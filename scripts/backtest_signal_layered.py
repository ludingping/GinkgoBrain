"""
Backtest + attribution for the signal_layered paradigm (Phase 3).

Produces per the §7 spec:
  models/saved/{run_name}/attribution.jsonl
  models/saved/{run_name}/trade_report.md
  models/saved/{run_name}/signal_decision_plot.png

Usage:
  python scripts/backtest_signal_layered.py \\
      --model models/saved/BTCUSDT_ppo_4h_signal/best_model.zip \\
      --config config/stage2_4h_signal.yaml \\
      [--test-start 2025-04-01] [--run-name BTCUSDT_ppo_4h_signal]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agents.ppo_shared import resample_ohlcv                          # noqa: E402
from envs.signal_layered_env import (                                  # noqa: E402
    SignalLayeredEnv, TARGET_POSITION, load_signal_list,
)
from utils.db import read_ohlcv                                        # noqa: E402
from utils.indicators import add_indicators                            # noqa: E402
from utils.signals import add_signals                                  # noqa: E402


# ═══════════════════════════════════════════════════════════════════ data

def load_data(cfg: dict, test_start: str | None, test_end: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (full_df_with_signals, backtest_df) based on config + optional date cut."""
    c = cfg["crypto"]
    tz = c.get("timezone", "Asia/Shanghai")
    start_utc = pd.Timestamp(c["start_date"], tz=tz).tz_convert("UTC").isoformat()
    end_utc = pd.Timestamp(c["end_date"], tz=tz).tz_convert("UTC").isoformat()

    print("Loading data from DB...")
    df_raw = read_ohlcv(
        c["symbol"], start=start_utc, end=end_utc,
        table=c.get("db_table", "public.crypto_kline_binance"), only_closed=True,
    )
    ts = df_raw["timestamp"]
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    df_raw["timestamp"] = ts.dt.tz_convert(tz)
    df_tf = resample_ohlcv(df_raw, c.get("timeframe", "4h").lower())
    df = add_indicators(df_tf)
    df = add_signals(df)

    if test_start:
        test_ts = pd.Timestamp(test_start, tz=tz)
        bt_df = df[df["timestamp"] >= test_ts].reset_index(drop=True)
    else:
        split = int(len(df) * cfg["crypto"].get("train_ratio", 0.75))
        bt_df = df.iloc[split:].reset_index(drop=True)
    if test_end:
        end_ts = pd.Timestamp(test_end, tz=tz)
        bt_df = bt_df[bt_df["timestamp"] < end_ts].reset_index(drop=True)
    print(f"Backtest period: {bt_df['timestamp'].iloc[0]} → "
          f"{bt_df['timestamp'].iloc[-1]} ({len(bt_df)} bars)")
    return df, bt_df


# ═══════════════════════════════════════════════════════════════════ run

def run_backtest(
    model,
    env: SignalLayeredEnv,
    df_bt: pd.DataFrame,
    signal_cols: list[str],
) -> tuple[list[dict], list[dict]]:
    """
    Full deterministic rollout. Returns (steps_log, trades_log).

    steps_log: one entry per env step — full TC-C1 attribution.jsonl spec.
    trades_log: entry per position change — for trade_report.md.
    """
    from stable_baselines3.common.utils import obs_as_tensor

    obs, _ = env.reset()
    steps_log: list[dict] = []
    prev_target: float | None = None
    open_trade: dict | None = None
    trades_log: list[dict] = []
    trade_idx = 0

    done = False
    while not done:
        step_idx = env.current_step
        price = float(env._close[step_idx])
        atr = float(env._atr[step_idx])

        # Raw signal values at this step
        sig_vals = {col: float(df_bt.loc[step_idx, col]) for col in signal_cols
                    if step_idx < len(df_bt)}

        # Policy inference — get action probs as well
        obs_t = obs_as_tensor(obs[None], model.policy.device)
        with torch.no_grad():
            dist = model.policy.get_distribution(obs_t)
            probs = dist.distribution.probs.cpu().numpy().squeeze().tolist()
        action = int(np.argmax(probs))  # deterministic = argmax

        obs, reward, done, _, info = env.step(action)

        target_ratio = TARGET_POSITION[action]
        rb = info["reward_breakdown"]

        # ── State vector (mirrors §5.3 / TC-C1 spec)
        state = {
            "position_ratio": float(info["position_ratio"]),
            "unrealized_pnl": float(
                (price - env._entry_price) / env._entry_price
                if env._entry_price and env.position > 0 else 0.0
            ),
            "steps_since_stop": int(env._steps_since_stop),
            "cumulative_log_return": float(np.clip(env._cum_log_return, -1, 1)),
        }

        # Timestamp
        ts_val = df_bt.get("timestamp", None)
        ts_str = (str(ts_val.iloc[step_idx]) if ts_val is not None
                  and step_idx < len(df_bt) else "")

        record = {
            "step": step_idx,
            "timestamp": ts_str,
            "price": price,
            "atr": atr,
            "signals": sig_vals,
            "state": state,
            "action": action,
            "target_ratio": target_ratio,
            "action_probs": [round(p, 5) for p in probs],
            "reward": float(reward),
            "reward_breakdown": {k: float(v) for k, v in rb.items()},
            "stop_loss_triggered": bool(info.get("stop_loss_triggered", False)),
        }
        steps_log.append(record)

        # ── Trade tracking (position changes)
        if prev_target is not None and target_ratio != prev_target:
            if open_trade is not None:
                # Close the previous trade
                duration_bars = step_idx - open_trade["entry_step"]
                entry_val = open_trade["entry_portfolio"]
                exit_val = info["portfolio_value"]
                pnl_pct = (exit_val - entry_val) / max(entry_val, 1e-8)
                open_trade.update({
                    "exit_step": step_idx,
                    "exit_timestamp": ts_str,
                    "exit_price": price,
                    "exit_target": target_ratio,
                    "exit_reason": ("atr_stop" if info.get("stop_loss_triggered")
                                   else "signal_change"),
                    "duration_bars": duration_bars,
                    "pnl_pct": pnl_pct,
                    "exit_portfolio": exit_val,
                    # Top signals at entry (by |value|)
                    "top_signals_at_entry": sorted(
                        open_trade["entry_signals"].items(),
                        key=lambda x: abs(x[1]), reverse=True
                    )[:6],
                })
                trades_log.append(open_trade)
                trade_idx += 1

            # Open a new trade record
            open_trade = {
                "trade_id": trade_idx,
                "entry_step": step_idx,
                "entry_timestamp": ts_str,
                "entry_price": price,
                "entry_target": target_ratio,
                "from_ratio": prev_target,
                "entry_portfolio": info["portfolio_value"],
                "entry_signals": sig_vals,
            }
        elif open_trade is None and target_ratio != 0.0:
            # First entry from flat
            open_trade = {
                "trade_id": trade_idx,
                "entry_step": step_idx,
                "entry_timestamp": ts_str,
                "entry_price": price,
                "entry_target": target_ratio,
                "from_ratio": 0.0,
                "entry_portfolio": info["portfolio_value"],
                "entry_signals": sig_vals,
            }

        prev_target = target_ratio

    return steps_log, trades_log


# ═══════════════════════════════════════════════════════════════════ reports

def write_attribution_jsonl(steps_log: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in steps_log:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {out_path}  ({len(steps_log):,} records)")


def write_trade_report(
    trades_log: list[dict],
    steps_log: list[dict],
    env: SignalLayeredEnv,
    signal_cols: list[str],
    out_path: Path,
) -> None:
    """TC-C2 compliant trade_report.md."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pv_series = [10_000.0]
    for s in steps_log:
        pv_series.append(pv_series[-1] * np.exp(s["reward_breakdown"]["log_return"]))
    pv_arr = np.array(pv_series)
    total_ret = pv_arr[-1] / pv_arr[0] - 1
    max_dd = np.min(pv_arr / np.maximum.accumulate(pv_arr)) - 1
    wins = [t for t in trades_log if t.get("pnl_pct", 0) > 0]
    win_rate = len(wins) / max(len(trades_log), 1)
    gross_profit = sum(t["pnl_pct"] for t in wins)
    gross_loss = abs(sum(t.get("pnl_pct", 0) for t in trades_log
                         if t.get("pnl_pct", 0) <= 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Compute simple Sharpe from step log_returns
    rets = np.array([s["reward_breakdown"]["log_return"] for s in steps_log])
    sharpe = (rets.mean() / (rets.std() + 1e-10)) * np.sqrt(2190)

    lines: list[str] = []
    lines.append("# Signal-Layered Backtest Report\n")
    lines.append(f"Generated: `{dt.datetime.now().isoformat(timespec='seconds')}`\n")
    lines.append("## Summary\n")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total return | {total_ret:+.2%} |")
    lines.append(f"| Max drawdown | {max_dd:.2%} |")
    lines.append(f"| Sharpe (annualised) | {sharpe:.3f} |")
    lines.append(f"| Trades | {len(trades_log)} |")
    lines.append(f"| Win rate | {win_rate:.1%} |")
    lines.append(f"| Profit factor | {profit_factor:.2f} |")
    lines.append(f"| Steps | {len(steps_log):,} |")
    lines.append("")

    # Action distribution
    from collections import Counter
    action_ctr = Counter(s["action"] for s in steps_log)
    total_s = len(steps_log)
    lines.append("## Action distribution\n")
    lines.append("| Action | Target | Steps | % |")
    lines.append("|--------|--------|-------|---|")
    for a in range(5):
        cnt = action_ctr.get(a, 0)
        lines.append(f"| {a} | {int(TARGET_POSITION[a]*100)}% | {cnt} | {100*cnt/total_s:.1f}% |")
    lines.append("")

    # Per-trade detail
    lines.append("## Trade log\n")
    for t in trades_log:
        tid = t["trade_id"] + 1
        frm = int(t["from_ratio"] * 100)
        to = int(t["entry_target"] * 100)
        direction = "加仓" if to > frm else "减仓"
        dur = t.get("duration_bars", "?")
        pnl = t.get("pnl_pct", 0)
        reason = t.get("exit_reason", "open")
        lines.append(f"### Trade #{tid}")
        lines.append(f"- Time: `{t['entry_timestamp']}`")
        lines.append(f"- Action: {direction} {frm}% → {to}%")
        lines.append(f"- Exit: `{t.get('exit_timestamp','—')}` ({dur} bars), reason=`{reason}`")
        lines.append(f"- P&L: {pnl:+.2%}")
        if t.get("top_signals_at_entry"):
            lines.append("- Top signals at entry:")
            for name, val in t["top_signals_at_entry"]:
                direction_str = "↑ bullish" if val > 0.2 else ("↓ bearish" if val < -0.2
                                                                 else "≈ neutral")
                lines.append(f"  - `{name}` = {val:+.3f}  {direction_str}")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}  ({len(trades_log)} trades)")


def write_signal_plot(
    steps_log: list[dict],
    trades_log: list[dict],
    signal_cols: list[str],
    out_path: Path,
) -> None:
    """TC-C3 multi-panel signal-decision plot."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prices  = [s["price"] for s in steps_log]
    pos_rat = [s["state"]["position_ratio"] for s in steps_log]
    steps   = list(range(len(steps_log)))

    # Signal group membership
    trend_sigs  = [c for c in signal_cols if "trend" in c]
    mom_sigs    = [c for c in signal_cols if "mom" in c]
    vol_sigs    = [c for c in signal_cols if "vol_" in c or "volume" in c]
    regime_sigs = [c for c in signal_cols if "regime" in c]

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(5, 1, figure=fig, hspace=0.35,
                           height_ratios=[3, 1.2, 1.2, 1.2, 0.9])
    axes = [fig.add_subplot(gs[i]) for i in range(5)]

    # ── Price + position shading
    ax0 = axes[0]
    ax0.plot(steps, prices, linewidth=0.8, color="#2c3e50", label="Price")
    ax0.fill_between(steps, prices,
                     alpha=0.08, color="steelblue", label="BTC price")
    # Position shading bands
    pos_arr = np.array(pos_rat)
    ax0_twin = ax0.twinx()
    ax0_twin.fill_between(steps, pos_arr, alpha=0.25, color="#e67e22", label="position")
    ax0_twin.set_ylim(0, 1.5)
    ax0_twin.set_ylabel("Position ratio", fontsize=8, color="#e67e22")
    # Trade arrows
    for t in trades_log:
        es = t["entry_step"]
        if es < len(steps_log):
            ep = steps_log[es]["price"]
            color = "#27ae60" if t["entry_target"] > t["from_ratio"] else "#c0392b"
            marker = "^" if t["entry_target"] > t["from_ratio"] else "v"
            ax0.scatter(es, ep, marker=marker, s=60, color=color, zorder=5)
    ax0.set_title("Price + Position (buy ▲ sell ▼)", fontsize=10)
    ax0.set_ylabel("Price (USD)")
    ax0.tick_params(axis="x", labelbottom=False)

    # ── Signal panels helper
    palette = ["#2980b9", "#e74c3c", "#27ae60", "#8e44ad", "#f39c12", "#16a085"]

    def _plot_sigs(ax, sig_group, title):
        for i, col in enumerate(sig_group):
            vals = [s["signals"].get(col, 0) for s in steps_log]
            ax.plot(steps, vals, linewidth=0.6, alpha=0.85,
                    color=palette[i % len(palette)], label=col.replace("sig_", ""))
        ax.axhline(0, color="gray", linewidth=0.4, linestyle="--")
        ax.set_ylim(-1.2, 1.2)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=6, loc="upper left", ncol=3, framealpha=0.5)
        ax.tick_params(axis="x", labelbottom=False)
        ax.set_ylabel("Signal", fontsize=8)

    _plot_sigs(axes[1], trend_sigs, "Trend signals")
    _plot_sigs(axes[2], mom_sigs, "Momentum signals")

    # Vol signals + ATR stop markers
    _plot_sigs(axes[3], vol_sigs + regime_sigs, "Volatility + Regime signals")
    stop_steps = [i for i, s in enumerate(steps_log) if s["stop_loss_triggered"]]
    if stop_steps:
        axes[3].scatter(stop_steps,
                        [steps_log[i]["signals"].get(signal_cols[0], 0) for i in stop_steps],
                        marker="x", s=80, color="red", zorder=6, label="ATR stop")
        axes[3].legend(fontsize=6, loc="upper left", ncol=4, framealpha=0.5)

    # ── Position ratio
    axes[4].fill_between(steps, pos_arr, alpha=0.55, color="#e67e22")
    axes[4].axhline(0.5, color="gray", linewidth=0.4, linestyle="--")
    axes[4].set_ylim(0, 1.1)
    axes[4].set_title("Position ratio", fontsize=9)
    axes[4].set_xlabel("Step")
    axes[4].set_ylabel("Ratio")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Wrote {out_path}")


# ═══════════════════════════════════════════════════════════════════ main

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path,
                        default=Path("models/saved/BTCUSDT_ppo_4h_signal/best_model.zip"))
    parser.add_argument("--config", type=Path,
                        default=Path("config/stage2_4h_signal.yaml"))
    parser.add_argument("--test-start", default=None,
                        help="ISO date string; if set, backtest from this date. "
                             "Otherwise uses 1-train_ratio tail of dataset.")
    parser.add_argument("--test-end", default=None,
                        help="ISO date string; if set, exclude bars at/after this date.")
    parser.add_argument("--run-name", default=None,
                        help="Output directory under models/saved/. "
                             "Defaults to model parent dir name.")
    args = parser.parse_args()

    if not args.model.exists():
        print(f"ERROR: model not found at {args.model}", file=sys.stderr)
        return 1

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    signal_cols = load_signal_list(cfg["signals"]["config_path"])
    print(f"Loaded {len(signal_cols)} signals")

    _, bt_df = load_data(cfg, args.test_start, args.test_end)

    _ALLOWED = {"window_size", "initial_balance", "commission",
                "risk_aversion_coef", "excess_return_coef",
                "stop_atr_mult", "stop_cooldown_steps"}
    env_cfg = {k: v for k, v in cfg["env"].items() if k in _ALLOWED}
    env_cfg["random_start"] = False

    env = SignalLayeredEnv(bt_df, signal_cols=signal_cols, **env_cfg)

    from stable_baselines3 import PPO
    print(f"Loading model from {args.model}...")
    model = PPO.load(str(args.model))

    steps_log, trades_log = run_backtest(model, env, bt_df, signal_cols)

    run_name = args.run_name or args.model.parent.name
    out_dir = REPO_ROOT / "models" / "saved" / run_name
    log_dir = REPO_ROOT / "models" / "logs" / run_name

    write_attribution_jsonl(steps_log, log_dir / "attribution.jsonl")
    write_trade_report(trades_log, steps_log, env, signal_cols,
                       out_dir / "trade_report.md")
    write_signal_plot(steps_log, trades_log, signal_cols,
                      out_dir / "signal_decision_plot.png")

    # Print summary
    pv_end = env._portfolio_value(float(env._close[env.current_step]))
    total_ret = pv_end / env.initial_balance - 1
    rets = np.array([s["reward_breakdown"]["log_return"] for s in steps_log])
    sharpe = (rets.mean() / (rets.std() + 1e-10)) * np.sqrt(2190)
    print(f"\n── Backtest Summary ──────────────────")
    print(f"  Steps: {len(steps_log):,}")
    print(f"  Trades: {len(trades_log)}")
    print(f"  Total return: {total_ret:+.2%}")
    print(f"  Sharpe (ann.): {sharpe:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
