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
      [--test-start 2025-04-01] [--test-end 2026-04-13] [--run-name BTCUSDT_ppo_4h_signal]

Slice selection (v3): --test-start/--test-end win; otherwise the config's
`crypto.val_start` → `crypto.test_start` window (the EvalCallback validation
slice); otherwise the legacy 1-train_ratio tail. Every slice is prefixed with
`SignalLayeredEnv.warmup_rows(window_size)` bars so the first decision lands on
the requested start. Sharpe is annualised from the config timeframe.

Baselines (same slice, same env costs) are appended to the report: pure
buy&hold, a `sig_mtf_4h_trend > 0 → 100%` rule, and constant positions
p ∈ {0, 25, 50, 75, 100}% — the last row-set doubles as the reward table used to
calibrate `risk_aversion_coef`.
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
from utils.data_loader import merge_contract_data                      # noqa: E402
from utils.data_loader import (  # noqa: E402
    check_contract_coverage, neutral_fill_contract_columns, required_contract_sources,
)
from utils.db import (                                                  # noqa: E402
    read_funding,
    read_liquidation_agg,
    read_ohlcv,
    read_open_interest,
)
from utils.indicators import add_indicators                            # noqa: E402
from utils.metrics import annualised_sharpe, bars_per_year             # noqa: E402
from utils.signals import add_contract_signals, add_signals            # noqa: E402
from utils.splits import slice_by_dates                                # noqa: E402

# Env kwargs the backtest forwards from the config. Must stay a superset of
# everything that changes *dynamics* (min_hold / stop / cooldown / commission),
# otherwise the backtest silently diverges from training.
_ALLOWED_ENV_KEYS = {
    "window_size", "initial_balance", "commission",
    "risk_aversion_coef", "excess_return_coef",
    "action_inertia_coef", "trade_penalty_coef",
    "stop_atr_mult", "stop_cooldown_steps", "min_hold_steps",
}

RULE_SIGNAL = "sig_mtf_4h_trend"      # rule baseline: long 100 % when > 0
GATE_SMA_DAYS = 200                   # Spider overlay: last closed UTC daily close > SMA200


def daily_sma_gate(full_df: pd.DataFrame, bt_df: pd.DataFrame,
                   sma_days: int = GATE_SMA_DAYS) -> np.ndarray:
    """Per-bar bool (aligned to bt_df) mirroring Spider's regime gate
    (`overlay.compute_regime_ok`).

    Computed on `full_df` because SMA200 needs 200 *days* of history — far more
    than the slice's warmup prefix. Daily bars are **UTC** days (Spider fetches
    Binance 1d klines) whatever the frame's display tz. The decision at a bar
    closing at time c may use the last *fully closed* day, i.e. floor(c) − 1 day
    (c == D+1 00:00 exactly → day D). Gate is False while the SMA is not warmed up.
    """
    ts = pd.DatetimeIndex(full_df["timestamp"]).tz_convert("UTC")
    close = pd.Series(full_df["close"].to_numpy(), index=ts)
    daily = close.resample("1D", closed="left", label="left").last()
    sma = daily.rolling(sma_days).mean()
    ok_daily = (daily > sma) & sma.notna()

    bt_ts = pd.DatetimeIndex(bt_df["timestamp"]).tz_convert("UTC")
    # Bar duration = modal spacing (a data gap between the first two rows must
    # not shift every close time in the slice).
    bar = pd.Series(bt_ts).diff().dropna().mode().iloc[0]
    last_closed_day = (bt_ts + bar).floor("D") - pd.Timedelta(days=1)
    return ok_daily.reindex(last_closed_day, fill_value=False).to_numpy(dtype=bool)


def gated(act_fn, gate: np.ndarray):
    """Wrap an act_fn so the target is forced to 0 % when the gate is off.

    Applied *before* env.step, so the env's min_hold lock may delay a gate
    exit by ≤ min_hold_steps bars. Spider should apply the gate *after* its
    lock so the gate can always flatten.
    """
    def act(obs, env):
        action, probs = act_fn(obs, env)
        if not gate[env.current_step]:
            return 0, probs
        return action, probs
    return act


# ═══════════════════════════════════════════════════════════════════ data

def load_data(
    cfg: dict,
    test_start: str | None,
    test_end: str | None = None,
    *,
    with_contracts: bool = False,
    prefix_rows: int = 0,
    required_contracts=frozenset(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (full_df_with_signals, backtest_df) based on config + optional date cut.

    Args:
        with_contracts: True 时跨库 join funding/OI/liquidation 到 OHLCV，与
            train.py / signal_linear_baseline 的 with_contracts 镜像一致。
        prefix_rows: history bars prepended to the slice (= env warmup rows).
    """
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

    if with_contracts:
        ccxt_symbol = c["symbol"] if "/" in c["symbol"] else (
            f"{c['symbol'][:-4]}/USDT" if c["symbol"].endswith("USDT") else c["symbol"]
        )
        print(f"Loading contract metadata for {ccxt_symbol}...")
        df_funding = read_funding(ccxt_symbol, start=start_utc, end=end_utc)
        df_oi      = read_open_interest(ccxt_symbol, start=start_utc, end=end_utc)
        df_liq     = read_liquidation_agg(ccxt_symbol, start=start_utc, end=end_utc)
        for sub in (df_funding, df_oi, df_liq):
            if sub.empty:
                continue
            sub_ts = sub["timestamp"]
            if sub_ts.dt.tz is None:
                sub_ts = sub_ts.dt.tz_localize("UTC")
            sub["timestamp"] = sub_ts.dt.tz_convert(tz)
        df_tf = merge_contract_data(
            df_tf, df_funding=df_funding, df_oi=df_oi, df_liq=df_liq,
        )
        check_contract_coverage(df_tf, required=required_contracts)
        df_tf = neutral_fill_contract_columns(df_tf)

    df = add_indicators(df_tf)
    df = add_signals(df)
    df = add_contract_signals(df, timeframe=c.get("timeframe", "4h"))

    # Slice priority: CLI dates → config val window → legacy ratio tail.
    if not test_start and c.get("val_start"):
        test_start = c["val_start"]
        test_end = test_end or c.get("test_start")
        print(f"[slice] using config validation window {test_start} → {test_end}")
    if test_start:
        bt_df = slice_by_dates(df, test_start, test_end, prefix_rows=prefix_rows)
    else:
        split = int(len(df) * c.get("train_ratio", 0.75))
        bt_df = df.iloc[max(0, split - prefix_rows):].reset_index(drop=True)
        if test_end:
            bt_df = slice_by_dates(bt_df, None, test_end)
    if len(bt_df) <= prefix_rows + 1:
        raise SystemExit(
            f"backtest slice has {len(bt_df)} rows (need > {prefix_rows + 1}); check that the "
            f"config end_date ({c.get('end_date')}) covers the requested window"
        )
    first_decision = bt_df["timestamp"].iloc[min(prefix_rows, len(bt_df) - 1)]
    print(f"Backtest period: {first_decision} → {bt_df['timestamp'].iloc[-1]} "
          f"({len(bt_df) - prefix_rows} decision bars, +{prefix_rows} warmup)")
    return df, bt_df


# ═══════════════════════════════════════════════════════════════════ run

def is_maskable_model(model) -> bool:
    return "Maskable" in type(model.policy).__name__


def model_class_from_zip(path: Path) -> str:
    """'MaskablePPO' or 'PPO', read from the zip's serialized `policy_class`.

    SB3 stores `data` as JSON; `policy_class[":serialized:"]` is a base64
    cloudpickle whose bytes contain the class's module path. Checking that field
    for the sb3_contrib maskable module is targeted (unlike a substring search
    over the whole blob, which any docstring or kwarg could trip).
    """
    import base64
    import zipfile

    with zipfile.ZipFile(path) as zf:
        data = json.loads(zf.read("data"))
    serialized = data.get("policy_class", {}).get(":serialized:", "")
    raw = base64.b64decode(serialized) if serialized else b""
    return "MaskablePPO" if b"sb3_contrib.common.maskable" in raw else "PPO"


def load_model(path: Path):
    """PPO.load or MaskablePPO.load, decided by the policy class stored in the zip."""
    from sb3_contrib import MaskablePPO
    from stable_baselines3 import PPO

    cls = MaskablePPO if model_class_from_zip(path) == "MaskablePPO" else PPO
    print(f"Loading {cls.__name__} from {path}...")
    return cls.load(str(path))


def make_policy_act_fn(model):
    """Deterministic (argmax) action + probs; honours env.action_masks() for MaskablePPO."""
    from stable_baselines3.common.utils import obs_as_tensor

    maskable = is_maskable_model(model)

    def act(obs: np.ndarray, env: SignalLayeredEnv) -> tuple[int, list[float]]:
        obs_t = obs_as_tensor(obs[None], model.policy.device)
        with torch.no_grad():
            if maskable:
                dist = model.policy.get_distribution(
                    obs_t, action_masks=env.action_masks()[None]
                )
            else:
                dist = model.policy.get_distribution(obs_t)
            probs = dist.distribution.probs.cpu().numpy().squeeze().tolist()
        return int(np.argmax(probs)), probs

    return act


def run_backtest(
    act_fn,
    env: SignalLayeredEnv,
    df_bt: pd.DataFrame,
    signal_cols: list[str],
) -> tuple[list[dict], list[dict]]:
    """
    Full deterministic rollout. Returns (steps_log, trades_log).

    act_fn(obs, env) -> (action, probs | None). Trades are tracked on the
    *executed* action (env may override under min_hold / cooldown / stop).

    steps_log: one entry per env step — full TC-C1 attribution.jsonl spec.
    trades_log: entry per position change — for trade_report.md.
    """
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

        action, probs = act_fn(obs, env)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        executed = int(env._prev_action)

        target_ratio = TARGET_POSITION[executed]
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
            "action_executed": executed,
            "target_ratio": target_ratio,
            "action_probs": [round(p, 5) for p in probs] if probs is not None else None,
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


# ═══════════════════════════════════════════════════════════════════ baselines

def summarise(steps_log: list[dict], trades_log: list[dict], periods_per_year: int) -> dict:
    """Common metrics for the PPO run and every baseline (same definitions)."""
    rets = np.array([s["reward_breakdown"]["log_return"] for s in steps_log])
    rewards = np.array([s["reward"] for s in steps_log])
    pv = 10_000.0 * np.exp(np.concatenate([[0.0], np.cumsum(rets)]))
    return {
        "total_return": float(pv[-1] / pv[0] - 1),
        "max_drawdown": float(np.min(pv / np.maximum.accumulate(pv)) - 1),
        "sharpe": annualised_sharpe(rets, periods_per_year),
        "trades": len(trades_log),
        "steps": len(steps_log),
        "mean_reward": float(rewards.mean()) if len(rewards) else 0.0,
    }


def buy_and_hold_metrics(
    bt_df: pd.DataFrame, signal_cols: list[str], env_cfg: dict, periods_per_year: int,
) -> dict:
    """Pure price path from the first decision bar, one entry fee (same keys as summarise)."""
    probe = SignalLayeredEnv(bt_df, signal_cols=signal_cols, **env_cfg)
    start = probe._min_start()
    close = probe._close
    log_rets = np.diff(np.log(close[start:]))
    commission = float(env_cfg.get("commission", 0.0005))
    pv = 10_000.0 * (1 - commission) * np.exp(np.concatenate([[0.0], np.cumsum(log_rets)]))
    return {
        "total_return": float(pv[-1] / 10_000.0 - 1),
        "max_drawdown": float(np.min(pv / np.maximum.accumulate(pv)) - 1),
        "sharpe": annualised_sharpe(log_rets, periods_per_year),
        "trades": 1, "steps": int(len(log_rets)), "mean_reward": float("nan"),
    }


def run_baselines(
    bt_df: pd.DataFrame,
    signal_cols: list[str],
    env_cfg: dict,
    periods_per_year: int,
    policy_act_fn=None,
    full_df: pd.DataFrame | None = None,
) -> dict[str, dict]:
    """Reference strategies on the same slice.

    * buy_and_hold — pure price path from the first decision bar, one entry fee.
    * gate_only — Spider's regime gate alone (100 % when UTC daily close > SMA200).
    * ppo_gate — the policy with the gate clamping it to 0 % (≈ Spider production).
    * rule_4h_trend — `RULE_SIGNAL > 0 → 100 %, else 0 %` through the env
      (so it pays the same commission / stop / min_hold as the agent).
    * const_p — hold a fixed target through the env; `mean_reward` per step
      across p is the reward table used to sanity-check `risk_aversion_coef`.
    """
    out: dict[str, dict] = {}
    gate = daily_sma_gate(full_df if full_df is not None else bt_df, bt_df)

    def rollout(name: str, act_fn) -> None:
        env = SignalLayeredEnv(bt_df, signal_cols=signal_cols, **env_cfg)
        steps, trades = run_backtest(act_fn, env, bt_df, signal_cols)
        out[name] = summarise(steps, trades, periods_per_year)

    out["buy_and_hold"] = buy_and_hold_metrics(bt_df, signal_cols, env_cfg, periods_per_year)

    rollout("gate_only", gated(lambda obs, env: (4, None), gate))
    if policy_act_fn is not None:
        rollout("ppo_gate", gated(policy_act_fn, gate))

    if RULE_SIGNAL in signal_cols:
        col = bt_df[RULE_SIGNAL].to_numpy()

        def rule_act(obs, env):
            return (4 if col[env.current_step] > 0 else 0), None

        rollout("rule_4h_trend", rule_act)

    for a in range(5):
        rollout(f"const_{int(TARGET_POSITION[a] * 100)}pct", lambda obs, env, a=a: (a, None))

    return out


def format_baseline_table(ppo: dict, baselines: dict[str, dict]) -> list[str]:
    rows = [("ppo (this run)", ppo)] + list(baselines.items())
    lines = ["| Strategy | Total return | Max DD | Sharpe | Trades | mean reward/step |",
             "|----------|-------------:|-------:|-------:|-------:|-----------------:|"]
    for name, m in rows:
        mr = "—" if m["mean_reward"] != m["mean_reward"] else f"{m['mean_reward']:+.5f}"
        lines.append(f"| {name} | {m['total_return']:+.2%} | {m['max_drawdown']:.2%} | "
                     f"{m['sharpe']:.3f} | {m['trades']} | {mr} |")
    return lines


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
    *,
    periods_per_year: int,
    period_label: str = "",
    baselines: dict[str, dict] | None = None,
) -> None:
    """TC-C2 compliant trade_report.md (+ v3 baseline table)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m = summarise(steps_log, trades_log, periods_per_year)
    wins = [t for t in trades_log if t.get("pnl_pct", 0) > 0]
    win_rate = len(wins) / max(len(trades_log), 1)
    gross_profit = sum(t["pnl_pct"] for t in wins)
    gross_loss = abs(sum(t.get("pnl_pct", 0) for t in trades_log
                         if t.get("pnl_pct", 0) <= 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    lines: list[str] = []
    lines.append("# Signal-Layered Backtest Report\n")
    lines.append(f"Generated: `{dt.datetime.now().isoformat(timespec='seconds')}`  ")
    if period_label:
        lines.append(f"Period: `{period_label}`  ")
    lines.append(f"Sharpe annualisation: sqrt({periods_per_year}) bars/year\n")
    lines.append("## Summary\n")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total return | {m['total_return']:+.2%} |")
    lines.append(f"| Max drawdown | {m['max_drawdown']:.2%} |")
    lines.append(f"| Sharpe (annualised) | {m['sharpe']:.3f} |")
    lines.append(f"| Trades | {len(trades_log)} |")
    lines.append(f"| Win rate | {win_rate:.1%} |")
    lines.append(f"| Profit factor | {profit_factor:.2f} |")
    lines.append(f"| Steps | {len(steps_log):,} |")
    lines.append(f"| Mean reward / step | {m['mean_reward']:+.5f} |")
    lines.append("")

    if baselines:
        lines.append("## Baselines (same slice, same env costs)\n")
        lines.extend(format_baseline_table(m, baselines))
        lines.append("")
        lines.append("`const_*` rows hold a fixed target through the env (ATR stop still "
                     "active); their `mean reward/step` spread is the exposure tax implied "
                     "by `risk_aversion_coef` — it should be small relative to plausible alpha.")
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
    parser.add_argument("--no-baselines", action="store_true",
                        help="Skip buy&hold / rule / constant-position baselines.")
    args = parser.parse_args()

    if not args.model.exists():
        print(f"ERROR: model not found at {args.model}", file=sys.stderr)
        return 1

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    signal_cols = load_signal_list(cfg["signals"]["config_path"])
    print(f"Loaded {len(signal_cols)} signals")

    needs_contracts = any(
        s.startswith(("sig_funding_", "sig_oi_", "sig_liq_")) for s in signal_cols
    )
    if needs_contracts:
        print("[backtest] candidate pool contains contract signals → enabling cross-db merge")

    env_cfg = {k: v for k, v in cfg["env"].items() if k in _ALLOWED_ENV_KEYS}
    env_cfg["random_start"] = False
    env_cfg["max_episode_steps"] = None       # one uninterrupted pass
    prefix_rows = SignalLayeredEnv.warmup_rows(env_cfg.get("window_size", 24))
    periods_per_year = bars_per_year(cfg["crypto"].get("timeframe", "4h"))

    full_df, bt_df = load_data(cfg, args.test_start, args.test_end,
                               with_contracts=needs_contracts, prefix_rows=prefix_rows,
                               required_contracts=required_contract_sources(signal_cols))
    period_label = (f"{bt_df['timestamp'].iloc[prefix_rows]} → "
                    f"{bt_df['timestamp'].iloc[-1]}")

    env = SignalLayeredEnv(bt_df, signal_cols=signal_cols, **env_cfg)
    model = load_model(args.model)

    policy_act = make_policy_act_fn(model)
    steps_log, trades_log = run_backtest(policy_act, env, bt_df, signal_cols)

    baselines = None
    if not args.no_baselines:
        print("Running baselines...")
        baselines = run_baselines(bt_df, signal_cols, env_cfg, periods_per_year,
                                  policy_act_fn=policy_act, full_df=full_df)

    run_name = args.run_name or args.model.parent.name
    out_dir = REPO_ROOT / "models" / "saved" / run_name
    log_dir = REPO_ROOT / "models" / "logs" / run_name

    write_attribution_jsonl(steps_log, log_dir / "attribution.jsonl")
    write_trade_report(trades_log, steps_log, env, signal_cols,
                       out_dir / "trade_report.md",
                       periods_per_year=periods_per_year,
                       period_label=period_label, baselines=baselines)
    write_signal_plot(steps_log, trades_log, signal_cols,
                      out_dir / "signal_decision_plot.png")

    summary = summarise(steps_log, trades_log, periods_per_year)
    if baselines is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "baselines.json").write_text(
            json.dumps({"period": period_label, "periods_per_year": periods_per_year,
                        "ppo": summary, "baselines": baselines}, indent=2),
            encoding="utf-8",
        )

    print(f"\n── Backtest Summary ({period_label}) ──")
    print(f"  Steps: {summary['steps']:,}   Trades: {summary['trades']}")
    print(f"  Total return: {summary['total_return']:+.2%}   "
          f"Max DD: {summary['max_drawdown']:.2%}   "
          f"Sharpe (ann., {periods_per_year}): {summary['sharpe']:.3f}")
    if baselines:
        print("\n".join(format_baseline_table(summary, baselines)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
