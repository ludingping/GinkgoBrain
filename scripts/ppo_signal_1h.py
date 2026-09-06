"""
Live 1h trading signal from a trained signal-layered PPO (BTC/USDT by default).

Pulls the most recent 1m klines from Postgres, resamples to the config
timeframe, recomputes indicators + sig_* features, deterministically replays
the environment from a fixed anchor date (`crypto.test_start` in the config, or
--anchor), and reports the policy's decision on the latest *closed* bar.

Replay-from-anchor instead of a persisted state file: the env state is a pure
function of (model, closed bars, anchor), so every run is reproducible, there is
nothing to drift or corrupt, and ~3–4k policy forward passes take seconds.

Usage:
  uv run python scripts/ppo_signal_1h.py                       # default config/model
  uv run python scripts/ppo_signal_1h.py --lookback-days 400 --verify   # parity check
  uv run python scripts/ppo_signal_1h.py --as-of "2026-09-01 12:00:00"

Outputs:
  reports/daily_signals/btc_1h_<YYYYMMDD_HH>.md  and  btc_1h_latest.md
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agents.ppo_shared import resample_ohlcv                          # noqa: E402
from envs.signal_layered_env import (                                  # noqa: E402
    SignalLayeredEnv, TARGET_POSITION, load_signal_list,
)
from scripts.backtest_signal_layered import (                          # noqa: E402
    _ALLOWED_ENV_KEYS, load_model, make_policy_act_fn, run_backtest, summarise,
)
from utils.db import read_ohlcv                                        # noqa: E402
from utils.indicators import add_indicators                            # noqa: E402
from utils.metrics import bars_per_year                                # noqa: E402
from utils.signals import add_contract_signals, add_signals            # noqa: E402
from utils.splits import slice_by_dates                                # noqa: E402

# Default lookback = the config's `start_date`, i.e. the exact frame the model
# was trained on. A shorter window (--lookback-days) reproduces every signal to
# ~1e-8 except `sig_volume_obv_slope`: OBV is a running cumsum, so
# `obv.pct_change(5)` depends on where the sum starts (verified 2026-09-05:
# max |Δ| = 2.0 with a 400-day window). Until that signal is redefined to be
# start-invariant, only the full-history load is bit-exact. --verify checks.
RECENT_BARS = 24


# ═══════════════════════════════════════════════════════════════════ data

def load_bars(cfg: dict, start: pd.Timestamp, end: pd.Timestamp | None) -> pd.DataFrame:
    """1m → timeframe OHLCV with indicators + signals. Drops an unfinished last bucket."""
    c = cfg["crypto"]
    tz = c.get("timezone", "Asia/Shanghai")
    timeframe = c.get("timeframe", "1h").lower()

    end_utc = (end or pd.Timestamp.now(tz=tz)).tz_convert("UTC")
    df_raw = read_ohlcv(
        c["symbol"], start=start.tz_convert("UTC").isoformat(), end=end_utc.isoformat(),
        table=c.get("db_table", "public.crypto_kline_binance"), only_closed=True,
    )
    if df_raw.empty:
        raise RuntimeError(f"No klines for {c['symbol']} in [{start}, {end_utc})")
    ts = df_raw["timestamp"]
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    df_raw["timestamp"] = ts.dt.tz_convert(tz)
    last_1m = df_raw["timestamp"].max()

    df_tf = resample_ohlcv(df_raw, timeframe)
    # A bucket [T, T+tf) is closed only if its last 1m bar (open_time T+tf−1m) exists.
    bucket = pd.Timedelta(timeframe)
    complete = df_tf["timestamp"] + bucket - pd.Timedelta(minutes=1) <= last_1m
    dropped = int((~complete).sum())
    df_tf = df_tf[complete].reset_index(drop=True)
    if dropped:
        print(f"[data] dropped {dropped} unfinished {timeframe} bucket(s); "
              f"last 1m bar = {last_1m}")

    df = add_indicators(df_tf)
    df = add_signals(df)
    df = add_contract_signals(df, timeframe=timeframe)
    return df.reset_index(drop=True)


def verify_feature_parity(cfg: dict, signal_cols: list[str], df_short: pd.DataFrame,
                          end: pd.Timestamp | None, n_check: int = 500) -> float:
    """Recompute signals from the config's full start_date and diff the tail."""
    c = cfg["crypto"]
    tz = c.get("timezone", "Asia/Shanghai")
    full_start = pd.Timestamp(c["start_date"], tz=tz)
    print(f"[verify] recomputing signals from {full_start} ...")
    df_full = load_bars(cfg, full_start, end)
    tail_ts = df_short["timestamp"].iloc[-n_check:]
    a = df_short.set_index("timestamp").loc[tail_ts, signal_cols + ["atr", "close"]]
    b = df_full.set_index("timestamp").loc[tail_ts, signal_cols + ["atr", "close"]]
    diff = (a - b).abs()
    worst = diff.max().sort_values(ascending=False)
    print("[verify] max |Δ| per column (last "
          f"{n_check} bars):\n{worst.head(6).to_string()}")
    return float(worst.max())


# ═══════════════════════════════════════════════════════════════════ report

def _fmt_pct(x: float) -> str:
    return f"{x:+.2%}"


def build_report(
    *, cfg: dict, model_path: Path, env: SignalLayeredEnv, df_bt: pd.DataFrame,
    steps_log: list[dict], trades_log: list[dict], signal_cols: list[str],
    action: int, probs: list[float], anchor: pd.Timestamp,
) -> str:
    c = cfg["crypto"]
    t = env.current_step
    ts_now = df_bt["timestamp"].iloc[t]
    close_now = float(env._close[t])
    mask = env.action_masks()
    locked = not mask.all()
    target = TARGET_POSITION[action]
    cur_ratio = float(env.position * close_now / env._portfolio_value(close_now))
    unreal = ((close_now - env._entry_price) / env._entry_price
              if env._entry_price and env.position > 0 else 0.0)
    ppy = bars_per_year(c.get("timeframe", "1h"))
    m = summarise(steps_log, trades_log, ppy)

    if target > cur_ratio + 0.02:
        verdict = f"加仓 → {target:.0%}"
    elif target < cur_ratio - 0.02:
        verdict = f"减仓 → {target:.0%}"
    else:
        verdict = f"维持 {target:.0%}"

    L: list[str] = []
    L.append(f"# {c['symbol']} {c.get('timeframe', '1h')} PPO signal\n")
    L.append(f"- As of bar: `{ts_now}` (close {close_now:,.2f})")
    L.append(f"- Generated: `{dt.datetime.now().isoformat(timespec='seconds')}`")
    L.append(f"- Model: `{model_path}`")
    L.append(f"- Replay anchor: `{anchor}` ({len(steps_log):,} bars replayed)\n")

    L.append("## Decision\n")
    L.append("| | |")
    L.append("|---|---|")
    L.append(f"| **Target position** | **{target:.0%}** — {verdict} |")
    L.append(f"| Current position (replayed book) | {cur_ratio:.0%} |")
    L.append(f"| Unrealised P&L | {_fmt_pct(unreal)} |")
    L.append(f"| Action lock | {'🔒 min_hold / cooldown active' if locked else 'free'} |")
    L.append(f"| Steps since last switch | {env._steps_since_trade} |")
    L.append(f"| Steps since stop | {env._steps_since_stop} |")
    L.append("| Action probs (0/25/50/75/100%) | "
             + " / ".join(f"{p:.2f}" for p in probs) + " |")
    L.append("")

    L.append("## Signals now (by |value|)\n")
    L.append("| Signal | Value | Read |")
    L.append("|--------|------:|------|")
    vals = {col: float(df_bt.loc[t, col]) for col in signal_cols}
    for name, v in sorted(vals.items(), key=lambda kv: abs(kv[1]), reverse=True)[:8]:
        read = "↑ bullish" if v > 0.2 else ("↓ bearish" if v < -0.2 else "≈ neutral")
        L.append(f"| `{name}` | {v:+.3f} | {read} |")
    L.append("")

    L.append(f"## Last {RECENT_BARS} bars\n")
    L.append("| Bar | Close | Executed | Target | Position | Stop |")
    L.append("|-----|------:|---------:|-------:|---------:|:----:|")
    for s in steps_log[-RECENT_BARS:]:
        L.append(f"| {s['timestamp'][:16]} | {s['price']:,.0f} | {s['action_executed']} | "
                 f"{s['target_ratio']:.0%} | {s['state']['position_ratio']:.0%} | "
                 f"{'✕' if s['stop_loss_triggered'] else ''} |")
    L.append("")

    L.append(f"## Replay since anchor ({anchor.date()} → {ts_now.date()})\n")
    L.append("| Total return | Max DD | Sharpe (ann.) | Trades | Stops |")
    L.append("|-------------:|-------:|--------------:|-------:|------:|")
    L.append(f"| {_fmt_pct(m['total_return'])} | {m['max_drawdown']:.2%} | "
             f"{m['sharpe']:.3f} | {m['trades']} | {len(env.stop_loss_events)} |")
    L.append("")
    L.append("> Positions are *targets* for a spot long-only book; the replayed book "
             "starts flat at the anchor with the config's initial balance.")
    return "\n".join(L)


# ═══════════════════════════════════════════════════════════════════ main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("config/stage2_1h_signal_v3.yaml"))
    ap.add_argument("--model", type=Path,
                    default=Path("models/saved/BTCUSDT_signal_layered_1h_v3/best_model.zip"))
    ap.add_argument("--anchor", default=None,
                    help="Replay start (config tz). Default: crypto.test_start from config.")
    ap.add_argument("--as-of", default=None,
                    help="Pretend 'now' is this timestamp (config tz); bars at/after are ignored.")
    ap.add_argument("--lookback-days", type=int, default=None,
                    help="Load only this many days of 1m bars instead of the config's "
                         "start_date. Faster, but sig_volume_obv_slope will differ (see header).")
    ap.add_argument("--out-dir", type=Path, default=Path("reports/daily_signals"))
    ap.add_argument("--verify", action="store_true",
                    help="Diff signals against a full-history recompute; exit 2 if they differ.")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    c = cfg["crypto"]
    tz = c.get("timezone", "Asia/Shanghai")

    signal_cols = load_signal_list(cfg["signals"]["config_path"])
    if any(s.startswith(("sig_funding_", "sig_oi_", "sig_liq_")) for s in signal_cols):
        raise NotImplementedError("contract signals are not supported by ppo_signal_1h.py")

    env_cfg = {k: v for k, v in cfg["env"].items() if k in _ALLOWED_ENV_KEYS}
    env_cfg["random_start"] = False
    env_cfg["max_episode_steps"] = None
    prefix_rows = SignalLayeredEnv.warmup_rows(env_cfg.get("window_size", 24))

    anchor = pd.Timestamp(args.anchor or c["test_start"], tz=tz)
    as_of = pd.Timestamp(args.as_of, tz=tz) if args.as_of else None
    if args.lookback_days is None:
        load_start = pd.Timestamp(c["start_date"], tz=tz)
    else:
        load_start = (min(anchor, (as_of or pd.Timestamp.now(tz=tz)))
                      - pd.Timedelta(days=args.lookback_days))

    print(f"Loading {c['symbol']} bars from {load_start} ...")
    df = load_bars(cfg, load_start, as_of)

    if args.verify:
        worst = verify_feature_parity(cfg, signal_cols, df, as_of)
        if worst > 1e-6:
            print(f"[verify] FAILED: max |Δ| = {worst:.3e} > 1e-6 — increase --lookback-days",
                  file=sys.stderr)
            return 2
        print("[verify] OK — lookback window reproduces full-history features")

    df_bt = slice_by_dates(df, anchor, None, prefix_rows=prefix_rows)
    if len(df_bt) <= prefix_rows + 1:
        raise RuntimeError("Not enough bars after the anchor to replay; move --anchor earlier")

    env = SignalLayeredEnv(df_bt, signal_cols=signal_cols, **env_cfg)
    model = load_model(args.model)
    act_fn = make_policy_act_fn(model)

    # Replay every closed bar up to the latest; env ends with current_step = last bar.
    steps_log, trades_log = run_backtest(act_fn, env, df_bt, signal_cols)
    assert env.current_step == len(df_bt) - 1
    obs = env._get_obs()
    action, probs = act_fn(obs, env)

    report = build_report(
        cfg=cfg, model_path=args.model, env=env, df_bt=df_bt, steps_log=steps_log,
        trades_log=trades_log, signal_cols=signal_cols, action=action, probs=probs,
        anchor=anchor,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ts_now = df_bt["timestamp"].iloc[-1]
    stamp = ts_now.strftime("%Y%m%d_%H")
    sym = c["symbol"].split("/")[0].lower()
    tf = c.get("timeframe", "1h")
    dated = args.out_dir / f"{sym}_{tf}_{stamp}.md"
    latest = args.out_dir / f"{sym}_{tf}_latest.md"
    dated.write_text(report, encoding="utf-8")
    latest.write_text(report, encoding="utf-8")

    print(f"\n{c['symbol']} {tf} @ {ts_now}: target position = "
          f"{TARGET_POSITION[action]:.0%} (action {action}, probs "
          f"{' / '.join(f'{p:.2f}' for p in probs)})")
    print(f"Wrote {dated}\nWrote {latest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
