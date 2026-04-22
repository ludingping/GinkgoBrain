"""
T4.3 + T4.4 — GBDT combiner backtest + SHAP attribution.

Loads a trained GBDTCombiner, runs a portfolio simulation on the test
set, writes:
  models/logs/{run_name}/attribution.jsonl      (SHAP values per step)
  models/saved/{run_name}/trade_report.md       (TC-C2-compatible report)
  models/saved/{run_name}/shap_importance.png   (T4.4 SHAP bar chart)

Portfolio model mirrors backtest_signal_layered.py:
  - Discrete target positions {0,1,2,3,4} → {0%,25%,50%,75%,100%}
  - Commission applied on rebalance (abs delta × commission)
  - No ATR stop (GBDT path; stop is in the env for PPO)
  - Long-only, no leverage

Usage:
  python scripts/backtest_gbdt.py \\
      --model models/saved/BTCUSDT_gbdt_1d \\
      --config config/stage4_1d_gbdt.yaml \\
      [--test-start 2024-07-01] [--run-name BTCUSDT_gbdt_1d]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agents.gbdt_combiner import GBDTCombiner            # noqa: E402
from agents.ppo_shared import resample_ohlcv             # noqa: E402
from envs.signal_layered_env import load_signal_list     # noqa: E402
from utils.db import read_ohlcv                          # noqa: E402
from utils.indicators import add_indicators              # noqa: E402
from utils.signals import add_signals                    # noqa: E402

TARGET_POSITION = {0: 0.00, 1: 0.25, 2: 0.50, 3: 0.75, 4: 1.00}
_TRADING_DAYS_PER_YEAR = 365   # crypto, continuous

_BARS_PER_YEAR = {
    "1m": 1440 * 365, "5min": 288 * 365, "15min": 96 * 365, "30min": 48 * 365,
    "1h": 24 * 365, "2h": 12 * 365, "4h": 6 * 365, "1d": 365,
}


# ─────────────────────────────────────────── risk overlays

def add_risk_overlay(df: pd.DataFrame, timeframe: str,
                     vol_window: int, regime_ma_days: int,
                     regime_resample: str = "1D") -> pd.DataFrame:
    """Append realized_vol_ann and regime_ok columns; computed on full df so
    slicing downstream inherits warmed-up values.

    regime_resample: pandas resample rule for the gate timeframe (e.g. "1D", "1h",
      "15min"). When != "1D", regime_ma_days is interpreted as "bars of that
      resample" rather than calendar days. Default "1D" preserves the original
      BTC 4h / 1h daily-SMA200 behavior byte-for-byte.
    """
    df = df.copy()
    bars_per_year = _BARS_PER_YEAR.get(timeframe, 24 * 365)

    log_ret = np.log(df["close"] / df["close"].shift(1))
    df["realized_vol_ann"] = (
        log_ret.rolling(vol_window).std() * np.sqrt(bars_per_year)
    ).bfill().fillna(0.6)

    gate_close = df.set_index("timestamp")["close"].resample(regime_resample).last().dropna()
    sma = gate_close.rolling(regime_ma_days).mean()
    regime_ok_gate = (gate_close > sma).astype(int)
    df["regime_ok"] = (
        regime_ok_gate.reindex(pd.DatetimeIndex(df["timestamp"]), method="ffill")
        .fillna(0).astype(int).values
    )
    return df


# ─────────────────────────────────────────── data

def load_data(cfg: dict, test_start: str | None, test_end: str | None = None,
              vol_window: int = 168, regime_ma_days: int = 200,
              regime_resample: str = "1D") -> pd.DataFrame:
    c = cfg["crypto"]
    tz = c.get("timezone", "Asia/Shanghai")
    start_utc = pd.Timestamp(c["start_date"], tz=tz).tz_convert("UTC").isoformat()
    end_utc = pd.Timestamp(c["end_date"], tz=tz).tz_convert("UTC").isoformat()

    print("Loading data from DB...")
    df_raw = read_ohlcv(
        c["symbol"], start=start_utc, end=end_utc,
        table=c.get("db_table", "public.crypto_kline_binance"),
        only_closed=True,
    )
    ts = df_raw["timestamp"]
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    df_raw["timestamp"] = ts.dt.tz_convert(tz)
    timeframe = c.get("timeframe", "1d").lower()
    df_tf = resample_ohlcv(df_raw, timeframe)
    df = add_indicators(df_tf)
    df = add_signals(df)
    df = add_risk_overlay(df, timeframe, vol_window, regime_ma_days, regime_resample)

    tz_ = c.get("timezone", "Asia/Shanghai")
    if test_start:
        cut = pd.Timestamp(test_start, tz=tz_)
        bt_df = df[df["timestamp"] >= cut].reset_index(drop=True)
    else:
        split = int(len(df) * c.get("train_ratio", 0.80))
        bt_df = df.iloc[split:].reset_index(drop=True)
    if test_end:
        end_cut = pd.Timestamp(test_end, tz=tz_)
        bt_df = bt_df[bt_df["timestamp"] < end_cut].reset_index(drop=True)

    print(f"Backtest period: {bt_df['timestamp'].iloc[0]} → "
          f"{bt_df['timestamp'].iloc[-1]} ({len(bt_df)} bars)")
    return bt_df


# ─────────────────────────────────────────── portfolio simulation

def run_backtest(
    combiner: GBDTCombiner,
    bt_df: pd.DataFrame,
    signal_cols: list[str],
    commission: float = 0.001,
    position_mode: str = "discrete",
    position_scale: float = 1.0,
    vol_target: float = 0.0,
    use_regime_gate: bool = False,
    rebalance_threshold: float = 0.01,
) -> tuple[list[dict], list[dict], pd.DataFrame]:
    """
    Return (steps_log, trades_log, equity_df).

    position_mode:
      - "discrete": map predicted Sharpe through config thresholds to 5 buckets
        {0, 0.25, 0.5, 0.75, 1.0}
      - "continuous": target_ratio = clip(predicted_sharpe / position_scale, 0, 1)

    Risk overlays (applied after raw ratio):
      - vol_target > 0: scale by min(vol_target / realized_vol_ann, 1.0)
      - use_regime_gate: force ratio to 0 when regime_ok == 0
    """
    close = bt_df["close"].values
    n = len(bt_df)

    positions_series = combiner.predict(bt_df, signal_cols)
    sharpe_series = combiner.predict_sharpe(bt_df, signal_cols)
    target_ratios = np.zeros(n, dtype=float)
    realized_vol = bt_df["realized_vol_ann"].values if "realized_vol_ann" in bt_df.columns else np.full(n, 0.6)
    regime_ok = bt_df["regime_ok"].values if "regime_ok" in bt_df.columns else np.ones(n, dtype=int)

    # SHAP for every step
    X = bt_df[signal_cols].values.astype(np.float32)
    shap_vals = combiner.shap_values(X)   # (n, n_features)
    ev = combiner.shap_expected_value()

    portfolio = 10_000.0
    position_ratio = 0.0
    steps_log: list[dict] = []
    trades_log: list[dict] = []
    equity = np.zeros(n)
    open_trade: dict | None = None
    trade_idx = 0

    for i in range(n):
        if position_mode == "continuous":
            raw_ratio = float(np.clip(sharpe_series.iloc[i] / position_scale, 0.0, 1.0))
            action = -1
        else:
            action = int(positions_series.iloc[i])
            raw_ratio = TARGET_POSITION[action]

        vol_scale = 1.0
        if vol_target > 0.0:
            rv = float(realized_vol[i])
            vol_scale = min(vol_target / max(rv, 1e-4), 1.0)
        regime_scale = 1.0 if (not use_regime_gate or regime_ok[i] == 1) else 0.0
        target_ratio = raw_ratio * vol_scale * regime_scale

        price = float(close[i])
        shap_i = {col: round(float(shap_vals[i, j]), 6)
                  for j, col in enumerate(signal_cols)}
        ts_str = str(bt_df["timestamp"].iloc[i]) if "timestamp" in bt_df.columns else ""

        # Deadband rebalance: only move to target when delta exceeds threshold
        delta = abs(target_ratio - position_ratio)
        if delta >= rebalance_threshold:
            cost = portfolio * delta * commission
            portfolio -= cost
            effective_ratio = target_ratio
        else:
            cost = 0.0
            effective_ratio = position_ratio

        target_ratios[i] = effective_ratio

        # Price return for one bar (using next bar's close; last bar stays flat)
        if i < n - 1:
            ret = float(close[i + 1]) / price - 1
        else:
            ret = 0.0
        portfolio *= 1 + effective_ratio * ret

        equity[i] = portfolio
        position_ratio = effective_ratio

        sig_vals = {col: round(float(bt_df[col].iloc[i]), 6)
                    for col in signal_cols}

        record = {
            "step": i,
            "timestamp": ts_str,
            "price": price,
            "action": action,
            "target_ratio": effective_ratio,
            "predicted_sharpe": round(float(sharpe_series.iloc[i]), 4),
            "signals": sig_vals,
            "shap_values": shap_i,
            "shap_expected_value": round(ev, 6),
            "portfolio_value": round(portfolio, 4),
            "commission_paid": round(cost, 4),
        }
        steps_log.append(record)

        # Trade tracking (only when a rebalance actually executed)
        prev_ratio = (steps_log[-2]["target_ratio"] if i > 0
                      else 0.0)
        if effective_ratio != prev_ratio:
            if open_trade is not None:
                entry_val = open_trade["entry_portfolio"]
                pnl_pct = (portfolio - entry_val) / max(entry_val, 1e-8)
                open_trade.update({
                    "exit_step": i,
                    "exit_timestamp": ts_str,
                    "exit_price": price,
                    "exit_target": effective_ratio,
                    "duration_bars": i - open_trade["entry_step"],
                    "pnl_pct": pnl_pct,
                    "exit_portfolio": portfolio,
                    "top_shap_at_entry": sorted(
                        open_trade["entry_shap"].items(),
                        key=lambda x: abs(x[1]), reverse=True
                    )[:4],
                })
                trades_log.append(open_trade)
                trade_idx += 1

            if effective_ratio > 0.0 or (i > 0 and prev_ratio > 0.0):
                open_trade = {
                    "trade_id": trade_idx,
                    "entry_step": i,
                    "entry_timestamp": ts_str,
                    "entry_price": price,
                    "entry_target": effective_ratio,
                    "from_ratio": prev_ratio,
                    "entry_portfolio": portfolio,
                    "entry_shap": shap_i,
                }

    equity_df = pd.DataFrame({
        "timestamp": bt_df["timestamp"].values,
        "portfolio_value": equity,
        "position_ratio": target_ratios,
        "close": close,
    })
    return steps_log, trades_log, equity_df


# ─────────────────────────────────────────── metrics

def _sharpe(returns: np.ndarray, ann: float = _TRADING_DAYS_PER_YEAR) -> float:
    if len(returns) < 2 or returns.std() < 1e-10:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(ann))


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / (peak + 1e-8)
    return float(dd.min())


def compute_metrics(equity_df: pd.DataFrame, trades_log: list[dict]) -> dict:
    equity = equity_df["portfolio_value"].values
    close = equity_df["close"].values
    daily_ret = np.diff(equity) / equity[:-1]
    bh_ret = np.diff(close) / close[:-1]

    total_return = float((equity[-1] / equity[0] - 1) * 100)
    bh_return = float((close[-1] / close[0] - 1) * 100)
    sharpe = _sharpe(daily_ret)
    max_dd = _max_drawdown(equity)
    n_trades = len(trades_log)
    wins = [t for t in trades_log if t.get("pnl_pct", 0) > 0]
    losses = [t for t in trades_log if t.get("pnl_pct", 0) <= 0]
    win_rate = len(wins) / max(n_trades, 1)
    gross_profit = sum(t["pnl_pct"] for t in wins)
    gross_loss = abs(sum(t["pnl_pct"] for t in losses))
    profit_factor = gross_profit / max(gross_loss, 1e-8)

    return {
        "total_return_pct": round(total_return, 2),
        "buy_hold_return_pct": round(bh_return, 2),
        "sharpe": round(sharpe, 4),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "n_trades": n_trades,
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 4),
    }


# ─────────────────────────────────────────── outputs

def write_attribution_jsonl(steps_log: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in steps_log:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {out_path}  ({len(steps_log)} lines)")


def write_trade_report(
    metrics: dict,
    trades_log: list[dict],
    equity_df: pd.DataFrame,
    out_path: Path,
    cfg: dict,
    run_name: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    c = cfg["crypto"]
    lines: list[str] = []

    lines += [
        f"# Trade Report — {run_name}",
        "",
        f"**Model**: GBDT Sharpe-regression combiner  ",
        f"**Symbol**: {c['symbol']}  |  **Timeframe**: {c.get('timeframe','1d')}  ",
        f"**Period**: {equity_df['timestamp'].iloc[0]} → {equity_df['timestamp'].iloc[-1]}  ",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total Return | **{metrics['total_return_pct']:+.2f}%** |",
        f"| Buy & Hold | {metrics['buy_hold_return_pct']:+.2f}% |",
        f"| Sharpe | {metrics['sharpe']:.4f} |",
        f"| Max Drawdown | {metrics['max_drawdown_pct']:.2f}% |",
        f"| Trades | {metrics['n_trades']} |",
        f"| Win Rate | {metrics['win_rate']*100:.1f}% |",
        f"| Profit Factor | {metrics['profit_factor']:.4f} |",
        "",
        "## Trade List (last 20)",
        "",
        "| # | Entry | Exit | From→To | PnL% | Duration | Top SHAP drivers |",
        "|---|-------|------|---------|------|----------|-----------------|",
    ]

    for t in trades_log[-20:]:
        top_shap = ", ".join(
            f"{k}={v:+.3f}" for k, v in t.get("top_shap_at_entry", [])[:3]
        )
        lines.append(
            f"| {t['trade_id']} "
            f"| {str(t['entry_timestamp'])[:10]} "
            f"| {str(t.get('exit_timestamp',''))[:10]} "
            f"| {t['from_ratio']:.0%}→{t['entry_target']:.0%} "
            f"| {t.get('pnl_pct', 0)*100:+.2f}% "
            f"| {t.get('duration_bars', 0)}d "
            f"| {top_shap} |"
        )

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")


def write_shap_plot(
    combiner: GBDTCombiner,
    X: np.ndarray,
    signal_cols: list[str],
    out_path: Path,
) -> None:
    """T4.4: SHAP bar chart — mean |SHAP| per feature."""
    shap_vals = combiner.shap_values(X)
    mean_abs = np.abs(shap_vals).mean(axis=0)

    order = np.argsort(mean_abs)[::-1]
    features = [signal_cols[i] for i in order]
    values = mean_abs[order]

    fig, ax = plt.subplots(figsize=(10, max(4, len(features) * 0.4)))
    bars = ax.barh(features[::-1], values[::-1],
                   color=plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(features))))
    ax.set_xlabel("Mean |SHAP value| (impact on predicted Sharpe)")
    ax.set_title("GBDT Feature Importance — SHAP (test set)")
    ax.axvline(0, color="black", linewidth=0.5)

    for bar, val in zip(bars, values[::-1]):
        ax.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def write_equity_plot(equity_df: pd.DataFrame, out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                   gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(equity_df["timestamp"], equity_df["portfolio_value"],
             label="Portfolio", color="steelblue")
    ax1b = ax1.twinx()
    ax1b.plot(equity_df["timestamp"], equity_df["close"],
              label="BTC Close", color="orange", alpha=0.4, linewidth=0.8)
    ax1.set_ylabel("Portfolio Value ($)")
    ax1b.set_ylabel("BTC Price")
    ax1.set_title("GBDT Combiner — Equity Curve vs BTC")
    ax1.legend(loc="upper left")
    ax1b.legend(loc="upper right")

    ax2.fill_between(equity_df["timestamp"], equity_df["position_ratio"],
                     alpha=0.6, color="steelblue", label="Position ratio")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Position")
    ax2.set_xlabel("Date")
    ax2.legend()

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


# ─────────────────────────────────────────── main

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path,
                        default=Path("models/saved/BTCUSDT_gbdt_1d"))
    parser.add_argument("--config", type=Path,
                        default=Path("config/stage4_1d_gbdt.yaml"))
    parser.add_argument("--test-start", default=None)
    parser.add_argument("--test-end", default=None)
    parser.add_argument("--commission", type=float, default=0.001)
    parser.add_argument("--position-mode", choices=["discrete", "continuous"],
                        default="discrete")
    parser.add_argument("--position-scale", type=float, default=1.0,
                        help="In continuous mode, target_ratio = clip(pred_sharpe / scale, 0, 1)")
    parser.add_argument("--vol-target", type=float, default=0.0,
                        help="Annualized vol target; 0 disables. e.g. 0.20 for 20%")
    parser.add_argument("--vol-window", type=int, default=168,
                        help="Rolling window (bars) for realized vol")
    parser.add_argument("--regime-gate", action="store_true",
                        help="Force position to 0 when close < daily SMA")
    parser.add_argument("--regime-ma-days", type=int, default=200,
                        help="SMA window (bars of --regime-resample) for regime gate")
    parser.add_argument("--regime-resample", default="1D",
                        help="Resample rule for regime gate timeframe (e.g. '1D', '1h', '15min')")
    parser.add_argument("--rebalance-threshold", type=float, default=0.01,
                        help="Deadband: skip rebalance when |target - current| < this")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    model_path = args.model
    if not model_path.with_suffix(".lgb").exists():
        # Accept path without extension too
        model_path = Path(str(model_path))
    if not model_path.with_suffix(".lgb").exists():
        print(f"ERROR: model not found at {model_path}.lgb", file=sys.stderr)
        return 1

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_name = args.run_name or model_path.stem
    signal_cols = load_signal_list(cfg["signals"]["config_path"])
    print(f"Loaded {len(signal_cols)} signals")

    # Load data
    bt_df = load_data(cfg, args.test_start, args.test_end,
                      vol_window=args.vol_window, regime_ma_days=args.regime_ma_days,
                      regime_resample=args.regime_resample)
    missing = [s for s in signal_cols if s not in bt_df.columns]
    if missing:
        print(f"ERROR: missing signal columns: {missing}", file=sys.stderr)
        return 1

    # Load model
    print(f"Loading combiner from {model_path} ...")
    combiner = GBDTCombiner.load(model_path)
    print(f"  Features: {combiner.feature_names}")
    print(f"  Val R²: {combiner.val_r2}")

    # Backtest
    steps_log, trades_log, equity_df = run_backtest(
        combiner, bt_df, signal_cols, commission=args.commission,
        position_mode=args.position_mode, position_scale=args.position_scale,
        vol_target=args.vol_target, use_regime_gate=args.regime_gate,
        rebalance_threshold=args.rebalance_threshold,
    )
    metrics = compute_metrics(equity_df, trades_log)

    print(f"\n── Backtest metrics ─────────────────────────────────────────")
    for k, v in metrics.items():
        print(f"  {k:<30} {v}")

    # Write outputs
    base = REPO_ROOT / "models" / "saved" / run_name
    logs_base = REPO_ROOT / "models" / "logs" / run_name

    write_attribution_jsonl(steps_log, logs_base / "attribution.jsonl")
    write_trade_report(metrics, trades_log, equity_df,
                       base / "trade_report.md", cfg, run_name)

    X = bt_df[signal_cols].values.astype("float32")
    write_shap_plot(combiner, X, signal_cols, base / "shap_importance.png")
    write_equity_plot(equity_df, base / "equity_curve.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
