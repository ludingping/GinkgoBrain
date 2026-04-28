"""
StatArb ETH-BTC 价差套利 — verdict-grade smoke backtest.

设计：docs/GinkgoBrain/StatArb-ETH-BTC价差-设计.md（v0.2）
测试：docs/GinkgoBrain/StatArb-ETH-BTC价差-测试用例.md

核心范式：
- 价差 ε_t = log(P_eth) - β·log(P_btc) - α，β 用 90 天滚动 OLS 估
- ADF 协整检验，maxlag=4 固定（防 autolag 跳跃噪声）
- z-score 用 shift(1) 防当期 spread 进入自身分母
- Hedge-ratio sizing：D_eth = D_total/(1+β)，D_btc = β·D_eth（让 PnL 严格跟随 dε_t）
- z>+2 / z<-2 入场；|z|<0.5 / |z|>4 / hold>30 天 出场
- ADF p≥0.10 时停止开仓

Usage:
    uv run python scripts/statarb_eth_btc_smoke.py
    uv run python scripts/statarb_eth_btc_smoke.py --z-entry 2.5 --z-exit 0.3
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agents.ppo_shared import resample_ohlcv  # noqa: E402
from utils.db import read_ohlcv  # noqa: E402

# ─── Constants ──────────────────────────────────────────────────────────────

BARS_PER_DAY_4H = 6  # 24 / 4
ADF_MAXLAG = 4       # 4h × 4 = 16 小时市场记忆，固定避免 autolag 跳跃


# ─── Data pipeline ──────────────────────────────────────────────────────────

def load_eth_btc_pair(
    start: str,
    end: str,
    timeframe: str = "4h",
) -> pd.DataFrame:
    """
    Load ETH/USDT and BTC/USDT, resample to timeframe, inner-join on timestamp.

    Returns DataFrame with columns:
        timestamp, btc_close, eth_close, log_btc, log_eth
    """
    print(f"Loading BTC/USDT from DB ({start} → {end})...")
    btc_1m = read_ohlcv("BTC/USDT", start=start, end=end, only_closed=True)
    print(f"  {len(btc_1m):,} rows of 1m → resample to {timeframe}")
    btc = resample_ohlcv(btc_1m, timeframe)[["timestamp", "close"]].rename(
        columns={"close": "btc_close"}
    )

    print(f"Loading ETH/USDT from DB ({start} → {end})...")
    eth_1m = read_ohlcv("ETH/USDT", start=start, end=end, only_closed=True)
    print(f"  {len(eth_1m):,} rows of 1m → resample to {timeframe}")
    eth = resample_ohlcv(eth_1m, timeframe)[["timestamp", "close"]].rename(
        columns={"close": "eth_close"}
    )

    df = btc.merge(eth, on="timestamp", how="inner").sort_values("timestamp").reset_index(drop=True)
    df["log_btc"] = np.log(df["btc_close"])
    df["log_eth"] = np.log(df["eth_close"])
    return df


def check_data_integrity(df: pd.DataFrame) -> None:
    """Abort on grossly broken data; warn on borderline."""
    print("\n=== 数据完整性预检 ===")
    n = len(df)
    print(f"  inner-join 后样本数: {n:,}")
    assert n > 0, "df 为空"
    assert df["timestamp"].is_unique, "时间戳有重复"
    assert (df["btc_close"] > 0).all() and (df["eth_close"] > 0).all(), "close 含非正数"
    assert df["btc_close"].notna().all() and df["eth_close"].notna().all(), "close 含 NaN"
    print(f"  时间范围 : {df['timestamp'].min()} → {df['timestamp'].max()}")
    if n < 5000:
        raise SystemExit(f"valid_rows={n} < 5000，样本不足，abort")
    if n < 12000:
        print(f"  WARN: valid_rows={n} < 12000，少于 5.5 年 4h；继续但请知悉")
    print("  ✓ 数据完整性 OK")


# ─── Cointegration math ─────────────────────────────────────────────────────

def compute_rolling_beta(
    df: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    """
    Rolling OLS β/α on log_eth ~ log_btc with right-open slice (no leak).

    For index i, fits regression on log_btc/log_eth[i-window : i] (i excluded).
    Adds columns: beta, alpha, spread = log_eth - β·log_btc - α.
    """
    out = df.copy()
    log_btc = out["log_btc"].to_numpy()
    log_eth = out["log_eth"].to_numpy()
    n = len(out)
    beta = np.full(n, np.nan)
    alpha = np.full(n, np.nan)
    for i in range(window, n):
        x = log_btc[i - window : i]
        y = log_eth[i - window : i]
        b, a = np.polyfit(x, y, 1)  # returns [slope, intercept]
        beta[i] = b
        alpha[i] = a
    out["beta"] = beta
    out["alpha"] = alpha
    out["spread"] = out["log_eth"] - out["beta"] * out["log_btc"] - out["alpha"]
    return out


def compute_rolling_adf(
    spread: pd.Series,
    adf_window: int,
    maxlag: int = ADF_MAXLAG,
    step: Optional[int] = None,
) -> pd.DataFrame:
    """
    Rolling ADF on spread with fixed maxlag (avoids autolag instability).

    Tests every `step` bars (default = adf_window, so once per window).
    Returns DataFrame indexed by spread.index with columns:
        adf_pvalue (ffilled), adf_usedlag.
    """
    if step is None:
        step = adf_window
    n = len(spread)
    pvals = np.full(n, np.nan)
    lags = np.full(n, np.nan)
    first_idx = spread.first_valid_index()
    if first_idx is None:
        return pd.DataFrame({"adf_pvalue": pvals, "adf_usedlag": lags}, index=spread.index)
    start = spread.index.get_loc(first_idx) + adf_window
    for i in range(start, n, step):
        seg = spread.iloc[i - adf_window : i].dropna()
        if len(seg) >= 50:
            r = adfuller(seg, maxlag=maxlag, autolag=None)
            pvals[i] = r[1]
            lags[i] = r[2]
    out = pd.DataFrame({"adf_pvalue": pvals, "adf_usedlag": lags}, index=spread.index)
    out["adf_pvalue"] = out["adf_pvalue"].ffill()
    return out


def compute_zscore(spread: pd.Series, window: int) -> pd.Series:
    """
    Z-score with shift(1) to exclude current spread from its own mean/std.

    spread[t] standardized using statistics over [t-window-1 .. t-1].
    """
    rolling_mean = spread.rolling(window).mean().shift(1)
    rolling_std = spread.rolling(window).std().shift(1)
    return (spread - rolling_mean) / rolling_std


# ─── Sizing & PnL ───────────────────────────────────────────────────────────

def hedge_ratio_legs(beta: float, d_total: float) -> tuple[float, float]:
    """
    Hedge-ratio sizing: D_eth = D_total / (1 + β), D_btc = β · D_eth.

    Total leg dollar = D_eth + D_btc = D_total invariant.
    Math: ΔPnL = D_eth · dε (PnL strictly tracks spread change).
    """
    if beta <= -1:
        raise ValueError(f"beta={beta} 致 (1+β) ≤ 0，无法分配两腿")
    d_eth = d_total / (1.0 + beta)
    d_btc = d_eth * beta
    return d_eth, d_btc


def compute_trade_pnl(
    direction: int,
    beta_open: float,
    eth_open: float,
    btc_open: float,
    eth_close: float,
    btc_close: float,
    d_total: float,
    commission: float,
) -> dict:
    """
    Trade PnL with hedge-ratio sizing.

    direction = +1 (long spread = long ETH + short BTC)
              = -1 (short spread)
    """
    d_eth, d_btc = hedge_ratio_legs(beta_open, d_total)
    qty_eth = direction * d_eth / eth_open
    qty_btc = -direction * d_btc / btc_open
    pnl_eth = qty_eth * (eth_close - eth_open)
    pnl_btc = qty_btc * (btc_close - btc_open)
    fee_open = (d_eth + d_btc) * commission
    fee_close = (abs(qty_eth * eth_close) + abs(qty_btc * btc_close)) * commission
    return {
        "d_eth": d_eth,
        "d_btc": d_btc,
        "qty_eth": qty_eth,
        "qty_btc": qty_btc,
        "pnl_eth": pnl_eth,
        "pnl_btc": pnl_btc,
        "fee": fee_open + fee_close,
        "pnl_net": pnl_eth + pnl_btc - fee_open - fee_close,
    }


# ─── State machine ──────────────────────────────────────────────────────────

@dataclass
class BacktestParams:
    z_entry: float = 2.0
    z_exit: float = 0.5
    z_stop: float = 4.0
    hold_max_days: int = 30
    bars_per_day: int = BARS_PER_DAY_4H
    adf_threshold: float = 0.10
    initial_balance: float = 10000.0
    leg_dollar_frac: float = 0.5
    commission: float = 0.0005


@dataclass
class Trade:
    open_idx: int
    open_ts: pd.Timestamp
    direction: int  # +1 long spread, -1 short spread
    beta_open: float
    eth_open: float
    btc_open: float
    d_eth: float
    d_btc: float
    close_idx: int = -1
    close_ts: Optional[pd.Timestamp] = None
    eth_close: float = 0.0
    btc_close: float = 0.0
    pnl_net: float = 0.0
    fee: float = 0.0
    reason: str = ""


def run_backtest(
    df: pd.DataFrame,
    params: BacktestParams,
) -> tuple[list[Trade], pd.Series]:
    """
    Iterate state machine: open on z-extreme + ADF pass, close on z_revert / stop / timeout.

    Required df columns: timestamp, btc_close, eth_close, beta, spread, z_score, adf_pvalue.
    Returns (trades, equity_curve).
    """
    n = len(df)
    equity = np.full(n, params.initial_balance, dtype=float)
    trades: list[Trade] = []
    open_trade: Optional[Trade] = None
    hold_max_bars = params.hold_max_days * params.bars_per_day

    ts = df["timestamp"].to_numpy()
    btc_close_arr = df["btc_close"].to_numpy()
    eth_close_arr = df["eth_close"].to_numpy()
    beta_arr = df["beta"].to_numpy()
    z_arr = df["z_score"].to_numpy()
    adf_p_arr = df["adf_pvalue"].to_numpy()

    realized_pnl = 0.0
    for i in range(n):
        if not np.isfinite(z_arr[i]) or not np.isfinite(beta_arr[i]):
            equity[i] = params.initial_balance + realized_pnl
            continue

        if open_trade is None:
            # 检查是否开仓
            if np.isfinite(adf_p_arr[i]) and adf_p_arr[i] < params.adf_threshold:
                if z_arr[i] < -params.z_entry:
                    direction = +1  # long spread
                elif z_arr[i] > params.z_entry:
                    direction = -1  # short spread
                else:
                    direction = 0
                if direction != 0:
                    d_total = params.initial_balance * params.leg_dollar_frac
                    d_eth, d_btc = hedge_ratio_legs(beta_arr[i], d_total)
                    open_trade = Trade(
                        open_idx=i,
                        open_ts=ts[i],
                        direction=direction,
                        beta_open=beta_arr[i],
                        eth_open=eth_close_arr[i],
                        btc_open=btc_close_arr[i],
                        d_eth=d_eth,
                        d_btc=d_btc,
                    )
        else:
            # 检查是否平仓
            hold_bars = i - open_trade.open_idx
            reason = None
            if abs(z_arr[i]) < params.z_exit:
                reason = "z_revert"
            elif abs(z_arr[i]) > params.z_stop:
                reason = "stop_loss"
            elif hold_bars >= hold_max_bars:
                reason = "timeout"
            if reason is not None:
                pnl_info = compute_trade_pnl(
                    direction=open_trade.direction,
                    beta_open=open_trade.beta_open,
                    eth_open=open_trade.eth_open,
                    btc_open=open_trade.btc_open,
                    eth_close=eth_close_arr[i],
                    btc_close=btc_close_arr[i],
                    d_total=open_trade.d_eth + open_trade.d_btc,
                    commission=params.commission,
                )
                open_trade.close_idx = i
                open_trade.close_ts = ts[i]
                open_trade.eth_close = eth_close_arr[i]
                open_trade.btc_close = btc_close_arr[i]
                open_trade.pnl_net = pnl_info["pnl_net"]
                open_trade.fee = pnl_info["fee"]
                open_trade.reason = reason
                realized_pnl += pnl_info["pnl_net"]
                trades.append(open_trade)
                open_trade = None

        # mark-to-market（持仓中加 unrealized；MTM 不算 fee）
        if open_trade is not None:
            unreal = compute_trade_pnl(
                direction=open_trade.direction,
                beta_open=open_trade.beta_open,
                eth_open=open_trade.eth_open,
                btc_open=open_trade.btc_open,
                eth_close=eth_close_arr[i],
                btc_close=btc_close_arr[i],
                d_total=open_trade.d_eth + open_trade.d_btc,
                commission=0.0,
            )
            equity[i] = params.initial_balance + realized_pnl + unreal["pnl_eth"] + unreal["pnl_btc"]
        else:
            equity[i] = params.initial_balance + realized_pnl

    return trades, pd.Series(equity, index=df.index, name="equity")


# ─── Metrics ────────────────────────────────────────────────────────────────

def compute_metrics(
    equity: pd.Series,
    trades: list[Trade],
    bars_per_year: int = BARS_PER_DAY_4H * 365,
) -> dict:
    """Sharpe / total_return / max_dd / calmar / trade stats."""
    rets = equity.pct_change().dropna()
    if len(rets) < 2 or rets.std() == 0:
        sharpe = 0.0
    else:
        sharpe = float(rets.mean() / rets.std() * np.sqrt(bars_per_year))
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) > 0 else 0.0
    running_max = equity.cummax()
    dd = (equity - running_max) / running_max
    max_dd = float(dd.min()) if len(dd) > 0 else 0.0
    years = len(equity) / bars_per_year if bars_per_year > 0 else 1.0
    if len(equity):
        cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / max(years, 1e-9)) - 1
    else:
        cagr = 0.0
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else float("inf")

    n_trades = len(trades)
    wins = sum(1 for t in trades if t.pnl_net > 0)
    avg_hold_days = (
        np.mean([(t.close_idx - t.open_idx) / BARS_PER_DAY_4H for t in trades])
        if trades else 0.0
    )
    reasons: dict[str, int] = {}
    for t in trades:
        reasons[t.reason] = reasons.get(t.reason, 0) + 1

    return {
        "sharpe": sharpe,
        "total_return": total_return,
        "max_dd": max_dd,
        "calmar": calmar,
        "n_trades": n_trades,
        "win_rate": wins / n_trades if n_trades else 0.0,
        "avg_hold_days": float(avg_hold_days),
        "reasons": reasons,
    }


# ─── Plotting ───────────────────────────────────────────────────────────────

def plot_diagnostics(
    df: pd.DataFrame,
    equity: pd.Series,
    trades: list[Trade],
    train_end_idx: int,
    output_dir: Path,
    output_stem: str,
) -> list[Path]:
    """Generate 4 PNGs: β/ADF rolling, spread+z, equity vs BTC hold, trades on z."""
    output_dir.mkdir(parents=True, exist_ok=True)
    png_paths = []

    ts = df["timestamp"]

    # 1. β + ADF p-value
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(ts, df["beta"], color="C0", lw=0.8)
    axes[0].axvline(ts.iloc[train_end_idx], ls="--", c="gray", label="Train/Test split")
    axes[0].set_ylabel("Rolling β (90d)")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.3)
    axes[1].plot(ts, df["adf_pvalue"], color="C1", lw=0.8)
    axes[1].axhline(0.10, ls=":", c="red", label="p=0.10 阈值")
    axes[1].axvline(ts.iloc[train_end_idx], ls="--", c="gray")
    axes[1].set_ylabel("ADF p-value")
    axes[1].set_xlabel("Time")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3)
    fig.suptitle("协整稳定性诊断：β 滚动 + ADF p-value")
    fig.tight_layout()
    p = output_dir / f"{output_stem}_cointegration.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    png_paths.append(p)

    # 2. Spread + z-score
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(ts, df["spread"], color="C2", lw=0.5)
    axes[0].axvline(ts.iloc[train_end_idx], ls="--", c="gray")
    axes[0].set_ylabel("Spread ε_t")
    axes[0].grid(alpha=0.3)
    axes[1].plot(ts, df["z_score"], color="C3", lw=0.5)
    axes[1].axhline(2, ls=":", c="green", lw=0.8)
    axes[1].axhline(-2, ls=":", c="green", lw=0.8, label="z=±2 入场")
    axes[1].axhline(0.5, ls=":", c="orange", lw=0.8)
    axes[1].axhline(-0.5, ls=":", c="orange", lw=0.8, label="z=±0.5 出场")
    axes[1].axvline(ts.iloc[train_end_idx], ls="--", c="gray")
    axes[1].set_ylabel("Z-score (shift(1))")
    axes[1].set_xlabel("Time")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3)
    fig.suptitle("Spread + Z-score（shift(1) 防泄漏）")
    fig.tight_layout()
    p = output_dir / f"{output_stem}_spread_zscore.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    png_paths.append(p)

    # 3. Equity vs BTC buy-and-hold
    fig, ax = plt.subplots(figsize=(12, 5))
    initial = equity.iloc[0]
    btc_norm = df["btc_close"] / df["btc_close"].iloc[0] * initial
    eth_norm = df["eth_close"] / df["eth_close"].iloc[0] * initial
    ax.plot(ts, equity, label="StatArb equity", color="C0", lw=1.2)
    ax.plot(ts, btc_norm, label="BTC buy-and-hold", color="orange", lw=0.8, alpha=0.7)
    ax.plot(ts, eth_norm, label="ETH buy-and-hold", color="purple", lw=0.8, alpha=0.7)
    ax.axvline(ts.iloc[train_end_idx], ls="--", c="gray", label="Train/Test split")
    ax.set_ylabel("Portfolio value (USD)")
    ax.set_xlabel("Time")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.suptitle("StatArb equity vs BTC/ETH buy-and-hold")
    fig.tight_layout()
    p = output_dir / f"{output_stem}_equity.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    png_paths.append(p)

    # 4. Trades on z-score
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(ts, df["z_score"], color="gray", lw=0.4, alpha=0.6)
    ax.axhline(0, color="black", lw=0.4)
    ax.axhline(2, ls=":", c="green", lw=0.6)
    ax.axhline(-2, ls=":", c="green", lw=0.6)
    ax.axhline(4, ls=":", c="red", lw=0.6)
    ax.axhline(-4, ls=":", c="red", lw=0.6)
    for t in trades:
        col = "green" if t.pnl_net > 0 else "red"
        z_open = df["z_score"].iloc[t.open_idx]
        z_close = df["z_score"].iloc[t.close_idx] if t.close_idx >= 0 else None
        ax.scatter(ts.iloc[t.open_idx], z_open, marker="^", color=col, s=20, alpha=0.7)
        if z_close is not None:
            ax.scatter(ts.iloc[t.close_idx], z_close, marker="v", color=col, s=20, alpha=0.7)
    ax.axvline(ts.iloc[train_end_idx], ls="--", c="gray")
    ax.set_ylabel("Z-score")
    ax.set_xlabel("Time")
    ax.set_title("交易点位：▲ 开仓 / ▼ 平仓（绿盈红亏）")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = output_dir / f"{output_stem}_trades.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    png_paths.append(p)

    return png_paths


# ─── Recap markdown ─────────────────────────────────────────────────────────

def _format_metrics_table(name: str, m: dict) -> str:
    return (
        f"### {name} 段业绩\n\n"
        f"| 指标 | 值 |\n|---|---|\n"
        f"| Sharpe | {m['sharpe']:.3f} |\n"
        f"| Total Return | {m['total_return']*100:.2f}% |\n"
        f"| Max DD | {m['max_dd']*100:.2f}% |\n"
        f"| Calmar | {m['calmar']:.3f} |\n"
        f"| 总交易数 | {m['n_trades']} |\n"
        f"| 胜率 | {m['win_rate']*100:.1f}% |\n"
        f"| 平均持仓天数 | {m['avg_hold_days']:.2f} |\n"
        f"| 出场原因 | {m['reasons']} |\n"
    )


def _format_verdict(test_metrics: dict, train_adf_pass_rate: float, test_adf_pass_rate: float) -> tuple[bool, str]:
    """5-criterion AND verdict per design §9.1."""
    checks = [
        ("OOS Sharpe > 0.5", test_metrics["sharpe"] > 0.5, f"{test_metrics['sharpe']:.3f}"),
        ("ADF 通过率 > 70%", test_adf_pass_rate > 0.70, f"{test_adf_pass_rate*100:.1f}%"),
        ("交易数 ≥ 30", test_metrics["n_trades"] >= 30, f"{test_metrics['n_trades']}"),
        ("Max DD < 20%", abs(test_metrics["max_dd"]) < 0.20, f"{test_metrics['max_dd']*100:.2f}%"),
        ("Calmar > 0.5", test_metrics["calmar"] > 0.5, f"{test_metrics['calmar']:.3f}"),
    ]
    all_pass = all(c[1] for c in checks)
    lines = ["### Done 判定（设计 §9.1）\n", "| 指标 | 阈值 | 实测 | 结果 |", "|---|---|---|---|"]
    for name, passed, val in checks:
        lines.append(f"| {name} | — | {val} | {'✅' if passed else '❌'} |")
    lines.append(f"\n**Verdict: {'✅ 通过 — 进产品化' if all_pass else '❌ 不通过 — 进 B2 横截面 PCA / 关 ETH 线'}**\n")
    return all_pass, "\n".join(lines)


def write_recap(
    args,
    df: pd.DataFrame,
    train_metrics: dict,
    test_metrics: dict,
    train_adf_pass: float,
    test_adf_pass: float,
    trades: list[Trade],
    train_end_ts: pd.Timestamp,
    png_paths: list[Path],
    output_path: Path,
) -> bool:
    """Write recap markdown; returns True if verdict pass."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    verdict_pass, verdict_md = _format_verdict(test_metrics, train_adf_pass, test_adf_pass)

    summary = (
        f"**Verdict**: {'✅ 通过' if verdict_pass else '❌ 不通过'} — "
        f"OOS Sharpe={test_metrics['sharpe']:.2f}, "
        f"trades={test_metrics['n_trades']}, "
        f"DD={test_metrics['max_dd']*100:.1f}%, "
        f"ADF 通过率={test_adf_pass*100:.0f}%。"
    )

    trade_rows = []
    for t in trades:
        if t.close_ts is None:
            continue
        direction_label = "long_spread" if t.direction > 0 else "short_spread"
        trade_rows.append(
            f"| {t.open_ts} | {t.close_ts} | {direction_label} | {t.beta_open:.3f} | "
            f"{t.d_eth:.0f} | {t.d_btc:.0f} | {t.pnl_net:+.2f} | {t.fee:.2f} | {t.reason} |"
        )
    trade_table = (
        "| open_ts | close_ts | direction | β@open | D_eth | D_btc | pnl_net | fee | reason |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
        + "\n".join(trade_rows[:200])
        + (f"\n\n*（共 {len(trade_rows)} 笔，仅展示前 200 笔）*" if len(trade_rows) > 200 else "")
    )

    def relpath(p: Path) -> str:
        try:
            return str(p.relative_to(output_path.parent))
        except ValueError:
            return str(p)

    img_md = "\n".join(f"![{p.stem}]({relpath(p)})" for p in png_paths)

    md = f"""# StatArb ETH-BTC 价差套利 — 复盘报告

> **日期**：2026-04-29（脚本运行日期；数据 cutoff {df['timestamp'].max()}）
> **设计**：[../StatArb-ETH-BTC价差-设计.md](../StatArb-ETH-BTC价差-设计.md)（v0.2）
> **测试**：[../StatArb-ETH-BTC价差-测试用例.md](../StatArb-ETH-BTC价差-测试用例.md)

## 1. 执行摘要

{summary}

## 2. 运行参数

| 参数 | 值 |
|---|---|
| timeframe | {args.timeframe} |
| 样本期 | {args.start} ~ {args.end} |
| β rolling window | {args.beta_window_days} 天 ({args.beta_window_days * BARS_PER_DAY_4H} 根 4h) |
| ADF window | {args.adf_window_days} 天 (maxlag={ADF_MAXLAG}, autolag=None) |
| z-score | entry=±{args.z_entry} / exit=±{args.z_exit} / stop=±{args.z_stop} |
| 持仓上限 | {args.hold_max_days} 天 |
| Train/Test 切点 | train_ratio={args.train_ratio}（{train_end_ts}） |
| 手续费 | {args.commission*100:.3f}% 单边 ×4 笔 = {args.commission*4*100:.2f}% 双边 |
| 仓位 | initial={args.initial_balance:.0f}, leg_dollar_frac={args.leg_dollar_frac}, hedge-ratio sizing |

## 3. 协整稳定性诊断

- ADF 通过率（Train）：{train_adf_pass*100:.1f}%
- ADF 通过率（Test）：{test_adf_pass*100:.1f}%
- ADF usedlag 是否稳定 = {ADF_MAXLAG}：{(df['adf_usedlag'].dropna() == ADF_MAXLAG).all()}
- β 中位数（全样本）：{df['beta'].median():.3f}
- β 95% 分位（全样本）：{df['beta'].quantile(0.95):.3f}
- β 5% 分位（全样本）：{df['beta'].quantile(0.05):.3f}

## 4. 业绩指标

{_format_metrics_table('Train (IS)', train_metrics)}

{_format_metrics_table('Test (OOS)', test_metrics)}

## 5. {verdict_md}

## 6. 可视化

{img_md}

## 7. 完整 Trade List

{trade_table}

## 8. 下一步

{'**进入产品化设计**：下一份设计稿 `StatArb-ETH-BTC价差-产品化设计.md`，加 perp + 资金费率 / paper trading / 多 pair 扩展。' if verdict_pass else '**关闭单 pair 尝试**：进入 §6 失败诊断；按情况转 B2 横截面 PCA 残差或彻底关闭 ETH 线。'}
"""
    output_path.write_text(md)
    print(f"\n✓ 复盘报告写入: {output_path}")
    return verdict_pass


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="StatArb ETH-BTC 4h 价差套利 verdict-grade smoke")
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2026-04-13")
    parser.add_argument("--timeframe", default="4h")
    parser.add_argument("--beta-window-days", type=int, default=90)
    parser.add_argument("--adf-window-days", type=int, default=30)
    parser.add_argument("--z-entry", type=float, default=2.0)
    parser.add_argument("--z-exit", type=float, default=0.5)
    parser.add_argument("--z-stop", type=float, default=4.0)
    parser.add_argument("--hold-max-days", type=int, default=30)
    parser.add_argument("--train-ratio", type=float, default=0.50)
    parser.add_argument("--commission", type=float, default=0.0005)
    parser.add_argument("--initial-balance", type=float, default=10000.0)
    parser.add_argument("--leg-dollar-frac", type=float, default=0.5)
    parser.add_argument(
        "--output",
        default="docs/GinkgoBrain/复盘/2026-04-29_statarb_eth_btc.md",
        help="复盘报告路径（相对 GinkgoRoad 仓库根；绝对路径直接使用）",
    )
    args = parser.parse_args()

    # 1. Load & integrity
    df = load_eth_btc_pair(args.start, args.end, args.timeframe)
    check_data_integrity(df)

    # 2. Rolling β + spread
    beta_window = args.beta_window_days * BARS_PER_DAY_4H
    print(f"\n=== 滚动 β 拟合 (window={beta_window} bars = {args.beta_window_days} 天) ===")
    df = compute_rolling_beta(df, window=beta_window)
    print(f"  β 中位数: {df['beta'].median():.4f}")
    print(f"  β 范围  : [{df['beta'].quantile(0.05):.3f}, {df['beta'].quantile(0.95):.3f}]")

    # 3. Z-score
    df["z_score"] = compute_zscore(df["spread"], window=beta_window)

    # 4. ADF
    adf_window = args.adf_window_days * BARS_PER_DAY_4H
    print(f"\n=== 滚动 ADF (window={adf_window} bars = {args.adf_window_days} 天, maxlag={ADF_MAXLAG}) ===")
    adf_df = compute_rolling_adf(df["spread"], adf_window=adf_window, maxlag=ADF_MAXLAG)
    df["adf_pvalue"] = adf_df["adf_pvalue"].values
    df["adf_usedlag"] = adf_df["adf_usedlag"].values
    valid_adf = df["adf_pvalue"].dropna()
    overall_pass = (valid_adf < 0.10).mean() if len(valid_adf) > 0 else 0.0
    print(f"  ADF 整体通过率: {overall_pass*100:.1f}%")
    if len(df["adf_usedlag"].dropna()) > 0:
        print(f"  ADF usedlag 全部 == {ADF_MAXLAG}: {(df['adf_usedlag'].dropna() == ADF_MAXLAG).all()}")

    # 5. Walk-forward split
    train_end_idx = int(len(df) * args.train_ratio)
    train_end_ts = df["timestamp"].iloc[train_end_idx]
    print(f"\n=== Walk-forward split: train_ratio={args.train_ratio} ===")
    print(f"  Train: 0 ~ {train_end_idx} ({df['timestamp'].iloc[0]} ~ {train_end_ts})")
    print(f"  Test : {train_end_idx} ~ {len(df)} ({train_end_ts} ~ {df['timestamp'].iloc[-1]})")

    # 6. Backtest
    params = BacktestParams(
        z_entry=args.z_entry,
        z_exit=args.z_exit,
        z_stop=args.z_stop,
        hold_max_days=args.hold_max_days,
        bars_per_day=BARS_PER_DAY_4H,
        initial_balance=args.initial_balance,
        leg_dollar_frac=args.leg_dollar_frac,
        commission=args.commission,
    )
    print("\n=== 回测全样本 ===")
    trades, equity = run_backtest(df, params)
    print(f"  总交易数: {len(trades)}")

    # 7. Split metrics
    train_equity = equity.iloc[:train_end_idx]
    test_equity = equity.iloc[train_end_idx:]
    train_trades = [t for t in trades if t.open_idx < train_end_idx]
    test_trades = [t for t in trades if t.open_idx >= train_end_idx]

    train_metrics = compute_metrics(train_equity, train_trades)
    test_metrics = compute_metrics(test_equity, test_trades)

    # 8. ADF pass rate per split
    train_adf_valid = df["adf_pvalue"].iloc[:train_end_idx].dropna()
    test_adf_valid = df["adf_pvalue"].iloc[train_end_idx:].dropna()
    train_adf_pass = (train_adf_valid < 0.10).mean() if len(train_adf_valid) > 0 else 0.0
    test_adf_pass = (test_adf_valid < 0.10).mean() if len(test_adf_valid) > 0 else 0.0

    # 9. Print summary
    print("\n=== 业绩 ===")
    print(f"  Train Sharpe={train_metrics['sharpe']:.3f}, Return={train_metrics['total_return']*100:.2f}%, "
          f"DD={train_metrics['max_dd']*100:.2f}%, n_trades={train_metrics['n_trades']}")
    print(f"  Test  Sharpe={test_metrics['sharpe']:.3f}, Return={test_metrics['total_return']*100:.2f}%, "
          f"DD={test_metrics['max_dd']*100:.2f}%, n_trades={test_metrics['n_trades']}")
    print(f"  ADF 通过率 Train={train_adf_pass*100:.1f}% / Test={test_adf_pass*100:.1f}%")

    # 10. Plots & recap
    print("\n=== 生成复盘报告 ===")
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path("/home/davidlyu/projects/GinkgoRoad") / args.output
    output_dir = output_path.parent
    png_paths = plot_diagnostics(
        df, equity, trades, train_end_idx,
        output_dir=output_dir,
        output_stem=output_path.stem,
    )
    verdict_pass = write_recap(
        args, df, train_metrics, test_metrics,
        train_adf_pass, test_adf_pass,
        trades, train_end_ts, png_paths, output_path,
    )

    print(f"\n[verdict: {'✓ 通过' if verdict_pass else '✗ 不通过'} done 判定]")
    return 0 if verdict_pass else 1


if __name__ == "__main__":
    sys.exit(main())
