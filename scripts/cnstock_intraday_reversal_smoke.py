"""
300866 5min T+0 局部反转 — verdict-grade smoke backtest.

设计：docs/GinkgoBrain/300866-5min-T0反转-设计.md（v0.3）
测试：docs/GinkgoBrain/300866-5min-T0反转-测试用例.md

核心范式：
- 不预测方向，捕捉极端 z-score 偏离 → 回归
- portfolio_ratio ∈ {0.5, 0.75, 1.0} 三档
- z>+2 → 0.5（卖底仓一半）；z<-2 → 1.0（用现金买回）；|z|<0.5 → 0.75（中性）

A 股微观结构（v0.2/v0.3）：
- T+1：当日买入次日才能卖（available_shares 追踪）
- 滑点：买×1.001 / 卖×0.999
- 印花税：卖出 0.10%
- 涨跌停：±20% 截断
- 前复权：5m 是未复权，day 是已复权 → 反推 adj_factor 修复

Usage:
    uv run python scripts/cnstock_intraday_reversal_smoke.py \\
        --code 300866 --start 2023-04-04 --end 2026-04-22 \\
        --sma-window 24 --z-entry 2.0 --z-exit 0.5
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

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.db import read_cnstock_kline_5m, read_cnstock_kline_day  # noqa: E402

# ─── Constants ──────────────────────────────────────────────────────────────

MIN_LOT_SHARES = 100
COMMISSION_BUY = 0.00025
COMMISSION_SELL = 0.00125
SLIPPAGE = 0.001
LIMIT_THRESHOLD = 0.195
BARS_PER_DAY = 48
TRADING_DAYS_PER_YEAR = 252


# ─── Data loading & forward adjustment ──────────────────────────────────────


def load_aligned_5m(
    code: str,
    start: str,
    end: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Load 5m + day data, apply forward-adjustment, return aligned 5m DataFrame.
    """
    print(f"Loading 5m K-line for {code} from {start} to {end}...")
    df_5m = read_cnstock_kline_5m(code, start=start, end=end)
    print(f"  {len(df_5m):,} rows raw")

    print(f"Loading day K-line for {code}...")
    day_start = (pd.Timestamp(start) - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    df_day = read_cnstock_kline_day(code, start=day_start, end=end)
    print(f"  {len(df_day):,} rows day")

    df_5m = df_5m.copy()
    df_5m["ts_cst"] = df_5m["trade_time"].dt.tz_convert("Asia/Shanghai")
    df_5m["hour_cst"] = df_5m["ts_cst"].dt.hour
    df_5m["minute_cst"] = df_5m["ts_cst"].dt.minute
    df_5m["date_cst"] = df_5m["ts_cst"].dt.date

    df_day = df_day.copy()
    df_day["ts_cst"] = df_day["trade_time"].dt.tz_convert("Asia/Shanghai")
    df_day["date_cst"] = df_day["ts_cst"].dt.date

    in_morning = (
        ((df_5m["hour_cst"] == 9) & (df_5m["minute_cst"] >= 35))
        | (df_5m["hour_cst"] == 10)
        | ((df_5m["hour_cst"] == 11) & (df_5m["minute_cst"] <= 30))
    )
    in_afternoon = (
        ((df_5m["hour_cst"] == 13))
        | (df_5m["hour_cst"] == 14)
        | ((df_5m["hour_cst"] == 15) & (df_5m["minute_cst"] == 0))
    )
    df_5m = df_5m[in_morning | in_afternoon].reset_index(drop=True)
    df_5m["session_bar"] = df_5m.groupby("date_cst").cumcount() + 1

    last_5m_unadj = df_5m.groupby("date_cst")["close_price"].last()
    day_close_indexed = df_day.set_index("date_cst")["close_price"]
    adj_factor = (day_close_indexed / last_5m_unadj).rename("adj_factor").dropna()
    print(
        f"  adj_factor: median={adj_factor.median():.4f}, "
        f"min={adj_factor.min():.4f}, max={adj_factor.max():.4f}"
    )

    df_5m = df_5m.merge(adj_factor.reset_index(), on="date_cst", how="inner")
    for col in ["open_price", "high_price", "low_price", "close_price"]:
        df_5m[col] = df_5m[col] * df_5m["adj_factor"]

    daily_pre_close = day_close_indexed.shift(1).rename("daily_pre_close")
    df_5m = df_5m.merge(daily_pre_close.reset_index(), on="date_cst", how="left")

    last_5m_adj = df_5m.groupby("date_cst")["close_price"].last()
    diff = (last_5m_adj - day_close_indexed).abs() / day_close_indexed
    valid = diff.dropna()
    if len(valid) > 0:
        max_diff = valid.max()
        if max_diff > 1e-4:
            raise SystemExit(
                f"复权对齐校验失败：5m_last_close vs day_close max diff = {max_diff*100:.6f}%"
            )
        print(f"  ✓ 复权对齐校验通过：max diff = {max_diff*100:.6f}%")

    return df_5m, df_day, adj_factor


def check_data_integrity(df_5m: pd.DataFrame) -> None:
    print("\n=== 数据完整性预检 ===")
    n = len(df_5m)
    print(f"  5m 样本数: {n:,}")
    if n < 30000:
        raise SystemExit(f"valid_rows={n} < 30000，abort")
    daily_bars = df_5m.groupby("date_cst").size()
    print(f"  单日 5m bar 数中位 / 最小 / 最大: {daily_bars.median():.0f} / {daily_bars.min()} / {daily_bars.max()}")
    if daily_bars.median() < 40:
        print(f"  WARN: 单日 bar 数中位偏少")
    susp_pct = (df_5m["suspension"] == 1).mean()
    print(f"  suspension=1 占比: {susp_pct*100:.2f}%")
    if susp_pct > 0.10:
        print(f"  WARN: suspension > 10%")
    print(f"  时间范围: {df_5m['ts_cst'].min()} → {df_5m['ts_cst'].max()}")


# ─── Z-score (shift(1) 防泄漏) ──────────────────────────────────────────────


def compute_zscore(close: pd.Series, window: int) -> pd.Series:
    rolling_mean = close.rolling(window).mean().shift(1)
    rolling_std = close.rolling(window).std().shift(1)
    return (close - rolling_mean) / rolling_std


# ─── Backtest ───────────────────────────────────────────────────────────────


@dataclass
class BacktestParams:
    sma_window: int = 24
    z_entry: float = 2.0
    z_exit: float = 0.5
    initial_balance: float = 200_000.0
    max_daily_switches: int = 4


@dataclass
class Trade:
    open_idx: int
    open_ts: pd.Timestamp
    direction: int
    ratio_from: float
    ratio_to: float
    delta_shares_ideal: int
    delta_shares_actual: int
    exec_price: float
    cash_after: float
    shares_after: int
    available_after: int
    truncated_by_t1: bool
    truncated_by_limit: bool


def run_backtest(
    df: pd.DataFrame,
    params: BacktestParams,
) -> tuple[list[Trade], pd.Series, dict]:
    n = len(df)
    close = df["close_price"].to_numpy()
    z = df["z_score"].to_numpy()
    suspended = (df["suspension"] == 1).to_numpy()
    sess_bar = df["session_bar"].to_numpy()
    daily_pc = df["daily_pre_close"].to_numpy()
    dates = df["date_cst"].to_numpy()
    ts = df["ts_cst"].to_numpy()

    initial_close = float(close[0])
    shares = int((params.initial_balance * 1.0) // (initial_close * MIN_LOT_SHARES) * MIN_LOT_SHARES)
    cash = params.initial_balance - shares * initial_close
    available = shares
    ratio = 1.0
    daily_switch_count = 0
    last_date = None

    equity = np.zeros(n)
    trades: list[Trade] = []
    t1_truncations = 0
    t1_pnl_loss_estimate = 0.0

    for i in range(n):
        if dates[i] != last_date:
            daily_switch_count = 0
            available = shares
            last_date = dates[i]

        is_first_bar = sess_bar[i] == 1
        is_tail = sess_bar[i] >= 46
        is_susp = bool(suspended[i])

        is_limit_up = False
        is_limit_down = False
        if not np.isnan(daily_pc[i]) and daily_pc[i] > 0:
            change = (close[i] - daily_pc[i]) / daily_pc[i]
            is_limit_up = change >= LIMIT_THRESHOLD
            is_limit_down = change <= -LIMIT_THRESHOLD

        is_locked = (
            is_susp or is_first_bar or is_tail
            or daily_switch_count >= params.max_daily_switches
        )

        new_ratio = ratio
        if not is_locked and not np.isnan(z[i]):
            if z[i] > params.z_entry:
                new_ratio = 0.5
            elif z[i] < -params.z_entry:
                new_ratio = 1.0
            elif abs(z[i]) < params.z_exit:
                new_ratio = 0.75

        truncated_by_limit = False
        if is_limit_up and new_ratio > ratio:
            new_ratio = ratio
            truncated_by_limit = True
        if is_limit_down and new_ratio < ratio:
            new_ratio = ratio
            truncated_by_limit = True

        if new_ratio != ratio:
            portfolio_value = cash + shares * close[i]
            target_shares_ideal = int(
                (portfolio_value * new_ratio) // (close[i] * MIN_LOT_SHARES) * MIN_LOT_SHARES
            )
            delta_ideal = target_shares_ideal - shares

            if delta_ideal > 0:
                exec_price = close[i] * (1 + SLIPPAGE)
                cost_per_share = exec_price * (1 + COMMISSION_BUY)
                max_buyable = int((cash // (cost_per_share * MIN_LOT_SHARES)) * MIN_LOT_SHARES)
                actual_delta = min(delta_ideal, max_buyable)
                if actual_delta >= MIN_LOT_SHARES:
                    cash -= actual_delta * cost_per_share
                    shares += actual_delta
                    trades.append(Trade(
                        open_idx=i, open_ts=ts[i], direction=+1,
                        ratio_from=ratio, ratio_to=new_ratio,
                        delta_shares_ideal=delta_ideal, delta_shares_actual=actual_delta,
                        exec_price=float(exec_price), cash_after=cash,
                        shares_after=shares, available_after=available,
                        truncated_by_t1=False, truncated_by_limit=truncated_by_limit,
                    ))
                    daily_switch_count += 1
                    ratio = (shares * close[i]) / (cash + shares * close[i])

            elif delta_ideal < 0:
                exec_price = close[i] * (1 - SLIPPAGE)
                want_sell = abs(delta_ideal)
                sellable = min(want_sell, available)
                sellable = (sellable // MIN_LOT_SHARES) * MIN_LOT_SHARES
                if sellable >= MIN_LOT_SHARES:
                    actual_delta = -sellable
                    truncated_by_t1 = sellable < want_sell
                    if truncated_by_t1:
                        t1_truncations += 1
                        t1_pnl_loss_estimate += (want_sell - sellable) * exec_price * 0.005
                    cash += sellable * exec_price * (1 - COMMISSION_SELL)
                    shares -= sellable
                    available -= sellable
                    trades.append(Trade(
                        open_idx=i, open_ts=ts[i], direction=-1,
                        ratio_from=ratio, ratio_to=new_ratio,
                        delta_shares_ideal=delta_ideal, delta_shares_actual=actual_delta,
                        exec_price=float(exec_price), cash_after=cash,
                        shares_after=shares, available_after=available,
                        truncated_by_t1=truncated_by_t1,
                        truncated_by_limit=truncated_by_limit,
                    ))
                    daily_switch_count += 1
                    ratio = (shares * close[i]) / (cash + shares * close[i])

        equity[i] = cash + shares * close[i]

    stats = {
        "n_trades": len(trades),
        "n_t1_truncations": t1_truncations,
        "t1_pnl_loss_estimate": t1_pnl_loss_estimate,
        "n_limit_truncations": sum(1 for t in trades if t.truncated_by_limit),
        "final_shares": shares,
        "final_cash": cash,
    }
    return trades, pd.Series(equity, index=df.index, name="equity"), stats


# ─── Metrics ────────────────────────────────────────────────────────────────


def compute_metrics(equity: pd.Series, df: pd.DataFrame) -> dict:
    if len(equity) < 2:
        return {"sharpe": 0.0, "total_return_pct": 0.0, "max_drawdown_pct": 0.0, "calmar": 0.0}

    daily_eq = equity.groupby(df["date_cst"].values).last()
    daily_ret = daily_eq.pct_change().dropna()

    if len(daily_ret) < 2 or daily_ret.std() == 0:
        sharpe = 0.0
    else:
        sharpe = float(daily_ret.mean() / daily_ret.std() * np.sqrt(TRADING_DAYS_PER_YEAR))

    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1)
    peak = equity.cummax()
    dd = float(((equity - peak) / peak).min())
    n_years = len(daily_eq) / TRADING_DAYS_PER_YEAR
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / max(n_years, 1e-9)) - 1 if n_years > 0 else 0.0
    calmar = float(cagr / abs(dd)) if dd < 0 else float("inf")

    return {
        "sharpe": sharpe,
        "total_return_pct": total_return * 100,
        "max_drawdown_pct": dd * 100,
        "calmar": calmar,
    }


def buy_and_hold_metrics(df: pd.DataFrame, initial_balance: float) -> tuple[pd.Series, dict]:
    close = df["close_price"].to_numpy()
    initial_shares = int((initial_balance) // (close[0] * MIN_LOT_SHARES) * MIN_LOT_SHARES)
    initial_cost = initial_shares * close[0] * (1 + COMMISSION_BUY)
    cash = initial_balance - initial_cost
    eq = cash + initial_shares * close
    eq_series = pd.Series(eq, index=df.index)
    metrics = compute_metrics(eq_series, df)
    return eq_series, metrics


# ─── Plotting ───────────────────────────────────────────────────────────────


def plot_diagnostics(
    df: pd.DataFrame,
    equity: pd.Series,
    bh_equity: pd.Series,
    trades: list[Trade],
    adj_factor: pd.Series,
    train_end_idx: int,
    output_dir: Path,
    output_stem: str,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    ts = df["ts_cst"]

    # 1. adj_factor
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(adj_factor.index, adj_factor.values, color="C0", lw=1.0)
    ax.set_ylabel("adj_factor")
    ax.set_xlabel("Date")
    ax.set_title("Forward-adjustment factor over time")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = output_dir / f"{output_stem}_adj_factor.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(p)

    # 2. z-score + trades
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(ts, df["z_score"], color="gray", lw=0.4, alpha=0.5)
    ax.axhline(0, color="black", lw=0.5)
    ax.axhline(2, ls=":", c="green", lw=0.6)
    ax.axhline(-2, ls=":", c="green", lw=0.6, label="z=+/-2 entry")
    ax.axhline(0.5, ls=":", c="orange", lw=0.5)
    ax.axhline(-0.5, ls=":", c="orange", lw=0.5, label="z=+/-0.5 exit")
    for t in trades:
        col = "red" if t.direction < 0 else "green"
        marker = "v" if t.direction < 0 else "^"
        ax.scatter(ts.iloc[t.open_idx], df["z_score"].iloc[t.open_idx],
                   marker=marker, color=col, s=20, alpha=0.7)
    ax.axvline(ts.iloc[train_end_idx], ls="--", c="black", label="Train/Test split")
    ax.set_ylabel("Z-score (shift(1))")
    ax.set_xlabel("Time")
    ax.set_title("z-score + trades (^ buy / v sell)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = output_dir / f"{output_stem}_zscore_trades.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(p)

    # 3. equity
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(ts, equity, label="Reversal strategy", color="C0", lw=1.0)
    ax.plot(ts, bh_equity, label="Buy-and-hold", color="orange", lw=0.8, alpha=0.8)
    ax.axvline(ts.iloc[train_end_idx], ls="--", c="black")
    ax.set_ylabel("Portfolio value (CNY)")
    ax.set_xlabel("Time")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title("Equity: Reversal vs Buy-and-Hold")
    fig.tight_layout()
    p = output_dir / f"{output_stem}_equity.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(p)

    # 4. monthly switches + PnL
    if len(trades) > 0:
        trade_dates = pd.to_datetime([t.open_ts for t in trades])
        df_trades = pd.DataFrame({"date": trade_dates})
        df_trades["month"] = df_trades["date"].dt.to_period("M").astype(str)
        monthly_count = df_trades.groupby("month").size()
        eq_monthly = equity.groupby(df["ts_cst"].dt.to_period("M").astype(str).values).last()
        eq_monthly_pnl = eq_monthly.diff().fillna(0)
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        axes[0].bar(monthly_count.index, monthly_count.values, color="C2")
        axes[0].set_ylabel("# switches/month")
        axes[0].grid(alpha=0.3, axis="y")
        colors = ["C2" if v >= 0 else "C3" for v in eq_monthly_pnl.values]
        axes[1].bar(eq_monthly_pnl.index, eq_monthly_pnl.values, color=colors)
        axes[1].axhline(0, color="black", lw=0.5)
        axes[1].set_ylabel("Monthly PnL (CNY)")
        axes[1].set_xlabel("Month")
        axes[1].tick_params(axis="x", rotation=45)
        axes[1].grid(alpha=0.3, axis="y")
        fig.suptitle("Monthly switches + PnL")
        fig.tight_layout()
        p = output_dir / f"{output_stem}_monthly.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths.append(p)

    return paths


# ─── Recap ──────────────────────────────────────────────────────────────────


def _verdict_check(
    metrics_strat: dict, metrics_bh: dict, n_trades: int, n_days: int
) -> tuple[str, dict]:
    avg_daily = n_trades / max(n_days, 1)
    return_excess = metrics_strat["total_return_pct"] - metrics_bh["total_return_pct"]
    checks = {
        "OOS 净 Total Return - BH": (return_excess, return_excess > 5.0),
        "OOS 净 Sharpe": (metrics_strat["sharpe"], metrics_strat["sharpe"] > 0.7),
        "OOS Max DD": (metrics_strat["max_drawdown_pct"], abs(metrics_strat["max_drawdown_pct"]) < 15.0),
        "OOS 总切换数": (n_trades, n_trades >= 30),
        "OOS 单日平均切换": (avg_daily, avg_daily <= 3.0),
    }
    overall = "✅ 通过" if all(c[1] for c in checks.values()) else "❌ 不通过"
    return overall, checks


def write_recap(
    args, df: pd.DataFrame, train_end_idx: int, train_end_ts,
    metrics_train_strat: dict, metrics_train_bh: dict,
    metrics_test_strat: dict, metrics_test_bh: dict,
    n_trades_train: int, n_trades_test: int,
    stats: dict, adj_stats: dict,
    png_paths: list[Path], output_path: Path,
) -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    test_dates = df.iloc[train_end_idx:]["date_cst"].nunique()
    overall, checks = _verdict_check(metrics_test_strat, metrics_test_bh, n_trades_test, test_dates)
    pass_verdict = overall.startswith("✅")

    def relpath(p: Path) -> str:
        try:
            return str(p.relative_to(output_path.parent))
        except ValueError:
            return str(p)
    img_md = "\n".join(f"![{p.stem}]({relpath(p)})" for p in png_paths)

    summary = (
        f"**Verdict**: {overall} — "
        f"OOS Sharpe={metrics_test_strat['sharpe']:.2f}（BH {metrics_test_bh['sharpe']:.2f}），"
        f"Total Return Δ={metrics_test_strat['total_return_pct']-metrics_test_bh['total_return_pct']:+.2f}%，"
        f"trades={n_trades_test}（avg {n_trades_test/max(test_dates,1):.2f}/day）"
    )

    md = f"""# 300866 5min T+0 局部反转 — 复盘报告

> **日期**：2026-04-30
> **设计**：[../300866-5min-T0反转-设计.md](../300866-5min-T0反转-设计.md)（v0.3）
> **测试**：[../300866-5min-T0反转-测试用例.md](../300866-5min-T0反转-测试用例.md)

## 1. 执行摘要

{summary}

## 2. 运行参数

| 参数 | 值 |
|---|---|
| 标的 | {args.code} |
| 样本期 | {args.start} ~ {args.end} |
| SMA window | {args.sma_window} |
| z 阈值 | entry=±{args.z_entry} / exit=±{args.z_exit} |
| 单日切换上限 | {args.max_daily_switches} |
| Train/Test 切点 | train_ratio={args.train_ratio}（{train_end_ts}） |
| 初始资金 | ¥{args.initial_balance:,.0f} |
| 滑点 | ±{SLIPPAGE*100:.2f}%（单边） |
| 手续费 | 买 {COMMISSION_BUY*100:.3f}% / 卖 {COMMISSION_SELL*100:.3f}% |

## 3. 复权统计

- adj_factor 范围: [{adj_stats['min']:.4f}, {adj_stats['max']:.4f}]
- adj_factor 中位: {adj_stats['median']:.4f}
- 累计稀释（最早 → 最近）: {(1/adj_stats['min']-1)*100:.1f}%

## 4. Mask 与 T+1 统计

| 指标 | 值 |
|---|---|
| 总切换数 | {stats['n_trades']} |
| T+1 截断次数 | {stats['n_t1_truncations']} |
| T+1 估算损失 | ¥{stats['t1_pnl_loss_estimate']:,.0f} |
| 涨跌停截断次数 | {stats['n_limit_truncations']} |

## 5. 业绩对比

### Train (IS) 段

| 指标 | Buy-and-Hold | Reversal | Δ |
|---|---|---|---|
| Sharpe | {metrics_train_bh['sharpe']:.3f} | {metrics_train_strat['sharpe']:.3f} | {metrics_train_strat['sharpe']-metrics_train_bh['sharpe']:+.3f} |
| Total Return | {metrics_train_bh['total_return_pct']:+.2f}% | {metrics_train_strat['total_return_pct']:+.2f}% | {metrics_train_strat['total_return_pct']-metrics_train_bh['total_return_pct']:+.2f}% |
| Max DD | {metrics_train_bh['max_drawdown_pct']:.2f}% | {metrics_train_strat['max_drawdown_pct']:.2f}% | {metrics_train_strat['max_drawdown_pct']-metrics_train_bh['max_drawdown_pct']:+.2f}% |
| Calmar | {metrics_train_bh['calmar']:.3f} | {metrics_train_strat['calmar']:.3f} | — |
| 总切换 | — | {n_trades_train} | — |

### Test (OOS) 段 ⭐核心

| 指标 | Buy-and-Hold | Reversal | Δ |
|---|---|---|---|
| Sharpe | {metrics_test_bh['sharpe']:.3f} | {metrics_test_strat['sharpe']:.3f} | {metrics_test_strat['sharpe']-metrics_test_bh['sharpe']:+.3f} |
| Total Return | {metrics_test_bh['total_return_pct']:+.2f}% | {metrics_test_strat['total_return_pct']:+.2f}% | {metrics_test_strat['total_return_pct']-metrics_test_bh['total_return_pct']:+.2f}% |
| Max DD | {metrics_test_bh['max_drawdown_pct']:.2f}% | {metrics_test_strat['max_drawdown_pct']:.2f}% | {metrics_test_strat['max_drawdown_pct']-metrics_test_bh['max_drawdown_pct']:+.2f}% |
| Calmar | {metrics_test_bh['calmar']:.3f} | {metrics_test_strat['calmar']:.3f} | — |
| 总切换 | — | {n_trades_test} | — |

## 6. Done 判定（设计 §9）

| 指标 | 通过阈值 | 实测 | 结果 |
|---|---|---|---|
| OOS 净 Return - BH > 5% | > 5% | {checks['OOS 净 Total Return - BH'][0]:+.2f}% | {'✅' if checks['OOS 净 Total Return - BH'][1] else '❌'} |
| OOS 净 Sharpe > 0.7 | > 0.7 | {checks['OOS 净 Sharpe'][0]:.3f} | {'✅' if checks['OOS 净 Sharpe'][1] else '❌'} |
| OOS Max DD < 15% | < 15% | {checks['OOS Max DD'][0]:.2f}% | {'✅' if checks['OOS Max DD'][1] else '❌'} |
| OOS 总切换 ≥ 30 | ≥ 30 | {checks['OOS 总切换数'][0]} | {'✅' if checks['OOS 总切换数'][1] else '❌'} |
| OOS 单日平均切换 ≤ 3 | ≤ 3 | {checks['OOS 单日平均切换'][0]:.2f} | {'✅' if checks['OOS 单日平均切换'][1] else '❌'} |

**Verdict: {overall}**

## 7. 可视化

{img_md}

## 8. 下一步

{'**进入多股票泛化**：在 5-10 只消费电子横向验证 alpha 是否泛化；如稳定则接 paper trading。' if pass_verdict else '**关闭单股线 / 调阈值**：参考设计 §9.3 失败诊断 checklist。'}
"""
    output_path.write_text(md)
    print(f"\n✓ 复盘报告写入: {output_path}")
    return pass_verdict


# ─── Main ───────────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--code", default="300866")
    p.add_argument("--start", default="2023-04-04")
    p.add_argument("--end", default="2026-04-22")
    p.add_argument("--sma-window", type=int, default=24)
    p.add_argument("--z-entry", type=float, default=2.0)
    p.add_argument("--z-exit", type=float, default=0.5)
    p.add_argument("--train-ratio", type=float, default=0.50)
    p.add_argument("--initial-balance", type=float, default=200_000.0)
    p.add_argument("--max-daily-switches", type=int, default=4)
    p.add_argument(
        "--output",
        default="docs/GinkgoBrain/复盘/2026-05-01_300866_intraday_reversal.md",
    )
    args = p.parse_args()

    df, df_day, adj_factor = load_aligned_5m(args.code, args.start, args.end)
    check_data_integrity(df)
    adj_stats = {
        "min": float(adj_factor.min()),
        "max": float(adj_factor.max()),
        "median": float(adj_factor.median()),
    }

    df["z_score"] = compute_zscore(df["close_price"], args.sma_window)

    train_end_idx = int(len(df) * args.train_ratio)
    train_end_ts = df["ts_cst"].iloc[train_end_idx]
    print(f"\n=== Walk-forward split ===")
    print(f"  Train: 0 ~ {train_end_idx} ({df['ts_cst'].iloc[0]} ~ {train_end_ts})")
    print(f"  Test : {train_end_idx} ~ {len(df)} ({train_end_ts} ~ {df['ts_cst'].iloc[-1]})")

    params = BacktestParams(
        sma_window=args.sma_window, z_entry=args.z_entry, z_exit=args.z_exit,
        initial_balance=args.initial_balance,
        max_daily_switches=args.max_daily_switches,
    )
    print("\n=== Reversal backtest ===")
    trades, equity, stats = run_backtest(df, params)
    print(f"  总切换 = {stats['n_trades']}, T+1 截断 = {stats['n_t1_truncations']}, "
          f"涨跌停截断 = {stats['n_limit_truncations']}")

    print("\n=== Buy-and-hold baseline ===")
    bh_equity, _ = buy_and_hold_metrics(df, args.initial_balance)

    train_strat = compute_metrics(equity.iloc[:train_end_idx], df.iloc[:train_end_idx])
    test_strat = compute_metrics(equity.iloc[train_end_idx:], df.iloc[train_end_idx:])
    train_bh = compute_metrics(bh_equity.iloc[:train_end_idx], df.iloc[:train_end_idx])
    test_bh = compute_metrics(bh_equity.iloc[train_end_idx:], df.iloc[train_end_idx:])

    n_trades_train = sum(1 for t in trades if t.open_idx < train_end_idx)
    n_trades_test = sum(1 for t in trades if t.open_idx >= train_end_idx)

    print("\n=== Metrics ===")
    print(f"  Train BH    : Sharpe={train_bh['sharpe']:.3f}, Return={train_bh['total_return_pct']:+.2f}%, DD={train_bh['max_drawdown_pct']:.2f}%")
    print(f"  Train Strat : Sharpe={train_strat['sharpe']:.3f}, Return={train_strat['total_return_pct']:+.2f}%, DD={train_strat['max_drawdown_pct']:.2f}%, trades={n_trades_train}")
    print(f"  Test  BH    : Sharpe={test_bh['sharpe']:.3f}, Return={test_bh['total_return_pct']:+.2f}%, DD={test_bh['max_drawdown_pct']:.2f}%")
    print(f"  Test  Strat : Sharpe={test_strat['sharpe']:.3f}, Return={test_strat['total_return_pct']:+.2f}%, DD={test_strat['max_drawdown_pct']:.2f}%, trades={n_trades_test}")

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path("/home/davidlyu/projects/GinkgoRoad") / args.output
    png_paths = plot_diagnostics(
        df, equity, bh_equity, trades, adj_factor,
        train_end_idx, output_path.parent, output_path.stem,
    )
    pass_verdict = write_recap(
        args, df, train_end_idx, train_end_ts,
        train_strat, train_bh, test_strat, test_bh,
        n_trades_train, n_trades_test, stats, adj_stats,
        png_paths, output_path,
    )
    print(f"\n[verdict: {'✓ 通过' if pass_verdict else '✗ 不通过 / 边缘'} done 判定]")
    return 0 if pass_verdict else 1


if __name__ == "__main__":
    sys.exit(main())
