"""非单调 alpha 验证 spike — 检查 Spearman IC 是否漏掉非线性结构.

3 个验证：
1. 分桶平均收益曲线（score 分 10 桶 × 平均 T+4 forward return）
   → 检查是不是 U 形 / 阶梯型 / 反向单调
2. 极端值 portfolio（Top 1% / 3% / 5% / 10% / 20% vs universe 平均）
   → 检查 Top 1% 是不是真有 alpha 但被 Top 20 (5%) dilute 掉
3. Bottom20 vs Top20 累计收益对比
   → 检查反向使用是否有 alpha（v0.2 IC=-0.0125 暗示）

用 v0.2 score = Σ(10日大单净流入) / Σ(10日总成交). **不预先过滤价格区间**.
fwd_ret 用 close.shift(-1) 当起点（修复 close[T] 数据泄漏）.
"""
from __future__ import annotations

import logging
import sys
from datetime import date
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from spike_v0_2_accumulation_ic import (
    N_WINDOW,
    load_data,
    pivot_matrices,
)

logger = logging.getLogger(__name__)

START = date(2025, 4, 2)
END = date(2026, 5, 12)
HORIZON = 4
OUTPUT_DIR = Path("/home/davidlyu/projects/GinkgoBrain/reports/spike_v0_2_nonlinearity")


def compute_score_v0_2(mats: dict) -> pd.DataFrame:
    inflow = mats["inflow"]
    amount = mats["amount"]
    cum_inflow = inflow.rolling(window=N_WINDOW, min_periods=N_WINDOW).sum()
    cum_amount = amount.rolling(window=N_WINDOW, min_periods=N_WINDOW).sum()
    score = cum_inflow / cum_amount.replace(0, np.nan)
    return score


def compute_forward_return(close: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """T+1 → T+horizon 累计收益（开盘买入更接近策略）.
    用 close.shift(-1) 起点，避开 T 日数据泄漏."""
    return close.shift(-horizon) / close.shift(-1) - 1.0


# ============================================================================

def bucket_average_returns(
    score: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    n_buckets: int = 10,
) -> tuple[pd.Series, pd.DataFrame]:
    per_day_bucket_rets = []
    for t in score.index:
        s = score.loc[t].dropna()
        r = fwd_ret.loc[t].reindex(s.index)
        mask = s.notna() & r.notna()
        if mask.sum() < n_buckets * 2:
            continue
        s_v = s[mask]
        r_v = r[mask]
        try:
            buckets = pd.qcut(s_v, q=n_buckets, labels=False, duplicates="drop")
        except ValueError:
            continue
        row: dict = {}
        for b in range(n_buckets):
            mask_b = buckets == b
            if mask_b.sum() > 0:
                row[b] = float(r_v[mask_b].mean())
            else:
                row[b] = np.nan
        row["trade_date"] = t
        per_day_bucket_rets.append(row)

    df = pd.DataFrame(per_day_bucket_rets).set_index("trade_date")
    bucket_cols = [c for c in df.columns if isinstance(c, int)]
    bucket_mean = df[bucket_cols].mean(axis=0)
    return bucket_mean, df[bucket_cols]


def plot_bucket_curve(bucket_mean: pd.Series, df: pd.DataFrame, out_dir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x = list(range(len(bucket_mean)))
    ax1.bar(x, bucket_mean.values * 100, color="C0", alpha=0.8, edgecolor="black")
    ax1.axhline(0, color="black", linewidth=0.6)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"B{i+1}" for i in range(len(bucket_mean))])
    ax1.set_xlabel("score quantile bucket (B1=lowest, B10=highest)")
    ax1.set_ylabel("Mean Forward Return (%, T+1->T+4)")
    ax1.set_title("Bucket Mean Forward Return - Monotonic?")
    ax1.grid(alpha=0.3, axis="y")

    cum = (1 + df.fillna(0)).cumprod()
    cmap = plt.cm.coolwarm
    for i, col in enumerate(cum.columns):
        c = cmap(i / max(1, len(cum.columns) - 1))
        ax2.plot(cum.index, cum[col], label=f"B{col+1}", linewidth=1.3, alpha=0.85, color=c)
    ax2.set_title("Cumulative Per-Bucket Return Over Time")
    ax2.set_ylabel("Cumulative (1+r)")
    ax2.legend(ncol=2, fontsize=8, loc="best")
    ax2.grid(alpha=0.3)
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.tight_layout()
    plt.savefig(out_dir / "bucket_curves.png", dpi=110)


def extreme_portfolio_returns(
    score: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    pcts: list[float],
) -> dict:
    results = {}
    universe_avg_daily = []
    for t in score.index:
        s = score.loc[t].dropna()
        r = fwd_ret.loc[t].reindex(s.index)
        mask = s.notna() & r.notna()
        if mask.sum() < 50:
            continue
        s_v = s[mask].sort_values(ascending=False)
        r_v = r[mask]
        universe_avg_daily.append({"trade_date": t, "u_avg": float(r_v.mean())})
        for p in pcts:
            n = max(1, int(len(s_v) * p))
            top_codes = s_v.head(n).index
            mean_ret = float(r_v.loc[top_codes].mean())
            results.setdefault(p, []).append({"trade_date": t, "ret": mean_ret, "n": n})
    out: dict = {"universe_avg": pd.DataFrame(universe_avg_daily).set_index("trade_date")}
    for p, rows in results.items():
        out[p] = pd.DataFrame(rows).set_index("trade_date")
    return out


def plot_extreme_portfolios(out: dict, out_dir: Path) -> pd.DataFrame:
    u_avg = out["universe_avg"]["u_avg"].mean()
    print(f"\n--- 极端值 portfolio 平均 T+1->T+4 收益 ---")
    print(f"  Universe avg:    {u_avg*100:+.3f}%")
    rows = []
    for p in sorted([k for k in out.keys() if isinstance(k, float)]):
        ret_mean = out[p]["ret"].mean()
        n_mean = out[p]["n"].mean()
        excess = ret_mean - u_avg
        # t-stat for daily excess
        diff = out[p]["ret"] - out["universe_avg"]["u_avg"].reindex(out[p].index)
        t_stat = float(diff.mean() / diff.std() * np.sqrt(len(diff))) if diff.std() > 0 else float("nan")
        print(f"  Top {p*100:>4.1f}% (avg {n_mean:.0f} stocks): {ret_mean*100:+.3f}% | excess {excess*100:+.3f}% | t-stat {t_stat:+.2f}")
        rows.append({"pct": p, "mean_ret_pct": round(ret_mean * 100, 4),
                     "excess_pct": round(excess * 100, 4),
                     "t_stat": round(t_stat, 3),
                     "avg_n": int(n_mean)})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "extreme_portfolios.csv", index=False)

    fig, ax = plt.subplots(figsize=(13, 5))
    for p in sorted([k for k in out.keys() if isinstance(k, float)]):
        cum = (1 + out[p]["ret"].fillna(0)).cumprod()
        ax.plot(cum.index, cum.values, label=f"Top {p*100:.0f}% (mean {out[p]['ret'].mean()*100:+.2f}%)", linewidth=1.4)
    u_cum = (1 + out["universe_avg"]["u_avg"].fillna(0)).cumprod()
    ax.plot(u_cum.index, u_cum.values, label=f"Universe avg ({u_avg*100:+.2f}%)", linewidth=2.0, linestyle="--", color="black")
    ax.set_title("Cumulative T+1->T+4 Return: Extreme Portfolios vs Universe")
    ax.set_ylabel("Cumulative (1+r)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.tight_layout()
    plt.savefig(out_dir / "extreme_portfolios.png", dpi=110)
    return df


def top_vs_bottom_20(score: pd.DataFrame, fwd_ret: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    rows = []
    for t in score.index:
        s = score.loc[t].dropna()
        r = fwd_ret.loc[t].reindex(s.index)
        mask = s.notna() & r.notna()
        if mask.sum() < n * 2:
            continue
        s_v = s[mask].sort_values(ascending=False)
        r_v = r[mask]
        top_n = s_v.head(n).index
        bot_n = s_v.tail(n).index
        rows.append({
            "trade_date": t,
            "top20_ret": float(r_v.loc[top_n].mean()),
            "bot20_ret": float(r_v.loc[bot_n].mean()),
            "univ_ret": float(r_v.mean()),
            "long_short_ret": float(r_v.loc[top_n].mean() - r_v.loc[bot_n].mean()),
        })
    return pd.DataFrame(rows).set_index("trade_date")


def plot_top_vs_bottom(df: pd.DataFrame, out_dir: Path) -> None:
    ls_mean = df["long_short_ret"].mean()
    ls_t = float(ls_mean / df["long_short_ret"].std() * np.sqrt(len(df)))
    print(f"\n--- Top20 vs Bottom20 mean T+1->T+4 收益 ---")
    print(f"  Top20:    {df['top20_ret'].mean()*100:+.3f}%")
    print(f"  Bottom20: {df['bot20_ret'].mean()*100:+.3f}%")
    print(f"  Universe: {df['univ_ret'].mean()*100:+.3f}%")
    print(f"  Long-Short (Top-Bot): {ls_mean*100:+.3f}% | t-stat {ls_t:+.2f}")
    df.to_csv(out_dir / "top_vs_bottom.csv")

    fig, ax = plt.subplots(figsize=(13, 5))
    cum_top = (1 + df["top20_ret"]).cumprod()
    cum_bot = (1 + df["bot20_ret"]).cumprod()
    cum_uni = (1 + df["univ_ret"]).cumprod()
    ax.plot(cum_top.index, cum_top, label=f"Top20 (mean {df['top20_ret'].mean()*100:+.3f}%)", linewidth=1.6)
    ax.plot(cum_bot.index, cum_bot, label=f"Bottom20 (mean {df['bot20_ret'].mean()*100:+.3f}%)", linewidth=1.6, color="C3")
    ax.plot(cum_uni.index, cum_uni, label=f"Universe (mean {df['univ_ret'].mean()*100:+.3f}%)", linewidth=2, linestyle="--", color="black")
    ax.set_title(f"Top20 vs Bottom20 by score - Cumulative T+1->T+4 (Long-Short t={ls_t:+.2f})")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.tight_layout()
    plt.savefig(out_dir / "top_vs_bottom.png", dpi=110)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    mats = pivot_matrices(data["mf"], data["kline"])
    score = compute_score_v0_2(mats)
    fwd_ret = compute_forward_return(mats["close"], horizon=HORIZON)
    logger.info("score shape: %s, fwd_ret shape: %s", score.shape, fwd_ret.shape)

    print("\n========== 验证 1: 分桶平均收益曲线 ==========")
    bucket_mean, bucket_df = bucket_average_returns(score, fwd_ret, n_buckets=10)
    print("Bucket | mean fwd_ret (%)")
    for b, v in bucket_mean.items():
        print(f"  B{b+1} (quantile {b*10}-{(b+1)*10}%): {v*100:+.4f}%")
    plot_bucket_curve(bucket_mean, bucket_df, OUTPUT_DIR)

    sorted_buckets = bucket_mean.sort_values().index.tolist()
    natural = list(range(len(bucket_mean)))
    is_mono_inc = sorted_buckets == natural
    is_mono_dec = sorted_buckets == list(reversed(natural))
    print(f"\nMonotonic ascending? {is_mono_inc}  (high score → high return ?)")
    print(f"Monotonic descending? {is_mono_dec}  (low score → high return ?)")

    print("\n========== 验证 2: 极端值 portfolio ==========")
    extreme = extreme_portfolio_returns(score, fwd_ret, pcts=[0.01, 0.03, 0.05, 0.10, 0.20])
    plot_extreme_portfolios(extreme, OUTPUT_DIR)

    print("\n========== 验证 3: Bottom20 vs Top20 ==========")
    tvb = top_vs_bottom_20(score, fwd_ret, n=20)
    plot_top_vs_bottom(tvb, OUTPUT_DIR)

    print(f"\nAll outputs in: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
