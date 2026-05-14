"""v0.4 IC 验证 spike — 主力 vs 散户对立信号 (super - small ratio).

新角度（v0.2 完全没用到）：
- score = super_net_inflow_ratio - small_net_inflow_ratio
- 含义：超大单买入占比 减 小单买入占比
- 越正 = 主力越买 + 散户越卖（对立结构 → 预期上涨）
- 越负 = 主力越卖 + 散户越买（派发结构 → 预期下跌）

4 个变种：
- v0.4_d1: 单日 spread
- v0.4_d5: 5 日 rolling mean spread
- v0.4_d10: 10 日 rolling mean spread
- v0.4_d20: 20 日 rolling mean spread

每个变种跑全样本 IC + 分桶单调 + Top vs Bottom + 按年度切片.
**无 C1/C3 过滤**——看纯信号 alpha 是否单调.

数据：mongo treasure.stock_fund_flow（2022-2025 实际有效段）
fwd_ret：price.shift(-4) / price.shift(-1) - 1（修过 close[T] 泄漏）
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

logger = logging.getLogger(__name__)

HORIZON = 4
TOP_N = 20
OUTPUT_DIR = Path("/home/davidlyu/projects/GinkgoBrain/reports/spike_v0_4_super_minus_small")
MONGO_HOST = "192.168.1.69"
MONGO_PORT = 27017
MONGO_USER = "admin"
MONGO_DB = "treasure"
MONGO_COLL = "stock_fund_flow"

EFFECTIVE_START_DATE = pd.Timestamp("2022-01-01")
MIN_COVERAGE_DAYS = 500
ROLLING_WINDOWS = [1, 5, 10, 20]


def load_and_filter() -> pd.DataFrame:
    from pymongo import MongoClient
    pwd = os.environ.get("MONGO_PWD")
    if not pwd:
        raise RuntimeError("MONGO_PWD env var required")
    client = MongoClient(host=MONGO_HOST, port=MONGO_PORT,
                         username=MONGO_USER, password=pwd, authSource="admin")
    col = client[MONGO_DB][MONGO_COLL]
    logger.info("Loading mongo (>=2022, super/small ratio fields) ...")
    rows = list(col.find(
        {"date": {"$gte": EFFECTIVE_START_DATE.to_pydatetime()}},
        {"_id": 0, "stock_id": 1, "date": 1, "price": 1,
         "super_net_inflow_ratio": 1, "small_net_inflow_ratio": 1},
    ))
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["super_net_inflow_ratio"] = pd.to_numeric(df["super_net_inflow_ratio"], errors="coerce")
    df["small_net_inflow_ratio"] = pd.to_numeric(df["small_net_inflow_ratio"], errors="coerce")
    cov = df.groupby("stock_id").size()
    keep = cov[cov >= MIN_COVERAGE_DAYS].index
    df = df[df["stock_id"].isin(keep)]
    logger.info("Filtered to %d rows / %d stocks", len(df), len(keep))
    return df


def pivot_panel(df: pd.DataFrame) -> dict:
    super_pivot = df.pivot_table(index="date", columns="stock_id",
                                  values="super_net_inflow_ratio", aggfunc="first")
    small_pivot = df.pivot_table(index="date", columns="stock_id",
                                  values="small_net_inflow_ratio", aggfunc="first")
    price_pivot = df.pivot_table(index="date", columns="stock_id",
                                  values="price", aggfunc="first")
    spread_pivot = super_pivot - small_pivot
    daily_size = price_pivot.notna().sum(axis=1)
    max_size = int(daily_size.max())
    valid_mask = daily_size >= max_size * 0.5
    valid_dates = price_pivot.index[valid_mask]
    return {
        "super": super_pivot.loc[valid_dates],
        "small": small_pivot.loc[valid_dates],
        "spread": spread_pivot.loc[valid_dates],
        "price": price_pivot.loc[valid_dates],
    }


def compute_forward_return(price: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    return price.shift(-horizon) / price.shift(-1) - 1.0


def compute_rolling_spread(spread: pd.DataFrame, window: int) -> pd.DataFrame:
    if window == 1:
        return spread
    return spread.rolling(window=window, min_periods=window).mean()


def bucket_returns(score: pd.DataFrame, fwd_ret: pd.DataFrame,
                   n_buckets: int = 10) -> tuple[pd.Series, pd.DataFrame]:
    per_day = []
    for t in score.index:
        s = score.loc[t].dropna()
        r = fwd_ret.loc[t].reindex(s.index)
        mask = s.notna() & r.notna()
        if mask.sum() < n_buckets * 2:
            continue
        try:
            buckets = pd.qcut(s[mask], q=n_buckets, labels=False, duplicates="drop")
        except ValueError:
            continue
        row = {b: float(r[mask][buckets == b].mean()) for b in range(n_buckets)
               if (buckets == b).sum() > 0}
        row["trade_date"] = t
        per_day.append(row)
    df = pd.DataFrame(per_day).set_index("trade_date")
    bucket_cols = [c for c in df.columns if isinstance(c, int)]
    return df[bucket_cols].mean(axis=0), df[bucket_cols]


def top_vs_bottom(score: pd.DataFrame, fwd_ret: pd.DataFrame, n: int = TOP_N) -> pd.DataFrame:
    rows = []
    for t in score.index:
        s = score.loc[t].dropna()
        r = fwd_ret.loc[t].reindex(s.index)
        mask = s.notna() & r.notna()
        if mask.sum() < n * 2:
            continue
        s_v = s[mask].sort_values(ascending=False)
        r_v = r[mask]
        rows.append({
            "trade_date": t,
            "top_ret": float(r_v.loc[s_v.head(n).index].mean()),
            "bot_ret": float(r_v.loc[s_v.tail(n).index].mean()),
            "univ_ret": float(r_v.mean()),
        })
    df = pd.DataFrame(rows).set_index("trade_date")
    df["long_short"] = df["top_ret"] - df["bot_ret"]
    return df


def daily_ic_series(score: pd.DataFrame, fwd_ret: pd.DataFrame) -> pd.Series:
    out = []
    for t in score.index:
        s = score.loc[t]
        r = fwd_ret.loc[t]
        mask = s.notna() & r.notna()
        if mask.sum() < 5:
            out.append((t, np.nan))
            continue
        out.append((t, float(s[mask].corr(r[mask], method="spearman"))))
    return pd.Series([v for _, v in out], index=[t for t, _ in out], name="ic")


def yearly_ic(ic_series: pd.Series) -> pd.DataFrame:
    df = ic_series.dropna().to_frame(name="ic")
    df["year"] = df.index.year
    return df.groupby("year")["ic"].agg(
        mean="mean", std="std", count="count",
        t_stat=lambda x: float(x.mean() / x.std() * np.sqrt(len(x))) if x.std() > 0 else 0.0,
    )


def write_report(out_dir: Path, results: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n========== 4 个 rolling window 对比 ==========")
    print(f"{'window':>8} | {'IC mean':>9} | {'t-stat':>7} | {'n':>4} | {'B1':>8} | {'B10':>8} | {'B10-B1':>9} | {'L-S':>8} | {'L-S t':>7}")
    summary = []
    for w, r in results.items():
        ic_mean = r['ic_stats'].get('mean', 0.0)
        ic_t = r['ic_stats'].get('t_stat', 0.0)
        ic_n = r['ic_stats'].get('n', 0)
        b1 = r['bucket_mean'].iloc[0]
        b10 = r['bucket_mean'].iloc[-1]
        ls_mean = r['tvb']['long_short'].mean()
        ls_t = float(ls_mean / r['tvb']['long_short'].std() * np.sqrt(len(r['tvb'])))
        summary.append({
            "window": w, "ic_mean": round(ic_mean, 4), "ic_tstat": round(ic_t, 2),
            "ic_n": ic_n,
            "B1_pct": round(b1 * 100, 4), "B10_pct": round(b10 * 100, 4),
            "B10_minus_B1_pct": round((b10 - b1) * 100, 4),
            "long_short_pct": round(ls_mean * 100, 4),
            "long_short_tstat": round(ls_t, 2),
        })
        print(f"  d{w:>3}     | {ic_mean:>+8.4f} | {ic_t:>+7.2f} | {ic_n:>4} | {b1*100:>+7.3f}% | {b10*100:>+7.3f}% | {(b10-b1)*100:>+8.3f}% | {ls_mean*100:>+7.3f}% | {ls_t:>+7.2f}")

    pd.DataFrame(summary).to_csv(out_dir / "window_summary.csv", index=False)

    print("\n========== 按年度 IC ==========")
    for w, r in results.items():
        print(f"\n--- d{w} ---")
        print(r["yearly_ic"].round(4).to_string())
        r["yearly_ic"].to_csv(out_dir / f"yearly_ic_d{w}.csv")

    # 图 1: 4 个 window bucket
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, (w, r) in zip(axes.flat, results.items()):
        bm = r["bucket_mean"]
        x = list(range(len(bm)))
        colors = ["C2" if bm.iloc[i] > 0 else "C3" for i in x]
        ax.bar(x, bm.values * 100, color=colors, alpha=0.85, edgecolor="black")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(x); ax.set_xticklabels([f"B{i+1}" for i in x])
        ic_t = r['ic_stats'].get('t_stat', 0.0)
        ax.set_title(f"d{w} | IC mean={r['ic_stats']['mean']:.4f} t={ic_t:+.2f}")
        ax.set_ylabel("fwd_ret %")
        ax.grid(alpha=0.3, axis="y")
    plt.suptitle("v0.4 super - small spread: Bucket Mean fwd_ret", fontsize=12)
    plt.tight_layout(); plt.savefig(out_dir / "bucket_curves.png", dpi=110); plt.close()

    # 图 2: Top vs Bottom cumulative
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, (w, r) in zip(axes.flat, results.items()):
        tvb = r["tvb"]
        cum_top = (1 + tvb["top_ret"]).cumprod()
        cum_bot = (1 + tvb["bot_ret"]).cumprod()
        cum_uni = (1 + tvb["univ_ret"]).cumprod()
        ax.plot(cum_top.index, cum_top, label="Top20", linewidth=1.4)
        ax.plot(cum_bot.index, cum_bot, label="Bot20", linewidth=1.4, color="C3")
        ax.plot(cum_uni.index, cum_uni, label="Univ", linewidth=2, linestyle="--", color="black")
        ls_t = float(tvb['long_short'].mean() / tvb['long_short'].std() * np.sqrt(len(tvb)))
        ax.set_title(f"d{w} | L-S t={ls_t:+.2f}")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax.set_yscale("log")
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.suptitle("v0.4 Top20 vs Bottom20 cumulative T+1→T+4 (log)", fontsize=12)
    plt.tight_layout(); plt.savefig(out_dir / "top_vs_bottom.png", dpi=110); plt.close()

    # 图 3: 按年度 IC
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, (w, r) in zip(axes.flat, results.items()):
        y = r["yearly_ic"]
        colors = ["C2" if (m > 0.02 and t > 3) else ("C3" if (m < -0.02 and t < -3) else "C7")
                  for m, t in zip(y["mean"], y["t_stat"])]
        ax.bar(y.index.astype(str), y["mean"] * 100, color=colors, alpha=0.85, edgecolor="black")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.axhline(2, color="green", linestyle="--", alpha=0.6)
        ax.axhline(-2, color="red", linestyle="--", alpha=0.6)
        ax.set_title(f"d{w}")
        ax.set_ylabel("IC mean (%)")
        ax.grid(alpha=0.3, axis="y")
    plt.suptitle("v0.4 super - small spread: Yearly IC mean", fontsize=12)
    plt.tight_layout(); plt.savefig(out_dir / "yearly_ic.png", dpi=110); plt.close()

    print(f"\nAll outputs in: {out_dir}")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_and_filter()
    panel = pivot_panel(df)
    fwd_ret = compute_forward_return(panel["price"])
    logger.info("Trade dates: %d, stocks: %d", len(panel["price"]), panel["price"].shape[1])

    results = {}
    for w in ROLLING_WINDOWS:
        logger.info("--- rolling window d%d ---", w)
        score = compute_rolling_spread(panel["spread"], w)
        ic_series = daily_ic_series(score, fwd_ret)
        ic_valid = ic_series.dropna()
        ic_stats = {
            "mean": float(ic_valid.mean()) if len(ic_valid) > 0 else 0.0,
            "std": float(ic_valid.std()) if len(ic_valid) > 0 else 0.0,
            "n": len(ic_valid),
            "t_stat": float(ic_valid.mean() / ic_valid.std() * np.sqrt(len(ic_valid)))
                if ic_valid.std() > 0 else 0.0,
        }
        bucket_mean, bucket_df = bucket_returns(score, fwd_ret)
        tvb = top_vs_bottom(score, fwd_ret)
        yearly = yearly_ic(ic_series)
        results[w] = {
            "ic_stats": ic_stats,
            "bucket_mean": bucket_mean, "bucket_df": bucket_df,
            "tvb": tvb, "yearly_ic": yearly,
        }

    write_report(OUTPUT_DIR, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
