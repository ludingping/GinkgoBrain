"""v0.3 IC 验证 spike — 用 MongoDB treasure.stock_fund_flow 14 年数据.

数据源（替代 cnstock_daily_money_flow）：
- 14 年（2011-08 ~ 2025-10）vs 旧 13 月
- 含完整市场周期：2015 股灾 / 2018 熊市 / 2020 疫情 / 2022 熊市 / 2024 反弹
- 字段清晰：main_net_inflow / *_ratio / price / quote_change（不需要单位修正）

复用 v0.2 信号设计（参数化迁移）：
- C1 持续性：过去 N=10 日中 main_net_inflow > 0 天数 ≥ 7
- C3 价格温和：过去 N 日累计涨幅 ∈ [-5%, +8%]
- score: 过去 N 日 main_net_inflow_ratio 均值（替代旧 Σnet/Σamount；新数据无 total_amount）
- fwd_ret: price.shift(-horizon) / price.shift(-1) - 1（T+1 close 起点，已修 close[T] 泄漏）

输出 reports/spike_v0_3/：
- ic_stats.json（含全样本 + 按年度切片 IC）
- bucket_curves.png / extreme_portfolios.png / top_vs_bottom.png
- regime_ic_by_year.png（**重要：看 alpha 在熊市/牛市是否持续**）

环境变量 MONGO_PWD 提供密码（不写进代码避免 commit 泄漏）.
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

N_WINDOW = 10
POSITIVE_DAYS_MIN = 7
PRICE_RANGE = (-0.05, 0.08)
TOP_N = 20
HORIZON = 4
OUTPUT_DIR = Path("/home/davidlyu/projects/GinkgoBrain/reports/spike_v0_3")

MONGO_HOST = "192.168.1.69"
MONGO_PORT = 27017
MONGO_USER = "admin"
MONGO_DB = "treasure"
MONGO_COLL = "stock_fund_flow"


def load_mongo() -> pd.DataFrame:
    from pymongo import MongoClient
    pwd = os.environ.get("MONGO_PWD")
    if not pwd:
        raise RuntimeError("MONGO_PWD env var required")
    logger.info("Connecting MongoDB %s:%d ...", MONGO_HOST, MONGO_PORT)
    client = MongoClient(
        host=MONGO_HOST, port=MONGO_PORT,
        username=MONGO_USER, password=pwd, authSource="admin",
    )
    col = client[MONGO_DB][MONGO_COLL]
    n = col.count_documents({})
    logger.info("Total docs: %d", n)
    cursor = col.find({}, {
        "_id": 0, "stock_id": 1, "name": 1, "date": 1, "price": 1,
        "quote_change": 1, "main_net_inflow": 1, "main_net_inflow_ratio": 1,
    })
    rows = list(cursor)
    logger.info("Fetched %d rows", len(rows))
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["main_net_inflow"] = pd.to_numeric(df["main_net_inflow"], errors="coerce")
    df["main_net_inflow_ratio"] = pd.to_numeric(df["main_net_inflow_ratio"], errors="coerce")
    return df


def pivot_matrices(df: pd.DataFrame) -> dict:
    logger.info("Pivoting matrices ...")
    inflow_mat = df.pivot_table(
        index="date", columns="stock_id", values="main_net_inflow", aggfunc="first"
    )
    ratio_mat = df.pivot_table(
        index="date", columns="stock_id", values="main_net_inflow_ratio", aggfunc="first"
    )
    price_mat = df.pivot_table(
        index="date", columns="stock_id", values="price", aggfunc="first"
    )
    logger.info("inflow_mat shape: %s", inflow_mat.shape)
    return {"inflow": inflow_mat, "ratio": ratio_mat, "price": price_mat}


def compute_signal(mats: dict, n: int = N_WINDOW, m: int = POSITIVE_DAYS_MIN,
                   price_range: tuple = PRICE_RANGE) -> dict:
    inflow = mats["inflow"]
    ratio = mats["ratio"]
    price = mats["price"]
    positive_days = (inflow > 0).rolling(window=n, min_periods=n).sum()
    score = ratio.rolling(window=n, min_periods=n).mean()
    price_change_n = price / price.shift(n - 1) - 1.0
    mask_c1 = positive_days >= m
    mask_c3 = (price_change_n >= price_range[0]) & (price_change_n <= price_range[1])
    final_mask = mask_c1 & mask_c3 & score.notna()
    score_filtered = score.where(final_mask)
    return {"score": score, "score_filtered": score_filtered,
            "positive_days": positive_days, "price_change_n": price_change_n,
            "final_mask": final_mask}


def compute_forward_return(price: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    return price.shift(-horizon) / price.shift(-1) - 1.0


def bucket_average_returns(score: pd.DataFrame, fwd_ret: pd.DataFrame,
                          n_buckets: int = 10):
    per_day = []
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
        row = {b: float(r_v[buckets == b].mean()) for b in range(n_buckets)
               if (buckets == b).sum() > 0}
        row["trade_date"] = t
        per_day.append(row)
    df = pd.DataFrame(per_day).set_index("trade_date")
    bucket_cols = [c for c in df.columns if isinstance(c, int)]
    return df[bucket_cols].mean(axis=0), df[bucket_cols]


def extreme_portfolio_returns(score: pd.DataFrame, fwd_ret: pd.DataFrame,
                              pcts: list = [0.01, 0.05, 0.10, 0.20]) -> dict:
    out = {}
    u_avg = []
    for t in score.index:
        s = score.loc[t].dropna()
        r = fwd_ret.loc[t].reindex(s.index)
        mask = s.notna() & r.notna()
        if mask.sum() < 50:
            continue
        s_v = s[mask].sort_values(ascending=False)
        r_v = r[mask]
        u_avg.append({"trade_date": t, "u_avg": float(r_v.mean())})
        for p in pcts:
            n = max(1, int(len(s_v) * p))
            ret_mean = float(r_v.loc[s_v.head(n).index].mean())
            out.setdefault(p, []).append({"trade_date": t, "ret": ret_mean, "n": n})
    return {
        "universe_avg": pd.DataFrame(u_avg).set_index("trade_date"),
        **{p: pd.DataFrame(rows).set_index("trade_date") for p, rows in out.items()},
    }


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


def yearly_ic(score: pd.DataFrame, fwd_ret: pd.DataFrame) -> pd.DataFrame:
    daily_ic = []
    for t in score.index:
        s = score.loc[t]
        r = fwd_ret.loc[t]
        mask = s.notna() & r.notna()
        if mask.sum() < 5:
            daily_ic.append({"trade_date": t, "ic": np.nan})
            continue
        ic = float(s[mask].corr(r[mask], method="spearman"))
        daily_ic.append({"trade_date": t, "ic": ic})
    df = pd.DataFrame(daily_ic).set_index("trade_date").dropna()
    df["year"] = df.index.year
    return df.groupby("year")["ic"].agg(
        mean="mean", std="std", count="count",
        t_stat=lambda x: float(x.mean() / x.std() * np.sqrt(len(x))) if x.std() > 0 else 0.0,
    )


def write_reports(out_dir: Path, bucket_mean, bucket_df, extreme, tvb, yearly) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    u_avg = extreme["universe_avg"]["u_avg"].mean()
    print("\n========== 全样本 IC + 极端 portfolio ==========")
    print("Bucket | mean fwd_ret (%)")
    for b, v in bucket_mean.items():
        print(f"  B{b+1} ({b*10}-{(b+1)*10}%): {v*100:+.4f}%")
    sorted_idx = bucket_mean.sort_values().index.tolist()
    is_mono = sorted_idx == list(range(len(bucket_mean)))
    print(f"\nMonotonic ascending? {is_mono}")
    print(f"\nUniverse avg: {u_avg*100:+.3f}%")
    extreme_summary = {}
    for p in sorted([k for k in extreme.keys() if isinstance(k, float)]):
        m = extreme[p]["ret"].mean()
        diff = extreme[p]["ret"] - extreme["universe_avg"]["u_avg"].reindex(extreme[p].index)
        t = float(diff.mean() / diff.std() * np.sqrt(len(diff))) if diff.std() > 0 else 0.0
        n_avg = extreme[p]["n"].mean()
        print(f"  Top {p*100:>4.1f}% (avg {n_avg:.0f}): {m*100:+.3f}% | excess {(m-u_avg)*100:+.3f}% | t={t:+.2f}")
        extreme_summary[f"Top_{int(p*100)}pct"] = {
            "mean_pct": round(m * 100, 4),
            "excess_pct": round((m - u_avg) * 100, 4),
            "t_stat": round(t, 3),
        }

    print(f"\nTop20 vs Bottom20:")
    print(f"  Top: {tvb['top_ret'].mean()*100:+.3f}%")
    print(f"  Bot: {tvb['bot_ret'].mean()*100:+.3f}%")
    ls_t = float(tvb['long_short'].mean() / tvb['long_short'].std() * np.sqrt(len(tvb)))
    print(f"  L-S: {tvb['long_short'].mean()*100:+.3f}% | t={ls_t:+.2f}")

    print("\n========== 按年度 IC (regime 检验) ==========")
    print(yearly.round(4).to_string())

    payload = {
        "params": {
            "N_window": N_WINDOW, "positive_days_min": POSITIVE_DAYS_MIN,
            "price_range": list(PRICE_RANGE), "horizon": HORIZON,
            "score": "rolling mean of main_net_inflow_ratio over N days",
        },
        "bucket_mean_pct": {f"B{b+1}": float(v) * 100 for b, v in bucket_mean.items()},
        "monotonic_ascending": bool(is_mono),
        "extreme_portfolios": extreme_summary,
        "top_vs_bottom": {
            "top_mean_pct": round(tvb["top_ret"].mean() * 100, 4),
            "bot_mean_pct": round(tvb["bot_ret"].mean() * 100, 4),
            "long_short_pct": round(tvb["long_short"].mean() * 100, 4),
            "long_short_tstat": round(ls_t, 3),
        },
        "yearly_ic": yearly.round(5).to_dict(orient="index"),
    }
    (out_dir / "ic_stats.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    tvb.to_csv(out_dir / "top_vs_bottom.csv")
    yearly.to_csv(out_dir / "yearly_ic.csv")

    # 图 1: bucket
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x = list(range(len(bucket_mean)))
    ax1.bar(x, bucket_mean.values * 100, color="C0", alpha=0.8, edgecolor="black")
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.set_xticks(x); ax1.set_xticklabels([f"B{i+1}" for i in range(len(bucket_mean))])
    ax1.set_ylabel("Mean Forward Return (%)")
    ax1.set_title("Bucket Mean Forward Return (14y)")
    ax1.grid(alpha=0.3, axis="y")
    cum = (1 + bucket_df.fillna(0)).cumprod()
    cmap = plt.cm.coolwarm
    for i, col in enumerate(cum.columns):
        ax2.plot(cum.index, cum[col], color=cmap(i / max(1, len(cum.columns) - 1)),
                 linewidth=1.2, alpha=0.85, label=f"B{col+1}")
    ax2.set_title("Cumulative Per-Bucket Return")
    ax2.legend(ncol=2, fontsize=8, loc="best"); ax2.grid(alpha=0.3)
    ax2.set_yscale("log")
    plt.tight_layout(); plt.savefig(out_dir / "bucket_curves.png", dpi=110); plt.close()

    # 图 2: 极端 portfolio
    fig, ax = plt.subplots(figsize=(14, 5))
    for p in sorted([k for k in extreme.keys() if isinstance(k, float)]):
        cum = (1 + extreme[p]["ret"].fillna(0)).cumprod()
        ax.plot(cum.index, cum, label=f"Top {p*100:.0f}% (mean {extreme[p]['ret'].mean()*100:+.3f}%)",
                linewidth=1.3)
    u_cum = (1 + extreme["universe_avg"]["u_avg"].fillna(0)).cumprod()
    ax.plot(u_cum.index, u_cum, label=f"Universe avg ({u_avg*100:+.3f}%)",
            linewidth=2, linestyle="--", color="black")
    ax.set_title("Cumulative T+1->T+4 Return: Extreme Portfolios (14y, log scale)")
    ax.legend(); ax.grid(alpha=0.3); ax.set_yscale("log")
    plt.tight_layout(); plt.savefig(out_dir / "extreme_portfolios.png", dpi=110); plt.close()

    # 图 3: Top vs Bottom
    fig, ax = plt.subplots(figsize=(14, 5))
    cum_top = (1 + tvb["top_ret"]).cumprod()
    cum_bot = (1 + tvb["bot_ret"]).cumprod()
    cum_uni = (1 + tvb["univ_ret"]).cumprod()
    ax.plot(cum_top.index, cum_top, label=f"Top20 ({tvb['top_ret'].mean()*100:+.3f}%)", linewidth=1.5)
    ax.plot(cum_bot.index, cum_bot, label=f"Bot20 ({tvb['bot_ret'].mean()*100:+.3f}%)", linewidth=1.5, color="C3")
    ax.plot(cum_uni.index, cum_uni, label=f"Univ ({tvb['univ_ret'].mean()*100:+.3f}%)",
            linewidth=2, linestyle="--", color="black")
    ax.set_title(f"Top20 vs Bottom20 by score (14y, log) - L-S t={ls_t:+.2f}")
    ax.legend(); ax.grid(alpha=0.3); ax.set_yscale("log")
    plt.tight_layout(); plt.savefig(out_dir / "top_vs_bottom.png", dpi=110); plt.close()

    # 图 4: 按年度 IC
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    colors = ["C2" if (m > 0.02 and t > 3) else ("C3" if (m < -0.02 and t < -3) else "C7")
              for m, t in zip(yearly["mean"], yearly["t_stat"])]
    years_str = yearly.index.astype(str)
    ax1.bar(years_str, yearly["mean"] * 100, color=colors, alpha=0.85, edgecolor="black")
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.axhline(2, color="green", linestyle="--", alpha=0.6, label="threshold +2%")
    ax1.set_ylabel("IC mean (%)")
    ax1.set_title("Yearly IC mean (Green=PASS, Red=REVERSE, Gray=NEUTRAL)")
    ax1.legend(); ax1.grid(alpha=0.3, axis="y")
    ax2.bar(years_str, yearly["t_stat"], color=colors, alpha=0.85, edgecolor="black")
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.axhline(3, color="green", linestyle="--", alpha=0.6, label="t=±3 significance")
    ax2.axhline(-3, color="red", linestyle="--", alpha=0.6)
    ax2.set_ylabel("IC t-stat")
    ax2.set_title("Yearly IC t-stat")
    ax2.legend(); ax2.grid(alpha=0.3, axis="y")
    plt.tight_layout(); plt.savefig(out_dir / "regime_ic_by_year.png", dpi=110); plt.close()

    print(f"\nAll outputs in: {out_dir}")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_mongo()
    logger.info("Date range: %s -> %s | unique stocks: %d",
                df["date"].min(), df["date"].max(), df["stock_id"].nunique())
    mats = pivot_matrices(df)
    logger.info("Computing signal ...")
    signal_info = compute_signal(mats)
    logger.info("Computing forward return ...")
    fwd_ret = compute_forward_return(mats["price"], horizon=HORIZON)

    print("\n========== 验证 1: 分桶 ==========")
    bucket_mean, bucket_df = bucket_average_returns(signal_info["score_filtered"], fwd_ret, n_buckets=10)
    print("\n========== 验证 2: 极端 portfolio ==========")
    extreme = extreme_portfolio_returns(signal_info["score_filtered"], fwd_ret,
                                        pcts=[0.01, 0.03, 0.05, 0.10, 0.20])
    print("\n========== 验证 3: Top20 vs Bottom20 ==========")
    tvb = top_vs_bottom(signal_info["score_filtered"], fwd_ret, n=TOP_N)
    print("\n========== 验证 4: 按年度 IC ==========")
    yearly = yearly_ic(signal_info["score_filtered"], fwd_ret)

    write_reports(OUTPUT_DIR, bucket_mean, bucket_df, extreme, tvb, yearly)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
