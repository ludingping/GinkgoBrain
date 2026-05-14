"""MongoDB stock_fund_flow 数据质量审计 spike.

目标：搞清楚 14 年数据里"哪部分合格、哪部分不合格"，为下一步 spike 提供 universe 过滤标准。

5 个审计维度：
1. 按年规模：每年 docs / unique stocks / unique dates / 平均每日 universe size
2. 按股覆盖：每股 trade_dates 总数、起止日期、是否全期覆盖
3. 字段缺失 / 零值率：主要是 main_net_inflow / price 是否大量为 0 或 NaN
4. 字段一致性：main_net_inflow ≈ super + big 的偏差分布
5. 数值合理性：*_ratio 字段是否超出 [-100, 100]

输出 reports/spike_v0_3_data_quality/：
- yearly_summary.csv  每年 universe 状态
- per_stock_coverage.csv  每股覆盖度
- yearly_universe_size.png
- coverage_distribution.png
- first_date_distribution.png
- recommended_universe.md  基于诊断的"合格 universe"建议
"""
from __future__ import annotations

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

OUTPUT_DIR = Path("/home/davidlyu/projects/GinkgoBrain/reports/spike_v0_3_data_quality")
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
    client = MongoClient(host=MONGO_HOST, port=MONGO_PORT,
                         username=MONGO_USER, password=pwd, authSource="admin")
    col = client[MONGO_DB][MONGO_COLL]
    logger.info("Loading 全部 11 字段 for audit ...")
    rows = list(col.find({}, {
        "_id": 0, "stock_id": 1, "name": 1, "date": 1, "price": 1,
        "main_net_inflow": 1, "main_net_inflow_ratio": 1,
        "super_net_inflow": 1, "big_net_inflow": 1,
        "middle_net_inflow": 1, "small_net_inflow": 1,
    }))
    logger.info("Fetched %d rows", len(rows))
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    for c in ["price", "main_net_inflow", "main_net_inflow_ratio",
              "super_net_inflow", "big_net_inflow",
              "middle_net_inflow", "small_net_inflow"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["year"] = df["date"].dt.year
    return df


def audit_yearly(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("year")
    out = pd.DataFrame({
        "docs": g.size(),
        "unique_stocks": g["stock_id"].nunique(),
        "unique_dates": g["date"].nunique(),
    })
    out["avg_universe_per_day"] = (out["docs"] / out["unique_dates"]).round(0).astype(int)
    out["nan_main_pct"] = (g["main_net_inflow"].apply(lambda x: x.isna().mean()) * 100).round(2)
    out["zero_main_pct"] = (g["main_net_inflow"].apply(lambda x: (x == 0).mean()) * 100).round(2)
    out["nan_price_pct"] = (g["price"].apply(lambda x: x.isna().mean()) * 100).round(2)
    return out.reset_index()


def audit_per_stock(df: pd.DataFrame, all_trade_dates: pd.DatetimeIndex) -> pd.DataFrame:
    total_dates = len(all_trade_dates)
    g = df.groupby("stock_id")
    out = pd.DataFrame({
        "n_days": g.size(),
        "first_date": g["date"].min(),
        "last_date": g["date"].max(),
        "name": g["name"].first(),
    })
    out["coverage_pct"] = (out["n_days"] / total_dates * 100).round(1)
    out["span_years"] = ((out["last_date"] - out["first_date"]).dt.days / 365.25).round(1)
    out["zero_main_pct"] = (g["main_net_inflow"].apply(lambda x: (x == 0).mean()) * 100).round(1)
    out["nan_price_pct"] = (g["price"].apply(lambda x: x.isna().mean()) * 100).round(1)
    return out.reset_index().sort_values("n_days", ascending=False)


def audit_field_consistency(df: pd.DataFrame) -> dict:
    valid = df.dropna(subset=["main_net_inflow", "super_net_inflow", "big_net_inflow"])
    expected_main = valid["super_net_inflow"] + valid["big_net_inflow"]
    diff = valid["main_net_inflow"] - expected_main
    main_abs = valid["main_net_inflow"].abs().replace(0, np.nan)
    rel_diff = (diff.abs() / main_abs).dropna()
    return {
        "n_rows": len(valid),
        "exact_equal_pct": round((diff.abs() < 1).mean() * 100, 2),
        "rel_diff_median_pct": round(rel_diff.median() * 100, 3),
        "rel_diff_p95_pct": round(rel_diff.quantile(0.95) * 100, 3),
        "rel_diff_p99_pct": round(rel_diff.quantile(0.99) * 100, 3),
    }


def audit_ratio_range(df: pd.DataFrame) -> dict:
    r = df["main_net_inflow_ratio"].dropna()
    out_of_range = ((r < -100) | (r > 100))
    return {
        "n_rows": len(r),
        "min": float(r.min()),
        "max": float(r.max()),
        "p01": float(r.quantile(0.01)),
        "p99": float(r.quantile(0.99)),
        "out_of_pct100_count": int(out_of_range.sum()),
        "out_of_pct100_ratio": round(out_of_range.mean() * 100, 4),
    }


def write_report(out_dir: Path, df: pd.DataFrame, yearly: pd.DataFrame,
                 per_stock: pd.DataFrame, consistency: dict, ratio_range: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    yearly.to_csv(out_dir / "yearly_summary.csv", index=False)
    per_stock.to_csv(out_dir / "per_stock_coverage.csv", index=False)

    print("\n========== 按年规模 ==========")
    print(yearly.to_string(index=False))

    print("\n========== 字段一致性 main ≈ super + big ==========")
    for k, v in consistency.items():
        print(f"  {k}: {v}")

    print("\n========== ratio 字段数值合理性 ==========")
    for k, v in ratio_range.items():
        print(f"  {k}: {v}")

    print("\n========== 每股覆盖度分布 ==========")
    print(f"  总 unique stocks: {len(per_stock):,}")
    cov_bins = [0, 30, 90, 180, 250, 500, 1000, 2000, 4000]
    cov_labels = ["<30d", "30-90d", "90-180d", "180-250d", "250-500d", "500-1000d", "1000-2000d", "2000+d"]
    per_stock["cov_bucket"] = pd.cut(per_stock["n_days"], bins=cov_bins, labels=cov_labels)
    cov_dist = per_stock["cov_bucket"].value_counts().sort_index()
    for label, n in cov_dist.items():
        pct = n / len(per_stock) * 100
        bar = "█" * int(pct / 2)
        print(f"  {str(label):>12s}: {n:>5} ({pct:>5.1f}%) {bar}")

    # 图 1: 按年规模
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    ax = axes[0]
    ax.bar(yearly["year"].astype(str), yearly["avg_universe_per_day"], color="C0", alpha=0.85, edgecolor="black")
    ax.set_ylabel("Avg universe size per day")
    ax.set_title("Yearly: Average Universe Size per Trade Date")
    ax.axhline(100, color="orange", linestyle="--", alpha=0.6, label="min for IC validity (100)")
    ax.axhline(500, color="green", linestyle="--", alpha=0.6, label="healthy (500)")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    ax = axes[1]
    ax.bar(yearly["year"].astype(str), yearly["unique_stocks"], color="C2", alpha=0.85, edgecolor="black")
    ax.set_ylabel("Unique stocks")
    ax.set_title("Yearly: Unique Stocks (rolling)")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout(); plt.savefig(out_dir / "yearly_universe_size.png", dpi=110); plt.close()

    # 图 2: 覆盖度分布
    fig, ax = plt.subplots(figsize=(12, 5))
    cov_dist.plot(kind="bar", color="C1", alpha=0.85, edgecolor="black", ax=ax)
    ax.set_ylabel("Stocks")
    ax.set_xlabel("Trade days covered")
    ax.set_title(f"Per-Stock Coverage Distribution (total {len(per_stock):,} stocks)")
    ax.grid(alpha=0.3, axis="y")
    plt.xticks(rotation=0)
    plt.tight_layout(); plt.savefig(out_dir / "coverage_distribution.png", dpi=110); plt.close()

    # 图 3: 首次出现日期
    fig, ax = plt.subplots(figsize=(14, 5))
    first_dates = per_stock["first_date"].sort_values()
    ax.hist(first_dates, bins=40, color="C3", alpha=0.85, edgecolor="black")
    ax.set_xlabel("First date in collection")
    ax.set_ylabel("Number of stocks")
    ax.set_title("Distribution of Per-Stock First Appearance Date")
    ax.grid(alpha=0.3, axis="y")
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xticks(rotation=45)
    plt.tight_layout(); plt.savefig(out_dir / "first_date_distribution.png", dpi=110); plt.close()

    # 推荐合格 universe
    healthy_mask = yearly["avg_universe_per_day"] >= 500
    healthy_years = yearly.loc[healthy_mask, "year"].tolist()
    first_healthy = min(healthy_years) if healthy_years else None
    stocks_with_2yr_plus = int((per_stock["n_days"] >= 500).sum())
    stocks_with_3yr_plus = int((per_stock["n_days"] >= 750).sum())
    md = [
        "# v0.3 数据质量审计 - 推荐合格 universe",
        "",
        f"## 1. 时间窗口",
        f"- 全 collection: {df['date'].min().date()} → {df['date'].max().date()}",
        f"- 健康年份（avg universe ≥ 500/day）: {healthy_years}",
        f"- **推荐回测起点：{first_healthy}-01-01** （早于此 universe 不足以做截面排序）",
        "",
        f"## 2. 股票筛选",
        f"- 总 unique stocks: {len(per_stock):,}",
        f"- 覆盖 ≥ 2 年（500 trade days）: {stocks_with_2yr_plus:,} 只",
        f"- 覆盖 ≥ 3 年（750 trade days）: {stocks_with_3yr_plus:,} 只",
        f"- **推荐：仅用覆盖 ≥ 2 年的股**",
        "",
        f"## 3. 字段一致性",
        f"- main_net_inflow ≈ super + big：偏差 < 1 元 的占 {consistency['exact_equal_pct']}%",
        f"  - 中位偏差: {consistency['rel_diff_median_pct']}%",
        f"  - p95 偏差: {consistency['rel_diff_p95_pct']}%",
        f"  - p99 偏差: {consistency['rel_diff_p99_pct']}%",
        f"- {'⚠️ main 与 super+big 偏差较大，需用 super+big 自己合成' if consistency['rel_diff_median_pct'] > 1 else '✓ main 字段可用'}",
        "",
        f"## 4. ratio 字段合理性",
        f"- 范围: [{ratio_range['min']:.2f}, {ratio_range['max']:.2f}]",
        f"- 1%/99% 分位: [{ratio_range['p01']:.2f}, {ratio_range['p99']:.2f}]",
        f"- 超 ±100 的: {ratio_range['out_of_pct100_count']} 行 ({ratio_range['out_of_pct100_ratio']}%)",
        f"- {'⚠️ ratio 有 outlier，需 winsorize' if ratio_range['out_of_pct100_ratio'] > 1 else '✓ ratio 字段干净'}",
        "",
        "## 5. 下一步建议",
        f"- v0.3-refined spike: 仅使用 {first_healthy} 起的数据 × 覆盖 ≥ 2 年的股",
        f"- score 公式：用 (super + big) 自己合成 main 避免一致性偏差",
        f"- 重跑 IC + 分桶 + 极端 portfolio + 按年度 IC",
    ]
    (out_dir / "recommended_universe.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n=== Recommended universe report: {out_dir / 'recommended_universe.md'}")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_mongo()
    all_trade_dates = pd.DatetimeIndex(sorted(df["date"].unique()))
    logger.info("Auditing %d rows, %d stocks, %d trade dates ...",
                len(df), df["stock_id"].nunique(), len(all_trade_dates))

    yearly = audit_yearly(df)
    per_stock = audit_per_stock(df, all_trade_dates)
    consistency = audit_field_consistency(df)
    ratio_range = audit_ratio_range(df)

    write_report(OUTPUT_DIR, df, yearly, per_stock, consistency, ratio_range)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
