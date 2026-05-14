"""v0.4 阈值扫描 + 成本压力测试 spike.

承接 spike_v0_4_avoid_filter.py 的发现（Excl Bot d1 5% 跨周期跑赢 BM2 +64.6%）.

5 个阈值 × 3 个成本水平 = 15 个 portfolio：
- thresholds: Bot 1%, 3%, 5%, 10%, 15%
- daily frictions: 0.0001（轻）, 0.0005（中），0.001（重）

也记录每个阈值的 daily turnover，用于校准成本假设.
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

OUTPUT_DIR = Path("/home/davidlyu/projects/GinkgoBrain/reports/spike_v0_4_threshold_cost")
MONGO_HOST = "192.168.1.69"
MONGO_PORT = 27017
MONGO_USER = "admin"
MONGO_DB = "treasure"
MONGO_COLL = "stock_fund_flow"
EFFECTIVE_START_DATE = pd.Timestamp("2022-01-01")
MIN_COVERAGE_DAYS = 500

THRESHOLDS = [0.01, 0.03, 0.05, 0.10, 0.15]
FRICTIONS = [0.0001, 0.0005, 0.001]


def load_and_filter() -> pd.DataFrame:
    from pymongo import MongoClient
    pwd = os.environ.get("MONGO_PWD")
    if not pwd:
        raise RuntimeError("MONGO_PWD env var required")
    client = MongoClient(host=MONGO_HOST, port=MONGO_PORT,
                         username=MONGO_USER, password=pwd, authSource="admin")
    col = client[MONGO_DB][MONGO_COLL]
    logger.info("Loading mongo ...")
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
    spread = super_pivot - small_pivot
    daily_size = price_pivot.notna().sum(axis=1)
    max_size = int(daily_size.max())
    valid_dates = price_pivot.index[daily_size >= max_size * 0.5]
    return {"spread_d1": spread.loc[valid_dates],
            "price": price_pivot.loc[valid_dates]}


def portfolio_for_threshold(spread_d1: pd.DataFrame, price: pd.DataFrame,
                            threshold: float, friction: float) -> dict:
    universe_mask = price.notna() & (price > 0)
    daily_ret = price.pct_change()
    bot_thr = spread_d1.quantile(threshold, axis=1)
    is_bot = spread_d1.le(bot_thr, axis=0).fillna(False)
    final_mask = universe_mask & (~is_bot)

    # ⚠️ 修复 lookahead bias：T-1 日的 spread 决定 T 日持仓
    # （否则等于用 T 日盘后才知道的数据决定 T 日已发生收益）
    final_mask = final_mask.shift(1).fillna(False).astype(bool)

    masked = daily_ret.where(final_mask)
    daily_port = masked.mean(axis=1, skipna=True) - friction
    daily_port = daily_port.fillna(0.0)
    eq = (1 + daily_port).cumprod()

    mask_int = final_mask.astype(int)
    daily_turn = (mask_int.diff().abs().sum(axis=1) / mask_int.sum(axis=1).replace(0, np.nan)).fillna(0.0)

    cum = float(eq.iloc[-1] - 1)
    n = max(1, len(eq) - 1)
    ann = float(eq.iloc[-1] ** (252 / n) - 1) if eq.iloc[-1] > 0 else float("nan")
    sh = float(daily_port.mean() / daily_port.std() * np.sqrt(252)) if daily_port.std() > 0 else 0.0
    dd = float((eq / eq.cummax() - 1).min())
    return {"daily_ret": daily_port, "equity": eq,
            "cum": cum, "ann": ann, "sharpe": sh, "max_dd": dd,
            "avg_turnover": float(daily_turn.mean())}


def bm2_baseline(price: pd.DataFrame, friction: float) -> dict:
    universe_mask = price.notna() & (price > 0)
    daily_ret = price.pct_change()
    daily_port = daily_ret.where(universe_mask).mean(axis=1, skipna=True) - friction
    daily_port = daily_port.fillna(0.0)
    eq = (1 + daily_port).cumprod()
    cum = float(eq.iloc[-1] - 1)
    n = max(1, len(eq) - 1)
    ann = float(eq.iloc[-1] ** (252 / n) - 1)
    sh = float(daily_port.mean() / daily_port.std() * np.sqrt(252)) if daily_port.std() > 0 else 0.0
    dd = float((eq / eq.cummax() - 1).min())
    return {"daily_ret": daily_port, "equity": eq,
            "cum": cum, "ann": ann, "sharpe": sh, "max_dd": dd, "avg_turnover": 0.0}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_and_filter()
    panel = pivot_panel(df)
    logger.info("Trade dates: %d, stocks: %d", len(panel["price"]), panel["price"].shape[1])

    rows = []
    results = {}
    bm2_by_friction = {}
    for friction in FRICTIONS:
        bm2 = bm2_baseline(panel["price"], friction)
        bm2_by_friction[friction] = bm2
        rows.append({
            "name": "BM2", "threshold": 0.0, "friction": friction,
            "cum_pct": round(bm2["cum"] * 100, 2),
            "ann_pct": round(bm2["ann"] * 100, 2),
            "sharpe": round(bm2["sharpe"], 2),
            "max_dd_pct": round(bm2["max_dd"] * 100, 2),
            "avg_turnover_pct": 0.0, "excess_vs_bm2_pct": 0.0,
        })

    print("\n========== 15 个 portfolio + 3 个 BM2 ==========")
    print(f"{'thr%':>5} | {'fric':>6} | {'cum%':>9} | {'ann%':>8} | {'sharpe':>7} | {'MaxDD%':>9} | {'turn%':>6} | {'excess':>9}")
    for thr in THRESHOLDS:
        for friction in FRICTIONS:
            r = portfolio_for_threshold(panel["spread_d1"], panel["price"], thr, friction)
            bm2 = bm2_by_friction[friction]
            excess = r["cum"] - bm2["cum"]
            results[(thr, friction)] = r
            rows.append({
                "name": f"Excl Bot {thr*100:.0f}%",
                "threshold": thr, "friction": friction,
                "cum_pct": round(r["cum"] * 100, 2),
                "ann_pct": round(r["ann"] * 100, 2),
                "sharpe": round(r["sharpe"], 2),
                "max_dd_pct": round(r["max_dd"] * 100, 2),
                "avg_turnover_pct": round(r["avg_turnover"] * 100, 2),
                "excess_vs_bm2_pct": round(excess * 100, 2),
            })
            print(f"  {thr*100:>3.0f}% | {friction:>6.4f} | {r['cum']*100:>+8.2f}% | {r['ann']*100:>+7.2f}% | {r['sharpe']:>+7.2f} | {r['max_dd']*100:>+8.2f}% | {r['avg_turnover']*100:>+5.2f}% | {excess*100:>+8.2f}%")
        print()

    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)

    # Yearly @ friction=0.0001
    print("\n========== 按年度 excess vs BM2 (friction=0.0001) ==========")
    bm2_eq = bm2_by_friction[0.0001]["equity"]
    bm2_yr = bm2_eq.groupby(bm2_eq.index.year).apply(
        lambda g: float(g.iloc[-1] / g.iloc[0] - 1) if len(g) > 1 else 0.0
    )
    yearly_data = {"BM2": bm2_yr * 100}
    for thr in THRESHOLDS:
        eq = results[(thr, 0.0001)]["equity"]
        yr = eq.groupby(eq.index.year).apply(
            lambda g: float(g.iloc[-1] / g.iloc[0] - 1) if len(g) > 1 else 0.0
        )
        yearly_data[f"Excl Bot {thr*100:.0f}%"] = yr * 100
    yearly_df = pd.DataFrame(yearly_data)
    excess_df = yearly_df.subtract(yearly_df["BM2"], axis=0).round(2)
    print("Yearly cum%:")
    print(yearly_df.round(2).to_string())
    print("\nExcess vs BM2:")
    print(excess_df.round(2).to_string())
    yearly_df.to_csv(OUTPUT_DIR / "yearly_by_threshold.csv")

    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps({"params": {"thresholds": THRESHOLDS, "frictions": FRICTIONS},
                    "rows": rows}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    # 图 1: alpha vs threshold 在 3 个 friction
    fig, ax = plt.subplots(figsize=(12, 5))
    for friction in FRICTIONS:
        excess = [results[(t, friction)]["cum"] * 100 - bm2_by_friction[friction]["cum"] * 100
                  for t in THRESHOLDS]
        ax.plot([t * 100 for t in THRESHOLDS], excess, "-o",
                label=f"friction {friction:.4f}/d", linewidth=1.6)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Bot threshold %")
    ax.set_ylabel("4y cum excess vs BM2 (%)")
    ax.set_title("v0.4: 4y cum excess by Bot threshold, per friction")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "threshold_alpha_curve.png", dpi=110); plt.close()

    # 图 2: cumulative @ friction=0.0001
    fig, ax = plt.subplots(figsize=(14, 6))
    bm2_eq = bm2_by_friction[0.0001]["equity"]
    ax.plot(bm2_eq.index, bm2_eq.values, "k--", label="BM2", linewidth=1.6)
    cmap = plt.cm.viridis
    for i, thr in enumerate(THRESHOLDS):
        eq = results[(thr, 0.0001)]["equity"]
        ax.plot(eq.index, eq.values, color=cmap(i / max(1, len(THRESHOLDS) - 1)),
                label=f"Excl Bot {thr*100:.0f}%", linewidth=1.5)
    ax.set_title("v0.4 Threshold scan: Cumulative @ friction=0.0001")
    ax.legend(); ax.grid(alpha=0.3); ax.set_yscale("log")
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "cumulative_compare.png", dpi=110); plt.close()

    # 图 3: yearly excess bar
    fig, ax = plt.subplots(figsize=(14, 6))
    years = excess_df.index.tolist()
    width = 0.15
    x = np.arange(len(years))
    for i, thr in enumerate(THRESHOLDS):
        col = f"Excl Bot {thr*100:.0f}%"
        ax.bar(x + i * width - 2 * width, excess_df[col].values, width, label=col)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x); ax.set_xticklabels(years)
    ax.set_ylabel("Excess vs BM2 (%)"); ax.set_xlabel("Year")
    ax.set_title("v0.4 Yearly excess vs BM2 (friction=0.0001)")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "annual_excess.png", dpi=110); plt.close()

    print(f"\nAll outputs in: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
