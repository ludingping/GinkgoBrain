"""v0.4 避雷过滤器 portfolio spike — long-only 验证.

基于 v0.4 IC spike 发现（super - small spread 跨周期反向显著）：
- d10 B10（持续 spread Top 10%）跑输（v0.2 "持续吸筹" 反向 alpha 同款）
- d1 Bot 5%（单日"散户极端买 + 主力极端卖"）跑输

假设：long-only 框架下，剔除这两类"避雷股"+ 等权剩余 universe → 跑赢 BM2

4 个对比 portfolio：
1. BM2          全 universe 等权日度再平衡（baseline）
2. Excl_Top_d10  剔除 d10 spread Top 10%，剩余等权
3. Excl_Bot_d1   剔除 d1 spread Bot 5%，剩余等权
4. Excl_Both     同时剔除两个尾部

每日等权日度再平衡，含 0.01% 摩擦.
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

OUTPUT_DIR = Path("/home/davidlyu/projects/GinkgoBrain/reports/spike_v0_4_avoid_filter")
MONGO_HOST = "192.168.1.69"
MONGO_PORT = 27017
MONGO_USER = "admin"
MONGO_DB = "treasure"
MONGO_COLL = "stock_fund_flow"

EFFECTIVE_START_DATE = pd.Timestamp("2022-01-01")
MIN_COVERAGE_DAYS = 500
DAILY_FRICTION = 0.0001
EXCL_TOP_PCT = 0.10
EXCL_BOT_PCT = 0.05
D_ROLLING = 10


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
            "spread_d10": spread.rolling(D_ROLLING, min_periods=D_ROLLING).mean().loc[valid_dates],
            "price": price_pivot.loc[valid_dates]}


def daily_returns(price: pd.DataFrame) -> pd.DataFrame:
    return price.pct_change()


def portfolio_returns(daily_ret: pd.DataFrame, mask_in: pd.DataFrame,
                      friction: float = DAILY_FRICTION) -> pd.Series:
    masked = daily_ret.where(mask_in)
    daily_port = masked.mean(axis=1, skipna=True) - friction
    return daily_port.fillna(0.0)


def cumulative_equity(daily_port: pd.Series) -> pd.Series:
    return (1 + daily_port).cumprod()


def annualize_return(equity: pd.Series) -> float:
    n = len(equity) - 1
    if n <= 0 or equity.iloc[-1] <= 0:
        return float("nan")
    return float(equity.iloc[-1] ** (252 / n) - 1)


def sharpe_ratio(daily_ret: pd.Series, periods: int = 252) -> float:
    r = daily_ret.dropna()
    if len(r) < 2 or r.std() < 1e-12:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(periods))


def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1).min())


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_and_filter()
    panel = pivot_panel(df)
    logger.info("Trade dates: %d, stocks: %d", len(panel["price"]), panel["price"].shape[1])

    daily_ret = daily_returns(panel["price"])
    universe_mask = panel["price"].notna() & (panel["price"] > 0)

    # d10 Top 10% mask（剔除）
    d10 = panel["spread_d10"]
    d10_thr = d10.quantile(1 - EXCL_TOP_PCT, axis=1)
    is_top_d10 = d10.ge(d10_thr, axis=0).fillna(False)
    excl_top_d10_mask = universe_mask & (~is_top_d10)

    # d1 Bot 5% mask（剔除）
    d1 = panel["spread_d1"]
    d1_thr = d1.quantile(EXCL_BOT_PCT, axis=1)
    is_bot_d1 = d1.le(d1_thr, axis=0).fillna(False)
    excl_bot_d1_mask = universe_mask & (~is_bot_d1)

    excl_both_mask = excl_top_d10_mask & excl_bot_d1_mask

    # ⚠️ 修复 lookahead bias：T-1 日的 spread 决定 T 日持仓
    excl_top_d10_mask = excl_top_d10_mask.shift(1).fillna(False).astype(bool)
    excl_bot_d1_mask = excl_bot_d1_mask.shift(1).fillna(False).astype(bool)
    excl_both_mask = excl_both_mask.shift(1).fillna(False).astype(bool)
    universe_mask = universe_mask.shift(1).fillna(False).astype(bool)  # BM2 也 shift 保持一致

    portfolios = {
        "BM2 (full universe)": universe_mask,
        "Excl Top d10 10%": excl_top_d10_mask,
        "Excl Bot d1 5%": excl_bot_d1_mask,
        "Excl Both (避雷双过滤)": excl_both_mask,
    }

    results = {}
    print("\n========== Portfolio 对比（全样本 4 年）==========")
    print(f"{'name':<28} | {'cum%':>9} | {'ann%':>8} | {'sharpe':>7} | {'MaxDD%':>9}")
    for name, mask in portfolios.items():
        port_ret = portfolio_returns(daily_ret, mask)
        eq = cumulative_equity(port_ret)
        cum = float(eq.iloc[-1] - 1) if len(eq) > 0 else 0.0
        ann = annualize_return(eq)
        sh = sharpe_ratio(port_ret)
        dd = max_drawdown(eq)
        results[name] = {"daily_ret": port_ret, "equity": eq,
                         "cum": cum, "ann": ann, "sharpe": sh, "max_dd": dd}
        print(f"  {name:<26} | {cum*100:>+8.2f}% | {ann*100:>+7.2f}% | {sh:>+7.2f} | {dd*100:>+8.2f}%")

    # 按年度
    print(f"\n========== 按年度累积涨幅 ==========")
    yearly_table = {}
    for name, r in results.items():
        eq = r["equity"]
        yr = eq.groupby(eq.index.year).apply(
            lambda g: float(g.iloc[-1] / g.iloc[0] - 1) if len(g) > 1 else 0.0
        )
        yearly_table[name] = yr
    cmp = pd.DataFrame({name: yearly_table[name] for name in results}) * 100
    cmp["Excl Top - BM2"] = cmp["Excl Top d10 10%"] - cmp["BM2 (full universe)"]
    cmp["Excl Bot - BM2"] = cmp["Excl Bot d1 5%"] - cmp["BM2 (full universe)"]
    cmp["Excl Both - BM2"] = cmp["Excl Both (避雷双过滤)"] - cmp["BM2 (full universe)"]
    print(cmp.round(2).to_string())
    cmp.to_csv(OUTPUT_DIR / "yearly_comparison.csv")

    # JSON
    summary = {
        "params": {
            "excl_top_pct": EXCL_TOP_PCT, "excl_bot_pct": EXCL_BOT_PCT,
            "d_rolling": D_ROLLING, "daily_friction": DAILY_FRICTION,
            "data": "MongoDB treasure.stock_fund_flow >=2022, coverage >=500d",
        },
        "portfolios": {
            name: {
                "cum_pct": round(r["cum"] * 100, 2),
                "ann_pct": round(r["ann"] * 100, 2),
                "sharpe": round(r["sharpe"], 2),
                "max_dd_pct": round(r["max_dd"] * 100, 2),
            }
            for name, r in results.items()
        },
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    # 图 1: cumulative
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ["black", "C0", "C2", "C3"]
    styles = ["--", "-", "-", "-"]
    for (name, r), c, s in zip(results.items(), colors, styles):
        ax.plot(r["equity"].index, r["equity"].values, label=name, color=c, linewidth=1.6, linestyle=s)
    ax.set_title("v0.4 避雷过滤器 vs BM2: Cumulative Equity (4y)")
    ax.set_ylabel("Cumulative (1+r)")
    ax.legend(); ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "cumulative.png", dpi=110); plt.close()

    # 图 2: 相对 BM2
    fig, ax = plt.subplots(figsize=(14, 5))
    bm2_eq = results["BM2 (full universe)"]["equity"]
    for name, r in results.items():
        if name == "BM2 (full universe)":
            continue
        rel = r["equity"] / bm2_eq
        ax.plot(rel.index, rel.values,
                label=f"{name}  (final {(rel.iloc[-1]-1)*100:+.2f}%)", linewidth=1.6)
    ax.axhline(1.0, color="black", linewidth=0.8)
    ax.set_title("v0.4 避雷过滤器 / BM2 (relative)")
    ax.set_ylabel("Portfolio / BM2")
    ax.legend(); ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "relative_to_bm2.png", dpi=110); plt.close()

    print(f"\nAll outputs in: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
