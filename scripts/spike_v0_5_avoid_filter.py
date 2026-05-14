"""v0.5 避雷过滤器 portfolio 验证 - 3 个新 score + v0.4 ref × 3 个成本.

基于 v0.5 multi_spread 扫描发现：
- big-small d1: L-S t=+8.12（v0.4 super-small 的 1.6 倍）
- main_vs_retail d1: L-S t=+6.20, L-S mean +0.89%（最大单次量级）
- composite = (big-small + main_vs_retail) / 2: 双信号合成

3 个新 score + 1 个 v0.4 baseline 参照 × 3 个 friction = 12 portfolio + 3 BM2

⚠️ 严格 shift(1) mask 避免 lookahead bias.
每个 portfolio：剔除 Bot 5%，剩余等权日度再平衡.
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

OUTPUT_DIR = Path("/home/davidlyu/projects/GinkgoBrain/reports/spike_v0_5_avoid_filter")
MONGO_HOST = "192.168.1.69"
MONGO_PORT = 27017
MONGO_USER = "admin"
MONGO_DB = "treasure"
MONGO_COLL = "stock_fund_flow"
EFFECTIVE_START_DATE = pd.Timestamp("2022-01-01")
MIN_COVERAGE_DAYS = 500
EXCL_BOT_PCT = 0.05
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
         "super_net_inflow_ratio": 1, "big_net_inflow_ratio": 1,
         "middle_net_inflow_ratio": 1, "small_net_inflow_ratio": 1},
    ))
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    for c in ["price", "super_net_inflow_ratio", "big_net_inflow_ratio",
              "middle_net_inflow_ratio", "small_net_inflow_ratio"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    cov = df.groupby("stock_id").size()
    keep = cov[cov >= MIN_COVERAGE_DAYS].index
    df = df[df["stock_id"].isin(keep)]
    logger.info("Filtered to %d rows / %d stocks", len(df), len(keep))
    return df


def pivot_panel(df: pd.DataFrame) -> dict:
    def piv(col):
        return df.pivot_table(index="date", columns="stock_id", values=col, aggfunc="first")
    panel = {
        "super": piv("super_net_inflow_ratio"),
        "big": piv("big_net_inflow_ratio"),
        "middle": piv("middle_net_inflow_ratio"),
        "small": piv("small_net_inflow_ratio"),
        "price": piv("price"),
    }
    daily_size = panel["price"].notna().sum(axis=1)
    max_size = int(daily_size.max())
    valid_dates = panel["price"].index[daily_size >= max_size * 0.5]
    for k in panel:
        panel[k] = panel[k].loc[valid_dates]
    return panel


def portfolio_avoid_bot(score: pd.DataFrame, price: pd.DataFrame,
                       threshold: float, friction: float) -> dict:
    universe_mask = price.notna() & (price > 0)
    daily_ret = price.pct_change()
    bot_thr = score.quantile(threshold, axis=1)
    is_bot = score.le(bot_thr, axis=0).fillna(False)
    final_mask = universe_mask & (~is_bot)
    # ⚠️ shift(1): T-1 score 决定 T 持仓
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
    universe_mask = universe_mask.shift(1).fillna(False).astype(bool)
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

    scores = {
        "big-small d1": panel["big"] - panel["small"],
        "main_vs_retail d1": (panel["super"] + panel["big"]) - (panel["middle"] + panel["small"]),
        "composite": ((panel["big"] - panel["small"])
                      + ((panel["super"] + panel["big"]) - (panel["middle"] + panel["small"]))) / 2,
        "v0.4 super-small d1 (ref)": panel["super"] - panel["small"],
    }

    rows = []
    results = {}
    bm2_by_friction = {}

    print("\n========== Portfolio 对比 (Bot 5% 剔除, shift(1) point-in-time) ==========")
    print(f"{'score':<28} | {'fric':>6} | {'cum%':>9} | {'ann%':>8} | {'sharpe':>7} | {'MaxDD%':>9} | {'turn%':>6} | {'excess':>9}")
    for friction in FRICTIONS:
        bm2 = bm2_baseline(panel["price"], friction)
        bm2_by_friction[friction] = bm2
        rows.append({
            "name": "BM2", "score": "BM2", "friction": friction,
            "cum_pct": round(bm2["cum"] * 100, 2),
            "ann_pct": round(bm2["ann"] * 100, 2),
            "sharpe": round(bm2["sharpe"], 2),
            "max_dd_pct": round(bm2["max_dd"] * 100, 2),
            "avg_turnover_pct": 0.0, "excess_pct": 0.0,
        })
        print(f"  {'BM2':<26} | {friction:>6.4f} | {bm2['cum']*100:>+8.2f}% | {bm2['ann']*100:>+7.2f}% | {bm2['sharpe']:>+7.2f} | {bm2['max_dd']*100:>+8.2f}% | {0.0:>+5.2f}% | {0.0:>+8.2f}%")
        for score_name, score in scores.items():
            r = portfolio_avoid_bot(score, panel["price"], EXCL_BOT_PCT, friction)
            excess = r["cum"] - bm2["cum"]
            results[(score_name, friction)] = r
            rows.append({
                "name": f"Excl Bot 5%: {score_name}",
                "score": score_name, "friction": friction,
                "cum_pct": round(r["cum"] * 100, 2),
                "ann_pct": round(r["ann"] * 100, 2),
                "sharpe": round(r["sharpe"], 2),
                "max_dd_pct": round(r["max_dd"] * 100, 2),
                "avg_turnover_pct": round(r["avg_turnover"] * 100, 2),
                "excess_pct": round(excess * 100, 2),
            })
            print(f"  {score_name:<26} | {friction:>6.4f} | {r['cum']*100:>+8.2f}% | {r['ann']*100:>+7.2f}% | {r['sharpe']:>+7.2f} | {r['max_dd']*100:>+8.2f}% | {r['avg_turnover']*100:>+5.2f}% | {excess*100:>+8.2f}%")
        print()

    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)

    # 按年度
    print("\n========== 按年度 cum% (@friction=0.0001) ==========")
    bm2_eq = bm2_by_friction[0.0001]["equity"]
    bm2_yr = bm2_eq.groupby(bm2_eq.index.year).apply(
        lambda g: float(g.iloc[-1] / g.iloc[0] - 1) if len(g) > 1 else 0.0
    )
    yearly_data = {"BM2": bm2_yr * 100}
    for name in scores:
        eq = results[(name, 0.0001)]["equity"]
        yr = eq.groupby(eq.index.year).apply(
            lambda g: float(g.iloc[-1] / g.iloc[0] - 1) if len(g) > 1 else 0.0
        )
        yearly_data[name] = yr * 100
    yearly_df = pd.DataFrame(yearly_data)
    excess_df = yearly_df.subtract(yearly_df["BM2"], axis=0).round(2)
    print(yearly_df.round(2).to_string())
    print("\nExcess vs BM2:")
    print(excess_df.round(2).to_string())
    yearly_df.to_csv(OUTPUT_DIR / "yearly_by_score.csv")

    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps({"params": {"scores": list(scores.keys()),
                                "frictions": FRICTIONS,
                                "excl_bot_pct": EXCL_BOT_PCT},
                    "rows": rows}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    # 图 1: cumulative
    fig, ax = plt.subplots(figsize=(14, 6))
    bm2_eq = bm2_by_friction[0.0001]["equity"]
    ax.plot(bm2_eq.index, bm2_eq.values, "k--",
            label=f"BM2 ({bm2_by_friction[0.0001]['cum']*100:+.2f}%)", linewidth=1.6)
    colors = ["C0", "C2", "C3", "C7"]
    for (name, _), c in zip(scores.items(), colors):
        eq = results[(name, 0.0001)]["equity"]
        ax.plot(eq.index, eq.values, color=c, linewidth=1.5,
                label=f"{name} ({results[(name, 0.0001)]['cum']*100:+.2f}%)")
    ax.set_title("v0.5 portfolio - Cumulative @ friction=0.0001 (point-in-time)")
    ax.legend(); ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "cumulative.png", dpi=110); plt.close()

    # 图 2: relative
    fig, ax = plt.subplots(figsize=(14, 5))
    bm2_eq = bm2_by_friction[0.0001]["equity"]
    for (name, _), c in zip(scores.items(), colors):
        rel = results[(name, 0.0001)]["equity"] / bm2_eq
        ax.plot(rel.index, rel.values, color=c, linewidth=1.6,
                label=f"{name} (final {(rel.iloc[-1]-1)*100:+.2f}%)")
    ax.axhline(1.0, color="black", linewidth=0.8)
    ax.set_title("v0.5 portfolio / BM2 (relative)")
    ax.set_ylabel("Portfolio / BM2"); ax.legend(); ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "relative_to_bm2.png", dpi=110); plt.close()

    # 图 3: alpha vs friction
    fig, ax = plt.subplots(figsize=(10, 5))
    for (name, _), c in zip(scores.items(), colors):
        excess = [results[(name, f)]["cum"] * 100 - bm2_by_friction[f]["cum"] * 100
                  for f in FRICTIONS]
        ax.plot([f * 10000 for f in FRICTIONS], excess, "-o", color=c, label=name, linewidth=1.6)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("friction (bps/day)")
    ax.set_ylabel("4y cum excess vs BM2 (%)")
    ax.set_title("v0.5 alpha vs friction across signals")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "alpha_vs_friction.png", dpi=110); plt.close()

    print(f"\nAll outputs in: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
