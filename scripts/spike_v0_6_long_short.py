"""v0.6 Long-Short market-neutral 验证 - 复用 v0.8 GBDT OOS pred.

5 portfolios on OOS period (2024-01-02 → 2025-10-09, 1.7y):
  1. BM2 (univ 等权): 基准
  2. Long Top 5% (only): 选 GBDT pred 最高 5%, 等权
  3. Avoid Bot 5% (v0.8 baseline): 剔除 pred 最低 5%, 剩余等权
  4. L-S 理想型: 50% long Top 5% + 50% short Bot 5% (cash-equal)
     -- A 股 Bot 5% 做空 实际不可行, 仅 reference
  5. L-S 可执行型: 100% long Top 5% + 100% short BM2 (univ proxy for 期指 hedge)
     -- 期指/沪深500 ETF 短可实现 (basis spread 暂忽略)

Key questions:
  Q1: long_top 单 vs avoid_bot - alpha 在 Top 端还是 Bot 端?
  Q2: ideal L-S (#4) 是否 ~v0.8 L-S t-stat 量级
  Q3: 可执行 L-S (#5) vs ideal (#4) 执行损耗

Gate:
  L-S 可执行 (#5) 4y 年化 alpha > +4%/y → 做空 hedge 有效
  否则 long-only 仍是最佳, 转向 v0.9 扩数据源
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from spike_v0_7_price_overlay import load_and_filter, pivot_panel  # type: ignore

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("/home/davidlyu/projects/GinkgoBrain/reports/spike_v0_6_long_short")
GBDT_PRED_PATH = Path("/home/davidlyu/projects/GinkgoBrain/reports/spike_v0_8_gbdt/oos_predictions.parquet")

TOP_PCT = 0.05
BOT_PCT = 0.05
FRICTION = 0.0001


def build_masks(pred_pivot: pd.DataFrame, price: pd.DataFrame):
    """Returns (is_top, is_bot, in_universe), all aligned, all shift(1) applied."""
    universe_mask = price.notna() & (price > 0)
    valid_pred = pred_pivot.where(universe_mask)
    top_thr = valid_pred.quantile(1 - TOP_PCT, axis=1)
    bot_thr = valid_pred.quantile(BOT_PCT, axis=1)
    is_top = valid_pred.ge(top_thr, axis=0).fillna(False) & universe_mask
    is_bot = valid_pred.le(bot_thr, axis=0).fillna(False) & universe_mask
    is_top = is_top.shift(1).fillna(False).astype(bool)
    is_bot = is_bot.shift(1).fillna(False).astype(bool)
    universe_mask_t1 = universe_mask.shift(1).fillna(False).astype(bool)
    return is_top, is_bot, universe_mask_t1


def equity_metrics(daily_port: pd.Series, oos_dates: pd.DatetimeIndex) -> dict:
    eq = (1 + daily_port.fillna(0.0)).cumprod()
    eq_oos = eq.loc[eq.index.isin(oos_dates)]
    if len(eq_oos) < 2:
        return {"equity": eq_oos, "cum": 0.0, "ann": 0.0, "sharpe": 0.0, "max_dd": 0.0,
                "first_date": "", "last_date": ""}
    eq_norm = eq_oos / eq_oos.iloc[0]
    dr = daily_port.loc[eq_oos.index]
    cum = float(eq_norm.iloc[-1] - 1)
    n = max(1, len(eq_oos) - 1)
    ann = float(eq_norm.iloc[-1] ** (252 / n) - 1) if eq_norm.iloc[-1] > 0 else float("nan")
    sh = float(dr.mean() / dr.std() * np.sqrt(252)) if dr.std() > 0 else 0.0
    dd = float((eq_norm / eq_norm.cummax() - 1).min())
    return {"equity": eq_norm, "cum": cum, "ann": ann, "sharpe": sh, "max_dd": dd,
            "first_date": str(eq_oos.index[0].date()), "last_date": str(eq_oos.index[-1].date())}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading v0.8 GBDT predictions ...")
    oos = pd.read_parquet(GBDT_PRED_PATH)
    logger.info("OOS predictions: %d rows, %s → %s",
                len(oos), oos["date"].min(), oos["date"].max())

    df = load_and_filter()
    panel = pivot_panel(df)
    price = panel["price"]
    logger.info("Trade dates: %d, stocks: %d", len(price), price.shape[1])

    pred_pivot = oos.pivot_table(index="date", columns="stock_id", values="pred", aggfunc="first")
    pred_pivot = pred_pivot.reindex(index=price.index, columns=price.columns)

    oos_dates = pd.DatetimeIndex(sorted(oos["date"].unique()))
    logger.info("OOS days: %d (%s → %s)", len(oos_dates), oos_dates.min().date(), oos_dates.max().date())

    daily_ret = price.pct_change()
    is_top, is_bot, in_univ = build_masks(pred_pivot, price)

    ret_top = daily_ret.where(is_top).mean(axis=1, skipna=True).fillna(0.0)
    ret_bot = daily_ret.where(is_bot).mean(axis=1, skipna=True).fillna(0.0)
    ret_bm2 = daily_ret.where(in_univ).mean(axis=1, skipna=True).fillna(0.0)
    not_bot = in_univ & (~is_bot)
    ret_avoid_bot = daily_ret.where(not_bot).mean(axis=1, skipna=True).fillna(0.0)

    def turn(mask):
        m_int = mask.astype(int)
        return float((m_int.diff().abs().sum(axis=1) /
                      m_int.sum(axis=1).replace(0, np.nan)).fillna(0.0).loc[oos_dates].mean())

    portfolios = {}

    bm2_dr = ret_bm2 - FRICTION
    portfolios["BM2 (univ 等权)"] = {**equity_metrics(bm2_dr, oos_dates),
                                      "daily_ret": bm2_dr, "avg_turnover": 0.0,
                                      "exposure_net": 1.0, "exposure_gross": 1.0}

    top_dr = ret_top - FRICTION
    portfolios["Long Top 5% (only)"] = {**equity_metrics(top_dr, oos_dates),
                                         "daily_ret": top_dr, "avg_turnover": turn(is_top),
                                         "exposure_net": 1.0, "exposure_gross": 1.0}

    abv_dr = ret_avoid_bot - FRICTION
    portfolios["Avoid Bot 5% (v0.8)"] = {**equity_metrics(abv_dr, oos_dates),
                                          "daily_ret": abv_dr, "avg_turnover": turn(not_bot),
                                          "exposure_net": 1.0, "exposure_gross": 1.0}

    ls_ideal_dr = 0.5 * ret_top - 0.5 * ret_bot - FRICTION
    portfolios["L-S 理想 (long Top - short Bot, 50/50)"] = {
        **equity_metrics(ls_ideal_dr, oos_dates),
        "daily_ret": ls_ideal_dr,
        "avg_turnover": (turn(is_top) + turn(is_bot)) / 2,
        "exposure_net": 0.0, "exposure_gross": 1.0}

    ls_exec_dr = ret_top - ret_bm2 - FRICTION
    portfolios["L-S 可执行 (long Top 100% - short BM2 100%)"] = {
        **equity_metrics(ls_exec_dr, oos_dates),
        "daily_ret": ls_exec_dr,
        "avg_turnover": turn(is_top),
        "exposure_net": 0.0, "exposure_gross": 2.0}

    print("\n========== 5 portfolios OOS (2024-01-02 → 2025-10-09, ~1.7y) ==========")
    print(f"{'Portfolio':<48} {'cum':>9} {'ann':>9} {'Sharpe':>7} {'MaxDD':>9} {'turn':>6} {'net':>5} {'gross':>6}")
    rows = []
    for name, p in portfolios.items():
        print(f"{name:<48} {p['cum']*100:>+7.2f}% {p['ann']*100:>+7.2f}% {p['sharpe']:>+7.2f} {p['max_dd']*100:>+7.2f}% {p['avg_turnover']*100:>+5.2f}% {p['exposure_net']:>+5.2f} {p['exposure_gross']:>+5.2f}")
        rows.append({
            "portfolio": name,
            "cum_pct": round(p["cum"] * 100, 2),
            "ann_pct": round(p["ann"] * 100, 2),
            "sharpe": round(p["sharpe"], 2),
            "max_dd_pct": round(p["max_dd"] * 100, 2),
            "avg_turn_pct": round(p["avg_turnover"] * 100, 2),
            "exposure_net": p["exposure_net"],
            "exposure_gross": p["exposure_gross"],
        })
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "portfolios.csv", index=False)

    top_ann = portfolios["Long Top 5% (only)"]["ann"]
    bot_only_dr = ret_bot - FRICTION
    bot_only_pf = equity_metrics(bot_only_dr, oos_dates)
    bm2_ann = portfolios["BM2 (univ 等权)"]["ann"]

    print(f"\n========== Alpha 拆解 (vs BM2 {bm2_ann*100:+.2f}%/y) ==========")
    print(f"  Top 5% alone (long)      : ann {top_ann*100:+.2f}% / 超额 {(top_ann - bm2_ann)*100:+.2f}%  ← long 端 alpha")
    print(f"  Bot 5% alone (long, ref) : ann {bot_only_pf['ann']*100:+.2f}% / 超额 {(bot_only_pf['ann'] - bm2_ann)*100:+.2f}%  ← short 端反向 alpha (正数 = short 有钱)")
    avoid_ann = portfolios["Avoid Bot 5% (v0.8)"]["ann"]
    print(f"  Avoid Bot 5% (v0.8)      : ann {avoid_ann*100:+.2f}% / 超额 {(avoid_ann - bm2_ann)*100:+.2f}%  ← long-only 综合")
    ls_ideal_ann = portfolios["L-S 理想 (long Top - short Bot, 50/50)"]["ann"]
    ls_exec_ann = portfolios["L-S 可执行 (long Top 100% - short BM2 100%)"]["ann"]
    print(f"  L-S 理想 50/50           : ann {ls_ideal_ann*100:+.2f}%  ← absolute (net 0)")
    print(f"  L-S 可执行 (vs BM2)      : ann {ls_exec_ann*100:+.2f}%  ← absolute (net 0)")

    print(f"\n========== Verdict ==========")
    short_end_alpha = -(bot_only_pf['ann'] - bm2_ann)
    long_end_alpha = top_ann - bm2_ann
    print(f"  Long 端 alpha (Top 5% - BM2): {long_end_alpha*100:+.2f}%/y")
    print(f"  Short 端 alpha (BM2 - Bot 5%): {short_end_alpha*100:+.2f}%/y")
    print(f"  对称否: {'对称' if abs(long_end_alpha - short_end_alpha) < 0.02 else ('long 主导' if long_end_alpha > short_end_alpha + 0.02 else 'short 主导')}")
    print(f"\n  Gate (L-S 可执行 > +4%/y): {'PASS ✓' if ls_exec_ann > 0.04 else 'FAIL ✗'}")
    if ls_exec_ann > 0.04:
        print(f"  ← L-S 范式有效, 推进 v0.6 production-grade hedge")
    else:
        print(f"  ← L-S 范式有限, long-only avoid 仍是最佳, 转 v0.9 扩数据源")

    (OUTPUT_DIR / "summary.json").write_text(json.dumps({
        "params": {"top_pct": TOP_PCT, "bot_pct": BOT_PCT, "friction": FRICTION,
                   "oos_start": str(oos_dates.min().date()),
                   "oos_end": str(oos_dates.max().date())},
        "portfolios": rows,
        "alpha_decomp": {
            "long_end_pct": long_end_alpha * 100,
            "short_end_pct": short_end_alpha * 100,
            "ls_ideal_50_50_pct": ls_ideal_ann * 100,
            "ls_exec_vs_bm2_pct": ls_exec_ann * 100,
        },
        "gate": {"ls_exec_gt_4pct": ls_exec_ann > 0.04, "ls_exec_pct": ls_exec_ann * 100},
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = {"BM2 (univ 等权)": "k",
              "Long Top 5% (only)": "C2",
              "Avoid Bot 5% (v0.8)": "C0",
              "L-S 理想 (long Top - short Bot, 50/50)": "C3",
              "L-S 可执行 (long Top 100% - short BM2 100%)": "C1"}
    styles = {"BM2 (univ 等权)": "--"}
    for name, p in portfolios.items():
        eq = p["equity"]
        if len(eq) < 2:
            continue
        ax.plot(eq.index, eq.values, color=colors.get(name, "C7"),
                linestyle=styles.get(name, "-"), linewidth=1.6,
                label=f"{name} ({p['cum']*100:+.2f}%, ann {p['ann']*100:+.2f}%)")
    ax.axhline(1.0, color="gray", linewidth=0.6)
    ax.set_title(f"v0.6 Long-Short - OOS 1.7y ({portfolios['BM2 (univ 等权)']['first_date']} → {portfolios['BM2 (univ 等权)']['last_date']})")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=30)
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "cumulative.png", dpi=110); plt.close()

    fig, ax = plt.subplots(figsize=(10, 5))
    labels = ["Long Top 5%\nalpha vs BM2", "Short Bot 5%\nalpha vs BM2\n(BM2-Bot)",
              "Avoid Bot 5%\n(v0.8)", "L-S 50/50\n(net 0)", "L-S vs BM2\n(net 0)"]
    values = [long_end_alpha*100, short_end_alpha*100,
              (avoid_ann - bm2_ann)*100, ls_ideal_ann*100, ls_exec_ann*100]
    cs = ["C2", "C3", "C0", "C5", "C1"]
    ax.bar(labels, values, color=cs)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("年化 alpha (%)")
    ax.set_title("v0.6 alpha decomposition - long vs short end")
    for i, v in enumerate(values):
        ax.text(i, v + (0.1 if v >= 0 else -0.3), f"{v:+.2f}%", ha="center", fontsize=10)
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "decomp.png", dpi=110); plt.close()

    print(f"\nAll outputs in: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
