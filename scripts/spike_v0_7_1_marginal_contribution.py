"""v0.7.1 边际贡献验证 - 单 20d_momentum / 单 5d_reversal portfolio 对照.

Hypothesis to test:
  v0.7 报 base + 20d_momentum portfolio +7.66%, base 单 +5.54%, lift +38%.
  但 20d_momentum 单 ls_t=-8.67 已远强于 base 单 ls_t=+0.72.
  问: combined portfolio 的 lift 是 base 的边际贡献, 还是其实只是 20d_momentum 自己 ?

Test by running 5 portfolios side-by-side (avoid Bot 5%, friction=0.0001):
  A: main_vs_retail_d1 (base alone)        -> 应 = +5.54% (v0.7 复现)
  B: -20d_momentum (cand alone, 取负让 high mom 落 Bot)
  C: base + 20d_momentum (combined, rank-mean)
  D: 5d_reversal (cand alone)
  E: base + 5d_reversal (combined 对照)

Verdict:
  若 B ≈ C: base 边际贡献为 0, fund-flow 信号被 price reversal 完全覆盖
  若 B < C: 互补, base 真有独立 alpha 维度
  若 B > C: base 拖累了 cand (不太可能)

Reuse v0.7 functions via import.
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

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from spike_v0_7_price_overlay import (  # type: ignore
    load_and_filter, pivot_panel, build_signals,
    rank_mean_combine, portfolio_avoid_bot, bm2_baseline,
)

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("/home/davidlyu/projects/GinkgoBrain/reports/spike_v0_7_1_marginal_contribution")
EXCL_BOT_PCT = 0.05
FRICTION = 0.0001


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_filter()
    panel = pivot_panel(df)
    logger.info("Trade dates: %d, stocks: %d", len(panel["price"]), panel["price"].shape[1])

    sigs = build_signals(panel)
    base = sigs["base_main_vs_retail_d1"]
    mom20 = sigs["candidates"]["20d_momentum"]
    rev5 = sigs["candidates"]["5d_reversal"]

    # 注: portfolio_avoid_bot 删 Bot 5% (score 最低端).
    #   - 20d_momentum: high mom -> low fwd, 想 cut high mom 那一端
    #     -> 用 -20d_momentum: 原 high 变 low, 落到 Bot 5%, 被 avoid_bot 剔除 ✓
    #   - 5d_reversal: 原 ls_t=+8.29 positive (high 5d_rev -> high fwd),
    #     想 cut low 5d_rev 那一端 -> 直接用 5d_reversal: 原 low 已在 Bot 5%, 被剔除 ✓

    portfolios = {
        "A: main_vs_retail_d1 (base only)": base,
        "B: -20d_momentum (cand only)": -mom20,
        "C: base + -20d_momentum (rank-mean)": rank_mean_combine(base, -mom20, wa=0.5),
        "D: 5d_reversal (cand only)": rev5,
        "E: base + 5d_reversal (rank-mean)": rank_mean_combine(base, rev5, wa=0.5),
    }

    bm2 = bm2_baseline(panel["price"], FRICTION)
    print(f"\n========== avoid Bot 5% @ friction={FRICTION} ==========")
    print(f"  {'BM2 (univ 等权)':<42} cum {bm2['cum']*100:+7.2f}%  ann {bm2['ann']*100:+6.2f}%  Sharpe {bm2['sharpe']:+.2f}  DD {bm2['max_dd']*100:+6.2f}%")

    results = {}
    rows = []
    for name, score in portfolios.items():
        r = portfolio_avoid_bot(score, panel["price"], EXCL_BOT_PCT, FRICTION)
        excess = r["cum"] - bm2["cum"]
        results[name] = r
        rows.append({
            "portfolio": name,
            "cum_pct": round(r["cum"] * 100, 2),
            "ann_pct": round(r["ann"] * 100, 2),
            "sharpe": round(r["sharpe"], 2),
            "max_dd_pct": round(r["max_dd"] * 100, 2),
            "avg_turn_pct": round(r["avg_turnover"] * 100, 2),
            "excess_pct": round(excess * 100, 2),
        })
        print(f"  {name:<42} cum {r['cum']*100:+7.2f}%  ann {r['ann']*100:+6.2f}%  Sharpe {r['sharpe']:+.2f}  DD {r['max_dd']*100:+6.2f}%  turn {r['avg_turnover']*100:+5.2f}%  excess {excess*100:+6.2f}%")

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(OUTPUT_DIR / "summary.csv", index=False)

    a_cum = results["A: main_vs_retail_d1 (base only)"]["cum"]
    b_cum = results["B: -20d_momentum (cand only)"]["cum"]
    c_cum = results["C: base + -20d_momentum (rank-mean)"]["cum"]
    d_cum = results["D: 5d_reversal (cand only)"]["cum"]
    e_cum = results["E: base + 5d_reversal (rank-mean)"]["cum"]

    print("\n========== 边际贡献分析 ==========")
    print(f"  base (A) vs cand20 (B): A={a_cum*100:+.2f}%  B={b_cum*100:+.2f}%")
    print(f"    -> 谁强: {'cand20' if b_cum > a_cum else 'base'}")
    print(f"  combined20 (C) vs cand20 (B): C={c_cum*100:+.2f}%  B={b_cum*100:+.2f}%  lift={(c_cum-b_cum)*100:+.2f}pp")
    print(f"    -> base 在组合中的边际贡献: {(c_cum-b_cum)*100:+.2f}pp")
    print(f"  base (A) vs cand5 (D): A={a_cum*100:+.2f}%  D={d_cum*100:+.2f}%")
    print(f"    -> 谁强: {'cand5' if d_cum > a_cum else 'base'}")
    print(f"  combined5 (E) vs cand5 (D): E={e_cum*100:+.2f}%  D={d_cum*100:+.2f}%  lift={(e_cum-d_cum)*100:+.2f}pp")
    print(f"    -> base 在组合中的边际贡献: {(e_cum-d_cum)*100:+.2f}pp")

    verdict20 = "互补" if c_cum > b_cum + 0.005 else ("无贡献" if abs(c_cum - b_cum) <= 0.005 else "拖累")
    verdict5 = "互补" if e_cum > d_cum + 0.005 else ("无贡献" if abs(e_cum - d_cum) <= 0.005 else "拖累")
    print(f"\nverdict (20d_momentum): {verdict20}")
    print(f"verdict (5d_reversal):  {verdict5}")

    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps({
            "params": {"excl_bot_pct": EXCL_BOT_PCT, "friction": FRICTION},
            "bm2": {"cum_pct": bm2["cum"]*100, "ann_pct": bm2["ann"]*100, "sharpe": bm2["sharpe"]},
            "portfolios": rows,
            "marginal_analysis": {
                "20d_momentum": {
                    "base_alone": a_cum*100, "cand_alone": b_cum*100, "combined": c_cum*100,
                    "base_marginal_pp": (c_cum - b_cum)*100, "verdict": verdict20,
                },
                "5d_reversal": {
                    "base_alone": a_cum*100, "cand_alone": d_cum*100, "combined": e_cum*100,
                    "base_marginal_pp": (e_cum - d_cum)*100, "verdict": verdict5,
                },
            },
        }, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(bm2["equity"].index, bm2["equity"].values, "k--",
            label=f"BM2 ({bm2['cum']*100:+.2f}%)", linewidth=1.6)
    colors = ["C0", "C2", "C3", "C4", "C5"]
    for (name, r), c in zip(results.items(), colors):
        ax.plot(r["equity"].index, r["equity"].values, color=c, linewidth=1.5,
                label=f"{name} ({r['cum']*100:+.2f}%)")
    ax.set_title(f"v0.7.1 marginal contribution - avoid Bot {int(EXCL_BOT_PCT*100)}% @ friction={FRICTION}")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "cumulative.png", dpi=110); plt.close()

    print(f"\nAll outputs in: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
