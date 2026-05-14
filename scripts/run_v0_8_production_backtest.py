"""v0.8 GBDT realistic backtest (mongo panel + production simulator).

两个 mode:
  --mode scope_a : Full 4y in-sample backtest
                   (用 train_v0_8_production.py 训练好的 full 模型 predict 所有日期, simulate)
  --mode scope_b : Walk-forward F1+F2 OOS backtest
                   (内部 re-train F1: 2022-04..2023-12; F2: 2022-04..2024-12;
                    stitch OOS pred 2024-01..2025-12, simulate)

Usage:
    MONGO_PWD='...' uv run --with pymongo --with lightgbm --with joblib \\
        --with matplotlib python scripts/run_v0_8_production_backtest.py --mode scope_a
    ...                                                                  --mode scope_b
    ...                                                                  --mode both

Outputs:
    reports/v0_8_production/scope_{a,b}_summary.json
    reports/v0_8_production/scope_{a,b}_cumulative.png
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from strategies.cn_a_big_money_rotation.data import load_mongo_money_flow_panel
from strategies.cn_a_big_money_rotation.signal import (
    build_v0_8_feature_panel, build_v0_8_inference_panel,
    train_v0_8_model, load_v0_8_model, predict_v0_8,
    V0_8_ALL_FEATS,
)
from strategies.cn_a_big_money_rotation.v0_8_simulator import (
    V0_8_BacktestConfig, simulate_v0_8, predictions_to_pivot,
)

logger = logging.getLogger(__name__)

MODEL_PATH = (Path(__file__).resolve().parent.parent
              / "strategies/cn_a_big_money_rotation/models/v0_8_gbdt_full_4y.joblib")
REPORT_DIR = Path("/home/davidlyu/projects/GinkgoBrain/reports/v0_8_production")
TRAIN_START = "2022-01-01"

FOLDS = [
    ("F1", "2022-04-01", "2023-12-31", "2024-01-01", "2024-12-31"),
    ("F2", "2022-04-01", "2024-12-31", "2025-01-01", "2025-12-31"),
]


def _result_to_row(name: str, r) -> dict:
    return {
        "portfolio": name,
        "n_days": r.n_days,
        "first_date": r.first_date,
        "last_date": r.last_date,
        "cum_pct": round(r.cum_return * 100, 2),
        "ann_pct": round(r.annualized * 100, 2),
        "sharpe": round(r.sharpe, 2),
        "max_dd_pct": round(r.max_drawdown * 100, 2),
        "avg_turn_pct": round(r.avg_turnover * 100, 2),
        "bm2_cum_pct": round(r.bm2_cum * 100, 2),
        "bm2_ann_pct": round(r.bm2_ann * 100, 2),
        "excess_pct": round(r.excess * 100, 2),
        "excess_ann_pct": round(r.excess_ann * 100, 2),
    }


def run_scope_a(panel: dict, friction_grid: list[float]) -> dict:
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found: {MODEL_PATH}; run train_v0_8_production.py first")
    model = load_v0_8_model(MODEL_PATH)
    logger.info("Loaded model: %s", MODEL_PATH)

    inf_df = build_v0_8_inference_panel(panel)
    logger.info("Inference panel: %d rows", len(inf_df))
    pred_series = predict_v0_8(model, inf_df)
    pred_pivot = predictions_to_pivot(pred_series)
    logger.info("Pred pivot: %d dates × %d stocks", *pred_pivot.shape)

    out = {"mode": "scope_a", "predictions_in_sample": True,
           "first_date": str(pred_pivot.index.min().date()),
           "last_date": str(pred_pivot.index.max().date()),
           "n_pred_dates": len(pred_pivot)}

    print("\n========== Scope A (in-sample full 4y) ==========")
    print(f"{'Variant':<60} {'cum':>9} {'ann':>9} {'Sharpe':>7} {'DD':>9} {'turn':>6} {'excess':>9}")
    rows = []

    cfg_simplified = V0_8_BacktestConfig(
        bot_pct=0.05, friction=0.0001,
        apply_limit_up_filter=False, apply_holding_lock=False,
    )
    r_simp = simulate_v0_8(pred_pivot, panel, cfg_simplified)
    name = "Simplified (no limit, no lock) avoid Bot 5%"
    print(f"{name:<60} {r_simp.cum_return*100:>+7.2f}% {r_simp.annualized*100:>+7.2f}% {r_simp.sharpe:>+7.2f} {r_simp.max_drawdown*100:>+7.2f}% {r_simp.avg_turnover*100:>+5.2f}% {r_simp.excess*100:>+7.2f}%")
    rows.append(_result_to_row(name, r_simp))

    for friction in friction_grid:
        cfg_realistic = V0_8_BacktestConfig(
            bot_pct=0.05, friction=friction,
            apply_limit_up_filter=True, apply_holding_lock=True, holding_min_days=3,
        )
        r = simulate_v0_8(pred_pivot, panel, cfg_realistic)
        name = f"Realistic (limit+lock 3d) avoid Bot 5% @ f={friction}"
        print(f"{name:<60} {r.cum_return*100:>+7.2f}% {r.annualized*100:>+7.2f}% {r.sharpe:>+7.2f} {r.max_drawdown*100:>+7.2f}% {r.avg_turnover*100:>+5.2f}% {r.excess*100:>+7.2f}%")
        rows.append(_result_to_row(name, r))

    out["rows"] = rows

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(r_simp.equity_curve.index, r_simp.equity_curve.values, "C0--",
            linewidth=1.4, label=f"Simplified ({r_simp.cum_return*100:+.2f}%)")
    cfg_real = V0_8_BacktestConfig(bot_pct=0.05, friction=0.0001,
                                     apply_limit_up_filter=True, apply_holding_lock=True)
    r_real = simulate_v0_8(pred_pivot, panel, cfg_real)
    ax.plot(r_real.equity_curve.index, r_real.equity_curve.values, "C2-",
            linewidth=1.6, label=f"Realistic f=0.0001 ({r_real.cum_return*100:+.2f}%)")
    ax.set_title("v0.8 production Scope A (in-sample 4y, simplified vs realistic)")
    ax.legend(); ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout(); plt.savefig(REPORT_DIR / "scope_a_cumulative.png", dpi=110); plt.close()

    return out


def run_scope_b(panel: dict) -> dict:
    logger.info("Building feature panel for walk-forward ...")
    df_long = build_v0_8_feature_panel(panel)

    all_preds = []
    for fold_name, ts, te, vs, ve in FOLDS:
        train = df_long[(df_long["date"] >= ts) & (df_long["date"] <= te)]
        test = df_long[(df_long["date"] >= vs) & (df_long["date"] <= ve)]
        logger.info("Fold %s: train %d rows (%s..%s), test %d rows (%s..%s)",
                    fold_name, len(train), ts, te, len(test), vs, ve)
        if len(test) == 0:
            continue
        model = train_v0_8_model(train)
        X_test = test[V0_8_ALL_FEATS].values
        preds = model.predict(X_test)
        fold_df = test[["date", "stock_id"]].copy()
        fold_df["pred"] = preds
        fold_df["fold"] = fold_name
        all_preds.append(fold_df)

    oos_df = pd.concat(all_preds, ignore_index=True)
    pred_pivot = oos_df.pivot_table(index="date", columns="stock_id", values="pred",
                                     aggfunc="first")
    logger.info("Stitched OOS pred pivot: %d dates × %d stocks", *pred_pivot.shape)

    out = {"mode": "scope_b", "predictions_in_sample": False,
           "first_date": str(pred_pivot.index.min().date()),
           "last_date": str(pred_pivot.index.max().date()),
           "n_pred_dates": len(pred_pivot),
           "folds": [list(f) for f in FOLDS]}

    print("\n========== Scope B (walk-forward OOS) ==========")
    print(f"{'Variant':<60} {'cum':>9} {'ann':>9} {'Sharpe':>7} {'DD':>9} {'turn':>6} {'excess':>9}")
    rows = []

    cfg_simplified = V0_8_BacktestConfig(
        bot_pct=0.05, friction=0.0001,
        apply_limit_up_filter=False, apply_holding_lock=False,
    )
    r_simp = simulate_v0_8(pred_pivot, panel, cfg_simplified)
    name = "Simplified (no limit, no lock) avoid Bot 5%"
    print(f"{name:<60} {r_simp.cum_return*100:>+7.2f}% {r_simp.annualized*100:>+7.2f}% {r_simp.sharpe:>+7.2f} {r_simp.max_drawdown*100:>+7.2f}% {r_simp.avg_turnover*100:>+5.2f}% {r_simp.excess*100:>+7.2f}%")
    rows.append(_result_to_row(name, r_simp))

    for friction in [0.0001, 0.0005, 0.001]:
        cfg_realistic = V0_8_BacktestConfig(
            bot_pct=0.05, friction=friction,
            apply_limit_up_filter=True, apply_holding_lock=True, holding_min_days=3,
        )
        r = simulate_v0_8(pred_pivot, panel, cfg_realistic)
        name = f"Realistic (limit+lock 3d) avoid Bot 5% @ f={friction}"
        print(f"{name:<60} {r.cum_return*100:>+7.2f}% {r.annualized*100:>+7.2f}% {r.sharpe:>+7.2f} {r.max_drawdown*100:>+7.2f}% {r.avg_turnover*100:>+5.2f}% {r.excess*100:>+7.2f}%")
        rows.append(_result_to_row(name, r))

    out["rows"] = rows

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(r_simp.equity_curve.index, r_simp.equity_curve.values, "C0--",
            linewidth=1.4, label=f"Simplified ({r_simp.cum_return*100:+.2f}%)")
    cfg_real = V0_8_BacktestConfig(bot_pct=0.05, friction=0.0001,
                                     apply_limit_up_filter=True, apply_holding_lock=True)
    r_real = simulate_v0_8(pred_pivot, panel, cfg_real)
    ax.plot(r_real.equity_curve.index, r_real.equity_curve.values, "C2-",
            linewidth=1.6, label=f"Realistic f=0.0001 ({r_real.cum_return*100:+.2f}%)")
    ax.set_title("v0.8 production Scope B (walk-forward OOS, simplified vs realistic)")
    ax.legend(); ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=30)
    plt.tight_layout(); plt.savefig(REPORT_DIR / "scope_b_cumulative.png", dpi=110); plt.close()

    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["scope_a", "scope_b", "both"], default="both")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    panel = load_mongo_money_flow_panel(TRAIN_START)

    if args.mode in ("scope_a", "both"):
        result = run_scope_a(panel, friction_grid=[0.0001, 0.0005, 0.001])
        (REPORT_DIR / "scope_a_summary.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    if args.mode in ("scope_b", "both"):
        result = run_scope_b(panel)
        (REPORT_DIR / "scope_b_summary.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print(f"\nAll outputs in: {REPORT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
