"""训练 v0.8 GBDT 生产模型 (on full 4y mongo data), 持久化到 strategies/.../models/.

Usage:
    MONGO_PWD='...' uv run --with pymongo --with lightgbm --with joblib python scripts/train_v0_8_production.py

Outputs:
    strategies/cn_a_big_money_rotation/models/v0_8_gbdt_full_4y.joblib
    reports/v0_8_production/train_summary.json
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from strategies.cn_a_big_money_rotation.data import load_mongo_money_flow_panel
from strategies.cn_a_big_money_rotation.signal import (
    build_v0_8_feature_panel, train_v0_8_model, save_v0_8_model,
    V0_8_ALL_FEATS, V0_8_DEFAULT_LGBM_PARAMS,
)

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent / "strategies/cn_a_big_money_rotation/models"
REPORT_DIR = Path("/home/davidlyu/projects/GinkgoBrain/reports/v0_8_production")
TRAIN_START = "2022-01-01"  # mongo earliest useful


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    panel = load_mongo_money_flow_panel(TRAIN_START)
    logger.info("Building feature panel ...")
    df_long = build_v0_8_feature_panel(panel)
    logger.info("Feature panel: %d rows, date range %s → %s",
                len(df_long), df_long["date"].min(), df_long["date"].max())

    logger.info("Training final GBDT model on full 4y ...")
    model = train_v0_8_model(df_long)

    model_path = MODEL_DIR / "v0_8_gbdt_full_4y.joblib"
    save_v0_8_model(model, model_path)
    logger.info("Saved model: %s", model_path)

    fi = dict(zip(V0_8_ALL_FEATS, model.feature_importances_.tolist()))
    fi_sorted = sorted(fi.items(), key=lambda x: -x[1])
    print("\nFeature importance (gain, sorted):")
    for f, imp in fi_sorted:
        print(f"  {f:<32} {imp:>8.0f}")

    summary = {
        "train_start": TRAIN_START,
        "train_end_actual": str(df_long["date"].max().date()),
        "n_rows": len(df_long),
        "n_stocks": int(df_long["stock_id"].nunique()),
        "n_dates": int(df_long["date"].nunique()),
        "features": V0_8_ALL_FEATS,
        "lgbm_params": V0_8_DEFAULT_LGBM_PARAMS,
        "feature_importance": dict(fi_sorted),
        "model_path": str(model_path),
    }
    (REPORT_DIR / "train_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    logger.info("Train summary: %s", REPORT_DIR / "train_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
