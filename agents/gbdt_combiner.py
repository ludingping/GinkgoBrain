"""
Layer 3 Combiner — 1d GBDT path (§6 of 信号分层架构与可解释RL设计.md).

Trains a LightGBM regressor to predict future N-bar Sharpe from signal
features, then maps the prediction to a discrete position via fixed
thresholds.  SHAP attributions are computed via TreeExplainer so every
prediction is fully explainable.

Public API
----------
GBDTCombiner(cfg)
    .fit(df, signal_cols)          → self
    .predict(df, signal_cols)      → pd.Series of int position [0..4]
    .shap_values(X)                → np.ndarray (n_samples, n_features)
    .save(path)
    .load(path)                    → GBDTCombiner

build_label_df(df, horizon, vol_floor, clip_range)
    → df with column "target_sharpe"

sharpe_to_position(s, thresholds)  → int 0..4
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

class _BoosterAdapter:
    """Thin wrapper so loaded Booster has the same .predict() interface as LGBMRegressor."""

    def __init__(self, booster: lgb.Booster) -> None:
        self._booster = booster

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._booster.predict(X)

    @property
    def best_iteration_(self) -> int | None:
        return None

    @property
    def booster_(self) -> lgb.Booster:
        return self._booster

    def get_params(self) -> dict:
        return {}


VOL_FLOOR: float = 1e-4
CLIP_RANGE: tuple[float, float] = (-5.0, 5.0)
DEFAULT_HORIZON: int = 5

DEFAULT_THRESHOLDS: tuple[float, float, float, float] = (1.0, 0.5, 0.0, -0.5)

DEFAULT_LGB_PARAMS: dict = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbose": -1,
}


# ─────────────────────────────────────────── label construction

def build_label_df(
    df: pd.DataFrame,
    horizon: int = DEFAULT_HORIZON,
    vol_floor: float = VOL_FLOOR,
    clip_range: tuple[float, float] = CLIP_RANGE,
    *,
    _bypass_protection: bool = False,   # test-only: disable both guards
) -> pd.DataFrame:
    """
    Add column 'target_sharpe' to df (in-place copy).

    Two-layer protection (§6.2.2):
      ① vol_floor prevents division-by-zero on flat markets
      ② clip_range prevents extreme labels from outlier returns
    """
    out = df.copy()
    close = out["close"]
    future_return = close.shift(-horizon) / close - 1
    future_vol = close.pct_change().rolling(horizon).std().shift(-horizon)

    if not _bypass_protection:
        future_vol = np.maximum(future_vol, vol_floor)

    target = future_return / future_vol

    if not _bypass_protection:
        target = np.clip(target, *clip_range)

    out["target_sharpe"] = target
    return out


# ─────────────────────────────────────────── mapping

def sharpe_to_position(
    s: float,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> int:
    """
    Map a predicted Sharpe scalar to a discrete position index 0..4.

    thresholds = (t3, t2, t1, t0) descending; default (1.0, 0.5, 0.0, -0.5).
      s > t3 → 4, s > t2 → 3, s > t1 → 2, s > t0 → 1, else → 0
    """
    t3, t2, t1, t0 = thresholds
    if s > t3:
        return 4
    if s > t2:
        return 3
    if s > t1:
        return 2
    if s > t0:
        return 1
    return 0


# ─────────────────────────────────────────── combiner

class GBDTCombiner:
    """
    LightGBM-based signal combiner for the 1d trading path.

    Parameters
    ----------
    cfg : dict
        Recognised keys (all optional):
          horizon          int   forward window for label (default 5)
          vol_floor        float volatility floor        (default 1e-4)
          clip_range       list  [lo, hi] label clip     (default [-5, 5])
          thresholds       list  4 Sharpe→position gates (default [1,.5,0,-.5])
          train_ratio      float train/val split         (default 0.8)
          lgb_params       dict  overrides for LGBMRegressor
          early_stopping_rounds int                      (default 30)
    """

    def __init__(self, cfg: dict | None = None) -> None:
        cfg = cfg or {}
        self.horizon: int = cfg.get("horizon", DEFAULT_HORIZON)
        self.vol_floor: float = cfg.get("vol_floor", VOL_FLOOR)
        clip = cfg.get("clip_range", list(CLIP_RANGE))
        self.clip_range: tuple[float, float] = (clip[0], clip[1])
        self.thresholds: tuple[float, float, float, float] = tuple(
            cfg.get("thresholds", list(DEFAULT_THRESHOLDS))
        )
        self.train_ratio: float = cfg.get("train_ratio", 0.8)
        self.early_stopping_rounds: int = cfg.get("early_stopping_rounds", 30)

        params = {**DEFAULT_LGB_PARAMS, **cfg.get("lgb_params", {})}
        self._model = lgb.LGBMRegressor(**params)
        self._explainer: shap.TreeExplainer | None = None
        self._feature_names: list[str] = []
        self._val_r2: float | None = None

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame, signal_cols: list[str]) -> "GBDTCombiner":
        """
        Build labels, split train/val, fit LGBMRegressor.

        df must already contain signal columns and a 'close' column.
        Rows with NaN labels (last `horizon` bars) are dropped automatically.
        """
        labeled = build_label_df(
            df, self.horizon, self.vol_floor, self.clip_range
        )
        labeled = labeled.dropna(subset=["target_sharpe"] + signal_cols)

        X = labeled[signal_cols].values.astype(np.float32)
        y = labeled["target_sharpe"].values.astype(np.float32)
        self._feature_names = signal_cols[:]

        split = int(len(X) * self.train_ratio)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        callbacks = [
            lgb.early_stopping(self.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=-1),
        ]
        self._model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=callbacks,
        )

        # Compute val R²
        y_pred_val = self._model.predict(X_val)
        ss_res = np.sum((y_val - y_pred_val) ** 2)
        ss_tot = np.sum((y_val - y_val.mean()) ** 2)
        self._val_r2 = float(1 - ss_res / (ss_tot + 1e-12))

        self._explainer = shap.TreeExplainer(self._model)
        return self

    # ── predict ──────────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame, signal_cols: list[str]) -> pd.Series:
        """Return int Series of position indices (0..4) aligned to df.index."""
        X = df[signal_cols].values.astype(np.float32)
        sharpe_pred = self._model.predict(X)
        positions = np.array(
            [sharpe_to_position(s, self.thresholds) for s in sharpe_pred],
            dtype=np.int32,
        )
        return pd.Series(positions, index=df.index, name="position")

    def predict_sharpe(self, df: pd.DataFrame, signal_cols: list[str]) -> pd.Series:
        """Return raw predicted Sharpe values."""
        X = df[signal_cols].values.astype(np.float32)
        return pd.Series(
            self._model.predict(X).astype(np.float32),
            index=df.index,
            name="predicted_sharpe",
        )

    # ── SHAP ─────────────────────────────────────────────────────────────────

    def shap_values(self, X: np.ndarray) -> np.ndarray:
        """
        Compute SHAP values for a feature matrix X (n_samples, n_features).
        Returns np.ndarray of shape (n_samples, n_features).
        """
        if self._explainer is None:
            raise RuntimeError("Call fit() before shap_values()")
        return self._explainer.shap_values(X)

    def shap_expected_value(self) -> float:
        if self._explainer is None:
            raise RuntimeError("Call fit() before shap_expected_value()")
        return float(self._explainer.expected_value)

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save model + metadata to {path}.lgb and {path}.meta.json."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lgb_path = path.with_suffix(".lgb")
        self._model.booster_.save_model(str(lgb_path))
        meta = {
            "horizon": self.horizon,
            "vol_floor": self.vol_floor,
            "clip_range": list(self.clip_range),
            "thresholds": list(self.thresholds),
            "train_ratio": self.train_ratio,
            "early_stopping_rounds": self.early_stopping_rounds,
            "feature_names": self._feature_names,
            "val_r2": self._val_r2,
            "lgb_params": self._model.get_params(),
        }
        path.with_suffix(".meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False)
        )

    @classmethod
    def load(cls, path: str | Path) -> "GBDTCombiner":
        """Load from {path}.lgb and {path}.meta.json."""
        path = Path(path)
        meta_path = path.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text())

        obj = cls(meta)
        obj._feature_names = meta["feature_names"]
        obj._val_r2 = meta.get("val_r2")

        booster = lgb.Booster(model_file=str(path.with_suffix(".lgb")))
        # Store booster directly; wrap in a thin adapter so predict() works
        obj._model = _BoosterAdapter(booster)
        obj._explainer = shap.TreeExplainer(booster)
        return obj

    # ── properties ───────────────────────────────────────────────────────────

    @property
    def val_r2(self) -> float | None:
        return self._val_r2

    @property
    def feature_names(self) -> list[str]:
        return self._feature_names[:]

    @property
    def best_iteration(self) -> int:
        bi = getattr(self._model, "best_iteration_", None)
        if bi is not None:
            return int(bi)
        booster = getattr(self._model, "booster_", None)
        if booster is not None:
            return int(booster.best_iteration)
        return -1
