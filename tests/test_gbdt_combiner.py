"""
TC-D1~D4 — GBDTCombiner unit + integration tests.

TC-D1: label double-protection (VOL_FLOOR + clip)
TC-D2: sharpe_to_position mapping
TC-D3: GBDT training convergence (early stopping + R² > 0.01)
TC-D4: SHAP attribution (dimension + additive property)
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.gbdt_combiner import (
    GBDTCombiner,
    build_label_df,
    sharpe_to_position,
    VOL_FLOOR,
    CLIP_RANGE,
)


# ─────────────────────────────────────────── fixtures

def _make_ohlcv(n: int = 300, seed: int = 42, trend: float = 0.0002) -> pd.DataFrame:
    """Synthetic 1d OHLCV with a mild upward drift."""
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1 + rng.normal(trend, 0.015, n))
    df = pd.DataFrame({
        "open":   close * (1 + rng.uniform(-0.005, 0.005, n)),
        "high":   close * (1 + rng.uniform(0.000, 0.010, n)),
        "low":    close * (1 - rng.uniform(0.000, 0.010, n)),
        "close":  close,
        "volume": rng.integers(1_000, 10_000, n).astype(float),
    })
    df.index = pd.date_range("2020-01-01", periods=n, freq="D")
    return df


def _make_signals(df: pd.DataFrame, n_signals: int = 5, seed: int = 0,
                  add_predictive: bool = False) -> list[str]:
    """Attach synthetic signal columns to df in-place; return column names."""
    rng = np.random.default_rng(seed)
    cols = [f"sig_{i}" for i in range(n_signals)]
    for col in cols:
        df[col] = rng.uniform(-1, 1, len(df))
    if add_predictive:
        # sig_pred: correlated with true future Sharpe label (low noise → clear signal)
        from agents.gbdt_combiner import build_label_df
        labeled_tmp = build_label_df(df)
        ts = labeled_tmp["target_sharpe"].fillna(0).values
        noise = rng.normal(0, 0.3, len(df))
        raw = ts + noise
        df["sig_pred"] = (raw - raw.mean()) / (raw.std() + 1e-8)
        cols = cols + ["sig_pred"]
    return cols


# ─────────────────────────────────────────── TC-D1

class TestLabelProtection:
    """TC-D1: VOL_FLOOR prevents inf/NaN; clip ensures |target_sharpe| ≤ 5."""

    def test_flat_market_no_inf_nan(self):
        """5 identical prices → future_vol≈0 → without floor sharpe explodes."""
        n_flat = 20
        df = _make_ohlcv(n=200)
        # Overwrite middle 20 bars with constant price
        flat_price = df["close"].iloc[50]
        df.loc[df.index[50:50 + n_flat], "close"] = flat_price
        labeled = build_label_df(df, horizon=5)
        ts = labeled["target_sharpe"].dropna()
        assert not ts.isin([np.inf, -np.inf]).any(), "inf found in target_sharpe"
        assert not ts.isna().any(), "NaN found in target_sharpe"

    def test_flat_explodes_without_floor(self):
        """Bypassing protection → target_sharpe blows up on a flat segment."""
        n_flat = 10
        df = _make_ohlcv(n=200)
        flat_price = df["close"].iloc[50]
        df.loc[df.index[50:50 + n_flat], "close"] = flat_price
        labeled = build_label_df(df, horizon=5, _bypass_protection=True)
        ts = labeled["target_sharpe"].dropna()
        has_extreme = ts.isin([np.inf, -np.inf]).any() or (ts.abs() > 5).any()
        assert has_extreme, "expected blow-up without protection"

    def test_clip_bounds_all_values(self):
        """All non-NaN target_sharpe values must be in [-5, 5]."""
        df = _make_ohlcv(n=500)
        # Inject a huge spike
        df.loc[df.index[100], "close"] = df["close"].iloc[100] * 5.0
        labeled = build_label_df(df, horizon=5)
        ts = labeled["target_sharpe"].dropna()
        lo, hi = CLIP_RANGE
        assert (ts >= lo).all(), f"found value below {lo}"
        assert (ts <= hi).all(), f"found value above {hi}"

    def test_vol_floor_value(self):
        assert VOL_FLOOR == 1e-4, "VOL_FLOOR must be 1e-4 per spec"


# ─────────────────────────────────────────── TC-D2

class TestSharpeToPosition:
    """TC-D2: mapping rule correctness."""

    def test_spec_example(self):
        """[2.0, 0.8, 0.3, -0.2, -1.0] → [4, 3, 2, 1, 0]."""
        inputs  = [2.0, 0.8, 0.3, -0.2, -1.0]
        expected = [4,   3,   2,   1,    0]
        for s, exp in zip(inputs, expected):
            assert sharpe_to_position(s) == exp, f"sharpe={s}: expected {exp}"

    def test_boundary_inclusive(self):
        """Values exactly at boundaries map to the lower bucket."""
        assert sharpe_to_position(1.0) == 3   # not > 1.0
        assert sharpe_to_position(0.5) == 2
        assert sharpe_to_position(0.0) == 1
        assert sharpe_to_position(-0.5) == 0

    def test_boundary_above(self):
        eps = 1e-9
        assert sharpe_to_position(1.0 + eps) == 4
        assert sharpe_to_position(0.5 + eps) == 3
        assert sharpe_to_position(0.0 + eps) == 2
        assert sharpe_to_position(-0.5 + eps) == 1

    def test_custom_thresholds(self):
        thresholds = (2.0, 1.0, 0.0, -1.0)
        assert sharpe_to_position(2.5, thresholds) == 4
        assert sharpe_to_position(1.5, thresholds) == 3
        assert sharpe_to_position(0.5, thresholds) == 2
        assert sharpe_to_position(-0.5, thresholds) == 1
        assert sharpe_to_position(-1.5, thresholds) == 0

    def test_output_range(self):
        for s in np.linspace(-10, 10, 100):
            p = sharpe_to_position(float(s))
            assert 0 <= p <= 4


# ─────────────────────────────────────────── TC-D3

class TestGBDTTraining:
    """TC-D3: training convergence — early stopping + R² + serializability."""

    @pytest.fixture(scope="class")
    def fitted_combiner(self):
        df = _make_ohlcv(n=1000, trend=0.0003)
        sig_cols = _make_signals(df, add_predictive=True)
        combiner = GBDTCombiner({"n_estimators": 500, "early_stopping_rounds": 30})
        combiner.fit(df, sig_cols)
        return combiner, df, sig_cols

    def test_early_stopping_before_500(self, fitted_combiner):
        """Best iteration must be < n_estimators (early stopping fired)."""
        combiner, _, _ = fitted_combiner
        best = combiner.best_iteration
        assert best > 0, "best_iteration not set"
        assert best < 500, f"no early stopping: best_iteration={best}"

    def test_val_r2_positive(self, fitted_combiner):
        """Validation R² must exceed 0.01 (non-trivial fit)."""
        combiner, _, _ = fitted_combiner
        r2 = combiner.val_r2
        assert r2 is not None
        assert r2 > 0.01, f"val R²={r2:.4f} does not exceed 0.01"

    def test_save_load_roundtrip(self, fitted_combiner, tmp_path):
        """Model must survive save → load and produce identical predictions."""
        combiner, df, sig_cols = fitted_combiner
        save_path = tmp_path / "test_model"
        combiner.save(save_path)

        loaded = GBDTCombiner.load(save_path)
        X = df[sig_cols].values.astype(np.float32)
        preds_orig = combiner._model.predict(X)
        preds_load = loaded._model.predict(X)
        np.testing.assert_allclose(preds_orig, preds_load, rtol=1e-5)

    def test_predict_positions_in_range(self, fitted_combiner):
        combiner, df, sig_cols = fitted_combiner
        positions = combiner.predict(df, sig_cols)
        assert positions.between(0, 4).all()
        assert len(positions) == len(df)


# ─────────────────────────────────────────── TC-D4

class TestSHAPAttribution:
    """TC-D4: SHAP dimension + additive decomposition."""

    @pytest.fixture(scope="class")
    def fitted_state(self):
        df = _make_ohlcv(n=400, trend=0.0003)
        sig_cols = _make_signals(df, add_predictive=True)
        combiner = GBDTCombiner({"n_estimators": 200, "early_stopping_rounds": 20})
        combiner.fit(df, sig_cols)
        labeled = build_label_df(df).dropna(subset=["target_sharpe"])
        X = labeled[sig_cols].values.astype(np.float32)
        return combiner, X, sig_cols

    def test_shap_shape(self, fitted_state):
        """shap_values must return (n_samples, n_features)."""
        combiner, X, sig_cols = fitted_state
        sv = combiner.shap_values(X)
        assert sv.shape == (len(X), len(sig_cols)), (
            f"Expected ({len(X)}, {len(sig_cols)}), got {sv.shape}"
        )

    def test_additive_decomposition(self, fitted_state):
        """sum(shap_values[i]) + expected_value ≈ model_prediction[i]."""
        combiner, X, sig_cols = fitted_state
        sv = combiner.shap_values(X)
        ev = combiner.shap_expected_value()
        preds = combiner._model.predict(X)

        # Check every sample
        reconstructed = sv.sum(axis=1) + ev
        np.testing.assert_allclose(reconstructed, preds, atol=1e-4,
                                   err_msg="SHAP additive property violated")

    def test_shap_feature_count_matches_signals(self, fitted_state):
        combiner, X, sig_cols = fitted_state
        sv = combiner.shap_values(X[:1])
        assert sv.shape[1] == len(sig_cols)

    def test_feature_names_stored(self, fitted_state):
        combiner, _, sig_cols = fitted_state
        assert combiner.feature_names == sig_cols
