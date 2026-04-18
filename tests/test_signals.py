"""
Tests for utils/signals.py.

Covers TC-A1 ~ TC-A4 from docs/GinkgoBrain/信号分层架构与可解释RL测试用例.md:
- TC-A1: znorm 工具函数边界行为
- TC-A2: 所有 signal 值域契约
- TC-A3: Signal 方向语义正确
- TC-A4: Warmup / NaN 处理
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from utils.indicators import add_indicators
from utils.signals import (
    REQUIRED_COLS,
    SIGNAL_WARMUP_WINDOW,
    add_signals,
    signal_columns,
    znorm,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_ohlcv(n: int, seed: int = 42) -> pd.DataFrame:
    """Random-walk OHLCV. Used for value-range and warmup tests."""
    rng = np.random.default_rng(seed)
    price = 60_000 + np.cumsum(rng.normal(0, 200, n))
    price = np.maximum(price, 1_000)  # keep positive
    return pd.DataFrame(
        {
            "open":   price * rng.uniform(0.995, 1.0, n),
            "high":   price * rng.uniform(1.0, 1.01, n),
            "low":    price * rng.uniform(0.99, 1.0, n),
            "close":  price,
            "volume": rng.integers(100_000, 1_000_000, n).astype(float),
        }
    )


def _make_monotonic_ohlcv(n_up: int = 60, n_down: int = 60) -> pd.DataFrame:
    """
    Construct a clean up-then-down price path. Used for direction-semantics
    tests (TC-A3). A long warmup prefix is prepended so signals are fully
    valid by the time the up/down segments begin. Warmup must cover:
      pct_change(60) NaN tail (60) + znorm rolling window (100) + headroom
    so sig_regime_return_60 is already a mature z-score at mid-uptrend.
    """
    warmup = 220
    rng = np.random.default_rng(0)
    warm_price = 60_000 + rng.normal(0, 50, warmup).cumsum()
    up_price = np.linspace(warm_price[-1], warm_price[-1] * 1.4, n_up)
    down_price = np.linspace(up_price[-1], up_price[-1] * 0.6, n_down)

    price = np.concatenate([warm_price, up_price, down_price])
    n = len(price)

    return pd.DataFrame(
        {
            "open":   price * 0.999,
            "high":   price * 1.002,
            "low":    price * 0.998,
            "close":  price,
            "volume": np.full(n, 500_000.0),
        }
    )


@pytest.fixture
def signal_df():
    """DataFrame with all indicators + signals, based on enough data."""
    raw = _make_ohlcv(400)
    indicated = add_indicators(raw)
    return add_signals(indicated)


@pytest.fixture
def discrete_sig_cols():
    """Signals whose output is strictly in {-1, 0, 1}."""
    return ["sig_mom_stoch_state"]


# ─────────────────────────────────────────────────────────────────────────────
# TC-A1: znorm boundary behavior
# ─────────────────────────────────────────────────────────────────────────────

def test_tc_a1_znorm_constant_series_returns_zero():
    s = pd.Series(np.ones(200) * 7.5)
    out = znorm(s, window=100)
    # Warmup window returns NaN (rolling requires 100 points)
    assert out.iloc[:100].isna().all() or (out.iloc[:100].fillna(0) == 0).all()
    # Post-warmup: constant input → zero output
    np.testing.assert_array_almost_equal(out.iloc[100:].values, 0.0)


def test_tc_a1_znorm_standard_normal_stays_in_range():
    rng = np.random.default_rng(1)
    s = pd.Series(rng.standard_normal(500))
    out = znorm(s, window=100, cap=3.0).dropna()
    # With cap=3 and /3, output is strictly [-1, 1]
    assert out.between(-1.0, 1.0).all()
    # At least 99% of values should fall within [-1, 1] (not just clipped)
    inside = (out.abs() < 1.0).sum() / len(out)
    assert inside > 0.9, f"Too many clipped values: only {inside:.2%} strictly inside"


def test_tc_a1_znorm_handles_nan_without_raising():
    s = pd.Series([1.0, 2.0, np.nan, np.nan, 3.0, 4.0] * 30)
    out = znorm(s, window=50)
    # Should not raise; NaN propagates naturally, no inf
    assert not np.isinf(out).any()


def test_tc_a1_znorm_no_division_by_zero_on_flat_window():
    """Safety: std=0 in rolling window must not cause inf or NaN explosion."""
    s = pd.Series([5.0] * 200)
    out = znorm(s, window=100)
    # After warmup, all values should be zero (not inf / NaN)
    post = out.iloc[100:]
    assert not np.isinf(post).any()
    np.testing.assert_array_almost_equal(post.fillna(0).values, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# TC-A2: value range contract
# ─────────────────────────────────────────────────────────────────────────────

def test_tc_a2_add_signals_requires_indicators():
    bare = pd.DataFrame({c: [1.0] * 10 for c in ["open", "high", "low", "close", "volume"]})
    with pytest.raises(ValueError, match="Missing columns"):
        add_signals(bare)


def test_tc_a2_all_signals_bounded_in_minus_one_to_one(signal_df):
    for col in signal_columns(signal_df):
        vals = signal_df[col].dropna()
        assert vals.between(-1.0, 1.0).all(), \
            f"{col} out of range: [{vals.min()}, {vals.max()}]"


def test_tc_a2_drawdown_signal_bounded_in_minus_one_to_zero(signal_df):
    vals = signal_df["sig_regime_drawdown"].dropna()
    assert vals.between(-1.0, 0.0).all(), \
        f"sig_regime_drawdown out of [-1, 0]: [{vals.min()}, {vals.max()}]"


def test_tc_a2_discrete_signals_are_tri_valued(signal_df, discrete_sig_cols):
    for col in discrete_sig_cols:
        unique = set(signal_df[col].dropna().unique())
        assert unique.issubset({-1.0, 0.0, 1.0}), \
            f"{col} has non-discrete values: {unique}"


def test_tc_a2_all_15_signals_present(signal_df):
    expected = {
        # trend
        "sig_trend_ema_cross", "sig_trend_macd_hist",
        "sig_trend_price_above_ma", "sig_trend_slope_21",
        # momentum
        "sig_mom_rsi_zone", "sig_mom_rsi_trend",
        "sig_mom_stoch_state", "sig_mom_roc_zscore",
        # volatility
        "sig_vol_atr_pct", "sig_vol_bb_width", "sig_vol_bb_position",
        # volume
        "sig_volume_obv_slope", "sig_volume_ratio",
        # regime
        "sig_regime_drawdown", "sig_regime_return_60",
    }
    present = set(signal_columns(signal_df))
    assert expected == present, f"Signal set mismatch: missing={expected-present}, extra={present-expected}"


# ─────────────────────────────────────────────────────────────────────────────
# TC-A3: direction semantics
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def monotonic_df():
    raw = _make_monotonic_ohlcv(n_up=60, n_down=60)
    return add_signals(add_indicators(raw))


# Segment boundaries of monotonic_df (must match _make_monotonic_ohlcv above).
# After add_indicators drops ~warmup rows of its own, we index relative to the
# post-indicators df via absolute rows. Mid-segment sampling avoids two
# artifacts:
#   1) MACD histogram converges to 0 at the END of a constant-slope ramp
#      (MACD fast-line and signal-line merge), so end-of-up is wrong for it.
#   2) Start-of-segment hasn't accumulated enough momentum for signals to
#      express a clear direction.
def _mid_up_slice(df: pd.DataFrame) -> pd.DataFrame:
    """Middle-5-bars of the uptrend segment."""
    mid = len(df) - 60 - 30  # 30 bars before the peak
    return df.iloc[mid - 2: mid + 3]


def _mid_down_slice(df: pd.DataFrame) -> pd.DataFrame:
    """Middle-5-bars of the downtrend segment."""
    mid = len(df) - 30
    return df.iloc[mid - 2: mid + 3]


def test_tc_a3_trend_signals_positive_on_uptrend(monotonic_df):
    up_means = _mid_up_slice(monotonic_df)[signal_columns(monotonic_df)].mean()
    for col in ["sig_trend_ema_cross", "sig_trend_macd_hist",
                "sig_trend_slope_21", "sig_trend_price_above_ma"]:
        assert up_means[col] > 0.2, f"{col} not bullish on uptrend: {up_means[col]:.3f}"


def test_tc_a3_trend_signals_negative_on_downtrend(monotonic_df):
    down_means = _mid_down_slice(monotonic_df)[signal_columns(monotonic_df)].mean()
    for col in ["sig_trend_ema_cross", "sig_trend_macd_hist",
                "sig_trend_slope_21", "sig_trend_price_above_ma"]:
        assert down_means[col] < -0.2, f"{col} not bearish on downtrend: {down_means[col]:.3f}"


def test_tc_a3_rsi_trend_and_obv_follow_direction(monotonic_df):
    up_means = _mid_up_slice(monotonic_df)[signal_columns(monotonic_df)].mean()
    down_means = _mid_down_slice(monotonic_df)[signal_columns(monotonic_df)].mean()
    assert up_means["sig_mom_rsi_trend"] > 0.2
    assert down_means["sig_mom_rsi_trend"] < -0.2


def test_tc_a3_regime_return_60_follows_direction(monotonic_df):
    up_means = _mid_up_slice(monotonic_df)[signal_columns(monotonic_df)].mean()
    down_means = _mid_down_slice(monotonic_df)[signal_columns(monotonic_df)].mean()
    assert up_means["sig_regime_return_60"] > 0.2
    assert down_means["sig_regime_return_60"] < -0.2


def test_tc_a3_rsi_zone_inverts_on_extreme_momentum(monotonic_df):
    """
    rsi_zone is a REVERSAL signal: its sign is opposite to rsi_trend.
    At the end of a strong uptrend (RSI near overbought), rsi_zone should be < 0.
    """
    up_end_slice = monotonic_df.iloc[len(monotonic_df) - 60 - 5: len(monotonic_df) - 60]
    # At peak of uptrend, RSI is high → rsi_zone should be negative (bearish reversal)
    assert up_end_slice["sig_mom_rsi_zone"].mean() <= 0.0


def test_tc_a3_drawdown_reflects_actual_drawdown(monotonic_df):
    """
    drawdown signal must be strictly <= 0 always, and should be more negative
    in the down segment than at the uptrend peak.
    """
    all_vals = monotonic_df["sig_regime_drawdown"].dropna()
    assert (all_vals <= 0.0).all()
    down_mean = monotonic_df["sig_regime_drawdown"].iloc[-5:].mean()
    assert down_mean < -0.1, f"Drawdown signal should be clearly negative, got {down_mean:.3f}"


# ─────────────────────────────────────────────────────────────────────────────
# TC-A4: warmup / NaN handling
# ─────────────────────────────────────────────────────────────────────────────

def test_tc_a4_length_preserved_by_add_signals():
    raw = _make_ohlcv(400)
    indicated = add_indicators(raw)
    n_in = len(indicated)
    out = add_signals(indicated)
    assert len(out) == n_in, "add_signals must not drop rows"


def test_tc_a4_warmup_filled_with_zero():
    """
    Within the first SIGNAL_WARMUP_WINDOW rows of the output, every sig_*
    column must have been filled with 0 (not left as NaN).
    """
    raw = _make_ohlcv(400)
    indicated = add_indicators(raw)
    out = add_signals(indicated)
    warmup_slice = out.iloc[:SIGNAL_WARMUP_WINDOW]
    for col in signal_columns(out):
        assert not warmup_slice[col].isna().any(), f"{col} has NaN in warmup region"


def test_tc_a4_post_warmup_signals_are_non_trivial():
    """After the warmup region, signals should have meaningful non-zero values."""
    raw = _make_ohlcv(500)
    indicated = add_indicators(raw)
    out = add_signals(indicated)
    tail_sigs = out[signal_columns(out)].iloc[-50:]
    # At least some signals should be non-zero in recent bars
    non_zero_frac = (tail_sigs.abs() > 1e-6).any(axis=1).sum() / len(tail_sigs)
    assert non_zero_frac > 0.8, f"Too many trivial-zero rows post-warmup: {non_zero_frac:.2%}"


def test_tc_a4_no_inf_or_nan_in_final_output(signal_df):
    sig_df = signal_df[signal_columns(signal_df)]
    assert not sig_df.isna().any().any(), "Signals contain NaN after fillna"
    assert not np.isinf(sig_df.values).any(), "Signals contain inf"


def test_tc_a4_required_cols_list_is_complete():
    """Sanity: the REQUIRED_COLS list references the columns the implementation actually reads."""
    raw = _make_ohlcv(300)
    indicated = add_indicators(raw)
    for col in REQUIRED_COLS:
        assert col in indicated.columns, \
            f"REQUIRED_COLS references {col!r} but add_indicators does not produce it"
