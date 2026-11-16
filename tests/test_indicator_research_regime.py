import numpy as np
import pandas as pd
import pytest

from indicator_research.regime import (
    add_regime_indicators,
    choppiness_index,
    efficiency_ratio,
    rolling_log_regression,
    vertical_horizontal_filter,
)


def test_efficiency_ratio_known_shapes_and_zero_path():
    index = pd.date_range("2024-01-01", periods=12, freq="D")
    trend = pd.Series(np.arange(12.0), index=index)
    alternating = pd.Series([10.0, 11.0] * 6, index=index)
    constant = pd.Series(10.0, index=index)

    assert efficiency_ratio(trend, 10).iloc[-1] == pytest.approx(1.0)
    assert efficiency_ratio(alternating, 10).iloc[-1] == pytest.approx(0.0)
    assert np.isnan(efficiency_ratio(constant, 10).iloc[-1])
    assert efficiency_ratio(trend, 10).iloc[:10].isna().all()


def test_choppiness_index_matches_definition_and_handles_zero_range():
    close = pd.Series([1.0, 2.0, 3.0, 4.0])
    result = choppiness_index(close, close, close, window=3)
    expected = 100 * np.log10(3.0 / 2.0) / np.log10(3)
    assert result.iloc[-1] == pytest.approx(expected)

    constant = pd.Series(2.0, index=close.index)
    assert choppiness_index(constant, constant, constant, window=3).isna().all()


def test_vhf_known_shapes_and_zero_path():
    trend = pd.Series(np.arange(30.0))
    alternating = pd.Series([10.0, 11.0] * 15)
    constant = pd.Series(10.0, index=trend.index)

    assert vertical_horizontal_filter(trend, 20).iloc[-1] == pytest.approx(1.0)
    assert vertical_horizontal_filter(alternating, 20).iloc[-1] == pytest.approx(1.0 / 19.0)
    assert np.isnan(vertical_horizontal_filter(constant, 20).iloc[-1])


def test_rolling_log_regression_recovers_slope_and_r2():
    x = np.arange(40, dtype=float)
    beta = 0.01
    close = pd.Series(np.exp(2.0 + beta * x))
    result = rolling_log_regression(close, window=20)

    assert result["reg_slope20"].iloc[-1] == pytest.approx(beta)
    assert result["reg_r2_20"].iloc[-1] == pytest.approx(1.0)
    assert result.iloc[:19].isna().all().all()

    constant = rolling_log_regression(pd.Series(np.ones(40)), window=20)
    assert constant["reg_slope20"].iloc[-1] == pytest.approx(0.0)
    assert np.isnan(constant["reg_r2_20"].iloc[-1])


def test_add_regime_indicators_preserves_rows_and_is_prefix_invariant():
    index = pd.date_range("2020-01-01", periods=280, freq="D")
    x = np.arange(len(index), dtype=float)
    close = pd.Series(80.0 + 0.08 * x + 4.0 * np.sin(x / 9.0), index=index)
    frame = pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + 1.2,
            "low": close - 1.1,
            "close": close,
            "volume": 100_000 + 500 * (x % 17),
        },
        index=index,
    )

    full = add_regime_indicators(frame)
    prefix = add_regime_indicators(frame.iloc[:250])

    assert len(full) == len(frame)
    assert full.index.equals(frame.index)
    assert full["sma20"].iloc[:19].isna().all()
    assert full["sma200"].iloc[:199].isna().all()
    for column in full.columns:
        pd.testing.assert_series_equal(
            full[column].iloc[:250], prefix[column], check_names=False, rtol=1e-12, atol=1e-12
        )


def test_add_regime_indicators_validates_input():
    with pytest.raises(ValueError, match="missing OHLCV columns"):
        add_regime_indicators(pd.DataFrame({"close": [1.0]}))
    with pytest.raises(ValueError, match="positive close"):
        add_regime_indicators(
            pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0], "close": [0.0], "volume": [1.0]})
        )
