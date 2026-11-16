import numpy as np
import pandas as pd

from indicator_research.volatility import add_volatility_regimes


def make_volatility_frame():
    return pd.DataFrame(
        {
            "bb_pct_b": [np.nan, 0.5, 1.2, 0.5, -0.2, 0.5, 1.1, 0.5],
            "bb_width": [1.0, 1.0, 1.1, 1.2, 1.3, 1.3, 1.4, 1.3],
            "bb_width_rank126": [np.nan, 0.2, 0.2, 0.05, 0.2, 0.05, 0.2, 0.2],
        },
        index=pd.date_range("2026-01-01", periods=8, freq="B"),
    )


def test_volatility_states_and_start_end_events():
    result = add_volatility_regimes(make_volatility_frame())

    assert result["volatility_regime"].fillna("warmup").tolist() == [
        "warmup", "neutral", "trend", "range", "trend", "range", "trend", "neutral",
    ]
    assert np.flatnonzero(result["volatility_trend_start"]).tolist() == [2, 4, 6]
    assert np.flatnonzero(result["volatility_trend_end"]).tolist() == [3, 5, 7]
    assert np.flatnonzero(result["volatility_range_start"]).tolist() == [3, 5]
    assert np.flatnonzero(result["volatility_range_end"]).tolist() == [4, 6]


def test_volatility_events_are_prefix_invariant():
    frame = make_volatility_frame()

    full = add_volatility_regimes(frame)
    prefix = add_volatility_regimes(frame.iloc[:7])

    pd.testing.assert_frame_equal(full.iloc[:7], prefix)
