import numpy as np
import pandas as pd

from indicator_research.direction import (
    add_donchian_states,
    add_momentum_range_states,
    direction_runs,
)


def test_donchian_direction_persists_until_opposite_breakout():
    frame = pd.DataFrame(
        {
            "close": [90.0, 101.0, 102.0, 99.0, 95.0, 79.0, 85.0, 103.0],
            "donchian_upper20": [100.0] * 8,
            "donchian_lower20": [80.0] * 8,
        },
        index=pd.date_range("2026-01-01", periods=8, freq="B"),
    )

    result = add_donchian_states(frame)

    assert result["donchian_direction"].fillna("warmup").tolist() == [
        "warmup", "bull", "bull", "bull", "bull", "bear", "bear", "bull",
    ]
    assert result["donchian_signal"].fillna("none").tolist() == [
        "none", "bull", "none", "none", "none", "bear", "none", "bull",
    ]
    pd.testing.assert_frame_equal(result.iloc[:6], add_donchian_states(frame.iloc[:6]))


def test_momentum_range_requires_all_non_extreme_conditions():
    frame = pd.DataFrame(
        {
            "rsi14": [np.nan, 30.0, 71.0, 50.0, 50.0, 50.0, 50.0],
            "kdj_k": [np.nan, 20.0, 50.0, 81.0, 50.0, 50.0, 50.0],
            "kdj_d": [np.nan, 80.0, 50.0, 50.0, 85.0, 50.0, 50.0],
            "cci20": [np.nan, -100.0, 0.0, 0.0, 0.0, 101.0, 0.0],
        },
        index=pd.date_range("2026-01-01", periods=7, freq="B"),
    )

    result = add_momentum_range_states(frame)

    assert result["momentum_range"].fillna("warmup").tolist() == [
        "warmup", "range", "other", "other", "other", "other", "range",
    ]
    assert direction_runs(result["momentum_range"]) == [
        ("warmup", 0, 1),
        ("range", 1, 2),
        ("other", 2, 6),
        ("range", 6, 7),
    ]
