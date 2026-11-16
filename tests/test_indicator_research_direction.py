import numpy as np
import pandas as pd

from indicator_research.direction import add_direction_states, add_volume_direction_states, direction_runs


def make_direction_frame():
    return pd.DataFrame(
        {
            "close": [100.0, 110.0, 110.0, 90.0, 90.0, 100.0, 100.0, 100.0],
            "sma200": [np.nan, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            "sma200_slope20": [np.nan, 0.01, -0.01, -0.01, -0.01, 0.01, 0.01, 0.01],
            "dif": [np.nan, 1.0, 0.5, -0.5, -1.0, 1.0, 1.0, 0.8],
            "dea": [np.nan, 0.0, 0.6, 0.0, -0.5, 0.5, 0.8, 0.8],
        },
        index=pd.date_range("2026-01-01", periods=8, freq="B"),
    )


def test_direction_states_and_entry_signals():
    frame = make_direction_frame()

    result = add_direction_states(frame)

    assert "long_direction" not in frame.columns
    assert result["long_direction"].fillna("warmup").tolist() == [
        "warmup", "bull", "neutral", "bear", "bear", "bull", "bull", "bull",
    ]
    assert result["long_signal"].fillna("none").tolist() == [
        "none", "bull", "none", "bear", "none", "bull", "none", "none",
    ]
    assert result["macd_direction"].fillna("warmup").tolist() == [
        "warmup", "bull", "bear", "bear", "bear", "bull", "bull", "neutral",
    ]
    assert result["macd_signal"].fillna("none").tolist() == [
        "none", "bull", "bear", "none", "none", "bull", "none", "none",
    ]


def test_direction_runs_cover_every_row_with_exclusive_ends():
    result = add_direction_states(make_direction_frame())

    assert direction_runs(result["long_direction"]) == [
        ("warmup", 0, 1),
        ("bull", 1, 2),
        ("neutral", 2, 3),
        ("bear", 3, 5),
        ("bull", 5, 8),
    ]
    assert direction_runs(result["macd_signal"]) == [
        ("warmup", 0, 1),
        ("bull", 1, 2),
        ("bear", 2, 3),
        ("warmup", 3, 5),
        ("bull", 5, 6),
        ("warmup", 6, 8),
    ]
    assert direction_runs(pd.Series([], dtype="string")) == []


def test_direction_states_are_prefix_invariant():
    frame = make_direction_frame()

    full = add_direction_states(frame)
    prefix = add_direction_states(frame.iloc[:6])

    pd.testing.assert_frame_equal(full.iloc[:6], prefix)


def test_volume_direction_states_entry_points_and_prefix_invariance():
    frame = pd.DataFrame({
        "close": [100.0] * 20 + [110.0, 110.0, 90.0, 90.0, 100.0, 100.0],
        "obv": [0.0] * 20 + [1e6, 1e6, -1e6, -1e6, 1e6, 1e6],
    })

    result = add_volume_direction_states(frame)

    assert "volume_direction" not in frame.columns
    assert result["volume_direction"].fillna("warmup").tolist() == (
        ["warmup"] * 20 + ["bull", "bull", "bear", "bear", "neutral", "neutral"]
    )
    assert result["volume_signal"].fillna("none").tolist() == (
        ["none"] * 20 + ["bull", "none", "bear", "none", "none", "none"]
    )
    prefix = add_volume_direction_states(frame.iloc[:25])
    pd.testing.assert_frame_equal(result.iloc[:25], prefix)
