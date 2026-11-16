import pandas as pd

from indicator_research.divergence import (
    MOMENTUM_INDICATORS,
    VOLUME_INDICATORS,
    add_momentum_divergences,
    add_volume_divergences,
)

SIGNAL_COLUMNS = [
    f"{prefix}_{direction}_divergence"
    for prefix in MOMENTUM_INDICATORS
    for direction in ("bullish", "bearish")
]


def make_divergence_frame():
    frame = pd.DataFrame(
        {
            "low": [12.0, 11.0, 10.0, 11.0, 12.0, 13.0, 12.0, 11.0, 10.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
            "high": [13.0, 14.0, 15.0, 14.0, 13.0, 12.0, 13.0, 14.0, 15.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0],
        }
    )
    for column in MOMENTUM_INDICATORS.values():
        frame[column] = 50.0
        frame.loc[[2, 9], column] = 20.0
    return frame


def test_detects_bullish_and_bearish_divergences_at_confirmation():
    frame = make_divergence_frame()
    frame.loc[9, ["rsi14", "roc20"]] = 25.0
    frame.loc[9, ["kdj_k", "cci20"]] = 15.0

    result = add_momentum_divergences(frame)

    for prefix in ("rsi", "roc"):
        assert result[f"{prefix}_bullish_divergence"].sum() == 1
        assert result[f"{prefix}_bullish_divergence"].iloc[11]
        assert not result[f"{prefix}_bearish_divergence"].any()
    for prefix in ("kdj", "cci"):
        assert result[f"{prefix}_bearish_divergence"].sum() == 1
        assert result[f"{prefix}_bearish_divergence"].iloc[11]
        assert not result[f"{prefix}_bullish_divergence"].any()


def test_equal_indicator_extremes_do_not_create_divergence():
    result = add_momentum_divergences(make_divergence_frame())

    assert not result[SIGNAL_COLUMNS].any().any()


def test_signals_do_not_appear_before_confirmation_and_are_prefix_invariant():
    frame = make_divergence_frame()
    frame.loc[9, ["rsi14", "roc20"]] = 25.0
    frame.loc[9, ["kdj_k", "cci20"]] = 15.0

    full = add_momentum_divergences(frame)
    before_confirmation = add_momentum_divergences(frame.iloc[:11])
    confirmed_prefix = add_momentum_divergences(frame.iloc[:12])

    assert not before_confirmation[SIGNAL_COLUMNS].any().any()
    pd.testing.assert_frame_equal(full[SIGNAL_COLUMNS].iloc[:12], confirmed_prefix[SIGNAL_COLUMNS])


def test_volume_divergences_are_confirmed_without_lookahead():
    frame = make_divergence_frame()
    frame["obv"] = 1_000.0
    frame.loc[[2, 9], "obv"] = [900.0, 1_100.0]
    frame["mfi14"] = 50.0
    frame.loc[[2, 9], "mfi14"] = [40.0, 30.0]
    signal_columns = [
        f"{prefix}_{direction}_divergence"
        for prefix in VOLUME_INDICATORS
        for direction in ("bullish", "bearish")
    ]

    full = add_volume_divergences(frame)
    before_confirmation = add_volume_divergences(frame.iloc[:11])
    confirmed_prefix = add_volume_divergences(frame.iloc[:12])

    assert full["obv_bullish_divergence"].sum() == 1
    assert full["obv_bullish_divergence"].iloc[11]
    assert not full["obv_bearish_divergence"].any()
    assert full["mfi_bearish_divergence"].sum() == 1
    assert full["mfi_bearish_divergence"].iloc[11]
    assert not full["mfi_bullish_divergence"].any()
    assert not before_confirmation[signal_columns].any().any()
    pd.testing.assert_frame_equal(full[signal_columns].iloc[:12], confirmed_prefix[signal_columns])
