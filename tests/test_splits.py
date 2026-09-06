"""Tests for utils/splits.py — date-pinned train/val/test with warmup prefix."""
from __future__ import annotations

import pandas as pd
import pytest

from utils.splits import slice_by_dates, split_by_dates


def make_hourly(n: int = 1000, tz: str = "Asia/Shanghai") -> pd.DataFrame:
    ts = pd.date_range("2025-01-01", periods=n, freq="1h", tz=tz)
    return pd.DataFrame({"timestamp": ts, "close": range(n)})


def test_slice_by_dates_prefix_lands_first_decision_on_start():
    df = make_hourly()
    start = "2025-01-10 00:00:00"          # index 216
    out = slice_by_dates(df, start, prefix_rows=206)
    assert len(out) == len(df) - 216 + 206
    assert out["timestamp"].iloc[206] == pd.Timestamp(start, tz="Asia/Shanghai")


def test_slice_by_dates_rejects_insufficient_prefix():
    """Clamping at the head would move the first decision past `start`."""
    df = make_hourly()
    with pytest.raises(ValueError, match="prefix_rows"):
        slice_by_dates(df, "2025-01-01 05:00:00", prefix_rows=206)
    # exactly enough history is fine
    out = slice_by_dates(df, "2025-01-01 05:00:00", prefix_rows=5)
    assert out["timestamp"].iloc[0] == df["timestamp"].iloc[0]


def test_slice_end_is_exclusive():
    df = make_hourly()
    out = slice_by_dates(df, "2025-01-02", "2025-01-03")
    assert len(out) == 24
    assert out["timestamp"].iloc[-1] == pd.Timestamp("2025-01-02 23:00", tz="Asia/Shanghai")


def test_split_by_dates_train_never_sees_val_and_test_is_held_out():
    df = make_hourly()
    val_start, test_start = "2025-01-20", "2025-02-01"
    sp = split_by_dates(df, val_start, test_start, prefix_rows=10)

    v0 = pd.Timestamp(val_start, tz="Asia/Shanghai")
    t0 = pd.Timestamp(test_start, tz="Asia/Shanghai")
    assert sp.train["timestamp"].max() < v0
    assert sp.val["timestamp"].iloc[10] == v0
    assert sp.val["timestamp"].max() < t0
    assert sp.test is not None
    assert sp.test["timestamp"].iloc[10] == t0
    # no test row (beyond the prefix) appears in val
    assert sp.val["timestamp"].max() < sp.test["timestamp"].iloc[10]


def test_split_by_dates_without_test():
    df = make_hourly()
    sp = split_by_dates(df, "2025-01-20", prefix_rows=0)
    assert sp.test is None
    assert len(sp.train) + len(sp.val) == len(df)


def test_split_by_dates_rejects_test_before_val():
    df = make_hourly()
    with pytest.raises(ValueError):
        split_by_dates(df, "2025-02-01", "2025-01-20")


def test_split_by_dates_accepts_tz_aware_inputs():
    df = make_hourly()
    v0 = pd.Timestamp("2025-01-20", tz="UTC")
    sp = split_by_dates(df, v0, prefix_rows=0)
    assert sp.val["timestamp"].iloc[0] == v0.tz_convert("Asia/Shanghai")


# ─── time_folds ──────────────────────────────────────────────────────────────

def _ts_index(n: int, freq: str = "4h") -> pd.Series:
    return pd.Series(pd.date_range("2021-01-01", periods=n, freq=freq, tz="Asia/Shanghai"))


def test_time_folds_expanding_window_covers_tail_only_once():
    from utils.splits import time_folds
    ts = _ts_index(400)
    folds = time_folds(ts, n_splits=4)
    assert len(folds) == 4
    # expanding: each fold's val starts where the previous ended
    for prev, cur in zip(folds, folds[1:]):
        assert prev.val_end == cur.val_start
    assert folds[-1].val_end > ts.iloc[-1]
    # val slices are disjoint and cover the last 4/5 of the range
    assert folds[0].val_start == ts.iloc[80]


def test_time_folds_embargo_gap_before_val_start():
    from utils.splits import time_folds
    ts = _ts_index(400)
    folds = time_folds(ts, n_splits=4, embargo=pd.Timedelta("24h"))
    for f in folds:
        assert f.train_end == f.val_start - pd.Timedelta("24h")


def test_time_folds_masks_exclude_embargo_rows_and_are_disjoint():
    from utils.splits import time_folds
    ts = _ts_index(400)
    f = time_folds(ts, n_splits=4, embargo=pd.Timedelta("24h"))[0]
    tr, va = f.masks(ts)
    assert not (tr & va).any()
    assert ts[tr].max() < f.val_start - pd.Timedelta("24h")
    assert ts[va].min() == f.val_start
    # 6 bars of 4h sit inside the 24h embargo
    assert tr.sum() == 80 - 6


def test_time_folds_rejects_duplicate_or_unsorted_ok_but_too_few_points():
    from utils.splits import time_folds
    with pytest.raises(ValueError):
        time_folds(_ts_index(3), n_splits=4)
