"""Date-based train / val / test splitting for time-series DataFrames.

Why not `split_df(ratio)`: a ratio split moves the validation boundary every
time `end_date` is extended, which silently changes what "OOS" means between
runs. Pinning `val_start` / `test_start` keeps the comparison stable and lets a
held-out test slice stay invisible to the EvalCallback that picks best_model.

Every slice is preceded by `prefix_rows` of history so an env that skips
`warmup_rows` bars before its first decision lands exactly on the requested
start date. Signals must already be computed on the full frame (they are
causal — rolling / ewm / shift — so this is not leakage).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DateSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame | None


def _to_ts(value: str | pd.Timestamp | None, like: pd.Series) -> pd.Timestamp | None:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    tz = like.dt.tz
    if tz is not None and ts.tz is None:
        ts = ts.tz_localize(tz)
    elif tz is not None:
        ts = ts.tz_convert(tz)
    return ts


def slice_by_dates(
    df: pd.DataFrame,
    start: str | pd.Timestamp | None,
    end: str | pd.Timestamp | None = None,
    *,
    prefix_rows: int = 0,
    ts_col: str = "timestamp",
) -> pd.DataFrame:
    """Rows with start <= ts < end, plus up to `prefix_rows` rows before start."""
    if ts_col not in df.columns:
        raise ValueError(f"slice_by_dates: DataFrame has no `{ts_col}` column")
    ts = df[ts_col]
    start_ts = _to_ts(start, ts)
    end_ts = _to_ts(end, ts)
    i0 = int(ts.searchsorted(start_ts, side="left")) if start_ts is not None else 0
    i1 = int(ts.searchsorted(end_ts, side="left")) if end_ts is not None else len(df)
    lo = i0 - int(prefix_rows)
    if start_ts is not None and lo < 0:
        # Silently clamping would shift the first decision past `start`.
        raise ValueError(
            f"slice_by_dates: only {i0} rows precede {start_ts}, need prefix_rows={prefix_rows}; "
            "load data from an earlier start_date"
        )
    return df.iloc[max(0, lo):i1].reset_index(drop=True)


def split_by_dates(
    df: pd.DataFrame,
    val_start: str | pd.Timestamp,
    test_start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    *,
    prefix_rows: int = 0,
    ts_col: str = "timestamp",
) -> DateSplit:
    """train = [.., val_start), val = [val_start, test_start), test = [test_start, end).

    val / test carry `prefix_rows` of history; train never sees rows at or
    after `val_start`.
    """
    ts = df[ts_col]
    val_ts = _to_ts(val_start, ts)
    test_ts = _to_ts(test_start, ts)
    if test_ts is not None and test_ts <= val_ts:
        raise ValueError(f"test_start ({test_ts}) must be after val_start ({val_ts})")

    train = df[ts < val_ts].reset_index(drop=True)
    val = slice_by_dates(df, val_ts, test_ts if test_ts is not None else end,
                         prefix_rows=prefix_rows, ts_col=ts_col)
    test = None
    if test_ts is not None:
        test = slice_by_dates(df, test_ts, end, prefix_rows=prefix_rows, ts_col=ts_col)
    if len(train) == 0 or len(val) <= prefix_rows:
        raise ValueError(
            f"split_by_dates produced an empty slice: train={len(train)}, val={len(val)}"
        )
    return DateSplit(train=train, val=val, test=test)
