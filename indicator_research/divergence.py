"""Causal swing-point divergence detection for momentum and volume charts."""
import numpy as np
import pandas as pd

MOMENTUM_INDICATORS = {
    "rsi": "rsi14",
    "kdj": "kdj_k",
    "roc": "roc20",
    "cci": "cci20",
}
VOLUME_INDICATORS = {
    "obv": "obv",
    "mfi": "mfi14",
}


def _swing_points(values: np.ndarray, left: int, right: int, kind: str) -> list[int]:
    if kind not in {"low", "high"}:
        raise ValueError("kind must be 'low' or 'high'")
    points = []
    for center in range(left, len(values) - right):
        window = values[center - left:center + right + 1]
        if np.isnan(window).any():
            continue
        current = window[left]
        neighbors = np.delete(window, left)
        if kind == "low" and np.all(current < neighbors):
            points.append(center)
        if kind == "high" and np.all(current > neighbors):
            points.append(center)
    return points


def _pivot_pairs(points: list[int], min_gap: int, max_gap: int):
    for position, current in enumerate(points):
        for previous in reversed(points[:position]):
            gap = current - previous
            if gap > max_gap:
                break
            if gap >= min_gap:
                yield previous, current
                break


def _confirmed_divergences(
    price: np.ndarray,
    indicator: np.ndarray,
    pivots: list[int],
    right: int,
    min_gap: int,
    max_gap: int,
    kind: str,
) -> np.ndarray:
    signals = np.zeros(len(price), dtype=bool)
    for previous, current in _pivot_pairs(pivots, min_gap, max_gap):
        confirmation = current + right
        values = np.array([
            price[previous],
            price[current],
            indicator[previous],
            indicator[current],
            indicator[confirmation],
        ])
        if not np.isfinite(values).all():
            continue
        if kind == "low":
            matched = price[current] < price[previous] and indicator[current] > indicator[previous]
        else:
            matched = price[current] > price[previous] and indicator[current] < indicator[previous]
        if matched:
            signals[confirmation] = True
    return signals


def _add_divergences(
    data: pd.DataFrame,
    indicators: dict[str, str],
    left: int,
    right: int,
    min_gap: int,
    max_gap: int,
) -> pd.DataFrame:
    if left < 1 or right < 1:
        raise ValueError("swing windows must be positive")
    if min_gap <= right:
        raise ValueError("min_gap must exceed the right confirmation window")
    if max_gap < min_gap:
        raise ValueError("max_gap must be at least min_gap")

    out = data.copy()
    lows = out["low"].to_numpy(dtype=float)
    highs = out["high"].to_numpy(dtype=float)
    low_pivots = _swing_points(lows, left, right, "low")
    high_pivots = _swing_points(highs, left, right, "high")

    for prefix, column in indicators.items():
        indicator = out[column].to_numpy(dtype=float)
        out[f"{prefix}_bullish_divergence"] = _confirmed_divergences(
            lows, indicator, low_pivots, right, min_gap, max_gap, "low"
        )
        out[f"{prefix}_bearish_divergence"] = _confirmed_divergences(
            highs, indicator, high_pivots, right, min_gap, max_gap, "high"
        )
    return out


def add_momentum_divergences(
    data: pd.DataFrame,
    left: int = 2,
    right: int = 2,
    min_gap: int = 5,
    max_gap: int = 60,
) -> pd.DataFrame:
    return _add_divergences(data, MOMENTUM_INDICATORS, left, right, min_gap, max_gap)


def add_volume_divergences(
    data: pd.DataFrame,
    left: int = 2,
    right: int = 2,
    min_gap: int = 5,
    max_gap: int = 60,
) -> pd.DataFrame:
    return _add_divergences(data, VOLUME_INDICATORS, left, right, min_gap, max_gap)
