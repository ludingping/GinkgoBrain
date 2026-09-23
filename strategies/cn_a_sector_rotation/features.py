"""§3.1 单行业趋势状态、§3.2 相对强度。纯函数，宽表（日期 × 代码）进出，历史不足处为 NA。"""
from __future__ import annotations

import pandas as pd


def sma(close: pd.DataFrame, n: int) -> pd.DataFrame:
    """简单均线，窗口不足 n 个有效值 → NaN。"""
    return close.rolling(n, min_periods=n).mean()


def in_trend(close: pd.DataFrame, n: int = 120, slope_lookback: int = 20) -> pd.DataFrame:
    """§3.1：close > SMA(n) 且 SMA(n) > SMA(n).shift(slope_lookback)。返回可空布尔（历史不足 → <NA>）。"""
    m = sma(close, n)
    above = close > m
    rising = m > m.shift(slope_lookback)
    valid = close.notna() & m.notna() & m.shift(slope_lookback).notna()
    return (above & rising).astype("boolean").where(valid)


def strength(close: pd.DataFrame, L: int = 60) -> pd.DataFrame:
    """§3.2：过去 L 个交易日的收益率 close / close.shift(L) − 1。"""
    return close / close.shift(L) - 1.0


def below_sma(close: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """§3.4 exit_fast：close < SMA(n)。历史不足 → <NA>。"""
    m = sma(close, n)
    return (close < m).astype("boolean").where(close.notna() & m.notna())


__all__ = ["sma", "in_trend", "strength", "below_sma"]
