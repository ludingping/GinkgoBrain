"""OHLCV indicators for trend, momentum, volatility, and regime research."""
from __future__ import annotations

import numpy as np
import pandas as pd
import ta

SMA_WINDOWS = (20, 50, 200, 250)
DONCHIAN_WINDOW = 20
MACD_WINDOWS = (12, 26, 9)
RSI_WINDOW = 14
KDJ_WINDOW = 9
ROC_WINDOW = 20
CCI_WINDOW = 20
ADX_WINDOW = 14
ATR_WINDOW = 14
BOLLINGER_WINDOW = 20
BOLLINGER_STD = 2.0
SQUEEZE_WINDOW = 126
VOLUME_WINDOW = 20
MFI_WINDOW = 14
CHOP_WINDOW = 14
ER_WINDOW = 10
VHF_WINDOW = 20
REGRESSION_WINDOW = 20

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


def efficiency_ratio(close: pd.Series, window: int = ER_WINDOW) -> pd.Series:
    displacement = (close - close.shift(window)).abs()
    path = close.diff().abs().rolling(window).sum()
    return (displacement / path.replace(0.0, np.nan)).rename(f"er{window}")


def choppiness_index(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = CHOP_WINDOW,
) -> pd.Series:
    previous_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    path = true_range.rolling(window).sum()
    price_range = high.rolling(window).max() - low.rolling(window).min()
    ratio = path / price_range.replace(0.0, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        chop = 100.0 * np.log10(ratio) / np.log10(window)
    return chop.rename(f"chop{window}")


def vertical_horizontal_filter(close: pd.Series, window: int = VHF_WINDOW) -> pd.Series:
    price_range = close.rolling(window).max() - close.rolling(window).min()
    path = close.diff().abs().rolling(window - 1).sum()
    return (price_range / path.replace(0.0, np.nan)).rename(f"vhf{window}")


def _regression_values(log_close: pd.Series, window: int) -> pd.DataFrame:
    x = np.arange(window, dtype=float)
    centered_x = x - x.mean()
    denominator = np.sum(centered_x**2)

    def slope(y: np.ndarray) -> float:
        return float(np.sum(centered_x * (y - y.mean())) / denominator)

    def r_squared(y: np.ndarray) -> float:
        centered_y = y - y.mean()
        total = np.sum(centered_y**2)
        if total <= 0.0:
            return np.nan
        beta = np.sum(centered_x * centered_y) / denominator
        residual = np.sum((centered_y - beta * centered_x) ** 2)
        return float(1.0 - residual / total)

    slopes = log_close.rolling(window).apply(slope, raw=True)
    r2 = log_close.rolling(window).apply(r_squared, raw=True)
    return pd.DataFrame({f"reg_slope{window}": slopes, f"reg_r2_{window}": r2})


def rolling_log_regression(close: pd.Series, window: int = REGRESSION_WINDOW) -> pd.DataFrame:
    if (close.dropna() <= 0).any():
        raise ValueError("log regression requires positive close prices")
    return _regression_values(np.log(close), window)


def add_regime_indicators(df: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"missing OHLCV columns: {missing}")
    if (df["close"].dropna() <= 0).any():
        raise ValueError("regime indicators require positive close prices")

    out = df.copy()
    close = out["close"]
    high = out["high"]
    low = out["low"]
    volume = out["volume"]

    for window in SMA_WINDOWS:
        out[f"sma{window}"] = close.rolling(window).mean()
    out["sma200_slope20"] = out["sma200"] / out["sma200"].shift(20) - 1.0

    fast, slow, signal = MACD_WINDOWS
    macd = ta.trend.MACD(close, window_fast=fast, window_slow=slow, window_sign=signal)
    out["dif"] = macd.macd()
    out["dea"] = macd.macd_signal()
    out["macd_hist"] = 2.0 * macd.macd_diff()

    out[f"rsi{RSI_WINDOW}"] = ta.momentum.RSIIndicator(close, window=RSI_WINDOW).rsi()
    stoch = ta.momentum.StochasticOscillator(high, low, close, window=KDJ_WINDOW)
    out["kdj_k"] = stoch.stoch()
    out["kdj_d"] = stoch.stoch_signal()
    out["kdj_j"] = 3.0 * out["kdj_k"] - 2.0 * out["kdj_d"]
    out[f"roc{ROC_WINDOW}"] = close.pct_change(ROC_WINDOW) * 100.0
    out[f"cci{CCI_WINDOW}"] = ta.trend.CCIIndicator(high, low, close, window=CCI_WINDOW).cci()

    adx = ta.trend.ADXIndicator(high, low, close, window=ADX_WINDOW)
    out["plus_di"] = adx.adx_pos()
    out["minus_di"] = adx.adx_neg()
    out[f"adx{ADX_WINDOW}"] = adx.adx()

    atr = ta.volatility.AverageTrueRange(high, low, close, window=ATR_WINDOW)
    out[f"atr{ATR_WINDOW}"] = atr.average_true_range()
    out[f"natr{ATR_WINDOW}"] = out[f"atr{ATR_WINDOW}"] / close * 100.0

    bollinger = ta.volatility.BollingerBands(
        close, window=BOLLINGER_WINDOW, window_dev=BOLLINGER_STD
    )
    out["bb_middle"] = bollinger.bollinger_mavg()
    out["bb_upper"] = bollinger.bollinger_hband()
    out["bb_lower"] = bollinger.bollinger_lband()
    out["bb_width"] = (out["bb_upper"] - out["bb_lower"]) / out["bb_middle"].replace(0.0, np.nan)
    band_range = (out["bb_upper"] - out["bb_lower"]).replace(0.0, np.nan)
    out["bb_pct_b"] = (close - out["bb_lower"]) / band_range
    out[f"bb_width_rank{SQUEEZE_WINDOW}"] = out["bb_width"].rolling(SQUEEZE_WINDOW).rank(pct=True)

    out["obv"] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    out[f"mfi{MFI_WINDOW}"] = ta.volume.MFIIndicator(
        high, low, close, volume, window=MFI_WINDOW
    ).money_flow_index()
    out[f"volume_ratio{VOLUME_WINDOW}"] = volume / volume.rolling(VOLUME_WINDOW).mean().replace(0.0, np.nan)

    out[f"donchian_upper{DONCHIAN_WINDOW}"] = high.rolling(DONCHIAN_WINDOW).max()
    out[f"donchian_lower{DONCHIAN_WINDOW}"] = low.rolling(DONCHIAN_WINDOW).min()
    out[f"donchian_middle{DONCHIAN_WINDOW}"] = (
        out[f"donchian_upper{DONCHIAN_WINDOW}"] + out[f"donchian_lower{DONCHIAN_WINDOW}"]
    ) / 2.0

    out = out.join(choppiness_index(high, low, close, CHOP_WINDOW))
    out = out.join(efficiency_ratio(close, ER_WINDOW))
    out = out.join(vertical_horizontal_filter(close, VHF_WINDOW))
    out = out.join(rolling_log_regression(close, REGRESSION_WINDOW))
    return out
