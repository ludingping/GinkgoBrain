"""Technical indicator computation using the `ta` library."""
import pandas as pd
import ta


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append common technical indicators to an OHLCV DataFrame.

    Input columns required: open, high, low, close, volume
    Returns a copy with NaN rows dropped.
    """
    df = df.copy()

    # Trend
    df["ema_9"] = ta.trend.EMAIndicator(df["close"], window=9).ema_indicator()
    df["ema_21"] = ta.trend.EMAIndicator(df["close"], window=21).ema_indicator()
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()

    # Momentum
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    stoch = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"])
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    # Volatility
    bb = ta.volatility.BollingerBands(df["close"])
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"]).average_true_range()

    # Volume
    df["obv"] = ta.volume.OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()

    return df.dropna().reset_index(drop=True)


def add_multi_timeframe_indicators(
    df_1m: pd.DataFrame,
    timeframes: list[str] = ["5min", "15min"],
) -> pd.DataFrame:
    """
    Compute indicators on higher timeframes and merge back to 1m index.
    Uses forward-fill to align higher-TF values to each 1m bar without look-ahead bias.
    """
    df = df_1m.copy()

    # Ensure DatetimeIndex for resampling
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")

    result = df.copy()

    for tf in timeframes:
        # Prefix: "5min" → "m5", "15min" → "m15"
        minutes = int("".join(filter(str.isdigit, tf)))
        prefix = f"m{minutes}"

        # Resample to higher TF (closed on left, label on left = bar open time)
        ohlcv_tf = df.resample(tf, closed="left", label="left").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()

        # Compute indicators on higher-TF candles
        ind = add_indicators(ohlcv_tf)   # reuse existing function

        # Select only the columns we want
        cols_to_keep = {
            "ema_9":        f"{prefix}_ema_9",
            "ema_21":       f"{prefix}_ema_21",
            "rsi":          f"{prefix}_rsi",
            "macd":         f"{prefix}_macd",
            "macd_signal":  f"{prefix}_macd_signal",
        }
        ind = ind[[c for c in cols_to_keep if c in ind.columns]].rename(
            columns=cols_to_keep
        )

        ind = ind.shift(1)
        ind = ind.reindex(result.index, method="ffill")
        result = pd.concat([result, ind], axis=1)

    if "timestamp" in df_1m.columns:
        result = result.reset_index()   # restore timestamp column
    return result.dropna().reset_index(drop=True)
