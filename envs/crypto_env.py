"""Crypto trading environment (same structure as stock, different data source)."""
import pandas as pd
from .stock_env import StockTradingEnv


class CryptoTradingEnv(StockTradingEnv):
    """
    Gymnasium environment for crypto trading.

    Identical to StockTradingEnv but accepts funding rates and other
    crypto-specific columns gracefully via the base OHLCV handling.
    Swap out commission default to reflect typical CEX taker fee.
    """

    def __init__(self, df: pd.DataFrame, commission: float = 0.0005, **kwargs):
        super().__init__(df, commission=commission, **kwargs)
