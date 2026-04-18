"""Stock trading environment."""
import numpy as np
import pandas as pd
from .base_env import BaseTradingEnv


class StockTradingEnv(BaseTradingEnv):
    """
    Gymnasium environment for stock trading using OHLCV + technical indicators.

    Expected df columns: open, high, low, close, volume, [indicator columns...]
    """

    REQUIRED_COLS = ["open", "high", "low", "close", "volume"]

    def __init__(self, df: pd.DataFrame, **kwargs):
        missing = [c for c in self.REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame missing columns: {missing}")
        super().__init__(df, **kwargs)

    def _count_features(self) -> int:
        # OHLCV (5) + any extra indicator columns + position history (2)
        extra = [c for c in self.df.columns if c not in self.REQUIRED_COLS and c != "timestamp"]
        return len(self.REQUIRED_COLS) + len(extra) + 2

    def _get_obs(self) -> np.ndarray:
        frame = self.df.iloc[self.current_step - self.window_size: self.current_step].copy()
        
        last_close = frame["close"].iloc[-1] + 1e-8

        # Group C: OBV to pct_change
        if "obv" in frame.columns:
            frame["obv"] = frame["obv"].pct_change().fillna(0).clip(-1, 1)

        # Group A: Price related columns
        PRICE_COLS = [
            "open", "high", "low", "close",
            "ema_9", "ema_21", "bb_upper", "bb_lower",
            "macd", "macd_signal", "atr",
            "m5_ema_9", "m5_ema_21", "m5_macd", "m5_macd_signal",
            "m15_ema_9",              "m15_macd",
        ]
        for col in PRICE_COLS:
            if col in frame.columns:
                frame[col] = frame[col] / last_close

        # Normalize volume
        vol_max = frame["volume"].max() + 1e-8
        frame["volume"] = frame["volume"] / vol_max

        # Group B: 0-100 to 0-1
        for col in ["rsi", "stoch_k", "stoch_d", "m5_rsi", "m15_rsi"]:
            if col in frame.columns:
                frame[col] = frame[col] / 100.0

        obs_frame = frame.drop(columns=["timestamp"], errors="ignore")
        obs = obs_frame.values.astype(np.float32)

        # Append position history buffer
        obs = np.concatenate([obs, self._pos_history], axis=1)
        return obs
