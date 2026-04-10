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
        # OHLCV (5) + any extra indicator columns + position ratio (1)
        extra = [c for c in self.df.columns if c not in self.REQUIRED_COLS]
        return len(self.REQUIRED_COLS) + len(extra) + 1

    def _get_obs(self) -> np.ndarray:
        frame = self.df.iloc[self.current_step - self.window_size: self.current_step].copy()
        # Normalize price columns relative to last close
        last_close = frame["close"].iloc[-1] + 1e-8
        for col in ["open", "high", "low", "close"]:
            frame[col] = frame[col] / last_close
        # Normalize volume
        vol_max = frame["volume"].max() + 1e-8
        frame["volume"] = frame["volume"] / vol_max

        obs = frame.values.astype(np.float32)

        # Append position ratio column
        price = float(self.df.loc[self.current_step, "close"])
        portfolio = self._portfolio_value(price) + 1e-8
        pos_ratio = np.full((self.window_size, 1), self.position * price / portfolio, dtype=np.float32)
        obs = np.concatenate([obs, pos_ratio], axis=1)
        return obs
