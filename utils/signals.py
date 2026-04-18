"""
Signal Extractor layer — converts raw OHLCV + technical indicators into
bounded, directional signals for the Combiner layer.

Design contract (see `docs/GinkgoBrain/信号分层架构与可解释RL设计.md` §4):
- All sig_* columns output in [-1, 1]
- Positive = bullish, Negative = bearish, 0 = neutral
  (exception: volatility group expresses state, not direction — see §4.2.3)
- Warmup / NaN filled with 0 (neutral)
- Rolling z-score (znorm) used for all quantities without fixed value ranges,
  to survive volatility regime shifts in crypto markets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_COLS = [
    # OHLCV
    "open", "high", "low", "close", "volume",
    # Indicators produced by utils/indicators.py
    "ema_9", "ema_21",
    "macd", "macd_signal",
    "rsi",
    "stoch_k", "stoch_d",
    "bb_upper", "bb_lower",
    "atr",
    "obv",
]

SIGNAL_WARMUP_WINDOW = 100  # must match §5.5.2 in the design doc


def znorm(s: pd.Series, window: int = 100, cap: float = 3.0) -> pd.Series:
    """
    Rolling z-score normalization, output clipped to [-1, 1].

    Chosen over fixed-constant tanh because crypto markets have strong
    volatility regime shifts — a constant scale that discriminates this year
    may compress signals into [-0.1, 0.1] next year.
    """
    mean = s.rolling(window).mean()
    std = s.rolling(window).std() + 1e-8
    z = (s - mean) / std
    return z.clip(-cap, cap) / cap


def _rsi_zone(rsi: pd.Series) -> pd.Series:
    """Piecewise mapping of RSI to a reversal signal (sign opposite to trend)."""
    conditions = [
        rsi < 30,
        (rsi >= 30) & (rsi < 45),
        (rsi >= 45) & (rsi <= 55),
        (rsi > 55) & (rsi <= 70),
        rsi > 70,
    ]
    choices = [0.8, 0.3, 0.0, -0.3, -0.8]
    out = np.select(conditions, choices, default=np.nan)
    return pd.Series(out, index=rsi.index, dtype=float)


def _stoch_cross(stoch_k: pd.Series, stoch_d: pd.Series) -> pd.Series:
    """+1 on upward K-over-D cross, -1 on downward, else 0."""
    k_gt_d = stoch_k > stoch_d
    k_lt_d = stoch_k < stoch_d
    prev_k_gt_d = k_gt_d.shift(1, fill_value=False)
    prev_k_lt_d = k_lt_d.shift(1, fill_value=False)
    crossed_up = k_gt_d & prev_k_lt_d
    crossed_down = k_lt_d & prev_k_gt_d
    sig = pd.Series(0.0, index=stoch_k.index, dtype=float)
    sig[crossed_up] = 1.0
    sig[crossed_down] = -1.0
    return sig


def add_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append all sig_* columns to a DataFrame that already contains OHLCV +
    indicators from utils.indicators.add_indicators.

    Output preserves input length; warmup rows have sig_* == 0 (neutral).
    """
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"add_signals requires indicators to be computed first. "
            f"Missing columns: {missing}"
        )

    df = df.copy()
    close = df["close"]

    # ─── Trend group ──────────────────────────────────────────────────
    df["sig_trend_ema_cross"] = znorm((df["ema_9"] - df["ema_21"]) / df["ema_21"])
    df["sig_trend_macd_hist"] = znorm((df["macd"] - df["macd_signal"]) / close)
    # ATR-normalized distance from MA: "how many ATRs above/below ema_21".
    # Continuous (not a regime flag) so it carries magnitude info distinct from
    # sig_trend_ema_cross (which is EMA-vs-EMA, not price-vs-EMA).
    df["sig_trend_price_above_ma"] = ((close - df["ema_21"]) / df["atr"]).clip(-3, 3) / 3
    df["sig_trend_slope_21"] = znorm(df["ema_21"].pct_change(5))

    # ─── Momentum group ───────────────────────────────────────────────
    df["sig_mom_rsi_zone"] = _rsi_zone(df["rsi"])
    df["sig_mom_rsi_trend"] = (df["rsi"] - 50.0) / 50.0
    df["sig_mom_stoch_state"] = _stoch_cross(df["stoch_k"], df["stoch_d"])
    df["sig_mom_roc_zscore"] = znorm(close.pct_change(5))

    # ─── Volatility group ─────────────────────────────────────────────
    # NOTE: volatility signals express state (high/low volatility, position
    # within BB), not direction. Their sign is NOT bullish/bearish.
    atr_pct = df["atr"] / close
    df["sig_vol_atr_pct"] = atr_pct.rolling(100).rank(pct=True) * 2.0 - 1.0

    bb_width = (df["bb_upper"] - df["bb_lower"]) / close
    df["sig_vol_bb_width"] = bb_width.rolling(100).rank(pct=True) * 2.0 - 1.0

    bb_range = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
    df["sig_vol_bb_position"] = (2.0 * (close - df["bb_lower"]) / bb_range - 1.0).clip(-1, 1)

    # ─── Volume group ─────────────────────────────────────────────────
    df["sig_volume_obv_slope"] = znorm(df["obv"].pct_change(5))

    vol_mean_20 = df["volume"].rolling(20).mean()
    df["sig_volume_ratio"] = znorm(df["volume"] / vol_mean_20)

    # ─── Regime group ─────────────────────────────────────────────────
    rolling_max = close.rolling(100).max()
    df["sig_regime_drawdown"] = (close / rolling_max - 1.0).clip(-1.0, 0.0)
    df["sig_regime_return_60"] = znorm(close.pct_change(60))

    # ─── Final: fill warmup NaN with 0 (neutral) ──────────────────────
    sig_cols = [c for c in df.columns if c.startswith("sig_")]
    df[sig_cols] = df[sig_cols].fillna(0.0)

    return df


def signal_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of sig_* columns present in the DataFrame, in order."""
    return [c for c in df.columns if c.startswith("sig_")]
