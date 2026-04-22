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

    # ─── Multi-timeframe context (higher-TF signals broadcast to this TF) ──
    # On 4h/1d data these will approximately duplicate the existing trend/regime
    # signals; signal_elimination's correlation filter handles the dedup.
    if "timestamp" in df.columns and len(df) > 100:
        idx = pd.DatetimeIndex(df["timestamp"])
        df_ts = df.set_index("timestamp")

        ohlc_4h = df_ts["close"].resample("4h", closed="left", label="left").last().dropna()
        ema9_4h = ohlc_4h.ewm(span=9, adjust=False).mean()
        ema21_4h = ohlc_4h.ewm(span=21, adjust=False).mean()
        trend_4h = np.sign(ema9_4h - ema21_4h)
        df["sig_mtf_4h_trend"] = trend_4h.reindex(idx, method="ffill").fillna(0.0).values

        ohlc_1d = df_ts["close"].resample("1d", closed="left", label="left").last().dropna()
        ret_60_1d = ohlc_1d.pct_change(60)
        regime_1d = znorm(ret_60_1d)
        df["sig_mtf_1d_regime"] = regime_1d.reindex(idx, method="ffill").fillna(0.0).values
    else:
        df["sig_mtf_4h_trend"] = 0.0
        df["sig_mtf_1d_regime"] = 0.0

    # ─── Intra-bar microstructure (high-frequency candle shape) ───────
    hl_range = (df["high"] - df["low"]).replace(0, np.nan)
    df["sig_hf_range_position"] = (2.0 * (close - df["low"]) / hl_range - 1.0).clip(-1, 1)
    df["sig_hf_body_ratio"] = ((close - df["open"]) / hl_range).clip(-1, 1)

    vol_mean_24 = df["volume"].rolling(24).mean()
    vol_std_24 = df["volume"].rolling(24).std() + 1e-8
    vol_z_24 = (df["volume"] - vol_mean_24) / vol_std_24
    df["sig_hf_vol_zscore_24"] = vol_z_24.clip(-3, 3) / 3

    # ─── Short-horizon mean reversion (tighter than 60-bar regime) ────
    ema_24 = close.ewm(span=24, adjust=False).mean()
    dev_atr = (close - ema_24) / (df["atr"] + 1e-8)
    df["sig_mr_deviation_atr"] = dev_atr.clip(-3, 3) / 3

    rsi_extreme = pd.Series(0.0, index=df.index, dtype=float)
    rsi_extreme[df["rsi"] < 25] = 1.0
    rsi_extreme[df["rsi"] > 75] = -1.0
    df["sig_mr_rsi_extreme"] = rsi_extreme

    # ─── Final: fill warmup NaN with 0 (neutral) ──────────────────────
    sig_cols = [c for c in df.columns if c.startswith("sig_")]
    df[sig_cols] = df[sig_cols].fillna(0.0)

    return df


def signal_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of sig_* columns present in the DataFrame, in order."""
    return [c for c in df.columns if c.startswith("sig_")]


# =============================================================================
# Contract signals (funding / OI / liquidation)
#
# 6 signals introduced by docs/GinkgoSpider/合约数据-采集与信号-设计.md §7.1.
# These are computed from columns merged by utils.data_loader.merge_contract_data:
#   funding_rate, sum_open_interest, liq_long_usd, liq_short_usd
#
# All 6 outputs honor the same value-range contract ([-1, 1]); warmup NaN is
# filled with 0 (neutral).
#
# ``add_contract_signals`` is intentionally a separate entry point from
# ``add_signals`` so the existing 22-signal pipeline keeps its invariants.
# Missing contract columns → the corresponding sig_* is set to 0 (neutral),
# so callers with partial data (e.g. SUI/ASTER before listing) still produce
# a uniform-shape DataFrame.
# =============================================================================

CONTRACT_SIGNAL_COLS = [
    "sig_funding_current",
    "sig_funding_trend",
    "sig_oi_change_zscore",
    "sig_liq_long_zscore",
    "sig_liq_short_zscore",
    "sig_liq_imbalance",
]

# 默认假设输入为 5min K 线；将 24h/1h 等窗口按 5min bars 换算
BARS_PER_HOUR_5MIN = 12
BARS_PER_DAY_5MIN = 288


def compute_sig_funding_current(
    funding_rate: pd.Series, window: int = 200,
) -> pd.Series:
    """当前 funding 值 → [-1, 1]。Rolling z-score 抗波动率 regime 漂移。"""
    return znorm(funding_rate, window=window)


def compute_sig_funding_trend(
    funding_rate: pd.Series,
    mean_window: int = BARS_PER_DAY_5MIN,
    znorm_window: int = 200,
) -> pd.Series:
    """24h funding 均值趋势 → z-score。正值表示过去 24h 多头付费多头趋势延续。"""
    avg = funding_rate.rolling(mean_window, min_periods=1).mean()
    return znorm(avg, window=znorm_window)


def compute_sig_oi_change_zscore(
    open_interest: pd.Series,
    roll: int = BARS_PER_HOUR_5MIN,
    znorm_window: int = 200,
) -> pd.Series:
    """OI 变化率（12 根 5min = 1h 滚动均值）z-score。OI 快速增加 → 杠杆进场。"""
    pct = open_interest.pct_change()
    smoothed = pct.rolling(roll, min_periods=1).mean()
    return znorm(smoothed, window=znorm_window)


def compute_sig_liq_long_zscore(
    liq_long_usd: pd.Series, window: int = 100,
) -> pd.Series:
    """多头爆仓量 z-score。极大值表示多头被动出清。"""
    return znorm(liq_long_usd, window=window)


def compute_sig_liq_short_zscore(
    liq_short_usd: pd.Series, window: int = 100,
) -> pd.Series:
    """空头爆仓量 z-score。"""
    return znorm(liq_short_usd, window=window)


def compute_sig_liq_imbalance(
    liq_long_usd: pd.Series,
    liq_short_usd: pd.Series,
    epsilon_window: int = 96,
    epsilon_ratio: float = 0.1,
) -> pd.Series:
    """
    爆仓不平衡（long - short） / (long + short + ε)，带 Laplace 平滑。

    ε = ``epsilon_ratio`` × rolling_mean(long + short, ``epsilon_window``)
      - 低流动性 / 低成交量时段：总量远低于 24h 均值 → ε 主导分母，信号衰减趋 0
      - 高活跃时段：总量远大于 24h 均值 → ε 可忽略，信号 ≈ 原公式

    默认 window=96 对应 15min bucket × 4/h × 24h。若输入是 5min K 线频率，
    调用方需先把 liq 列 broadcast 到 5min（由 merge_contract_data 完成），
    此处 window 代表"15min 桶数"，不是 5min 行数。

    返回值域 [-1, 1]。
    """
    total = (liq_long_usd.fillna(0.0) + liq_short_usd.fillna(0.0))
    # ε 的 rolling mean 带 min_periods=1，冷启动时用实际已有样本均值
    ma = total.rolling(epsilon_window, min_periods=1).mean()
    eps = epsilon_ratio * ma
    denom = total + eps
    # 极端边界：整个窗口都是 0 → denom=0 → 返回 0
    diff = (liq_long_usd.fillna(0.0) - liq_short_usd.fillna(0.0))
    out = diff / denom.where(denom > 0, np.nan)
    out = out.fillna(0.0).clip(-1.0, 1.0)
    return out


def add_contract_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append 6 contract-market signals to ``df``. Operates in-place on a copy.

    Required optional columns on ``df`` (any subset — missing ones mean the
    corresponding sig_* column is all-zero neutral):
      - ``funding_rate``               → sig_funding_current, sig_funding_trend
      - ``sum_open_interest``          → sig_oi_change_zscore
      - ``liq_long_usd``/liq_short_usd → sig_liq_{long,short}_zscore,
                                         sig_liq_imbalance

    Warmup NaN filled with 0 (neutral), consistent with ``add_signals``.
    """
    df = df.copy()
    n = len(df)
    zeros = pd.Series(0.0, index=df.index, dtype=float)

    if "funding_rate" in df.columns:
        fr = pd.to_numeric(df["funding_rate"], errors="coerce")
        df["sig_funding_current"] = compute_sig_funding_current(fr)
        df["sig_funding_trend"] = compute_sig_funding_trend(fr)
    else:
        df["sig_funding_current"] = zeros
        df["sig_funding_trend"] = zeros

    if "sum_open_interest" in df.columns:
        oi = pd.to_numeric(df["sum_open_interest"], errors="coerce")
        df["sig_oi_change_zscore"] = compute_sig_oi_change_zscore(oi)
    else:
        df["sig_oi_change_zscore"] = zeros

    has_liq = "liq_long_usd" in df.columns and "liq_short_usd" in df.columns
    if has_liq:
        long_ = pd.to_numeric(df["liq_long_usd"], errors="coerce").fillna(0.0)
        short_ = pd.to_numeric(df["liq_short_usd"], errors="coerce").fillna(0.0)
        df["sig_liq_long_zscore"] = compute_sig_liq_long_zscore(long_)
        df["sig_liq_short_zscore"] = compute_sig_liq_short_zscore(short_)
        df["sig_liq_imbalance"] = compute_sig_liq_imbalance(long_, short_)
    else:
        df["sig_liq_long_zscore"] = zeros
        df["sig_liq_short_zscore"] = zeros
        df["sig_liq_imbalance"] = zeros

    df[CONTRACT_SIGNAL_COLS] = df[CONTRACT_SIGNAL_COLS].fillna(0.0)
    return df
