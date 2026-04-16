"""
Shared building blocks for the four-stage SB3 PPO curriculum.

Used by `notebooks/crypto_ppo_stage{1..4}*.ipynb`. Implements P0/P5/P8/P9/P10
of `docs/training/PPO训练优化-观察归一化与奖励整形.md`.

- `add_indicators`                 — base 1m TA features
- `add_multi_timeframe_indicators` — P6 MTF features merged via shift(1)+ffill
- `resample_ohlcv`                 — 1m → 1h/4h/1d
- `CryptoPPOEnv`                   — gym env (P0/P1/P5/P8/P10 reward+obs)
- `TradingCNN`                     — P9 1D-CNN with AdaptiveAvgPool decoupling window_size
- `warm_start_policy`              — cross-stage weight transfer (P9 §14.3 + channel padding)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import ta
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# ─── Feature engineering ────────────────────────────────────────────────────

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Trend / momentum / volatility / volume indicators. Preserves DatetimeIndex."""
    out = df.copy()
    out["ema_9"]       = ta.trend.EMAIndicator(out["close"], window=9).ema_indicator()
    out["ema_21"]      = ta.trend.EMAIndicator(out["close"], window=21).ema_indicator()
    macd               = ta.trend.MACD(out["close"])
    out["macd"]        = macd.macd()
    out["macd_signal"] = macd.macd_signal()
    out["rsi"]         = ta.momentum.RSIIndicator(out["close"], window=14).rsi()
    stoch              = ta.momentum.StochasticOscillator(out["high"], out["low"], out["close"])
    out["stoch_k"]     = stoch.stoch()
    out["stoch_d"]     = stoch.stoch_signal()
    bb                 = ta.volatility.BollingerBands(out["close"])
    out["bb_upper"]    = bb.bollinger_hband()
    out["bb_lower"]    = bb.bollinger_lband()
    out["atr"]         = ta.volatility.AverageTrueRange(out["high"], out["low"], out["close"]).average_true_range()
    out["obv"]         = ta.volume.OnBalanceVolumeIndicator(out["close"], out["volume"]).on_balance_volume()
    return out


def add_multi_timeframe_indicators(
    df_1m: pd.DataFrame,
    timeframes=("5min", "15min"),
) -> pd.DataFrame:
    """
    Merge higher-TF indicators onto 1m bars via `shift(1) + ffill` (P6).

    `shift(1)` ensures that at 1m bar T the higher-TF value reflects the
    *last closed* candle strictly before T — no look-ahead.
    """
    df = df_1m.copy()
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    result = df.copy()

    keep_by_tf = {
        "5min":  ["ema_9", "ema_21", "rsi", "macd", "macd_signal"],
        "15min": ["ema_9", "rsi", "macd"],
    }

    for tf in timeframes:
        minutes = int("".join(filter(str.isdigit, tf)))
        prefix  = f"m{minutes}"

        ohlcv_tf = df.resample(tf, closed="left", label="left").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna()

        ind_tf = add_indicators(ohlcv_tf)
        cols   = [c for c in keep_by_tf[tf] if c in ind_tf.columns]
        ind_tf = ind_tf[cols].rename(columns={c: f"{prefix}_{c}" for c in cols})

        ind_tf = ind_tf.shift(1).reindex(result.index, method="ffill")
        result = pd.concat([result, ind_tf], axis=1)

    return result.reset_index().dropna().reset_index(drop=True)


def resample_ohlcv(df_1m: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample 1m OHLCV to higher timeframe (e.g. '1H', '4H', '1D')."""
    df = df_1m.copy()
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    out = df.resample(timeframe, closed="left", label="left").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna()
    return out.reset_index()


# ─── Trading env (P0/P1/P5/P8/P10) ──────────────────────────────────────────

class CryptoPPOEnv(gym.Env):
    """
    Stage1–4 PPO trading env shared across the curriculum.

    P0  — group-wise normalization in `_get_obs` (ref = current-window last_close)
    P1  — `trade_penalty_coef * commission` on every trade event
    P5  — action inertia penalty on holding-state flips, with pos history buffer
    P8  — log-return reward with asymmetric risk aversion (negative scaled by 1+α)
    P10 — `random_start=True` reset for sub-proc rollout decorrelation
    """
    metadata = {"render_modes": []}

    PRICE_COLS = {
        "open", "high", "low", "close",
        "ema_9", "ema_21", "bb_upper", "bb_lower",
        "macd", "macd_signal", "atr",
        "m5_ema_9", "m5_ema_21", "m5_macd", "m5_macd_signal",
        "m15_ema_9", "m15_macd",
    }
    PCT_COLS = {"rsi", "stoch_k", "stoch_d", "m5_rsi", "m15_rsi"}

    def __init__(
        self,
        features: np.ndarray,
        feature_cols,
        window_size: int,
        initial_balance: float    = 10_000.0,
        commission: float         = 0.0005,
        trade_penalty_coef: float = 1.0,
        action_inertia_coef: float = 0.001,
        risk_aversion_coef: float  = 2.0,
        random_start: bool        = False,
    ):
        super().__init__()
        self.features            = features
        self.feature_cols        = list(feature_cols)
        self.window_size         = window_size
        self.initial_balance     = initial_balance
        self.commission          = commission
        self.trade_penalty_coef  = trade_penalty_coef
        self.action_inertia_coef = action_inertia_coef
        self.risk_aversion_coef  = risk_aversion_coef
        self.random_start        = random_start
        self.n_feat              = features.shape[1]

        self._price_idx = np.array(
            [i for i, c in enumerate(self.feature_cols) if c in self.PRICE_COLS],
            dtype=np.int64,
        )
        self._pct_idx = np.array(
            [i for i, c in enumerate(self.feature_cols) if c in self.PCT_COLS],
            dtype=np.int64,
        )
        self._obv_idx    = self.feature_cols.index("obv")    if "obv"    in self.feature_cols else -1
        self._volume_idx = self.feature_cols.index("volume") if "volume" in self.feature_cols else -1
        self._close_idx  = self.feature_cols.index("close")

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.window_size, self.n_feat + 2),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)
        self._reset_state()

    def _reset_state(self):
        self.balance           = self.initial_balance
        self.position          = 0.0
        self.step_idx          = self.window_size
        self._last_trade_cost  = 0.0
        self._prev_is_holding  = 0
        self._pos_history      = np.zeros((self.window_size, 2), dtype=np.float32)
        self.trades            = []
        self.portfolio_history = []

    def _portfolio_value(self, price: float) -> float:
        return self.balance + self.position * price

    def _update_pos_history(self, price: float):
        port      = self._portfolio_value(price) + 1e-8
        pos_ratio = self.position * price / port
        is_hold   = float(self.position > 0)
        self._pos_history = np.roll(self._pos_history, -1, axis=0)
        self._pos_history[-1] = [pos_ratio, is_hold]

    def _get_obs(self) -> np.ndarray:
        window = self.features[self.step_idx - self.window_size : self.step_idx].copy()
        last_close = float(window[-1, self._close_idx]) + 1e-8

        if self._obv_idx >= 0:
            obv = window[:, self._obv_idx]
            prev = np.concatenate([[obv[0]], obv[:-1]])
            pct  = np.where(np.abs(prev) > 1e-8, (obv - prev) / prev, 0.0)
            window[:, self._obv_idx] = np.clip(pct, -1.0, 1.0)

        if self._price_idx.size > 0:
            window[:, self._price_idx] = window[:, self._price_idx] / last_close

        if self._volume_idx >= 0:
            vmax = float(window[:, self._volume_idx].max()) + 1e-8
            window[:, self._volume_idx] = window[:, self._volume_idx] / vmax

        if self._pct_idx.size > 0:
            window[:, self._pct_idx] = window[:, self._pct_idx] / 100.0

        return np.concatenate([window, self._pos_history], axis=1).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        if self.random_start:
            max_start = len(self.features) - 101
            self.step_idx = int(self.np_random.integers(self.window_size, max_start))
        return self._get_obs(), {}

    def _execute_action(self, action: int, price: float):
        self._last_trade_cost = 0.0
        if action == 1 and self.balance > 0:
            units = (self.balance * (1.0 - self.commission)) / price
            self.position += units
            self.balance   = 0.0
            self._last_trade_cost = self.commission
            self.trades.append({"step": self.step_idx, "side": "buy",  "price": price})
        elif action == 2 and self.position > 0:
            self.balance += self.position * price * (1.0 - self.commission)
            self.position = 0.0
            self._last_trade_cost = self.commission
            self.trades.append({"step": self.step_idx, "side": "sell", "price": price})

    def step(self, action: int):
        price     = float(self.features[self.step_idx, self._close_idx])
        prev_port = self._portfolio_value(price)

        self._execute_action(action, price)

        self.step_idx += 1
        done = self.step_idx >= len(self.features) - 1

        new_price = float(self.features[self.step_idx, self._close_idx])
        new_port  = self._portfolio_value(new_price)
        self.portfolio_history.append(new_port)

        log_ret = float(np.log(max(new_port, 1e-8) / max(prev_port, 1e-8)))
        reward  = log_ret * (1.0 + self.risk_aversion_coef) if log_ret < 0 else log_ret
        reward -= self._last_trade_cost * self.trade_penalty_coef

        curr_is_holding = int(self.position > 0)
        if curr_is_holding != self._prev_is_holding:
            reward -= self.action_inertia_coef
        self._prev_is_holding = curr_is_holding

        self._update_pos_history(new_price)

        info = {
            "portfolio_value": new_port,
            "balance":         self.balance,
            "position":        self.position,
            "n_trades":        len(self.trades),
            "log_return":      log_ret,
            "traded":          bool(self._last_trade_cost > 0),
        }
        return self._get_obs(), float(reward), done, False, info


# ─── CNN feature extractor (P9) ─────────────────────────────────────────────

class TradingCNN(BaseFeaturesExtractor):
    """
    1D-CNN feature extractor.

    `AdaptiveAvgPool1d(POOL_SIZE)` after the conv stack collapses the time
    axis to a fixed length, decoupling `features_dim` from `window_size`.
    This is what lets stage1→stage4 transfer Linear/policy weights even when
    `window_size` changes (20 → 48 → 120).
    """
    POOL_SIZE = 8

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_steps, n_features = observation_space.shape

        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=3, padding=1),            nn.ReLU(),
            nn.Conv1d(64,         64, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(64,         32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(self.POOL_SIZE),
        )
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * self.POOL_SIZE, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs.permute(0, 2, 1)   # (B, window, feat) → (B, feat, window)
        return self.linear(self.cnn(x))


# ─── Cross-stage warm start ─────────────────────────────────────────────────

def warm_start_policy(
    new_model: PPO,
    prev_model_path: str,
    in_channel_mapping: dict[int, int] | None = None,
) -> dict:
    """
    Transfer compatible weights from a saved PPO model into `new_model.policy`.

    All same-shape parameters are copied. The first Conv1d (`*cnn.0.weight`)
    is treated specially when `in_channel_mapping` is provided: each
    `{curr_ch: prev_ch}` entry copies that input-channel slice from the old
    weight into the new weight, leaving unmapped channels at their random init.
    Used for stage3 → stage4 where MTF features expand the channel count
    (18 → 26) but OHLCV+1m indicators retain their semantic position.

    Returns: dict with `moved`, `expanded`, `skipped` lists for inspection.
    """
    prev = PPO.load(prev_model_path, device="cpu")
    src  = prev.policy.state_dict()
    dst  = new_model.policy.state_dict()

    moved, expanded, skipped = [], [], []

    for k, v_src in src.items():
        if k not in dst:
            skipped.append((k, "not in new policy"))
            continue
        v_dst = dst[k]

        if v_dst.shape == v_src.shape:
            dst[k] = v_src.clone()
            moved.append(k)
            continue

        # Selective channel copy on first Conv1d weight only.
        is_first_conv_w = (
            in_channel_mapping is not None
            and k.endswith("cnn.0.weight")
            and v_dst.dim() == 3
            and v_dst.shape[0] == v_src.shape[0]      # out_channels match
            and v_dst.shape[2] == v_src.shape[2]      # kernel_size match
        )
        if is_first_conv_w:
            new_w = v_dst.clone()
            for curr_idx, prev_idx in in_channel_mapping.items():
                if prev_idx >= v_src.shape[1] or curr_idx >= v_dst.shape[1]:
                    continue
                new_w[:, curr_idx, :] = v_src[:, prev_idx, :]
            dst[k] = new_w
            expanded.append(k)
            continue

        skipped.append((k, f"shape {tuple(v_src.shape)} ≠ {tuple(v_dst.shape)}"))

    new_model.policy.load_state_dict(dst, strict=True)

    return {"moved": moved, "expanded": expanded, "skipped": skipped}


# ─── Canonical feature column lists per stage ──────────────────────────────

FEATURE_COLS_BASE = [
    # OHLCV
    "open", "high", "low", "close", "volume",
    # 1m technical indicators
    "ema_9", "ema_21", "macd", "macd_signal",
    "rsi", "stoch_k", "stoch_d",
    "bb_upper", "bb_lower", "atr", "obv",
]  # 16 cols → +2 pos = 18 channels (stages 1/2/3)

FEATURE_COLS_MTF = FEATURE_COLS_BASE + [
    "m5_ema_9", "m5_ema_21", "m5_rsi", "m5_macd", "m5_macd_signal",
    "m15_ema_9", "m15_rsi", "m15_macd",
]  # 24 cols → +2 pos = 26 channels (stage 4)


# Channel mapping for stage3 (18 ch) → stage4 (26 ch) warm start.
# Stage3 layout: [0..15]=OHLCV+1m, [16,17]=pos
# Stage4 layout: [0..15]=OHLCV+1m, [16..23]=MTF (random init), [24,25]=pos
STAGE3_TO_STAGE4_CHANNEL_MAP = {i: i for i in range(16)} | {24: 16, 25: 17}
