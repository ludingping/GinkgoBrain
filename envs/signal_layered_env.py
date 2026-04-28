"""
Signal-layered trading environment (Phase 2, 4h PPO path).

Implements §5.1–5.4 of `docs/GinkgoBrain/信号分层架构与可解释RL设计.md`:
- Action space:    Discrete(5) target position {0, 25, 50, 75, 100}%.
- Observation:     (window_size, n_signals + 4_state) — only sig_* columns +
                   4 state features. No raw OHLCV or indicator columns.
- ATR stop loss:   env-side safety net, not an RL action. Cooldown after stop.
- Reward:          log_return + asymmetric risk aversion + excess_return
                   - trade_cost. No inertia / missed-opportunity / stop penalty.

Keeps the legacy StockTradingEnv / CryptoTradingEnv untouched — new paradigm
coexists with the old one and is selected via `--paradigm signal_layered`.
"""
from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import yaml
from gymnasium import spaces


TARGET_POSITION: dict[int, float] = {
    0: 0.00,   # 空仓
    1: 0.25,   # 轻仓
    2: 0.50,   # 半仓
    3: 0.75,   # 重仓
    4: 1.00,   # 满仓
}

MIN_REBALANCE_PCT = 0.02        # 仓位变动小于 2% 不交易
SIGNAL_WARMUP_WINDOW = 100      # 必须与 utils/signals.py 的 znorm window 一致
SAFETY_BUFFER = 10              # 留给 rolling rank / regime_drawdown

STATE_DIM = 4                   # position_ratio, unrealized_pnl,
                                # steps_since_stop, cumulative_log_return

REWARD_KEYS = (                 # 与 §5.4.2 显式声明的项 + trade-frequency 惩罚
    "log_return",
    "risk_aversion_adjustment",
    "excess_return",
    "trade_cost",
    "action_inertia",
    "trade_penalty",
)


def load_signal_list(config_path: str | Path) -> list[str]:
    """Read the `signals` list from a `signals_v*.yaml` config."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    sigs = cfg.get("signals")
    if not sigs:
        raise ValueError(f"{config_path}: missing/empty `signals` field")
    return list(sigs)


class SignalLayeredEnv(gym.Env):
    """Phase 2 env driven by pre-computed sig_* signals + a minimal state vector."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        signal_cols: list[str],
        window_size: int = 24,
        initial_balance: float = 10_000.0,
        commission: float = 0.0005,
        risk_aversion_coef: float = 0.5,
        excess_return_coef: float = 0.5,
        action_inertia_coef: float = 0.0,
        trade_penalty_coef: float = 0.0,
        stop_atr_mult: float = 2.0,
        stop_cooldown_steps: int = 3,
        random_start: bool = False,
        render_mode: str | None = None,
    ):
        super().__init__()

        if "atr" not in df.columns:
            raise ValueError("SignalLayeredEnv requires an `atr` column for stop loss.")
        if "close" not in df.columns:
            raise ValueError("SignalLayeredEnv requires a `close` column.")
        missing = [c for c in signal_cols if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame missing signal columns: {missing}")

        self.df = df.reset_index(drop=True)
        self.signal_cols = list(signal_cols)
        self.n_signals = len(self.signal_cols)

        self.window_size = int(window_size)
        self.initial_balance = float(initial_balance)
        self.commission = float(commission)
        self.risk_aversion_coef = float(risk_aversion_coef)
        self.excess_return_coef = float(excess_return_coef)
        self.action_inertia_coef = float(action_inertia_coef)
        self.trade_penalty_coef = float(trade_penalty_coef)
        self.stop_atr_mult = float(stop_atr_mult)
        self.stop_cooldown_steps = int(stop_cooldown_steps)
        self.random_start = bool(random_start)
        self.render_mode = render_mode

        # Pre-extract signal matrix (float32) for fast obs slicing.
        self._sig_mat = self.df[self.signal_cols].to_numpy(dtype=np.float32)
        self._close = self.df["close"].to_numpy(dtype=np.float64)
        self._atr = self.df["atr"].to_numpy(dtype=np.float64)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, self.n_signals + STATE_DIM),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(5)

        self._reset_state()

    # ------------------------------------------------------------------ state

    def _reset_state(self) -> None:
        self.balance: float = self.initial_balance
        self.position: float = 0.0
        self._prev_action: int = 0
        self._entry_price: float | None = None
        self._steps_since_stop: int = self.window_size  # "long ago"
        self._cum_log_return: float = 0.0
        self.current_step: int = self._min_start()
        self.total_reward: float = 0.0
        self._last_trade_cost: float = 0.0
        self._last_reward_breakdown: dict[str, float] = {k: 0.0 for k in REWARD_KEYS}
        self._last_stop_triggered: bool = False
        self.trades: list[dict] = []
        self.stop_loss_events: list[dict] = []
        # Rolling state buffer for observation (position_ratio, unrealized_pnl,
        # steps_since_stop_norm, cum_log_return_clipped).
        self._state_history = np.zeros((self.window_size, STATE_DIM), dtype=np.float32)

    def _min_start(self) -> int:
        """Earliest legal step; enforces warmup + window + buffer (§5.5.2)."""
        return SIGNAL_WARMUP_WINDOW + self.window_size + SAFETY_BUFFER

    # ------------------------------------------------------------------ reset

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        if self.random_start:
            lo = self._min_start()
            hi = len(self.df) - 2
            if hi > lo:
                self.current_step = int(self.np_random.integers(lo, hi))
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    # ------------------------------------------------------------------ step

    def step(self, action: int):
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}; expected 0..4")

        price = float(self._close[self.current_step])
        atr = float(self._atr[self.current_step])
        prev_portfolio = self._portfolio_value(price)

        # ── 1. ATR stop loss (safety net, runs before RL action) ──────────
        self._last_stop_triggered = False
        if self.position > 0 and self._entry_price is not None:
            loss_pct = (price - self._entry_price) / self._entry_price
            stop_threshold = -(self.stop_atr_mult * atr / self._entry_price)
            if loss_pct < stop_threshold:
                self._force_close(price, reason="atr_stop")
                self._last_stop_triggered = True
                action = 0   # override RL action this step
                self._steps_since_stop = 0

        # ── 2. Cooldown: any non-zero action within cooldown → 0 ──────────
        in_cooldown = (
            self._steps_since_stop < self.stop_cooldown_steps
            and not self._last_stop_triggered
        )
        if in_cooldown and action != 0:
            action = 0

        # ── 3. Execute RL action ─────────────────────────────────────────
        self._execute_action(action, price)

        # ── 4. Advance time and recompute portfolio ──────────────────────
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        new_price = float(self._close[self.current_step])
        new_portfolio = self._portfolio_value(new_price)

        # ── 5. Reward ────────────────────────────────────────────────────
        log_return = float(np.log(max(new_portfolio, 1e-8) / max(prev_portfolio, 1e-8)))

        risk_adj = 0.0
        if log_return < 0:
            risk_adj = log_return * self.risk_aversion_coef   # negative

        excess = 0.0
        if self.excess_return_coef > 0:
            market_return = float(np.log(max(new_price, 1e-8) / max(price, 1e-8)))
            excess = self.excess_return_coef * (log_return - market_return)

        trade_cost_term = -self._last_trade_cost

        # ── 5b. Trade-frequency 惩罚（防 PPO 在 5min noise 上反复横跳）──
        # action_inertia: 仓位档位变化幅度的连续惩罚（|target_ratio - prev_target_ratio|）
        # trade_penalty: 任何 action 切换的固定惩罚（离散，每次切换固定扣分）
        action_delta = abs(TARGET_POSITION[action] - TARGET_POSITION[self._prev_action])
        inertia_term = -self.action_inertia_coef * action_delta
        trade_pen_term = -self.trade_penalty_coef * (1.0 if action != self._prev_action else 0.0)

        reward = log_return + risk_adj + excess + trade_cost_term + inertia_term + trade_pen_term
        self._last_reward_breakdown = {
            "log_return": log_return,
            "risk_aversion_adjustment": risk_adj,
            "excess_return": excess,
            "trade_cost": trade_cost_term,
            "action_inertia": inertia_term,
            "trade_penalty": trade_pen_term,
        }
        self.total_reward += reward
        self._cum_log_return += log_return
        if not self._last_stop_triggered:
            self._steps_since_stop = min(self._steps_since_stop + 1, self.window_size)
        self._prev_action = action

        # ── 6. Roll state history forward ────────────────────────────────
        self._update_state_history(new_price)

        obs = self._get_obs()
        info = self._get_info()
        return obs, reward, done, False, info

    # ------------------------------------------------------------------ action

    def _execute_action(self, action: int, price: float) -> None:
        self._last_trade_cost = 0.0
        target_ratio = TARGET_POSITION[action]
        portfolio = self._portfolio_value(price)
        if portfolio <= 0:
            return

        target_value = portfolio * target_ratio
        current_value = self.position * price
        delta_value = target_value - current_value

        if abs(delta_value) / portfolio < MIN_REBALANCE_PCT:
            return

        if delta_value > 0:
            units = (delta_value * (1 - self.commission)) / price
            new_position = self.position + units
            new_value = new_position * price
            if self._entry_price is None or self.position == 0:
                self._entry_price = price
            else:
                prev_cost = self._entry_price * self.position
                self._entry_price = (prev_cost + price * units) / new_position \
                    if new_position > 0 else price
            self.position = new_position
            self.balance -= delta_value
            self._last_trade_cost = self.commission * (delta_value / portfolio)
            self.trades.append({
                "step": self.current_step, "side": "buy",
                "price": price, "target": target_ratio,
                "position_value": new_value,
            })
        else:
            units_to_sell = abs(delta_value) / price
            proceeds = units_to_sell * price * (1 - self.commission)
            self.balance += proceeds
            self.position -= units_to_sell
            if self.position <= 1e-10:
                self.position = 0.0
                self._entry_price = None
            self._last_trade_cost = self.commission * (abs(delta_value) / portfolio)
            self.trades.append({
                "step": self.current_step, "side": "sell",
                "price": price, "target": target_ratio,
                "position_value": self.position * price,
            })

    def _force_close(self, price: float, reason: str) -> None:
        """Liquidate all position, log a stop loss event, reset entry price."""
        if self.position <= 0:
            return
        proceeds = self.position * price * (1 - self.commission)
        cost = self.commission * (self.position * price
                                  / max(self._portfolio_value(price), 1e-8))
        self.balance += proceeds
        self.stop_loss_events.append({
            "step": self.current_step,
            "price": price,
            "entry_price": self._entry_price,
            "reason": reason,
        })
        self.position = 0.0
        self._entry_price = None
        self._last_trade_cost = cost

    # ------------------------------------------------------------------ obs

    def _update_state_history(self, price: float) -> None:
        portfolio = self._portfolio_value(price) + 1e-8
        pos_ratio = float(np.clip(self.position * price / portfolio, 0.0, 1.0))
        if self._entry_price is not None and self.position > 0:
            unrealized = (price - self._entry_price) / self._entry_price
        else:
            unrealized = 0.0
        unrealized = float(np.clip(unrealized, -0.5, 0.5))
        steps_norm = float(np.clip(self._steps_since_stop / max(self.window_size, 1),
                                   0.0, 1.0))
        cum_clip = float(np.clip(self._cum_log_return, -1.0, 1.0))

        self._state_history = np.roll(self._state_history, -1, axis=0)
        self._state_history[-1] = [pos_ratio, unrealized, steps_norm, cum_clip]

    def _get_obs(self) -> np.ndarray:
        lo = self.current_step - self.window_size
        hi = self.current_step
        if lo < 0:
            # should not happen once _min_start is respected
            sig_window = np.zeros((self.window_size, self.n_signals), dtype=np.float32)
            sig_window[-hi:] = self._sig_mat[:hi]
        else:
            sig_window = self._sig_mat[lo:hi]
        obs = np.concatenate([sig_window, self._state_history], axis=1)
        return obs.astype(np.float32, copy=False)

    # ------------------------------------------------------------------ info

    def _portfolio_value(self, price: float) -> float:
        return self.balance + self.position * price

    def _get_info(self) -> dict:
        price = float(self._close[self.current_step])
        portfolio = self._portfolio_value(price)
        pos_ratio = (self.position * price / portfolio) if portfolio > 0 else 0.0
        return {
            "step": self.current_step,
            "price": price,
            "portfolio_value": portfolio,
            "balance": self.balance,
            "position": self.position,
            "position_ratio": pos_ratio,
            "total_reward": self.total_reward,
            "reward_breakdown": dict(self._last_reward_breakdown),
            "stop_loss_triggered": self._last_stop_triggered,
            "steps_since_stop": self._steps_since_stop,
        }

    def render(self):
        info = self._get_info()
        print(
            f"step={info['step']} price={info['price']:.2f} "
            f"portfolio={info['portfolio_value']:.2f} "
            f"pos_ratio={info['position_ratio']:.2%}"
        )
