"""Base trading environment shared by stock and crypto envs."""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from abc import abstractmethod


class BaseTradingEnv(gym.Env):
    """
    Base class for trading environments.

    Observation space: window of OHLCV + technical indicators, normalized.
    Action space: Discrete(3) — 0=Hold, 1=Buy, 2=Sell
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        df,
        window_size: int = 20,
        initial_balance: float = 10_000.0,
        commission: float = 0.001,
        trade_penalty_coef: float = 1.0,
        action_inertia_coef: float = 0.5,
        risk_aversion_coef: float = 2.0,
        excess_return_coef: float = 0.0,
        random_start: bool = False,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.commission = commission
        self.trade_penalty_coef = trade_penalty_coef
        self.action_inertia_coef = action_inertia_coef
        self.risk_aversion_coef = risk_aversion_coef
        self.excess_return_coef = excess_return_coef
        self.random_start = random_start
        self.render_mode = render_mode

        self.n_features = self._count_features()

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size, self.n_features),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)  # 0=Hold, 1=Buy, 2=Sell

        self._reset_state()

    @abstractmethod
    def _count_features(self) -> int:
        """Return number of features per time step."""

    @abstractmethod
    def _get_obs(self) -> np.ndarray:
        """Return observation array of shape (window_size, n_features)."""

    def _reset_state(self):
        self.balance = self.initial_balance
        self.position = 0.0   # units held
        self.current_step = self.window_size
        self.total_reward = 0.0
        self.trades: list[dict] = []
        self._last_trade_cost = 0.0
        self._prev_is_holding = 0
        self._pos_history = np.zeros((self.window_size, 2), dtype=np.float32)

    def _update_pos_history(self, price: float):
        portfolio = self._portfolio_value(price) + 1e-8
        pos_ratio = self.position * price / portfolio
        is_holding = float(self.position > 0)
        self._pos_history = np.roll(self._pos_history, -1, axis=0)
        self._pos_history[-1] = [pos_ratio, is_holding]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        if self.random_start:
            max_start = len(self.df) - 101
            if max_start > self.window_size:
                self.current_step = int(self.np_random.integers(self.window_size, max_start))
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int):
        price = float(self.df.loc[self.current_step, "close"])
        prev_portfolio = self._portfolio_value(price)

        self._execute_action(action, price)

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        new_price = float(self.df.loc[self.current_step, "close"])
        new_portfolio = self._portfolio_value(new_price)

        self._update_pos_history(new_price)

        # P8: log return + asymmetric risk penalty
        log_return = float(np.log(new_portfolio / max(prev_portfolio, 1e-8)))
        if log_return < 0:
            reward = log_return * (1.0 + self.risk_aversion_coef)
        else:
            reward = log_return

        # P11: excess return over market (alpha reward)
        # Rewards being in cash during drops, penalizes missing rallies
        if self.excess_return_coef > 0:
            market_return = float(np.log(new_price / max(price, 1e-8)))
            excess_return = log_return - market_return
            reward += self.excess_return_coef * excess_return

        # P1: trade penalty
        reward -= self._last_trade_cost * self.trade_penalty_coef

        # P5: action inertia penalty
        curr_is_holding = int(self.position > 0)
        if curr_is_holding != self._prev_is_holding:
            reward -= self.action_inertia_coef
        self._prev_is_holding = curr_is_holding

        self.total_reward += reward
        obs = self._get_obs()
        info = self._get_info()
        return obs, reward, done, False, info

    def _execute_action(self, action: int, price: float):
        self._last_trade_cost = 0.0
        if action == 1 and self.balance > 0:   # Buy
            units = (self.balance * (1 - self.commission)) / price
            self.position += units
            self.balance = 0.0
            self._last_trade_cost = self.commission
            self.trades.append({"step": self.current_step, "side": "buy", "price": price})
        elif action == 2 and self.position > 0:  # Sell
            self.balance += self.position * price * (1 - self.commission)
            self.position = 0.0
            self._last_trade_cost = self.commission
            self.trades.append({"step": self.current_step, "side": "sell", "price": price})

    def _portfolio_value(self, price: float) -> float:
        return self.balance + self.position * price

    def _get_info(self) -> dict:
        price = float(self.df.loc[self.current_step, "close"])
        return {
            "portfolio_value": self._portfolio_value(price),
            "balance": self.balance,
            "position": self.position,
            "total_reward": self.total_reward,
            "step": self.current_step,
        }

    def render(self):
        info = self._get_info()
        print(
            f"Step: {info['step']} | Portfolio: {info['portfolio_value']:.2f} "
            f"| Balance: {info['balance']:.2f} | Position: {info['position']:.4f}"
        )
