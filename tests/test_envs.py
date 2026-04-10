"""Basic sanity tests for trading environments."""
import numpy as np
import pandas as pd
import pytest
from gymnasium.utils.env_checker import check_env

from envs import StockTradingEnv, CryptoTradingEnv


def make_dummy_df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    price = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame(
        {
            "open": price * rng.uniform(0.99, 1.0, n),
            "high": price * rng.uniform(1.0, 1.01, n),
            "low": price * rng.uniform(0.99, 1.0, n),
            "close": price,
            "volume": rng.integers(100_000, 1_000_000, n).astype(float),
        }
    )


@pytest.fixture
def dummy_df():
    return make_dummy_df()


def test_stock_env_gymnasium_check(dummy_df):
    env = StockTradingEnv(dummy_df, window_size=10)
    check_env(env, warn=True)


def test_crypto_env_gymnasium_check(dummy_df):
    env = CryptoTradingEnv(dummy_df, window_size=10)
    check_env(env, warn=True)


def test_stock_env_episode(dummy_df):
    env = StockTradingEnv(dummy_df, window_size=10, initial_balance=10_000)
    obs, info = env.reset()
    assert obs.shape == (10, env.n_features)

    done = False
    steps = 0
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1

    assert steps > 0
    assert "portfolio_value" in info


def test_buy_sell_mechanics(dummy_df):
    env = StockTradingEnv(dummy_df, window_size=10, initial_balance=10_000, commission=0.0)
    env.reset()

    env.step(1)  # Buy
    assert env.position > 0
    assert env.balance == 0.0

    env.step(2)  # Sell
    assert env.position == 0.0
    assert env.balance > 0
