"""Evaluation metrics for trained policies."""
import numpy as np
import pandas as pd
from stable_baselines3.common.vec_env import VecEnv


# Bars per year by timeframe (24/7 crypto). Canonical copy — scripts that
# annualise a Sharpe ratio must derive the factor from the config timeframe,
# never hardcode one (the 4h constant 2190 halved every reported 1h Sharpe).
BARS_PER_YEAR: dict[str, int] = {
    "1m": 1440 * 365, "5min": 288 * 365, "15min": 96 * 365, "30min": 48 * 365,
    "1h": 24 * 365, "2h": 12 * 365, "4h": 6 * 365, "1d": 365,
}


def bars_per_year(timeframe: str) -> int:
    key = str(timeframe).lower()
    if key not in BARS_PER_YEAR:
        raise ValueError(f"Unknown timeframe '{timeframe}'; known: {sorted(BARS_PER_YEAR)}")
    return BARS_PER_YEAR[key]


def annualised_sharpe(returns, periods_per_year: int) -> float:
    """Sharpe of a per-bar return series, annualised by sqrt(periods_per_year)."""
    r = np.asarray(returns, dtype=float)
    if r.size < 2:
        return 0.0
    return float(r.mean() / (r.std() + 1e-10) * np.sqrt(periods_per_year))


def evaluate_policy(model, env, n_episodes: int = 10) -> dict:
    """
    Run `n_episodes` episodes and return aggregated performance metrics.

    Returns:
        dict with keys: mean_reward, std_reward, mean_portfolio_return,
                        sharpe_ratio, max_drawdown, win_rate
    """
    episode_rewards = []
    portfolio_returns = []

    for _ in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        initial_value = info.get("portfolio_value", 1.0)
        portfolio_values = [initial_value]

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            portfolio_values.append(info.get("portfolio_value", portfolio_values[-1]))

        episode_rewards.append(ep_reward)
        portfolio_values = np.array(portfolio_values)
        portfolio_returns.append((portfolio_values[-1] / portfolio_values[0]) - 1)

    returns = np.array(portfolio_returns)
    sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(n_episodes)

    all_pv = np.array(portfolio_returns) + 1
    cummax = np.maximum.accumulate(all_pv)
    max_drawdown = float(np.min((all_pv - cummax) / (cummax + 1e-8)))

    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_portfolio_return": float(returns.mean()),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "win_rate": float((returns > 0).mean()),
    }
