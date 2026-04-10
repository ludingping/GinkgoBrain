"""Evaluation metrics for trained policies."""
import numpy as np
import pandas as pd
from stable_baselines3.common.vec_env import VecEnv


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
