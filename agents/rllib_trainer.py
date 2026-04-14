"""RLlib-based trainer for trading environments."""
from pathlib import Path
from typing import Callable, Type

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.algorithms.td3 import TD3Config
from ray.rllib.algorithms.dqn import DQNConfig

ALGORITHM_CONFIGS: dict[str, type] = {
    "ppo": PPOConfig,
    "sac": SACConfig,
    "td3": TD3Config,
    "dqn": DQNConfig,
}


class RLlibTrainer:
    """
    Wraps RLlib training loop with a similar interface to the SB3 Trainer.

    RLlib works with env *classes* (or creator callables) rather than env
    instances, so pass in the class and its constructor kwargs separately.

    Usage:
        from envs import CryptoTradingEnv
        trainer = RLlibTrainer(
            env_cls=CryptoTradingEnv,
            env_config={"df": train_df, "window_size": 20},
            algo="ppo",
            run_name="btc_ppo_rllib",
        )
        trainer.train(n_iterations=200)
        trainer.save()
    """

    def __init__(
        self,
        env_cls: Type,
        env_config: dict,
        algo: str = "ppo",
        run_name: str = "rllib_run",
        model_dir: str = "models/saved",
        log_dir: str = "models/logs",
        algo_kwargs: dict | None = None,
        num_workers: int = 0,
        framework: str = "torch",
    ):
        self.run_name = run_name
        self.model_dir = Path(model_dir)
        self.log_dir = Path(log_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        config_cls = ALGORITHM_CONFIGS.get(algo.lower())
        if config_cls is None:
            raise ValueError(
                f"Unknown algorithm '{algo}'. Choose from: {list(ALGORITHM_CONFIGS)}"
            )

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, log_to_driver=False)

        env_name = f"trading_{run_name}"
        tune.register_env(env_name, lambda cfg: env_cls(**cfg))

        self._algo_kwargs = algo_kwargs or {}
        config = (
            config_cls()
            .environment(env=env_name, env_config=env_config)
            .framework(framework)
            .env_runners(num_env_runners=num_workers)
            .training(**self._algo_kwargs)
        )
        self.algo = config.build()
        self._checkpoint_path: str | None = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, n_iterations: int = 100, log_interval: int = 10):
        """Run *n_iterations* of RLlib training (each iteration = one SGD pass
        over the collected rollouts).

        Args:
            n_iterations: Number of training iterations.
            log_interval:  Print a progress line every N iterations.
        """
        print(f"[RLlib] Starting '{self.run_name}' — {n_iterations} iterations")
        for i in range(1, n_iterations + 1):
            result = self.algo.train()
            if i % log_interval == 0 or i == n_iterations:
                ep_rew = result.get("env_runners", {}).get(
                    "episode_reward_mean",
                    result.get("episode_reward_mean", float("nan")),
                )
                print(
                    f"  iter {i:>4}/{n_iterations} | "
                    f"episode_reward_mean={ep_rew:.4f}"
                )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, dirname: str | None = None) -> Path:
        """Save a RLlib checkpoint and return its directory path."""
        save_dir = self.model_dir / (dirname or self.run_name)
        save_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = self.algo.save(str(save_dir))
        # RLlib 2.x returns a Checkpoint object; extract the path string.
        path_str = getattr(checkpoint, "path", str(checkpoint))
        self._checkpoint_path = path_str
        print(f"[RLlib] Checkpoint saved to {path_str}")
        return Path(path_str)

    @classmethod
    def load(
        cls,
        checkpoint_path: str,
        env_cls: Type,
        env_config: dict,
        algo: str = "ppo",
        run_name: str = "rllib_loaded",
    ) -> "RLlibTrainer":
        """Restore a trainer from a saved checkpoint."""
        config_cls = ALGORITHM_CONFIGS[algo.lower()]

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, log_to_driver=False)

        env_name = f"trading_{run_name}"
        tune.register_env(env_name, lambda cfg: env_cls(**cfg))

        config = (
            config_cls()
            .environment(env=env_name, env_config=env_config)
            .framework("torch")
        )
        instance = cls.__new__(cls)
        instance.run_name = run_name
        instance.model_dir = Path("models/saved")
        instance.log_dir = Path("models/logs")
        instance._checkpoint_path = checkpoint_path
        instance.algo = config.build()
        instance.algo.restore(checkpoint_path)
        return instance

    # ------------------------------------------------------------------
    # Inference helper
    # ------------------------------------------------------------------

    def predict(self, obs):
        """Return the greedy action for a single observation."""
        return self.algo.compute_single_action(obs, explore=False)
