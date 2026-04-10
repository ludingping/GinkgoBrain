"""High-level trainer wrapping SB3 algorithms."""
import os
from pathlib import Path

from stable_baselines3 import PPO, A2C, SAC, TD3
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor

ALGORITHMS = {
    "ppo": PPO,
    "a2c": A2C,
    "sac": SAC,
    "td3": TD3,
}


class Trainer:
    """
    Wraps SB3 training loop with sensible defaults for trading environments.

    Usage:
        trainer = Trainer(env, eval_env, algo="ppo", run_name="btc_ppo_v1")
        trainer.train(total_timesteps=500_000)
        trainer.save()
    """

    def __init__(
        self,
        env,
        eval_env=None,
        algo: str = "ppo",
        run_name: str = "run",
        model_dir: str = "models/saved",
        log_dir: str = "models/logs",
        policy: str = "MlpPolicy",
        algo_kwargs: dict | None = None,
    ):
        self.run_name = run_name
        self.model_dir = Path(model_dir)
        self.log_dir = Path(log_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        algo_cls = ALGORITHMS.get(algo.lower())
        if algo_cls is None:
            raise ValueError(f"Unknown algorithm '{algo}'. Choose from: {list(ALGORITHMS)}")

        wrapped_env = Monitor(env)
        self.model = algo_cls(
            policy,
            wrapped_env,
            tensorboard_log=str(self.log_dir),
            **(algo_kwargs or {}),
        )

        self.callbacks = []
        if eval_env is not None:
            self.callbacks.append(
                EvalCallback(
                    Monitor(eval_env),
                    best_model_save_path=str(self.model_dir / run_name),
                    log_path=str(self.log_dir / run_name),
                    eval_freq=10_000,
                    deterministic=True,
                    render=False,
                )
            )
        self.callbacks.append(
            CheckpointCallback(
                save_freq=50_000,
                save_path=str(self.model_dir / run_name / "checkpoints"),
                name_prefix=run_name,
            )
        )

    def train(self, total_timesteps: int = 500_000):
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=self.callbacks,
            tb_log_name=self.run_name,
            progress_bar=True,
        )

    def save(self, filename: str | None = None):
        path = self.model_dir / (filename or self.run_name)
        self.model.save(str(path))
        print(f"Model saved to {path}.zip")
        return path

    @classmethod
    def load(cls, path: str, env, algo: str = "ppo"):
        algo_cls = ALGORITHMS[algo.lower()]
        model = algo_cls.load(path, env=Monitor(env))
        instance = cls.__new__(cls)
        instance.model = model
        return instance
