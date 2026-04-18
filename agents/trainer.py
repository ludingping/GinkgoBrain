"""High-level trainer wrapping SB3 algorithms."""
import os
from pathlib import Path

from stable_baselines3 import PPO, A2C, SAC, TD3
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
)
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn


ALGORITHMS = {
    "ppo": PPO,
    "a2c": A2C,
    "sac": SAC,
    "td3": TD3,
}

class TradingCNN(BaseFeaturesExtractor):
    """
    1D-CNN feature extractor for (window_size, n_features) observations.
    Processes the time dimension with causal-style convolutions.
    """

    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_steps, n_features = observation_space.shape

        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        # Compute output dim dynamically
        with torch.no_grad():
            sample = torch.zeros(1, n_features, n_steps)
            cnn_out = self.cnn(sample)
            cnn_flat = cnn_out.flatten(1).shape[1]

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cnn_flat, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (batch, window_size, n_features) → transpose to (batch, n_features, window_size)
        x = obs.permute(0, 2, 1)
        return self.linear(self.cnn(x))


class Trainer:
    """
    Wraps SB3 training loop with sensible defaults for trading environments.

    Usage:
        def make_env():
            return CryptoTradingEnv(...)
        trainer = Trainer(make_env, algo="ppo", run_name="btc_ppo_v1")
        trainer.train(total_timesteps=500_000)
        trainer.save()
    """

    def __init__(
        self,
        env_fn,
        eval_env_fn=None,
        algo: str = "ppo",
        run_name: str = "run",
        model_dir: str = "models/saved",
        log_dir: str = "models/logs",
        policy: str = "MlpPolicy",
        algo_kwargs: dict | None = None,
        n_envs: int = 1,
        normalize_obs: bool = True,
        normalize_reward: bool = False,
        clip_obs: float = 10.0,
    ):
        self.run_name = run_name
        self.model_dir = Path(model_dir)
        self.log_dir = Path(log_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        algo_cls = ALGORITHMS.get(algo.lower())
        if algo_cls is None:
            raise ValueError(f"Unknown algorithm '{algo}'. Choose from: {list(ALGORITHMS)}")

        if n_envs > 1:
            venv = SubprocVecEnv([env_fn] * n_envs, start_method="fork")
        else:
            venv = DummyVecEnv([env_fn])

        if normalize_obs:
            venv = VecNormalize(
                venv,
                norm_obs=True,
                norm_reward=normalize_reward,
                clip_obs=clip_obs,
            )
        self.venv = venv

        self.model = algo_cls(
            policy,
            self.venv,
            tensorboard_log=str(self.log_dir),
            **(algo_kwargs or {}),
        )

        self.callbacks = []
        if eval_env_fn is not None:
            eval_venv = DummyVecEnv([eval_env_fn])
            if normalize_obs:
                eval_venv = VecNormalize(
                    eval_venv,
                    norm_obs=True,
                    norm_reward=False,
                    training=False,    # don't update stats during eval
                )
            self.eval_venv = eval_venv

            # Sync normalization stats from train env → eval env before each eval
            from stable_baselines3.common.callbacks import CallbackList, BaseCallback

            class _SyncNormCallback(BaseCallback):
                """Copy running mean/var from train VecNormalize to eval VecNormalize."""
                def __init__(self, train_venv, eval_venv):
                    super().__init__()
                    self._train = train_venv
                    self._eval = eval_venv

                def _on_step(self) -> bool:
                    if isinstance(self._train, VecNormalize) and isinstance(self._eval, VecNormalize):
                        self._eval.obs_rms = self._train.obs_rms
                        self._eval.ret_rms = self._train.ret_rms
                    return True

            self.callbacks.append(_SyncNormCallback(self.venv, eval_venv))
            self.callbacks.append(
                EvalCallback(
                    self.eval_venv,
                    best_model_save_path=str(self.model_dir / run_name),
                    log_path=str(self.log_dir / run_name),
                    eval_freq=max(10_000 // n_envs, 1),
                    deterministic=True,
                    render=False,
                )
            )
        self.callbacks.append(
            CheckpointCallback(
                save_freq=max(50_000 // n_envs, 1),
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
        if isinstance(self.venv, VecNormalize):
            self.venv.save(str(path) + "_vecnorm.pkl")
        print(f"Model saved to {path}.zip")
        return path

    @classmethod
    def load(cls, path: str, env_fn, algo: str = "ppo", normalize_obs: bool = True):
        algo_cls = ALGORITHMS[algo.lower()]

        # Handle path with or without .zip safely
        base_path = path[:-4] if path.endswith('.zip') else path
        model_path = base_path + ".zip"

        venv = DummyVecEnv([env_fn])
        if normalize_obs:
            vecnorm_path = base_path + "_vecnorm.pkl"
            if os.path.exists(vecnorm_path):
                 venv = VecNormalize.load(vecnorm_path, venv)
                 venv.training = False
                 venv.norm_reward = False

        model = algo_cls.load(model_path, env=venv)
        instance = cls.__new__(cls)
        instance.model = model
        instance.venv = venv
        return instance
