"""
TC-C4: PPO policy feature importance via Integrated Gradients.

Computes, for each of the 5 target actions, the attribution of every
signal (and state) feature to the policy logit — without any extra
library (no SHAP, no captum).

Method — Integrated Gradients (Sundararajan et al., 2017):
  IG_i(x) = (x_i - x'_i) × ∫₀¹ ∂F(x' + α(x-x')) / ∂x_i dα

  where x is an actual observation, x' is the zero baseline (neutral),
  and F is the action logit function of the policy network.
  The integral is approximated by the trapezoidal rule over N_STEPS steps.

Output:
  models/saved/{run_name}/feature_importance.json
    {
      "method": "integrated_gradients",
      "n_samples": 200,
      "features": ["sig_trend_macd_hist", ..., "state_position_ratio", ...],
      "actions": {
          "action_0_0pct": {"sig_trend_macd_hist": 0.034, ...},
          ...
      },
      "overall": {"sig_trend_macd_hist": 0.061, ...}   ← mean |IG| across actions
    }

Usage:
  python scripts/feature_importance.py \\
      --model models/saved/BTCUSDT_ppo_4h_signal/best_model.zip \\
      --config config/stage2_4h_signal.yaml \\
      [--n-samples 200] [--ig-steps 50] [--run-name BTCUSDT_ppo_4h_signal]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agents.ppo_shared import resample_ohlcv                  # noqa: E402
from envs.signal_layered_env import SignalLayeredEnv, load_signal_list  # noqa: E402
from utils.db import read_ohlcv                               # noqa: E402
from utils.indicators import add_indicators                   # noqa: E402
from utils.signals import add_signals                         # noqa: E402

TARGET_POSITION = {0: 0.00, 1: 0.25, 2: 0.50, 3: 0.75, 4: 1.00}


# ─────────────────────────────────────────── Integrated Gradients

def _policy_logits(policy, obs_t: torch.Tensor) -> torch.Tensor:
    """obs_t: (B, W, F) → logits: (B, n_actions)."""
    with torch.enable_grad():
        features = policy.extract_features(obs_t, policy.features_extractor)
        latent_pi = policy.mlp_extractor.forward_actor(features)
        return policy.action_net(latent_pi)


def integrated_gradients(
    policy,
    obs: np.ndarray,          # (W, F) single observation
    n_steps: int = 50,
) -> np.ndarray:              # (n_actions, W, F)
    """
    Compute IG attributions for all 5 actions simultaneously.
    Baseline is the zero observation (all signals neutral, state = 0).
    """
    device = next(policy.parameters()).device
    W, F = obs.shape
    n_actions = 5

    x = torch.tensor(obs, dtype=torch.float32, device=device)          # (W, F)
    baseline = torch.zeros_like(x)
    delta = x - baseline

    # Build interpolation path: (n_steps+1, W, F)
    alphas = torch.linspace(0.0, 1.0, n_steps + 1, device=device)
    path = baseline.unsqueeze(0) + alphas.view(-1, 1, 1) * delta.unsqueeze(0)
    path.requires_grad_(True)

    # Forward all interpolated points in one batch → (n_steps+1, n_actions)
    logits = _policy_logits(policy, path)

    # Sum logits per action (enables one backward per action)
    action_igs = []
    for a in range(n_actions):
        # Compute grad w.r.t. path for this action's logit sum
        if path.grad is not None:
            path.grad.zero_()
        grad = torch.autograd.grad(
            logits[:, a].sum(), path,
            retain_graph=(a < n_actions - 1),
            create_graph=False,
        )[0]                                           # (n_steps+1, W, F)

        # Trapezoidal rule: average neighbouring gradients
        avg_grad = (grad[:-1] + grad[1:]).mean(dim=0)  # (W, F)
        ig = avg_grad * delta                           # (W, F)
        action_igs.append(ig.detach().cpu().numpy())

    return np.stack(action_igs)  # (n_actions, W, F)


# ─────────────────────────────────────────── data + rollout

def collect_observations(
    model,
    bt_df,
    signal_cols: list[str],
    env_cfg: dict,
    n_samples: int,
) -> np.ndarray:
    """
    Run a deterministic rollout on bt_df, collect all obs arrays, then
    return a random subsample of size n_samples.
    """
    from stable_baselines3.common.utils import obs_as_tensor

    env = SignalLayeredEnv(bt_df, signal_cols=signal_cols, **env_cfg)
    obs_arr, _ = env.reset()
    obs_list: list[np.ndarray] = []

    done = False
    while not done:
        obs_list.append(obs_arr.copy())
        obs_t = obs_as_tensor(obs_arr[None], model.policy.device)
        with torch.no_grad():
            action, _ = model.predict(obs_arr, deterministic=True)
        obs_arr, _, done, _, _ = env.step(int(action))

    # Subsample evenly across the episode
    total = len(obs_list)
    if total <= n_samples:
        return np.stack(obs_list)
    idx = np.round(np.linspace(0, total - 1, n_samples)).astype(int)
    return np.stack([obs_list[i] for i in idx])


# ─────────────────────────────────────────── main

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path,
                        default=Path("models/saved/BTCUSDT_ppo_4h_signal/best_model.zip"))
    parser.add_argument("--config", type=Path,
                        default=Path("config/stage2_4h_signal.yaml"))
    parser.add_argument("--test-start", default=None)
    parser.add_argument("--n-samples", type=int, default=200,
                        help="Number of obs to average IG over (more = slower but stabler)")
    parser.add_argument("--ig-steps", type=int, default=50,
                        help="Trapezoidal integration steps (50 is good enough)")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    if not args.model.exists():
        print(f"ERROR: model not found at {args.model}", file=sys.stderr)
        return 1

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    signal_cols = load_signal_list(cfg["signals"]["config_path"])
    state_cols = [
        "state_position_ratio", "state_unrealized_pnl",
        "state_steps_since_stop", "state_cumulative_log_return",
    ]
    feature_names = signal_cols + state_cols
    n_signals = len(signal_cols)
    print(f"Signals: {n_signals}, State features: {len(state_cols)}, "
          f"Total per timestep: {len(feature_names)}")

    # ── Load + resample data
    c = cfg["crypto"]
    tz = c.get("timezone", "Asia/Shanghai")
    start_utc = pd.Timestamp(c["start_date"], tz=tz).tz_convert("UTC").isoformat()
    end_utc = pd.Timestamp(c["end_date"], tz=tz).tz_convert("UTC").isoformat()

    print("Loading data...")
    df_raw = read_ohlcv(c["symbol"], start=start_utc, end=end_utc,
                        table=c.get("db_table", "public.crypto_kline_binance"),
                        only_closed=True)
    ts = df_raw["timestamp"]
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    df_raw["timestamp"] = ts.dt.tz_convert(tz)
    df_tf = resample_ohlcv(df_raw, c.get("timeframe", "4h").lower())
    df = add_indicators(df_tf)
    df = add_signals(df)

    if args.test_start:
        bt_df = df[df["timestamp"] >= pd.Timestamp(args.test_start, tz=tz)].reset_index(drop=True)
    else:
        split = int(len(df) * c.get("train_ratio", 0.75))
        bt_df = df.iloc[split:].reset_index(drop=True)
    print(f"Backtest period: {bt_df['timestamp'].iloc[0]} → {bt_df['timestamp'].iloc[-1]}")

    _ALLOWED = {"window_size", "initial_balance", "commission",
                "risk_aversion_coef", "excess_return_coef",
                "stop_atr_mult", "stop_cooldown_steps"}
    env_cfg = {k: v for k, v in cfg["env"].items() if k in _ALLOWED}
    env_cfg["random_start"] = False

    # ── Load model
    from stable_baselines3 import PPO
    print(f"Loading {args.model}...")
    model = PPO.load(str(args.model))
    policy = model.policy
    policy.eval()
    device = next(policy.parameters()).device
    print(f"Policy on {device}")

    # ── Collect observations
    print(f"Collecting {args.n_samples} observations from deterministic rollout...")
    obs_batch = collect_observations(model, bt_df, signal_cols, env_cfg, args.n_samples)
    print(f"  Got {len(obs_batch)} obs, shape {obs_batch.shape}")

    # ── Run Integrated Gradients
    n_actions = 5
    W, F = obs_batch.shape[1], obs_batch.shape[2]
    # Accumulator: (n_actions, W, F)
    ig_sum = np.zeros((n_actions, W, F), dtype=np.float64)

    print(f"Running IG ({args.ig_steps} steps) on {len(obs_batch)} samples...")
    for i, obs in enumerate(obs_batch):
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(obs_batch)}")
        ig = integrated_gradients(policy, obs, n_steps=args.ig_steps)  # (n_actions, W, F)
        ig_sum += ig

    ig_mean = ig_sum / len(obs_batch)  # (n_actions, W, F)

    # ── Aggregate: sum |IG| over window dimension → (n_actions, F)
    ig_per_feature = np.abs(ig_mean).sum(axis=1)  # (n_actions, F)

    # Normalize each action's importances to sum to 1
    ig_normed = ig_per_feature / (ig_per_feature.sum(axis=1, keepdims=True) + 1e-12)

    # Overall = mean across actions of |IG|
    overall = ig_normed.mean(axis=0)  # (F,)

    # ── Build JSON
    action_labels = {a: f"action_{a}_{int(TARGET_POSITION[a]*100)}pct"
                     for a in range(n_actions)}

    actions_dict = {}
    for a in range(n_actions):
        label = action_labels[a]
        actions_dict[label] = {
            name: round(float(ig_normed[a, i]), 6)
            for i, name in enumerate(feature_names)
        }
        # Sort descending by importance
        actions_dict[label] = dict(
            sorted(actions_dict[label].items(), key=lambda x: -x[1])
        )

    overall_dict = dict(
        sorted(
            {name: round(float(overall[i]), 6)
             for i, name in enumerate(feature_names)}.items(),
            key=lambda x: -x[1]
        )
    )

    result = {
        "method": "integrated_gradients",
        "model": str(args.model),
        "n_samples": len(obs_batch),
        "ig_steps": args.ig_steps,
        "features": feature_names,
        "actions": actions_dict,
        "overall": overall_dict,
    }

    run_name = args.run_name or args.model.parent.name
    out_path = REPO_ROOT / "models" / "saved" / run_name / "feature_importance.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nWrote {out_path}")

    # ── Print summary table
    print("\n── Overall feature importance (top 11 signals) ────────────────")
    print(f"  {'Feature':<35} {'Importance':>10}")
    print(f"  {'-'*35} {'-'*10}")
    for name, val in list(overall_dict.items())[:len(feature_names)]:
        marker = " ◀" if name in signal_cols else ""
        print(f"  {name:<35} {val:>10.4f}{marker}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
