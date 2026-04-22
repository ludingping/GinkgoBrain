"""
Backtest a PPO signal-layered model with optional risk overlays.

Replays the model action-by-action through SignalLayeredEnv to collect the
discrete actions, then runs a parallel portfolio simulation that applies:
  - vol-targeting: target_ratio *= min(vol_target / realized_vol, 1.0)
  - regime gate:   target_ratio  = 0 when close < daily SMA
  - deadband:      skip rebalance when |delta| < threshold

This intentionally mirrors run_backtest in backtest_gbdt.py so results between
GBDT and PPO are apples-to-apples.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from stable_baselines3 import PPO                             # noqa: E402
from envs.signal_layered_env import (                         # noqa: E402
    SignalLayeredEnv, TARGET_POSITION, load_signal_list,
)
from scripts.backtest_gbdt import load_data, _BARS_PER_YEAR   # noqa: E402


def collect_actions(model: PPO, env: SignalLayeredEnv) -> tuple[np.ndarray, np.ndarray]:
    """Replay PPO deterministically; return (actions, step_indices into bt_df)."""
    obs, _ = env.reset()
    actions: list[int] = []
    step_indices: list[int] = []
    done = False
    while not done:
        obs_t = torch.as_tensor(obs[None], device=model.policy.device)
        with torch.no_grad():
            dist = model.policy.get_distribution(obs_t)
            probs = dist.distribution.probs.cpu().numpy().squeeze()
        a = int(np.argmax(probs))
        step_indices.append(int(env.current_step))
        actions.append(a)
        obs, _, done, _, _ = env.step(a)
    return np.array(actions, dtype=int), np.array(step_indices, dtype=int)


def simulate(
    closes: np.ndarray,
    raw_ratios: np.ndarray,
    realized_vol: np.ndarray,
    regime_ok: np.ndarray,
    vol_target: float,
    use_regime_gate: bool,
    rebalance_threshold: float,
    commission: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    n = len(closes)
    portfolio = 10_000.0
    pos = 0.0
    equity = np.zeros(n)
    final_ratios = np.zeros(n)
    n_rebalances = 0
    for i in range(n):
        vol_scale = 1.0
        if vol_target > 0.0:
            vol_scale = min(vol_target / max(float(realized_vol[i]), 1e-4), 1.0)
        regime_scale = 1.0 if (not use_regime_gate or regime_ok[i] == 1) else 0.0
        tgt = raw_ratios[i] * vol_scale * regime_scale

        delta = abs(tgt - pos)
        if delta >= rebalance_threshold:
            portfolio -= portfolio * delta * commission
            pos = float(tgt)
            n_rebalances += 1
        final_ratios[i] = pos

        ret = float(closes[i + 1] / closes[i] - 1) if i < n - 1 else 0.0
        portfolio *= 1 + pos * ret
        equity[i] = portfolio
    return equity, final_ratios, n_rebalances


def compute_metrics(equity: np.ndarray, closes: np.ndarray, timeframe: str) -> dict:
    bars_per_year = _BARS_PER_YEAR.get(timeframe, 24 * 365)
    daily = np.diff(equity) / equity[:-1]
    if daily.std() > 1e-10:
        sharpe = float(daily.mean() / daily.std() * np.sqrt(bars_per_year))
    else:
        sharpe = 0.0
    peak = np.maximum.accumulate(equity)
    dd = float(((equity - peak) / (peak + 1e-8)).min())
    return {
        "total_return_pct": round(float(equity[-1] / equity[0] - 1) * 100, 2),
        "buy_hold_return_pct": round(float(closes[-1] / closes[0] - 1) * 100, 2),
        "sharpe": round(sharpe, 4),
        "max_drawdown_pct": round(dd * 100, 2),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--test-start", default=None)
    p.add_argument("--test-end", default=None)
    p.add_argument("--vol-target", type=float, default=0.0)
    p.add_argument("--vol-window", type=int, default=168)
    p.add_argument("--regime-gate", action="store_true")
    p.add_argument("--regime-ma-days", type=int, default=200)
    p.add_argument("--rebalance-threshold", type=float, default=0.01)
    p.add_argument("--commission", type=float, default=0.001)
    args = p.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    timeframe = cfg["crypto"].get("timeframe", "4h").lower()

    signal_cols = load_signal_list(cfg["signals"]["config_path"])
    print(f"Loaded {len(signal_cols)} signals")

    bt_df = load_data(cfg, args.test_start, args.test_end,
                      vol_window=args.vol_window, regime_ma_days=args.regime_ma_days)

    _ALLOWED = {"window_size", "initial_balance", "commission",
                "risk_aversion_coef", "excess_return_coef",
                "stop_atr_mult", "stop_cooldown_steps"}
    env_cfg = {k: v for k, v in cfg["env"].items() if k in _ALLOWED}
    env_cfg["random_start"] = False
    env = SignalLayeredEnv(bt_df, signal_cols, **env_cfg)

    print(f"Loading PPO model from {args.model}")
    model = PPO.load(args.model, device="cpu")
    actions, step_indices = collect_actions(model, env)
    print(f"Collected {len(actions)} actions, steps {step_indices[0]}..{step_indices[-1]}")

    closes = bt_df["close"].values[step_indices]
    rv = bt_df["realized_vol_ann"].values[step_indices]
    reg = bt_df["regime_ok"].values[step_indices]
    raw_ratios = np.array([TARGET_POSITION[int(a)] for a in actions])

    equity, final_ratios, n_rebalances = simulate(
        closes, raw_ratios, rv, reg,
        args.vol_target, args.regime_gate, args.rebalance_threshold, args.commission,
    )
    m = compute_metrics(equity, closes, timeframe)
    m["n_rebalances"] = n_rebalances
    m["mean_position"] = round(float(final_ratios.mean()), 3)
    m["frac_flat"] = round(float((final_ratios == 0).mean()), 3)

    print(f"\n── Backtest metrics ─────────────────────────────────────────")
    for k, v in m.items():
        print(f"  {k:<30} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
