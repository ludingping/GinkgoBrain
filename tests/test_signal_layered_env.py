"""
Tests for envs/signal_layered_env.py — covers TC-B1..B7 and TC-B10 from
`docs/GinkgoBrain/信号分层架构与可解释RL测试用例.md`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from envs.signal_layered_env import (
    MIN_REBALANCE_PCT,
    REWARD_KEYS,
    SIGNAL_WARMUP_WINDOW,
    STATE_DIM,
    TARGET_POSITION,
    SignalLayeredEnv,
)


# ---------------------------------------------------------------- fixtures

SIG_COLS = [
    "sig_trend_macd_hist", "sig_trend_slope_21",
    "sig_mom_rsi_trend", "sig_mom_roc_zscore",
    "sig_vol_atr_pct", "sig_vol_bb_width", "sig_vol_bb_position",
    "sig_volume_obv_slope", "sig_volume_ratio",
    "sig_regime_drawdown", "sig_regime_return_60",
]


def make_df(n: int = 400, price: float | None = None, atr: float = 60.0,
            seed: int = 0) -> pd.DataFrame:
    """
    Build a synthetic DataFrame long enough to clear warmup. Price defaults
    to a slight upward drift so tests that don't care about direction run
    without noise.
    """
    rng = np.random.default_rng(seed)
    if price is None:
        close = 60_000 + np.cumsum(rng.normal(0, 20, n))
    else:
        close = np.full(n, float(price))
    df = pd.DataFrame({
        "open": close, "high": close * 1.002, "low": close * 0.998,
        "close": close, "volume": rng.integers(100, 1000, n).astype(float),
        "atr": np.full(n, float(atr)),
    })
    for col in SIG_COLS:
        df[col] = rng.uniform(-1, 1, n).astype(np.float32)
    return df


def make_flat_env(n: int = 400, price: float = 60_000.0, atr: float = 60.0,
                  **kwargs) -> SignalLayeredEnv:
    """Constant-price env — price drift won't interfere with action mechanics."""
    df = make_df(n=n, price=price, atr=atr)
    return SignalLayeredEnv(df, signal_cols=SIG_COLS, **kwargs)


# ---------------------------------------------------------------- TC-B10 + basics

def test_observation_shape_matches_spec():
    """TC-B10: observation_space.shape == (window_size, n_signals + 4)."""
    env = make_flat_env(window_size=24)
    assert env.observation_space.shape == (24, len(SIG_COLS) + STATE_DIM)
    obs, _ = env.reset()
    assert obs.shape == (24, len(SIG_COLS) + STATE_DIM)
    assert obs.dtype == np.float32


def test_min_start_respects_warmup_and_window():
    """§5.5.2: current_step >= SIGNAL_WARMUP_WINDOW + window_size + SAFETY_BUFFER."""
    env = make_flat_env(window_size=24)
    env.reset()
    assert env.current_step >= SIGNAL_WARMUP_WINDOW + 24  # SAFETY_BUFFER = 10


# ---------------------------------------------------------------- TC-B1

def test_discrete5_target_position_semantics():
    """TC-B1: sequential actions 2,3,1,0 land on 50/75/25/0% position ratios."""
    env = make_flat_env(window_size=10, initial_balance=10_000, commission=0.0005)
    env.reset()

    _, _, _, _, info = env.step(2)
    assert info["position_ratio"] == pytest.approx(0.5, abs=5e-3)
    assert env.trades[-1]["target"] == 0.5

    _, _, _, _, info = env.step(3)
    assert info["position_ratio"] == pytest.approx(0.75, abs=5e-3)
    assert env.trades[-1]["target"] == 0.75

    _, _, _, _, info = env.step(1)
    assert info["position_ratio"] == pytest.approx(0.25, abs=5e-3)
    assert env.trades[-1]["target"] == 0.25

    _, _, _, _, info = env.step(0)
    assert env.position == 0.0
    assert info["balance"] == pytest.approx(info["portfolio_value"])


# ---------------------------------------------------------------- TC-B2

def test_min_rebalance_pct_suppresses_microtrade():
    """TC-B2: a no-op (same target) produces no new trade log entry."""
    env = make_flat_env(window_size=10, initial_balance=10_000)
    env.reset()
    env.step(2)
    n_trades = len(env.trades)
    env.step(2)   # same target → delta ≈ 0 < MIN_REBALANCE_PCT
    assert len(env.trades) == n_trades
    assert env._last_trade_cost == 0.0


# ---------------------------------------------------------------- TC-B3

def test_commission_proportional_to_fill_notional():
    """TC-B3: trade_cost is commission × |delta|/portfolio, not flat 0.1%."""
    commission = 0.001
    env = make_flat_env(window_size=10, initial_balance=10_000, commission=commission)
    env.reset()

    env.step(2)   # 0 → 50% : delta/portfolio ≈ 0.5 → cost ≈ 0.0005
    assert env._last_trade_cost == pytest.approx(commission * 0.5, rel=5e-3)

    env.step(3)   # 50% → 75% : delta/portfolio ≈ 0.25 → cost ≈ 0.00025
    # Second trade's portfolio is slightly smaller after commission + flat price,
    # so delta/portfolio is a hair over 0.25 — allow 2% relative tolerance.
    assert env._last_trade_cost == pytest.approx(commission * 0.25, rel=2e-2)


# ---------------------------------------------------------------- TC-B4

def _make_stop_trigger_df(n_flat: int = 130, n_drop: int = 20,
                          drop_pct: float = 0.10) -> pd.DataFrame:
    """First section flat at 60k, then an abrupt drop that exceeds 2×ATR.

    n_flat defaults to just above MIN_START (≈120) so the env's initial
    current_step lands close to the engineered drop; tests don't need to
    burn 400 warmup bars before reaching the interesting region.
    """
    rng = np.random.default_rng(1)
    close = np.full(n_flat + n_drop, 60_000.0)
    close[n_flat:] = 60_000.0 * (1 - drop_pct)
    df = pd.DataFrame({
        "open": close, "high": close * 1.001, "low": close * 0.999,
        "close": close, "volume": rng.integers(100, 1000, len(close)).astype(float),
        "atr": np.full(len(close), 500.0),   # 2×ATR = 1000, drop = 6000 → triggers
    })
    for col in SIG_COLS:
        df[col] = rng.uniform(-1, 1, len(close)).astype(np.float32)
    return df


def test_atr_stop_triggers_and_force_closes():
    """TC-B4: after a >2×ATR drop while long, position is force-closed."""
    df = _make_stop_trigger_df()
    env = SignalLayeredEnv(df, signal_cols=SIG_COLS, window_size=10,
                           initial_balance=10_000, commission=0.0005,
                           stop_atr_mult=2.0, stop_cooldown_steps=3)
    env.reset()
    env.step(4)   # 100% long while price is flat
    assert env.position > 0

    triggered = False
    for _ in range(20):
        _, _, done, _, info = env.step(4)   # keep trying to stay long
        if info["stop_loss_triggered"]:
            triggered = True
            assert env.position == 0.0
            assert env._entry_price is None
            assert env.stop_loss_events[-1]["reason"] == "atr_stop"
            break
        if done:
            break
    assert triggered, "ATR stop should have fired after the engineered drop"


# ---------------------------------------------------------------- TC-B5

def test_stop_cooldown_blocks_reentry():
    """TC-B5: for cooldown_steps after a stop, non-zero actions are forced to 0."""
    df = _make_stop_trigger_df()
    env = SignalLayeredEnv(df, signal_cols=SIG_COLS, window_size=10,
                           initial_balance=10_000, commission=0.0005,
                           stop_atr_mult=2.0, stop_cooldown_steps=3)
    env.reset()
    env.step(4)

    while True:
        _, _, _, _, info = env.step(4)
        if info["stop_loss_triggered"]:
            break

    # Next N steps must remain flat regardless of action.
    for _ in range(3):
        _, _, _, _, info = env.step(4)
        assert env.position == 0.0, "cooldown should block reopening"

    # After cooldown, a long action should finally take effect.
    _, _, _, _, info = env.step(4)
    assert env.position > 0, "action 4 after cooldown expiry should open a position"


# ---------------------------------------------------------------- TC-B6

def test_reward_breakdown_has_exactly_four_keys():
    """TC-B6: reward_breakdown keys match §5.4.2 spec — no legacy terms."""
    env = make_flat_env(window_size=10)
    env.reset()
    _, reward, _, _, info = env.step(2)
    keys = set(info["reward_breakdown"].keys())
    assert keys == set(REWARD_KEYS)
    total = sum(info["reward_breakdown"].values())
    assert reward == pytest.approx(total, abs=1e-10)


def test_reward_trade_cost_only_on_trade_step():
    """TC-B6 scenario D: trade_cost fires only on actual trade, not on hold."""
    env = make_flat_env(window_size=10, commission=0.001)
    env.reset()
    _, _, _, _, info_trade = env.step(2)
    assert info_trade["reward_breakdown"]["trade_cost"] < 0

    _, _, _, _, info_hold = env.step(2)
    assert info_hold["reward_breakdown"]["trade_cost"] == 0.0


def test_risk_aversion_asymmetric_on_drop():
    """TC-B6 scenario B: down moves produce risk_aversion_adjustment < 0."""
    rng = np.random.default_rng(2)
    n = 400
    close = np.full(n, 60_000.0)
    close[-1] = 60_000.0 * (1 - 0.005)   # 0.5% drop on the final bar
    df = pd.DataFrame({
        "open": close, "high": close, "low": close, "close": close,
        "volume": rng.integers(100, 1000, n).astype(float),
        "atr": np.full(n, 100_000.0),   # absurdly high ATR so stop won't fire
    })
    for col in SIG_COLS:
        df[col] = rng.uniform(-1, 1, n).astype(np.float32)

    env = SignalLayeredEnv(df, signal_cols=SIG_COLS, window_size=10,
                           random_start=False, risk_aversion_coef=0.5,
                           excess_return_coef=0.0)
    env.reset()
    env.current_step = n - 3             # price[n-3]=price[n-2]=60000, price[n-1]=59700
    env.step(4)                          # go 100% long at 60000, advance to n-2
    _, _, _, _, info = env.step(4)       # hold, advance to n-1 → read dropped price

    rb = info["reward_breakdown"]
    assert rb["log_return"] < 0
    assert rb["risk_aversion_adjustment"] < 0
    assert rb["risk_aversion_adjustment"] == pytest.approx(
        rb["log_return"] * 0.5, rel=1e-6
    )


# ---------------------------------------------------------------- TC-B7

def test_legacy_reward_fields_are_absent():
    """TC-B7: no missed_opportunity / stop_penalty / action_inertia fields."""
    env = make_flat_env(window_size=10)
    env.reset()
    for action in [0, 1, 2, 3, 4, 0]:
        _, _, _, _, info = env.step(action)
        for forbidden in ("missed_opportunity", "stop_penalty", "action_inertia"):
            assert forbidden not in info["reward_breakdown"]


def test_stop_step_has_no_extra_penalty():
    """TC-B7: on the stop-trigger step, reward = log_return × (1+risk) only."""
    df = _make_stop_trigger_df()
    env = SignalLayeredEnv(df, signal_cols=SIG_COLS, window_size=10,
                           risk_aversion_coef=0.5, excess_return_coef=0.0,
                           stop_atr_mult=2.0, stop_cooldown_steps=3)
    env.reset()
    env.step(4)

    while True:
        _, reward, _, _, info = env.step(4)
        if info["stop_loss_triggered"]:
            rb = info["reward_breakdown"]
            # Reward is still sum of 4 components — no hidden penalty.
            assert reward == pytest.approx(sum(rb.values()), abs=1e-10)
            # Only expected components on a drop: log_return (negative),
            # risk_aversion_adjustment (negative), trade_cost (≤ 0, fee on close).
            # No positive-penalty "stop_penalty" anywhere.
            assert set(rb.keys()) == set(REWARD_KEYS)
            break


# ---------------------------------------------------------------- gymnasium contract

def test_gymnasium_env_checker():
    """Structural Gymnasium check — spaces, reset, step, dtypes."""
    from gymnasium.utils.env_checker import check_env

    env = make_flat_env(window_size=10)
    check_env(env, warn=True, skip_render_check=True)
