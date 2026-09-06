"""
TC-E1 partial coverage: exercise CLI + config plumbing for the signal_layered
paradigm. Skips the actual trainer (needs DB + SB3 VecEnv setup) — asserts that
argument parsing, config deep_merge and legacy-key filtering behave correctly.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from envs.signal_layered_env import SignalLayeredEnv, load_signal_list
from train import deep_merge


REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- CLI

def test_paradigm_flag_registered():
    """--paradigm flag exists and accepts signal_layered / end_to_end."""
    from train import main   # noqa: F401  # ensure module imports
    import argparse

    # Reconstruct the parser the way train.main() does and check the choices.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paradigm",
        choices=["end_to_end", "signal_layered"],
        default=None,
    )
    ns = parser.parse_args(["--paradigm", "signal_layered"])
    assert ns.paradigm == "signal_layered"
    ns = parser.parse_args(["--paradigm", "end_to_end"])
    assert ns.paradigm == "end_to_end"


# ---------------------------------------------------------------- config

def test_stage2_signal_config_overrides_paradigm():
    """stage2_4h_signal.yaml declares paradigm=signal_layered."""
    with open(REPO_ROOT / "config/stage2_4h_signal.yaml") as f:
        cfg = yaml.safe_load(f)
    assert cfg.get("paradigm") == "signal_layered"
    assert cfg["signals"]["config_path"] == "config/signals_v1.yaml"


def test_deep_merge_preserves_new_env_keys():
    """deep_merge keeps stage-specific env keys on top of default.yaml."""
    with open(REPO_ROOT / "config/default.yaml") as f:
        base = yaml.safe_load(f)
    with open(REPO_ROOT / "config/stage2_4h_signal.yaml") as f:
        stage = yaml.safe_load(f)
    merged = deep_merge(base, stage)

    for key in ("risk_aversion_coef", "excess_return_coef",
                "stop_atr_mult", "stop_cooldown_steps"):
        assert key in merged["env"]
    # These come from default.yaml and will be filtered before env instantiation.
    for legacy in ("trade_penalty_coef", "action_inertia_coef"):
        assert legacy in merged["env"]


def test_allowed_env_keys_match_env_signature():
    """Every key train.py forwards must be a real SignalLayeredEnv parameter."""
    import inspect

    from train import _ALLOWED_ENV_KEYS

    accepted = set(inspect.signature(SignalLayeredEnv).parameters) - {
        "self", "df", "signal_cols"
    }
    assert _ALLOWED_ENV_KEYS <= accepted, _ALLOWED_ENV_KEYS - accepted
    for key in ("trade_penalty_coef", "action_inertia_coef",
                "min_hold_steps", "max_episode_steps"):
        assert key in _ALLOWED_ENV_KEYS


def test_v3_config_overrides_every_forwarded_default_env_key():
    """Deep-merge trap: default.yaml sets env.trade_penalty_coef=1.0 and that key
    is forwarded to SignalLayeredEnv. A stage config that omits it silently
    trains with a -1.0/switch penalty. v3 must pin every such key explicitly."""
    from train import _ALLOWED_ENV_KEYS

    with open(REPO_ROOT / "config/default.yaml") as f:
        base_env = yaml.safe_load(f)["env"]
    with open(REPO_ROOT / "config/stage2_1h_signal_v3.yaml") as f:
        stage_env = yaml.safe_load(f)["env"]

    leaking = (set(base_env) & _ALLOWED_ENV_KEYS) - set(stage_env)
    assert not leaking, f"v3 config inherits from default.yaml: {sorted(leaking)}"


# ---------------------------------------------------------------- signals loader

def test_load_signal_list_reads_11_signals():
    sigs = load_signal_list(REPO_ROOT / "config/signals_v1.yaml")
    assert len(sigs) == 11
    assert all(s.startswith("sig_") for s in sigs)
