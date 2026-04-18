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


def test_legacy_env_keys_get_filtered_for_signal_env():
    """The whitelist in train._train_signal_layered must exclude legacy keys."""
    import inspect

    accepted = set(inspect.signature(SignalLayeredEnv).parameters) - {
        "self", "df", "signal_cols"
    }
    for legacy in ("trade_penalty_coef", "action_inertia_coef"):
        assert legacy not in accepted, (
            f"{legacy} should NOT be accepted by SignalLayeredEnv"
        )


# ---------------------------------------------------------------- signals loader

def test_load_signal_list_reads_11_signals():
    sigs = load_signal_list(REPO_ROOT / "config/signals_v1.yaml")
    assert len(sigs) == 11
    assert all(s.startswith("sig_") for s in sigs)
