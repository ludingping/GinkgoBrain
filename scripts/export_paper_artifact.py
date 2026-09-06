"""
Export a trained signal-layered PPO run as a paper-trading artifact for
GinkgoSpider (`ginkgo_spider/paper/artifacts/<name>.zip` + `<name>.signals.yaml`).

Writes into <out>/ (default: artifacts/ in this repo; copy to Spider by hand):
  <name>.zip                 best_model.zip of the run (MaskablePPO or PPO)
  <name>.signals.yaml        the elimination yaml the run trained on, plus
                             `artifact:` provenance, `serving:` contract
                             (obs / state features / lock / costs) and a
                             `fingerprint:` block (GinkgoSpider issue #21)
  <name>.golden.json         deterministic synthetic OHLCV input for the fingerprint
  <name>.golden_signals.json expected `add_signals` output on it (last 500 bars)

Serving-side check (issue #21): recompute signals on golden.json with the
serving copy of add_indicators/add_signals and compare to golden_signals.json
with atol = serving.fingerprint.atol. A mismatch means the signal definitions
(or `ta`/numpy behaviour) drifted from what the model was trained on.

Usage:
  uv run python scripts/export_paper_artifact.py --config config/stage2_4h_signal_v2.yaml \\
      --run BTCUSDT_signal_layered_4h_v2 --name btc_4h_ppo_gate_v2
  uv run python scripts/export_paper_artifact.py --verify artifacts/btc_4h_ppo_gate_v2.signals.yaml
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from envs.signal_layered_env import (                                  # noqa: E402
    MIN_REBALANCE_PCT, STATE_DIM, TARGET_POSITION, SignalLayeredEnv, load_signal_list,
)
from utils.indicators import add_indicators                            # noqa: E402
from utils.signals import add_signals                                  # noqa: E402

GOLDEN_SEED = 20260905
GOLDEN_BARS = 1200
GOLDEN_CHECK_ROWS = 500
GOLDEN_START = "2024-01-01T00:00:00Z"
FINGERPRINT_DECIMALS = 9
FINGERPRINT_ATOL = 1e-6

# Signals the paper side cannot compute from its rolling ~500-bar window.
UNSERVABLE_PREFIXES = ("sig_mtf_", "sig_funding_", "sig_oi_", "sig_liq_")

STATE_FEATURES = [
    {"name": "position_ratio",
     "definition": "position * close_t / portfolio_value_t, clipped to [0, 1]"},
    {"name": "unrealized_pnl",
     "definition": "(close_t - entry_price) / entry_price if long else 0, clipped to [-0.5, 0.5]; "
                   "entry_price is the volume-weighted average of buys since last flat"},
    {"name": "steps_since_stop_norm",
     "definition": "min(steps_since_stop, window_size) / window_size; constant 1.0 when the "
                   "ATR stop is disabled (stop_atr_mult <= 0)"},
    {"name": "cum_log_return",
     "definition": "log(portfolio_value_t / initial_balance) accumulated UNCLIPPED across the "
                   "episode, clipped to [-1, 1] only when written into the observation"},
]

OBS_CONTRACT = (
    "obs shape (window_size, n_signals + 4). Row i = bar t-window_size+1+i. The LAST row is "
    "the latest CLOSED bar t: its signals are computed from OHLCV through close[t]; its state "
    "features are evaluated at close[t] with the position held BEFORE this bar's action. The "
    "action chosen from this obs fills at close[t]. Signal columns in `signals` order, then "
    "the 4 state columns in `serving.state_features` order."
)


# ═══════════════════════════════════════════════════════════════════ golden

def make_golden_ohlcv(seed: int = GOLDEN_SEED, n: int = GOLDEN_BARS,
                      timeframe: str = "4h", start: str = GOLDEN_START) -> pd.DataFrame:
    """Deterministic synthetic OHLCV (numpy default_rng, calls in this exact order)."""
    rng = np.random.default_rng(seed)
    log_ret = rng.normal(0.0, 0.01, n)
    close = 30_000.0 * np.exp(np.cumsum(log_ret))
    open_ = np.concatenate([[close[0]], close[:-1]])
    hi_ext = np.abs(rng.normal(0.0, 0.003, n))
    lo_ext = np.abs(rng.normal(0.0, 0.003, n))
    high = np.maximum(open_, close) * (1.0 + hi_ext)
    low = np.minimum(open_, close) * (1.0 - lo_ext)
    volume = rng.lognormal(10.0, 0.5, n)
    ts = pd.date_range(pd.Timestamp(start), periods=n, freq=timeframe, tz="UTC")
    return pd.DataFrame({"timestamp": ts, "open": open_, "high": high,
                         "low": low, "close": close, "volume": volume})


def golden_signals(golden: pd.DataFrame, signal_cols: list[str],
                   n_rows: int = GOLDEN_CHECK_ROWS) -> pd.DataFrame:
    df = add_signals(add_indicators(golden.copy()))
    out = df[["timestamp", *signal_cols]].iloc[-n_rows:].reset_index(drop=True)
    out[signal_cols] = out[signal_cols].astype(float).round(FINGERPRINT_DECIMALS)
    return out


def _df_to_json(df: pd.DataFrame) -> dict:
    cols = {}
    for c in df.columns:
        if c == "timestamp":
            cols[c] = [t.isoformat() for t in pd.DatetimeIndex(df[c]).tz_convert("UTC")]
        else:
            cols[c] = [float(v) for v in df[c].to_numpy()]
    return {"columns": list(df.columns), "n": int(len(df)), "data": cols}


def _json_to_df(obj: dict) -> pd.DataFrame:
    df = pd.DataFrame({c: obj["data"][c] for c in obj["columns"]})
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def fingerprint_sha256(expected: pd.DataFrame, signal_cols: list[str]) -> str:
    payload = expected[signal_cols].to_csv(index=False, float_format=f"%.{FINGERPRINT_DECIMALS}f")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ═══════════════════════════════════════════════════════════════════ helpers

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _model_class_from_zip(path: Path) -> str:
    from scripts.backtest_signal_layered import model_class_from_zip
    return model_class_from_zip(path)


def _git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=REPO_ROOT, text=True).strip()
    except Exception:            # noqa: BLE001 — provenance only
        return "unknown"


def _versions() -> dict:
    import gymnasium, stable_baselines3, torch  # noqa: E401
    try:
        import sb3_contrib
        contrib = sb3_contrib.__version__
    except ImportError:
        contrib = None
    # str(): torch.__version__ is a str subclass PyYAML refuses to represent.
    return {
        "python": platform.python_version(),
        "stable_baselines3": str(stable_baselines3.__version__),
        "sb3_contrib": str(contrib) if contrib else None,
        "torch": str(torch.__version__),
        "gymnasium": str(gymnasium.__version__),
        "numpy": str(np.__version__),
        "pandas": str(pd.__version__),
    }


# ═══════════════════════════════════════════════════════════════════ export

def export(config_path: Path, run: str, name: str, out_dir: Path) -> Path:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    env_cfg = cfg["env"]
    c = cfg["crypto"]
    signals_path = REPO_ROOT / cfg["signals"]["config_path"]
    with open(signals_path, encoding="utf-8") as f:
        signals_yaml = yaml.safe_load(f)
    signal_cols = load_signal_list(signals_path)

    bad = [s for s in signal_cols if s.startswith(UNSERVABLE_PREFIXES)]
    if bad:
        raise SystemExit(f"refusing to export: serving cannot compute {bad}")

    run_dir = REPO_ROOT / "models" / "saved" / run
    model_src = run_dir / "best_model.zip"
    if not model_src.exists():
        raise SystemExit(f"model not found: {model_src}")
    model_class = _model_class_from_zip(model_src)
    if bool(cfg["training"].get("action_masking")) != (model_class == "MaskablePPO"):
        raise SystemExit(f"config action_masking={cfg['training'].get('action_masking')} "
                         f"but zip policy is {model_class}")

    out_dir.mkdir(parents=True, exist_ok=True)
    model_dst = out_dir / f"{name}.zip"
    shutil.copyfile(model_src, model_dst)

    golden = make_golden_ohlcv(timeframe=c.get("timeframe", "4h"))
    expected = golden_signals(golden, signal_cols)
    (out_dir / f"{name}.golden.json").write_text(json.dumps(_df_to_json(golden)), encoding="utf-8")
    (out_dir / f"{name}.golden_signals.json").write_text(
        json.dumps(_df_to_json(expected)), encoding="utf-8")

    window = int(env_cfg["window_size"])
    stop_mult = float(env_cfg.get("stop_atr_mult", 2.0))
    payload = dict(signals_yaml)   # version / generated_at / source / thresholds / signals / eliminated
    payload["artifact"] = {
        "name": name,
        "exported_at": dt.datetime.now().isoformat(timespec="seconds"),
        "model_zip": model_dst.name,
        "model_sha256": _sha256_file(model_dst),
        "source_run": run,
        "training_config": str(config_path),
        "ginkgobrain_commit": _git_rev(),
        "versions": _versions(),
    }
    payload["serving"] = {
        "model_class": model_class,
        "load": ("sb3_contrib.MaskablePPO.load(zip); policy.get_distribution(obs, action_masks=mask)"
                 if model_class == "MaskablePPO" else "stable_baselines3.PPO.load(zip)"),
        "symbol": c["symbol"],
        "timeframe": c.get("timeframe", "4h"),
        "bar_alignment": "UTC-aligned buckets (training tz Asia/Shanghai; 8 mod 4 == 0)",
        "window_size": window,
        "n_signals": len(signal_cols),
        "state_dim": STATE_DIM,
        "obs_shape": [window, len(signal_cols) + STATE_DIM],
        "obs_contract": OBS_CONTRACT,
        "state_features": STATE_FEATURES,
        "action_space": {int(k): float(v) for k, v in TARGET_POSITION.items()},
        "min_hold_steps": int(env_cfg.get("min_hold_steps", 0)),
        "lock_rule": ("after an executed switch to a new target level, the action is frozen at "
                      "that level for min_hold_steps bars (mask = one-hot of the last executed "
                      "level). Persist last_executed_level and bars_since_change. Apply any "
                      "serving-side risk gate AFTER the lock so the gate can always flatten."),
        "min_rebalance_pct": MIN_REBALANCE_PCT,
        "commission": float(env_cfg["commission"]),
        "stop_loss": ("none" if stop_mult <= 0 else
                      f"ATR x {stop_mult}, cooldown {env_cfg.get('stop_cooldown_steps')} bars"),
        "min_bars": max(500, SignalLayeredEnv.warmup_rows(window) + window),
        "episode_note": ("training episodes are max_episode_steps="
                         f"{env_cfg.get('max_episode_steps')} bars with cum_log_return reset to 0; "
                         "serving should reset cum_log_return on a comparable cadence or accept drift"),
    }
    payload["fingerprint"] = {
        "method": "ginkgo_golden_v1",
        "golden_input": f"{name}.golden.json",
        "expected_output": f"{name}.golden_signals.json",
        "golden_spec": {"seed": GOLDEN_SEED, "n_bars": GOLDEN_BARS, "start": GOLDEN_START,
                        "timeframe": c.get("timeframe", "4h"),
                        "generator": "scripts/export_paper_artifact.py::make_golden_ohlcv"},
        "check_rows": GOLDEN_CHECK_ROWS,
        "round_decimals": FINGERPRINT_DECIMALS,
        "atol": FINGERPRINT_ATOL,
        "sha256": fingerprint_sha256(expected, signal_cols),
    }

    yaml_dst = out_dir / f"{name}.signals.yaml"
    with open(yaml_dst, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True, width=100)
    for p in (model_dst, yaml_dst, out_dir / f"{name}.golden.json",
              out_dir / f"{name}.golden_signals.json"):
        print(f"Wrote {p}")
    return yaml_dst


def verify(yaml_path: Path) -> int:
    with open(yaml_path, encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    fp = payload["fingerprint"]
    signal_cols = list(payload["signals"])
    golden = _json_to_df(json.loads((yaml_path.parent / fp["golden_input"]).read_text()))
    expected = _json_to_df(json.loads((yaml_path.parent / fp["expected_output"]).read_text()))
    actual = golden_signals(golden, signal_cols, n_rows=int(fp["check_rows"]))
    diff = (actual[signal_cols].to_numpy() - expected[signal_cols].to_numpy())
    worst = float(np.nanmax(np.abs(diff)))
    sha_ok = fingerprint_sha256(actual, signal_cols) == fp["sha256"]
    print(f"[verify] max |Δ| = {worst:.3e} (atol {fp['atol']}); sha256 match = {sha_ok}")
    if worst > float(fp["atol"]):
        print("[verify] FAILED — signal definitions drifted from the exported artifact",
              file=sys.stderr)
        return 2
    print("[verify] OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, help="training config the run used")
    ap.add_argument("--run", help="run name under models/saved/")
    ap.add_argument("--name", help="artifact name, e.g. btc_4h_ppo_gate_v2")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "artifacts")
    ap.add_argument("--verify", type=Path, metavar="SIGNALS_YAML",
                    help="recompute the golden fingerprint with the current code and compare")
    args = ap.parse_args()

    if args.verify:
        return verify(args.verify)
    if not (args.config and args.run and args.name):
        ap.error("--config, --run and --name are required unless --verify is given")
    export(args.config, args.run, args.name, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
