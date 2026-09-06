"""
Model-free rule backtester — `gate_only` and `gate × signal` rules through
SignalLayeredEnv (same commission / min_hold / stop as training and Spider).

设计：GinkgoRoad/docs/GinkgoBrain/BTC4h-趋势底座-持仓信号增量-设计.md §4.4 / §6 步 3。
gate_only 是系统底座；每条候选规则必须在 val 与 test 两段都满足 §1.1 的四项验收：
  max DD ≤ 2/3 × gate_only   |   total return ≥ 80 % × gate_only
  Calmar > gate_only         |   ≤ 30 new trades / year

Usage:
    uv run python scripts/backtest_rules.py --config config/stage2_4h_signal_v2.yaml \\
        --period val=2024-10-01:2026-04-13 --period test=2026-04-13: \\
        --rule gate_only \\
        --rule "gate_reduce:signal=sig_funding_current,thr=0.667,level=2" \\
        --tag h1_funding

Rule grammar:  name[:k=v,k=v,...]
    gate_only                       100 % when the UTC daily SMA200 gate is on, else 0 %
    gate_reduce:signal=,thr=,level=[,side=above|below][,signal2=,thr2=,side2=]
                                    gate on & signal (>|<) thr [& signal2 (>|<) thr2] → `level`
                                    (0-4 = 0/25/50/75/100 %), gate on otherwise → 100 %, gate off → 0 %
                                    signals may be pool columns, raw contract columns, probe features
                                    (funding_cum_3d_z, …) or dist_sma200
    const:level=                    fixed target (sanity)
Outputs reports/backtest_rules_<tag>.md + .json (one table per period).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from envs.signal_layered_env import (                                  # noqa: E402
    SignalLayeredEnv, TARGET_POSITION, load_signal_list,
)
from scripts.backtest_signal_layered import (                          # noqa: E402
    _ALLOWED_ENV_KEYS, buy_and_hold_metrics, daily_sma_distance, daily_sma_gate,
    load_data, run_backtest, summarise,
)
from scripts.signal_ic_probe import (                                  # noqa: E402
    DIST_COL, PROBE_FEATURES, add_probe_features, required_sources as probe_required_sources,
)
from utils.data_loader import (                                        # noqa: E402
    CONTRACT_SOURCE_COLS, required_contract_sources,
)
from utils.metrics import bars_per_year                                # noqa: E402

FULL = 4                      # action index for 100 %
FLAT = 0

# §1.1 acceptance thresholds relative to gate_only
ACCEPT_MDD_RATIO = 2.0 / 3.0
ACCEPT_RETURN_RATIO = 0.80
ACCEPT_MAX_TRADES_PER_YEAR = 30.0


# ═══════════════════════════════════════════════════════════════════ rules

@dataclass(frozen=True)
class RuleSpec:
    name: str
    params: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        if not self.params:
            return self.name
        return self.name + ":" + ",".join(f"{k}={v}" for k, v in self.params.items())

    @property
    def signal(self) -> str | None:
        return self.params.get("signal")

    @property
    def signals(self) -> list[str]:
        return [self.params[k] for k in ("signal", "signal2") if k in self.params]


_RULE_NAMES = {"gate_only", "gate_reduce", "const"}


def _coerce(v: str):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def parse_rule(text: str) -> RuleSpec:
    name, _, rest = text.strip().partition(":")
    if name not in _RULE_NAMES:
        raise ValueError(f"unknown rule '{name}'; known: {sorted(_RULE_NAMES)}")
    params: dict = {}
    if rest:
        for kv in rest.split(","):
            k, sep, v = kv.partition("=")
            if not sep or not k or v == "":
                raise ValueError(f"bad rule param '{kv}' in '{text}' (expected k=v)")
            params[k.strip()] = _coerce(v.strip())
    if name == "gate_reduce":
        missing = {"signal", "thr", "level"} - params.keys()
        if missing:
            raise ValueError(f"gate_reduce needs {sorted(missing)}")
        params.setdefault("side", "above")
        if params["side"] not in ("above", "below"):
            raise ValueError("gate_reduce side must be 'above' or 'below'")
        if "signal2" in params or "thr2" in params:
            if {"signal2", "thr2"} - params.keys():
                raise ValueError("gate_reduce AND-condition needs both signal2= and thr2=")
            params.setdefault("side2", "above")
            if params["side2"] not in ("above", "below"):
                raise ValueError("gate_reduce side2 must be 'above' or 'below'")
    if name == "const" and "level" not in params:
        raise ValueError("const needs level=")
    if "level" in params and not (isinstance(params["level"], int) and 0 <= params["level"] <= 4):
        raise ValueError("level must be an int in 0..4")
    return RuleSpec(name, params)


def make_act_fn(spec: RuleSpec, bt_df: pd.DataFrame, gate: np.ndarray):
    """act_fn(obs, env) -> (action, None) for `run_backtest`."""
    if spec.name == "const":
        lvl = spec.params["level"]
        return lambda obs, env: (lvl, None)

    if spec.name == "gate_only":
        return lambda obs, env: ((FULL if gate[env.current_step] else FLAT), None)

    # gate_reduce (optionally AND-ed with a second condition)
    def _cond(sig_key: str, thr_key: str, side_key: str):
        col = spec.params[sig_key]
        if col not in bt_df.columns:
            raise ValueError(f"rule signal '{col}' not in data columns")
        x = bt_df[col].to_numpy(dtype=float)
        thr = float(spec.params[thr_key])
        above = spec.params[side_key] == "above"
        # NaN (e.g. SMA warmup) never satisfies a condition
        return (x > thr) if above else (x < thr)

    hit = _cond("signal", "thr", "side")
    if "signal2" in spec.params:
        hit = hit & _cond("signal2", "thr2", "side2")
    lvl = spec.params["level"]

    def act(obs, env):
        i = env.current_step
        if not gate[i]:
            return FLAT, None
        return (lvl if hit[i] else FULL), None

    return act


def rule_contract_sources(specs: list[RuleSpec]) -> set[str]:
    """Contract data sources referenced by the rules (signal names or raw columns)."""
    raw_to_src = {c: src for src, cols in CONTRACT_SOURCE_COLS.items() for c in cols}
    out: set[str] = set()
    for sp in specs:
        for col in sp.signals:
            if col in raw_to_src:
                out.add(raw_to_src[col])
            out |= probe_required_sources([col])
    return out


def add_rule_features(full_df: pd.DataFrame, bt_df: pd.DataFrame, specs: list[RuleSpec],
                      timeframe: str) -> pd.DataFrame:
    """Materialise probe features / dist_sma200 referenced by the rules onto bt_df."""
    names = [c for sp in specs for c in sp.signals if c in PROBE_FEATURES]
    out = add_probe_features(bt_df, names, timeframe)
    if any(DIST_COL in sp.signals for sp in specs):
        out[DIST_COL] = daily_sma_distance(full_df, bt_df)
    return out


# ═══════════════════════════════════════════════════════════════════ metrics

def enrich(m: dict, periods_per_year: int) -> dict:
    """Add annualised return, Calmar and trades/year to a `summarise` dict."""
    steps = max(int(m["steps"]), 1)
    years = steps / periods_per_year
    gross = 1.0 + m["total_return"]
    ann = gross ** (1.0 / years) - 1.0 if gross > 0 else -1.0
    mdd = abs(m["max_drawdown"])
    return {
        **m,
        "years": years,
        "ann_return": float(ann),
        "calmar": float(ann / mdd) if mdd > 1e-12 else float("inf") if ann > 0 else 0.0,
        "trades_per_year": float(m["trades"] / years),
    }


def evaluate_acceptance(rule: dict, gate: dict) -> dict:
    """§1.1 acceptance of a rule relative to gate_only (both enriched dicts)."""
    checks = {
        "mdd": abs(rule["max_drawdown"]) <= ACCEPT_MDD_RATIO * abs(gate["max_drawdown"]),
        "return": rule["total_return"] >= ACCEPT_RETURN_RATIO * gate["total_return"],
        "calmar": rule["calmar"] > gate["calmar"],
        "turnover": rule["trades_per_year"] <= ACCEPT_MAX_TRADES_PER_YEAR,
    }
    return {**checks, "pass": all(checks.values())}


# ═══════════════════════════════════════════════════════════════════ run

def run_rules_on_slice(
    specs: list[RuleSpec],
    full_df: pd.DataFrame,
    bt_df: pd.DataFrame,
    signal_cols: list[str],
    env_cfg: dict,
    periods_per_year: int,
    timeframe: str = "4h",
) -> dict[str, dict]:
    """Roll every rule (plus buy_and_hold) through the env on one slice."""
    gate = daily_sma_gate(full_df, bt_df)
    bt_df = add_rule_features(full_df, bt_df, specs, timeframe)
    out: dict[str, dict] = {
        "buy_and_hold": enrich(buy_and_hold_metrics(bt_df, signal_cols, env_cfg, periods_per_year),
                               periods_per_year),
    }
    for sp in specs:
        env = SignalLayeredEnv(bt_df, signal_cols=signal_cols, **env_cfg)
        steps, trades = run_backtest(make_act_fn(sp, bt_df, gate), env, bt_df, signal_cols)
        out[sp.label] = enrich(summarise(steps, trades, periods_per_year), periods_per_year)
    return out


def parse_period(text: str) -> tuple[str, str | None, str | None]:
    name, sep, rng = text.partition("=")
    if not sep:
        raise ValueError(f"period must be name=start:end, got '{text}'")
    start, _, end = rng.partition(":")
    return name.strip(), (start.strip() or None), (end.strip() or None)


def default_periods(c: dict) -> list[tuple[str, str | None, str | None]]:
    out = []
    if c.get("val_start"):
        out.append(("val", c["val_start"], c.get("test_start")))
    if c.get("test_start"):
        out.append(("test", c["test_start"], c.get("end_date")))
    return out


def format_table(results: dict[str, dict], gate_label: str = "gate_only") -> list[str]:
    lines = ["| Strategy | Total | Ann. | Max DD | Calmar | Sharpe | Trades | Trades/yr | Accept |",
             "|---|---:|---:|---:|---:|---:|---:|---:|:--|"]
    gate = results.get(gate_label)
    for name, m in results.items():
        if name in ("buy_and_hold", gate_label) or gate is None:
            acc = "—"
        else:
            a = evaluate_acceptance(m, gate)
            acc = "PASS" if a["pass"] else "FAIL(" + ",".join(k for k, v in a.items() if k != "pass" and not v) + ")"
        lines.append(f"| {name} | {m['total_return']:+.2%} | {m['ann_return']:+.2%} | "
                     f"{m['max_drawdown']:.2%} | {m['calmar']:.2f} | {m['sharpe']:.2f} | "
                     f"{m['trades']} | {m['trades_per_year']:.1f} | {acc} |")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--rule", action="append", default=[], help="rule spec; repeatable")
    ap.add_argument("--period", action="append", default=[],
                    help="name=start:end (end optional); repeatable. Default: config val/test windows")
    ap.add_argument("--tag", default=None, help="report name suffix (default: timestamp)")
    ap.add_argument("--with-contracts", action="store_true",
                    help="force the funding/OI/liq merge even if no rule references them")
    args = ap.parse_args()

    specs = [parse_rule(r) for r in (args.rule or ["gate_only"])]
    if not any(sp.name == "gate_only" for sp in specs):
        specs.insert(0, RuleSpec("gate_only"))

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    c = cfg["crypto"]
    signal_cols = load_signal_list(cfg["signals"]["config_path"])
    env_cfg = {k: v for k, v in cfg["env"].items() if k in _ALLOWED_ENV_KEYS}
    env_cfg["random_start"] = False
    env_cfg["max_episode_steps"] = None
    prefix_rows = SignalLayeredEnv.warmup_rows(env_cfg.get("window_size", 24))
    periods_per_year = bars_per_year(c.get("timeframe", "4h"))

    required = rule_contract_sources(specs) | required_contract_sources(signal_cols)
    with_contracts = args.with_contracts or bool(required)

    periods = [parse_period(p) for p in args.period] or default_periods(c)
    if not periods:
        print("ERROR: no --period given and config has no val_start/test_start", file=sys.stderr)
        return 1

    all_results: dict[str, dict] = {}
    lines = [f"# Rule backtest — {args.config.name}", "",
             f"- Generated: `{datetime.now().isoformat(timespec='seconds')}`",
             f"- Rules: " + "; ".join(f"`{sp.label}`" for sp in specs),
             f"- Env: commission={env_cfg.get('commission')}, min_hold={env_cfg.get('min_hold_steps')}, "
             f"stop_atr_mult={env_cfg.get('stop_atr_mult')}; Sharpe annualised by {periods_per_year}",
             f"- Acceptance vs gate_only: MDD ≤ {ACCEPT_MDD_RATIO:.2f}×, return ≥ {ACCEPT_RETURN_RATIO:.0%}×, "
             f"Calmar >, trades/yr ≤ {ACCEPT_MAX_TRADES_PER_YEAR:.0f}", ""]
    for name, start, end in periods:
        print(f"\n== period {name}: {start} → {end} ==")
        full_df, bt_df = load_data(cfg, start, end, with_contracts=with_contracts,
                                   prefix_rows=prefix_rows, required_contracts=required)
        res = run_rules_on_slice(specs, full_df, bt_df, signal_cols, env_cfg, periods_per_year,
                                 timeframe=c.get("timeframe", "4h"))
        label = f"{bt_df['timestamp'].iloc[prefix_rows]} → {bt_df['timestamp'].iloc[-1]}"
        all_results[name] = {"period": label, "results": res}
        lines += [f"## {name} — {label}", ""] + format_table(res) + [""]
        print("\n".join(format_table(res)))

    tag = args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_md = REPO_ROOT / "reports" / f"backtest_rules_{tag}.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    out_md.with_suffix(".json").write_text(
        json.dumps({"config": str(args.config), "rules": [sp.label for sp in specs],
                    "periods": all_results}, indent=2, default=str), encoding="utf-8")
    print(f"\nreport → {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
