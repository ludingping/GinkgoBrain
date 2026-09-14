"""
H7 portfolio layer — several assets each running `gate_only` (UTC daily close > SMA200,
cash otherwise), capital held at fixed weights with bar-level rebalancing.
Per-asset legs go through SignalLayeredEnv exactly like `backtest_rules.py` (commission,
min_hold), so leg returns already carry costs; the cross-asset rebalance itself is
cost-free here (documented approximation, design doc H7).
Usage:
    uv run python scripts/portfolio_gate_only.py --config config/stage2_4h_signal_v2.yaml \\
        --period pre_val=2021-08-01:2024-10-01 --period val=2024-10-01:2026-04-13 \\
        --period test=2026-04-13: --noharm-periods test --tag h7
Outputs reports/portfolio_gate_only_<tag>.md + .json.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from envs.signal_layered_env import SignalLayeredEnv, load_signal_list        # noqa: E402
from scripts.backtest_rules import (                                          # noqa: E402
    RuleSpec, enrich, evaluate_acceptance, make_act_fn, parse_period,
)
from scripts.backtest_signal_layered import (                                 # noqa: E402
    _ALLOWED_ENV_KEYS, daily_sma_gate, load_data, run_backtest,
)
from utils.metrics import annualised_sharpe, bars_per_year                    # noqa: E402

DEFAULT_SYMBOLS = ("BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT")
# Pre-registered weight variants (design doc H7). Keys are symbol prefixes.
VARIANTS: dict[str, dict[str, float]] = {
    "P-EW4":   {"BTC": 0.25, "ETH": 0.25, "SOL": 0.25, "BNB": 0.25},
    "P-EW2":   {"BTC": 0.5, "ETH": 0.5},
    "P-BTC50": {"BTC": 0.5, "ETH": 1 / 6, "SOL": 1 / 6, "BNB": 1 / 6},
}
BASE = "BTC"


def leg_log_returns(cfg: dict, symbol: str, start: str | None, end: str | None,
                    signal_cols: list[str], env_cfg: dict, prefix_rows: int) -> pd.Series:
    """Per-bar log return of gate_only on one asset, indexed by bar timestamp (UTC)."""
    c = copy.deepcopy(cfg)
    c["crypto"]["symbol"] = symbol
    full_df, bt_df = load_data(c, start, end, prefix_rows=prefix_rows)
    gate = daily_sma_gate(full_df, bt_df)
    env = SignalLayeredEnv(bt_df, signal_cols=signal_cols, **env_cfg)
    steps, _ = run_backtest(make_act_fn(RuleSpec("gate_only"), bt_df, gate), env, bt_df, signal_cols)
    ts = pd.to_datetime([s["timestamp"] for s in steps], utc=True)
    lr = np.array([s["reward_breakdown"]["log_return"] for s in steps], dtype=float)
    return pd.Series(lr, index=ts, name=symbol.split("/")[0])


def combine(legs: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Bar-level rebalanced portfolio log return from per-leg log returns (columns = symbols).
    Weights are renormalised over the columns present; bars missing any leg are dropped."""
    cols = [c for c in weights if c in legs.columns]
    if not cols:
        raise ValueError(f"none of {list(weights)} in legs {list(legs.columns)}")
    w = np.array([weights[c] for c in cols], dtype=float)
    w = w / w.sum()
    simple = np.expm1(legs[cols].dropna().to_numpy(dtype=float))
    port = simple @ w
    return pd.Series(np.log1p(port), index=legs[cols].dropna().index)


def metrics(lr: pd.Series, periods_per_year: int, trades: int = 0) -> dict:
    """Same definitions as backtest_signal_layered.summarise + backtest_rules.enrich."""
    rets = lr.to_numpy(dtype=float)
    pv = np.exp(np.concatenate([[0.0], np.cumsum(rets)]))
    m = {
        "total_return": float(pv[-1] / pv[0] - 1),
        "max_drawdown": float(np.min(pv / np.maximum.accumulate(pv)) - 1),
        "sharpe": annualised_sharpe(rets, periods_per_year),
        "trades": trades, "steps": int(len(rets)), "mean_reward": float("nan"),
    }
    return enrich(m, periods_per_year)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    ap.add_argument("--period", action="append", default=[], help="name=start:end; repeatable")
    ap.add_argument("--noharm-periods", default="")
    ap.add_argument("--tag", default=datetime.now().strftime("%Y%m%d_%H%M"))
    ap.add_argument("--accept-return-ratio", type=float, default=0.8)
    args = ap.parse_args()
    noharm = {p.strip() for p in args.noharm_periods.split(",") if p.strip()}
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    c = cfg["crypto"]
    signal_cols = load_signal_list(cfg["signals"]["config_path"])
    env_cfg = {k: v for k, v in cfg["env"].items() if k in _ALLOWED_ENV_KEYS}
    env_cfg["random_start"] = False
    env_cfg["max_episode_steps"] = None
    prefix_rows = SignalLayeredEnv.warmup_rows(env_cfg.get("window_size", 24))
    ppy = bars_per_year(c.get("timeframe", "4h"))
    periods = [parse_period(p) for p in args.period]
    if not periods:
        print("ERROR: give at least one --period", file=sys.stderr)
        return 1

    out_json: dict = {"config": str(args.config), "symbols": symbols, "variants": VARIANTS, "periods": {}}
    lines = [f"# H7 portfolio gate_only — {args.config.name}", "",
             f"- Generated: `{datetime.now().isoformat(timespec='seconds')}`",
             f"- Legs: {', '.join(symbols)} each `gate_only` (UTC daily close > SMA200); bar-level "
             f"rebalance to fixed weights, rebalance cost not modelled",
             f"- Env per leg: commission={env_cfg.get('commission')}, min_hold={env_cfg.get('min_hold_steps')}; "
             f"Sharpe annualised by {ppy}",
             f"- Acceptance vs BTC gate_only — improve: MDD ≤ 0.67×, return ≥ {args.accept_return_ratio:.0%}×, "
             f"Calmar >; noharm ({', '.join(sorted(noharm)) or '—'}): MDD not deeper, return ≥ {args.accept_return_ratio:.0%}×", ""]
    for name, start, end in periods:
        print(f"\n== period {name}: {start} → {end} ==")
        legs = pd.concat([leg_log_returns(cfg, s, start, end, signal_cols, env_cfg, prefix_rows)
                          for s in symbols], axis=1)
        aligned = legs.dropna()
        print(f"   aligned bars: {len(aligned)} (per leg: {legs.notna().sum().to_dict()})")
        res = {f"{col} gate_only": metrics(aligned[col], ppy) for col in aligned.columns}
        for vname, w in VARIANTS.items():
            res[vname] = metrics(combine(aligned, w), ppy)
        base = res[f"{BASE} gate_only"]
        mode = "noharm" if name in noharm else "improve"
        corr = aligned.corr()
        out_json["periods"][name] = {
            "window": f"{aligned.index[0]} → {aligned.index[-1]}", "mode": mode,
            "results": res, "corr": corr.round(4).to_dict(),
            "acceptance": {v: evaluate_acceptance(res[v], base, mode, args.accept_return_ratio)
                           for v in VARIANTS},
        }
        lines += [f"## {name} — {aligned.index[0]} → {aligned.index[-1]}  ({len(aligned)} bars)", "",
                  "| Strategy | Total | Ann. | Max DD | MDD ratio | Calmar | Sharpe | Accept (" + mode + ") |",
                  "|---|---:|---:|---:|---:|---:|---:|:--|"]
        for k, m in res.items():
            acc = ""
            if k in VARIANTS:
                a = out_json["periods"][name]["acceptance"][k]
                acc = "PASS" if a["pass"] else "FAIL(" + ",".join(x for x, ok in a.items()
                                                                 if x not in ("mode", "pass") and not ok) + ")"
            lines.append(f"| {k} | {m['total_return']:+.2%} | {m['ann_return']:+.2%} | {m['max_drawdown']:.2%} | "
                         f"{abs(m['max_drawdown']) / abs(base['max_drawdown']):.2f} | {m['calmar']:.2f} | "
                         f"{m['sharpe']:.2f} | {acc} |")
        lines += ["", "Pairwise correlation of per-bar gate_only log returns:", "",
                  "| | " + " | ".join(corr.columns) + " |", "|---|" + "---:|" * len(corr.columns)]
        for r in corr.index:
            lines.append(f"| {r} | " + " | ".join(f"{corr.loc[r, cc]:.2f}" for cc in corr.columns) + " |")
        lines.append("")
        print("\n".join(lines[-(len(res) + len(corr) + 8):]))

    rep = REPO_ROOT / "reports"
    rep.mkdir(exist_ok=True)
    (rep / f"portfolio_gate_only_{args.tag}.md").write_text("\n".join(lines), encoding="utf-8")
    (rep / f"portfolio_gate_only_{args.tag}.json").write_text(json.dumps(out_json, indent=2, default=str),
                                                             encoding="utf-8")
    print(f"\nwrote reports/portfolio_gate_only_{args.tag}.md / .json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
