"""
Export a rule-only paper-trading contract (`<name>.rules.yaml`) for GinkgoSpider.

Rule family `gate_exit_sma` (design doc §5 H3'b, V1):
    regime gate : last closed UTC daily close  >  SMA(gate_days)          (strict, = overlay.compute_regime_ok)
    exit        : flat when daily close  <  SMA(exit_days); equality keeps (= Brain dist_sma50 < 0 → level 0)
    target      : 1.0 when gate on and not exited, else 0.0
With `--exit-days 0` the contract is the pure `gate_only` control arm.

The contract carries a golden fingerprint: a deterministic synthetic daily close
series and the expected (regime_ok, exit_ok, target) rows. Spider recomputes the
rows with its own implementation at seed time and refuses to start on mismatch, so
the two repos cannot drift on boundary conventions.

Usage:
    uv run python scripts/export_rule_artifact.py --name btc_4h_gate_sma50_v1 --gate-days 200 --exit-days 50 \\
        --backtest-json reports/backtest_rules_h3b_exit_variants.json \\
        --backtest-json reports/backtest_rules_h3b_v1_confirm_pre_val.json \\
        --rule-label "gate_reduce:signal=dist_sma50,thr=0,level=0,side=below"
    uv run python scripts/export_rule_artifact.py --name btc_4h_gate_only --gate-days 200 --exit-days 0
    uv run python scripts/export_rule_artifact.py --verify artifacts/btc_4h_gate_sma50_v1.rules.yaml
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

GOLDEN_SEED = 20260907
GOLDEN_DAYS = 320
GOLDEN_START = "2024-01-01"
CHECK_ROWS = 60                 # trailing rows compared (gate + exit both flip inside them)
RULE_KIND = "gate_exit_sma"
RULE_KIND_VOL = "gate_vol_target"     # H6 A/B arm: gate × volatility-target sizing
DAYS_PER_YEAR = 365
LEVELS = 4                            # 0 / 25 / 50 / 75 / 100 %
FINGERPRINT_METHOD = "ginkgo_rule_golden_v1"
MIN_REBALANCE_PCT = 0.02
COMMISSION = 0.001
MIN_DAILY_BARS = 300


# ═══════════════════════════════════════════════════════════════════ rule

def rule_decisions(closes: pd.Series, gate_days: int, exit_days: int | None) -> pd.DataFrame:
    """Per-day (regime_ok, exit_ok, target) for the gate_exit_sma rule.

    Conventions (shared with Spider `overlay.rule_decisions`):
      regime_ok = close > SMA(gate_days)             strict; NaN SMA → False
      exit_ok   = close >= SMA(exit_days)            equality keeps the position; NaN SMA → False;
                                                     True everywhere when exit_days is falsy
      target    = 1.0 if regime_ok and exit_ok else 0.0
    """
    c = pd.to_numeric(closes, errors="coerce").astype(float)
    sma_g = c.rolling(int(gate_days)).mean()
    regime_ok = (c > sma_g) & sma_g.notna()
    if exit_days:
        sma_e = c.rolling(int(exit_days)).mean()
        exit_ok = (c >= sma_e) & sma_e.notna()
    else:
        exit_ok = pd.Series(True, index=c.index)
    target = np.where(regime_ok & exit_ok, 1.0, 0.0)
    return pd.DataFrame({"regime_ok": regime_ok.to_numpy(), "exit_ok": exit_ok.to_numpy(),
                         "target": target}, index=c.index)


def vol_rule_decisions(closes: pd.Series, gate_days: int, vol_window: int, vol_target: float,
                       floor_level: int = 1) -> pd.DataFrame:
    """Per-day (regime_ok, exit_ok, target) for the gate_vol_target rule.

    Conventions (shared with Spider `overlay.vol_rule_decisions` and identical to
    Brain `backtest_rules.vol_target_level` + `daily_realized_vol`):
      rvol      = std(Δlog close over the last vol_window days, ddof=1) × √365
      w         = min(1, vol_target / rvol);  rvol NaN or ≤ 0 (warmup) → level 4
      level     = floor(4·w + 0.5) clamped to [floor_level, 4]
      target    = level / 4 if regime_ok (close > SMA(gate_days), strict) else 0.0
      exit_ok   = True everywhere (no exit rule; kept so fingerprint rows share one schema)
    """
    c = pd.to_numeric(closes, errors="coerce").astype(float)
    sma_g = c.rolling(int(gate_days)).mean()
    regime_ok = (c > sma_g) & sma_g.notna()
    rvol = np.log(c).diff().rolling(int(vol_window)).std() * np.sqrt(DAYS_PER_YEAR)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = np.where(rvol.to_numpy() > 0, LEVELS * np.minimum(1.0, float(vol_target) / rvol.to_numpy()), np.nan)
    level = np.where(np.isfinite(raw), np.floor(raw + 0.5), LEVELS)
    level = np.clip(level, int(floor_level), LEVELS)
    target = np.where(regime_ok.to_numpy(), level / LEVELS, 0.0)
    return pd.DataFrame({"regime_ok": regime_ok.to_numpy(), "exit_ok": np.ones(len(c), dtype=bool),
                         "target": target}, index=c.index)


# ═══════════════════════════════════════════════════════════════════ golden

def make_golden_daily_closes(seed: int = GOLDEN_SEED, n: int = GOLDEN_DAYS,
                             start: str = GOLDEN_START) -> pd.Series:
    """Deterministic daily closes shaped so the trailing CHECK_ROWS days contain all
    three rule states: long (gate on & above SMA50), exited (gate on & below SMA50),
    and gate off (below SMA200). Piecewise path + tiny seeded noise (no exact ties)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    path = np.empty(n)
    path[:200] = 100.0                                              # flat: SMA200 ≈ 100
    path[200:275] = np.linspace(100.0, 135.0, 75)                  # ramp: gate on, above SMA50
    path[275:295] = np.linspace(135.0, 112.0, 20)                  # dip below SMA50, still > SMA200
    path[295:] = np.linspace(112.0, 80.0, n - 295)                  # crash: gate off
    noise = np.exp(rng.normal(0.0, 0.004, n))
    close = path * noise
    idx = pd.date_range(start, periods=n, freq="1D", tz="UTC")
    return pd.Series(np.round(close, 6), index=idx, name="close")


def make_golden_daily_closes_vol(seed: int = GOLDEN_SEED, n: int = GOLDEN_DAYS,
                                 start: str = GOLDEN_START) -> pd.Series:
    """Deterministic daily closes for the gate_vol_target fingerprint: the trailing
    CHECK_ROWS days contain a calm up-trend (level 4), a violent chop above SMA200
    (level 1–2), a medium-vol drift (level 2–3) and a crash below SMA200 (target 0)."""
    rng = np.random.default_rng(seed + 1)
    path = np.empty(n)
    path[:200] = 100.0
    path[200:262] = np.linspace(100.0, 140.0, 62)                  # calm ramp → level 4
    path[262:290] = 140.0                                           # chop segment (noise below)
    path[290:306] = np.linspace(140.0, 118.0, 16)                   # medium-vol drift
    path[306:] = np.linspace(118.0, 70.0, n - 306)                  # crash → gate off
    sigma = np.full(n, 0.004)
    sigma[262:290] = 0.06
    sigma[290:306] = 0.025
    close = path * np.exp(rng.normal(0.0, 1.0, n) * sigma)
    idx = pd.date_range(start, periods=n, freq="1D", tz="UTC")
    return pd.Series(np.round(close, 6), index=idx, name="close")


def expected_rows(decisions: pd.DataFrame, check_rows: int = CHECK_ROWS) -> list[dict]:
    tail = decisions.iloc[-check_rows:]
    return [{"day": ts.strftime("%Y-%m-%d"), "regime_ok": bool(r.regime_ok),
             "exit_ok": bool(r.exit_ok), "target": float(r.target)}
            for ts, r in tail.iterrows()]


def fingerprint_sha256(rows: list[dict]) -> str:
    blob = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


# ═══════════════════════════════════════════════════════════════════ export / verify

def _git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                                       text=True).strip()
    except Exception:
        return "unknown"


def _backtest_summary(json_paths: list[Path], rule_label: str | None) -> dict:
    out: dict = {}
    for jp in json_paths:
        d = json.loads(jp.read_text(encoding="utf-8"))
        for period, blk in d.get("periods", {}).items():
            res = blk["results"]
            entry = {"window": blk.get("period"), "gate_only": _slim(res.get("gate_only"))}
            if rule_label and rule_label in res:
                entry["rule"] = _slim(res[rule_label])
            out[f"{jp.name}:{period}"] = entry
    return out


def _slim(m: dict | None) -> dict | None:
    if not m:
        return None
    keys = ("total_return", "max_drawdown", "calmar", "sharpe", "trades", "trades_per_year")
    return {k: (round(float(m[k]), 4) if isinstance(m[k], (int, float)) else m[k]) for k in keys if k in m}


def build_payload(name: str, gate_days: int, exit_days: int | None, *,
                  symbol: str = "BTC/USDT", timeframe: str = "4h",
                  backtest: dict | None = None,
                  vol_target: float | None = None, vol_window: int | None = None,
                  floor_level: int = 1) -> dict:
    is_vol = vol_target is not None
    if is_vol:
        if exit_days:
            raise ValueError("gate_vol_target has no exit rule; drop --exit-days")
        if not vol_window or vol_window < 2:
            raise ValueError("gate_vol_target needs --vol-window ≥ 2 (days)")
        golden = make_golden_daily_closes_vol()
        dec = vol_rule_decisions(golden, gate_days, vol_window, vol_target, floor_level)
        golden_gen = "scripts/export_rule_artifact.py::make_golden_daily_closes_vol"
        design = "GinkgoRoad/docs/GinkgoBrain/BTC4h-趋势底座-持仓信号增量-设计.md H6（A/B 臂，未过 §1.1）"
    else:
        golden = make_golden_daily_closes()
        dec = rule_decisions(golden, gate_days, exit_days)
        golden_gen = "scripts/export_rule_artifact.py::make_golden_daily_closes"
        design = "GinkgoRoad/docs/GinkgoBrain/BTC4h-趋势底座-持仓信号增量-设计.md §5 H3'b"
    rows = expected_rows(dec)
    exit_block = ({"ma_days": int(exit_days),
                   "condition": "flat (target 0) when daily close < SMA(ma_days); equality keeps the position"}
                  if exit_days else {"ma_days": None, "condition": "none (pure regime gate)"})
    vol_block = ({"window_days": int(vol_window), "target_ann": float(vol_target),
                  "floor_level": int(floor_level), "levels": LEVELS,
                  "rvol": "std(Δlog daily close over window_days, ddof=1) × sqrt(365); NaN/≤0 → level 4",
                  "level": "floor(4·min(1, target_ann/rvol) + 0.5) clamped to [floor_level, 4]",
                  "target": "level/4 when gate on, 0 when gate off"}
                 if is_vol else None)
    return {
        "version": 1,
        "artifact": {
            "name": name, "kind": "rule", "exported_at": dt.datetime.now().isoformat(timespec="seconds"),
            "ginkgobrain_commit": _git_rev(),
            "design_doc": design,
            "generator": "scripts/export_rule_artifact.py",
        },
        "rule": {
            "kind": RULE_KIND_VOL if is_vol else RULE_KIND, "symbol": symbol, "timeframe": timeframe,
            "daily_bars": "UTC 1d klines; use only the last fully closed day (open_time + 1d <= now)",
            "decision_cadence": f"every closed {timeframe} bar; the daily inputs are those of the last closed UTC day",
            "gate": {"ma_days": int(gate_days), "condition": "daily close > SMA(ma_days) (strict)"},
            "exit": exit_block,
            **({"vol": vol_block} if is_vol else {}),
            "target": ({"on": "level/4", "off": 0.0} if is_vol else {"on": 1.0, "off": 0.0}),
            "min_rebalance_pct": MIN_REBALANCE_PCT,
            "commission": COMMISSION,
            "min_daily_bars": MIN_DAILY_BARS,
            "no_model": True,
        },
        "backtest": backtest or {},
        "fingerprint": {
            "method": FINGERPRINT_METHOD,
            "golden_spec": {"seed": GOLDEN_SEED, "n_days": GOLDEN_DAYS, "start": GOLDEN_START,
                            "generator": golden_gen},
            "golden_closes": [float(x) for x in golden.to_numpy()],
            "check_rows": CHECK_ROWS,
            "expected": rows,
            "sha256": fingerprint_sha256(rows),
        },
    }


def verify_payload(payload: dict) -> list[str]:
    """Recompute the golden decisions from the embedded closes; return mismatch messages."""
    fp, rule = payload["fingerprint"], payload["rule"]
    closes = pd.Series(fp["golden_closes"], index=pd.date_range(fp["golden_spec"]["start"],
                       periods=len(fp["golden_closes"]), freq="1D", tz="UTC"))
    if rule.get("kind") == RULE_KIND_VOL:
        v = rule["vol"]
        dec = vol_rule_decisions(closes, rule["gate"]["ma_days"], v["window_days"], v["target_ann"],
                                 v.get("floor_level", 1))
    else:
        dec = rule_decisions(closes, rule["gate"]["ma_days"], rule["exit"]["ma_days"])
    rows = expected_rows(dec, fp["check_rows"])
    problems = []
    if rows != fp["expected"]:
        bad = [i for i, (a, b) in enumerate(zip(rows, fp["expected"])) if a != b]
        problems.append(f"{len(bad)} of {len(rows)} golden rows differ (first at index {bad[0] if bad else '?'})")
    if fingerprint_sha256(fp["expected"]) != fp["sha256"]:
        problems.append("sha256 does not match the embedded expected rows")
    targets = {r["target"] for r in fp["expected"]}
    if len(targets) < 2:
        problems.append("golden window never switches target — fingerprint would not catch a flipped comparison")
    if rule.get("kind") == RULE_KIND_VOL and len(targets - {0.0, 1.0}) == 0:
        problems.append("golden window has no intermediate vol-target level — sizing formula is not exercised")
    return problems


def export(name: str, gate_days: int, exit_days: int | None, out_dir: Path,
           backtest: dict | None = None, **kind_kwargs) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = build_payload(name, gate_days, exit_days, backtest=backtest, **kind_kwargs)
    problems = verify_payload(payload)
    if problems:
        raise SystemExit("self-verification failed: " + "; ".join(problems))
    dst = out_dir / f"{name}.rules.yaml"
    with open(dst, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True, width=100)
    print(f"Wrote {dst}")
    return dst


def verify(path: Path) -> int:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    problems = verify_payload(payload)
    if problems:
        print(f"FAIL {path}: " + "; ".join(problems))
        return 1
    extra = (f"vol={payload['rule']['vol']['target_ann']}/{payload['rule']['vol']['window_days']}d"
             if payload['rule'].get('kind') == RULE_KIND_VOL else f"exit={payload['rule']['exit']['ma_days']}")
    print(f"OK {path}: rule={payload['rule']['kind']} gate={payload['rule']['gate']['ma_days']} "
          f"{extra} sha256={payload['fingerprint']['sha256'][:12]}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name")
    ap.add_argument("--gate-days", type=int, default=200)
    ap.add_argument("--exit-days", type=int, default=0, help="0 = no exit (pure gate)")
    ap.add_argument("--vol-target", type=float, default=None,
                    help="annualised vol target → kind gate_vol_target (e.g. 0.30)")
    ap.add_argument("--vol-window", type=int, default=None, help="realised-vol window in days (e.g. 30)")
    ap.add_argument("--floor-level", type=int, default=1, help="minimum level while gate on (default 1 = 25%%)")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "artifacts")
    ap.add_argument("--backtest-json", type=Path, action="append", default=[])
    ap.add_argument("--rule-label", default=None, help="rule label inside the backtest json to embed")
    ap.add_argument("--verify", type=Path, metavar="RULES_YAML")
    args = ap.parse_args()
    if args.verify:
        return verify(args.verify)
    if not args.name:
        ap.error("--name is required unless --verify is given")
    bt = _backtest_summary(args.backtest_json, args.rule_label) if args.backtest_json else None
    kind_kwargs = ({"vol_target": args.vol_target, "vol_window": args.vol_window,
                    "floor_level": args.floor_level} if args.vol_target is not None else {})
    export(args.name, args.gate_days, args.exit_days or None, args.out, backtest=bt, **kind_kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
