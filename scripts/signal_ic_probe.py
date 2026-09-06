"""
Single-signal IC probe — the L1/L2 falsification test of the design doc
(GinkgoRoad/docs/GinkgoBrain/BTC4h-趋势底座-持仓信号增量-设计.md §3 / §5).

For every (signal, horizon) it reports, on each expanding time fold's *validation
window* (same folds as signal_linear_baseline_pooled, so numbers are comparable):
    rank IC      Spearman(signal_t, fwd log return_{t→t+k})
    decile       mean fwd return of top-10 % signal bars minus bottom-10 % (bps)
    AUC          roc_auc(fwd return > 0, signal)   (< 0.5 ⇒ inverse relation)
No model is fitted — IC is model-free — so folds only test regime stability.

Pass rule (§5): |IC| ≥ IC_MIN on every fold, all folds same sign, and the most
recent fold agrees in sign.  Overlapping k-bar targets inflate t-stats; the
thresholds are on IC magnitude, not p-values.

Probe-only derived features (from the raw `funding_rate` column) live in
PROBE_FEATURES; promote one to utils/signals.py (value contract [-1, 1]) only
if it passes.

Usage:
    uv run python scripts/signal_ic_probe.py --config config/stage2_4h_signal_v2.yaml \\
        --signal sig_funding_current --signal sig_funding_trend \\
        --signal funding_cum_3d --signal funding_cum_7d --signal funding_rank_90d \\
        --horizon 6 --horizon 18 --tag h1_funding
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.backtest_signal_layered import daily_sma_gate               # noqa: E402
from scripts.signal_linear_baseline import load_df_from_config           # noqa: E402
from utils.data_loader import required_contract_sources                  # noqa: E402
from utils.indicators import add_indicators                              # noqa: E402
from utils.signals import add_contract_signals, add_signals              # noqa: E402
from utils.splits import TimeFold, time_folds                            # noqa: E402

IC_MIN = 0.03
TS_COL = "timestamp"
DECILE = 0.10


# ═══════════════════════════════════════════════════════════════════ features

def _bars(span: str, timeframe: str) -> int:
    return max(1, int(pd.Timedelta(span) / pd.Timedelta(timeframe)))


def funding_cum(df: pd.DataFrame, timeframe: str, span: str) -> pd.Series:
    """Rolling sum of the (predicted) funding rate over `span` — raw scale."""
    fr = pd.to_numeric(df["funding_rate"], errors="coerce")
    return fr.rolling(_bars(span, timeframe), min_periods=1).sum()


def funding_rank(df: pd.DataFrame, timeframe: str, span: str) -> pd.Series:
    """Rolling percentile rank of the funding rate over `span`, in [0, 1]."""
    fr = pd.to_numeric(df["funding_rate"], errors="coerce")
    w = _bars(span, timeframe)
    return fr.rolling(w, min_periods=max(2, w // 4)).rank(pct=True)


PROBE_FEATURES = {
    "funding_cum_3d": lambda df, tf: funding_cum(df, tf, "3D"),
    "funding_cum_7d": lambda df, tf: funding_cum(df, tf, "7D"),
    "funding_rank_90d": lambda df, tf: funding_rank(df, tf, "90D"),
}
_PROBE_SOURCE = {name: "funding" for name in PROBE_FEATURES}


def add_probe_features(df: pd.DataFrame, names: list[str], timeframe: str) -> pd.DataFrame:
    out = df.copy()
    for n in names:
        if n in PROBE_FEATURES:
            if "funding_rate" not in out.columns:
                raise ValueError(f"probe feature '{n}' needs a funding_rate column")
            out[n] = PROBE_FEATURES[n](out, timeframe)
    return out


def required_sources(signals: list[str]) -> set[str]:
    out = required_contract_sources(signals)
    out |= {_PROBE_SOURCE[s] for s in signals if s in _PROBE_SOURCE}
    return out


# ═══════════════════════════════════════════════════════════════════ metrics

def forward_log_return(close: pd.Series, horizon: int) -> pd.Series:
    return np.log(close.shift(-horizon) / close)


def rank_ic(signal: pd.Series, fwd: pd.Series) -> float:
    m = signal.notna() & fwd.notna()
    if m.sum() < 10 or signal[m].nunique() < 2:
        return float("nan")
    return float(spearmanr(signal[m], fwd[m]).correlation)


def decile_spread_bps(signal: pd.Series, fwd: pd.Series, q: float = DECILE) -> float:
    m = signal.notna() & fwd.notna()
    s, f = signal[m], fwd[m]
    if len(s) < 20:
        return float("nan")
    hi, lo = s.quantile(1 - q), s.quantile(q)
    return float((f[s >= hi].mean() - f[s <= lo].mean()) * 1e4)


def auc(signal: pd.Series, fwd: pd.Series) -> float:
    m = signal.notna() & fwd.notna()
    y = (fwd[m] > 0).astype(int)
    if y.nunique() < 2 or signal[m].nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y, signal[m]))


@dataclass(frozen=True)
class FoldStat:
    fold: int
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    n: int
    ic: float
    spread_bps: float
    auc: float


def verdict(ics: list[float], ic_min: float = IC_MIN) -> tuple[bool, str]:
    """(pass, reason) per §5: |IC| ≥ ic_min on all folds, same sign, last fold agrees."""
    if not ics or any(np.isnan(ics)):
        return False, "nan"
    signs = {np.sign(x) for x in ics}
    if len(signs) != 1 or 0 in signs:
        return False, "sign flips"
    weak = [k + 1 for k, x in enumerate(ics) if abs(x) < ic_min]
    if weak:
        return False, f"|IC|<{ic_min} on fold {weak}"
    return True, "ok"


def evaluate_signal(
    df: pd.DataFrame, signal: str, horizon: int, folds: list[TimeFold],
    mask: np.ndarray | None = None,
) -> list[FoldStat]:
    """Per-fold stats on the validation window; `mask` (bool per row) restricts the
    rows evaluated — e.g. gate-on bars only, since a gate × signal rule never acts
    while the gate is off."""
    fwd = forward_log_return(df["close"], horizon)
    ts = df[TS_COL]
    keep = pd.Series(True if mask is None else mask, index=df.index)
    out = []
    for f in folds:
        m = (ts >= f.val_start) & (ts < f.val_end) & keep
        s, r = df.loc[m, signal], fwd[m]
        out.append(FoldStat(f.fold, f.val_start, f.val_end, int((s.notna() & r.notna()).sum()),
                            rank_ic(s, r), decile_spread_bps(s, r), auc(s, r)))
    return out


# ═══════════════════════════════════════════════════════════════════ report

def format_horizon_table(stats: dict[str, list[FoldStat]], ic_min: float = IC_MIN) -> list[str]:
    n_folds = max(len(v) for v in stats.values())
    head = "| signal | " + " | ".join(f"IC f{k + 1}" for k in range(n_folds)) + \
           " | IC mean | AUC last | spread last (bps) | verdict |"
    lines = [head, "|---|" + "---:|" * (n_folds + 3) + ":--|"]
    for name, fs in stats.items():
        ics = [x.ic for x in fs]
        ok, why = verdict(ics, ic_min)
        cells = " | ".join(f"{x.ic:+.4f}" for x in fs)
        lines.append(f"| `{name}` | {cells} | {np.nanmean(ics):+.4f} | {fs[-1].auc:.4f} | "
                     f"{fs[-1].spread_bps:+.1f} | {'**PASS**' if ok else 'FAIL (' + why + ')'} |")
    return lines


def build_frame(config: Path, signals: list[str]) -> tuple[pd.DataFrame, str]:
    req = required_sources(signals)
    df_tf, tf = load_df_from_config(config, with_contracts=bool(req), required_contracts=req)
    df = add_indicators(df_tf)
    df = add_signals(df)
    df = add_contract_signals(df, timeframe=tf)
    df = add_probe_features(df, signals, tf)
    missing = [s for s in signals if s not in df.columns]
    if missing:
        raise ValueError(f"unknown signals {missing}; known probe features: {sorted(PROBE_FEATURES)}")
    return df.reset_index(drop=True), tf


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--signal", action="append", required=True, help="column or probe feature; repeatable")
    ap.add_argument("--horizon", action="append", type=int, default=None, help="bars ahead; repeatable (default 6, 18)")
    ap.add_argument("--cv-splits", type=int, default=3)
    ap.add_argument("--ic-min", type=float, default=IC_MIN)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--gate-on", action="store_true",
                    help="evaluate only bars where the UTC daily SMA200 gate is on "
                         "(the regime in which a gate × signal rule can act)")
    args = ap.parse_args()
    horizons = args.horizon or [6, 18]

    df, tf = build_frame(args.config, args.signal)
    mask = daily_sma_gate(df, df) if args.gate_on else None
    if mask is not None:
        print(f"[gate-on] {int(mask.sum()):,} / {len(mask):,} bars")
    lines = [f"# Signal IC probe — {args.config.name}", "",
             f"- Generated: `{datetime.now().isoformat(timespec='seconds')}`",
             f"- Timeframe **{tf}**, bars {len(df):,}, {df[TS_COL].iloc[0]} → {df[TS_COL].iloc[-1]}",
             f"- Folds: {args.cv_splits} expanding time folds (validation windows only; no fitting)",
             f"- Rows: {'gate-on bars only (' + str(int(mask.sum())) + ')' if mask is not None else 'all bars'}",
             f"- Pass: |IC| ≥ {args.ic_min} on every fold, same sign, latest fold agrees. "
             f"k-bar targets overlap, so IC magnitude is the criterion, not p-values.", ""]
    for h in horizons:
        folds = time_folds(df[TS_COL], args.cv_splits, embargo=h * pd.Timedelta(tf))
        stats = {s: evaluate_signal(df, s, h, folds, mask) for s in args.signal}
        lines += [f"## horizon k = {h} bars", "",
                  "Fold windows: " + "; ".join(f"f{f.fold} {f.val_start.date()}→{f.val_end.date()}" for f in folds), ""]
        lines += format_horizon_table(stats, args.ic_min) + [""]
        print(f"\n== k = {h} ==")
        print("\n".join(format_horizon_table(stats, args.ic_min)))

    tag = args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    out = REPO_ROOT / "reports" / f"signal_ic_probe_{tag}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nreport → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
