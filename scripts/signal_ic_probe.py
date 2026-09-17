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
import yaml
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.backtest_signal_layered import DAILY_FEATURES, daily_sma_gate    # noqa: E402
from scripts.signal_linear_baseline import _to_ccxt_symbol, load_df_from_config           # noqa: E402
from utils.data_loader import required_contract_sources                  # noqa: E402
from utils.indicators import add_indicators                              # noqa: E402
from utils.signals import add_contract_signals, add_signals, znorm       # noqa: E402
from utils.data_loader import read_position_ratio, resample_position_ratio  # noqa: E402
from utils.splits import TimeFold, time_folds                            # noqa: E402

IC_MIN = 0.03
N_MIN = 300              # rows per fold below which an IC is not evidence
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


def _oi(df: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(df["sum_open_interest"], errors="coerce")


def oi_level_z(df: pd.DataFrame, timeframe: str, span: str = "90D") -> pd.Series:
    """H2a: OI (base units) z-score vs its rolling `span` window, in [-1, 1] (cap 3)."""
    return znorm(_oi(df), window=_bars(span, timeframe))


def oi_change(df: pd.DataFrame, timeframe: str, span: str = "1D") -> pd.Series:
    """OI pct change over `span` (raw); ±inf (change from a zero OI row) → NaN."""
    return _oi(df).pct_change(_bars(span, timeframe)).replace([np.inf, -np.inf], np.nan)


def oi_capitulation(df: pd.DataFrame, timeframe: str, span: str = "1D") -> pd.Series:
    """H2b: forced long unwind proxy = |OI drop| when both OI and price fell over `span`, else 0.
    Positive values = capitulation; predicted to precede a rebound (IC > 0)."""
    k = _bars(span, timeframe)
    d_oi = oi_change(df, timeframe, span)
    ret = np.log(df["close"] / df["close"].shift(k))
    both_down = (d_oi < 0) & (ret < 0)
    return (-d_oi).where(both_down, 0.0)


def oi_new_longs(df: pd.DataFrame, timeframe: str, span: str = "1D") -> pd.Series:
    """H2c: OI up & price up over `span` → OI increase, else 0 (trend-continuation confirmation)."""
    k = _bars(span, timeframe)
    d_oi = oi_change(df, timeframe, span)
    ret = np.log(df["close"] / df["close"].shift(k))
    return d_oi.where((d_oi > 0) & (ret > 0), 0.0)


PROBE_FEATURES = {
    # ── H2 (open interest) ──
    "oi_level_90d_z": lambda df, tf: oi_level_z(df, tf, "90D"),
    "oi_chg_1d": lambda df, tf: oi_change(df, tf, "1D"),
    "oi_capitulation_1d": lambda df, tf: oi_capitulation(df, tf, "1D"),
    "oi_new_longs_1d": lambda df, tf: oi_new_longs(df, tf, "1D"),
    # ── H1 (funding) ──
    "funding_cum_3d": lambda df, tf: funding_cum(df, tf, "3D"),
    "funding_cum_7d": lambda df, tf: funding_cum(df, tf, "7D"),
    "funding_rank_90d": lambda df, tf: funding_rank(df, tf, "90D"),
    # bounded [-1, 1] versions (rolling 200-bar z, cap 3) — usable as rule thresholds
    "funding_cum_3d_z": lambda df, tf: znorm(funding_cum(df, tf, "3D"), window=200),
    "funding_cum_7d_z": lambda df, tf: znorm(funding_cum(df, tf, "7D"), window=200),
    # ── H4 / H5 (Binance long/short & taker ratios; utils.data_loader.read_position_ratio) ──
    "top_position_90d_z": lambda df, tf: znorm(df["top_position"], window=_bars("90D", tf)),
    "retail_minus_top": lambda df, tf: df["global_account"] - df["top_account"],
    "retail_minus_top_90d_z": lambda df, tf: znorm(df["global_account"] - df["top_account"],
                                                   window=_bars("90D", tf)),
    "taker_dev_z30d": lambda df, tf: znorm(df["taker_vol_mean"] - 1.0, window=_bars("30D", tf)),
}
_RATIO_FEATURE_COL = {"top_position_90d_z": "top_position", "retail_minus_top": "global_account",
                      "retail_minus_top_90d_z": "global_account", "taker_dev_z30d": "taker_vol_mean"}
_PROBE_SOURCE = {name: ("ratio" if name in _RATIO_FEATURE_COL else "oi" if name.startswith("oi_") else "funding")
                 for name in PROBE_FEATURES}
_PROBE_RAW_COL = {"oi": "sum_open_interest", "funding": "funding_rate"}
RATIO_MIN_COVERAGE = 0.80         # top_* have ~15 % Vision holes in 2022; below this the probe refuses
DIST_COL = "dist_sma200"          # close / UTC-daily SMA200 − 1 of the last closed day


def add_probe_features(df: pd.DataFrame, names: list[str], timeframe: str) -> pd.DataFrame:
    """Add requested probe-only features and any daily gate-aligned features referenced."""
    out = df.copy()
    for n in names:
        if n in PROBE_FEATURES:
            raw = _RATIO_FEATURE_COL.get(n) or _PROBE_RAW_COL[_PROBE_SOURCE[n]]
            if raw not in out.columns:
                raise ValueError(f"probe feature '{n}' needs a {raw} column")
            out[n] = PROBE_FEATURES[n](out, timeframe)
        elif n in DAILY_FEATURES and n not in out.columns:
            out[n] = DAILY_FEATURES[n](out, out)     # df is the full frame → warmup satisfied
    return out


_WHERE_OPS = {"<=": np.less_equal, ">=": np.greater_equal, "<": np.less, ">": np.greater}


def parse_where(text: str) -> tuple[str, str, float]:
    """`col<0.1` / `col>=-0.05` → (col, op, value)."""
    for op in ("<=", ">=", "<", ">"):          # two-char ops first
        if op in text:
            col, _, val = text.partition(op)
            col, val = col.strip(), val.strip()
            if not col or not val:
                break
            return col, op, float(val)
    raise ValueError(f"bad --where '{text}' (expected col<op>value, op in {list(_WHERE_OPS)})")


def where_mask(df: pd.DataFrame, clauses: list[tuple[str, str, float]]) -> np.ndarray:
    """AND of the clauses; NaN never satisfies a clause."""
    m = np.ones(len(df), dtype=bool)
    for col, op, val in clauses:
        if col not in df.columns:
            raise ValueError(f"--where column '{col}' not in data")
        x = df[col].to_numpy(dtype=float)
        m &= _WHERE_OPS[op](x, val) & ~np.isnan(x)
    return m


def required_sources(signals: list[str]) -> set[str]:
    out = required_contract_sources(signals)
    out |= {_PROBE_SOURCE[s] for s in signals if s in _PROBE_SOURCE}
    return out


# ═══════════════════════════════════════════════════════════════════ metrics

def forward_log_return(close: pd.Series, horizon: int) -> pd.Series:
    return np.log(close.shift(-horizon) / close)


def _finite(signal: pd.Series, fwd: pd.Series) -> pd.Series:
    """Rows usable for a metric: both finite (NaN and ±inf excluded — e.g. OI pct_change from 0)."""
    return pd.Series(np.isfinite(signal.to_numpy(dtype=float)) & np.isfinite(fwd.to_numpy(dtype=float)),
                     index=signal.index)


def rank_ic(signal: pd.Series, fwd: pd.Series) -> float:
    m = _finite(signal, fwd)
    if m.sum() < 10 or signal[m].nunique() < 2:
        return float("nan")
    return float(spearmanr(signal[m], fwd[m]).correlation)


def decile_spread_bps(signal: pd.Series, fwd: pd.Series, q: float = DECILE) -> float:
    m = _finite(signal, fwd)
    s, f = signal[m], fwd[m]
    if len(s) < 20:
        return float("nan")
    hi, lo = s.quantile(1 - q), s.quantile(q)
    return float((f[s >= hi].mean() - f[s <= lo].mean()) * 1e4)


def auc(signal: pd.Series, fwd: pd.Series) -> float:
    m = _finite(signal, fwd)
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


def verdict(ics: list[float], ic_min: float = IC_MIN,
            ns: list[int] | None = None, n_min: int = N_MIN) -> tuple[bool, str]:
    """(pass, reason) per §5: |IC| ≥ ic_min on all folds, same sign, last fold agrees,
    and every fold has at least `n_min` rows (small filtered subsets are not evidence)."""
    if not ics or any(np.isnan(ics)):
        return False, "nan"
    if ns is not None:
        thin = [k + 1 for k, n in enumerate(ns) if n < n_min]
        if thin:
            return False, f"n<{n_min} on fold {thin}"
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
        out.append(FoldStat(f.fold, f.val_start, f.val_end, int(_finite(s, r).sum()),
                            rank_ic(s, r), decile_spread_bps(s, r), auc(s, r)))
    return out


# ═══════════════════════════════════════════════════════════════════ report

def format_horizon_table(stats: dict[str, list[FoldStat]], ic_min: float = IC_MIN) -> list[str]:
    n_folds = max(len(v) for v in stats.values())
    head = "| signal | " + " | ".join(f"IC f{k + 1} (n)" for k in range(n_folds)) + \
           " | IC mean | AUC last | spread last (bps) | verdict |"
    lines = [head, "|---|" + "---:|" * (n_folds + 3) + ":--|"]
    for name, fs in stats.items():
        ics = [x.ic for x in fs]
        ok, why = verdict(ics, ic_min, ns=[x.n for x in fs])
        cells = " | ".join(f"{x.ic:+.4f} ({x.n})" for x in fs)
        lines.append(f"| `{name}` | {cells} | {np.nanmean(ics):+.4f} | {fs[-1].auc:.4f} | "
                     f"{fs[-1].spread_bps:+.1f} | {'**PASS**' if ok else 'FAIL (' + why + ')'} |")
    return lines


def merge_position_ratio(df: pd.DataFrame, config: Path, tf: str,
                         min_coverage: float = RATIO_MIN_COVERAGE, log=print) -> pd.DataFrame:
    """Attach the four L/S ratios (+ taker_vol_mean) resampled to `tf`, exact-joined on the
    bar's open timestamp (bucket-end value → known at bar close). Done *after* add_indicators
    so the ratio holes (NaN, never filled) cannot drop OHLCV rows via the global dropna."""
    with config.open(encoding="utf-8") as f:
        c = yaml.safe_load(f)["crypto"]
    tz = c.get("timezone", "Asia/Shanghai")
    start_utc = pd.Timestamp(c["start_date"], tz=tz).tz_convert("UTC").isoformat()
    end_utc = pd.Timestamp(c["end_date"], tz=tz).tz_convert("UTC").isoformat()
    raw = read_position_ratio(_to_ccxt_symbol(c["symbol"]), since=start_utc, until=end_utc)
    if raw.empty:
        raise ValueError("position ratio table returned no rows for the config window")
    # Bucket in the config tz so bucket edges coincide with the OHLCV bars (resample_ohlcv
    # runs in that tz): a UTC daily bucket would sit 8 h off an Asia/Shanghai daily bar.
    ts = raw["timestamp"]
    raw = raw.assign(timestamp=(ts.dt.tz_localize("UTC") if ts.dt.tz is None else ts).dt.tz_convert(tz))
    r = resample_position_ratio(raw, tf)
    out = df.merge(r, on="timestamp", how="left")
    for col in ("top_account", "top_position", "global_account", "taker_vol", "taker_vol_mean"):
        cov = out[col].notna().mean()
        first = out.loc[out[col].notna(), "timestamp"]
        log(f"[ratio] {col:16s} coverage={cov:6.1%}  "
            f"{first.iloc[0] if len(first) else 'ABSENT'} → {first.iloc[-1] if len(first) else ''}")
        if cov < min_coverage:
            raise ValueError(f"position ratio '{col}' coverage {cov:.1%} < {min_coverage:.0%}; "
                             "narrow start_date (clean window from 2023-01) or backfill")
    return out


def build_frame(config: Path, signals: list[str],
                extra_features: list[str] = (),
                min_ratio_coverage: float = RATIO_MIN_COVERAGE) -> tuple[pd.DataFrame, str]:
    req = required_sources(list(signals) + list(extra_features))
    need_ratio = "ratio" in req
    req_db = req - {"ratio"}
    df_tf, tf = load_df_from_config(config, with_contracts=bool(req_db), required_contracts=req_db)
    df = add_indicators(df_tf)
    df = add_signals(df)
    df = add_contract_signals(df, timeframe=tf)
    if need_ratio:
        df = merge_position_ratio(df, config, tf, min_coverage=min_ratio_coverage)
    df = add_probe_features(df, signals + extra_features, tf)
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
    ap.add_argument("--min-ratio-coverage", type=float, default=RATIO_MIN_COVERAGE,
                    help="refuse if any L/S ratio column covers less than this fraction of bars "
                         f"(default {RATIO_MIN_COVERAGE}; lower it only for a documented data-start reason)")
    ap.add_argument("--where", action="append", default=[],
                    help="row filter col<op>value (AND-ed, repeatable), e.g. dist_sma200<0.10")
    args = ap.parse_args()
    horizons = args.horizon or [6, 18]
    clauses = [parse_where(w) for w in args.where]

    df, tf = build_frame(args.config, args.signal, extra_features=[c for c, _, _ in clauses], min_ratio_coverage=args.min_ratio_coverage)
    mask = None
    if args.gate_on:
        mask = daily_sma_gate(df, df)
    if clauses:
        wm = where_mask(df, clauses)
        mask = wm if mask is None else (mask & wm)
    if mask is not None:
        print(f"[rows] {int(mask.sum()):,} / {len(mask):,} bars after gate/where filters")
    lines = [f"# Signal IC probe — {args.config.name}", "",
             f"- Generated: `{datetime.now().isoformat(timespec='seconds')}`",
             f"- Timeframe **{tf}**, bars {len(df):,}, {df[TS_COL].iloc[0]} → {df[TS_COL].iloc[-1]}",
             f"- Folds: {args.cv_splits} expanding time folds (validation windows only; no fitting)",
             f"- Rows: {'filtered (' + str(int(mask.sum())) + ') — gate_on=' + str(args.gate_on) + ' where=' + ' & '.join(args.where) if mask is not None else 'all bars'}",
             f"- Pass: |IC| ≥ {args.ic_min} on every fold, same sign, latest fold agrees, n ≥ {N_MIN} per fold. "
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
