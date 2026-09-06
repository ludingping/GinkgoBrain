"""Pooled multi-asset linear baseline: does more data raise OOS AUC?

Hypothesis under test (2026-09-06): the ~0.53 AUC of the 4h signal pool on
BTC alone is a *sample-size* limit (≈12k bars), not a feature limit. If it is,
fitting the same LogReg on BTC+ETH+BNB+SOL stacked should raise the AUC
measured on BTC's own validation slices. If pooling does nothing, the features
carry no more information and the next step is new data (contract signals),
not multi-asset PPO.

Three arms per fold, same time-based folds for all (see utils.splits.time_folds):

  single    fit on the asset's own history           → AUC on its own val slice
  pooled    fit on every asset's history             → AUC per asset + pooled
  transfer  fit on every *other* asset (target held out entirely)
                                                       → AUC on the target's val

Folds are cut by timestamp with an embargo of `horizon` bars, never by row
index: two assets at the same time are correlated, so an index split would
leak the validation window through a sibling asset.

Usage:
  uv run python -m scripts.signal_linear_baseline_pooled
  uv run python -m scripts.signal_linear_baseline_pooled --timeframe 1h --horizon 24 \
      --signals-config config/signals_v1_1h.yaml
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from agents.ppo_shared import resample_ohlcv  # noqa: E402
from scripts.signal_linear_baseline import (  # noqa: E402
    _BARS_PER_DAY,
    build_dataset_from_df,
    load_signal_list,
)
from utils.db import read_ohlcv  # noqa: E402
from utils.splits import TimeFold, time_folds  # noqa: E402

DEFAULT_SYMBOLS = ("BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT")
DEFAULT_SIGNALS = REPO_ROOT / "config" / "signals_v2_4h.yaml"
DEFAULT_REPORT_DIR = REPO_ROOT / "reports"
SYMBOL_COL = "symbol"
TS_COL = "timestamp"


# ─── data ────────────────────────────────────────────────────────────────────

def load_symbol_tf(
    symbol: str, timeframe: str, start: str, end: str, tz: str, cache_dir: Path | None,
) -> pd.DataFrame:
    """1m OHLCV from DB → resampled to `timeframe` in `tz`. Same path as train.py."""
    tag = symbol.replace("/", "") + f"_{timeframe}_{start[:10]}_{end[:10]}.parquet"
    cache = cache_dir / tag if cache_dir else None
    if cache is not None and cache.exists():
        return pd.read_parquet(cache)

    start_utc = pd.Timestamp(start, tz=tz).tz_convert("UTC").isoformat()
    end_utc = pd.Timestamp(end, tz=tz).tz_convert("UTC").isoformat()
    df_raw = read_ohlcv(symbol, start=start_utc, end=end_utc, only_closed=True)
    if df_raw.empty:
        raise ValueError(f"no OHLCV rows for {symbol} in [{start}, {end})")
    ts = df_raw[TS_COL]
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    df_raw[TS_COL] = ts.dt.tz_convert(tz)
    df_tf = df_raw if timeframe == "1m" else resample_ohlcv(df_raw, timeframe)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df_tf.to_parquet(cache, index=False)
    return df_tf


def build_pooled_frame(
    frames: dict[str, pd.DataFrame], signal_cols: list[str], horizon: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Stack per-asset (X, y) with symbol + timestamp columns; features per asset."""
    xs, ys = [], []
    for symbol, df in frames.items():
        X, y_cls, _ = build_dataset_from_df(df, signal_cols, horizon, extra_cols=(TS_COL,))
        X = X.assign(**{SYMBOL_COL: symbol})
        xs.append(X)
        ys.append(y_cls)
    X_all = pd.concat(xs, ignore_index=True)
    y_all = pd.concat(ys, ignore_index=True)
    return X_all, y_all


# ─── model ───────────────────────────────────────────────────────────────────

MODELS = ("logreg", "lgbm")

# Deliberately small GBDT: the question is whether *more rows* let a non-linear
# learner find structure a 10-parameter LogReg cannot, not to tune a model.
_LGBM_PARAMS = dict(
    n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=200,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
    verbose=-1, n_jobs=4,
)


def make_model(name: str):
    if name == "logreg":
        return LogisticRegression(max_iter=1000, solver="lbfgs")
    if name == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**_LGBM_PARAMS)
    raise ValueError(f"unknown model {name!r}; choose from {MODELS}")


def fit_auc(
    X_tr: pd.DataFrame, y_tr: pd.Series, X_va: pd.DataFrame, y_va: pd.Series,
    *, model_name: str = "logreg",
) -> float | None:
    """OOS AUC of `model_name` on standardised features; None if a side is degenerate."""
    if len(X_tr) == 0 or len(X_va) == 0 or y_tr.nunique() < 2 or y_va.nunique() < 2:
        return None
    scaler = StandardScaler().fit(X_tr)
    model = make_model(model_name)
    model.fit(scaler.transform(X_tr), y_tr)
    proba = model.predict_proba(scaler.transform(X_va))[:, 1]
    return float(roc_auc_score(y_va, proba))


@dataclass(frozen=True)
class FoldResult:
    fold: int
    arm: str            # single | pooled | transfer
    eval_symbol: str    # asset (or "ALL") the AUC is measured on
    n_train: int
    n_val: int
    auc: float | None


def evaluate_fold(
    fold: TimeFold, X: pd.DataFrame, y: pd.Series, signal_cols: list[str], symbols: list[str],
    *, model_name: str = "logreg",
) -> list[FoldResult]:
    tr_mask, va_mask = fold.masks(X[TS_COL])
    sym = X[SYMBOL_COL]
    feats = X[signal_cols]
    out: list[FoldResult] = []

    def record(arm: str, eval_symbol: str, tr: pd.Series, va: pd.Series) -> None:
        out.append(FoldResult(
            fold.fold, arm, eval_symbol, int(tr.sum()), int(va.sum()),
            fit_auc(feats[tr], y[tr], feats[va], y[va], model_name=model_name),
        ))

    for s in symbols:
        own = sym == s
        record("single", s, tr_mask & own, va_mask & own)
        record("pooled", s, tr_mask, va_mask & own)
        record("transfer", s, tr_mask & ~own, va_mask & own)
    record("pooled", "ALL", tr_mask, va_mask)
    return out


# ─── report ──────────────────────────────────────────────────────────────────

def summarise(results: list[FoldResult]) -> pd.DataFrame:
    df = pd.DataFrame([r.__dict__ for r in results]).dropna(subset=["auc"])
    g = df.groupby(["eval_symbol", "arm"])["auc"]
    return pd.DataFrame({
        "mean": g.mean(), "min": g.min(), "max": g.max(), "folds": g.count(),
    }).reset_index()


def write_report(
    path: Path, summary: pd.DataFrame, results: list[FoldResult], folds: list[TimeFold],
    *, symbols: list[str], target: str, signal_cols: list[str], signals_config: Path,
    timeframe: str, horizon: int, n_bars: dict[str, int], model_name: str,
) -> None:
    piv = summary.pivot(index="eval_symbol", columns="arm", values="mean")
    order = [*symbols, "ALL"]
    piv = piv.reindex([s for s in order if s in piv.index])
    lines = ["# Pooled Multi-Asset Linear Baseline\n"]
    lines.append(f"- Generated: `{dt.datetime.now().isoformat(timespec='seconds')}`  ")
    try:
        sig_rel = signals_config.resolve().relative_to(REPO_ROOT)
    except ValueError:
        sig_rel = signals_config.name
    lines.append(f"- Signals config: `{sig_rel}` ({len(signal_cols)} features)  ")
    lines.append(f"- Timeframe: **{timeframe}**, horizon **k = {horizon} bars**, model **{model_name}** on standardised features  ")
    lines.append(f"- Assets: {', '.join(symbols)}; target: **{target}**  ")
    lines.append("- Bars after warmup: " + ", ".join(f"{s} {n:,}" for s, n in n_bars.items()) + "  ")
    lines.append(f"- CV: {len(folds)} expanding time folds, embargo = {horizon} bars\n")

    lines.append("## AUC by arm (mean over folds)\n")
    lines.append("| eval on | single (own history) | pooled (all assets) | transfer (others only) | Δ pooled − single |")
    lines.append("|---|---|---|---|---|")
    for s, row in piv.iterrows():
        single = row.get("single", np.nan); pooled = row.get("pooled", np.nan)
        transfer = row.get("transfer", np.nan)
        delta = pooled - single if pd.notna(single) else np.nan
        f = lambda v: "—" if pd.isna(v) else f"{v:.4f}"
        fd = "—" if pd.isna(delta) else f"{delta:+.4f}"
        lines.append(f"| {s} | {f(single)} | {f(pooled)} | {f(transfer)} | {fd} |")
    lines.append("")

    lines.append("## Per-fold detail\n")
    lines.append("| fold | val window | eval on | arm | n_train | n_val | AUC |")
    lines.append("|---|---|---|---|---|---|---|")
    by_fold = {f.fold: f for f in folds}
    for r in results:
        f = by_fold[r.fold]
        win = f"{f.val_start:%Y-%m-%d} → {f.val_end:%Y-%m-%d}"
        auc = "—" if r.auc is None else f"{r.auc:.4f}"
        lines.append(f"| {r.fold} | {win} | {r.eval_symbol} | {r.arm} | {r.n_train:,} | {r.n_val:,} | {auc} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# ─── main ────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    ap.add_argument("--target", default="BTC/USDT")
    ap.add_argument("--timeframe", default="4h")
    ap.add_argument("--start", default="2021-01-01 00:00:00")
    ap.add_argument("--end", default="2026-09-04 00:00:00")
    ap.add_argument("--timezone", default="Asia/Shanghai")
    ap.add_argument("--signals-config", type=Path, default=DEFAULT_SIGNALS)
    ap.add_argument("--horizon", type=int, default=None, help="bars ahead; default 1 day")
    ap.add_argument("--cv-splits", type=int, default=3)
    ap.add_argument("--model", choices=MODELS, default="logreg")
    ap.add_argument("--cache-dir", type=Path, default=None, help="parquet cache for resampled bars")
    ap.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = ap.parse_args()

    tf = args.timeframe.lower()
    horizon = args.horizon or _BARS_PER_DAY[tf]
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if args.target not in symbols:
        print(f"ERROR: target {args.target} not in --symbols", file=sys.stderr)
        return 1
    signal_cols = load_signal_list(args.signals_config)
    print(f"{len(signal_cols)} signals, tf={tf}, horizon={horizon}, symbols={symbols}")

    frames = {}
    for s in symbols:
        frames[s] = load_symbol_tf(s, tf, args.start, args.end, args.timezone, args.cache_dir)
        print(f"  {s}: {len(frames[s]):,} bars")
    X, y = build_pooled_frame(frames, signal_cols, horizon)
    n_bars = X[SYMBOL_COL].value_counts().to_dict()
    print(f"pooled dataset: {len(X):,} rows, y balance={y.mean():.3f}")

    folds = time_folds(X[TS_COL], args.cv_splits, embargo=horizon * pd.Timedelta(tf))
    results: list[FoldResult] = []
    for f in folds:
        results.extend(evaluate_fold(f, X, y, signal_cols, symbols, model_name=args.model))
        print(f"fold {f.fold}: val {f.val_start:%Y-%m-%d} → {f.val_end:%Y-%m-%d} done")

    summary = summarise(results)
    piv = summary.pivot(index="eval_symbol", columns="arm", values="mean")
    print("\nAUC mean over folds:")
    print(piv.round(4).to_string())

    args.report_dir.mkdir(parents=True, exist_ok=True)
    report = args.report_dir / (
        f"signal_linear_baseline_pooled_{tf}_{args.model}_{args.signals_config.stem}.md"
    )
    write_report(report, summary, results, folds, symbols=symbols, target=args.target,
                 signal_cols=signal_cols, signals_config=args.signals_config,
                 timeframe=tf, horizon=horizon, n_bars=n_bars, model_name=args.model)
    print(f"\nWrote {report.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
