"""
TC-A6: Linear-baseline iron gate for Phase 1.

Rationale (design doc §8.1, T1.4): PPO is a far more expressive non-linear
function approximator than LogReg/LinReg. If even a linear model cannot
extract *any* directional predictive signal from the feature set, then PPO
has no Alpha to find either — no amount of hyperparameter tuning will fix
noise. This script blocks entry to Phase 2 until the signal set clears:

    LogReg AUC (iron gate): mean > --auc-threshold AND min > --auc-floor
    LinReg R² (diagnostic): mean > --r2-floor     (catastrophe check only)

AUC is the real gate: does the feature set carry *any* directional info?
R² on crypto 4h returns is structurally dominated by noise variance and
cannot pass a meaningful positive threshold even when AUC is clearly > 0.52.
R² is therefore retained as a sanity check — if R² is very negative
(e.g. < -0.10), the model is systematically worse than predicting the mean,
which would indicate a pipeline bug or feature mis-specification.

Target horizon is k bars ahead (default 6 → 1 day on 4h data), chosen to
match the combiner's multi-step rollout semantics. See design doc §8.1 T1.4
revision notes (2026-04-18) for the rationale behind these choices.

Inputs:
  - config/signals_v1.yaml          (signal list, produced by TC-A7)
  - data/cache/BTCUSDT_4h_*.parquet (OHLCV cache, produced by TC-A5 notebook)

Outputs:
  - reports/signal_linear_baseline.md
  - reports/signal_linear_baseline_artifacts/confusion_matrix.png
  - reports/signal_linear_baseline_artifacts/residuals.png
  - reports/signal_linear_baseline_artifacts/coefficients.png

Usage:
  python -m scripts.signal_linear_baseline
  python -m scripts.signal_linear_baseline --cv-splits 5 --auc-threshold 0.52 --r2-threshold 0.005
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sklearn.linear_model import LinearRegression, LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    confusion_matrix,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from agents.ppo_shared import resample_ohlcv  # noqa: E402
from utils.db import read_ohlcv  # noqa: E402
from utils.indicators import add_indicators  # noqa: E402
from utils.signals import SIGNAL_WARMUP_WINDOW, add_signals  # noqa: E402


DEFAULT_CACHE = (
    REPO_ROOT / "data" / "cache" / "BTCUSDT_4h_2020-01-01_2024-07-01.parquet"
)
DEFAULT_SIGNALS = REPO_ROOT / "config" / "signals_v1.yaml"
DEFAULT_REPORT_DIR = REPO_ROOT / "reports"

# Bars-per-day by timeframe — used to auto-derive the 1-day horizon when
# --horizon is not explicitly overridden. Keys are lowercased timeframe strings.
_BARS_PER_DAY = {
    "1m": 1440, "5min": 288, "15min": 96, "30min": 48,
    "1h": 24, "2h": 12, "4h": 6, "1d": 1,
}


def load_signal_list(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    return list(payload["signals"])


def load_df_from_config(config_path: Path) -> tuple[pd.DataFrame, str]:
    """Load + resample crypto OHLCV from DB per a training-style config yaml.

    Mirrors train.py::_load_crypto_df so baseline uses the same data path as
    training. Returns (df_tf, timeframe_lower).
    """
    with config_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    c = cfg["crypto"]
    tz = c.get("timezone", "Asia/Shanghai")
    start_utc = pd.Timestamp(c["start_date"], tz=tz).tz_convert("UTC").isoformat()
    end_utc = pd.Timestamp(c["end_date"], tz=tz).tz_convert("UTC").isoformat()

    df_raw = read_ohlcv(
        c["symbol"], start=start_utc, end=end_utc,
        table=c.get("db_table", "public.crypto_kline_binance"),
        only_closed=True,
    )
    ts = df_raw["timestamp"]
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    df_raw["timestamp"] = ts.dt.tz_convert(tz)

    tf = c.get("timeframe", "1d").lower()
    df_tf = df_raw if tf == "1m" else resample_ohlcv(df_raw, tf)
    return df_tf, tf


def build_dataset_from_df(
    df_raw: pd.DataFrame, signal_cols: list[str], horizon: int
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return (X, y_cls, y_reg) aligned. Drops warmup + last `horizon` rows."""
    df_ind = add_indicators(df_raw)
    df = add_signals(df_ind)
    df = df.iloc[SIGNAL_WARMUP_WINDOW:].reset_index(drop=True)

    missing = [c for c in signal_cols if c not in df.columns]
    if missing:
        raise ValueError(f"signals_v1.yaml lists columns not present: {missing}")

    fut_close = df["close"].shift(-horizon)
    y_cls = (fut_close > df["close"]).astype(int)
    y_reg = np.log(fut_close / df["close"])

    keep = y_reg.notna()
    return df.loc[keep, signal_cols].reset_index(drop=True), \
           y_cls.loc[keep].reset_index(drop=True), \
           y_reg.loc[keep].reset_index(drop=True)


def build_dataset(
    cache: Path, signal_cols: list[str], horizon: int
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Parquet-cache entry point (legacy 4h path). Reads parquet then delegates."""
    return build_dataset_from_df(pd.read_parquet(cache), signal_cols, horizon)


def run_cv(
    X: pd.DataFrame,
    y_cls: pd.Series,
    y_reg: pd.Series,
    n_splits: int,
) -> dict:
    """Run TimeSeriesSplit CV for LogReg and LinReg. Returns per-fold metrics."""
    tss = TimeSeriesSplit(n_splits=n_splits)
    fold_records = []
    last_fold = None

    for fold, (train_idx, val_idx) in enumerate(tss.split(X), start=1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_cls_tr, y_cls_val = y_cls.iloc[train_idx], y_cls.iloc[val_idx]
        y_reg_tr, y_reg_val = y_reg.iloc[train_idx], y_reg.iloc[val_idx]

        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        X_val_s = scaler.transform(X_val)

        logreg = LogisticRegression(max_iter=1000, solver="lbfgs")
        logreg.fit(X_tr_s, y_cls_tr)
        proba = logreg.predict_proba(X_val_s)[:, 1]
        pred_cls = (proba >= 0.5).astype(int)
        auc = float(roc_auc_score(y_cls_val, proba))

        linreg = LinearRegression()
        linreg.fit(X_tr_s, y_reg_tr)
        pred_reg = linreg.predict(X_val_s)
        r2 = float(r2_score(y_reg_val, pred_reg))

        fold_records.append({
            "fold": fold, "n_train": len(train_idx), "n_val": len(val_idx),
            "auc": auc, "r2": r2,
        })
        last_fold = {
            "y_cls_val": y_cls_val.values, "pred_cls": pred_cls,
            "y_reg_val": y_reg_val.values, "pred_reg": pred_reg,
            "logreg_coef": logreg.coef_[0], "linreg_coef": linreg.coef_,
        }

    return {"folds": fold_records, "last_fold": last_fold}


def verdict(
    folds: list[dict],
    auc_th: float, auc_floor: float,
    r2_floor: float,
) -> dict:
    aucs = np.array([f["auc"] for f in folds])
    r2s = np.array([f["r2"] for f in folds])
    auc_pass = aucs.mean() > auc_th and aucs.min() > auc_floor
    # R² is a catastrophe check only: mean must not be severely negative.
    r2_pass = r2s.mean() > r2_floor
    return {
        "auc_mean": float(aucs.mean()), "auc_std": float(aucs.std()),
        "auc_min": float(aucs.min()), "auc_max": float(aucs.max()),
        "r2_mean": float(r2s.mean()), "r2_std": float(r2s.std()),
        "r2_min": float(r2s.min()), "r2_max": float(r2s.max()),
        "auc_pass": bool(auc_pass),
        "r2_pass": bool(r2_pass),
        "overall_pass": bool(auc_pass and r2_pass),
    }


def plot_confusion(last: dict, out: Path) -> None:
    cm = confusion_matrix(last["y_cls_val"], last["pred_cls"])
    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["pred down", "pred up"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["true down", "true up"])
    ax.set_title("Confusion Matrix (last fold)")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, dpi=100); plt.close(fig)


def plot_residuals(last: dict, out: Path) -> None:
    resid = last["y_reg_val"] - last["pred_reg"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].scatter(last["pred_reg"], resid, s=4, alpha=0.35, color="#3b7dd8")
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].set_xlabel("predicted log-return"); axes[0].set_ylabel("residual")
    axes[0].set_title("Residuals vs Predicted (last fold)")
    axes[1].hist(resid, bins=60, color="#3b7dd8", edgecolor="white")
    axes[1].axvline(0, color="black", linewidth=0.5)
    axes[1].set_xlabel("residual"); axes[1].set_title("Residual distribution")
    fig.tight_layout()
    fig.savefig(out, dpi=100); plt.close(fig)


def plot_coefficients(last: dict, sig_cols: list[str], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, max(3.5, 0.28 * len(sig_cols))))
    for ax, coefs, title in [
        (axes[0], last["logreg_coef"], "LogReg coefficients (direction)"),
        (axes[1], last["linreg_coef"], "LinReg coefficients (log-return)"),
    ]:
        order = np.argsort(coefs)
        cols = [sig_cols[i] for i in order]
        vals = coefs[order]
        colors = ["#c0392b" if v < 0 else "#2e86de" for v in vals]
        ax.barh(range(len(cols)), vals, color=colors)
        ax.set_yticks(range(len(cols)))
        ax.set_yticklabels(cols, fontsize=8)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=100); plt.close(fig)


def write_report(
    report_path: Path,
    artifact_dir: Path,
    sig_cols: list[str],
    folds: list[dict],
    v: dict,
    auc_th: float, auc_floor: float,
    r2_floor: float,
    horizon: int,
    source_tag: str,
    signals_config: Path,
) -> None:
    rel = lambda p: str(p.relative_to(report_path.parent)).replace("\\", "/")
    lines: list[str] = []
    lines.append("# Linear Baseline Report — TC-A6\n")
    lines.append(f"- Generated: `{dt.datetime.now().isoformat(timespec='seconds')}`  ")
    sig_path = signals_config.resolve()
    try:
        sig_rel = sig_path.relative_to(REPO_ROOT)
    except ValueError:
        sig_rel = sig_path
    lines.append(f"- Signals config: `{sig_rel}` ({len(sig_cols)} features)  ")
    lines.append(f"- Data source: `{source_tag}`  ")
    lines.append(f"- Target horizon: **k = {horizon} bars**  ")
    lines.append(f"- CV: TimeSeriesSplit({len(folds)} folds, no shuffle)  ")
    lines.append(f"- Iron gate: AUC mean > **{auc_th}** AND min > **{auc_floor}**  ")
    lines.append(f"- R² diagnostic: mean > **{r2_floor}** (catastrophe check, not a positive-predictive gate)\n")

    verdict_str = "✅ PASS" if v["overall_pass"] else "❌ FAIL — Phase 2 frozen"
    lines.append(f"## Verdict: {verdict_str}\n")
    lines.append("| Metric | Rule | Min | Mean | Max | Pass |")
    lines.append("|--------|------|-----|------|-----|------|")
    lines.append(f"| ROC-AUC (LogReg, iron gate) | mean > {auc_th} AND min > {auc_floor} | {v['auc_min']:.4f} | {v['auc_mean']:.4f} | {v['auc_max']:.4f} | {'✅' if v['auc_pass'] else '❌'} |")
    lines.append(f"| R² (LinReg, diagnostic) | mean > {r2_floor} | {v['r2_min']:+.4f} | {v['r2_mean']:+.4f} | {v['r2_max']:+.4f} | {'✅' if v['r2_pass'] else '❌'} |")
    lines.append("")

    lines.append("## Per-fold metrics\n")
    lines.append("| Fold | n_train | n_val | ROC-AUC | R² |")
    lines.append("|------|---------|-------|---------|-----|")
    for f in folds:
        lines.append(f"| {f['fold']} | {f['n_train']:,} | {f['n_val']:,} | {f['auc']:.4f} | {f['r2']:+.4f} |")
    lines.append("")

    lines.append("## Diagnostics (last fold)\n")
    lines.append(f"![confusion matrix]({rel(artifact_dir / 'confusion_matrix.png')})")
    lines.append("")
    lines.append(f"![residuals]({rel(artifact_dir / 'residuals.png')})")
    lines.append("")
    lines.append("## Feature weights (last fold)\n")
    lines.append("Positive = bullish contribution; negative = bearish. Standardized features.\n")
    lines.append(f"![coefficients]({rel(artifact_dir / 'coefficients.png')})")
    lines.append("")

    if not v["overall_pass"]:
        lines.append("## Next steps (iron-gate failure)\n")
        lines.append("- Return to §4 signal design — not a hyperparameter issue.")
        lines.append("- Check whether any sig_* is degenerate post-warmup (flat, near-zero std).")
        lines.append("- Consider adding non-linear transformations that linear models miss "
                     "(interaction terms, regime-conditioned features).")
        lines.append("- Do NOT proceed to Phase 2 PPO training until this gate passes.")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None,
                        help="Training-style yaml (e.g. config/stage2_1h_signal.yaml). "
                             "When set, loads from DB and resamples; --cache is ignored. "
                             "Horizon auto-derived to 1 day in bars unless --horizon given.")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE,
                        help="Parquet OHLCV cache (legacy 4h path). Ignored if --config set.")
    parser.add_argument("--signals-config", type=Path, default=DEFAULT_SIGNALS)
    parser.add_argument("--cv-splits", type=int, default=3,
                        help="TimeSeriesSplit folds; default 3 ensures each fold "
                             "has ≥~1800 training bars on the 2020-2024 BTC 4h set")
    parser.add_argument("--horizon", type=int, default=None,
                        help="Prediction horizon in bars. Default: 6 for --cache path, "
                             "or timeframe-derived 1-day equivalent for --config path.")
    parser.add_argument("--auc-threshold", type=float, default=0.52,
                        help="AUC mean must exceed this")
    parser.add_argument("--auc-floor", type=float, default=0.50,
                        help="AUC min must exceed this (no fold clearly reversed)")
    parser.add_argument("--r2-floor", type=float, default=-0.10,
                        help="R² mean must exceed this (catastrophe check, not a positive-predictive gate)")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()

    if not args.signals_config.exists():
        print(f"ERROR: signals config not found: {args.signals_config}", file=sys.stderr)
        return 1

    sig_cols = load_signal_list(args.signals_config)
    print(f"Loaded {len(sig_cols)} signals from {args.signals_config.name}")

    if args.config is not None:
        if not args.config.exists():
            print(f"ERROR: config not found: {args.config}", file=sys.stderr)
            return 1
        df_tf, tf = load_df_from_config(args.config)
        if args.horizon is None:
            if tf not in _BARS_PER_DAY:
                print(f"ERROR: unknown timeframe '{tf}', pass --horizon explicitly",
                      file=sys.stderr)
                return 1
            args.horizon = _BARS_PER_DAY[tf]
        print(f"DB load: timeframe={tf}, bars={len(df_tf)}, horizon={args.horizon}")
        X, y_cls, y_reg = build_dataset_from_df(df_tf, sig_cols, horizon=args.horizon)
        source_tag = f"config:{args.config.name} (tf={tf})"
    else:
        if not args.cache.exists():
            print(f"ERROR: cache not found: {args.cache}", file=sys.stderr)
            return 1
        if args.horizon is None:
            args.horizon = 6
        tf = "4h"   # legacy parquet cache is 4h
        X, y_cls, y_reg = build_dataset(args.cache, sig_cols, horizon=args.horizon)
        source_tag = f"cache:{args.cache.name}"
    print(f"Dataset: X={X.shape}, horizon={args.horizon} bars, "
          f"y_cls balance={y_cls.mean():.3f} | source={source_tag}")

    cv = run_cv(X, y_cls, y_reg, n_splits=args.cv_splits)
    v = verdict(cv["folds"],
                args.auc_threshold, args.auc_floor,
                args.r2_floor)

    print(f"\nAUC  min={v['auc_min']:.4f}  mean={v['auc_mean']:.4f}  max={v['auc_max']:.4f}  "
          f"{'PASS' if v['auc_pass'] else 'FAIL'}")
    print(f"R²   min={v['r2_min']:+.4f}  mean={v['r2_mean']:+.4f}  max={v['r2_max']:+.4f}  "
          f"{'PASS' if v['r2_pass'] else 'FAIL'}")
    print(f"\nOverall: {'✅ PASS' if v['overall_pass'] else '❌ FAIL — Phase 2 frozen'}")

    # Per-timeframe suffix so 4h / 1h / 5min runs don't overwrite each other.
    suffix = f"_{tf}" if args.config is not None else ""
    artifact_dir = args.report_dir / f"signal_linear_baseline{suffix}_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion(cv["last_fold"], artifact_dir / "confusion_matrix.png")
    plot_residuals(cv["last_fold"], artifact_dir / "residuals.png")
    plot_coefficients(cv["last_fold"], sig_cols, artifact_dir / "coefficients.png")

    report_path = args.report_dir / f"signal_linear_baseline{suffix}.md"
    write_report(report_path, artifact_dir, sig_cols, cv["folds"], v,
                 args.auc_threshold, args.auc_floor,
                 args.r2_floor,
                 args.horizon,
                 source_tag, args.signals_config)
    print(f"\nWrote {report_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {artifact_dir.relative_to(REPO_ROOT)}/")

    return 0 if v["overall_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
