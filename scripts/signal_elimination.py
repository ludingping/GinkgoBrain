"""
TC-A7: Signal elimination script.

Applies two mechanical rules from §4.4 of the design doc:
  1. Single-signal Sharpe < --sharpe-threshold  → eliminate
  2. Pairwise |Pearson| > --corr-threshold      → keep the higher-Sharpe one

State-only signals (volatility group + sig_regime_drawdown) are exempt from
the Sharpe gate because their single-signal backtest is not semantically valid
(they express state, not direction).

Outputs:
  - config/signals_v1.yaml           (kept signal list, for env + GBDT loaders)
  - config/signal_elimination_log.md (human-readable decision log)

Usage:
  python -m scripts.signal_elimination
  python -m scripts.signal_elimination --dry-run
  python -m scripts.signal_elimination --cache data/cache/BTCUSDT_4h_2020-01-01_2024-07-01.parquet
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.ppo_shared import resample_ohlcv  # noqa: E402
from utils.data_loader import merge_contract_data  # noqa: E402
from utils.db import (  # noqa: E402
    read_funding,
    read_liquidation_agg,
    read_ohlcv,
    read_open_interest,
)
from utils.indicators import add_indicators  # noqa: E402
from utils.signals import (  # noqa: E402
    SIGNAL_WARMUP_WINDOW,
    add_contract_signals,
    add_signals,
    signal_columns,
)


# ── Signals exempt from Sharpe gate (state, not direction) ───────────────
STATE_ONLY_SIGNALS = {
    "sig_vol_atr_pct",
    "sig_vol_bb_width",
    "sig_vol_bb_position",
    "sig_regime_drawdown",
}

DEFAULT_CACHE = (
    REPO_ROOT / "data" / "cache" / "BTCUSDT_4h_2020-01-01_2024-07-01.parquet"
)
# bars/year by timeframe — used for Sharpe annualization under --config.
BARS_PER_YEAR = {
    "1m": 1440 * 365, "5min": 288 * 365, "15min": 96 * 365, "30min": 48 * 365,
    "1h": 24 * 365, "2h": 12 * 365, "4h": 6 * 365, "1d": 365,
}
PERIODS_PER_YEAR = BARS_PER_YEAR["4h"]  # legacy default for --cache path


def backtest_signal(sig: pd.Series, log_ret: pd.Series, periods_per_year: int) -> float:
    """Naive long-only backtest Sharpe per §4.3 of design doc."""
    sig_prev = sig.shift(1).fillna(0.0)
    pos = np.zeros(len(sig_prev), dtype=float)
    current = 0.0
    for i, v in enumerate(sig_prev.values):
        if v > 0.3:
            current = 1.0
        elif v < -0.3:
            current = 0.0
        pos[i] = current
    strat_ret = pos * log_ret.values
    mu, sd = strat_ret.mean(), strat_ret.std()
    if sd < 1e-12:
        return 0.0
    return float(mu / sd * np.sqrt(periods_per_year))


def compute_stats(
    df: pd.DataFrame, sig_cols: list[str], periods_per_year: int,
) -> tuple[dict, pd.DataFrame]:
    """Return (sharpe_by_signal, correlation_matrix)."""
    log_ret = np.log(df["close"] / df["close"].shift(1)).fillna(0.0)
    sharpe = {col: backtest_signal(df[col], log_ret, periods_per_year) for col in sig_cols}
    corr = df[sig_cols].corr(method="pearson")
    return sharpe, corr


def _to_ccxt_symbol(symbol: str) -> str:
    """`BTCUSDT` → `BTC/USDT`（合约表用 ccxt 现货格式存）。已含 `/` 时原样返回。"""
    if "/" in symbol:
        return symbol
    if symbol.endswith("USDT"):
        return f"{symbol[:-4]}/USDT"
    return symbol


def load_df_from_config(
    config_path: Path,
    *,
    with_contracts: bool = False,
) -> tuple[pd.DataFrame, str]:
    """Mirror signal_linear_baseline.load_df_from_config: DB → resampled df.

    Args:
        with_contracts: True 时跨库 join funding/OI/liquidation 到 OHLCV，并立刻
            对合约列 fillna(0)——避免下游 `add_indicators.dropna()` 误删合约起点
            之前的 OHLCV 行（设计稿 §4.2 + 5min ETH 调试经验）。
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

    if with_contracts:
        ccxt_symbol = _to_ccxt_symbol(c["symbol"])
        df_funding = read_funding(ccxt_symbol, start=start_utc, end=end_utc)
        df_oi      = read_open_interest(ccxt_symbol, start=start_utc, end=end_utc)
        df_liq     = read_liquidation_agg(ccxt_symbol, start=start_utc, end=end_utc)
        for sub in (df_funding, df_oi, df_liq):
            if sub.empty:
                continue
            sub_ts = sub["timestamp"]
            if sub_ts.dt.tz is None:
                sub_ts = sub_ts.dt.tz_localize("UTC")
            sub["timestamp"] = sub_ts.dt.tz_convert(tz)
        df_tf = merge_contract_data(
            df_tf, df_funding=df_funding, df_oi=df_oi, df_liq=df_liq,
        )
        # 防 add_indicators 全局 dropna 把合约起点前的 OHLCV 行误删
        for col in (
            "funding_rate", "sum_open_interest", "sum_open_interest_value",
            "liq_long_usd", "liq_short_usd", "liq_total_usd",
        ):
            if col in df_tf.columns:
                df_tf[col] = df_tf[col].fillna(0.0)
        if "funding_origin" in df_tf.columns:
            df_tf["funding_origin"] = df_tf["funding_origin"].fillna("")

    return df_tf, tf


def apply_elimination(
    sig_cols: list[str],
    sharpe: dict[str, float],
    corr: pd.DataFrame,
    sharpe_threshold: float,
    corr_threshold: float,
) -> tuple[list[str], list[dict]]:
    """Apply rules. Returns (kept_signals, elimination_log_entries)."""
    kept = set(sig_cols)
    entries: list[dict] = []

    # Rule 1: Sharpe gate (exempt state-only signals)
    for col in sig_cols:
        if col in STATE_ONLY_SIGNALS:
            continue
        if sharpe[col] < sharpe_threshold:
            kept.discard(col)
            entries.append({
                "signal": col,
                "reason": "sharpe_below_threshold",
                "detail": f"Sharpe={sharpe[col]:+.3f} < {sharpe_threshold}",
            })

    # Rule 2: correlation redundancy — greedy by |ρ| descending
    pairs: list[tuple[str, str, float]] = []
    for i, a in enumerate(sig_cols):
        for b in sig_cols[i + 1:]:
            pairs.append((a, b, float(corr.loc[a, b])))
    pairs.sort(key=lambda t: abs(t[2]), reverse=True)

    for a, b, rho in pairs:
        if abs(rho) <= corr_threshold:
            break
        if a not in kept or b not in kept:
            continue
        # Keep the higher-Sharpe one
        loser = a if sharpe[a] < sharpe[b] else b
        winner = b if loser == a else a
        kept.discard(loser)
        entries.append({
            "signal": loser,
            "reason": "correlation_redundancy",
            "detail": (
                f"|ρ|={abs(rho):.3f} ({rho:+.3f}) with {winner} "
                f"(kept: Sharpe={sharpe[winner]:+.3f} vs {sharpe[loser]:+.3f})"
            ),
        })

    final_order = [c for c in sig_cols if c in kept]
    return final_order, entries


def write_yaml(
    path: Path,
    kept: list[str],
    eliminated: list[dict],
    sharpe: dict[str, float],
    source_tag: str,
    periods_per_year: int,
    sharpe_threshold: float,
    corr_threshold: float,
) -> None:
    payload = {
        "version": 1,
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "source": {
            "tag": source_tag,
            "warmup_window": SIGNAL_WARMUP_WINDOW,
            "periods_per_year": periods_per_year,
        },
        "thresholds": {
            "sharpe_min": sharpe_threshold,
            "pearson_abs_max": corr_threshold,
        },
        "signals": kept,
        "eliminated": [
            {
                **entry,
                "sharpe": round(sharpe[entry["signal"]], 4),
            }
            for entry in eliminated
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True, width=100)


def write_log(
    path: Path,
    kept: list[str],
    eliminated: list[dict],
    sharpe: dict[str, float],
    sharpe_threshold: float,
    corr_threshold: float,
) -> None:
    lines: list[str] = []
    lines.append("# Signal Elimination Log — TC-A7\n")
    lines.append(f"- Generated: `{dt.datetime.now().isoformat(timespec='seconds')}`  ")
    lines.append(f"- Sharpe gate: `< {sharpe_threshold}`  ")
    lines.append(f"- Correlation gate: `|ρ| > {corr_threshold}`  ")
    lines.append(
        f"- State-only signals exempt from Sharpe gate: "
        f"`{sorted(STATE_ONLY_SIGNALS)}`\n"
    )
    lines.append("## Kept signals\n")
    for col in kept:
        tag = " *(state-only)*" if col in STATE_ONLY_SIGNALS else ""
        lines.append(f"- `{col}` — Sharpe `{sharpe[col]:+.3f}`{tag}")
    lines.append("")
    lines.append("## Eliminated signals\n")
    if not eliminated:
        lines.append("_None._\n")
    else:
        lines.append("| Signal | Sharpe | Reason | Detail |")
        lines.append("|--------|--------|--------|--------|")
        for e in eliminated:
            lines.append(
                f"| `{e['signal']}` | {sharpe[e['signal']]:+.3f} | "
                f"{e['reason']} | {e['detail']} |"
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None,
                        help="Training-style yaml (e.g. config/stage2_1h_signal.yaml). "
                             "When set, loads from DB and resamples; --cache is ignored.")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE,
                        help="Cached 4h OHLCV parquet (legacy path, ignored if --config set)")
    parser.add_argument("--sharpe-threshold", type=float, default=-0.2)
    parser.add_argument("--corr-threshold", type=float, default=0.95)
    parser.add_argument("--output", type=Path, default=None,
                        help="Output signals yaml. Default: config/signals_v1.yaml for "
                             "--cache, config/signals_v1_{tf}.yaml for --config")
    parser.add_argument("--log", type=Path, default=None,
                        help="Output decision log. Default matches --output naming.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print decisions but do not write files")
    parser.add_argument("--with-contracts", action="store_true",
                        help="跨库 join funding/OI/liquidation 让候选池含合约信号"
                             "（与 signal_linear_baseline.py 对称）")
    args = parser.parse_args()

    if args.config is not None:
        if not args.config.exists():
            print(f"ERROR: config not found: {args.config}", file=sys.stderr)
            return 1
        df_tf, tf = load_df_from_config(args.config, with_contracts=args.with_contracts)
        if tf not in BARS_PER_YEAR:
            print(f"ERROR: unknown timeframe '{tf}'", file=sys.stderr)
            return 1
        periods_per_year = BARS_PER_YEAR[tf]
        source_tag = f"config:{args.config.name} (tf={tf})"
        print(f"DB load: timeframe={tf}, bars={len(df_tf)}, bars/year={periods_per_year}")
        df_ind = add_indicators(df_tf)
        df = add_signals(df_ind)
        df = add_contract_signals(df)
        output_path = args.output or REPO_ROOT / "config" / f"signals_v1_{tf}.yaml"
        log_path = args.log or REPO_ROOT / "config" / f"signal_elimination_log_{tf}.md"
    else:
        if not args.cache.exists():
            print(f"ERROR: cache file not found: {args.cache}", file=sys.stderr)
            print("Run notebooks/signal_sanity.ipynb first to populate the cache.",
                  file=sys.stderr)
            return 1
        print(f"Loading cached 4h OHLCV: {args.cache}")
        df_raw = pd.read_parquet(args.cache)
        df_ind = add_indicators(df_raw)
        df = add_signals(df_ind)
        df = add_contract_signals(df)
        periods_per_year = PERIODS_PER_YEAR
        source_tag = f"cache:{args.cache.name}"
        output_path = args.output or REPO_ROOT / "config" / "signals_v1.yaml"
        log_path = args.log or REPO_ROOT / "config" / "signal_elimination_log.md"

    df_post = df.iloc[SIGNAL_WARMUP_WINDOW:].reset_index(drop=True)
    sig_cols = signal_columns(df_post)
    print(f"Rows post-warmup: {len(df_post):,}, signals: {len(sig_cols)}")

    sharpe, corr = compute_stats(df_post, sig_cols, periods_per_year)
    kept, eliminated = apply_elimination(
        sig_cols, sharpe, corr,
        sharpe_threshold=args.sharpe_threshold,
        corr_threshold=args.corr_threshold,
    )

    print(f"\nKept ({len(kept)}):")
    for col in kept:
        tag = " (state-only)" if col in STATE_ONLY_SIGNALS else ""
        print(f"  ✓ {col:<32s} Sharpe={sharpe[col]:+.3f}{tag}")

    print(f"\nEliminated ({len(eliminated)}):")
    for e in eliminated:
        print(f"  ✗ {e['signal']:<32s} Sharpe={sharpe[e['signal']]:+.3f}  "
              f"[{e['reason']}] {e['detail']}")

    if args.dry_run:
        print("\n--dry-run: no files written.")
        return 0

    write_yaml(output_path, kept, eliminated, sharpe,
               source_tag, periods_per_year,
               args.sharpe_threshold, args.corr_threshold)
    write_log(log_path, kept, eliminated, sharpe,
              args.sharpe_threshold, args.corr_threshold)
    print(f"\nWrote {output_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {log_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
