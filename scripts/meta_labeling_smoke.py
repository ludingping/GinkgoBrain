"""Meta-Labeling smoke test: 验证"用 ML 过滤基础信号假突破"是否在 ETH 上有 alpha。

设计（Lopez de Prado《Advances in Financial Machine Learning》§3.5）:
  Layer 1 (规则信号): SMA20 cross — close 上穿 SMA20 时给一个 long entry 候选
  Triple-Barrier 标注:
    - 上界 (take-profit) = entry + tp_atr_mult × ATR
    - 下界 (stop-loss)   = entry - sl_atr_mult × ATR
    - 时间界 = horizon bars
    - 哪个先触达 → label = 1（上界）/ 0（下/时间界）
  Layer 2 (ML): LightGBM binary classifier
    features = 22 signal values at entry bar
    target  = label (0/1)

判据:
  - AUC > 0.55 → Meta-Labeling 在 ETH 上有效，进入 Day 2 正式实施
  - AUC ~ 0.50 → ML 在过滤上也无能力，本路径关闭
  - AUC < 0.50 → 信号反向，可考虑 invert 或直接关闭

Usage:
    cd /home/davidlyu/projects/GinkgoBrain
    CUDA_VISIBLE_DEVICES= uv run python scripts/meta_labeling_smoke.py \\
        --timeframe 1h --symbol ETH/USDT --horizon 24
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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
)


def load_full_df(symbol: str, timeframe: str, start: str, end: str) -> pd.DataFrame:
    """加载 OHLCV + 合约 + 信号，返回完整 df。"""
    print(f"Loading {symbol} {timeframe} from DB ({start} → {end})...")
    df_raw = read_ohlcv(symbol, start=start, end=end, only_closed=True)
    ts = df_raw["timestamp"]
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    df_raw["timestamp"] = ts.dt.tz_convert("Asia/Shanghai")
    df_tf = df_raw if timeframe == "1m" else resample_ohlcv(df_raw, timeframe)
    print(f"  {len(df_tf):,} {timeframe} bars after resample")

    print(f"Loading contract metadata for {symbol}...")
    ccxt_symbol = symbol if "/" in symbol else f"{symbol[:-4]}/USDT"
    df_funding = read_funding(ccxt_symbol, start=start, end=end)
    df_oi      = read_open_interest(ccxt_symbol, start=start, end=end)
    df_liq     = read_liquidation_agg(ccxt_symbol, start=start, end=end)
    for sub in (df_funding, df_oi, df_liq):
        if sub.empty:
            continue
        sub_ts = sub["timestamp"]
        if sub_ts.dt.tz is None:
            sub_ts = sub_ts.dt.tz_localize("UTC")
        sub["timestamp"] = sub_ts.dt.tz_convert("Asia/Shanghai")
    df_tf = merge_contract_data(df_tf, df_funding=df_funding, df_oi=df_oi, df_liq=df_liq)
    for col in ("funding_rate", "sum_open_interest", "sum_open_interest_value",
                "liq_long_usd", "liq_short_usd", "liq_total_usd"):
        if col in df_tf.columns:
            df_tf[col] = df_tf[col].fillna(0.0)
    if "funding_origin" in df_tf.columns:
        df_tf["funding_origin"] = df_tf["funding_origin"].fillna("")

    df = add_indicators(df_tf)
    df = add_signals(df)
    df = add_contract_signals(df)
    df = df.iloc[SIGNAL_WARMUP_WINDOW:].reset_index(drop=True)
    return df


def layer1_sma_cross(df: pd.DataFrame, sma_window: int = 20) -> np.ndarray:
    """Layer 1 规则：close 上穿 SMA(sma_window) 返回 long entry mask。

    Returns:
        boolean array, True at bars where close crosses ABOVE sma (long entry).
    """
    sma = df["close"].rolling(sma_window).mean()
    above = df["close"] > sma
    cross_up = above & ~above.shift(1, fill_value=False)
    return cross_up.fillna(False).values


def triple_barrier_labels(
    df: pd.DataFrame,
    entry_mask: np.ndarray,
    horizon: int,
    tp_atr_mult: float,
    sl_atr_mult: float,
) -> tuple[np.ndarray, np.ndarray]:
    """三障碍标注（long-only）。

    Returns:
        (labels, valid_idx)
        labels ∈ {0, 1}: 1 = 上界先触达，0 = 下界或时间界先触达
        valid_idx: 实际有 entry 的 bar 索引（去除尾部 horizon 不足的）
    """
    close = df["close"].values
    atr = df["atr"].values
    n = len(df)
    entries = np.where(entry_mask)[0]
    valid = entries[entries < n - horizon - 1]

    labels = np.zeros(len(valid), dtype=int)
    for i, e in enumerate(valid):
        tp = close[e] + tp_atr_mult * atr[e]
        sl = close[e] - sl_atr_mult * atr[e]
        path = close[e + 1: e + 1 + horizon]
        hit_tp = np.where(path >= tp)[0]
        hit_sl = np.where(path <= sl)[0]
        first_tp = hit_tp[0] if len(hit_tp) > 0 else np.inf
        first_sl = hit_sl[0] if len(hit_sl) > 0 else np.inf
        if first_tp < first_sl:
            labels[i] = 1  # take-profit 先触达
        else:
            labels[i] = 0  # stop-loss 先触达 OR 时间界
    return labels, valid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="ETH/USDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--start", default="2022-01-01 00:00:00+00:00")
    parser.add_argument("--end", default="2026-04-13 00:00:00+00:00")
    parser.add_argument("--sma-window", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=24,
                        help="时间界 bar 数（1h × 24 = 1d；4h × 6 = 1d）")
    parser.add_argument("--tp-atr", type=float, default=2.0,
                        help="take-profit ATR 倍数")
    parser.add_argument("--sl-atr", type=float, default=1.0,
                        help="stop-loss ATR 倍数（默认 2:1 risk/reward）")
    parser.add_argument("--train-ratio", type=float, default=0.50)
    parser.add_argument("--threshold", type=float, default=0.55,
                        help="预测概率阈值（>thr 才执行 trade）")
    parser.add_argument("--invert", action="store_true",
                        help="反转 Layer 1：cross_down 时出 long entry（mean-reversion 规则）")
    args = parser.parse_args()

    df = load_full_df(args.symbol, args.timeframe, args.start, args.end)
    print(f"  total bars after warmup: {len(df):,}")

    entry_mask = layer1_sma_cross(df, sma_window=args.sma_window)
    if args.invert:
        # 反转：上穿原来是 entry → 改为 下穿/未上穿 是 entry
        # 实际意图：cross_down 出 long entry（逆势抄底）
        sma = df["close"].rolling(args.sma_window).mean()
        below = df["close"] < sma
        cross_down = below & ~below.shift(1, fill_value=False)
        entry_mask = cross_down.fillna(False).values
        print("[INVERTED] Layer 1: cross-DOWN as long entry (mean-reversion)")
    n_entries_total = int(entry_mask.sum())
    print(f"\nLayer 1 (SMA{args.sma_window} cross{' DOWN' if args.invert else ' UP'}): {n_entries_total} entry candidates")

    if n_entries_total < 100:
        print("ERROR: too few entries (<100), Layer 1 rule too strict")
        return 1

    labels, valid_idx = triple_barrier_labels(
        df, entry_mask, args.horizon, args.tp_atr, args.sl_atr,
    )
    n = len(valid_idx)
    print(f"  valid (with full horizon): {n}")
    print(f"  positive label rate (no ML): {labels.mean():.3f}  ← Layer 1 baseline win rate")

    sig_cols = [c for c in df.columns if c.startswith("sig_")]
    print(f"  features: {len(sig_cols)} signals")
    X = df.iloc[valid_idx][sig_cols].values
    y = labels

    cut = int(n * args.train_ratio)
    X_train, X_test = X[:cut], X[cut:]
    y_train, y_test = y[:cut], y[cut:]
    print(f"\nSplit: train={len(y_train)} ({y_train.mean():.3f} pos)  test={len(y_test)} ({y_test.mean():.3f} pos)")

    print("\nTraining LightGBM binary classifier...")
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score, accuracy_score

    clf = lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1,
    )
    clf.fit(X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(30, verbose=False)])
    n_trees = clf.best_iteration_ or clf.n_estimators
    print(f"  trained {n_trees} trees (early stopped if applicable)")

    proba_test = clf.predict_proba(X_test)[:, 1]
    pred_test = (proba_test >= args.threshold).astype(int)
    auc = roc_auc_score(y_test, proba_test) if len(np.unique(y_test)) > 1 else float("nan")
    acc_at_thr = accuracy_score(y_test, pred_test)

    test_idx = valid_idx[cut:]
    close = df["close"].values
    atr_at = df["atr"].values
    pnls = []
    for i, gi in enumerate(test_idx):
        if proba_test[i] < args.threshold:
            continue
        tp = close[gi] + args.tp_atr * atr_at[gi]
        sl = close[gi] - args.sl_atr * atr_at[gi]
        path = close[gi + 1: gi + 1 + args.horizon]
        hit_tp_idx = np.where(path >= tp)[0]
        hit_sl_idx = np.where(path <= sl)[0]
        first_tp = hit_tp_idx[0] if len(hit_tp_idx) > 0 else np.inf
        first_sl = hit_sl_idx[0] if len(hit_sl_idx) > 0 else np.inf
        if first_tp < first_sl:
            pnls.append(args.tp_atr * atr_at[gi] / close[gi])
        elif first_sl < first_tp:
            pnls.append(-args.sl_atr * atr_at[gi] / close[gi])
        else:
            end_close = close[gi + args.horizon] if gi + args.horizon < len(close) else close[-1]
            pnls.append((end_close - close[gi]) / close[gi])
    pnls = np.array(pnls)
    n_filtered = len(pnls)
    if n_filtered > 0:
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        win_rate = (pnls > 0).mean()
        profit_factor = wins.sum() / abs(losses.sum()) if losses.sum() < 0 else float("inf")
        avg_pnl = pnls.mean()
    else:
        win_rate = profit_factor = avg_pnl = float("nan")

    print("\n" + "=" * 64)
    print(f"=== Meta-Labeling Smoke Result ({args.symbol} {args.timeframe}) ===")
    print("=" * 64)
    print(f"  Layer 1 baseline win rate (no ML)   : {labels.mean():.3f}")
    print(f"  AUC (test set, ML filter)           : {auc:.4f}")
    print(f"  Accuracy at thr={args.threshold}             : {acc_at_thr:.4f}")
    print(f"  Filtered trades (test)              : {n_filtered}/{len(test_idx)} ({n_filtered/max(len(test_idx),1):.1%})")
    print(f"  Filtered win rate                   : {win_rate:.3f}")
    print(f"  Profit factor                       : {profit_factor:.3f}")
    print(f"  Avg per-trade return                : {avg_pnl:+.4%}")
    print()
    print("Verdict:")
    if np.isnan(auc):
        print("  ⚠️  AUC is NaN (only one class in test)")
    elif auc > 0.55:
        print(f"  ✅ AUC {auc:.4f} > 0.55 — Meta-Labeling has signal, proceed to Day 2 full impl")
    elif auc > 0.52:
        print(f"  🟡 AUC {auc:.4f} > 0.52 — marginal, need more Layer 1 rules + larger feature window")
    elif auc > 0.48:
        print(f"  ⚠️  AUC {auc:.4f} ≈ 0.50 — ML can't filter; consider Layer 1 rule change or invert")
    else:
        print(f"  ❌ AUC {auc:.4f} < 0.48 — signal inverted; flip predictions or close path")
    print()

    print("Top 10 feature importance:")
    fi = pd.Series(clf.feature_importances_, index=sig_cols).sort_values(ascending=False)
    for name, imp in fi.head(10).items():
        print(f"  {name:<32s} {int(imp)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
