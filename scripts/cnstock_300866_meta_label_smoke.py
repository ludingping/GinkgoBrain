"""
300866 meta-label v1 OOS smoke backtest.

读取 notebooks/cnstock_300866_signals.csv（来源：notebooks/cnstock_300866_pattern_discovery.ipynb §9
事件去重 scoreboard），把 8 个 meta-label 信号映射回原始 day 数据，按 70/30 时间序列切分做 OOS 测试，
单信号 + 组合两层验证，verdict 写到 docs/GinkgoBrain/复盘/300866_meta_label_v1.md。

⚠️ 已知偏差：信号库由全样本 pattern discovery 产出，TEST 段的数据已被 §3.4 看到过。
本测试只能回答"信号在 TEST 段是否仍能机械地复现"，不是干净 OOS。
真正干净的 OOS 需要 v2：先按 TRAIN 重新跑 §3.4，再在 TEST 上验证。

Usage:
    uv run python scripts/cnstock_300866_meta_label_smoke.py
    uv run python scripts/cnstock_300866_meta_label_smoke.py --train-frac 0.7
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.cnstock_intraday_reversal_smoke import load_aligned_5m  # noqa: E402

# ─── Constants ──────────────────────────────────────────────────────────────

DEFAULT_SIGNALS_CSV = REPO_ROOT / "notebooks" / "cnstock_300866_signals.csv"
DEFAULT_VERDICT     = REPO_ROOT / "docs" / "GinkgoBrain" / "复盘" / "300866_meta_label_v1.md"
TRADING_DAYS_PER_YEAR = 252

# 把 CSV 里的 indicator 名映射回 build_regime_columns() 里的列名
INDICATOR_TO_REGIME_COL: dict[str, str] = {
    "vol_21d":      "regime_vol",
    "trend":        "regime_trend",
    "drawdown_60d": "regime_dd",
    "volume_z_21d": "regime_vol_z",
    "rsi_14":       "regime_rsi",
}


# ─── Indicator computation (mirrors notebook §3.1-§3.3) ─────────────────────


def build_regime_columns(df_day: pd.DataFrame) -> pd.DataFrame:
    """给 df_day 加上 5 个 regime 列 — 必须与 notebook 完全一致，否则 trigger 对不上。"""
    df = df_day.copy()
    df["ret"]     = df["close_price"].pct_change()
    df["vol_21d"] = df["ret"].rolling(21).std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    # vol regime
    vp = df["vol_21d"] * 100
    lo, hi = vp.quantile(0.33), vp.quantile(0.67)
    df["regime_vol"] = pd.cut(vp, bins=[-np.inf, lo, hi, np.inf], labels=["低 vol", "中 vol", "高 vol"])

    # trend regime
    df["ma_60"]       = df["close_price"].rolling(60).mean()
    df["ma_60_slope"] = df["ma_60"].diff(5)

    def _trend(row: pd.Series) -> str | float:
        if pd.isna(row["ma_60"]) or pd.isna(row["ma_60_slope"]):
            return np.nan
        above = row["close_price"] > row["ma_60"]
        up    = row["ma_60_slope"] > 0
        if above and up:           return "1.强势(上+涨)"
        if above and not up:       return "2.顶背离(上+跌)"
        if not above and up:       return "3.反弹(下+涨)"
        return "4.弱势(下+跌)"

    df["regime_trend"] = df.apply(_trend, axis=1)

    # drawdown regime
    rh = df["close_price"].rolling(60, min_periods=20).max()
    df["dd_60d"] = df["close_price"] / rh - 1

    def _dd(d: float) -> str | float:
        if pd.isna(d):  return np.nan
        if d >= -0.05:  return "A.高点±5%"
        if d >= -0.15:  return "B.回撤5-15%"
        if d >= -0.25:  return "C.回撤15-25%"
        return                 "D.深回撤>25%"

    df["regime_dd"] = df["dd_60d"].apply(_dd)

    # volume z-score regime
    log_vol = np.log(df["volume"].replace(0, np.nan))
    df["vol_z_21d"] = (log_vol - log_vol.rolling(21).mean()) / log_vol.rolling(21).std()
    df["regime_vol_z"] = pd.cut(
        df["vol_z_21d"],
        bins=[-np.inf, -1.0, -0.5, 0.5, 1.0, np.inf],
        labels=["A.缩量<-1σ", "B.偏缩-1~-0.5σ", "C.中性±0.5σ", "D.偏放0.5~1σ", "E.放量>1σ"],
    )

    # RSI regime (Wilder)
    delta = df["close_price"].diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_g = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_l = loss.ewm(alpha=1/14, adjust=False).mean()
    df["rsi_14"] = 100 - 100 / (1 + avg_g / avg_l.replace(0, np.nan))
    df["regime_rsi"] = pd.cut(
        df["rsi_14"],
        bins=[0, 30, 50, 70, 100],
        labels=["A.超卖<30", "B.弱30-50", "C.强50-70", "D.超买>70"],
        include_lowest=True,
    )

    return df


# ─── Signal trigger / scoring ───────────────────────────────────────────────


def event_trigger_mask(regime_series: pd.Series, prev: str, to: str) -> pd.Series:
    """v2 directional：True on day where regime transitions from `prev` to `to`."""
    today_match = (regime_series == to).fillna(False)
    yest_match  = (regime_series.shift(1) == prev).fillna(False)
    return today_match & yest_match


@dataclass
class SignalScore:
    n: int
    mean_pct: float
    win_pct: float
    t: float
    sum_pct: float


def score_signal(
    df: pd.DataFrame,
    signal: pd.Series,
    period_mask: pd.Series,
    *,
    entry_lag: int = 1,
) -> SignalScore:
    """单个信号在 period_mask 范围内的触发收益统计。"""
    regime_col = INDICATOR_TO_REGIME_COL[signal["indicator"]]
    triggers = event_trigger_mask(
        df[regime_col], signal["prev_regime"], signal["regime"]
    ) & period_mask

    h = int(signal["h"])
    p = df["close_price"]
    fwd = p.shift(-(entry_lag + h)) / p.shift(-entry_lag) - 1

    rets = fwd[triggers].dropna()
    sign = +1 if signal["direction"] == "long" else -1
    pnl = rets * sign

    n = len(pnl)
    if n == 0:
        return SignalScore(n=0, mean_pct=float("nan"), win_pct=float("nan"),
                           t=float("nan"), sum_pct=float("nan"))

    mu, sd = pnl.mean(), pnl.std()
    t = mu / (sd / np.sqrt(n)) if sd > 0 and n > 1 else float("nan")
    return SignalScore(
        n=n,
        mean_pct=float(round(mu * 100, 3)),
        win_pct=float(round((pnl > 0).mean() * 100, 1)),
        t=float(round(t, 2)) if not np.isnan(t) else float("nan"),
        sum_pct=float(round(pnl.sum() * 100, 2)),
    )


# ─── Portfolio simulation ───────────────────────────────────────────────────


def simulate_portfolio(
    df: pd.DataFrame,
    signals: pd.DataFrame,
) -> tuple[pd.Series, np.ndarray]:
    """
    根据所有 signal 在每个交易日的活跃仓位累加得到组合 daily P&L（log return 加权）。

    返回:
        daily_pnl  shape (T,) 索引 = df.index
        position   shape (T, n_signals) 浮点矩阵，每列是一个信号的逐日仓位 (signed)
    """
    n_total = len(df)
    n_sigs  = len(signals)
    log_ret = np.log(df["close_price"]).diff().to_numpy()
    position = np.zeros((n_total, n_sigs), dtype=np.float64)

    for col_idx, sig in enumerate(signals.itertuples(index=False)):
        regime_col = INDICATOR_TO_REGIME_COL[sig.indicator]
        triggers   = event_trigger_mask(df[regime_col], sig.prev_regime, sig.regime).to_numpy()

        h = int(sig.h)
        sign = +1 if sig.direction == "long" else -1
        size = sig.size_pct / 100.0

        # 触发日 t_idx，T+1 entry，持仓 h 日 → 在 [t_idx+2, t_idx+h+2) 区间累计 log_ret
        for t_idx in np.where(triggers)[0]:
            start = t_idx + 2
            end   = min(t_idx + h + 2, n_total)
            if start >= n_total:
                continue
            position[start:end, col_idx] += sign * size

    daily_pnl_arr = (position * log_ret[:, None]).sum(axis=1)
    daily_pnl = pd.Series(daily_pnl_arr, index=df.index)
    return daily_pnl, position


# ─── Verdict logic ──────────────────────────────────────────────────────────


def signal_verdict(row: dict) -> str:
    if row["n_test"] < 3:
        return "💤 太少样本"
    if np.isnan(row["mean_test"]) or np.isnan(row["mean_train"]):
        return "💤 太少样本"
    same_dir = np.sign(row["mean_test"]) == np.sign(row["mean_train"])
    if not same_dir:
        return "❌ 方向反转"
    if abs(row["t_test"]) >= 1.5:
        return "✅ 可信"
    return "⚠️ 弱"


def overall_verdict(
    n_pass: int,
    test_strat_log: float,
    test_bh_log: float,
    test_sharpe: float,
) -> str:
    if test_sharpe > 0.5 and test_strat_log > test_bh_log and n_pass >= 3:
        return "✅ PASS — 信号库具备 OOS 持续力"
    if test_sharpe > 0 and n_pass >= 1:
        return "⚠️ MIXED — 部分信号可信，需筛选"
    return "❌ FAIL — 信号库 OOS 无效"


# ─── Main ───────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="300866 meta-label v1 OOS smoke")
    parser.add_argument("--code",         default="300866")
    parser.add_argument("--start",        default="2023-04-04")
    parser.add_argument("--end",          default="2026-04-22")
    parser.add_argument("--train-frac",   type=float, default=0.7)
    parser.add_argument("--signals-csv",  default=str(DEFAULT_SIGNALS_CSV))
    parser.add_argument("--verdict-path", default=str(DEFAULT_VERDICT))
    args = parser.parse_args()

    print(f"=== 300866 meta-label v1 OOS smoke ===")
    print(f"data: {args.start} → {args.end}")
    print(f"train_frac: {args.train_frac:.0%}")

    # 1) 数据 + regime 列
    _, df_day, _ = load_aligned_5m(args.code, args.start, args.end)
    df_day = df_day.copy()
    df_day["date"] = pd.to_datetime(df_day["date_cst"])
    df_day = df_day.sort_values("date").reset_index(drop=True)
    df = build_regime_columns(df_day)

    # 2) train/test split
    n_total = len(df)
    n_train = int(n_total * args.train_frac)
    train_mask = pd.Series(False, index=df.index)
    train_mask.iloc[:n_train] = True
    test_mask = ~train_mask

    train_start = df["date"].iloc[0].date()
    train_end   = df["date"].iloc[n_train - 1].date()
    test_start  = df["date"].iloc[n_train].date()
    test_end    = df["date"].iloc[-1].date()
    print(f"\ntrain: {train_start} → {train_end}  ({n_train} days)")
    print(f"test:  {test_start} → {test_end}  ({n_total - n_train} days)")

    # 3) 加载信号
    signals = pd.read_csv(args.signals_csv)
    print(f"\nloaded {len(signals)} signals from {args.signals_csv}")
    print(f"  tier dist: {dict(signals['tier'].value_counts())}")

    # 4) 单信号级 OOS 复现表
    rows = []
    for _, sig in signals.iterrows():
        train_score = score_signal(df, sig, train_mask)
        test_score  = score_signal(df, sig, test_mask)
        rows.append({
            "signal_id": sig["signal_id"],
            "tier":      sig["tier"],
            "direction": sig["direction"],
            "h":         int(sig["h"]),
            "size_pct":  int(sig["size_pct"]),
            "n_train":    train_score.n,
            "mean_train": train_score.mean_pct,
            "t_train":    train_score.t,
            "n_test":     test_score.n,
            "mean_test":  test_score.mean_pct,
            "t_test":     test_score.t,
            "sum_test":   test_score.sum_pct,
        })

    out = pd.DataFrame(rows)
    out["verdict"] = out.apply(lambda r: signal_verdict(r.to_dict()), axis=1)

    cols_show = ["signal_id", "tier", "direction", "h", "size_pct",
                 "n_train", "mean_train", "t_train",
                 "n_test", "mean_test", "t_test", "sum_test",
                 "verdict"]
    print("\n=== 单信号 OOS 复现表 ===")
    print(out[cols_show].to_string(index=False))

    # 5) 组合 P&L
    daily_pnl, position = simulate_portfolio(df, signals)
    log_ret = np.log(df["close_price"]).diff()

    train_pnl = daily_pnl[train_mask].dropna()
    test_pnl  = daily_pnl[test_mask].dropna()
    train_bh  = log_ret[train_mask].dropna().sum()
    test_bh   = log_ret[test_mask].dropna().sum()

    train_strat = train_pnl.sum()
    test_strat  = test_pnl.sum()

    def _sharpe(pnl: pd.Series) -> float:
        if len(pnl) == 0 or pnl.std() == 0:
            return float("nan")
        return float(pnl.mean() / pnl.std() * np.sqrt(TRADING_DAYS_PER_YEAR))

    train_sharpe = _sharpe(train_pnl)
    test_sharpe  = _sharpe(test_pnl)

    avg_gross_train = float(np.abs(position[:n_train]).sum(axis=1).mean())
    avg_gross_test  = float(np.abs(position[n_train:]).sum(axis=1).mean())

    print(f"\n=== 组合 P&L (log) ===")
    print(f"TRAIN: strat = {train_strat*100:+.1f}% vs B&H = {train_bh*100:+.1f}%   Δ={(train_strat-train_bh)*100:+.1f}%   Sharpe={train_sharpe:+.2f}   gross={avg_gross_train:.2f}x")
    print(f"TEST : strat = {test_strat*100:+.1f}% vs B&H = {test_bh*100:+.1f}%   Δ={(test_strat-test_bh)*100:+.1f}%   Sharpe={test_sharpe:+.2f}   gross={avg_gross_test:.2f}x")

    # 6) 总判定
    n_pass = int((out["verdict"] == "✅ 可信").sum())
    n_weak = int((out["verdict"] == "⚠️ 弱").sum())
    n_fail = int((out["verdict"] == "❌ 方向反转").sum())
    n_few  = int((out["verdict"] == "💤 太少样本").sum())
    verdict = overall_verdict(n_pass, test_strat, test_bh, test_sharpe)

    print(f"\n=== 总判定: {verdict} ===")
    print(f"  signals: ✅{n_pass}  ⚠️{n_weak}  ❌{n_fail}  💤{n_few}")

    # 7) verdict markdown
    verdict_path = Path(args.verdict_path)
    verdict_path.parent.mkdir(parents=True, exist_ok=True)

    md = f"""# 300866 meta-label v1 OOS verdict

**生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M")}
**判定**: {verdict}

## 数据
- 周期: {args.start} → {args.end}（共 {n_total} 个交易日）
- TRAIN: {train_start} → {train_end}（{n_train} 天，{args.train_frac:.0%}）
- TEST : {test_start} → {test_end}（{n_total - n_train} 天）

## 信号库
- 来源: `{Path(args.signals_csv).relative_to(REPO_ROOT)}`
- 共 {len(signals)} 条；tier 分布: {dict(signals['tier'].value_counts())}

## ⚠️ 已知偏差
信号库由**全样本** pattern discovery 产出，TEST 段的数据已被 §3.4 看到过。
本测试只能回答"信号在 TEST 段是否仍能机械地复现"，不是干净 OOS。
真正干净 OOS 需要 v2：先按 TRAIN 重新跑 §3.4，再在 TEST 上验证。

## Signal-level 复现

```
{out[cols_show].to_string(index=False)}
```

- ✅ 可信 (|t_test|≥1.5 同向): {n_pass}
- ⚠️ 弱 (同向但 t 不显著): {n_weak}
- ❌ 方向反转: {n_fail}
- 💤 样本不足 (n_test<3): {n_few}

## 组合 P&L (log return)

| 段 | 策略 | Buy & Hold | Δ | Sharpe | avg gross |
|---|---|---|---|---|---|
| TRAIN | {train_strat*100:+.1f}% | {train_bh*100:+.1f}% | {(train_strat-train_bh)*100:+.1f}% | {train_sharpe:+.2f} | {avg_gross_train:.2f}x |
| TEST  | {test_strat*100:+.1f}% | {test_bh*100:+.1f}% | {(test_strat-test_bh)*100:+.1f}% | {test_sharpe:+.2f} | {avg_gross_test:.2f}x |

## 下一步

{"- 继续推进：v2 重做（按 TRAIN 重跑 pattern discovery 再 OOS）" if n_pass >= 3 else "- 信号库重做或换股票；或先做单信号深挖（如 trend.反弹 假反弹陷阱）"}
"""
    verdict_path.write_text(md, encoding="utf-8")
    print(f"\n📝 verdict written to {verdict_path}")


if __name__ == "__main__":
    main()
