"""
A 股多股票 meta-label 横向稳健性扫描 v1.

对每只股票跑 notebooks/cnstock_300866_pattern_discovery.ipynb §3.1-§3.4 的
事件去重 meta-label 框架，产出 per-stock 信号库 + 横向 t-stat 矩阵。

核心问题：哪些 meta-label 在多只股票上**横截面一致显著**（≥3/4 stocks
同向 |t|≥2）？这种横向稳健的信号才有泛化价值，单股 OOS 显著但其他股票上
不显著的多半是数据集过拟合。

⚠️ 默认时间窗口与 300866 完全一致（2023-04-04 → 2026-04-22），便于横向比较。

Usage:
    uv run python scripts/cnstock_multi_stock_meta_label.py
    uv run python scripts/cnstock_multi_stock_meta_label.py \\
        --codes 300866,002594,600765,600893 \\
        --start 2023-04-04 --end 2026-04-22
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.db import read_cnstock_kline_day  # noqa: E402

# ─── Defaults ───────────────────────────────────────────────────────────────

DEFAULT_CODES = ["300866", "002594", "600765", "600893"]
CODE_NAMES = {
    "300866": "安克创新",
    "002594": "比亚迪",
    "600765": "中航重机",
    "600893": "航发动力",
}

DEFAULT_OUT_DIR     = REPO_ROOT / "notebooks"
DEFAULT_VERDICT     = REPO_ROOT / "docs" / "GinkgoBrain" / "复盘" / "cross_stock_meta_label_v1.md"
TRADING_DAYS_PER_YEAR = 252

INDICATORS = ["vol_21d", "trend", "drawdown_60d", "volume_z_21d", "rsi_14"]

INDICATOR_TO_REGIME_COL = {
    "vol_21d":      "regime_vol",
    "trend":        "regime_trend",
    "drawdown_60d": "regime_dd",
    "volume_z_21d": "regime_vol_z",
    "rsi_14":       "regime_rsi",
}


# ─── Data loading ───────────────────────────────────────────────────────────


def load_day(code: str, start: str, end: str) -> pd.DataFrame:
    """Load forward-adjusted day K-line for a single A-share code."""
    df = read_cnstock_kline_day(code, start=start, end=end)
    df = df.copy()
    df["ts_cst"] = df["trade_time"].dt.tz_convert("Asia/Shanghai")
    df["date"]   = pd.to_datetime(df["ts_cst"].dt.date)
    return df.sort_values("date").reset_index(drop=True)


# ─── Indicator computation (mirrors notebook §3.1-§3.3) ─────────────────────


def build_regime_columns(df_day: pd.DataFrame) -> pd.DataFrame:
    """5 个 day 级 regime 列，必须与 notebook 完全一致。"""
    df = df_day.copy()
    df["ret"]     = df["close_price"].pct_change()
    df["vol_21d"] = df["ret"].rolling(21).std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    # vol regime — 每只股票自己的 33%/67% 分位
    vp = df["vol_21d"] * 100
    lo, hi = vp.quantile(0.33), vp.quantile(0.67)
    df["regime_vol"] = pd.cut(vp, bins=[-np.inf, lo, hi, np.inf], labels=["低 vol", "中 vol", "高 vol"])

    # trend regime — close vs MA60 × MA60 slope
    df["ma_60"]       = df["close_price"].rolling(60).mean()
    df["ma_60_slope"] = df["ma_60"].diff(5)

    def _trend(row: pd.Series):
        if pd.isna(row["ma_60"]) or pd.isna(row["ma_60_slope"]):
            return np.nan
        above = row["close_price"] > row["ma_60"]
        up    = row["ma_60_slope"] > 0
        if above and up:           return "1.强势(上+涨)"
        if above and not up:       return "2.顶背离(上+跌)"
        if not above and up:       return "3.反弹(下+涨)"
        return "4.弱势(下+跌)"

    df["regime_trend"] = df.apply(_trend, axis=1)

    # drawdown regime — 距 60 日高点回撤
    rh = df["close_price"].rolling(60, min_periods=20).max()
    df["dd_60d"] = df["close_price"] / rh - 1

    def _dd(d):
        if pd.isna(d):  return np.nan
        if d >= -0.05:  return "A.高点±5%"
        if d >= -0.15:  return "B.回撤5-15%"
        if d >= -0.25:  return "C.回撤15-25%"
        return                 "D.深回撤>25%"

    df["regime_dd"] = df["dd_60d"].apply(_dd)

    # volume z-score
    log_vol = np.log(df["volume"].replace(0, np.nan))
    df["vol_z_21d"] = (log_vol - log_vol.rolling(21).mean()) / log_vol.rolling(21).std()
    df["regime_vol_z"] = pd.cut(
        df["vol_z_21d"],
        bins=[-np.inf, -1.0, -0.5, 0.5, 1.0, np.inf],
        labels=["A.缩量<-1σ", "B.偏缩-1~-0.5σ", "C.中性±0.5σ", "D.偏放0.5~1σ", "E.放量>1σ"],
    )

    # RSI Wilder
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


# ─── Meta-label scoring ─────────────────────────────────────────────────────


def regime_score(
    df: pd.DataFrame,
    regime_col: str,
    *,
    horizons: tuple[int, ...] = (1, 5, 21),
    indicator_name: str | None = None,
    entry_lag: int = 1,
) -> pd.DataFrame:
    """Directional regime transition scoring (v2).

    事件 = (prev_regime, to_regime) 边——同一个 to_regime 来自不同 prev_regime
    时经济含义可能完全相反（例如 "C → B" 是反弹回浅回撤、"A → B" 是高位破位），
    必须按 4 元 (indicator, prev_regime, to_regime, h) 分别统计，避免方向混淆。

    无前视 T+1 entry：信号在 t 日收盘观察到 (prev → to) 切换 → t+1 收盘建仓 → t+1+h 收盘平仓。

    输出列：indicator, prev_regime, regime (=to_regime), h, n, mean_pct,
            sharpe_ann, win_pct, t
    """
    p = df["close_price"]
    today = df[regime_col]
    yest  = df[regime_col].shift(1)
    is_event = (today != yest) & today.notna() & yest.notna()
    name = indicator_name or regime_col

    rows = []
    for h in horizons:
        fwd = p.shift(-(entry_lag + h)) / p.shift(-entry_lag) - 1
        joined = pd.DataFrame({
            "prev_regime": yest.astype("object"),
            "to_regime":   today.astype("object"),
            "fwd":         fwd,
            "is_event":    is_event,
        })
        joined = joined[joined["is_event"]].dropna(subset=["fwd", "prev_regime", "to_regime"])

        for (prev, to), g in joined.groupby(["prev_regime", "to_regime"]):
            n = len(g)
            if n < 5:
                continue
            mu, sd = g["fwd"].mean(), g["fwd"].std()
            sharpe = (mu / sd) * np.sqrt(TRADING_DAYS_PER_YEAR / h) if sd > 0 else np.nan
            t      = (mu / (sd / np.sqrt(n)))                       if sd > 0 else np.nan
            rows.append({
                "indicator":   name,
                "prev_regime": str(prev),
                "regime":      str(to),
                "h":           h,
                "n":           n,
                "mean_pct":    round(mu * 100, 3),
                "sharpe_ann":  round(sharpe, 2) if not np.isnan(sharpe) else np.nan,
                "win_pct":     round((g["fwd"] > 0).mean() * 100, 1),
                "t":           round(t, 2)      if not np.isnan(t)      else np.nan,
            })
    return pd.DataFrame(rows)


def build_event_scoreboard(df: pd.DataFrame) -> pd.DataFrame:
    """对 5 个 indicator 跑 directional regime_score，合并成大表。"""
    parts = []
    for name in INDICATORS:
        regime_col = INDICATOR_TO_REGIME_COL[name]
        parts.append(regime_score(df, regime_col, indicator_name=name))
    out = pd.concat(parts, ignore_index=True)
    out["abs_t"] = out["t"].abs()
    return out


# ─── Per-stock signal library (与 notebook §9 一致) ─────────────────────────


def _sanitize_regime(s: str) -> str:
    """把 regime label 里非字母数字 / 非 CJK 的字符替换成 `_`，用于 signal_id。"""
    return "".join(c if c.isalnum() or "一" <= c <= "鿿" else "_" for c in s)


def build_signal_library(scoreboard: pd.DataFrame) -> pd.DataFrame:
    """从 directional event scoreboard → 分级 + 仓位 + signal_id 信号库 (v2)。

    每条信号是一个有向边 (prev_regime → to_regime) 在某 horizon 的统计。
    signal_id 格式: ``{indicator}::{sanitized_prev}->{sanitized_to}::h{h}``
    """
    sig = scoreboard[scoreboard["abs_t"] >= 2.0].copy()

    def _tier(row):
        t, n = row["abs_t"], row["n"]
        if t >= 4.0 and n >= 50:  return ("high",   90)
        if t >= 2.5 and n >= 20:  return ("medium", 70)
        if t >= 2.0 and n >= 10:  return ("low",    50)
        if t >= 2.0 and n >= 5:   return ("trial",  25)
        return ("reject", 0)

    sig[["tier", "size_pct"]] = sig.apply(lambda r: pd.Series(_tier(r)), axis=1)
    sig = sig[sig["tier"] != "reject"].copy()
    sig["direction"] = np.where(sig["mean_pct"] > 0, "long", "short")

    sig["signal_id"] = (
        sig["indicator"]
        + "::" + sig["prev_regime"].map(_sanitize_regime)
        + "->" + sig["regime"].map(_sanitize_regime)
        + "::h" + sig["h"].astype(str)
    )

    tier_order = {"high": 0, "medium": 1, "low": 2, "trial": 3}
    sig["_ord"] = sig["tier"].map(tier_order)
    sig = sig.sort_values(["_ord", "abs_t"], ascending=[True, False]).drop(columns=["_ord", "abs_t"])

    return sig[[
        "signal_id", "indicator", "prev_regime", "regime", "h", "direction",
        "n", "mean_pct", "win_pct", "sharpe_ann", "t", "tier", "size_pct",
    ]]


# ─── Cross-stock aggregation ────────────────────────────────────────────────


def hypothesis_id(indicator: str, prev_regime: str, regime: str, h: int) -> str:
    """跨股票 hypothesis 标识 (v2)：indicator + (prev->to) + h（不含 direction）。"""
    return f"{indicator}::{_sanitize_regime(prev_regime)}->{_sanitize_regime(regime)}::h{h}"


def aggregate_cross_stock(
    scoreboards: dict[str, pd.DataFrame],
    *,
    sig_threshold: float = 2.0,
    direction_threshold: float = 1.5,
) -> pd.DataFrame:
    """
    把每只股票的 directional event scoreboard 拼成 (hypothesis × stock) 矩阵，
    hypothesis = (indicator, prev_regime, to_regime, h) 4 元，
    然后给每个 hypothesis 打 robustness 标签。
    """
    codes = list(scoreboards.keys())

    all_h: dict[str, dict] = {}
    for code, sb in scoreboards.items():
        for _, r in sb.iterrows():
            hid = hypothesis_id(r["indicator"], r["prev_regime"], r["regime"], int(r["h"]))
            if hid not in all_h:
                all_h[hid] = {
                    "hypothesis_id": hid,
                    "indicator":     r["indicator"],
                    "prev_regime":   r["prev_regime"],
                    "regime":        r["regime"],
                    "h":             int(r["h"]),
                }
            all_h[hid][f"n_{code}"]    = int(r["n"])
            all_h[hid][f"mean_{code}"] = float(r["mean_pct"])
            all_h[hid][f"t_{code}"]    = float(r["t"]) if not pd.isna(r["t"]) else np.nan

    df = pd.DataFrame(list(all_h.values()))

    for code in codes:
        for col_prefix in ("n_", "mean_", "t_"):
            col = f"{col_prefix}{code}"
            if col not in df.columns:
                df[col] = np.nan

    def _aggregate(row):
        ts = [row.get(f"t_{c}") for c in codes]
        means = [row.get(f"mean_{c}") for c in codes]
        n_long_pass  = sum(
            1 for t, m in zip(ts, means)
            if not pd.isna(t) and t >= sig_threshold and not pd.isna(m) and m > 0
        )
        n_short_pass = sum(
            1 for t, m in zip(ts, means)
            if not pd.isna(t) and t <= -sig_threshold and not pd.isna(m) and m < 0
        )
        n_long_dir   = sum(1 for t in ts if not pd.isna(t) and t >= direction_threshold)
        n_short_dir  = sum(1 for t in ts if not pd.isna(t) and t <= -direction_threshold)
        n_present    = sum(1 for t in ts if not pd.isna(t))
        return pd.Series({
            "n_present":     n_present,
            "n_long_pass":   n_long_pass,
            "n_short_pass":  n_short_pass,
            "n_long_dir":    n_long_dir,
            "n_short_dir":   n_short_dir,
        })

    df = pd.concat([df, df.apply(_aggregate, axis=1)], axis=1)

    def _tier(row):
        long_pass = row["n_long_pass"]
        short_pass = row["n_short_pass"]
        if long_pass >= 1 and short_pass >= 1:
            return "F (mixed)"
        passes = max(long_pass, short_pass)
        if passes >= 3:        return "A (≥3/4 same dir |t|≥2)"
        if passes == 2:        return "B (2/4 same dir |t|≥2)"
        if passes == 1:        return "C (single-stock only)"
        return                        "F (none |t|≥2)"

    def _dir(row):
        if row["n_long_pass"] >= 1 and row["n_short_pass"] >= 1:
            return "mixed"
        if row["n_long_pass"] > row["n_short_pass"]:
            return "long"
        if row["n_short_pass"] > row["n_long_pass"]:
            return "short"
        if row["n_long_dir"] > row["n_short_dir"]:
            return "long?"
        if row["n_short_dir"] > row["n_long_dir"]:
            return "short?"
        return "none"

    df["robustness"]  = df.apply(_tier, axis=1)
    df["verdict_dir"] = df.apply(_dir, axis=1)

    tier_rank = {"A (≥3/4 same dir |t|≥2)": 0, "B (2/4 same dir |t|≥2)": 1,
                 "C (single-stock only)": 2, "F (mixed)": 3, "F (none |t|≥2)": 4}
    df["_rank"] = df["robustness"].map(tier_rank)
    df["max_pass"] = df[["n_long_pass", "n_short_pass"]].max(axis=1)
    df["max_abs_t"] = df[[f"t_{c}" for c in codes]].abs().max(axis=1)
    df = df.sort_values(["_rank", "max_pass", "max_abs_t"],
                        ascending=[True, False, False]).drop(columns=["_rank"])

    base_cols = ["hypothesis_id", "indicator", "prev_regime", "regime", "h"]
    stock_cols = []
    for c in codes:
        stock_cols += [f"n_{c}", f"mean_{c}", f"t_{c}"]
    summary_cols = ["n_present", "n_long_pass", "n_short_pass",
                    "max_pass", "max_abs_t", "robustness", "verdict_dir"]
    return df[base_cols + stock_cols + summary_cols]


# ─── Main ───────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="A 股多股票 meta-label 横向稳健性扫描 v1")
    parser.add_argument("--codes",        default=",".join(DEFAULT_CODES))
    parser.add_argument("--start",        default="2023-04-04")
    parser.add_argument("--end",          default="2026-04-22")
    parser.add_argument("--out-dir",      default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--verdict-path", default=str(DEFAULT_VERDICT))
    args = parser.parse_args()

    codes = [c.strip() for c in args.codes.split(",") if c.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== 多股票 meta-label 横向扫描 ===")
    print(f"codes:  {codes}")
    print(f"window: {args.start} → {args.end}\n")

    scoreboards: dict[str, pd.DataFrame] = {}
    libraries:   dict[str, pd.DataFrame] = {}
    n_days_per:  dict[str, int]          = {}

    for code in codes:
        name = CODE_NAMES.get(code, "?")
        print(f"--- {code} {name} ---")
        df_day = load_day(code, args.start, args.end)
        if len(df_day) < 100:
            print(f"   ⚠️ 仅 {len(df_day)} 个交易日，跳过")
            continue

        df = build_regime_columns(df_day)
        sb = build_event_scoreboard(df)
        lib = build_signal_library(sb)

        scoreboards[code] = sb
        libraries[code]   = lib
        n_days_per[code]  = len(df_day)

        per_stock_csv = out_dir / f"cnstock_{code}_signals.csv"
        lib.to_csv(per_stock_csv, index=False, encoding="utf-8")
        print(f"   {len(df_day)} days  →  {len(lib)} signals "
              f"({dict(lib['tier'].value_counts()) if len(lib) else '∅'})")
        print(f"   wrote {per_stock_csv}")

    if len(scoreboards) < 2:
        print("\n⚠️ 至少需要 2 只股票才能做横向聚合，退出。")
        return

    print(f"\n--- aggregating across {len(scoreboards)} stocks ---")
    matrix = aggregate_cross_stock(scoreboards)

    matrix_csv = out_dir / "cross_stock_signal_matrix.csv"
    matrix.to_csv(matrix_csv, index=False, encoding="utf-8")
    print(f"wrote {matrix_csv}")

    tier_counts = matrix["robustness"].value_counts()
    print(f"\n=== robustness tier 分布 ===")
    for tier in ["A (≥3/4 same dir |t|≥2)", "B (2/4 same dir |t|≥2)",
                 "C (single-stock only)", "F (mixed)", "F (none |t|≥2)"]:
        if tier in tier_counts:
            print(f"   {tier}: {tier_counts[tier]}")

    show_cols = ["hypothesis_id", "indicator", "regime", "h",
                 *[f"t_{c}" for c in scoreboards.keys()],
                 "max_pass", "robustness", "verdict_dir"]
    print(f"\n=== Tier A + B hypotheses（横向稳健）===")
    ab = matrix[matrix["robustness"].isin([
        "A (≥3/4 same dir |t|≥2)", "B (2/4 same dir |t|≥2)"
    ])]
    if len(ab) == 0:
        print("   ∅ 无横向稳健信号")
    else:
        print(ab[show_cols].to_string(index=False))

    print(f"\n=== 300866 显著但其他股票不显著（C tier）===")
    if "300866" in scoreboards:
        c_only = matrix[
            (matrix["robustness"] == "C (single-stock only)") &
            (matrix["t_300866"].abs() >= 2.0)
        ]
        if len(c_only):
            cols_c = ["hypothesis_id", *[f"t_{c}" for c in scoreboards.keys()], "verdict_dir"]
            print(c_only[cols_c].to_string(index=False))
        else:
            print("   ∅")

    verdict_path = Path(args.verdict_path)
    verdict_path.parent.mkdir(parents=True, exist_ok=True)

    md_lines = [
        f"# A 股多股票 meta-label 横向稳健性 v1",
        f"",
        f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**窗口**: {args.start} → {args.end}",
        f"**股票数**: {len(scoreboards)}",
        f"",
        f"## 方法",
        f"对每只股票：",
        f"1. 加载日线（已前复权）",
        f"2. 计算 5 个 day 级 indicator regime（vol / trend / drawdown / volume_z / RSI）",
        f"3. 事件去重后跑 `regime_score(h ∈ {{1,5,21}})` → 生成 event scoreboard",
        f"4. 按 |t|≥2 + (n, t) 分级 → 生成 per-stock 信号库",
        f"",
        f"横向聚合：把每只股票的 scoreboard 按 `(indicator, regime, h)` 拼成 hypothesis 矩阵，",
        f"统计每个 hypothesis 在多少只股票上同向 |t|≥2，分四档：",
        f"",
        f"| Tier | 标准 |",
        f"|------|------|",
        f"| A | ≥3/4 stocks 同向 \\|t\\|≥2 |",
        f"| B | 2/4 stocks 同向 \\|t\\|≥2 |",
        f"| C | 仅 1 只股票 \\|t\\|≥2 |",
        f"| F | 0 只 OR 不同股票方向反转 |",
        f"",
        f"## 数据覆盖",
        f"",
        f"| 股票 | 名称 | 交易日数 | 候选信号数 |",
        f"|------|------|---------|------------|",
    ]
    for code in scoreboards.keys():
        md_lines.append(
            f"| {code} | {CODE_NAMES.get(code, '?')} | {n_days_per[code]} | {len(libraries[code])} |"
        )

    md_lines += [
        f"",
        f"## Robustness Tier 分布",
        f"",
    ]
    for tier in ["A (≥3/4 same dir |t|≥2)", "B (2/4 same dir |t|≥2)",
                 "C (single-stock only)", "F (mixed)", "F (none |t|≥2)"]:
        cnt = tier_counts.get(tier, 0)
        md_lines.append(f"- **{tier}**: {cnt}")

    md_lines += [
        f"",
        f"## Tier A + B（横向稳健）",
        f"",
    ]
    if len(ab):
        md_lines.append("```")
        md_lines.append(ab[show_cols].to_string(index=False))
        md_lines.append("```")
    else:
        md_lines.append("∅ 无横向稳健信号——所有 §3.4 显著的信号都是 stock-specific")
        md_lines.append("")
        md_lines.append("**含义**：之前在 300866 上找到的 8 条信号没有一条能横向泛化。")
        md_lines.append("这本身是个重要结论——这些信号本质是单股特征，不能作为通用 alpha。")

    md_lines += [
        f"",
        f"## 完整矩阵",
        f"",
        f"参见 `{matrix_csv.relative_to(REPO_ROOT)}`，共 {len(matrix)} 个 hypothesis。",
        f"",
        f"## 下一步",
        f"",
    ]
    if len(ab) >= 1:
        md_lines += [
            f"- Tier A/B 信号 ({len(ab)} 条) 进入下一阶段：构造组合策略，扣手续费滑点后做 walk-forward",
            f"- Tier C 信号需小心——可能是单股偶然或确实存在 stock-specific alpha，待多窗口验证",
        ]
    else:
        md_lines += [
            f"- 横向不一致 → meta-label 框架需要修正（可能是 day 层级颗粒度太粗）",
            f"- 候选方向：(1) 改 5min 粒度 (2) 用行业 / 板块作为额外条件 (3) 引入横截面 cross-section ranking",
        ]
    md_lines.append("")

    verdict_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\n📝 verdict written to {verdict_path}")


if __name__ == "__main__":
    main()
