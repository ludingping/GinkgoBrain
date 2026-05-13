"""IC 验证 spike：v0.2 "主力吸筹建仓" 信号（持续流入 + 价格温和）.

设计思路（与 v0.1 momentum 反方向）：
- C1 持续性：过去 N=10 个交易日中，大单净流入为正的天数 ≥ M=7
- C3 价格温和：过去 N 日累计涨幅 ∈ [-5%, +8%]
- 排序：通过过滤的标的按 ``Σ big_net_inflow / Σ total_amount`` 降序，取 Top20
- IC：score 与 T+4 累计相对收益的 Spearman

输出 reports/spike_v0_2/ 下：
- ic_stats.json
- ic_series.csv
- daily_top20.csv
- universe_after_filter.csv
- ic_plot.png

非完整 backtest（不撮合、不算 equity），只看信号本身有无 alpha.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import date
from pathlib import Path

# 项目根加入 sys.path（让本脚本可被 python 直接调用）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from strategies.cn_a_big_money_rotation.data import (
    compute_money_flow_factors,
    load_kline_close_range,
    load_money_flow_range,
    load_security_list_active,
    load_snapshot_latest,
    recover_float_share,
)
from strategies.cn_a_big_money_rotation.evaluate import compute_ic_stats

logger = logging.getLogger(__name__)

# ============================================================================
# 参数（用户已定）
# ============================================================================

N_WINDOW = 10                # C1 / C3 滚动窗口
POSITIVE_DAYS_MIN = 7        # C1: ≥7 日正流入
PRICE_RANGE = (-0.05, 0.08)  # C3: v0.2 原版 [-5%, +8%]（universe 健康）
TOP_N = 20
FORWARD_IC_HORIZONS = [1, 4, 10, 20]  # 多 horizon 扫描
START = date(2025, 4, 2)
END = date(2026, 5, 12)
OUTPUT_DIR = Path("/home/davidlyu/projects/GinkgoBrain/reports/spike_v0_2_multi_horizon")


def load_data() -> dict:
    logger.info("Loading sec_list / snapshot / money_flow / kline ...")
    sec_list = load_security_list_active()
    snapshot = load_snapshot_latest()
    mf = load_money_flow_range(START.isoformat(), END.isoformat())
    kline = load_kline_close_range(START.isoformat(), END.isoformat())

    float_share_map: dict[str, float] = {}
    anomalies = 0
    for row in snapshot.itertuples():
        r = recover_float_share({
            "float_market_cap": getattr(row, "float_market_cap", None),
            "current_price": getattr(row, "current_price", None),
            "total_volume": getattr(row, "total_volume", None),
            "turnover_rate": getattr(row, "turnover_rate", None),
        })
        if r.share is not None:
            float_share_map[row.stock_code] = float(r.share)
        if r.anomaly:
            anomalies += 1
    logger.info("float_share recovered for %d stocks (anomalies=%d)", len(float_share_map), anomalies)

    mf_with_factors = compute_money_flow_factors(mf)
    return {
        "sec_list": sec_list,
        "snapshot": snapshot,
        "mf": mf_with_factors,
        "kline": kline,
        "float_share_map": float_share_map,
    }


def pivot_matrices(mf: pd.DataFrame, kline: pd.DataFrame) -> dict:
    inflow_mat = mf.pivot_table(
        index="trade_date", columns="stock_code", values="big_net_inflow", aggfunc="first"
    )
    amount_mat = mf.pivot_table(
        index="trade_date", columns="stock_code", values="total_amount", aggfunc="first"
    )
    close_mat = kline.pivot_table(
        index="trade_date", columns="stock_code", values="close_price", aggfunc="first"
    )
    kline_amount_mat = kline.pivot_table(
        index="trade_date", columns="stock_code", values="amount", aggfunc="first"
    )
    common_dates = sorted(set(inflow_mat.index) & set(close_mat.index))
    common_codes = sorted(set(inflow_mat.columns) & set(close_mat.columns))
    return {
        "inflow": inflow_mat.loc[common_dates, common_codes],
        "amount": amount_mat.loc[common_dates, common_codes],
        "close": close_mat.loc[common_dates, common_codes],
        "kline_amount": kline_amount_mat.loc[common_dates, common_codes],
    }


def compute_v0_2_signal(mats: dict) -> dict:
    inflow = mats["inflow"]
    amount = mats["amount"]
    close = mats["close"]

    positive_days = (inflow > 0).rolling(window=N_WINDOW, min_periods=N_WINDOW).sum()
    cum_inflow = inflow.rolling(window=N_WINDOW, min_periods=N_WINDOW).sum()
    cum_amount = amount.rolling(window=N_WINDOW, min_periods=N_WINDOW).sum()
    score = cum_inflow / cum_amount.replace(0, np.nan)
    price_change_n = close / close.shift(N_WINDOW - 1) - 1

    mask_c1 = positive_days >= POSITIVE_DAYS_MIN
    mask_c3 = (price_change_n >= PRICE_RANGE[0]) & (price_change_n <= PRICE_RANGE[1])
    mask_score = score.notna()
    final_mask = mask_c1 & mask_c3 & mask_score
    score_filtered = score.where(final_mask)

    return {
        "score": score,
        "score_filtered": score_filtered,
        "positive_days": positive_days,
        "price_change_n": price_change_n,
        "cum_inflow": cum_inflow,
        "cum_amount": cum_amount,
        "mask_c1": mask_c1,
        "mask_c3": mask_c3,
        "final_mask": final_mask,
    }


def select_top_n_per_day(score_filtered: pd.DataFrame, n: int = TOP_N) -> dict:
    out = {}
    for t, row in score_filtered.iterrows():
        valid = row.dropna()
        if len(valid) == 0:
            out[t] = []
            continue
        sorted_codes = sorted(valid.index, key=lambda c: (-valid[c], c))
        out[t] = sorted_codes[:n]
    return out


def compute_daily_ic(
    score_filtered: pd.DataFrame,
    close: pd.DataFrame,
    horizon: int = 4,
) -> pd.Series:
    fwd_ret = close.shift(-horizon) / close - 1.0
    fwd_rel = fwd_ret.sub(fwd_ret.mean(axis=1), axis=0)
    ic_list = []
    for t in score_filtered.index:
        s = score_filtered.loc[t]
        r = fwd_rel.loc[t]
        mask = s.notna() & r.notna()
        if mask.sum() < 5:
            ic_list.append(np.nan)
            continue
        ic_list.append(float(s[mask].corr(r[mask], method="spearman")))
    return pd.Series(ic_list, index=score_filtered.index, name="ic")


def write_reports(
    out_dir: Path,
    score_filtered: pd.DataFrame,
    signal_info: dict,
    top_n_per_day: dict,
    ic_series: pd.Series,
    sec_list: pd.DataFrame,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = compute_ic_stats(ic_series.dropna())
    print(f"\n=== v0.2 IC Stats ===")
    print(f"  mean   = {stats.mean:.4f}")
    print(f"  std    = {stats.std:.4f}")
    print(f"  t-stat = {stats.t_stat:.3f}")
    print(f"  n      = {stats.n}")
    verdict_pass = stats.mean > 0.02 and stats.t_stat > 3
    print(f"  Verdict: {'PASS 门槛' if verdict_pass else 'FAIL 门槛'}")

    (out_dir / "ic_stats.json").write_text(
        json.dumps({
            "ic_mean": stats.mean,
            "ic_std": stats.std,
            "ic_tstat": stats.t_stat,
            "ic_n": stats.n,
            "params": {
                "N_window": N_WINDOW,
                "positive_days_min": POSITIVE_DAYS_MIN,
                "price_range": list(PRICE_RANGE),
                "top_n": TOP_N,
                "forward_ic_horizons": FORWARD_IC_HORIZONS,
            },
            "pass_thresholds": verdict_pass,
        }, indent=2, default=str),
        encoding="utf-8",
    )

    ic_series.to_csv(out_dir / "ic_series.csv", index_label="trade_date")

    final_mask = signal_info["final_mask"]
    mask_c1 = signal_info["mask_c1"]
    mask_c3 = signal_info["mask_c3"]
    universe_stats = pd.DataFrame({
        "after_c1_persistence": mask_c1.sum(axis=1),
        "after_c3_price": mask_c3.sum(axis=1),
        "after_all": final_mask.sum(axis=1),
        "selected_top20": [len(top_n_per_day.get(t, [])) for t in mask_c1.index],
    }, index=mask_c1.index)
    universe_stats.to_csv(out_dir / "universe_after_filter.csv", index_label="trade_date")
    print(f"\n=== Universe 过滤效果（取近 5 日） ===")
    print(universe_stats.tail(5).to_string())

    name_map = dict(zip(sec_list["stock_code"], sec_list["stock_name"]))
    rows = []
    for t, codes in top_n_per_day.items():
        for rank, code in enumerate(codes, 1):
            s = score_filtered.loc[t, code] if code in score_filtered.columns else np.nan
            cum_inf = signal_info["cum_inflow"].loc[t, code] if code in signal_info["cum_inflow"].columns else np.nan
            pos_days = signal_info["positive_days"].loc[t, code] if code in signal_info["positive_days"].columns else np.nan
            price_chg = signal_info["price_change_n"].loc[t, code] if code in signal_info["price_change_n"].columns else np.nan
            rows.append({
                "trade_date": t,
                "rank": rank,
                "stock_code": code,
                "stock_name": name_map.get(code),
                "score": float(s) if pd.notna(s) else None,
                "cum_inflow_yi": float(cum_inf) / 1e8 if pd.notna(cum_inf) else None,
                "positive_days_in_10": int(pos_days) if pd.notna(pos_days) else None,
                "price_change_10d_pct": float(price_chg) * 100 if pd.notna(price_chg) else None,
            })
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "daily_top20.csv", index=False)
        print(f"\ndaily_top20.csv rows: {len(rows):,}")

    fig, axes = plt.subplots(3, 1, figsize=(13, 10))
    valid_ic = ic_series.dropna()

    ax = axes[0]
    ax.plot(valid_ic.index, valid_ic.values, color="C0", alpha=0.5, linewidth=0.8, label="daily IC")
    ax.plot(valid_ic.index, valid_ic.rolling(20).mean(), color="C0", linewidth=2.2, label="20d MA")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axhline(0.02, color="green", linestyle="--", alpha=0.6, label="门槛 0.02")
    ax.set_title(f"v0.2 Daily IC Time Series (N={N_WINDOW}, M≥{POSITIVE_DAYS_MIN}, price∈{PRICE_RANGE})")
    ax.set_ylabel("IC")
    ax.legend(); ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    ax = axes[1]
    ax.hist(valid_ic.values, bins=40, color="C0", alpha=0.6, edgecolor="black")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(valid_ic.mean(), color="red", linewidth=2, label=f"mean={valid_ic.mean():.4f}")
    ax.axvline(0.02, color="green", linestyle="--", label="门槛 0.02")
    ax.set_title(f"IC Distribution (mean={valid_ic.mean():.4f}, std={valid_ic.std():.4f}, t-stat={stats.t_stat:.2f})")
    ax.set_xlabel("Daily IC"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[2]
    cum = valid_ic.fillna(0).cumsum()
    ax.plot(cum.index, cum.values, color="C0", linewidth=1.6)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Cumulative Daily IC (sum)")
    ax.set_ylabel("cumsum")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "ic_plot.png", dpi=110)
    print(f"\nWrote {out_dir / 'ic_plot.png'}")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    data = load_data()
    mats = pivot_matrices(data["mf"], data["kline"])
    logger.info("matrix shape: inflow=%s close=%s", mats["inflow"].shape, mats["close"].shape)

    signal_info = compute_v0_2_signal(mats)
    score_filtered = signal_info["score_filtered"]
    logger.info("score_filtered non-NaN cells: %d / %d (%.2f%%)",
                int(score_filtered.notna().sum().sum()),
                int(score_filtered.size),
                100 * float(score_filtered.notna().sum().sum()) / max(1, score_filtered.size))

    top20_per_day = select_top_n_per_day(score_filtered, n=TOP_N)
    n_selected_days = sum(1 for v in top20_per_day.values() if v)
    logger.info("Days with >=1 selected: %d / %d", n_selected_days, len(top20_per_day))

    # 多 horizon 扫描
    all_ic = {}
    summary_rows = []
    for h in FORWARD_IC_HORIZONS:
        logger.info("Computing IC for horizon T+%d ...", h)
        ic_h = compute_daily_ic(score_filtered, mats["close"], horizon=h)
        all_ic[h] = ic_h
        stats = compute_ic_stats(ic_h.dropna())
        verdict_pass = stats.mean > 0.02 and stats.t_stat > 3
        verdict_reverse = stats.mean < -0.02 and stats.t_stat < -3
        flag = "PASS" if verdict_pass else ("REVERSE-SIG" if verdict_reverse else "NO-ALPHA")
        summary_rows.append({
            "horizon": f"T+{h}",
            "ic_mean": round(stats.mean, 4),
            "ic_std": round(stats.std, 4),
            "t_stat": round(stats.t_stat, 3),
            "n": stats.n,
            "verdict": flag,
        })

    summary_df = pd.DataFrame(summary_rows)
    print("\n=== Multi-Horizon IC Summary (signal=v0.2 持续吸筹+价格温和) ===")
    print(summary_df.to_string(index=False))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(OUTPUT_DIR / "horizon_ic_summary.csv", index=False)

    # 保存 4 个 horizon 的 daily IC series
    df_all = pd.DataFrame({f"ic_T+{h}": s for h, s in all_ic.items()})
    df_all.to_csv(OUTPUT_DIR / "ic_series_multi_horizon.csv", index_label="trade_date")

    # 画图：4 个 horizon cumsum
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, (h, s) in zip(axes.flat, all_ic.items()):
        v = s.dropna()
        cum = v.fillna(0).cumsum()
        ax.plot(cum.index, cum.values, linewidth=1.6)
        ax.axhline(0, color="black", linewidth=0.5)
        stats = compute_ic_stats(v)
        title_flag = "✓" if (stats.mean > 0.02 and stats.t_stat > 3) else (
            "↓ REVERSE" if (stats.mean < -0.02 and stats.t_stat < -3) else "≈ 0"
        )
        ax.set_title(f"T+{h} | mean={stats.mean:.4f} t={stats.t_stat:.2f} {title_flag}")
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.suptitle("v0.2 信号 在不同 forward horizon 上的累积 IC", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "multi_horizon_cumsum_ic.png", dpi=110)
    print(f"\nWrote {OUTPUT_DIR / 'multi_horizon_cumsum_ic.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
