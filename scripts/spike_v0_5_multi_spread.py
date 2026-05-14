"""v0.5 多档资金流向 spread 组合 IC 扫描.

7 个 spread 变种：
1. super - big        (主力内部分化)
2. super - middle     (超大单 vs 中户)
3. super - small      (v0.4 baseline)
4. big - middle       (大户 vs 中户)
5. big - small        (大户 vs 散户)
6. middle - small     (中户 vs 散户)
7. (super + big) - (middle + small) = main vs retail (聚合主力 vs 聚合散户)

每个 × 4 rolling window (d1, d5, d10, d20)：
- 全样本 IC + t-stat
- 分桶单调（10 buckets）
- Top vs Bottom L-S
- 按年度 IC（regime 稳定性）

目标：找出 t-stat 最高、单调性最稳定、跨周期最 robust 的组合.

只做 IC 验证（用修过的 fwd_ret = price.shift(-h)/price.shift(-1) - 1）；
不上 portfolio simulation（避免 lookahead bug 重蹈覆辙）。
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

HORIZON = 4
TOP_N = 20
OUTPUT_DIR = Path("/home/davidlyu/projects/GinkgoBrain/reports/spike_v0_5_multi_spread")
MONGO_HOST = "192.168.1.69"
MONGO_PORT = 27017
MONGO_USER = "admin"
MONGO_DB = "treasure"
MONGO_COLL = "stock_fund_flow"
EFFECTIVE_START_DATE = pd.Timestamp("2022-01-01")
MIN_COVERAGE_DAYS = 500
ROLLING_WINDOWS = [1, 5, 10, 20]


def load_and_filter() -> pd.DataFrame:
    from pymongo import MongoClient
    pwd = os.environ.get("MONGO_PWD")
    if not pwd:
        raise RuntimeError("MONGO_PWD env var required")
    client = MongoClient(host=MONGO_HOST, port=MONGO_PORT,
                         username=MONGO_USER, password=pwd, authSource="admin")
    col = client[MONGO_DB][MONGO_COLL]
    logger.info("Loading mongo ...")
    rows = list(col.find(
        {"date": {"$gte": EFFECTIVE_START_DATE.to_pydatetime()}},
        {"_id": 0, "stock_id": 1, "date": 1, "price": 1,
         "super_net_inflow_ratio": 1, "big_net_inflow_ratio": 1,
         "middle_net_inflow_ratio": 1, "small_net_inflow_ratio": 1},
    ))
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    for c in ["price", "super_net_inflow_ratio", "big_net_inflow_ratio",
              "middle_net_inflow_ratio", "small_net_inflow_ratio"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    cov = df.groupby("stock_id").size()
    keep = cov[cov >= MIN_COVERAGE_DAYS].index
    df = df[df["stock_id"].isin(keep)]
    logger.info("Filtered to %d rows / %d stocks", len(df), len(keep))
    return df


def pivot_panel(df: pd.DataFrame) -> dict:
    def piv(col):
        return df.pivot_table(index="date", columns="stock_id", values=col, aggfunc="first")
    panel = {
        "super": piv("super_net_inflow_ratio"),
        "big": piv("big_net_inflow_ratio"),
        "middle": piv("middle_net_inflow_ratio"),
        "small": piv("small_net_inflow_ratio"),
        "price": piv("price"),
    }
    daily_size = panel["price"].notna().sum(axis=1)
    max_size = int(daily_size.max())
    valid_dates = panel["price"].index[daily_size >= max_size * 0.5]
    for k in panel:
        panel[k] = panel[k].loc[valid_dates]
    return panel


SPREAD_DEFINITIONS = {
    "super-big": lambda d: d["super"] - d["big"],
    "super-middle": lambda d: d["super"] - d["middle"],
    "super-small": lambda d: d["super"] - d["small"],
    "big-middle": lambda d: d["big"] - d["middle"],
    "big-small": lambda d: d["big"] - d["small"],
    "middle-small": lambda d: d["middle"] - d["small"],
    "main_vs_retail": lambda d: (d["super"] + d["big"]) - (d["middle"] + d["small"]),
}


def compute_forward_return(price: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    return price.shift(-horizon) / price.shift(-1) - 1.0


def daily_ic_series(score: pd.DataFrame, fwd_ret: pd.DataFrame) -> pd.Series:
    out = []
    for t in score.index:
        s = score.loc[t]
        r = fwd_ret.loc[t]
        mask = s.notna() & r.notna()
        if mask.sum() < 5:
            out.append((t, np.nan))
            continue
        out.append((t, float(s[mask].corr(r[mask], method="spearman"))))
    return pd.Series([v for _, v in out], index=[t for t, _ in out], name="ic")


def bucket_returns(score: pd.DataFrame, fwd_ret: pd.DataFrame, n_buckets: int = 10) -> pd.Series:
    per_day = []
    for t in score.index:
        s = score.loc[t].dropna()
        r = fwd_ret.loc[t].reindex(s.index)
        mask = s.notna() & r.notna()
        if mask.sum() < n_buckets * 2:
            continue
        try:
            buckets = pd.qcut(s[mask], q=n_buckets, labels=False, duplicates="drop")
        except ValueError:
            continue
        row = {b: float(r[mask][buckets == b].mean()) for b in range(n_buckets)
               if (buckets == b).sum() > 0}
        per_day.append(row)
    df = pd.DataFrame(per_day)
    bucket_cols = [c for c in df.columns if isinstance(c, int)]
    return df[bucket_cols].mean(axis=0)


def top_vs_bottom(score: pd.DataFrame, fwd_ret: pd.DataFrame, n: int = TOP_N) -> tuple[float, float]:
    rows = []
    for t in score.index:
        s = score.loc[t].dropna()
        r = fwd_ret.loc[t].reindex(s.index)
        mask = s.notna() & r.notna()
        if mask.sum() < n * 2:
            continue
        s_v = s[mask].sort_values(ascending=False)
        r_v = r[mask]
        top = float(r_v.loc[s_v.head(n).index].mean())
        bot = float(r_v.loc[s_v.tail(n).index].mean())
        rows.append(top - bot)
    if not rows:
        return 0.0, 0.0
    arr = np.array(rows)
    if arr.std() == 0:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(arr.mean() / arr.std() * np.sqrt(len(arr)))


def yearly_ic_stats(ic_series: pd.Series) -> pd.DataFrame:
    df = ic_series.dropna().to_frame(name="ic")
    df["year"] = df.index.year
    return df.groupby("year")["ic"].agg(
        mean="mean",
        t_stat=lambda x: float(x.mean() / x.std() * np.sqrt(len(x))) if x.std() > 0 else 0.0,
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_and_filter()
    panel = pivot_panel(df)
    fwd_ret = compute_forward_return(panel["price"], horizon=HORIZON)
    logger.info("Trade dates: %d, stocks: %d", len(panel["price"]), panel["price"].shape[1])

    base_spreads = {name: fn(panel) for name, fn in SPREAD_DEFINITIONS.items()}

    summary_rows = []
    yearly_all = {}
    bucket_all = {}

    for name, base in base_spreads.items():
        for w in ROLLING_WINDOWS:
            key = f"{name}_d{w}"
            score = base if w == 1 else base.rolling(w, min_periods=w).mean()
            ic = daily_ic_series(score, fwd_ret)
            ic_valid = ic.dropna()
            ic_mean = float(ic_valid.mean()) if len(ic_valid) > 0 else 0.0
            ic_t = float(ic_valid.mean() / ic_valid.std() * np.sqrt(len(ic_valid))) if ic_valid.std() > 0 else 0.0
            ic_n = len(ic_valid)
            bm = bucket_returns(score, fwd_ret)
            ls_mean, ls_t = top_vs_bottom(score, fwd_ret)
            yr = yearly_ic_stats(ic)
            sorted_idx = bm.sort_values().index.tolist()
            is_mono_inc = sorted_idx == list(range(len(bm)))
            is_mono_dec = sorted_idx == list(reversed(range(len(bm))))
            yr_signs = (yr["mean"] > 0).astype(int) - (yr["mean"] < 0).astype(int)
            n_same_sign = int(((yr_signs == np.sign(ic_mean)) | (yr_signs == 0)).sum())
            summary_rows.append({
                "name": name,
                "window": w,
                "ic_mean": round(ic_mean, 4),
                "ic_tstat": round(ic_t, 2),
                "ic_n": ic_n,
                "B1_pct": round(bm.iloc[0] * 100, 3) if len(bm) > 0 else 0.0,
                "B10_pct": round(bm.iloc[-1] * 100, 3) if len(bm) >= 10 else 0.0,
                "B10_minus_B1_pct": round((bm.iloc[-1] - bm.iloc[0]) * 100, 3) if len(bm) >= 10 else 0.0,
                "monotonic": "↑" if is_mono_inc else ("↓" if is_mono_dec else "·"),
                "ls_mean_pct": round(ls_mean * 100, 3),
                "ls_tstat": round(ls_t, 2),
                "yr_same_sign": f"{n_same_sign}/{len(yr)}",
            })
            yearly_all[key] = yr
            bucket_all[key] = bm
        logger.info("Done: %s", name)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)

    print("\n========== 7 spread × 4 window = 28 variants ==========")
    pd.set_option("display.max_rows", 100)
    pd.set_option("display.max_columns", 100)
    print(summary.to_string(index=False))

    summary["abs_ls_t"] = summary["ls_tstat"].abs()
    summary["abs_ic_t"] = summary["ic_tstat"].abs()
    print("\n========== Top 5 by |L-S t-stat| ==========")
    print(summary.sort_values("abs_ls_t", ascending=False).head(5)[
        ["name","window","ic_mean","ic_tstat","B10_minus_B1_pct","monotonic","ls_mean_pct","ls_tstat","yr_same_sign"]
    ].to_string(index=False))

    print("\n========== Top 5 by |IC t-stat| ==========")
    print(summary.sort_values("abs_ic_t", ascending=False).head(5)[
        ["name","window","ic_mean","ic_tstat","B10_minus_B1_pct","monotonic","ls_mean_pct","ls_tstat","yr_same_sign"]
    ].to_string(index=False))

    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps({"params": {"spreads": list(SPREAD_DEFINITIONS.keys()),
                                "windows": ROLLING_WINDOWS, "horizon": HORIZON},
                    "rows": summary_rows}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    # 图 1: heatmap |ls_tstat|
    pivot_ls_t = summary.pivot(index="name", columns="window", values="ls_tstat")
    max_abs = max(abs(pivot_ls_t.min().min()), abs(pivot_ls_t.max().max()))
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot_ls_t.values, cmap="RdBu_r", aspect="auto",
                   vmin=-max_abs, vmax=max_abs)
    ax.set_xticks(range(len(pivot_ls_t.columns)))
    ax.set_xticklabels([f"d{w}" for w in pivot_ls_t.columns])
    ax.set_yticks(range(len(pivot_ls_t.index)))
    ax.set_yticklabels(pivot_ls_t.index)
    ax.set_title("v0.5 Long-Short t-stat (spread × window)")
    for i in range(len(pivot_ls_t.index)):
        for j in range(len(pivot_ls_t.columns)):
            v = pivot_ls_t.iloc[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    color="white" if abs(v) > 2 else "black", fontsize=10)
    fig.colorbar(im, ax=ax, label="L-S t-stat")
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "heatmap_ls_tstat.png", dpi=110); plt.close()

    # 图 2: heatmap IC t-stat
    pivot_ic_t = summary.pivot(index="name", columns="window", values="ic_tstat")
    max_abs = max(abs(pivot_ic_t.min().min()), abs(pivot_ic_t.max().max()))
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot_ic_t.values, cmap="RdBu_r", aspect="auto",
                   vmin=-max_abs, vmax=max_abs)
    ax.set_xticks(range(len(pivot_ic_t.columns)))
    ax.set_xticklabels([f"d{w}" for w in pivot_ic_t.columns])
    ax.set_yticks(range(len(pivot_ic_t.index)))
    ax.set_yticklabels(pivot_ic_t.index)
    ax.set_title("v0.5 IC t-stat (spread × window)")
    for i in range(len(pivot_ic_t.index)):
        for j in range(len(pivot_ic_t.columns)):
            v = pivot_ic_t.iloc[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    color="white" if abs(v) > 2 else "black", fontsize=10)
    fig.colorbar(im, ax=ax, label="IC t-stat")
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "heatmap_ic_tstat.png", dpi=110); plt.close()

    # 图 3: Top 4 by |L-S t-stat| bucket curves
    top4 = summary.sort_values("abs_ls_t", ascending=False).head(4)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, (_, row) in zip(axes.flat, top4.iterrows()):
        key = f"{row['name']}_d{row['window']}"
        bm = bucket_all[key]
        colors = ["C2" if bm.iloc[i] > 0 else "C3" for i in range(len(bm))]
        ax.bar(range(len(bm)), bm.values * 100, color=colors, alpha=0.85, edgecolor="black")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(range(len(bm))); ax.set_xticklabels([f"B{i+1}" for i in range(len(bm))])
        ax.set_title(f"{key} | IC t={row['ic_tstat']:+.2f} | L-S t={row['ls_tstat']:+.2f}")
        ax.set_ylabel("fwd_ret %")
        ax.grid(alpha=0.3, axis="y")
    plt.suptitle("v0.5 Top 4 by |L-S t-stat| — buckets", fontsize=12)
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "top4_buckets.png", dpi=110); plt.close()

    # 图 4: Top 4 yearly IC
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, (_, row) in zip(axes.flat, top4.iterrows()):
        key = f"{row['name']}_d{row['window']}"
        yr = yearly_all[key]
        colors = ["C2" if (m > 0.02 and t > 3) else ("C3" if (m < -0.02 and t < -3) else "C7")
                  for m, t in zip(yr["mean"], yr["t_stat"])]
        ax.bar(yr.index.astype(str), yr["t_stat"], color=colors, alpha=0.85, edgecolor="black")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.axhline(3, color="green", linestyle="--", alpha=0.6)
        ax.axhline(-3, color="red", linestyle="--", alpha=0.6)
        ax.set_ylabel("Yearly IC t-stat"); ax.set_title(key)
        ax.grid(alpha=0.3, axis="y")
    plt.suptitle("v0.5 Top 4 by |L-S t-stat| — yearly IC", fontsize=12)
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "top4_yearly_ic.png", dpi=110); plt.close()

    print(f"\nAll outputs in: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
