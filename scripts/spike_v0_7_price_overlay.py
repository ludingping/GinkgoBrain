"""v0.7 价格/量价正交维度叠加 main_vs_retail d1.

Hypothesis:
  main_vs_retail d1 (v0.5 winner, IC L-S t=+6.20, portfolio +5.54%) 是 fund-flow 维度;
  叠加 1 个正交 (corr < 0.3) 价格/动量信号可以推到 +8~10%.

Mongo 只有 fund-flow + price (无 volume), 候选限定纯价格类:
  1. 1d reversal: -quote_change
  2. 5d reversal: -ret_5d
  3. 20d momentum: ret_20d
  4. 60d momentum: ret_60d
  5. 20d volatility: std(ret_1d, 20d)
  6. 20d position: (close - low_20d) / (high_20d - low_20d)

Step 1: 计算 base (main_vs_retail d1) + 6 个 price 候选 (T 时点)
Step 2: 跑 7x7 IC corr 矩阵 (截面 rank 的 Spearman) - 平均后 corr
Step 3: 单 IC L-S t-stat (验证有 alpha)
Step 4: 取 base + 1 个 corr < 0.3 候选, 用 rank-mean 合成
Step 5: combined IC + portfolio (avoid Bot 5%, shift(1), friction=0.0001)

⚠️ 全部严格 shift(1) point-in-time, 避免 L7 lookahead bias.
⚠️ IC fwd_ret 用 close.shift(-h)/close.shift(-1)-1 起点, 避免 L2 close[T] leak.
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
import matplotlib.dates as mdates
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("/home/davidlyu/projects/GinkgoBrain/reports/spike_v0_7_price_overlay")
MONGO_HOST = "192.168.1.69"
MONGO_PORT = 27017
MONGO_USER = "admin"
MONGO_DB = "treasure"
MONGO_COLL = "stock_fund_flow"
EFFECTIVE_START_DATE = pd.Timestamp("2022-01-01")
MIN_COVERAGE_DAYS = 500

EXCL_BOT_PCT = 0.05
FRICTION = 0.0001
IC_HORIZON = 5  # 5-day fwd return for IC
CORR_THRESHOLD = 0.3  # 候选 vs base IC corr 上限


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
        {"_id": 0, "stock_id": 1, "date": 1, "price": 1, "quote_change": 1,
         "super_net_inflow_ratio": 1, "big_net_inflow_ratio": 1,
         "middle_net_inflow_ratio": 1, "small_net_inflow_ratio": 1},
    ))
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    for c in ["price", "quote_change", "super_net_inflow_ratio",
              "big_net_inflow_ratio", "middle_net_inflow_ratio",
              "small_net_inflow_ratio"]:
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
        "price": piv("price"),
        "quote_change": piv("quote_change"),
        "super": piv("super_net_inflow_ratio"),
        "big": piv("big_net_inflow_ratio"),
        "middle": piv("middle_net_inflow_ratio"),
        "small": piv("small_net_inflow_ratio"),
    }
    daily_size = panel["price"].notna().sum(axis=1)
    max_size = int(daily_size.max())
    valid_dates = panel["price"].index[daily_size >= max_size * 0.5]
    for k in panel:
        panel[k] = panel[k].loc[valid_dates]
    return panel


def build_signals(panel: dict) -> dict:
    price = panel["price"]
    qc = panel["quote_change"]

    base = (panel["super"] + panel["big"]) - (panel["middle"] + panel["small"])

    ret_5d = price / price.shift(5) - 1
    ret_20d = price / price.shift(20) - 1
    ret_60d = price / price.shift(60) - 1
    daily_ret = price.pct_change()
    vol_20d = daily_ret.rolling(20).std()
    high_20d = price.rolling(20).max()
    low_20d = price.rolling(20).min()
    position_20d = (price - low_20d) / (high_20d - low_20d).replace(0, np.nan)

    candidates = {
        "1d_reversal": -qc,
        "5d_reversal": -ret_5d,
        "20d_momentum": ret_20d,
        "60d_momentum": ret_60d,
        "20d_volatility": vol_20d,
        "20d_position": position_20d,
    }
    return {"base_main_vs_retail_d1": base, "candidates": candidates}


def compute_fwd_ret(price: pd.DataFrame, horizon: int) -> pd.DataFrame:
    return price.shift(-horizon) / price.shift(-1) - 1


def cross_sectional_rank(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.rank(axis=1, pct=True)


def ic_corr_matrix(signals: dict) -> pd.DataFrame:
    names = list(signals.keys())
    ranks = {n: cross_sectional_rank(s) for n, s in signals.items()}
    common_dates = None
    for r in ranks.values():
        idx = r.dropna(how="all").index
        common_dates = idx if common_dates is None else common_dates.intersection(idx)
    common_dates = common_dates[-min(800, len(common_dates)):]

    n = len(names)
    mat = np.full((n, n), np.nan)
    counts = np.zeros((n, n), dtype=int)
    for date in common_dates:
        row_data = {}
        for name in names:
            row = ranks[name].loc[date].dropna() if date in ranks[name].index else None
            if row is not None and len(row) > 20:
                row_data[name] = row
        if len(row_data) < 2:
            continue
        common_stocks = set.intersection(*(set(r.index) for r in row_data.values()))
        if len(common_stocks) < 20:
            continue
        for i, a in enumerate(names):
            if a not in row_data:
                continue
            for j, b in enumerate(names):
                if b not in row_data:
                    continue
                if i > j:
                    continue
                stocks = list(common_stocks)
                va = row_data[a].loc[stocks].values
                vb = row_data[b].loc[stocks].values
                if np.std(va) == 0 or np.std(vb) == 0:
                    continue
                c = np.corrcoef(va, vb)[0, 1]
                if np.isnan(c):
                    continue
                if np.isnan(mat[i, j]):
                    mat[i, j] = 0.0
                mat[i, j] += c
                counts[i, j] += 1
    out = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i, n):
            if counts[i, j] > 0:
                out[i, j] = mat[i, j] / counts[i, j]
                out[j, i] = out[i, j]
    return pd.DataFrame(out, index=names, columns=names)


def signal_ic_stats(signal: pd.DataFrame, fwd_ret: pd.DataFrame) -> dict:
    sig = signal.copy()
    fwd = fwd_ret.copy()
    common = sig.index.intersection(fwd.index)
    sig = sig.loc[common]
    fwd = fwd.loc[common]

    ic_daily = []
    ls_daily = []
    bucket_means = [[] for _ in range(10)]
    for date in sig.index:
        s = sig.loc[date].dropna()
        f = fwd.loc[date].dropna()
        common_stocks = s.index.intersection(f.index)
        if len(common_stocks) < 30:
            continue
        s = s.loc[common_stocks]
        f = f.loc[common_stocks]
        if s.std() == 0 or f.std() == 0:
            continue
        rho, _ = spearmanr(s, f)
        if not np.isnan(rho):
            ic_daily.append(rho)
        ranks = s.rank(pct=True)
        for i in range(10):
            lo, hi = i / 10, (i + 1) / 10
            mask = (ranks > lo) & (ranks <= hi) if i > 0 else (ranks >= lo) & (ranks <= hi)
            if mask.sum() > 0:
                bucket_means[i].append(f[mask].mean())
        top_thr = ranks.quantile(0.95)
        bot_thr = ranks.quantile(0.05)
        top = f[ranks >= top_thr].mean()
        bot = f[ranks <= bot_thr].mean()
        if not (np.isnan(top) or np.isnan(bot)):
            ls_daily.append(top - bot)

    ic_arr = np.asarray(ic_daily)
    ls_arr = np.asarray(ls_daily)
    n_ic = len(ic_arr)
    n_ls = len(ls_arr)
    ic_mean = float(ic_arr.mean()) if n_ic > 0 else float("nan")
    ic_std = float(ic_arr.std(ddof=1)) if n_ic > 1 else float("nan")
    ic_t = ic_mean / ic_std * np.sqrt(n_ic) if (n_ic > 1 and ic_std > 0) else float("nan")
    ls_mean = float(ls_arr.mean()) if n_ls > 0 else float("nan")
    ls_std = float(ls_arr.std(ddof=1)) if n_ls > 1 else float("nan")
    ls_t = ls_mean / ls_std * np.sqrt(n_ls) if (n_ls > 1 and ls_std > 0) else float("nan")
    bucket_avg = [float(np.mean(b)) if len(b) > 0 else float("nan") for b in bucket_means]
    return {
        "n_days": n_ic,
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "ic_t": float(ic_t),
        "ls_mean": ls_mean,
        "ls_t": float(ls_t),
        "buckets": bucket_avg,
        "bucket_b1": bucket_avg[0],
        "bucket_b10": bucket_avg[-1],
    }


def rank_mean_combine(a: pd.DataFrame, b: pd.DataFrame, wa: float = 0.5) -> pd.DataFrame:
    ra = a.rank(axis=1, pct=True)
    rb = b.rank(axis=1, pct=True)
    return ra * wa + rb * (1 - wa)


def portfolio_avoid_bot(score: pd.DataFrame, price: pd.DataFrame,
                       threshold: float, friction: float) -> dict:
    universe_mask = price.notna() & (price > 0)
    daily_ret = price.pct_change()
    bot_thr = score.quantile(threshold, axis=1)
    is_bot = score.le(bot_thr, axis=0).fillna(False)
    final_mask = universe_mask & (~is_bot)
    final_mask = final_mask.shift(1).fillna(False).astype(bool)

    masked = daily_ret.where(final_mask)
    daily_port = masked.mean(axis=1, skipna=True) - friction
    daily_port = daily_port.fillna(0.0)
    eq = (1 + daily_port).cumprod()

    mask_int = final_mask.astype(int)
    daily_turn = (mask_int.diff().abs().sum(axis=1) / mask_int.sum(axis=1).replace(0, np.nan)).fillna(0.0)

    cum = float(eq.iloc[-1] - 1)
    n = max(1, len(eq) - 1)
    ann = float(eq.iloc[-1] ** (252 / n) - 1) if eq.iloc[-1] > 0 else float("nan")
    sh = float(daily_port.mean() / daily_port.std() * np.sqrt(252)) if daily_port.std() > 0 else 0.0
    dd = float((eq / eq.cummax() - 1).min())
    return {"daily_ret": daily_port, "equity": eq,
            "cum": cum, "ann": ann, "sharpe": sh, "max_dd": dd,
            "avg_turnover": float(daily_turn.mean())}


def bm2_baseline(price: pd.DataFrame, friction: float) -> dict:
    universe_mask = price.notna() & (price > 0)
    universe_mask = universe_mask.shift(1).fillna(False).astype(bool)
    daily_ret = price.pct_change()
    daily_port = daily_ret.where(universe_mask).mean(axis=1, skipna=True) - friction
    daily_port = daily_port.fillna(0.0)
    eq = (1 + daily_port).cumprod()
    cum = float(eq.iloc[-1] - 1)
    n = max(1, len(eq) - 1)
    ann = float(eq.iloc[-1] ** (252 / n) - 1)
    sh = float(daily_port.mean() / daily_port.std() * np.sqrt(252)) if daily_port.std() > 0 else 0.0
    dd = float((eq / eq.cummax() - 1).min())
    return {"daily_ret": daily_port, "equity": eq,
            "cum": cum, "ann": ann, "sharpe": sh, "max_dd": dd, "avg_turnover": 0.0}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_filter()
    panel = pivot_panel(df)
    logger.info("Trade dates: %d, stocks: %d", len(panel["price"]), panel["price"].shape[1])

    sigs = build_signals(panel)
    base_name = "main_vs_retail_d1"
    all_signals = {base_name: sigs["base_main_vs_retail_d1"]}
    all_signals.update(sigs["candidates"])

    print("\n========== Step 2: 7x7 IC corr matrix ==========")
    corr_mat = ic_corr_matrix(all_signals)
    print(corr_mat.round(3).to_string())
    corr_mat.to_csv(OUTPUT_DIR / "corr_matrix.csv")

    base_corrs = corr_mat[base_name].drop(base_name).sort_values(key=lambda x: x.abs())
    print(f"\nCandidates corr to {base_name}:")
    for name, c in base_corrs.items():
        flag = "ORTHOGONAL" if abs(c) < CORR_THRESHOLD else "correlated"
        print(f"  {name:<24} {c:+.3f}  {flag}")

    print(f"\n========== Step 3: 单信号 IC + L-S (h={IC_HORIZON}d, close.shift(-1) start) ==========")
    fwd_ret = compute_fwd_ret(panel["price"], IC_HORIZON)
    ic_rows = []
    for name, sig in all_signals.items():
        stats = signal_ic_stats(sig, fwd_ret)
        ic_rows.append({
            "signal": name,
            "n_days": stats["n_days"],
            "ic_mean_%": round(stats["ic_mean"] * 100, 3),
            "ic_t": round(stats["ic_t"], 2),
            "ls_mean_%": round(stats["ls_mean"] * 100, 3),
            "ls_t": round(stats["ls_t"], 2),
            "b1_%": round(stats["bucket_b1"] * 100, 3),
            "b10_%": round(stats["bucket_b10"] * 100, 3),
        })
    ic_df = pd.DataFrame(ic_rows)
    print(ic_df.to_string(index=False))
    ic_df.to_csv(OUTPUT_DIR / "single_ic.csv", index=False)

    eligible = []
    for name, c in base_corrs.items():
        if abs(c) >= CORR_THRESHOLD:
            continue
        row = next(r for r in ic_rows if r["signal"] == name)
        if abs(row["ls_t"]) < 2.0:
            continue
        eligible.append((name, c, row["ls_t"]))
    print(f"\n========== Step 4: 合格候选 (corr<{CORR_THRESHOLD} AND |ls_t|>=2) ==========")
    for name, c, t in eligible:
        print(f"  {name}: corr={c:+.3f}, ls_t={t:+.2f}")
    if not eligible:
        print("  无合格候选 - 单信号收尾")

    base_sig = all_signals[base_name]
    combos = {base_name: base_sig}
    for name, c, t in eligible:
        cand_sig = all_signals[name]
        single_t = next(r for r in ic_rows if r["signal"] == name)["ls_t"]
        cand_for_combine = -cand_sig if single_t < 0 else cand_sig
        combos[f"base + {name}"] = rank_mean_combine(base_sig, cand_for_combine, wa=0.5)

    print(f"\n========== Step 5: combined IC + L-S ==========")
    combo_ic_rows = []
    for name, sig in combos.items():
        stats = signal_ic_stats(sig, fwd_ret)
        combo_ic_rows.append({
            "signal": name,
            "ic_mean_%": round(stats["ic_mean"] * 100, 3),
            "ic_t": round(stats["ic_t"], 2),
            "ls_mean_%": round(stats["ls_mean"] * 100, 3),
            "ls_t": round(stats["ls_t"], 2),
        })
    combo_ic_df = pd.DataFrame(combo_ic_rows)
    print(combo_ic_df.to_string(index=False))
    combo_ic_df.to_csv(OUTPUT_DIR / "combined_ic.csv", index=False)

    print(f"\n========== Step 6: avoid Bot 5% portfolio @ friction={FRICTION} ==========")
    bm2 = bm2_baseline(panel["price"], FRICTION)
    print(f"  {'BM2 (equal-weight univ)':<40} cum {bm2['cum']*100:+7.2f}%  ann {bm2['ann']*100:+6.2f}%  Sharpe {bm2['sharpe']:+.2f}  DD {bm2['max_dd']*100:+6.2f}%")
    port_results = {}
    port_rows = []
    for name, sig in combos.items():
        r = portfolio_avoid_bot(sig, panel["price"], EXCL_BOT_PCT, FRICTION)
        excess = r["cum"] - bm2["cum"]
        port_results[name] = r
        port_rows.append({
            "signal": name,
            "cum_pct": round(r["cum"] * 100, 2),
            "ann_pct": round(r["ann"] * 100, 2),
            "sharpe": round(r["sharpe"], 2),
            "max_dd_pct": round(r["max_dd"] * 100, 2),
            "avg_turn_pct": round(r["avg_turnover"] * 100, 2),
            "excess_pct": round(excess * 100, 2),
        })
        print(f"  {name:<40} cum {r['cum']*100:+7.2f}%  ann {r['ann']*100:+6.2f}%  Sharpe {r['sharpe']:+.2f}  DD {r['max_dd']*100:+6.2f}%  turn {r['avg_turnover']*100:+5.2f}%  excess {excess*100:+6.2f}%")
    port_df = pd.DataFrame(port_rows)
    port_df.to_csv(OUTPUT_DIR / "portfolio_summary.csv", index=False)

    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps({
            "params": {
                "ic_horizon": IC_HORIZON, "friction": FRICTION,
                "excl_bot_pct": EXCL_BOT_PCT, "corr_threshold": CORR_THRESHOLD,
                "candidates": list(sigs["candidates"].keys()),
            },
            "single_ic": ic_rows,
            "eligible": [(n, float(c), float(t)) for n, c, t in eligible],
            "combined_ic": combo_ic_rows,
            "portfolio": port_rows,
            "bm2": {"cum_pct": bm2["cum"]*100, "ann_pct": bm2["ann"]*100, "sharpe": bm2["sharpe"]},
        }, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr_mat.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_mat)))
    ax.set_yticks(range(len(corr_mat)))
    ax.set_xticklabels(corr_mat.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr_mat.index)
    for i in range(len(corr_mat)):
        for j in range(len(corr_mat)):
            v = corr_mat.iloc[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if abs(v) > 0.5 else "black", fontsize=9)
    ax.set_title("v0.7 IC corr matrix (avg cross-section Spearman)")
    plt.colorbar(im, ax=ax)
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "corr_matrix.png", dpi=110); plt.close()

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(bm2["equity"].index, bm2["equity"].values, "k--",
            label=f"BM2 ({bm2['cum']*100:+.2f}%)", linewidth=1.6)
    colors = ["C0", "C2", "C3", "C1", "C4", "C5", "C6"]
    for (name, r), c in zip(port_results.items(), colors):
        ax.plot(r["equity"].index, r["equity"].values, color=c, linewidth=1.5,
                label=f"{name} ({r['cum']*100:+.2f}%)")
    ax.set_title(f"v0.7 portfolio - avoid Bot {int(EXCL_BOT_PCT*100)}% @ friction={FRICTION}")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "cumulative.png", dpi=110); plt.close()

    print(f"\nAll outputs in: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
