"""v0.8 GBDT 多信号合成 - LightGBM 学 11 features 非线性 + 交叉.

Features (11, all at T):
  Fund-flow 5: super/big/middle/small/main_net_inflow_ratio
  Price/mom 6: quote_change, ret_5d, ret_20d, ret_60d, vol_20d, position_20d

Target: cross-sectional rank of 5d fwd_ret (close.shift(-5)/close.shift(-1)-1, L2 fix)

Walk-forward 3 folds:
  F1: train 2022-04..2023-12 (~1.5y), test 2024-01..2024-12 (1y)
  F2: train 2022-04..2024-12 (~2.5y), test 2025-01..2025-12 (1y)
  F3: train 2022-04..2025-12 (~3.5y), test 2026-01..2026-05 (4mo)

LGBM hyperparams (conservative):
  n_estimators=300, max_depth=5, lr=0.05, num_leaves=31,
  min_child_samples=200, feature_fraction=0.8, bagging_fraction=0.8

Eval:
  OOS IC mean + L-S t-stat per fold + overall
  OOS portfolio: avoid Bot 5% by predicted rank
  对比 single -20d_momentum (+8.49% 4y)

Gate:
  OOS L-S t > 3 across 3 folds
  OOS portfolio 4y excess > +10%
  Feature importance 含 fund-flow + price 各 2+
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import lightgbm as lgb
from scipy.stats import spearmanr

from spike_v0_7_price_overlay import load_and_filter, pivot_panel  # type: ignore

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("/home/davidlyu/projects/GinkgoBrain/reports/spike_v0_8_gbdt")
EXCL_BOT_PCT = 0.05
FRICTION = 0.0001
IC_HORIZON = 5

FOLDS = [
    ("F1", "2022-04-01", "2023-12-31", "2024-01-01", "2024-12-31"),
    ("F2", "2022-04-01", "2024-12-31", "2025-01-01", "2025-12-31"),
    ("F3", "2022-04-01", "2025-12-31", "2026-01-01", "2026-12-31"),
]

LGBM_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "n_estimators": 300,
    "max_depth": 5,
    "num_leaves": 31,
    "min_child_samples": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}

FUND_FLOW_FEATS = ["super_net_inflow_ratio", "big_net_inflow_ratio",
                   "middle_net_inflow_ratio", "small_net_inflow_ratio",
                   "main_net_inflow_ratio"]
PRICE_FEATS = ["quote_change", "ret_5d", "ret_20d", "ret_60d",
               "vol_20d", "position_20d"]
ALL_FEATS = FUND_FLOW_FEATS + PRICE_FEATS


def build_full_panel(panel: dict) -> pd.DataFrame:
    """Long-format panel with all 11 features + target rank."""
    price = panel["price"]
    qc = panel["quote_change"]

    ret_5d = price / price.shift(5) - 1
    ret_20d = price / price.shift(20) - 1
    ret_60d = price / price.shift(60) - 1
    daily_ret = price.pct_change()
    vol_20d = daily_ret.rolling(20).std()
    high_20d = price.rolling(20).max()
    low_20d = price.rolling(20).min()
    position_20d = (price - low_20d) / (high_20d - low_20d).replace(0, np.nan)

    fwd_ret = price.shift(-IC_HORIZON) / price.shift(-1) - 1
    target_rank = fwd_ret.rank(axis=1, pct=True)

    feature_pivots = {
        "super_net_inflow_ratio": panel["super"],
        "big_net_inflow_ratio": panel["big"],
        "middle_net_inflow_ratio": panel["middle"],
        "small_net_inflow_ratio": panel["small"],
        "main_net_inflow_ratio": panel["super"] + panel["big"],
        "quote_change": qc,
        "ret_5d": ret_5d,
        "ret_20d": ret_20d,
        "ret_60d": ret_60d,
        "vol_20d": vol_20d,
        "position_20d": position_20d,
    }

    frames = []
    for name, pv in feature_pivots.items():
        s = pv.stack(dropna=False).rename(name)
        frames.append(s)
    frames.append(target_rank.stack(dropna=False).rename("target_rank"))
    frames.append(fwd_ret.stack(dropna=False).rename("fwd_ret"))

    df_long = pd.concat(frames, axis=1).reset_index()
    df_long.columns.values[:2] = ["date", "stock_id"]
    df_long = df_long.dropna(subset=ALL_FEATS + ["target_rank"])
    return df_long


def run_fold(df_long: pd.DataFrame, fold_name: str, train_start: str, train_end: str,
             test_start: str, test_end: str) -> dict:
    train_mask = (df_long["date"] >= train_start) & (df_long["date"] <= train_end)
    test_mask = (df_long["date"] >= test_start) & (df_long["date"] <= test_end)
    train = df_long[train_mask]
    test = df_long[test_mask]
    logger.info("Fold %s: train %d (%s..%s), test %d (%s..%s)",
                fold_name, len(train), train_start, train_end, len(test), test_start, test_end)

    if len(test) == 0:
        logger.warning("Fold %s test empty, skip", fold_name)
        return None

    X_train = train[ALL_FEATS].values
    y_train = train["target_rank"].values
    X_test = test[ALL_FEATS].values

    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    out = test[["date", "stock_id", "fwd_ret", "target_rank"]].copy()
    out["pred"] = preds
    out["fold"] = fold_name

    fi = dict(zip(ALL_FEATS, model.feature_importances_))
    return {"predictions": out, "feature_importance": fi}


def evaluate_oos(oos: pd.DataFrame) -> dict:
    ic_daily = []
    ls_daily = []
    for date, grp in oos.groupby("date"):
        if len(grp) < 30:
            continue
        p = grp["pred"].values
        f = grp["fwd_ret"].values
        mask = ~(np.isnan(p) | np.isnan(f))
        if mask.sum() < 30 or p[mask].std() == 0 or f[mask].std() == 0:
            continue
        rho, _ = spearmanr(p[mask], f[mask])
        if not np.isnan(rho):
            ic_daily.append(rho)
        rk = pd.Series(p[mask]).rank(pct=True).values
        f_arr = f[mask]
        top = f_arr[rk >= 0.95].mean() if (rk >= 0.95).sum() > 0 else np.nan
        bot = f_arr[rk <= 0.05].mean() if (rk <= 0.05).sum() > 0 else np.nan
        if not (np.isnan(top) or np.isnan(bot)):
            ls_daily.append(top - bot)
    ic_arr = np.asarray(ic_daily)
    ls_arr = np.asarray(ls_daily)
    ic_mean = float(ic_arr.mean()) if len(ic_arr) > 0 else np.nan
    ic_std = float(ic_arr.std(ddof=1)) if len(ic_arr) > 1 else np.nan
    ic_t = ic_mean / ic_std * np.sqrt(len(ic_arr)) if (len(ic_arr) > 1 and ic_std > 0) else np.nan
    ls_mean = float(ls_arr.mean()) if len(ls_arr) > 0 else np.nan
    ls_std = float(ls_arr.std(ddof=1)) if len(ls_arr) > 1 else np.nan
    ls_t = ls_mean / ls_std * np.sqrt(len(ls_arr)) if (len(ls_arr) > 1 and ls_std > 0) else np.nan
    return {"n_days": len(ic_arr), "ic_mean": ic_mean, "ic_t": float(ic_t),
            "ls_mean": ls_mean, "ls_t": float(ls_t)}


def portfolio_from_pred(oos: pd.DataFrame, price: pd.DataFrame, friction: float) -> dict:
    pred_pivot = oos.pivot_table(index="date", columns="stock_id", values="pred", aggfunc="first")
    pred_pivot = pred_pivot.reindex(index=price.index, columns=price.columns)

    universe_mask = price.notna() & (price > 0)
    daily_ret = price.pct_change()
    bot_thr = pred_pivot.quantile(EXCL_BOT_PCT, axis=1)
    is_bot = pred_pivot.le(bot_thr, axis=0).fillna(False)
    final_mask = universe_mask & (~is_bot)
    final_mask = final_mask.shift(1).fillna(False).astype(bool)

    masked = daily_ret.where(final_mask)
    daily_port = masked.mean(axis=1, skipna=True) - friction
    daily_port = daily_port.fillna(0.0)
    eq = (1 + daily_port).cumprod()

    has_pred = pred_pivot.notna().any(axis=1)
    eq_oos = eq[has_pred]
    daily_port_oos = daily_port[has_pred]
    if len(eq_oos) < 2:
        return {"equity": eq, "cum": 0.0, "ann": 0.0, "sharpe": 0.0, "max_dd": 0.0,
                "avg_turnover": 0.0, "n_oos_days": 0,
                "first_date": "", "last_date": ""}
    eq_oos_norm = eq_oos / eq_oos.iloc[0]
    cum = float(eq_oos_norm.iloc[-1] - 1)
    n = max(1, len(eq_oos) - 1)
    ann = float(eq_oos_norm.iloc[-1] ** (252 / n) - 1) if eq_oos_norm.iloc[-1] > 0 else float("nan")
    sh = float(daily_port_oos.mean() / daily_port_oos.std() * np.sqrt(252)) if daily_port_oos.std() > 0 else 0.0
    dd = float((eq_oos_norm / eq_oos_norm.cummax() - 1).min())
    mask_int = final_mask.astype(int)
    daily_turn = (mask_int.diff().abs().sum(axis=1) / mask_int.sum(axis=1).replace(0, np.nan)).fillna(0.0)
    avg_turn = float(daily_turn[has_pred].mean())
    return {"equity": eq_oos_norm, "cum": cum, "ann": ann, "sharpe": sh, "max_dd": dd,
            "avg_turnover": avg_turn, "n_oos_days": len(eq_oos),
            "first_date": str(eq_oos.index[0].date()), "last_date": str(eq_oos.index[-1].date())}


def bm2_oos(price: pd.DataFrame, oos_dates: pd.DatetimeIndex, friction: float) -> dict:
    universe_mask = price.notna() & (price > 0)
    universe_mask = universe_mask.shift(1).fillna(False).astype(bool)
    daily_ret = price.pct_change()
    daily_port = daily_ret.where(universe_mask).mean(axis=1, skipna=True) - friction
    daily_port = daily_port.fillna(0.0)
    eq = (1 + daily_port).cumprod()
    eq_oos = eq.loc[oos_dates]
    if len(eq_oos) < 2:
        return {"equity": eq, "cum": 0.0, "ann": 0.0}
    eq_oos_norm = eq_oos / eq_oos.iloc[0]
    cum = float(eq_oos_norm.iloc[-1] - 1)
    n = max(1, len(eq_oos) - 1)
    ann = float(eq_oos_norm.iloc[-1] ** (252 / n) - 1)
    return {"equity": eq_oos_norm, "cum": cum, "ann": ann}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_filter()
    panel = pivot_panel(df)
    logger.info("Trade dates: %d, stocks: %d", len(panel["price"]), panel["price"].shape[1])

    df_long = build_full_panel(panel)
    logger.info("Long-format panel: %d rows", len(df_long))

    print(f"\n========== Walk-forward 3 fold ==========")
    all_preds = []
    fold_results = []
    fi_per_fold = {}
    for fold_name, ts, te, vs, ve in FOLDS:
        r = run_fold(df_long, fold_name, ts, te, vs, ve)
        if r is None:
            continue
        all_preds.append(r["predictions"])
        fi_per_fold[fold_name] = r["feature_importance"]
        ev = evaluate_oos(r["predictions"])
        ev["fold"] = fold_name
        ev["n_test"] = len(r["predictions"])
        fold_results.append(ev)
        print(f"  {fold_name}: n_days={ev['n_days']:3d}  IC={ev['ic_mean']*100:+.3f}% (t={ev['ic_t']:+.2f})  L-S={ev['ls_mean']*100:+.3f}% (t={ev['ls_t']:+.2f})")

    oos_full = pd.concat(all_preds, ignore_index=True)
    oos_full.to_parquet(OUTPUT_DIR / "oos_predictions.parquet")
    overall = evaluate_oos(oos_full)
    overall["fold"] = "OVERALL"
    overall["n_test"] = len(oos_full)
    fold_results.append(overall)
    print(f"  OVERALL: n_days={overall['n_days']}  IC={overall['ic_mean']*100:+.3f}% (t={overall['ic_t']:+.2f})  L-S={overall['ls_mean']*100:+.3f}% (t={overall['ls_t']:+.2f})")

    fold_df = pd.DataFrame(fold_results)
    fold_df.to_csv(OUTPUT_DIR / "oos_ic_per_fold.csv", index=False)

    print(f"\n========== Feature Importance (avg gain across folds) ==========")
    fi_avg = {f: float(np.mean([fi_per_fold[fold].get(f, 0) for fold in fi_per_fold])) for f in ALL_FEATS}
    fi_sorted = sorted(fi_avg.items(), key=lambda x: -x[1])
    for f, imp in fi_sorted:
        kind = "FF" if f in FUND_FLOW_FEATS else "PR"
        print(f"  [{kind}] {f:<32} {imp:8.0f}")
    pd.DataFrame([{"feature": f, "kind": "fund_flow" if f in FUND_FLOW_FEATS else "price",
                   "importance": imp} for f, imp in fi_sorted]).to_csv(
        OUTPUT_DIR / "feature_importance.csv", index=False)

    print(f"\n========== OOS Portfolio (avoid Bot 5% @ friction={FRICTION}) ==========")
    pf = portfolio_from_pred(oos_full, panel["price"], FRICTION)
    bm2 = bm2_oos(panel["price"], pf["equity"].index, FRICTION)
    excess = pf["cum"] - bm2["cum"]
    print(f"  OOS period: {pf['first_date']} → {pf['last_date']}, {pf['n_oos_days']} days")
    print(f"  BM2 (univ 等权):        cum {bm2['cum']*100:+7.2f}%  ann {bm2['ann']*100:+6.2f}%")
    print(f"  GBDT avoid Bot 5%:     cum {pf['cum']*100:+7.2f}%  ann {pf['ann']*100:+6.2f}%  Sharpe {pf['sharpe']:+.2f}  DD {pf['max_dd']*100:+6.2f}%  turn {pf['avg_turnover']*100:+5.2f}%  excess {excess*100:+6.2f}%")

    print(f"\n  Reference: -20d_momentum single 4y +8.49% excess")
    print(f"  GBDT OOS excess: {excess*100:+.2f}% over {pf['n_oos_days']} OOS days")

    summary = {
        "params": {"folds": [list(f) for f in FOLDS], "features": ALL_FEATS,
                   "lgbm": LGBM_PARAMS, "friction": FRICTION,
                   "excl_bot_pct": EXCL_BOT_PCT, "ic_horizon": IC_HORIZON},
        "fold_results": fold_results,
        "feature_importance_avg": fi_avg,
        "portfolio": {"cum_pct": pf["cum"]*100, "ann_pct": pf["ann"]*100,
                      "sharpe": pf["sharpe"], "max_dd_pct": pf["max_dd"]*100,
                      "avg_turn_pct": pf["avg_turnover"]*100, "excess_pct": excess*100,
                      "n_oos_days": pf["n_oos_days"],
                      "first_date": pf["first_date"], "last_date": pf["last_date"]},
        "bm2_oos": {"cum_pct": bm2["cum"]*100, "ann_pct": bm2["ann"]*100},
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(bm2["equity"].index, bm2["equity"].values, "k--",
            label=f"BM2 ({bm2['cum']*100:+.2f}%)", linewidth=1.6)
    ax.plot(pf["equity"].index, pf["equity"].values, "C0", linewidth=1.6,
            label=f"GBDT avoid Bot 5% ({pf['cum']*100:+.2f}%)")
    ax.set_title(f"v0.8 GBDT OOS cumulative ({pf['first_date']} -> {pf['last_date']})")
    ax.legend(); ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "cumulative.png", dpi=110); plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    feats = [f for f, _ in fi_sorted]
    imps = [imp for _, imp in fi_sorted]
    colors = ["C0" if f in FUND_FLOW_FEATS else "C2" for f in feats]
    ax.barh(range(len(feats)), imps, color=colors)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats)
    ax.invert_yaxis()
    ax.set_xlabel("avg gain importance (across folds)")
    ax.set_title("v0.8 GBDT feature importance (blue=fund-flow, green=price)")
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "feat_importance.png", dpi=110); plt.close()

    print(f"\nAll outputs in: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
