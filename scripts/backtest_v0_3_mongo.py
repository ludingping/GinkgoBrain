"""v0.3 完整 backtest — MongoDB 4 年数据 (2022-2025) 跑 v0.2 信号.

目标：验证 v0.2 "持续吸筹 + 价格温和" 信号在含 2022 熊市 + 2023 震荡 + 2024-25 反弹的
跨周期样本上是否保留"低 DD 防守 + Sharpe ≥1"的特征。

数据源：MongoDB treasure.stock_fund_flow
- 实际有效段：2022-01 ~ 2025-10（≈900 trade days）
- 筛选：覆盖 ≥500 trade days 的股（4962 只 / 88.8%）

撮合简化（mongo 无 OHLC，简化合理性见 spike_v0_3_data_quality verdict）：
- T+1 撮合价 = T+1 close（mongo 只有 close，没有 open）
- 跳过一字涨跌停判定（limit flags 全 False）
- 跳过容量截断（cap_buy = 1e9 元上限）
- 保留 entry-only rebalance + 持有期 ≥3 + 成本 0.20%

输出 reports/v0.3_mongo_{ts}/:
- metrics.json / verdict.md / equity.csv / yearly_metrics.csv / daily_diagnostics.csv
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

from strategies.cn_a_big_money_rotation.portfolio import (
    CostParams, PortfolioState,
    apply_orders_at_t1, decide_orders, holdings_in_lock,
)
from strategies.cn_a_big_money_rotation.signal import (
    TARGET_TOP_N, compute_v0_2_panel, select_v0_2_top_n_for_date,
)
from strategies.cn_a_big_money_rotation.evaluate import (
    ICStats, StrategyMetrics,
    annualize_return, compute_information_ratio, compute_max_drawdown,
    compute_sharpe,
)

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("/home/davidlyu/projects/GinkgoBrain/reports") / f"v0.3_mongo_{datetime.now():%Y%m%d_%H%M%S}"
MONGO_HOST = "192.168.1.69"
MONGO_PORT = 27017
MONGO_USER = "admin"
MONGO_DB = "treasure"
MONGO_COLL = "stock_fund_flow"

MIN_COVERAGE_DAYS = 500
EFFECTIVE_START_DATE = pd.Timestamp("2022-01-01")
INITIAL_CAPITAL = 10_000_000.0
N_TARGET = TARGET_TOP_N
HOLDING_MIN_DAYS = 3
V0_2_N_WINDOW = 10
V0_2_POSITIVE_DAYS_MIN = 7
V0_2_PRICE_RANGE = (-0.05, 0.08)


def load_and_filter() -> pd.DataFrame:
    from pymongo import MongoClient
    pwd = os.environ.get("MONGO_PWD")
    if not pwd:
        raise RuntimeError("MONGO_PWD env var required")
    client = MongoClient(host=MONGO_HOST, port=MONGO_PORT,
                         username=MONGO_USER, password=pwd, authSource="admin")
    col = client[MONGO_DB][MONGO_COLL]
    logger.info("Loading mongo (>=2022 only) ...")
    rows = list(col.find(
        {"date": {"$gte": EFFECTIVE_START_DATE.to_pydatetime()}},
        {"_id": 0, "stock_id": 1, "name": 1, "date": 1, "price": 1,
         "main_net_inflow": 1, "main_net_inflow_ratio": 1},
    ))
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["main_net_inflow"] = pd.to_numeric(df["main_net_inflow"], errors="coerce")
    df["main_net_inflow_ratio"] = pd.to_numeric(df["main_net_inflow_ratio"], errors="coerce")
    cov = df.groupby("stock_id").size()
    keep = cov[cov >= MIN_COVERAGE_DAYS].index
    before = len(df)
    df = df[df["stock_id"].isin(keep)]
    logger.info("Filtered: %d -> %d rows (keep %d stocks with >=%d days)",
                before, len(df), len(keep), MIN_COVERAGE_DAYS)
    return df


def pivot_panel(df: pd.DataFrame) -> dict:
    inflow_pivot = df.pivot_table(index="date", columns="stock_id",
                                  values="main_net_inflow", aggfunc="first")
    ratio_pivot = df.pivot_table(index="date", columns="stock_id",
                                 values="main_net_inflow_ratio", aggfunc="first")
    price_pivot = df.pivot_table(index="date", columns="stock_id",
                                 values="price", aggfunc="first")
    # 反推 total_amount = |net_inflow / (ratio/100)|；ratio≈0 时 amount → NaN
    with np.errstate(divide="ignore", invalid="ignore"):
        amount_pivot = (inflow_pivot / (ratio_pivot / 100.0)).abs()
    amount_pivot = amount_pivot.replace([np.inf, -np.inf], np.nan)
    return {"inflow": inflow_pivot, "amount": amount_pivot,
            "price": price_pivot, "ratio": ratio_pivot}


def daily_universe_mongo(t, panel: dict) -> set[str]:
    if t not in panel["price"].index or t not in panel["inflow"].index:
        return set()
    price_row = panel["price"].loc[t]
    inflow_row = panel["inflow"].loc[t]
    valid = price_row.notna() & inflow_row.notna() & (price_row > 0)
    return set(valid[valid].index.tolist())


MIN_UNIVERSE_RATIO = 0.5  # trim：当日 universe < 历史 max × 此比例 → 视为数据稀疏天剔除


def run_backtest_mongo(panel: dict) -> dict:
    state = PortfolioState(cash=INITIAL_CAPITAL, current_step=0)
    cost_params = CostParams()
    trade_dates_raw = sorted(panel["price"].index)

    # Trim 数据稀疏天：先计算每天的 universe size，剔除"末尾断崖"
    daily_universe_sizes = panel["price"].notna().sum(axis=1)
    max_univ = int(daily_universe_sizes.max())
    threshold = max_univ * MIN_UNIVERSE_RATIO
    valid_dates_mask = daily_universe_sizes >= threshold
    trade_dates = [t for t in trade_dates_raw if valid_dates_mask.get(t, False)]
    n_dropped = len(trade_dates_raw) - len(trade_dates)
    logger.info("Trim: dropped %d sparse days (universe < %d/2). Backtest spans %d trade dates",
                n_dropped, max_univ, len(trade_dates))

    v0_2 = compute_v0_2_panel(
        panel["inflow"], panel["amount"], panel["price"],
        n_window=V0_2_N_WINDOW,
        positive_days_min=V0_2_POSITIVE_DAYS_MIN,
        price_range=V0_2_PRICE_RANGE,
    )
    logger.info("v0.2 panel: score_filtered non-NaN = %d",
                int(v0_2["score_filtered"].notna().sum().sum()))

    equity_log = []
    daily_returns = []
    bm2_equity = [1.0]
    selected_history = {}
    diag_rows = []
    pending_orders = []

    for step, t in enumerate(trade_dates):
        state.current_step = step
        universe = daily_universe_mongo(t, panel)
        close_map = {}
        if t in panel["price"].index:
            for c, v in panel["price"].loc[t].dropna().items():
                if v > 0:
                    close_map[c] = float(v)

        # 1. T+1 撮合（用 close 当 open）
        if pending_orders:
            apply_orders_at_t1(
                state=state, orders=pending_orders,
                t1_open=close_map,
                t1_limit_up_flag={},
                t1_limit_down_flag={},
                cap_buy_amount={c: 1e9 for c in close_map},
                cost_params=cost_params,
            )

        # 2. 当日 equity
        equity = state.total_assets(close_map)
        equity_log.append((t, equity))
        if len(equity_log) >= 2 and equity_log[-2][1] > 0:
            daily_returns.append((equity - equity_log[-2][1]) / equity_log[-2][1])
        else:
            daily_returns.append(0.0)

        # 3. 选股
        if t not in v0_2["score_filtered"].index:
            pending_orders = []
            bm2_equity.append(bm2_equity[-1])
            continue
        score_row = v0_2["score_filtered"].loc[t]
        sel_list = select_v0_2_top_n_for_date(score_row, universe, n=N_TARGET)
        selected_codes = set(sel_list)
        selected_history[t] = sel_list

        # 4. BM2 等权 universe
        if step > 0 and trade_dates[step - 1] in panel["price"].index:
            prev_close = panel["price"].loc[trade_dates[step - 1]]
            shared = (set(close_map.keys()) & set(prev_close.index) & universe)
            rets = []
            for c in shared:
                pc = prev_close.get(c, np.nan)
                cc = close_map.get(c, np.nan)
                if pd.notna(pc) and pc > 0 and pd.notna(cc):
                    rets.append((cc - pc) / pc)
            bm2_ret = float(np.mean(rets)) - 0.0001 if rets else 0.0
            bm2_equity.append(bm2_equity[-1] * (1 + bm2_ret))
        else:
            bm2_equity.append(bm2_equity[-1])

        # 5. 决定下日订单
        locked = holdings_in_lock(state.positions, step + 1, HOLDING_MIN_DAYS)
        total_assets = state.total_assets(close_map)
        orders = decide_orders(
            prev_positions=state.positions, new_selected=selected_codes,
            holding_locked=locked, cash_avail=state.cash,
            total_assets=total_assets, n_target=N_TARGET,
        )
        pending_orders = orders

        diag_rows.append({
            "trade_date": t,
            "universe_size": len(universe),
            "selected_size": len(selected_codes),
            "n_positions": len(state.positions),
            "cash": state.cash,
            "equity": equity,
        })
        if step % 100 == 0:
            logger.info("step %d/%d (%s) | univ=%d sel=%d pos=%d eq=%.0f",
                        step, len(trade_dates), t.date(), len(universe),
                        len(selected_codes), len(state.positions), equity)

    eq_idx = pd.DatetimeIndex([t for t, _ in equity_log])
    return {
        "equity": pd.Series([e for _, e in equity_log], index=eq_idx),
        "daily_ret": pd.Series(daily_returns, index=eq_idx),
        "bm2_equity": pd.Series(bm2_equity[:len(equity_log)], index=eq_idx),
        "selected_history": selected_history,
        "diagnostics": pd.DataFrame(diag_rows).set_index("trade_date") if diag_rows else None,
    }


def evaluate_and_report(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    eq = result["equity"]
    daily_ret = result["daily_ret"]
    bm2 = result["bm2_equity"]
    bm2_daily = bm2.pct_change().fillna(0.0)

    cum = float(eq.iloc[-1] / eq.iloc[0] - 1.0) if len(eq) > 1 else 0.0
    n = max(1, len(eq) - 1)
    ann = annualize_return(cum, n)
    bm2_cum = float(bm2.iloc[-1] - 1.0)
    bm2_ann = annualize_return(bm2_cum, n)
    excess_bm2 = ann - bm2_ann
    sharpe = compute_sharpe(daily_ret)
    ir_bm2 = compute_information_ratio(daily_ret, bm2_daily)
    max_dd = compute_max_drawdown(eq)

    print(f"\n========== v0.3 mongo backtest 全样本 ==========")
    print(f"  trade_dates:    {len(eq):,}")
    print(f"  cum return:     {cum*100:+.2f}%")
    print(f"  annualized:     {ann*100:+.2f}%")
    print(f"  BM2 等权 cum:   {bm2_cum*100:+.2f}%")
    print(f"  BM2 annualized: {bm2_ann*100:+.2f}%")
    print(f"  excess vs BM2:  {excess_bm2*100:+.2f}%")
    print(f"  sharpe:         {sharpe:+.2f}")
    print(f"  IR vs BM2:      {ir_bm2:+.2f}")
    print(f"  max drawdown:   {max_dd*100:+.2f}%")

    # 按年度
    df_yearly = pd.DataFrame({
        "equity": eq, "daily_ret": daily_ret, "bm2": bm2, "bm2_ret": bm2_daily,
    })
    df_yearly["year"] = df_yearly.index.year
    yearly_rows = []
    print(f"\n========== 按年度拆分 ==========")
    print(f"{'year':>6} | {'cum%':>9} | {'BM2%':>9} | {'excess%':>9} | {'sharpe':>7} | {'MaxDD%':>8} | {'days':>5}")
    for y, g in df_yearly.groupby("year"):
        y_eq = g["equity"]
        y_cum = float(y_eq.iloc[-1] / y_eq.iloc[0] - 1.0) if len(y_eq) > 1 else 0.0
        y_bm2 = float(g["bm2"].iloc[-1] / g["bm2"].iloc[0] - 1.0) if len(g["bm2"]) > 1 else 0.0
        y_sharpe = compute_sharpe(g["daily_ret"])
        y_dd = compute_max_drawdown(y_eq)
        excess = y_cum - y_bm2
        yearly_rows.append({
            "year": int(y), "cum_pct": round(y_cum * 100, 2),
            "bm2_pct": round(y_bm2 * 100, 2),
            "excess_pct": round(excess * 100, 2),
            "sharpe": round(y_sharpe, 2), "max_dd_pct": round(y_dd * 100, 2),
            "n_days": len(g),
        })
        print(f"  {int(y):>4} | {y_cum*100:>+8.2f}% | {y_bm2*100:>+8.2f}% | {excess*100:>+8.2f}% | {y_sharpe:>+7.2f} | {y_dd*100:>+7.2f}% | {len(g):>5}")
    pd.DataFrame(yearly_rows).to_csv(out_dir / "yearly_metrics.csv", index=False)

    # verdict
    if excess_bm2 >= 0.03 and sharpe >= 0.8 and abs(max_dd) <= 0.30:
        verdict_str = "PASS-prelim (跨周期超额 ≥3% + Sharpe ≥0.8 + DD ≤30%)"
    elif excess_bm2 < 0 and sharpe >= 1.5 and abs(max_dd) <= 0.15:
        verdict_str = "DEFENSIVE (跑输 BM2 但低 DD 高 Sharpe — 真防守 portfolio)"
    elif excess_bm2 < 0:
        verdict_str = "NO-ALPHA-vs-BM2 (跑输等权全市场)"
    else:
        verdict_str = "WEAK (软门槛部分通过)"
    print(f"\n*** VERDICT: {verdict_str} ***")

    pd.DataFrame({
        "equity": eq, "daily_return": daily_ret, "bm2_equity": bm2,
    }).to_csv(out_dir / "equity.csv", index_label="trade_date")

    if result["diagnostics"] is not None:
        result["diagnostics"].to_csv(out_dir / "daily_diagnostics.csv")

    payload = {
        "verdict": verdict_str,
        "params": {
            "data_source": "MongoDB treasure.stock_fund_flow",
            "effective_start": str(EFFECTIVE_START_DATE.date()),
            "min_coverage_days": MIN_COVERAGE_DAYS,
            "signal": "v0.2 (持续吸筹 + 价格温和 [-5%, +8%])",
            "撮合简化": "用 close 当 T+1 开盘价；跳过涨跌停 + 容量截断",
        },
        "metrics": {
            "cumulative_return": cum, "annualized_return": ann,
            "bm2_cum": bm2_cum, "bm2_annualized": bm2_ann,
            "excess_return_bm2": excess_bm2, "sharpe": sharpe,
            "ir_bm2": ir_bm2, "max_drawdown": max_dd,
            "n_trade_dates": len(eq),
        },
        "yearly": yearly_rows,
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    md = [
        f"# v0.3 MongoDB Backtest — {verdict_str}",
        "",
        f"**信号**: v0.2 (持续吸筹 + 价格温和 [-5%, +8%])",
        f"**数据**: MongoDB 4 年 ({EFFECTIVE_START_DATE.date()} → {eq.index[-1].date()})",
        f"**universe**: 覆盖 ≥{MIN_COVERAGE_DAYS} 天 的股",
        "",
        "## 全样本",
        f"- 累积: {cum*100:+.2f}%, 年化 {ann*100:+.2f}%",
        f"- BM2 等权累积: {bm2_cum*100:+.2f}%, 年化 {bm2_ann*100:+.2f}%",
        f"- 超额 vs BM2: {excess_bm2*100:+.2f}%",
        f"- Sharpe {sharpe:+.2f}, IR vs BM2 {ir_bm2:+.2f}, MaxDD {max_dd*100:+.2f}%",
        "",
        "## 按年度",
        "| year | cum% | BM2% | excess% | sharpe | MaxDD% | days |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in yearly_rows:
        md.append(f"| {r['year']} | {r['cum_pct']:+.2f}% | {r['bm2_pct']:+.2f}% | {r['excess_pct']:+.2f}% | {r['sharpe']:+.2f} | {r['max_dd_pct']:+.2f}% | {r['n_days']} |")
    md.extend([
        "",
        "## 撮合简化说明",
        "- mongo 无 OHLC，用 close 当 T+1 开盘买入价（简化）",
        "- 跳过一字涨跌停判定",
        "- 跳过容量截断（cap_buy = 1e9）",
        "- 方向性验证；若 PASS 再补真实撮合细节",
    ])
    (out_dir / "verdict.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nAll outputs in: {out_dir}")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    df = load_and_filter()
    panel = pivot_panel(df)
    logger.info("inflow %s, price %s", panel["inflow"].shape, panel["price"].shape)
    result = run_backtest_mongo(panel)
    evaluate_and_report(result, OUTPUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
