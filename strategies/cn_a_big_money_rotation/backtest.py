"""M6 回测主入口：把 M1-M5 串起来跑完整 backtest（设计 §10）.

入口：
- 函数：``run_backtest(BacktestConfig) -> BacktestResult``（供 M7 / notebook 调）
- CLI ：``python -m strategies.cn_a_big_money_rotation.backtest --start ... --end ...``

主循环（每个 trade_date T）：
1. 构造 universe_T（设计 §4，复用 universe.daily_universe）
2. 计算信号 score_A / score_B 与 rank_A / rank_B（M3 compute_signals）
3. 选 selected_T = Top20 by rank_A（M3 select_top_n）
4. 在 T+1（下一个 trade_date）开盘撮合：先 SELL，再 BUY，按 §7 容量截断与涨跌停规则
5. 当日 equity = cash + Σ position.shares × close_t

事后评估：
- daily IC：T 日 score_A 与 T+forward_horizon 累计相对收益的 Spearman 相关
- daily 收益：portfolio equity 日变化率
- 双基准：BM2 等权 universe 日度（含 0.01% 日度摩擦近似 0.20% 双边）；
         BM1 HS300 缺数据时 graceful fallback 至 vs 0% cash 比较
- verdict：M5 render_verdict()

输出 ``reports/v0.1_{run_id}/`` 下 ``metrics.json + verdict.md + equity.csv + daily_ic.csv``.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import (
    compute_money_flow_factors,
    load_kline_close_range,
    load_money_flow_range,
    load_security_list_active,
    load_snapshot_latest,
    recover_float_share,
)
from .universe import daily_universe
from .signal import compute_signals, select_top_n, dual_signal_overlap, TARGET_TOP_N
from .portfolio import (
    HOLDING_MIN_DAYS,
    CostParams,
    PortfolioState,
    apply_orders_at_t1,
    capacity_cap_for_buy,
    decide_orders,
    holdings_in_lock,
    is_one_word_limit_down,
    is_one_word_limit_up,
)
from .evaluate import (
    ICStats,
    StrategyMetrics,
    Verdict,
    annualize_return,
    classify_dual_signal,
    compute_ic,
    compute_ic_stats,
    compute_information_ratio,
    compute_max_drawdown,
    compute_sharpe,
    render_verdict,
)

logger = logging.getLogger(__name__)


# ============================================================================
# 配置 + 结果
# ============================================================================

@dataclass
class BacktestConfig:
    start: date
    end: date
    initial_capital: float = 10_000_000.0   # 1000 万
    n_target: int = TARGET_TOP_N
    holding_min_days: int = HOLDING_MIN_DAYS
    forward_ic_horizon: int = 4              # T+horizon 累计收益做 IC 验证
    output_dir: Path = Path("reports")


@dataclass
class BacktestResult:
    daily_returns: pd.Series
    equity_curve: pd.Series
    daily_ic_a: pd.Series
    daily_ic_b: pd.Series
    bm2_equity: pd.Series
    selected_history: dict[Any, list[str]]
    metrics: StrategyMetrics
    verdict: Verdict
    verdict_notes: dict[str, Any]
    extra_notes: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# 数据加载与预处理
# ============================================================================

def _load_all_data(cfg: BacktestConfig) -> dict[str, Any]:
    """一次性加载 backtest 所需的全部 PG 数据."""
    from sqlalchemy import text
    from utils.db import get_contract_engine

    logger.info("Loading security_list ...")
    sec_list = load_security_list_active()

    logger.info("Loading snapshot (latest) ...")
    snapshot = load_snapshot_latest()

    logger.info("Loading money_flow range %s → %s ...", cfg.start, cfg.end)
    money_flow = load_money_flow_range(cfg.start.isoformat(), cfg.end.isoformat())

    logger.info("Loading kline range %s → %s ...", cfg.start, cfg.end)
    kline = load_kline_close_range(cfg.start.isoformat(), cfg.end.isoformat())

    logger.info("Loading listing_date inference ...")
    sql = text(
        "SELECT stock_code, MIN(trade_time::date) AS listing_date "
        "FROM cnstock_kline_day GROUP BY stock_code"
    )
    with get_contract_engine().connect() as conn:
        listing_df = pd.read_sql(sql, conn, parse_dates=["listing_date"])
    listing_dates = {
        row.stock_code: row.listing_date.date()
        for row in listing_df.itertuples()
        if pd.notna(row.listing_date)
    }
    logger.info("listing_dates: %d stocks", len(listing_dates))

    return {
        "sec_list": sec_list,
        "snapshot": snapshot,
        "money_flow": money_flow,
        "kline": kline,
        "listing_dates": listing_dates,
    }


def _build_float_share_map(snapshot: pd.DataFrame) -> dict[str, float]:
    """对 snapshot 每行反推 float_share，返回 {stock_code: share}."""
    out: dict[str, float] = {}
    anomalies = 0
    for row in snapshot.itertuples():
        r = recover_float_share({
            "float_market_cap": getattr(row, "float_market_cap", None),
            "current_price": getattr(row, "current_price", None),
            "total_volume": getattr(row, "total_volume", None),
            "turnover_rate": getattr(row, "turnover_rate", None),
        })
        if r.share is None:
            continue
        if r.anomaly:
            anomalies += 1
        out[row.stock_code] = float(r.share)
    logger.info("float_share recovered for %d stocks (anomalies=%d)", len(out), anomalies)
    return out


def _build_kline_indexed(kline: pd.DataFrame) -> dict[Any, pd.DataFrame]:
    """按 trade_date 分组 kline DataFrame，每组按 stock_code index."""
    return {td: grp.set_index("stock_code") for td, grp in kline.groupby("trade_date")}


def _build_amount_history(kline: pd.DataFrame, window: int = 20) -> dict[str, dict[Any, pd.Series]]:
    """股票 → trade_date → 含当日的最多 window 日 amount Series."""
    histories: dict[str, dict[Any, pd.Series]] = {}
    for code, grp in kline.groupby("stock_code"):
        grp_sorted = grp.sort_values("trade_date")
        amounts = grp_sorted["amount"].astype(float).reset_index(drop=True)
        dates = grp_sorted["trade_date"].reset_index(drop=True)
        per_stock: dict[Any, pd.Series] = {}
        for i in range(len(dates)):
            start_idx = max(0, i - window + 1)
            per_stock[dates.iloc[i]] = amounts.iloc[start_idx:i + 1].reset_index(drop=True)
        histories[code] = per_stock
    return histories


def _limit_flags_for_day(
    today_indexed: pd.DataFrame,
    yesterday_indexed: pd.DataFrame | None,
) -> tuple[dict[str, bool], dict[str, bool]]:
    """精确判一字涨跌停：用昨收 × ±10% 作为涨跌停价（v0.1 主板规则简化）."""
    up_flag: dict[str, bool] = {}
    down_flag: dict[str, bool] = {}
    for code in today_indexed.index:
        row = today_indexed.loc[code]
        try:
            op = float(row["open_price"])
            hi = float(row["high_price"])
            lo = float(row["low_price"])
        except (KeyError, ValueError, TypeError):
            continue
        if any(pd.isna([op, hi, lo])):
            continue
        prev_close = np.nan
        if yesterday_indexed is not None and code in yesterday_indexed.index:
            try:
                prev_close = float(yesterday_indexed.loc[code, "close_price"])
            except (KeyError, ValueError, TypeError):
                prev_close = np.nan
        if pd.isna(prev_close) or prev_close <= 0:
            # 退化：仅 O==H==L 判定停板形态
            is_flat = abs(hi - lo) < 1e-6 and abs(op - lo) < 1e-6
            up_flag[code] = is_flat
            down_flag[code] = False
            continue
        limit_up = round(prev_close * 1.1, 2)
        limit_down = round(prev_close * 0.9, 2)
        up_flag[code] = is_one_word_limit_up(op, hi, lo, limit_up)
        down_flag[code] = is_one_word_limit_down(op, hi, lo, limit_down)
    return up_flag, down_flag


# ============================================================================
# 主循环
# ============================================================================

def run_backtest(cfg: BacktestConfig) -> BacktestResult:
    """跑完整 v0.1 backtest 并出 verdict."""
    data = _load_all_data(cfg)
    sec_list = data["sec_list"]
    snapshot = data["snapshot"]
    money_flow = data["money_flow"]
    kline = data["kline"]
    listing_dates = data["listing_dates"]

    float_share_map = _build_float_share_map(snapshot)
    kline_by_date = _build_kline_indexed(kline)
    amount_history = _build_amount_history(kline)

    money_flow_by_date_code: dict[Any, set[str]] = {
        td: set(grp["stock_code"].tolist())
        for td, grp in money_flow.groupby("trade_date")
    }
    money_flow_by_date_grouped: dict[Any, pd.DataFrame] = {
        td: grp.set_index("stock_code")
        for td, grp in money_flow.groupby("trade_date")
    }

    trade_dates = sorted(set(money_flow["trade_date"]) & set(kline["trade_date"]))
    logger.info("Backtest spans %d trade dates", len(trade_dates))

    state = PortfolioState(cash=cfg.initial_capital, current_step=0)
    cost_params = CostParams()

    equity_log: list[tuple[Any, float]] = []
    daily_returns: list[float] = []
    daily_ic_a: list[float] = []
    daily_ic_b: list[float] = []
    selected_history: dict[Any, list[str]] = {}
    score_history: dict[Any, pd.DataFrame] = {}
    bm2_equity: list[float] = [1.0]

    pending_orders: list = []
    pending_orders_metadata: dict | None = None

    for step, t in enumerate(trade_dates):
        state.current_step = step

        # ---- 1. 当日开盘撮合（用上一日决定的 pending_orders）----
        if pending_orders and t in kline_by_date:
            today_indexed = kline_by_date[t]
            t1_open = today_indexed["open_price"].astype(float).to_dict()
            yesterday = trade_dates[step - 1] if step > 0 else None
            yesterday_indexed = kline_by_date.get(yesterday)
            up_flag, down_flag = _limit_flags_for_day(today_indexed, yesterday_indexed)
            cap_buy_dict = pending_orders_metadata["cap_buy"] if pending_orders_metadata else {}
            apply_orders_at_t1(
                state=state,
                orders=pending_orders,
                t1_open=t1_open,
                t1_limit_up_flag=up_flag,
                t1_limit_down_flag=down_flag,
                cap_buy_amount=cap_buy_dict,
                cost_params=cost_params,
            )

        # ---- 2. 当日 equity ----
        close_map: dict[str, float] = {}
        if t in kline_by_date:
            close_map = kline_by_date[t]["close_price"].astype(float).to_dict()
        equity = state.total_assets(close_map)
        equity_log.append((t, equity))
        if len(equity_log) >= 2:
            prev = equity_log[-2][1]
            daily_returns.append((equity - prev) / prev if prev > 0 else 0.0)
        else:
            daily_returns.append(0.0)

        # ---- 3. 构造 universe ----
        if t not in kline_by_date:
            pending_orders = []
            pending_orders_metadata = None
            bm2_equity.append(bm2_equity[-1])
            continue
        today_kline = kline_by_date[t]
        kline_today_amount = today_kline["amount"].astype(float).to_dict()
        money_flow_codes = money_flow_by_date_code.get(t, set())
        history_for_t = {
            code: amount_history.get(code, {}).get(t, pd.Series(dtype=float))
            for code in sec_list["stock_code"]
        }
        # daily_universe 接受 date
        t_as_date = t.date() if hasattr(t, "date") else t
        universe = daily_universe(
            as_of=t_as_date,
            sec_list=sec_list,
            listing_dates=listing_dates,
            kline_today_amount=kline_today_amount,
            money_flow_today_codes=money_flow_codes,
            amount_history=history_for_t,
        )

        # ---- 4. 计算因子表 + 信号 ----
        if not universe or t not in money_flow_by_date_grouped:
            pending_orders = []
            pending_orders_metadata = None
            bm2_equity.append(bm2_equity[-1])
            continue
        mf_today = money_flow_by_date_grouped[t]
        codes_with_data = [c for c in universe if c in mf_today.index]
        if not codes_with_data:
            pending_orders = []
            pending_orders_metadata = None
            bm2_equity.append(bm2_equity[-1])
            continue
        factor_df = mf_today.loc[codes_with_data].reset_index()
        factor_df["float_mv"] = [
            float_share_map.get(c, np.nan) * close_map.get(c, np.nan)
            for c in factor_df["stock_code"]
        ]
        factor_df = compute_money_flow_factors(factor_df)
        signals = compute_signals(factor_df)
        score_history[t] = signals[["stock_code", "score_A", "score_B"]].copy()

        # ---- 5. 选 Top N ----
        selected = select_top_n(signals, n=cfg.n_target)
        selected_codes = set(selected["stock_code"].tolist())
        selected_history[t] = list(selected_codes)

        # ---- 6. BM2 等权 universe 日度收益 ----
        if step > 0:
            prev_t = trade_dates[step - 1]
            prev_kline = kline_by_date.get(prev_t)
            if prev_kline is not None:
                prev_close = prev_kline["close_price"].astype(float)
                shared = set(close_map.keys()) & set(prev_close.index) & universe
                rets = []
                for c in shared:
                    pc = prev_close.get(c, np.nan)
                    cc = close_map.get(c, np.nan)
                    if pc and pc > 0 and not pd.isna(cc):
                        rets.append((cc - pc) / pc)
                bm2_ret = float(np.mean(rets)) if rets else 0.0
                # 日度等权再平衡近似 0.01% 日度摩擦
                bm2_ret -= 0.0001
                bm2_equity.append(bm2_equity[-1] * (1 + bm2_ret))
            else:
                bm2_equity.append(bm2_equity[-1])

        # ---- 7. 决定下日订单 ----
        locked = holdings_in_lock(
            state.positions, current_step=step + 1, min_days=cfg.holding_min_days
        )
        total_assets = state.total_assets(close_map)
        orders = decide_orders(
            prev_positions=state.positions,
            new_selected=selected_codes,
            holding_locked=locked,
            cash_avail=state.cash,
            total_assets=total_assets,
            n_target=cfg.n_target,
        )
        cap_buy: dict[str, float] = {}
        for c in selected_codes:
            ah = amount_history.get(c, {}).get(t)
            if ah is not None and len(ah) > 0:
                cap_buy[c] = capacity_cap_for_buy(float(ah.mean()))
            else:
                cap_buy[c] = 0.0
        pending_orders = orders
        pending_orders_metadata = {"cap_buy": cap_buy}

        if step % 20 == 0:
            logger.info("step %d/%d (%s) | universe=%d selected=%d pos=%d cash=%.0f eq=%.0f",
                        step, len(trade_dates), t, len(universe), len(selected_codes),
                        len(state.positions), state.cash, equity)

    # ============================================================================
    # 事后评估
    # ============================================================================

    equity_series = pd.Series(
        [e for _, e in equity_log],
        index=pd.DatetimeIndex([t for t, _ in equity_log]),
        name="equity",
    )
    daily_ret_series = pd.Series(daily_returns, index=equity_series.index, name="daily_return")
    bm2_equity_series = pd.Series(
        bm2_equity[:len(equity_series)],
        index=equity_series.index,
        name="bm2_equity",
    )
    bm2_daily_ret = bm2_equity_series.pct_change().fillna(0.0)

    sorted_dates = sorted(score_history.keys())
    for i, t in enumerate(sorted_dates):
        target_idx = i + cfg.forward_ic_horizon
        if target_idx >= len(sorted_dates):
            continue
        t_future = sorted_dates[target_idx]
        if t not in kline_by_date or t_future not in kline_by_date:
            continue
        close_t_series = kline_by_date[t]["close_price"].astype(float)
        close_future_series = kline_by_date[t_future]["close_price"].astype(float)
        scores = score_history[t].set_index("stock_code")
        codes = scores.index
        rets = pd.Series(
            [(close_future_series.get(c, np.nan) - close_t_series.get(c, np.nan)) / close_t_series.get(c, np.nan)
             if close_t_series.get(c, 0) and close_t_series.get(c, 0) > 0 else np.nan
             for c in codes],
            index=codes,
        )
        rel_rets = rets - rets.mean()
        ic_a = compute_ic(scores["score_A"], rel_rets)
        ic_b = compute_ic(scores["score_B"], rel_rets)
        daily_ic_a.append(ic_a)
        daily_ic_b.append(ic_b)

    ic_a_series = pd.Series(daily_ic_a, name="ic_a")
    ic_b_series = pd.Series(daily_ic_b, name="ic_b")
    ic_a_stats = compute_ic_stats(ic_a_series)
    ic_b_stats = compute_ic_stats(ic_b_series)

    cum_ret = (equity_series.iloc[-1] / equity_series.iloc[0]) - 1.0 if len(equity_series) > 1 else 0.0
    n_periods = max(1, len(equity_series) - 1)
    ann_ret = annualize_return(cum_ret, n_periods)
    bm2_cum = bm2_equity_series.iloc[-1] - 1.0 if len(bm2_equity_series) > 0 else 0.0
    bm2_ann = annualize_return(bm2_cum, n_periods)
    excess_bm2 = ann_ret - bm2_ann
    excess_bm1 = ann_ret  # BM1 不可得 → 退化为 vs 0% cash
    ir_bm2 = compute_information_ratio(daily_ret_series, bm2_daily_ret)
    ir_bm1 = float("nan")
    max_dd = compute_max_drawdown(equity_series)
    max_rel_dd = float("nan")
    sharpe = compute_sharpe(daily_ret_series)
    monthly_turnover = 0.30  # 占位；M7 中可由 fills 精确算

    daily_overlap_list = []
    for t, sel_a in selected_history.items():
        if t not in score_history:
            continue
        s = score_history[t]
        s_b_sorted = s.sort_values("score_B", ascending=False)
        sel_b = s_b_sorted["stock_code"].head(cfg.n_target).tolist()
        daily_overlap_list.append(dual_signal_overlap(sel_a, sel_b))
    overlap_ab = float(np.mean(daily_overlap_list)) if daily_overlap_list else 0.0

    metrics = StrategyMetrics(
        ic_a=ic_a_stats,
        ic_b=ic_b_stats,
        cumulative_return=cum_ret,
        annualized_return=ann_ret,
        excess_return_bm1=excess_bm1,
        excess_return_bm2=excess_bm2,
        ir_bm1=ir_bm1,
        ir_bm2=ir_bm2,
        max_drawdown=max_dd,
        max_relative_dd_bm1=max_rel_dd,
        monthly_turnover=monthly_turnover,
        sharpe=sharpe,
        overlap_ab=overlap_ab,
    )

    # 月度 BM2 超额（UNSTABLE 检测）
    if len(equity_series) > 0:
        monthly_strat = equity_series.resample("ME").last().pct_change().fillna(0.0)
        monthly_bm2 = bm2_equity_series.resample("ME").last().pct_change().fillna(0.0)
        monthly_excess_vs_bm2 = monthly_strat - monthly_bm2
    else:
        monthly_excess_vs_bm2 = pd.Series(dtype=float)

    verdict, verdict_notes = render_verdict(metrics, monthly_excess_vs_bm2=monthly_excess_vs_bm2)
    dual_status = classify_dual_signal(ic_a_stats, ic_b_stats, overlap_ab)

    extra_notes: dict[str, Any] = {
        "bm1_unavailable": True,
        "dual_signal_status": dual_status.value,
        "n_trade_dates": len(trade_dates),
        "final_equity": float(equity_series.iloc[-1]) if len(equity_series) > 0 else cfg.initial_capital,
        "cumulative_return": float(cum_ret),
        "n_ic_observations": len(daily_ic_a),
    }

    return BacktestResult(
        daily_returns=daily_ret_series,
        equity_curve=equity_series,
        daily_ic_a=ic_a_series,
        daily_ic_b=ic_b_series,
        bm2_equity=bm2_equity_series,
        selected_history=selected_history,
        metrics=metrics,
        verdict=verdict,
        verdict_notes=verdict_notes,
        extra_notes=extra_notes,
    )


# ============================================================================
# 报告输出
# ============================================================================

def write_reports(result: BacktestResult, run_dir: Path) -> None:
    """落 metrics.json + verdict.md + equity.csv + daily_ic.csv."""
    run_dir.mkdir(parents=True, exist_ok=True)
    m = result.metrics
    payload = {
        "verdict": result.verdict.value,
        "verdict_notes": result.verdict_notes,
        "extra_notes": result.extra_notes,
        "metrics": {
            "ic_a": asdict(m.ic_a),
            "ic_b": asdict(m.ic_b),
            "cumulative_return": m.cumulative_return,
            "annualized_return": m.annualized_return,
            "excess_return_bm1": m.excess_return_bm1,
            "excess_return_bm2": m.excess_return_bm2,
            "ir_bm1": m.ir_bm1,
            "ir_bm2": m.ir_bm2,
            "max_drawdown": m.max_drawdown,
            "max_relative_dd_bm1": m.max_relative_dd_bm1,
            "monthly_turnover": m.monthly_turnover,
            "sharpe": m.sharpe,
            "overlap_ab": m.overlap_ab,
        },
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    md = [
        f"# v0.1 PRELIMINARY Verdict — {result.verdict.value}",
        "",
        f"**run dir**: `{run_dir}`",
        f"**verdict notes**: {result.verdict_notes}",
        f"**extra notes**: {result.extra_notes}",
        "",
        "## Metrics",
        f"- IC_A mean={m.ic_a.mean:.4f}, t-stat={m.ic_a.t_stat:.2f}, n={m.ic_a.n}",
        f"- IC_B mean={m.ic_b.mean:.4f}, t-stat={m.ic_b.t_stat:.2f}, n={m.ic_b.n}",
        f"- cum return: {m.cumulative_return*100:.2f}%",
        f"- annualized: {m.annualized_return*100:.2f}%",
        f"- excess vs BM1 (fallback vs 0% cash): {m.excess_return_bm1*100:.2f}%",
        f"- excess vs BM2 (等权 universe): {m.excess_return_bm2*100:.2f}%",
        f"- max drawdown: {m.max_drawdown*100:.2f}%",
        f"- sharpe (扣成本): {m.sharpe:.2f}",
        f"- IR vs BM2: {m.ir_bm2:.2f}",
        f"- dual signal overlap (A vs B Top-N): {m.overlap_ab*100:.1f}%",
        "",
        "## 后续",
        "- 若 verdict ∈ {PASS-prelim} → 数据攒到 ≥18 月后 v0.1.5 复跑（前 ⅔ IS + 后 ⅓ OOS）",
        "- 若 verdict ∈ {CHURNING, NO-ALPHA, SIZE-BETA-ONLY} → 关闭策略 + 归档 lesson",
    ]
    (run_dir / "verdict.md").write_text("\n".join(md), encoding="utf-8")

    df_eq = pd.DataFrame({
        "equity": result.equity_curve,
        "daily_return": result.daily_returns,
        "bm2_equity": result.bm2_equity,
    })
    df_eq.to_csv(run_dir / "equity.csv", index_label="trade_date")

    df_ic = pd.DataFrame({"ic_a": result.daily_ic_a, "ic_b": result.daily_ic_b})
    df_ic.to_csv(run_dir / "daily_ic.csv", index_label="day_idx")


# ============================================================================
# CLI
# ============================================================================

def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="cn_a_big_money_rotation v0.1 backtest")
    parser.add_argument("--start", type=_parse_date, required=True)
    parser.add_argument("--end", type=_parse_date, required=True)
    parser.add_argument("--initial-capital", type=float, default=1e7)
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    run_id = datetime.now().strftime("v0.1_%Y%m%d_%H%M%S")
    run_dir = args.output_dir / run_id
    logger.info("Run dir: %s", run_dir)

    cfg = BacktestConfig(
        start=args.start,
        end=args.end,
        initial_capital=args.initial_capital,
        output_dir=args.output_dir,
    )
    result = run_backtest(cfg)
    write_reports(result, run_dir)
    print(f"\nVERDICT: {result.verdict.value}")
    print(f"reports: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
