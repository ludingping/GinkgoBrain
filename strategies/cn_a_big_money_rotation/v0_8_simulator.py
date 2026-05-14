"""v0.8 GBDT realistic backtest 模块 (mongo panel 路径, 2026-05-14 加入).

vs `backtest.py` (cnstock postgres + 事件驱动):
- 这里是 panel-vectorized 实现 (mongo data 路径)
- 现实约束:
  * T+1 持仓 (T 信号 mask.shift(1))
  * 持有期 ≥3 日 (forward-fill over holding_min_days+1 天的 rolling max)
  * 一字涨跌停 approximation (|quote_change_T+1| >= 9.95 不能买入)
  * friction 日度 % 扣除
- 跳过 (mongo 数据局限):
  * Capacity 约束 (mongo 无 volume)
  * T+1 开盘价 (mongo 只有 close, 用 close approx)
  * 精确一字涨跌停 (mongo 无 high/low, 用 |quote_change| 近似)

设计意图: 是 v0.8 spike (简化) → production-grade 的桥梁; 注意 ceiling
仍受 mongo 字段限制. 真生产部署需 join cnstock kline 获取 volume/high/low.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class V0_8_BacktestConfig:
    bot_pct: float = 0.05                  # 剔除 pred 最低 5% (avoid_bot mode)
    top_n: int | None = None               # 若 ≥0 则选 Top n 模式 (优先 bot_pct)
    friction: float = 0.0001               # 日度 friction (单边)
    holding_min_days: int = 3              # 买入后锁定天数
    limit_up_thr: float = 9.95             # |quote_change| ≥ 此值视为一字涨跌停 (approx, mongo only)
    apply_limit_up_filter: bool = True
    apply_holding_lock: bool = True
    # 以下需 cnstock kline_panel 才生效:
    apply_precise_limit_up: bool = False   # True 时用 (open==high==low) 替换 |qc| approx (需 kline)
    min_avg_amount_20d: float = 0.0        # liquidity filter; >0 时排除 amount_20d < 此值的 stock (需 kline)


@dataclass
class V0_8_BacktestResult:
    daily_returns: pd.Series
    equity_curve: pd.Series
    cum_return: float
    annualized: float
    sharpe: float
    max_drawdown: float
    avg_turnover: float
    n_days: int
    first_date: str
    last_date: str
    bm2_cum: float
    bm2_ann: float
    excess: float
    excess_ann: float
    config: dict[str, Any] = field(default_factory=dict)


def _build_target_mask(
    pred_pivot: pd.DataFrame,
    universe_mask: pd.DataFrame,
    bot_pct: float,
    top_n: int | None,
) -> pd.DataFrame:
    valid_pred = pred_pivot.where(universe_mask)
    if top_n is not None and top_n > 0:
        rank = valid_pred.rank(axis=1, ascending=False, method="first")
        return (rank <= top_n).fillna(False) & universe_mask
    else:
        bot_thr = valid_pred.quantile(bot_pct, axis=1)
        is_bot = valid_pred.le(bot_thr, axis=0).fillna(False)
        return universe_mask & ~is_bot


def simulate_v0_8(
    pred_pivot: pd.DataFrame,
    panel: dict,
    cfg: V0_8_BacktestConfig,
    kline_panel: dict | None = None,
) -> V0_8_BacktestResult:
    """Vectorized realistic backtest with v0.8 GBDT predictions.

    Args:
        pred_pivot: GBDT 预测 rank, index=date, cols=stock_id, values=pred
        panel: mongo panel from load_mongo_money_flow_panel
        cfg: 配置
        kline_panel: 可选 cnstock kline panel from load_cnstock_kline_panel
                     (含 open/high/low/close/amount), 用于:
                     - 精确 limit_up (open==high==low) 替换 |qc| approx
                     - liquidity filter (avg_amount_20d 下限)
    """
    price = panel["price"]
    qc = panel["quote_change"]

    pred_pivot = pred_pivot.reindex(index=price.index, columns=price.columns)
    universe_mask = price.notna() & (price > 0)
    daily_ret = price.pct_change()

    # liquidity filter (need kline)
    if kline_panel is not None and cfg.min_avg_amount_20d > 0:
        amount = kline_panel["amount"].reindex(index=price.index, columns=price.columns)
        avg_amount_20d = amount.rolling(20, min_periods=10).mean()
        liquid_mask = avg_amount_20d >= cfg.min_avg_amount_20d
        universe_mask = universe_mask & liquid_mask.fillna(False)
        logger.info("Liquidity filter active: avg_amount_20d >= %.0f yuan", cfg.min_avg_amount_20d)

    target_mask_t = _build_target_mask(pred_pivot, universe_mask, cfg.bot_pct, cfg.top_n)
    intended_t1 = target_mask_t.shift(1).fillna(False).astype(bool)

    if cfg.apply_limit_up_filter:
        # 默认: |quote_change| ≥ thr approx
        limit_at_t1 = qc.abs().ge(cfg.limit_up_thr).fillna(False)
        # 升级: 用 cnstock kline (open==high==low) AND qc>=thr 精确一字涨跌停
        if kline_panel is not None and cfg.apply_precise_limit_up:
            opn = kline_panel["open"].reindex(index=price.index, columns=price.columns)
            high = kline_panel["high"].reindex(index=price.index, columns=price.columns)
            low = kline_panel["low"].reindex(index=price.index, columns=price.columns)
            # 一字: open == high == low; qc 用 mongo 的 (兼容)
            eps = (opn.abs().clip(lower=1.0) * 1e-6).fillna(1.0)
            is_one_word = (((opn - high).abs() <= eps) &
                            ((high - low).abs() <= eps) &
                            (qc.abs() >= cfg.limit_up_thr)).fillna(False)
            limit_at_t1 = is_one_word
            logger.info("Precise limit_up active: (open==high==low) AND |qc|>=%.2f", cfg.limit_up_thr)
        prev_intended = intended_t1.shift(1).fillna(False)
        is_new_entry = intended_t1 & ~prev_intended
        blocked = is_new_entry & limit_at_t1
        intended_t1 = intended_t1 & ~blocked

    if cfg.apply_holding_lock and cfg.holding_min_days > 0:
        lock_window = cfg.holding_min_days + 1
        intended_t1 = intended_t1.rolling(lock_window, min_periods=1).max().fillna(False).astype(bool)

    universe_t1 = universe_mask.shift(1).fillna(False).astype(bool)
    final_mask = intended_t1 & universe_t1

    masked = daily_ret.where(final_mask)
    daily_port = masked.mean(axis=1, skipna=True) - cfg.friction
    daily_port = daily_port.fillna(0.0)

    has_pred = pred_pivot.notna().any(axis=1)
    eq_full = (1 + daily_port).cumprod()
    eq_oos = eq_full[has_pred]
    dp_oos = daily_port[has_pred]

    if len(eq_oos) < 2:
        return V0_8_BacktestResult(
            daily_returns=dp_oos, equity_curve=eq_oos, cum_return=0.0, annualized=0.0,
            sharpe=0.0, max_drawdown=0.0, avg_turnover=0.0, n_days=0,
            first_date="", last_date="", bm2_cum=0.0, bm2_ann=0.0,
            excess=0.0, excess_ann=0.0, config=cfg.__dict__)

    eq_norm = eq_oos / eq_oos.iloc[0]
    cum = float(eq_norm.iloc[-1] - 1)
    n_steps = max(1, len(eq_oos) - 1)
    ann = float(eq_norm.iloc[-1] ** (252 / n_steps) - 1) if eq_norm.iloc[-1] > 0 else float("nan")
    sh = float(dp_oos.mean() / dp_oos.std() * np.sqrt(252)) if dp_oos.std() > 0 else 0.0
    dd = float((eq_norm / eq_norm.cummax() - 1).min())

    mask_int = final_mask.astype(int)
    daily_turn = (mask_int.diff().abs().sum(axis=1) /
                  mask_int.sum(axis=1).replace(0, np.nan)).fillna(0.0)
    avg_turn = float(daily_turn[has_pred].mean())

    bm2_daily = daily_ret.where(universe_t1).mean(axis=1, skipna=True) - cfg.friction
    bm2_daily = bm2_daily.fillna(0.0)
    bm2_eq_full = (1 + bm2_daily).cumprod()
    bm2_eq_oos = bm2_eq_full[has_pred]
    bm2_eq_norm = bm2_eq_oos / bm2_eq_oos.iloc[0]
    bm2_cum = float(bm2_eq_norm.iloc[-1] - 1)
    bm2_ann = float(bm2_eq_norm.iloc[-1] ** (252 / n_steps) - 1) if bm2_eq_norm.iloc[-1] > 0 else float("nan")

    return V0_8_BacktestResult(
        daily_returns=dp_oos, equity_curve=eq_norm, cum_return=cum, annualized=ann,
        sharpe=sh, max_drawdown=dd, avg_turnover=avg_turn, n_days=len(eq_oos),
        first_date=str(eq_oos.index[0].date()), last_date=str(eq_oos.index[-1].date()),
        bm2_cum=bm2_cum, bm2_ann=bm2_ann,
        excess=cum - bm2_cum, excess_ann=ann - bm2_ann, config=cfg.__dict__)


def predictions_to_pivot(predictions: pd.Series | pd.DataFrame) -> pd.DataFrame:
    """Series with MultiIndex (date, stock_id) OR DataFrame [date, stock_id, pred]
    → pivot (index=date, cols=stock_id, values=pred)."""
    if isinstance(predictions, pd.Series):
        if isinstance(predictions.index, pd.MultiIndex):
            return predictions.unstack(level="stock_id")
        raise ValueError("Series must have MultiIndex (date, stock_id)")
    if {"date", "stock_id", "pred"}.issubset(predictions.columns):
        return predictions.pivot_table(index="date", columns="stock_id",
                                       values="pred", aggfunc="first")
    raise ValueError("DataFrame must have columns [date, stock_id, pred]")
