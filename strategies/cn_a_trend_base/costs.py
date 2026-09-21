"""§1.2 成本与可交易性（所有 A 股回测共用一套）。"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

COMMISSION = 0.00025            # 单边佣金
SLIPPAGE = 0.001                # 单边滑点
STAMP_TAX_BEFORE = 0.001        # 卖出印花税，2023-08-27 及之前
STAMP_TAX_AFTER = 0.0005        # 2023-08-28 起
STAMP_TAX_SWITCH = date(2023, 8, 28)
LIMIT_MAIN = 0.098              # 主板涨跌停判定阈值（10% 留余量）
LIMIT_GEM_STAR = 0.198          # 创业板 300 / 科创板 688（20%）


def stamp_tax(d: date | pd.Timestamp) -> float:
    d = pd.Timestamp(d).date()
    return STAMP_TAX_AFTER if d >= STAMP_TAX_SWITCH else STAMP_TAX_BEFORE


def buy_cost() -> float:
    return COMMISSION + SLIPPAGE


def sell_cost(d: date | pd.Timestamp) -> float:
    return COMMISSION + SLIPPAGE + stamp_tax(d)


def limit_threshold(code: str) -> float:
    return LIMIT_GEM_STAR if code.startswith(("300", "301", "688", "689")) else LIMIT_MAIN


def limit_up_open(open_px: pd.DataFrame, pre_close: pd.DataFrame) -> pd.DataFrame:
    """T 日开盘已涨停（不可买入）。列 = 股票代码，阈值按板块。"""
    thr = pd.Series({c: limit_threshold(c) for c in open_px.columns})
    return (open_px / pre_close - 1.0) >= thr


def limit_down_open(open_px: pd.DataFrame, pre_close: pd.DataFrame) -> pd.DataFrame:
    """T 日开盘已跌停（不可卖出）。"""
    thr = pd.Series({c: limit_threshold(c) for c in open_px.columns})
    return (open_px / pre_close - 1.0) <= -thr


def turnover_cost(turnover: pd.Series) -> pd.Series:
    """日换手（Σ|Δw|，买卖各半）→ 当日成本比例。"""
    sell = pd.Series([sell_cost(d) for d in turnover.index], index=turnover.index)
    return turnover / 2.0 * (buy_cost() + sell)


__all__ = ["COMMISSION", "SLIPPAGE", "STAMP_TAX_SWITCH", "stamp_tax", "buy_cost", "sell_cost",
           "limit_threshold", "limit_up_open", "limit_down_open", "turnover_cost", "np"]
