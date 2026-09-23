"""§1.1 / §2.4 / §4 L0：申万行业指数面板（宽表）、BM-S（31 个一级行业等权、每月首个交易日再平衡、T+1 开盘、§1.3 成本）、
逐年覆盖统计、parquet 缓存读写。DB 读取放在 scripts/。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .rotation import EvalContext, Plan, run_executor

# §1.2 三段（登记时定下，不改）
SEGMENTS: dict[str, tuple[str, str | None]] = {
    "pre_val": ("2005-01-04", "2016-12-30"),
    "val": ("2017-01-03", "2022-12-30"),
    "test": ("2023-01-03", None),
}
BACKTEST_START = "2005-01-04"
COVERAGE_MIN = 0.95                 # L0：31 个行业每年覆盖 ≥ 95%
PANEL_FIELDS = ("close", "open", "high", "low", "amount")
# 申万 2021 版一级行业 31 个（cnstock_sw_industry 缺席时的兜底；有表以表为准）
SW_L1_CODES = ["801010", "801030", "801040", "801050", "801080", "801110", "801120", "801130", "801140", "801150",
               "801160", "801170", "801180", "801200", "801210", "801230", "801710", "801720", "801730", "801740",
               "801750", "801760", "801770", "801780", "801790", "801880", "801890", "801950", "801960", "801970",
               "801980"]


def to_wide(long: pd.DataFrame, col: str) -> pd.DataFrame:
    """长表（trade_date, index_code, …）→ 日期 × 代码 宽表，按日期排序；缺行保持 NaN（不填零）。"""
    wide = long.pivot(index="trade_date", columns="index_code", values=col).sort_index()
    wide.index = pd.DatetimeIndex(wide.index, name="trade_date")
    return wide.astype("float64")


def validate_panel(close: pd.DataFrame) -> None:
    """L0 "缺失报错不填零"：收盘价 ≤ 0 视为坏数据，报错并列出代码与日期。"""
    bad = (close <= 0).stack()
    bad = bad[bad]
    if len(bad):
        head = ", ".join(f"{c}@{d.date()}" for d, c in bad.index[:10])
        raise ValueError(f"panel has {len(bad)} non-positive close values, e.g. {head}")


def first_dates(close: pd.DataFrame) -> pd.Series:
    """每个代码的首个有效交易日。"""
    return close.apply(lambda s: s.first_valid_index())


def coverage_by_year(close: pd.DataFrame, codes: list[str]) -> pd.DataFrame:
    """逐年：候选数、日均可用数、最少可用数、覆盖率、首日之后的缺口天数（首日前不算缺失）。"""
    px = close.reindex(columns=codes)
    avail = px.notna()
    started = avail.cummax()                          # 首个有效日之后视为"应有数据"
    gaps = (started & ~avail).groupby(px.index.year).sum().sum(axis=1)
    by = avail.sum(axis=1).groupby(px.index.year)
    out = pd.DataFrame({"n_codes": len(codes), "mean_available": by.mean(), "min_available": by.min(), "gap_days": gaps})
    out["coverage"] = out["mean_available"] / out["n_codes"]
    out["ok"] = out["coverage"] >= COVERAGE_MIN
    out.index.name = "year"
    return out


def gap_dates(close: pd.DataFrame, codes: list[str], frac: float = 0.5) -> pd.Series:
    """L0 诊断：已开始有数据的行业中缺行占比 > frac 的日期（多为入库缺行的真实交易日）→ Series(日期 → 缺行数)。"""
    px = close.reindex(columns=codes)
    avail = px.notna(); started = avail.cummax()
    missing = (started & ~avail).sum(axis=1)
    return missing[missing > frac * started.sum(axis=1)]


def month_first_eval_days(index: pd.DatetimeIndex) -> pd.Series:
    """评估日 = 每月最后一个交易日（T 收盘决策 → 下月首个交易日 T+1 开盘成交）；首日也评估以便建仓。"""
    period = index.to_period("M")
    last_of_month = pd.Series(period != np.roll(period, -1), index=index)
    last_of_month.iloc[-1] = False                    # 样本末日之后无 T+1
    return last_of_month | pd.Series(index == index[0], index=index)


def equal_weight_selector(ctx: EvalContext) -> Plan:
    """BM-S：全部可用行业等权（不再可用的清零）。"""
    avail = np.flatnonzero(ctx.available)
    plan = {ctx.codes[i]: 1.0 / len(avail) for i in avail} if len(avail) else {}
    return {**plan, **{ctx.codes[i]: 0.0 for i in np.flatnonzero(ctx.held & ~ctx.available)}}


def bm_s(close: pd.DataFrame, open_px: pd.DataFrame, codes: list[str], start: str | None = None) -> pd.DataFrame:
    """§1.1 BM-S：codes 等权、每月首个交易日再平衡、T+1 开盘、§1.3 成本；返回 daily + idx_net（start 起建仓）。"""
    c = close.reindex(columns=codes); o = open_px.reindex(columns=codes)
    ev = month_first_eval_days(c.index)
    if start is not None:
        s = pd.Timestamp(start)
        first = c.index[c.index.searchsorted(s)]
        ev = (ev & (c.index >= s)) | (c.index == first)
    out = run_executor(c, o, equal_weight_selector, every=ev).daily.copy()
    out["idx_net"] = (1.0 + out["ret_net"]).cumprod()
    return out


def sw_levels(industry: pd.DataFrame | None, all_codes: list[str]) -> tuple[list[str], list[str]]:
    """(一级代码, 二级代码)。有 cnstock_sw_industry 以其 level 为准；否则一级 = SW_L1_CODES ∩ 面板，其余为二级。"""
    if industry is not None and len(industry):
        lv = industry.set_index("index_code")["level"].astype(int)
        return [c for c in all_codes if lv.get(c) == 1], [c for c in all_codes if lv.get(c) == 2]
    return [c for c in all_codes if c in SW_L1_CODES], [c for c in all_codes if c not in SW_L1_CODES]


def save_panel(out: Path, wide: dict[str, pd.DataFrame]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for k, v in wide.items():
        v.to_parquet(out / f"{k}.parquet")


def load_panel(out: Path, fields: tuple[str, ...] = PANEL_FIELDS) -> dict[str, pd.DataFrame]:
    return {k: pd.read_parquet(out / f"{k}.parquet") for k in fields}


__all__ = ["SEGMENTS", "BACKTEST_START", "COVERAGE_MIN", "PANEL_FIELDS", "SW_L1_CODES", "to_wide", "validate_panel",
           "first_dates", "coverage_by_year", "gap_dates", "month_first_eval_days", "equal_weight_selector", "bm_s", "sw_levels",
           "save_panel", "load_panel"]
