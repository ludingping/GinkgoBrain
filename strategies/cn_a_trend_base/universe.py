"""§2.3 宇宙掩码（向量化；输入为 日期 × 代码 的宽表）。

规则：非 ST（当前名称，含前视偏差，见设计 §2.2）、首根日线距今 ≥ MIN_LISTING_DAYS 个交易日、
当日成交额与 20 日均成交额 ≥ LIQUIDITY_FLOOR、当日未停牌且有成交、当日有收盘价。
北交所不在数据里（代码前缀仅 0/3/6）。
"""
from __future__ import annotations

import pandas as pd

ST_PREFIXES = ("ST", "*ST", "退")
MIN_LISTING_DAYS = 250          # 交易日
LIQUIDITY_BOTTOM_PCT = 0.10     # 剔除 20 日均成交额排名后 10%（近似中证全指的流动性剔除）
LIQUIDITY_WINDOW = 20
SEASONED_LOOKBACK = 20          # 窗口前 20 个交易日内出现过的代码视为已过次新期


def st_codes(names: pd.Series) -> set[str]:
    """stock_code → stock_name 的 Series；返回当前名称为 ST/*ST/退 前缀的代码集合。"""
    n = names.fillna("").astype(str).str.lstrip().str.upper()
    return set(n[n.str.startswith(ST_PREFIXES)].index)


def listing_age(close: pd.DataFrame, *, seasoned_at_start: int | None = MIN_LISTING_DAYS,
                lookback: int = SEASONED_LOOKBACK) -> pd.DataFrame:
    """每个代码自首根日线起经过的交易日数（首日 = 0；首日前 NaN）。

    数据窗口前 `lookback` 个交易日内就有行情的代码，真实上市日在窗口之前、不可知：按
    `seasoned_at_start` 个交易日补齐（默认视为已过次新期），否则整个 2016 年会被误判为次新。
    各股在采集里的首行日期并不对齐，所以看前 20 日而不是第一天。"""
    has = close.notna()
    first = has.cumsum().gt(0)
    age = (first.cumsum() - 1).astype("float64")
    if seasoned_at_start and len(close):
        present = has.iloc[:lookback].any()
        age.loc[:, present[present].index] += float(seasoned_at_start)
    return age.where(first)


def universe_mask(close: pd.DataFrame, amount: pd.DataFrame, suspension: pd.DataFrame,
                  st: set[str] | None = None, *, min_listing_days: int = MIN_LISTING_DAYS,
                  liquidity_bottom_pct: float = LIQUIDITY_BOTTOM_PCT, window: int = LIQUIDITY_WINDOW) -> pd.DataFrame:
    """T 日宇宙（bool 宽表）。注意：这是"T 日收盘后可知"的成员资格，持仓用 T−1 的掩码。

    流动性：20 日均成交额在当日所有有成交的代码里排名后 `liquidity_bottom_pct` 的剔除（相对规则，
    不随市场成交量水平漂移；固定金额门槛会让 2017 年宇宙只剩 1/3）。"""
    amt = amount.fillna(0.0)
    aged = listing_age(close).fillna(-1) >= min_listing_days
    trading = (suspension.fillna(1) == 0) & (amt > 0) & close.notna()
    avg_amt = amt.rolling(window, min_periods=1).mean().where(trading & aged)
    cutoff = avg_amt.quantile(liquidity_bottom_pct, axis=1)
    liquid = avg_amt.ge(cutoff, axis=0)
    mask = liquid & aged & trading
    if st:
        drop = [c for c in mask.columns if c in st]
        mask.loc[:, drop] = False
    return mask


def yearly_stats(mask: pd.DataFrame, suspension: pd.DataFrame, listed: pd.DataFrame,
                 limit_up: pd.DataFrame | None = None) -> pd.DataFrame:
    """每年：宇宙均值、停牌率（分母 = 当日有行情记录的代码）、（宇宙内）开盘涨停不可买占比。"""
    idx = mask.index.year
    susp = ((suspension == 1) & listed).sum(axis=1) / listed.sum(axis=1).replace(0, float("nan"))
    out = pd.DataFrame({
        "universe_mean": mask.sum(axis=1).groupby(idx).mean().round(0),
        "suspended_rate": susp.groupby(idx).mean().round(4),
    })
    if limit_up is not None:
        lu = (limit_up & mask).sum(axis=1) / mask.sum(axis=1).replace(0, float("nan"))
        out["limit_up_in_universe"] = lu.groupby(idx).mean().round(4)
    return out
