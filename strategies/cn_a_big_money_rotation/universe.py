"""M2 Universe 层：每个 T 日构造 universe_T（设计 §4 五步过滤）.

5 个纯函数 + 1 个组合主入口，零 DB 依赖；caller 负责把 PG 数据装好传进来.

过滤顺序（与设计 §4 一致）：
1. is_active 基底（caller 保证 sec_list 已 filter）
2. ST/*ST/退 前缀剔除
3. 上市 ≥ 250 个交易日（次新过滤）
4. 当日停牌：kline.amount==0 或 kline 行缺失 或 money_flow 行缺失
5. 流动性：当日成交额 ≥ 5000 万 AND 过去含当日 ≤20 日均成交额 ≥ 5000 万
"""
from __future__ import annotations

from datetime import date
from typing import Iterable, Mapping

import pandas as pd


ST_PREFIXES = ("ST", "*ST", "退")
MIN_LISTING_DAYS = 250
LIQUIDITY_FLOOR_RMB = 5e7  # 5000 万元
LIQUIDITY_WINDOW = 20      # 含当日的 20 日窗口


def is_st_name(name: str | None) -> bool:
    """股票名前缀以 ST / *ST / 退 开头 → True（设计 §4 步骤 2）.

    NULL / 空 → False（不剔除；调用方应负责剔除 NULL 股名行）.
    去前导空白后比较，对 ASCII 字母不区分大小写.
    """
    if name is None:
        return False
    stripped = name.lstrip()
    if not stripped:
        return False
    upper = stripped.upper()
    return upper.startswith(ST_PREFIXES)


def passes_listing_age(
    listing_date: date | None,
    as_of: date,
    min_days: int = MIN_LISTING_DAYS,
) -> bool:
    """上市 ≥ min_days 自然日 → True（设计 §4 步骤 3）.

    listing_date NULL → False（保守剔除，避免新股进入 universe）.
    """
    if listing_date is None:
        return False
    return (as_of - listing_date).days >= min_days


def passes_liquidity(
    daily_amounts: pd.Series,
    floor: float = LIQUIDITY_FLOOR_RMB,
) -> bool:
    """流动性双条件 AND（设计 §4 步骤 5）.

    输入 ``daily_amounts``：含当日（最后一个元素）的最多 20 日成交额序列.

    通过条件：
    - 当日成交额 ≥ floor
    - 序列均值 ≥ floor（不足 20 日窗口时用现有均值，不补 NaN）
    """
    if daily_amounts is None or len(daily_amounts) == 0:
        return False
    same_day = daily_amounts.iloc[-1]
    if pd.isna(same_day) or same_day < floor:
        return False
    mean_val = daily_amounts.mean()
    if pd.isna(mean_val) or mean_val < floor:
        return False
    return True


def is_suspended(
    stock_code: str,
    kline_today_amount: Mapping[str, float],
    money_flow_today_codes: set[str],
) -> bool:
    """T 日停牌判定（设计 §4 步骤 4）.

    任一条件触发即视为停牌：
    - kline 当日无行 → 停牌
    - kline.amount == 0（或 NaN）→ 无成交，视同停牌
    - money_flow 当日无行 → 数据缺失，从 universe 剔除
    """
    amt = kline_today_amount.get(stock_code)
    if amt is None or pd.isna(amt) or amt <= 0:
        return True
    if stock_code not in money_flow_today_codes:
        return True
    return False


def daily_universe(
    as_of: date,
    sec_list: pd.DataFrame,
    listing_dates: Mapping[str, date],
    kline_today_amount: Mapping[str, float],
    money_flow_today_codes: Iterable[str],
    amount_history: Mapping[str, pd.Series],
) -> set[str]:
    """构造 T 日 universe（设计 §4 五步联合）.

    Args:
        as_of: T 日.
        sec_list: 已 ``is_active=True`` 过滤的清单；需含 ``stock_code, stock_name``.
        listing_dates: stock_code → 上市日（``cnstock_kline_day`` 的 MIN(trade_date) 近似）.
        kline_today_amount: stock_code → T 日 ``amount``（成交额）.
        money_flow_today_codes: T 日 ``cnstock_daily_money_flow`` 出现过的 stock_code 集合.
        amount_history: stock_code → 含 T 日（最后元素）的最多 20 日成交额 Series.

    Returns:
        通过所有 5 步过滤的 stock_code 集合.
    """
    money_flow_set = set(money_flow_today_codes)

    # 步骤 1：is_active 基底
    name_map = dict(zip(sec_list["stock_code"], sec_list["stock_name"]))
    candidates = set(name_map.keys())

    # 步骤 2：ST/*ST/退 前缀剔除
    candidates = {c for c in candidates if not is_st_name(name_map.get(c))}

    # 步骤 3：次新过滤
    candidates = {
        c for c in candidates
        if passes_listing_age(listing_dates.get(c), as_of)
    }

    # 步骤 4：停牌过滤
    candidates = {
        c for c in candidates
        if not is_suspended(c, kline_today_amount, money_flow_set)
    }

    # 步骤 5：流动性下限
    candidates = {
        c for c in candidates
        if passes_liquidity(amount_history.get(c, pd.Series(dtype=float)))
    }

    return candidates
