"""M1 数据层：snapshot 反推 float_share + money_flow 因子派生 + SQL reader.

对应设计 §3.3 (float_share 反推) 与 §3.4 (派生因子)。

纯函数（recover_float_share / compute_historical_float_mv /
compute_money_flow_factors）零 DB 依赖，配合 fixture 单测；SQL reader
(load_money_flow_range / load_snapshot_latest / load_kline_close_range)
通过 ``utils.db.get_contract_engine()`` 读 ginkgo_bole 库。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sqlalchemy import text

from utils.db import get_contract_engine

logger = logging.getLogger(__name__)


# ============================================================================
# §3.3  float_share 反推
# ============================================================================

ANOMALY_DEVIATION_THRESHOLD = 0.05  # 主路径与互验路径偏差超过 5% 标异常


@dataclass
class FloatShareRecovery:
    """单只股票从 snapshot 反推 float_share 的结果."""

    share_main: float | None       # 主路径：float_market_cap / current_price
    share_check: float | None      # 互验：total_volume / (turnover_rate / 100)
    deviation_pct: float | None    # |main - check| / main；缺一方时为 None
    share: float | None            # 实际采用值：main 优先，main 不可用回退 check
    anomaly: bool                  # 偏差 > 5% 或 main 与 check 都失败
    notes: dict[str, bool] = field(default_factory=dict)


def recover_float_share(row: dict[str, Any]) -> FloatShareRecovery:
    """从 snapshot 一行反推 float_share（股本数量，单位股）.

    主路径：float_share = float_market_cap / current_price
    互验：float_share = total_volume / (turnover_rate / 100)

    异常处置：
    - current_price == 0 或 NULL → 主路径不可用，回退互验；标 main_path_zero_division
    - turnover_rate == 0 或 NULL → 互验不可用，不标异常（数据稀疏）
    - 偏差 > 5% → anomaly=True
    - 主+互验都不可用 → anomaly=True, share=None
    """
    notes: dict[str, bool] = {}

    fm = row.get("float_market_cap")
    cp = row.get("current_price")
    tv = row.get("total_volume")
    tr = row.get("turnover_rate")

    # ---- 主路径 ----
    share_main: float | None
    if cp is None or float(cp) == 0.0:
        share_main = None
        notes["main_path_zero_division"] = True
    elif fm is None:
        share_main = None
        notes["main_path_missing_mv"] = True
    else:
        share_main = float(fm) / float(cp)

    # ---- 互验路径 ----
    share_check: float | None
    if tr is None or float(tr) == 0.0:
        share_check = None
    elif tv is None:
        share_check = None
    else:
        share_check = float(tv) / (float(tr) / 100.0)

    # ---- 偏差 / share / anomaly ----
    deviation: float | None = None
    if share_main is not None and share_check is not None and share_main != 0:
        deviation = abs(share_main - share_check) / abs(share_main)

    share: float | None
    anomaly = False
    if share_main is not None:
        share = share_main
    elif share_check is not None:
        share = share_check
    else:
        share = None
        anomaly = True
        notes["both_paths_unavailable"] = True

    if deviation is not None and deviation > ANOMALY_DEVIATION_THRESHOLD:
        anomaly = True
        notes["share_recover_anomaly"] = True

    return FloatShareRecovery(
        share_main=share_main,
        share_check=share_check,
        deviation_pct=deviation,
        share=share,
        anomaly=anomaly,
        notes=notes,
    )


def compute_historical_float_mv(float_share: float, close_series: pd.Series) -> pd.Series:
    """历史每日 float_mv = float_share × close_t（设计 §3.3）.

    close NULL 处直接 NaN 传播（由 universe 层后续剔除）；
    设计已明确 13 月内股本变化对截面排序近乎无影响。
    """
    return float(float_share) * close_series


# ============================================================================
# §3.4  money_flow 因子派生
# ============================================================================

# 8 个 cje 字段（金额，单位元）
_BUY_FIELDS = ["zmbtdcje", "zmbddcje", "zmbzdcje", "zmbxdcje"]   # 主买 特/大/中/小
_SELL_FIELDS = ["zmstdcje", "zmsddcje", "zmszdcje", "zmsxdcje"]  # 主卖 特/大/中/小
ALL_CJE_FIELDS = _BUY_FIELDS + _SELL_FIELDS

# 大单族 = 特大 + 大；用于主信号
_BIG_BUY_FIELDS = ["zmbtdcje", "zmbddcje"]
_BIG_SELL_FIELDS = ["zmstdcje", "zmsddcje"]


def compute_money_flow_factors(mf: pd.DataFrame) -> pd.DataFrame:
    """从 cnstock_daily_money_flow 行计算派生因子（设计 §3.4）.

    输入列要求：8 个 ``zmb*cje`` / ``zms*cje`` 字段（NULL 视为 0，显式 COALESCE）.
    可选输入：``float_mv`` 列；若存在则一并输出 ``big_net_per_mv``.

    输出附加列：
    - big_net_inflow  = Σ 大单族主买 cje − Σ 大单族主卖 cje（元；正 = 主买净流入）
    - total_amount    = Σ 全档主买 cje + Σ 全档主卖 cje（元）
    - big_net_per_mv  = big_net_inflow / float_mv（仅 float_mv 列存在时输出）
    - big_net_ratio   = big_net_inflow / total_amount（total_amount > 0 时输出，否则 NaN）

    NULL/缺失字段统一按 0 处理，避免在缺特定档时丢整行（COALESCE 语义）.
    """
    out = mf.copy()

    # 缺失列补 0，已有列 NULL 也填 0
    for col in ALL_CJE_FIELDS:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    big_buy = out[_BIG_BUY_FIELDS].sum(axis=1)
    big_sell = out[_BIG_SELL_FIELDS].sum(axis=1)
    out["big_net_inflow"] = big_buy - big_sell

    total = out[ALL_CJE_FIELDS].sum(axis=1)
    out["total_amount"] = total

    # ratio：total == 0 时为 NaN（不入截面排序）
    out["big_net_ratio"] = np.where(
        total > 0,
        out["big_net_inflow"] / total.replace(0, np.nan),
        np.nan,
    )

    if "float_mv" in out.columns:
        float_mv = pd.to_numeric(out["float_mv"], errors="coerce")
        out["big_net_per_mv"] = np.where(
            float_mv > 0,
            out["big_net_inflow"] / float_mv.replace(0, np.nan),
            np.nan,
        )

    return out


# ============================================================================
# SQL readers（实读 ginkgo_bole 库；走 get_contract_engine）
# ============================================================================

def _stock_filter_clause(
    stock_codes: Iterable[str] | None,
    param_prefix: str = "s",
) -> tuple[str, dict]:
    """生成可选 `stock_code IN (:s0, :s1, ...)` 子句."""
    if not stock_codes:
        return "", {}
    codes = list(stock_codes)
    placeholders = ", ".join(f":{param_prefix}{i}" for i in range(len(codes)))
    params = {f"{param_prefix}{i}": c for i, c in enumerate(codes)}
    return f" AND stock_code IN ({placeholders})", params


def load_money_flow_range(
    start: str,
    end: str,
    stock_codes: Iterable[str] | None = None,
) -> pd.DataFrame:
    """读 cnstock_daily_money_flow 在 [start, end] 闭区间内的 8 个 cje 字段."""
    clause, params = _stock_filter_clause(stock_codes)
    params.update({"start": start, "end": end})
    cols = ", ".join(ALL_CJE_FIELDS)
    sql = text(
        f"SELECT trade_date, stock_code, {cols} "
        f"FROM cnstock_daily_money_flow "
        f"WHERE trade_date >= :start AND trade_date <= :end"
        f"{clause} "
        f"ORDER BY trade_date ASC, stock_code ASC"
    )
    with get_contract_engine().connect() as conn:
        df = pd.read_sql(sql, conn, params=params, parse_dates=["trade_date"])
    return df.reset_index(drop=True)


def load_snapshot_latest(stock_codes: Iterable[str] | None = None) -> pd.DataFrame:
    """读 cnstock_daily_snapshot 最新一天（设计上整表只 1 天）的反推所需字段."""
    clause, params = _stock_filter_clause(stock_codes)
    sql = text(
        f"SELECT stock_code, trade_date, "
        f"       float_market_cap, current_price, total_volume, turnover_rate "
        f"FROM cnstock_daily_snapshot "
        f"WHERE trade_date = (SELECT MAX(trade_date) FROM cnstock_daily_snapshot)"
        f"{clause} "
        f"ORDER BY stock_code ASC"
    )
    with get_contract_engine().connect() as conn:
        df = pd.read_sql(sql, conn, params=params, parse_dates=["trade_date"])
    return df.reset_index(drop=True)


def load_kline_close_range(
    start: str,
    end: str,
    stock_codes: Iterable[str] | None = None,
) -> pd.DataFrame:
    """读 cnstock_kline_day 在 [start, end] 闭区间内的 OHLC + 成交额.

    注意：列名是 ``open_price/high_price/low_price/close_price``（已前复权）.
    """
    clause, params = _stock_filter_clause(stock_codes)
    params.update({"start": start, "end": end})
    sql = text(
        f"SELECT trade_time::date AS trade_date, stock_code, "
        f"       open_price, high_price, low_price, close_price, "
        f"       volume, amount, suspension "
        f"FROM cnstock_kline_day "
        f"WHERE trade_time::date >= :start AND trade_time::date <= :end"
        f"{clause} "
        f"ORDER BY trade_time ASC, stock_code ASC"
    )
    with get_contract_engine().connect() as conn:
        df = pd.read_sql(sql, conn, params=params, parse_dates=["trade_date"])
    return df.reset_index(drop=True)


def load_security_list_active() -> pd.DataFrame:
    """读 cnstock_security_list 当前 is_active 行（含股票名，用于 ST 过滤）."""
    sql = text(
        "SELECT stock_code, exchange_code, stock_name, is_active, updated_at "
        "FROM cnstock_security_list "
        "WHERE is_active = TRUE "
        "ORDER BY stock_code ASC"
    )
    with get_contract_engine().connect() as conn:
        df = pd.read_sql(sql, conn)
    return df.reset_index(drop=True)
