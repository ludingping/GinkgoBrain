"""M3 信号层：双信号 + 截面排序 + tie-break + Top20（设计 §3.5, §5）.

设计原则：
- 主决策信号 ``score_A = big_net_per_mv``；同时输出诊断 ``score_B = big_net_ratio``
- 截面排序按 score **降序**（大者 rank=1），tie-break 用 ``stock_code`` 字典序升序
- ``selected_T`` 只由 ``rank_A`` 决定，``score_B / rank_B`` 仅供 §3.5 双信号交叉验证
- 纯函数，不接 DB，对调用顺序不敏感（同输入恒同输出）
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


TARGET_TOP_N = 20


def rank_cross_section(
    scores: pd.Series,
    stock_codes: pd.Series,
    ascending: bool = False,
) -> pd.Series:
    """对截面 ``scores`` 排名，tie-break 用 stock_code 字典序升序.

    Args:
        scores: 长度 = N 的 score 序列（可含 NaN）.
        stock_codes: 同长 stock_code 序列；与 scores 按 position 对齐.
        ascending: False 表示大者 rank=1（默认，配合 score 越大越好）；
                   True 表示小者 rank=1.

    Returns:
        长度 = N 的 ``int`` rank Series（1-based），index 与 scores 一致；
        NaN score → rank 排到末尾，便于 Top-N 截断自然丢弃.
    """
    if len(scores) != len(stock_codes):
        raise ValueError("scores 与 stock_codes 长度不一致")

    df = pd.DataFrame({
        "_score": scores.reset_index(drop=True),
        "_code": stock_codes.reset_index(drop=True),
    })
    n = len(df)

    # NaN 排到末尾（无论 ascending 与否）
    nan_mask = df["_score"].isna()
    valid = df[~nan_mask].copy()
    invalid = df[nan_mask].copy()

    # 主排序键：score；次排序键：stock_code 字典序升序
    valid = valid.sort_values(
        by=["_score", "_code"],
        ascending=[ascending, True],
        kind="mergesort",  # 稳定排序
    )

    # 重组：valid 在前，NaN 在后
    ordered = pd.concat([valid, invalid], ignore_index=False)

    # 重建 rank：position 1..N（基于 ordered 的 index）
    rank = pd.Series(np.arange(1, n + 1), index=ordered.index, dtype=int)
    # 回到原始 scores 顺序
    rank = rank.sort_index()
    rank.index = scores.index
    return rank


def compute_signals(factor_df: pd.DataFrame) -> pd.DataFrame:
    """计算 score_A / rank_A / score_B / rank_B 四列（设计 §3.5）.

    输入 ``factor_df`` 要求列：``stock_code, big_net_inflow, total_amount, float_mv``.
    （来自 data.compute_money_flow_factors 的派生输出.）

    输出附加列：
    - ``score_A``: big_net_inflow / float_mv（主决策信号）
    - ``score_B``: big_net_inflow / total_amount（诊断信号）
    - ``rank_A``: score_A 截面降序 rank（tie-break: stock_code 字典序）
    - ``rank_B``: score_B 截面降序 rank
    """
    out = factor_df.copy()

    out["score_A"] = np.where(
        out["float_mv"] > 0,
        out["big_net_inflow"] / out["float_mv"].replace(0, np.nan),
        np.nan,
    )
    out["score_B"] = np.where(
        out["total_amount"] > 0,
        out["big_net_inflow"] / out["total_amount"].replace(0, np.nan),
        np.nan,
    )

    out["rank_A"] = rank_cross_section(out["score_A"], out["stock_code"], ascending=False)
    out["rank_B"] = rank_cross_section(out["score_B"], out["stock_code"], ascending=False)

    return out


def select_top_n(
    signal_df: pd.DataFrame,
    n: int = TARGET_TOP_N,
    rank_col: str = "rank_A",
    score_col: str = "score_A",
) -> pd.DataFrame:
    """返回 rank_col ≤ n 的行（设计 §5）.

    universe 不足 n 时返回所有有效（score 非 NaN）行；
    NaN score 在 rank_cross_section 中已排到末尾，不会被选中.

    Args:
        signal_df: 含 ``rank_col``, ``score_col``, ``stock_code`` 的 DataFrame.
        n: 目标 Top-N.
        rank_col: 用于截取的 rank 列名（默认主决策信号 rank_A）.
        score_col: 配套 score 列名；用于"仅 NaN score 行才被排除"判定.

    Returns:
        长度 ≤ n 的 DataFrame，按 rank_col 升序排好.
    """
    valid_count = signal_df[score_col].notna().sum()
    effective_n = min(n, int(valid_count))
    out = signal_df[signal_df[rank_col] <= effective_n].copy()
    return out.sort_values(by=rank_col, kind="mergesort").reset_index(drop=True)


# ============================================================================
# v0.2 panel signal（持续吸筹 + 价格温和；2026-05-13 反向 bug 修复后验证有 alpha）
# ============================================================================

V0_2_N_WINDOW = 10
V0_2_POSITIVE_DAYS_MIN = 7
V0_2_PRICE_RANGE: tuple[float, float] = (-0.05, 0.08)


def compute_v0_2_panel(
    inflow_pivot: pd.DataFrame,
    amount_pivot: pd.DataFrame,
    close_pivot: pd.DataFrame,
    n_window: int = V0_2_N_WINDOW,
    positive_days_min: int = V0_2_POSITIVE_DAYS_MIN,
    price_range: tuple[float, float] = V0_2_PRICE_RANGE,
) -> dict[str, pd.DataFrame]:
    """计算 v0.2 panel 信号（date × stock_code 矩阵形式）.

    Args:
        inflow_pivot:  (trade_date × stock_code) 矩阵；每格 = 当日 big_net_inflow.
        amount_pivot:  (trade_date × stock_code) 矩阵；每格 = 当日 total_amount.
        close_pivot:   (trade_date × stock_code) 矩阵；每格 = 当日 close_price.
        n_window: 滚动窗口长度（默认 10 日）.
        positive_days_min: C1 持续性门槛（窗口内 ≥ 多少日大单净流入为正）.
        price_range: C3 价格温和区间 (low, high)，对应"过去 n 日累计涨幅".

    Returns:
        dict with keys:
          - score: 累计大单净流入 / 累计总成交额（无过滤）
          - score_filtered: 应用 C1 + C3 后保留的 score, 不通过的为 NaN
          - positive_days, cum_inflow, cum_amount, price_change_n: 中间量
          - mask_c1, mask_c3, final_mask: bool 矩阵
    """
    cum_inflow = inflow_pivot.rolling(window=n_window, min_periods=n_window).sum()
    cum_amount = amount_pivot.rolling(window=n_window, min_periods=n_window).sum()
    score = cum_inflow / cum_amount.replace(0, np.nan)
    positive_days = (inflow_pivot > 0).rolling(window=n_window, min_periods=n_window).sum()
    price_change_n = close_pivot / close_pivot.shift(n_window - 1) - 1.0

    mask_c1 = positive_days >= positive_days_min
    mask_c3 = (price_change_n >= price_range[0]) & (price_change_n <= price_range[1])
    mask_score = score.notna()
    final_mask = mask_c1 & mask_c3 & mask_score

    return {
        "score": score,
        "score_filtered": score.where(final_mask),
        "positive_days": positive_days,
        "cum_inflow": cum_inflow,
        "cum_amount": cum_amount,
        "price_change_n": price_change_n,
        "mask_c1": mask_c1,
        "mask_c3": mask_c3,
        "final_mask": final_mask,
    }


def select_v0_2_top_n_for_date(
    score_filtered_row: pd.Series,
    universe_codes: Iterable[str],
    n: int = TARGET_TOP_N,
) -> list[str]:
    """单日选 v0.2 Top-N：在 universe ∩ score_filtered 非 NaN 集合内按 score 降序排.

    tie-break：stock_code 字典序升序（确定性）.
    """
    u = set(universe_codes)
    s = score_filtered_row.reindex([c for c in score_filtered_row.index if c in u]).dropna()
    if len(s) == 0:
        return []
    sorted_codes = sorted(s.index, key=lambda c: (-float(s[c]), c))
    return sorted_codes[:n]


# ============================================================================


def dual_signal_overlap(
    selected_a: Iterable[str],
    selected_b: Iterable[str],
) -> float:
    """计算两个选股集合的 Jaccard 重合度（设计 §3.5 双信号交叉验证）.

    重合度 = |A ∩ B| / |A ∪ B|；两者空集时返回 0.
    """
    set_a = set(selected_a)
    set_b = set(selected_b)
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)
