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
