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


# ============================================================================
# v0.8 GBDT 多信号合成 (2026-05-14 加入)
# ============================================================================
# 11 features = 5 fund-flow + 6 price/momentum; target = 5d fwd_ret 截面 rank.
# 训练 = LightGBM regression on cross-sectional pct rank.

V0_8_FUND_FLOW_FEATS = [
    "super_net_inflow_ratio", "big_net_inflow_ratio",
    "middle_net_inflow_ratio", "small_net_inflow_ratio",
    "main_net_inflow_ratio",  # = super + big
]
V0_8_PRICE_FEATS = [
    "quote_change", "ret_5d", "ret_20d", "ret_60d", "vol_20d", "position_20d",
]
V0_8_ALL_FEATS = V0_8_FUND_FLOW_FEATS + V0_8_PRICE_FEATS
V0_8_IC_HORIZON = 5  # 5d fwd_ret 做 target

V0_8_DEFAULT_LGBM_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "n_estimators": 300,
    "max_depth": 5,
    "num_leaves": 31,
    "min_child_samples": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}


def _build_v0_8_features_only(panel: dict) -> dict[str, pd.DataFrame]:
    price = panel["price"]
    qc = panel["quote_change"]
    ret_5d = price / price.shift(5) - 1
    ret_20d = price / price.shift(20) - 1
    ret_60d = price / price.shift(60) - 1
    daily_ret = price.pct_change()
    vol_20d = daily_ret.rolling(20).std()
    high_20d = price.rolling(20).max()
    low_20d = price.rolling(20).min()
    position_20d = (price - low_20d) / (high_20d - low_20d).replace(0, np.nan)
    return {
        "super_net_inflow_ratio": panel["super"],
        "big_net_inflow_ratio": panel["big"],
        "middle_net_inflow_ratio": panel["middle"],
        "small_net_inflow_ratio": panel["small"],
        "main_net_inflow_ratio": panel["super"] + panel["big"],
        "quote_change": qc,
        "ret_5d": ret_5d,
        "ret_20d": ret_20d,
        "ret_60d": ret_60d,
        "vol_20d": vol_20d,
        "position_20d": position_20d,
    }


def build_v0_8_feature_panel(panel: dict, horizon: int = V0_8_IC_HORIZON) -> pd.DataFrame:
    """Long-format: V0_8_ALL_FEATS + target_rank + fwd_ret; 用于训练 (drop NaN)."""
    fps = _build_v0_8_features_only(panel)
    price = panel["price"]
    fwd_ret = price.shift(-horizon) / price.shift(-1) - 1
    target_rank = fwd_ret.rank(axis=1, pct=True)
    frames = [pv.stack(dropna=False).rename(name) for name, pv in fps.items()]
    frames.append(target_rank.stack(dropna=False).rename("target_rank"))
    frames.append(fwd_ret.stack(dropna=False).rename("fwd_ret"))
    df_long = pd.concat(frames, axis=1).reset_index()
    df_long.columns.values[:2] = ["date", "stock_id"]
    return df_long.dropna(subset=V0_8_ALL_FEATS + ["target_rank"])


def build_v0_8_inference_panel(panel: dict) -> pd.DataFrame:
    """Long-format inference: V0_8_ALL_FEATS only (无 target), 保留尾部 N 天."""
    fps = _build_v0_8_features_only(panel)
    frames = [pv.stack(dropna=False).rename(name) for name, pv in fps.items()]
    df_long = pd.concat(frames, axis=1).reset_index()
    df_long.columns.values[:2] = ["date", "stock_id"]
    return df_long.dropna(subset=V0_8_ALL_FEATS)


def train_v0_8_model(features_df: pd.DataFrame, lgbm_params: dict | None = None):
    """LGBMRegressor 预测 target_rank."""
    import lightgbm as lgb
    params = dict(V0_8_DEFAULT_LGBM_PARAMS)
    if lgbm_params:
        params.update(lgbm_params)
    X = features_df[V0_8_ALL_FEATS].values
    y = features_df["target_rank"].values
    model = lgb.LGBMRegressor(**params)
    model.fit(X, y)
    return model


def save_v0_8_model(model, path) -> None:
    import joblib
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, p)


def load_v0_8_model(path):
    import joblib
    return joblib.load(path)


def predict_v0_8(model, features_df: pd.DataFrame) -> pd.Series:
    """Returns Series indexed by (date, stock_id), values = pred."""
    X = features_df[V0_8_ALL_FEATS].values
    preds = model.predict(X)
    return pd.Series(preds, index=pd.MultiIndex.from_arrays(
        [features_df["date"], features_df["stock_id"]],
        names=["date", "stock_id"]), name="pred")


def select_v0_8_avoid_bot_for_date(
    predictions_pivot: pd.DataFrame,
    date,
    universe: Iterable[str] | None = None,
    bot_pct: float = 0.05,
) -> list[str]:
    """剔除 pred 最低 bot_pct, 返回剩余 stock_id sorted."""
    if date not in predictions_pivot.index:
        return []
    row = predictions_pivot.loc[date].dropna()
    if universe is not None:
        univ_set = set(universe)
        row = row[row.index.isin(univ_set)]
    if len(row) == 0:
        return []
    bot_thr = row.quantile(bot_pct)
    keep = row[row > bot_thr]
    return sorted(keep.index.tolist())


def select_v0_8_top_n_for_date(
    predictions_pivot: pd.DataFrame,
    date,
    universe: Iterable[str] | None = None,
    n: int = TARGET_TOP_N,
) -> list[str]:
    """选 pred 最高 n 只 (v0_1/v0_2 风格)."""
    if date not in predictions_pivot.index:
        return []
    row = predictions_pivot.loc[date].dropna()
    if universe is not None:
        univ_set = set(universe)
        row = row[row.index.isin(univ_set)]
    if len(row) == 0:
        return []
    return sorted(row.sort_values(ascending=False).head(n).index.tolist())
