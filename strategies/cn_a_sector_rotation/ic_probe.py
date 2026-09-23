"""§4 L1 / H-B3：行业 rank IC 探针。周频截面（每 every 个交易日）、分段统计（复用 xs_probe）、分位价差、子宇宙掩码。"""
from __future__ import annotations

import math

import pandas as pd

from strategies.cn_a_trend_base.xs_probe import FoldStat, daily_rank_ic, fold_stats, verdict

IC_MIN, T_MIN, N_MIN_WEEKS = 0.03, 2.0, 50          # §4 L1 门槛：IC 均值 ≥ 0.03、t ≥ 2；每段至少 50 个周截面


def weekly_rank_ic(feature: pd.DataFrame, fwd: pd.DataFrame, mask: pd.DataFrame, every: int = 5,
                   min_names: int = 10) -> pd.Series:
    """每 every 个交易日取一个截面，在 mask 成员上算 Spearman(feature, fwd)；成员 < min_names → NaN。"""
    rows = feature.index[::every]
    return daily_rank_ic(feature.loc[rows], fwd.loc[rows], mask.loc[rows], min_names=min_names)


def quantile_spread(feature: pd.DataFrame, fwd: pd.DataFrame, mask: pd.DataFrame, every: int = 5, q: int = 5,
                    min_names: int | None = None) -> pd.DataFrame:
    """每 every 日一个截面，按特征分 q 组的平均前向收益（Q1 最低 … Qq 最高）。成员 < min_names（默认 2q）的截面跳过。"""
    need = min_names if min_names is not None else 2 * q
    f = feature.where(mask); r = fwd.where(mask)
    rows = {}
    for d in feature.index[::every]:
        x, y = f.loc[d], r.loc[d]
        m = x.notna() & y.notna()
        if m.sum() < need:
            continue
        g = pd.qcut(x[m].rank(method="first"), q, labels=False)
        rows[d] = y[m].groupby(g).mean().reindex(range(q)).to_numpy()
    return pd.DataFrame.from_dict(rows, orient="index", columns=[f"Q{i + 1}" for i in range(q)]).sort_index()


def amount_split_masks(amount: pd.DataFrame, window: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """§2.4 子宇宙：20 日均成交额 ≥ 当日截面中位数 → 上半区（大板块），否则下半区。窗口不足 → 两边皆 False。"""
    avg = amount.rolling(window, min_periods=window).mean()
    med = avg.median(axis=1)
    valid = avg.notna()
    return avg.ge(med, axis=0) & valid, avg.lt(med, axis=0) & valid


def nw_lags(k: int, every: int) -> int:
    """重叠期修正的滞后数：k 日前向收益按 every 日采样 → ceil(k / every) 个重叠截面。"""
    return max(1, math.ceil(k / every))


def segment_stats(ic: pd.Series, names: pd.Series, segments: dict[str, tuple[str, str | None]], k: int,
                  every: int = 5) -> list[FoldStat]:
    """三段周频 IC 均值 / NW-t / 截面数，复用 xs_probe.fold_stats（lags 按重叠期）。"""
    return fold_stats(ic, names.reindex(ic.index), segments, lags=nw_lags(k, every))


def l1_verdict(stats: list[FoldStat]) -> tuple[bool, str]:
    """§4 L1：三段 IC ≥ 0.03、t ≥ 2、三段同号（含最近段不反号）、每段 ≥ N_MIN_WEEKS 个截面。登记方向 IC > 0。"""
    return verdict(stats, expected_sign=+1, ic_min=IC_MIN, t_min=T_MIN, n_min_days=N_MIN_WEEKS)


def spread_by_segment(qs: pd.DataFrame, segments: dict[str, tuple[str, str | None]]) -> dict[str, float]:
    """每段 Qq − Q1 平均价差（NaN = 无截面）。"""
    hi, lo = qs.columns[-1], qs.columns[0]
    out = {}
    for name, (s, e) in segments.items():
        d = qs.loc[s:e] if e else qs.loc[s:]
        out[name] = float((d[hi] - d[lo]).mean()) if len(d) else float("nan")
    return out


__all__ = ["IC_MIN", "T_MIN", "N_MIN_WEEKS", "weekly_rank_ic", "quantile_spread", "amount_split_masks", "nw_lags",
           "segment_stats", "l1_verdict", "spread_by_segment"]
