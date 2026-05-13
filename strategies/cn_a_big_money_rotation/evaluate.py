"""M5 评估层：IC / 双基准 / 5 种 verdict / 双信号交叉（设计 §8, §3.5）.

verdict 优先级（按设计 §8 表）：
1. NO-ALPHA      : IC mean ≤ 0.02 OR t-stat ≤ 3
2. CHURNING      : IC 过 + 年化超额 vs BM1 < 3% 红线
3. SIZE-BETA-ONLY: vs BM1 过 + vs BM2 ≤ 0
4. UNSTABLE      : 通过门槛 但 连续 3 月 vs BM2 累计 < −10%
5. PASS-prelim   : 全部门槛过

所有函数纯计算，零 DB 依赖；返回 dataclass / Enum.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


# ============================================================================
# 门槛常量（设计 §8）
# ============================================================================

IC_MEAN_THRESHOLD = 0.02
IC_TSTAT_THRESHOLD = 3.0
IR_BM1_THRESHOLD = 0.4               # 短样本下自 0.5 放宽
IR_BM2_THRESHOLD = 0.3
ANN_EXCESS_BM1_THRESHOLD = 0.05      # 5%
ANN_EXCESS_BM2_THRESHOLD = 0.03      # 3%
CHURNING_RED_LINE = 0.03             # 3% 红线
MAX_ABS_DD_THRESHOLD = 0.30
MAX_REL_DD_THRESHOLD = 0.15
MAX_TURNOVER_THRESHOLD = 1.0         # 月换手率 100%
SHARPE_THRESHOLD = 0.8

UNSTABLE_REL_LOSS_THRESHOLD = -0.10
UNSTABLE_WINDOW_MONTHS = 3

DUAL_SIGNAL_OVERLAP_THRESHOLD = 0.60

TRADING_DAYS_PER_YEAR = 252


# ============================================================================
# 枚举与数据结构
# ============================================================================

class Verdict(str, Enum):
    PASS_PRELIM = "PASS-prelim"
    CHURNING = "CHURNING"
    SIZE_BETA_ONLY = "SIZE-BETA-ONLY"
    NO_ALPHA = "NO-ALPHA"
    UNSTABLE = "UNSTABLE"


class DualSignalStatus(str, Enum):
    DUAL_CONFIRMED = "dual_signal_confirmed"
    ALPHA_MAY_BE_BETA_DRIVEN = "alpha_may_be_beta_driven"
    SWITCH_PRIMARY_SIGNAL = "switch_primary_signal_in_v0_1_5"
    NO_ALPHA = "no_alpha"


@dataclass
class ICStats:
    mean: float
    std: float
    t_stat: float
    n: int

    def passes(self,
               mean_thr: float = IC_MEAN_THRESHOLD,
               t_thr: float = IC_TSTAT_THRESHOLD) -> bool:
        return self.mean > mean_thr and self.t_stat > t_thr


@dataclass
class StrategyMetrics:
    ic_a: ICStats
    ic_b: ICStats
    cumulative_return: float
    annualized_return: float
    excess_return_bm1: float        # 年化超额 vs HS300
    excess_return_bm2: float        # 年化超额 vs 等权 universe
    ir_bm1: float
    ir_bm2: float
    max_drawdown: float             # 绝对回撤（负数）
    max_relative_dd_bm1: float
    monthly_turnover: float
    sharpe: float
    overlap_ab: float               # 双信号选股重合度


# ============================================================================
# 数学工具
# ============================================================================

def compute_ic(scores: pd.Series, forward_returns: pd.Series) -> float:
    """单日 Spearman rank correlation；不足 2 个有效对 → NaN."""
    if len(scores) != len(forward_returns):
        raise ValueError("scores 与 forward_returns 长度不一致")
    s = pd.to_numeric(scores, errors="coerce")
    r = pd.to_numeric(forward_returns, errors="coerce")
    mask = s.notna() & r.notna()
    if mask.sum() < 2:
        return float("nan")
    return float(s[mask].corr(r[mask], method="spearman"))


def compute_ic_stats(daily_ic: pd.Series) -> ICStats:
    """daily_ic：每日 IC Series；NaN 排除后算 mean / std / t-stat."""
    series = pd.to_numeric(daily_ic, errors="coerce").dropna()
    n = len(series)
    if n == 0:
        return ICStats(mean=float("nan"), std=float("nan"), t_stat=float("nan"), n=0)
    mean = float(series.mean())
    std = float(series.std(ddof=1)) if n > 1 else 0.0
    if std == 0 or np.isnan(std):
        t_stat = float("nan") if std == 0 else 0.0
    else:
        t_stat = mean / (std / np.sqrt(n))
    return ICStats(mean=mean, std=std, t_stat=t_stat, n=n)


def annualize_return(
    cumulative_return: float,
    n_periods: int,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """几何年化：(1+cum)^(periods_per_year/n) - 1."""
    if n_periods <= 0:
        return float("nan")
    base = 1.0 + cumulative_return
    if base <= 0:
        return -1.0
    return base ** (periods_per_year / n_periods) - 1.0


_STD_ZERO_TOL = 1e-12  # float 精度阈值：std 小于此视为零除


def compute_sharpe(
    daily_returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    risk_free_daily: float = 0.0,
) -> float:
    """年化 Sharpe = mean / std × sqrt(periods)；std≈0 → 0."""
    r = pd.to_numeric(daily_returns, errors="coerce").dropna()
    if len(r) < 2:
        return float("nan")
    excess = r - risk_free_daily
    std = excess.std(ddof=1)
    if std < _STD_ZERO_TOL or np.isnan(std):
        return 0.0
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def compute_information_ratio(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """年化 IR = 年化 alpha / 年化 tracking error."""
    s = pd.to_numeric(strategy_returns, errors="coerce")
    b = pd.to_numeric(benchmark_returns, errors="coerce")
    mask = s.notna() & b.notna()
    if mask.sum() < 2:
        return float("nan")
    diff = s[mask] - b[mask]
    std = diff.std(ddof=1)
    if std < _STD_ZERO_TOL or np.isnan(std):
        return 0.0
    return float(diff.mean() / std * np.sqrt(periods_per_year))


def compute_max_drawdown(equity_curve: pd.Series) -> float:
    """最大回撤（负数，单位百分比小数）."""
    eq = pd.to_numeric(equity_curve, errors="coerce").dropna()
    if len(eq) == 0:
        return float("nan")
    running_max = eq.cummax()
    dd = (eq - running_max) / running_max
    return float(dd.min())


def detect_unstable_window(
    monthly_excess_vs_bm2: pd.Series,
    window: int = UNSTABLE_WINDOW_MONTHS,
    threshold: float = UNSTABLE_REL_LOSS_THRESHOLD,
) -> bool:
    """检测是否存在连续 window 个月 vs BM2 累计 < threshold.

    monthly_excess_vs_bm2: 月度超额收益序列（小数）.
    """
    s = pd.to_numeric(monthly_excess_vs_bm2, errors="coerce").dropna()
    if len(s) < window:
        return False
    rolling_sum = s.rolling(window=window).sum()
    return bool((rolling_sum < threshold).any())


# ============================================================================
# 双信号交叉验证（设计 §3.5）
# ============================================================================

def classify_dual_signal(
    ic_a: ICStats,
    ic_b: ICStats,
    overlap: float,
    *,
    mean_thr: float = IC_MEAN_THRESHOLD,
    t_thr: float = IC_TSTAT_THRESHOLD,
    overlap_thr: float = DUAL_SIGNAL_OVERLAP_THRESHOLD,
) -> DualSignalStatus:
    a_pass = ic_a.passes(mean_thr, t_thr)
    b_pass = ic_b.passes(mean_thr, t_thr)
    if a_pass and b_pass:
        if overlap > overlap_thr:
            return DualSignalStatus.DUAL_CONFIRMED
        # 双 PASS 但重合度低：保守标 may_be_beta（A 仍为主决策）
        return DualSignalStatus.ALPHA_MAY_BE_BETA_DRIVEN
    if a_pass and not b_pass:
        return DualSignalStatus.ALPHA_MAY_BE_BETA_DRIVEN
    if not a_pass and b_pass:
        return DualSignalStatus.SWITCH_PRIMARY_SIGNAL
    return DualSignalStatus.NO_ALPHA


# ============================================================================
# Verdict 决策（设计 §8）
# ============================================================================

def render_verdict(
    metrics: StrategyMetrics,
    *,
    monthly_excess_vs_bm2: pd.Series | None = None,
) -> tuple[Verdict, dict[str, object]]:
    """根据 metrics 给出 verdict + notes.

    优先级：NO-ALPHA → CHURNING → SIZE-BETA-ONLY → UNSTABLE → PASS-prelim.
    """
    notes: dict[str, object] = {}

    # ---- 1. NO-ALPHA: IC 不过门槛 ----
    ic_a_pass = metrics.ic_a.passes()
    notes["ic_a_pass"] = ic_a_pass
    if not ic_a_pass:
        notes["reason"] = "ic_a_failed"
        return Verdict.NO_ALPHA, notes

    # ---- 2. CHURNING: 年化超额 vs BM1 < 3% 红线 ----
    if metrics.excess_return_bm1 < CHURNING_RED_LINE:
        notes["reason"] = "ic_pass_but_cost_eats_alpha"
        notes["excess_bm1"] = metrics.excess_return_bm1
        return Verdict.CHURNING, notes

    # ---- 3. SIZE-BETA-ONLY: BM1 过 + BM2 ≤ 0 ----
    if metrics.excess_return_bm2 <= 0:
        notes["reason"] = "bm1_excess_explained_by_size_beta"
        notes["excess_bm2"] = metrics.excess_return_bm2
        return Verdict.SIZE_BETA_ONLY, notes

    # ---- 4. UNSTABLE: 连续 3 月 vs BM2 累计 < -10% ----
    if monthly_excess_vs_bm2 is not None and detect_unstable_window(monthly_excess_vs_bm2):
        notes["reason"] = "consecutive_3m_underperform_bm2"
        return Verdict.UNSTABLE, notes

    # ---- 5. 其他软门槛（PASS-prelim 前置检查）----
    failed: list[str] = []
    if metrics.excess_return_bm1 < ANN_EXCESS_BM1_THRESHOLD:
        failed.append("excess_bm1_below_5pct")
    if metrics.excess_return_bm2 < ANN_EXCESS_BM2_THRESHOLD:
        failed.append("excess_bm2_below_3pct")
    if metrics.ir_bm1 < IR_BM1_THRESHOLD:
        failed.append("ir_bm1_below_0.4")
    if metrics.ir_bm2 < IR_BM2_THRESHOLD:
        failed.append("ir_bm2_below_0.3")
    if abs(metrics.max_drawdown) > MAX_ABS_DD_THRESHOLD:
        failed.append("max_dd_above_30pct")
    if abs(metrics.max_relative_dd_bm1) > MAX_REL_DD_THRESHOLD:
        failed.append("rel_dd_bm1_above_15pct")
    if metrics.monthly_turnover > MAX_TURNOVER_THRESHOLD:
        failed.append("turnover_above_100pct")
    if metrics.sharpe < SHARPE_THRESHOLD:
        failed.append("sharpe_below_0.8")

    if failed:
        notes["reason"] = "soft_thresholds_failed"
        notes["failed"] = failed
        return Verdict.NO_ALPHA, notes

    notes["reason"] = "all_thresholds_pass"
    return Verdict.PASS_PRELIM, notes
