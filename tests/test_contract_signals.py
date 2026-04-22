"""Contract signal tests (utils.signals.add_contract_signals + compute_sig_*).

Focuses on:
- Value-range contract [-1, 1]
- Warmup NaN → 0 (neutral)
- sig_liq_imbalance ε-smoothing defeating low-volume noise（核心回归点 §11.5）
- Missing columns → all-zero output (graceful partial-data handling)
- Extreme-edge invariants
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils.signals import (
    CONTRACT_SIGNAL_COLS,
    add_contract_signals,
    compute_sig_funding_current,
    compute_sig_funding_trend,
    compute_sig_liq_imbalance,
    compute_sig_liq_long_zscore,
    compute_sig_oi_change_zscore,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(n: int, freq: str = "5min") -> pd.DatetimeIndex:
    return pd.date_range("2026-01-01", periods=n, freq=freq, tz="UTC")


# ---------------------------------------------------------------------------
# Value-range + warmup contract
# ---------------------------------------------------------------------------

def test_add_contract_signals_value_range() -> None:
    n = 500
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "timestamp": _ts(n),
        "funding_rate": rng.normal(0, 1e-4, n),
        "sum_open_interest": 1e8 + rng.normal(0, 1e5, n).cumsum(),
        "liq_long_usd":  np.abs(rng.normal(5e5, 2e5, n)),
        "liq_short_usd": np.abs(rng.normal(5e5, 2e5, n)),
    })
    out = add_contract_signals(df)
    for col in CONTRACT_SIGNAL_COLS:
        assert col in out.columns
        v = out[col]
        assert v.between(-1.0, 1.0).all(), f"{col} out of range"
        assert not v.isna().any(), f"{col} has NaN"


def test_add_contract_signals_zero_warmup() -> None:
    """前 1-2 行 znorm 不足 → 应填 0（neutral）。"""
    n = 500
    df = pd.DataFrame({
        "timestamp": _ts(n),
        "funding_rate": np.linspace(0, 1e-4, n),
        "sum_open_interest": np.linspace(1e8, 1.1e8, n),
        "liq_long_usd":  np.full(n, 5e5),
        "liq_short_usd": np.full(n, 5e5),
    })
    out = add_contract_signals(df)
    # 前两行应为 0（znorm 要求 window>=100, std>0；少于 window 仍给 NaN，被 fillna(0) 填平）
    for col in CONTRACT_SIGNAL_COLS:
        assert out[col].iloc[0] == 0.0


def test_add_contract_signals_missing_columns_zero_neutral() -> None:
    """只有 OHLCV 没有合约列 → 6 个 sig 全为 0。"""
    n = 200
    df = pd.DataFrame({
        "timestamp": _ts(n),
        "open":  np.ones(n),
        "close": np.ones(n),
    })
    out = add_contract_signals(df)
    for col in CONTRACT_SIGNAL_COLS:
        assert (out[col] == 0.0).all()


def test_add_contract_signals_partial_funding_only() -> None:
    """funding 有但 OI/liq 缺 → funding 相关 sig 非零可能，其他应为 0。"""
    n = 400
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "timestamp": _ts(n),
        "funding_rate": rng.normal(0, 1e-4, n),
    })
    out = add_contract_signals(df)
    assert (out["sig_oi_change_zscore"] == 0.0).all()
    assert (out["sig_liq_long_zscore"] == 0.0).all()
    assert (out["sig_liq_short_zscore"] == 0.0).all()
    assert (out["sig_liq_imbalance"] == 0.0).all()
    # funding 相关至少应有非零值
    assert out["sig_funding_current"].abs().sum() > 0


# ---------------------------------------------------------------------------
# sig_liq_imbalance — ε-smoothing（核心）
# ---------------------------------------------------------------------------

def test_liq_imbalance_low_volume_spike_suppressed() -> None:
    """
    低流动性伪信号：市场历史总量 ~5e6，当前仅 (100, 10)。
    预期 signal 接近 0（< 0.01）。
    """
    window = 96
    # 前 window 个桶 total=5e6（多空各半）
    long_ = [2.5e6] * window + [100.0]
    short_ = [2.5e6] * window + [10.0]
    s_long = pd.Series(long_)
    s_short = pd.Series(short_)
    out = compute_sig_liq_imbalance(s_long, s_short, epsilon_window=window)
    last = out.iloc[-1]
    assert abs(last) < 0.01, f"low-volume spike not suppressed: {last}"


def test_liq_imbalance_active_market_signal_preserved() -> None:
    """
    活跃市场：历史 ~5e6，当前 (1e7, 1e6) → 多头被爆远大于空头。
    预期 signal ≈ 0.78（接近原公式 0.818）。
    """
    window = 96
    long_ = [2.5e6] * window + [1e7]
    short_ = [2.5e6] * window + [1e6]
    s_long = pd.Series(long_)
    s_short = pd.Series(short_)
    out = compute_sig_liq_imbalance(s_long, s_short, epsilon_window=window)
    last = out.iloc[-1]
    assert 0.7 < last < 0.9, f"active-market imbalance off: {last}"


def test_liq_imbalance_low_volume_but_consistent_regime_keeps_signal() -> None:
    """
    真实低量期：全市场都很低（200），当前 (100, 10)。
    ε 随之很小（~20）→ 信号 ~(90)/(110+20)=0.69，保留方向信息。
    """
    window = 96
    long_ = [100.0] * window + [100.0]
    short_ = [100.0] * window + [10.0]
    s_long = pd.Series(long_)
    s_short = pd.Series(short_)
    out = compute_sig_liq_imbalance(s_long, s_short, epsilon_window=window)
    last = out.iloc[-1]
    assert last > 0.4, f"signal over-suppressed in low-volume regime: {last}"


def test_liq_imbalance_all_zero_returns_zero() -> None:
    n = 100
    s = pd.Series(np.zeros(n))
    out = compute_sig_liq_imbalance(s, s, epsilon_window=96)
    assert (out == 0.0).all()


def test_liq_imbalance_extreme_single_side_near_one() -> None:
    window = 96
    long_ = [5e6] * window + [1e8]
    short_ = [5e6] * window + [0.0]
    out = compute_sig_liq_imbalance(
        pd.Series(long_), pd.Series(short_), epsilon_window=window,
    )
    last = out.iloc[-1]
    assert last > 0.95


def test_liq_imbalance_cold_start_window_uses_actual_samples() -> None:
    """
    min_periods=1 生效：样本不足 window 也能算（用已有样本均值）。
    """
    # 只有 10 个样本，window=96 → 冷启动期用 10 个样本的均值
    s_long = pd.Series([1e6] * 9 + [1e7])
    s_short = pd.Series([1e6] * 9 + [1e6])
    out = compute_sig_liq_imbalance(s_long, s_short, epsilon_window=96)
    last = out.iloc[-1]
    # ma ≈ (1e6*9 + 1.1e7) / 10 = 2e6，ε=2e5，denom=1.1e7+2e5=1.12e7，diff=9e6，sig≈0.80
    assert 0.7 < last < 0.9


def test_liq_imbalance_nan_inputs_treated_as_zero() -> None:
    """NaN liq 输入（比如启动阶段）应被 fillna(0) 安全处理。"""
    n = 100
    s_long = pd.Series([np.nan] * 50 + [1e6] * 50)
    s_short = pd.Series([np.nan] * 50 + [5e5] * 50)
    out = compute_sig_liq_imbalance(s_long, s_short, epsilon_window=96)
    # 前 50 个应为 0（NaN → 0 → diff=0）
    assert (out.iloc[:50] == 0.0).all()
    # 后续应有合理值
    assert out.iloc[-1] != 0.0


# ---------------------------------------------------------------------------
# sig_funding_current / sig_funding_trend
# ---------------------------------------------------------------------------

def test_funding_current_sustained_positive_means_positive() -> None:
    """funding 从历史均值附近向上突破 → 当前 z-score > 0。"""
    n = 500
    rng = np.random.default_rng(10)
    baseline = rng.normal(0, 1e-5, n)  # 历史噪音均值 0
    fr = pd.Series(baseline)
    # 最后一段拉到远高于历史均值
    fr.iloc[-50:] = 5e-4
    out = compute_sig_funding_current(fr, window=200)
    assert out.iloc[-1] > 0.5


def test_funding_current_constant_returns_zero() -> None:
    fr = pd.Series(np.full(400, 1e-4))
    out = compute_sig_funding_current(fr, window=200)
    # std=0 (+ 1e-8)，(x-mean)=0 → z=0
    assert (out.iloc[200:] == 0.0).all()


def test_funding_trend_smooths_noise() -> None:
    """加剧噪音时，trend（24h MA）应比 current 平滑（方差更小）。"""
    n = 500
    rng = np.random.default_rng(2)
    fr = pd.Series(rng.normal(0, 1e-3, n))
    cur = compute_sig_funding_current(fr, window=200)
    trend = compute_sig_funding_trend(fr, mean_window=288, znorm_window=200)
    # 跳过 warmup
    assert trend.iloc[300:].std() < cur.iloc[300:].std()


# ---------------------------------------------------------------------------
# sig_oi_change_zscore
# ---------------------------------------------------------------------------

def test_oi_change_zscore_rising_is_positive() -> None:
    """OI 持续上升段落的 z-score 应 > 0。"""
    n = 400
    # 前半 OI 平稳，后半加速上升
    oi = np.concatenate([
        np.linspace(1e8, 1.001e8, 200),
        np.linspace(1.001e8, 1.1e8, 200),
    ])
    out = compute_sig_oi_change_zscore(pd.Series(oi))
    assert out.iloc[-10:].mean() > 0


def test_oi_change_zscore_constant_returns_zero() -> None:
    oi = pd.Series(np.full(400, 1e8))
    out = compute_sig_oi_change_zscore(oi)
    # pct_change = 0 全程；znorm warmup 后为 0（std≈0 + 1e-8 → (0-0)/1e-8=0）
    # compute_sig_* 本身不 fillna（add_contract_signals 层统一处理），这里只看 warmup 后
    valid = out.dropna()
    assert len(valid) > 0
    assert (valid == 0.0).all()


# ---------------------------------------------------------------------------
# sig_liq_long / short z-score
# ---------------------------------------------------------------------------

def test_liq_long_zscore_spike_positive() -> None:
    n = 300
    rng = np.random.default_rng(3)
    base = pd.Series(np.abs(rng.normal(5e5, 1e5, n)))
    base.iloc[-1] = 5e7   # 大爆仓事件
    out = compute_sig_liq_long_zscore(base, window=100)
    assert out.iloc[-1] > 0.9
