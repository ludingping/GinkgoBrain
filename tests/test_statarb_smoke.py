"""
Tests for scripts/statarb_eth_btc_smoke.py.

对应 docs/GinkgoBrain/StatArb-ETH-BTC价差-测试用例.md Level 1（9 条 pytest 单测）：
1. β 滚动拟合无未来泄漏
2. Z-score shift(1) 防前视泄漏
3. ADF 固定 maxlag 确定性
4. Hedge-ratio sizing 数学正确性
5. Hedge-ratio PnL 跟随 dε
6. 状态机：开仓 / 出场 触发顺序
7. inner-join 数据对齐
8. np.polyfit 系数顺序回归锁
9. 手续费汇总公式
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.stattools import adfuller

from scripts.statarb_eth_btc_smoke import (
    BacktestParams,
    compute_rolling_adf,
    compute_rolling_beta,
    compute_trade_pnl,
    compute_zscore,
    hedge_ratio_legs,
    run_backtest,
)


# ─── Fixtures ───────────────────────────────────────────────────────────────


def _make_synthetic_pair(
    n: int = 500,
    beta_true: float = 1.0,
    alpha_true: float = 0.5,
    shock_at: int | None = None,
    shock_log_diff: float = 2.3,  # ≈ ln(10)
    seed: int = 42,
) -> pd.DataFrame:
    """log_eth = beta_true * log_btc + alpha_true + tiny noise；可在 shock_at 注入突变。"""
    rng = np.random.default_rng(seed)
    log_btc = np.cumsum(rng.normal(0, 0.01, n))
    log_eth = beta_true * log_btc + alpha_true + rng.normal(0, 0.001, n)
    if shock_at is not None:
        log_eth[shock_at:] += shock_log_diff
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=n, freq="4h", tz="UTC"),
            "btc_close": np.exp(log_btc) * 40000,
            "eth_close": np.exp(log_eth) * 2000,
            "log_btc": log_btc,
            "log_eth": log_eth,
        }
    )


# ─── 1. β 滚动拟合无未来泄漏 ─────────────────────────────────────────────────


def test_rolling_beta_normal_segment():
    """正常段 β ≈ true（容差 0.05）。"""
    df = _make_synthetic_pair(n=500, beta_true=1.0)
    out = compute_rolling_beta(df, window=100)
    assert abs(out["beta"].iloc[100] - 1.0) < 0.05


def test_rolling_beta_no_lookahead():
    """index=200 处插突变，β[200] 不能受影响（用的是 [100, 199] 的数据）。"""
    df = _make_synthetic_pair(n=500, beta_true=1.0, shock_at=200, shock_log_diff=2.3)
    out = compute_rolling_beta(df, window=100)
    assert abs(out["beta"].iloc[200] - 1.0) < 0.05, (
        f"β[200]={out['beta'].iloc[200]:.3f} 不应受 i=200 的突变影响（无未来泄漏）"
    )
    # β[201] 开始包含 i=200 的突变，应明显偏离
    assert out["beta"].iloc[201] > 1.05 or out["beta"].iloc[201] < 0.95, (
        "β[201] 应已开始反映突变"
    )


def test_rolling_beta_cold_start_nan():
    """i < window 时 β/α/spread 全为 NaN。"""
    df = _make_synthetic_pair(n=200)
    out = compute_rolling_beta(df, window=100)
    assert out["beta"].iloc[:100].isna().all()
    assert out["alpha"].iloc[:100].isna().all()
    assert out["spread"].iloc[:100].isna().all()
    assert out["beta"].iloc[100:].notna().all()


# ─── 2. Z-score 防前视泄漏 ──────────────────────────────────────────────────


def test_zscore_excludes_current():
    """spike 不被自己污染：correct（shift(1)）的 |z| 应远大于 bad（无 shift）。"""
    spread = pd.Series([0.0] * 100 + [10.0] + [0.0] * 10)
    z_correct = compute_zscore(spread, window=50)
    bad_mean = spread.rolling(50).mean()
    bad_std = spread.rolling(50).std()
    z_bad = (spread - bad_mean) / bad_std
    assert abs(z_correct.iloc[100]) > abs(z_bad.iloc[100]) * 1.5


def test_zscore_cold_start_nan():
    """前 window+1 根 z-score 应为 NaN（rolling 50 → 第一非 NaN 在 49；shift(1) → 50）。"""
    spread = pd.Series(np.random.RandomState(42).randn(200))
    z = compute_zscore(spread, window=50)
    assert z.iloc[:50].isna().all()
    assert z.iloc[50:].notna().all()


# ─── 3. ADF 固定 maxlag 确定性 ──────────────────────────────────────────────


def test_adf_deterministic():
    """同输入两次 adfuller(maxlag=4, autolag=None) p-value 完全相等。"""
    rng = np.random.default_rng(42)
    seg = pd.Series(rng.normal(0, 1, 180))
    r1 = adfuller(seg, maxlag=4, autolag=None)
    r2 = adfuller(seg, maxlag=4, autolag=None)
    assert r1[1] == r2[1]
    assert r1[2] == r2[2] == 4


def test_rolling_adf_usedlag_stable():
    """compute_rolling_adf 在不同段返回的 usedlag 应稳定 = maxlag。"""
    rng = np.random.default_rng(42)
    spread = pd.Series(rng.normal(0, 1, 600))
    out = compute_rolling_adf(spread, adf_window=180, maxlag=4)
    used_vals = out["adf_usedlag"].dropna().unique()
    assert (used_vals == 4).all(), f"usedlag 不稳定: {used_vals}"


# ─── 4. Hedge-ratio sizing 数学正确性 ───────────────────────────────────────


@pytest.mark.parametrize("beta,d_total", [
    (1.0, 5000.0),
    (1.05, 5000.0),
    (2.0, 5000.0),
    (0.5, 10000.0),
])
def test_hedge_ratio_sizing_invariant(beta, d_total):
    """D_eth + D_btc == D_total 恒成立。"""
    d_eth, d_btc = hedge_ratio_legs(beta, d_total)
    assert abs(d_eth + d_btc - d_total) < 1e-6
    assert abs(d_btc - beta * d_eth) < 1e-6


def test_hedge_ratio_sizing_specific_values():
    """β=1.05 时 D_eth ≈ 2439.02、D_btc ≈ 2560.97。"""
    d_eth, d_btc = hedge_ratio_legs(1.05, 5000.0)
    assert abs(d_eth - 2439.024390) < 1e-3
    assert abs(d_btc - 2560.975609) < 1e-3


def test_hedge_ratio_sizing_reject_negative_beta():
    """β ≤ -1 抛 ValueError。"""
    with pytest.raises(ValueError):
        hedge_ratio_legs(-1.0, 5000.0)
    with pytest.raises(ValueError):
        hedge_ratio_legs(-1.5, 5000.0)


# ─── 5. Hedge-ratio PnL 跟随 dε ─────────────────────────────────────────────


def test_pnl_tracks_spread_change_long_spread():
    """构造 dε=+0.002 → trade_pnl ≈ D_eth × 0.002（容差 1%）。"""
    beta = 1.05
    d_total = 5000.0
    eth_open, btc_open = 2000.0, 40000.0
    btc_ratio = 1.005
    btc_close = btc_open * btc_ratio
    delta_epsilon = 0.002
    eth_close = eth_open * np.exp(beta * np.log(btc_ratio) + delta_epsilon)

    info = compute_trade_pnl(
        direction=+1, beta_open=beta,
        eth_open=eth_open, btc_open=btc_open,
        eth_close=eth_close, btc_close=btc_close,
        d_total=d_total, commission=0.0,
    )
    expected = info["d_eth"] * delta_epsilon
    rel_err = abs(info["pnl_net"] - expected) / abs(expected)
    assert rel_err < 0.01, f"PnL={info['pnl_net']:.4f}, expected={expected:.4f}, err={rel_err:.4%}"


def test_pnl_tracks_spread_change_short_spread():
    """空价差（direction=-1）镜像情形。"""
    beta = 1.05
    d_total = 5000.0
    eth_open, btc_open = 2000.0, 40000.0
    btc_close = btc_open * 1.005
    delta_epsilon = -0.002
    eth_close = eth_open * np.exp(beta * np.log(1.005) + delta_epsilon)

    info = compute_trade_pnl(
        direction=-1, beta_open=beta,
        eth_open=eth_open, btc_open=btc_open,
        eth_close=eth_close, btc_close=btc_close,
        d_total=d_total, commission=0.0,
    )
    expected = info["d_eth"] * abs(delta_epsilon)
    assert info["pnl_net"] > 0
    rel_err = abs(info["pnl_net"] - expected) / abs(expected)
    assert rel_err < 0.01


def test_pnl_dollar_neutral_is_wrong():
    """反例：等额美元（v0.1 错误）显著偏离 D · dε。"""
    beta = 1.05
    d_per_leg = 2500.0
    eth_open, btc_open = 2000.0, 40000.0
    btc_close = btc_open * 1.005
    delta_epsilon = 0.002
    eth_close = eth_open * np.exp(beta * np.log(1.005) + delta_epsilon)

    qty_eth = +d_per_leg / eth_open
    qty_btc = -d_per_leg / btc_open
    pnl_wrong = qty_eth * (eth_close - eth_open) + qty_btc * (btc_close - btc_open)
    expected_correct = d_per_leg * delta_epsilon
    rel_err = abs(pnl_wrong - expected_correct) / abs(expected_correct)
    assert rel_err > 0.05, f"v0.1 错误实现误差只有 {rel_err:.4%}"


# ─── 6. 状态机：开仓 / 出场 ─────────────────────────────────────────────────


def _make_state_machine_df(z_series: list[float], adf_p: float = 0.05, beta: float = 1.0) -> pd.DataFrame:
    n = len(z_series)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=n, freq="4h", tz="UTC"),
            "btc_close": np.full(n, 40000.0),
            "eth_close": np.full(n, 2000.0),
            "beta": np.full(n, beta),
            "spread": np.zeros(n),
            "z_score": np.array(z_series, dtype=float),
            "adf_pvalue": np.full(n, adf_p),
        }
    )


def test_state_machine_z_revert():
    """z=-3 入 → z=0.3 出（reason=z_revert）。"""
    df = _make_state_machine_df([-3.0, -1.0, 0.3, 0.4])
    trades, _ = run_backtest(df, BacktestParams())
    assert len(trades) == 1
    assert trades[0].open_idx == 0
    assert trades[0].close_idx == 2
    assert trades[0].direction == +1
    assert trades[0].reason == "z_revert"


def test_state_machine_stop_loss():
    """z=-3 入 → z=-4.5 强平（reason=stop_loss）。"""
    df = _make_state_machine_df([-3.0, -3.5, -4.5, -3.0])
    trades, _ = run_backtest(df, BacktestParams())
    assert len(trades) == 1
    assert trades[0].close_idx == 2
    assert trades[0].reason == "stop_loss"


def test_state_machine_timeout():
    """z=-3 入 → 30 天 (180 bars) 后强平 timeout。"""
    z_series = [-3.0] + [-1.0] * 250
    df = _make_state_machine_df(z_series)
    trades, _ = run_backtest(df, BacktestParams())
    assert len(trades) == 1
    assert trades[0].reason == "timeout"
    assert trades[0].close_idx == 180


def test_state_machine_adf_blocks_entry():
    """adf_p=0.15 且 z=-3 时不开仓。"""
    df = _make_state_machine_df([-3.0, -3.0, -3.0], adf_p=0.15)
    trades, _ = run_backtest(df, BacktestParams())
    assert len(trades) == 0


def test_state_machine_adf_breakdown_keeps_old_position():
    """旧仓在 ADF 破裂后不强平，按 z_revert 正常出场。"""
    df = _make_state_machine_df([-3.0, -1.0, -1.0, 0.3], adf_p=0.05)
    df.loc[2:, "adf_pvalue"] = 0.20
    trades, _ = run_backtest(df, BacktestParams())
    assert len(trades) == 1
    assert trades[0].open_idx == 0
    assert trades[0].close_idx == 3
    assert trades[0].reason == "z_revert"


def test_state_machine_no_double_entry():
    """持仓中 z 再次极端，不重复开仓。"""
    df = _make_state_machine_df([-3.0, -3.0, -3.0, -3.0, 0.3])
    trades, _ = run_backtest(df, BacktestParams())
    assert len(trades) == 1


def test_state_machine_short_spread_entry():
    """z=+3 入 → direction=-1（short spread）。"""
    df = _make_state_machine_df([+3.0, +1.0, 0.3])
    trades, _ = run_backtest(df, BacktestParams())
    assert len(trades) == 1
    assert trades[0].direction == -1


# ─── 7. inner-join 数据对齐 ─────────────────────────────────────────────────


def test_inner_join_alignment():
    """ETH 时间戳缺失时 inner-join 丢弃单边孤儿。"""
    ts = pd.date_range("2023-01-01", periods=1000, freq="4h", tz="UTC")
    btc = pd.DataFrame({"timestamp": ts, "btc_close": np.arange(1000) * 100.0})
    eth = pd.DataFrame({"timestamp": ts[100:], "eth_close": np.arange(900) * 50.0})
    merged = btc.merge(eth, on="timestamp", how="inner")
    assert len(merged) == 900


def test_inner_join_random_gaps():
    """中间随机缺失 50 个 ts。"""
    rng = np.random.default_rng(7)
    ts = pd.date_range("2023-01-01", periods=1000, freq="4h", tz="UTC")
    drop_idx = rng.choice(1000, size=50, replace=False)
    keep_mask = np.ones(1000, dtype=bool)
    keep_mask[drop_idx] = False
    btc = pd.DataFrame({"timestamp": ts, "btc_close": np.arange(1000) * 100.0})
    eth = pd.DataFrame({"timestamp": ts[keep_mask], "eth_close": np.arange(950) * 50.0})
    merged = btc.merge(eth, on="timestamp", how="inner")
    assert len(merged) == 950


# ─── 8. np.polyfit 系数顺序回归锁 ───────────────────────────────────────────


def test_polyfit_returns_slope_then_intercept():
    """y = 2*x + 5 → polyfit(x,y,1) == [2.0, 5.0]，beta, alpha = ... 解构正确。"""
    x = np.arange(100, dtype=float)
    y = 2.0 * x + 5.0
    coefs = np.polyfit(x, y, 1)
    assert abs(coefs[0] - 2.0) < 1e-6, f"polyfit[0] (slope) = {coefs[0]}"
    assert abs(coefs[1] - 5.0) < 1e-6, f"polyfit[1] (intercept) = {coefs[1]}"
    beta, alpha = np.polyfit(x, y, 1)
    assert beta == coefs[0]
    assert alpha == coefs[1]


def test_compute_rolling_beta_recovers_known_relation():
    """已知 log_eth = 1.5*log_btc + 0.7 + tiny_noise 时，β ≈ 1.5、α ≈ 0.7。"""
    df = _make_synthetic_pair(n=300, beta_true=1.5, alpha_true=0.7)
    out = compute_rolling_beta(df, window=100)
    assert abs(out["beta"].iloc[-1] - 1.5) < 0.05
    assert abs(out["alpha"].iloc[-1] - 0.7) < 0.05


# ─── 9. 手续费汇总公式 ──────────────────────────────────────────────────────


def test_fee_no_price_change():
    """β=1.0、价格不变 → trade_pnl == -fee_total = -(D_eth+D_btc) × commission × 2。"""
    info = compute_trade_pnl(
        direction=+1, beta_open=1.0,
        eth_open=2000.0, btc_open=40000.0,
        eth_close=2000.0, btc_close=40000.0,
        d_total=5000.0, commission=0.0005,
    )
    expected_fee = 5.0
    assert abs(info["fee"] - expected_fee) < 1e-6
    assert abs(info["pnl_net"] - (-expected_fee)) < 1e-6


def test_fee_asymmetric_beta():
    """β=1.05 时 fee = (D_eth + D_btc) × 2 × commission = D_total × 2 × commission。"""
    info = compute_trade_pnl(
        direction=+1, beta_open=1.05,
        eth_open=2000.0, btc_open=40000.0,
        eth_close=2000.0, btc_close=40000.0,
        d_total=5000.0, commission=0.0005,
    )
    expected_fee = 5000.0 * 0.0005 * 2
    assert abs(info["fee"] - expected_fee) < 1e-6
