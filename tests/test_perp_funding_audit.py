"""
Tests for scripts/perp_funding_audit.py.

对应 docs/GinkgoBrain/BTC4h-Perp资金费率审计-测试用例.md Level 1（10 条 pytest 单测）：

1. Funding-then-rebalance 顺序：满仓避费 ⭐ 核心微观结构修正
2. Funding-then-rebalance 顺序：开仓延迟 ⭐
3. Funding 边界识别（hour % 8 == 0）
4. Funding 公式正确性
5. Direction sign（long-only 收负费率）
6. 累计正确性（multi-bar）
7. Funding 与 commission 独立累加
8. Pos=0 即使在边界也不扣费
9. MTM 用 post-rebalance pos
10. No-funding 模式回归与 backtest_ppo_overlay.py:simulate() 一致
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.perp_funding_audit import simulate_with_funding


# ─── 1. Funding-then-rebalance: 满仓避费 ⭐核心 ───────────────────────────────


def test_funding_charges_pre_trade_position_long_to_flat():
    """
    进入 bar 0 时 pos=1.0（前根满仓），bar 0 在 funding 边界 + tgt=0（要平仓避费）。
    funding 必须按 pos=1.0 扣费，不能按 0 扣（v0.1 错误实现的 bug）。
    """
    closes = np.array([100.0, 100.0])
    raw_ratios = np.array([0.0, 0.0])
    realized_vol = np.array([0.01, 0.01])
    regime_ok = np.array([1, 1])
    funding_rates = np.array([0.001, 0.0])
    initial_pos = 1.0

    equity, funding_paid, _ = simulate_with_funding(
        closes, raw_ratios, realized_vol, regime_ok, funding_rates,
        vol_target=0.0, use_regime_gate=False,
        rebalance_threshold=0.01, commission=0.0,
        use_funding=True, initial_balance=10000.0, initial_pos=initial_pos,
    )
    expected_funding = 10000.0 * 1.0 * 0.001
    assert abs(funding_paid[0] - expected_funding) < 1e-6, (
        f"funding={funding_paid[0]:.4f}, expected={expected_funding:.4f}（用 pre-trade pos=1.0）"
    )


# ─── 2. Funding-then-rebalance: 开仓延迟 ⭐核心 ──────────────────────────────


def test_funding_skips_post_trade_position_flat_to_long():
    """
    进入 bar 0 时 pos=0（前根空仓），bar 0 在 funding 边界 + tgt=1.0（开仓）。
    funding 必须为 0（pre-trade pos=0），不能按新仓位 1.0 扣（v0.1 错误）。
    """
    closes = np.array([100.0, 100.0])
    raw_ratios = np.array([1.0, 1.0])
    realized_vol = np.array([0.01, 0.01])
    regime_ok = np.array([1, 1])
    funding_rates = np.array([0.001, 0.0])

    _, funding_paid, n_rebal = simulate_with_funding(
        closes, raw_ratios, realized_vol, regime_ok, funding_rates,
        vol_target=0.0, use_regime_gate=False,
        rebalance_threshold=0.01, commission=0.0,
        use_funding=True, initial_balance=10000.0, initial_pos=0.0,
    )
    assert funding_paid[0] == 0.0, (
        f"funding={funding_paid[0]:.4f}, 应为 0（pre-trade pos=0）"
    )
    assert n_rebal == 1   # 仍正常调仓到满仓


# ─── 3. Funding 边界识别（外部驱动 funding_rates） ───────────────────────────


def test_funding_boundary_detection_only_at_hour_mod_8_zero():
    """
    模拟 load_funding_aligned 输出：仅在 idx 0/2/4 (= 00/08/16 boundary) 有非零 rate。
    模拟器对所有 funding_rates ≠ 0 的 bar 都扣费。
    """
    n = 6
    closes = np.full(n, 100.0)
    raw_ratios = np.full(n, 1.0)
    realized_vol = np.full(n, 0.01)
    regime_ok = np.ones(n, dtype=int)
    funding_rates = np.array([0.001, 0.0, 0.001, 0.0, 0.001, 0.0])

    _, funding_paid, _ = simulate_with_funding(
        closes, raw_ratios, realized_vol, regime_ok, funding_rates,
        vol_target=0.0, use_regime_gate=False,
        rebalance_threshold=0.01, commission=0.0,
        use_funding=True, initial_balance=10000.0, initial_pos=1.0,
    )
    nonzero_idx = np.where(funding_paid != 0.0)[0].tolist()
    assert nonzero_idx == [0, 2, 4], f"funding 在 idx={nonzero_idx}（应为 [0, 2, 4]）"


# ─── 4. Funding 公式正确性 ─────────────────────────────────────────────────


@pytest.mark.parametrize("pos,rate,expected_cost", [
    (1.0, 0.001, 10.0),     # 满仓正费率
    (0.5, 0.001, 5.0),      # 半仓正费率
    (1.0, -0.0005, -5.0),   # 满仓负费率（多头收钱）
    (1.0, 0.01, 100.0),     # 极端正费率
])
def test_funding_cost_formula(pos, rate, expected_cost):
    """对单根 bar 验证 cost = portfolio × pos × rate。"""
    closes = np.array([100.0, 100.0])
    raw_ratios = np.array([pos, pos])
    realized_vol = np.full(2, 0.01)
    regime_ok = np.ones(2, dtype=int)
    funding_rates = np.array([rate, 0.0])

    _, funding_paid, _ = simulate_with_funding(
        closes, raw_ratios, realized_vol, regime_ok, funding_rates,
        vol_target=0.0, use_regime_gate=False,
        rebalance_threshold=0.01, commission=0.0,
        use_funding=True, initial_balance=10000.0, initial_pos=pos,
    )
    assert abs(funding_paid[0] - expected_cost) < 1e-6


def test_funding_zero_position_no_cost():
    """空仓即使在 funding 边界也不扣费。"""
    closes = np.array([100.0, 100.0])
    raw_ratios = np.array([0.0, 0.0])
    realized_vol = np.full(2, 0.01)
    regime_ok = np.ones(2, dtype=int)
    funding_rates = np.array([0.001, 0.0])

    _, funding_paid, _ = simulate_with_funding(
        closes, raw_ratios, realized_vol, regime_ok, funding_rates,
        vol_target=0.0, use_regime_gate=False,
        rebalance_threshold=0.01, commission=0.0,
        use_funding=True, initial_balance=10000.0, initial_pos=0.0,
    )
    assert funding_paid[0] == 0.0


# ─── 5. Direction sign：long-only 收负费率 ─────────────────────────────────


def test_long_receives_negative_funding():
    """
    全程 funding rate=-0.0005（熊市 short 付 long）+ pos=1.0（多头）
    → funding_paid 全部 < 0（多头收钱）→ portfolio 净增长（无价格变动情况下）
    """
    n = 4
    closes = np.full(n, 100.0)
    raw_ratios = np.full(n, 1.0)
    realized_vol = np.full(n, 0.01)
    regime_ok = np.ones(n, dtype=int)
    funding_rates = np.array([-0.0005, 0.0, -0.0005, 0.0])

    equity, funding_paid, _ = simulate_with_funding(
        closes, raw_ratios, realized_vol, regime_ok, funding_rates,
        vol_target=0.0, use_regime_gate=False,
        rebalance_threshold=0.01, commission=0.0,
        use_funding=True, initial_balance=10000.0, initial_pos=1.0,
    )
    assert funding_paid.sum() < 0
    assert equity[-1] > 10000.0


# ─── 6. 累计正确性 ──────────────────────────────────────────────────────────


def test_funding_cumulative_3_events():
    """3 个 funding 边界，全程 pos=1.0，rate=0.001 → 累计 ≈ $30（含复利略小）。"""
    n = 6
    closes = np.full(n, 100.0)
    raw_ratios = np.full(n, 1.0)
    realized_vol = np.full(n, 0.01)
    regime_ok = np.ones(n, dtype=int)
    funding_rates = np.array([0.001, 0.0, 0.001, 0.0, 0.001, 0.0])

    _, funding_paid, _ = simulate_with_funding(
        closes, raw_ratios, realized_vol, regime_ok, funding_rates,
        vol_target=0.0, use_regime_gate=False,
        rebalance_threshold=0.01, commission=0.0,
        use_funding=True, initial_balance=10000.0, initial_pos=1.0,
    )
    total = funding_paid.sum()
    assert 29.5 < total < 30.0, f"累计 funding={total:.4f}，期望 [29.5, 30.0)"


# ─── 7. Funding 与 commission 独立累加 ───────────────────────────────────────


def test_funding_then_commission_compose():
    """
    bar 0：进入 pos=1.0，tgt=0.5，funding 边界 rate=0.001。
    顺序：funding(pos=1.0, $10) → rebalance(delta=0.5, $4.995) → MTM($0)
    portfolio: 10000 → 9990 → 9985.005
    """
    closes = np.array([100.0, 100.0])
    raw_ratios = np.array([0.5, 0.5])
    realized_vol = np.full(2, 0.01)
    regime_ok = np.ones(2, dtype=int)
    funding_rates = np.array([0.001, 0.0])

    equity, funding_paid, n_rebal = simulate_with_funding(
        closes, raw_ratios, realized_vol, regime_ok, funding_rates,
        vol_target=0.0, use_regime_gate=False,
        rebalance_threshold=0.01, commission=0.001,
        use_funding=True, initial_balance=10000.0, initial_pos=1.0,
    )
    assert abs(funding_paid[0] - 10.0) < 1e-6
    assert abs(equity[0] - 9985.005) < 1e-3
    assert n_rebal == 1


# ─── 8. Pos=0 多 bar 不扣费 ─────────────────────────────────────────────────


def test_zero_position_multi_bar_no_funding():
    """全程空仓 + 多个 funding 边界，funding_paid 应全 0。"""
    n = 6
    closes = np.full(n, 100.0)
    raw_ratios = np.zeros(n)
    realized_vol = np.full(n, 0.01)
    regime_ok = np.ones(n, dtype=int)
    funding_rates = np.array([0.001, 0.0, 0.001, 0.0, 0.001, 0.0])

    equity, funding_paid, n_rebal = simulate_with_funding(
        closes, raw_ratios, realized_vol, regime_ok, funding_rates,
        vol_target=0.0, use_regime_gate=False,
        rebalance_threshold=0.01, commission=0.001,
        use_funding=True, initial_balance=10000.0, initial_pos=0.0,
    )
    assert (funding_paid == 0.0).all()
    assert n_rebal == 0
    assert (equity == 10000.0).all()


# ─── 9. MTM 用 post-rebalance pos ──────────────────────────────────────────


def test_mtm_uses_post_rebalance_position():
    """
    bar 0: pos=0 → tgt=1.0（非 funding 边界）+ ret 0→1: +5%
    portfolio: 10000 - commission(delta=1.0, $10) = 9990 → MTM(+5%) = 10489.5
    """
    closes = np.array([100.0, 105.0, 105.0])
    raw_ratios = np.array([1.0, 1.0, 1.0])
    realized_vol = np.full(3, 0.01)
    regime_ok = np.ones(3, dtype=int)
    funding_rates = np.zeros(3)

    equity, _, _ = simulate_with_funding(
        closes, raw_ratios, realized_vol, regime_ok, funding_rates,
        vol_target=0.0, use_regime_gate=False,
        rebalance_threshold=0.01, commission=0.001,
        use_funding=True, initial_balance=10000.0, initial_pos=0.0,
    )
    assert abs(equity[0] - 10489.5) < 1e-3, f"equity[0]={equity[0]}"


# ─── 10. No-funding 模式回归 ⭐核心 ────────────────────────────────────────


def test_no_funding_mode_zero_funding_paid():
    """use_funding=False 时无论 funding_rates 多大，funding_paid 必须全 0。"""
    n = 10
    closes = np.linspace(100.0, 110.0, n)
    raw_ratios = np.full(n, 0.5)
    realized_vol = np.full(n, 0.02)
    regime_ok = np.ones(n, dtype=int)
    funding_rates = np.full(n, 0.005)

    _, funding_paid, _ = simulate_with_funding(
        closes, raw_ratios, realized_vol, regime_ok, funding_rates,
        vol_target=0.01, use_regime_gate=True,
        rebalance_threshold=0.01, commission=0.001,
        use_funding=False, initial_balance=10000.0, initial_pos=0.0,
    )
    assert (funding_paid == 0.0).all(), "use_funding=False 时 funding_paid 必须全 0"


def test_no_funding_matches_overlay_baseline():
    """
    use_funding=False + initial_balance=10000 + initial_pos=0 时，
    simulate_with_funding 输出应与 backtest_ppo_overlay.simulate() 严格一致。
    设计 §6 工程边界契约：新加 funding 模块不能改变 use_funding=False 时的行为。
    """
    from scripts.backtest_ppo_overlay import simulate as overlay_simulate

    np.random.seed(42)
    n = 50
    closes = 100 * np.cumprod(1 + np.random.randn(n) * 0.01)
    raw_ratios = np.random.choice([0.0, 0.25, 0.5, 0.75, 1.0], n)
    realized_vol = np.full(n, 0.02)
    regime_ok = np.random.choice([0, 1], n).astype(int)
    funding_rates = np.zeros(n)

    common = dict(
        closes=closes, raw_ratios=raw_ratios, realized_vol=realized_vol,
        regime_ok=regime_ok, vol_target=0.01, use_regime_gate=True,
        rebalance_threshold=0.01, commission=0.001,
    )
    eq_baseline, _, n_rebal_baseline = overlay_simulate(**common)
    eq_audit, _, n_rebal_audit = simulate_with_funding(
        funding_rates=funding_rates, use_funding=False,
        initial_balance=10000.0, initial_pos=0.0, **common,
    )
    np.testing.assert_array_almost_equal(eq_baseline, eq_audit, decimal=8)
    assert n_rebal_baseline == n_rebal_audit
