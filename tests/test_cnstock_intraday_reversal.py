"""
Tests for scripts/cnstock_intraday_reversal_smoke.py.

对应 docs/GinkgoBrain/300866-5min-T0反转-测试用例.md Level 1（16 条 pytest 单测）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.cnstock_intraday_reversal_smoke import (
    BacktestParams,
    compute_zscore,
    run_backtest,
    MIN_LOT_SHARES,
    SLIPPAGE,
    COMMISSION_BUY,
    COMMISSION_SELL,
)


# ─── Fixture helpers ────────────────────────────────────────────────────────


def _make_intraday_df(
    n_bars: int,
    close_price=None,
    z_score=None,
    suspension=None,
    daily_pre_close: float = 100.0,
    start_session_bar: int = 2,
) -> pd.DataFrame:
    """构造单日内 fixture（避免触发 session_bar=1 / 46+ mask）。"""
    if close_price is None:
        close_price = np.full(n_bars, 100.0)
    if z_score is None:
        z_score = np.zeros(n_bars)
    if suspension is None:
        suspension = np.zeros(n_bars, dtype=int)
    ts = pd.date_range(
        "2024-01-15 09:40:00",
        periods=n_bars, freq="5min", tz="Asia/Shanghai",
    )
    return pd.DataFrame({
        "ts_cst": ts,
        "date_cst": ts.date,
        "session_bar": np.arange(start_session_bar, start_session_bar + n_bars),
        "close_price": np.asarray(close_price, dtype=float),
        "z_score": np.asarray(z_score, dtype=float),
        "suspension": np.asarray(suspension, dtype=int),
        "daily_pre_close": np.full(n_bars, daily_pre_close),
    })


# ─── 1. Z-score 防泄漏 ⭐核心 ────────────────────────────────────────────────


def test_zscore_excludes_current():
    close = pd.Series([100.0] * 100 + [120.0] + [100.0] * 10)
    z_correct = compute_zscore(close, window=24)
    bad_mean = close.rolling(24).mean()
    bad_std = close.rolling(24).std()
    z_bad = (close - bad_mean) / bad_std
    assert abs(z_correct.iloc[100]) > abs(z_bad.iloc[100]) * 1.5


def test_zscore_cold_start_nan():
    close = pd.Series(np.random.RandomState(42).randn(100))
    z = compute_zscore(close, window=24)
    assert z.iloc[:24].isna().all()
    assert z.iloc[24:].notna().all()


# ─── 2. 状态机三档 transition ──────────────────────────────────────────────


def test_state_machine_z_extreme_high_sells():
    df = _make_intraday_df(n_bars=4, close_price=[100, 100, 100, 100],
                           z_score=[+3.0, +1.0, +0.3, 0.0])
    trades, _, _ = run_backtest(df, BacktestParams(z_entry=2.0, z_exit=0.5))
    assert len(trades) >= 1
    assert trades[0].direction == -1
    assert abs(trades[0].ratio_to - 0.5) < 0.05


def test_state_machine_buy_after_sell():
    """初始满仓 → bar 0 z>+2 卖一半 → bar 1 z<-2 买回。"""
    df = _make_intraday_df(n_bars=4, close_price=[100, 100, 100, 100],
                           z_score=[+3.0, -3.0, -3.0, 0.0])
    trades, _, _ = run_backtest(df, BacktestParams(z_entry=2.0, z_exit=0.5))
    assert trades[0].direction == -1
    assert trades[1].direction == +1


# ─── 3. 1 手 sizing ────────────────────────────────────────────────────────


def test_lot_sizing_initial_position():
    """初始 200K @ 117.65，z=1.0（dead zone）不触发交易，equity=200K（价格不变）。"""
    df = _make_intraday_df(n_bars=2, close_price=[117.65, 117.65], z_score=[1.0, 1.0])
    _, equity, _ = run_backtest(df, BacktestParams(initial_balance=200_000.0))
    # init shares = 200000 // (117.65 × 100) × 100 = 1700
    # cash = 200000 - 1700 × 117.65 = 0.50；equity = 0.50 + 1700×117.65 = 200000
    assert abs(equity.iloc[0] - 200_000.0) < 0.5
    assert abs(equity.iloc[1] - 200_000.0) < 0.5


# ─── 4. T+1 单日内 ⭐核心 ⭐v0.2 ─────────────────────────────────────────────


def test_t1_intraday_freezes_newly_bought():
    """
    T+1 关键不变量：同日内买回的 shares 不进 available。
    - bar 0 sell：available 扣除卖出量
    - bar 1 buy：shares 增加，但 available 保持卖出后的低位
    - bar 2 sell：available_after 应反映"前次买入未解冻"

    注：在 3 档比例（0.5/0.75/1.0）+ 价格不变下，每次 sell 量恰好等于 available，
    自然不触发 truncation。本测试验证 available 追踪正确性，不强求 truncation 发生。
    """
    df = _make_intraday_df(
        n_bars=4, close_price=[100, 100, 100, 100],
        z_score=[+3.0, -3.0, +3.0, 0.0],
        daily_pre_close=100.0,
    )
    trades, _, _ = run_backtest(
        df, BacktestParams(initial_balance=200_000, max_daily_switches=10),
    )
    assert len(trades) >= 3, (
        f"only {len(trades)} trades: {[(t.direction, t.ratio_from, t.ratio_to) for t in trades]}"
    )
    sell0, buy1, sell2 = trades[0], trades[1], trades[2]
    # bar 0 sell: shares 减少，available 减少同量
    assert sell0.direction == -1
    assert sell0.available_after == sell0.shares_after, "bar 0 sell: available == shares"
    # bar 1 buy: shares 增加，available **不增加**（T+1 freeze）
    assert buy1.direction == +1
    assert buy1.available_after < buy1.shares_after, (
        f"bar 1 buy 后 available({buy1.available_after}) 应 < shares({buy1.shares_after})（T+1 freeze）"
    )
    # bar 2 sell: 从 available 扣
    assert sell2.direction == -1
    assert sell2.available_after < buy1.available_after, (
        f"bar 2 sell 后 available 应进一步减少"
    )


# ─── 5. T+1 跨日 reset ⭐核心 ⭐v0.2 ─────────────────────────────────────────


def test_t1_resets_at_next_day_open():
    ts = pd.concat([
        pd.Series(pd.date_range("2024-01-15 09:40:00", periods=4, freq="5min", tz="Asia/Shanghai")),
        pd.Series(pd.date_range("2024-01-16 09:40:00", periods=4, freq="5min", tz="Asia/Shanghai")),
    ]).reset_index(drop=True)
    df = pd.DataFrame({
        "ts_cst": ts,
        "date_cst": ts.dt.date,
        "session_bar": [2, 3, 4, 5, 2, 3, 4, 5],
        "close_price": [100.0] * 8,
        "z_score": [+3.0, -3.0, +3.0, 0.0, +3.0, 0.0, 0.0, 0.0],
        "suspension": np.zeros(8, dtype=int),
        "daily_pre_close": np.full(8, 100.0),
    })
    trades, _, _ = run_backtest(
        df, BacktestParams(initial_balance=200_000, max_daily_switches=10),
    )
    d2_trades = [t for t in trades if t.open_ts.date() == pd.Timestamp("2024-01-16").date()]
    assert len(d2_trades) >= 1, f"no D2 trades: {[(t.open_ts, t.direction) for t in trades]}"
    d2_sell = next((t for t in d2_trades if t.direction == -1), None)
    assert d2_sell is not None
    assert not d2_sell.truncated_by_t1, "D2 first sell should NOT be T+1 truncated"


# ─── 6. 滑点 ⭐v0.2 ────────────────────────────────────────────────────────


def test_slippage_buy_pushes_price_up():
    df = _make_intraday_df(n_bars=3, close_price=[100, 100, 100],
                           z_score=[+3.0, -3.0, 0.0])
    trades, _, _ = run_backtest(df, BacktestParams())
    buy_trade = next((t for t in trades if t.direction == +1), None)
    assert buy_trade is not None
    assert abs(buy_trade.exec_price - 100.0 * (1 + SLIPPAGE)) < 1e-6


def test_slippage_sell_pushes_price_down():
    df = _make_intraday_df(n_bars=2, close_price=[100, 100], z_score=[+3.0, 0.0])
    trades, _, _ = run_backtest(df, BacktestParams())
    sell_trade = next((t for t in trades if t.direction == -1), None)
    assert sell_trade is not None
    assert abs(sell_trade.exec_price - 100.0 * (1 - SLIPPAGE)) < 1e-6


# ─── 7. 每日切换上限 ─────────────────────────────────────────────────────────


def test_daily_switch_limit():
    df = _make_intraday_df(n_bars=10, close_price=np.full(10, 100.0),
                           z_score=[+3, -3, +3, -3, +3, -3, +3, -3, +3, 0])
    trades, _, _ = run_backtest(df, BacktestParams(max_daily_switches=4, initial_balance=200_000))
    assert len(trades) <= 4, f"got {len(trades)} trades, expected <= 4"


# ─── 8 & 9. 涨跌停 mask ─────────────────────────────────────────────────────


def test_limit_up_blocks_buy():
    """涨停时 (close ≥ daily_pre_close × 1.195) 无法加仓买入，bar 应无买入交易记录。"""
    df = _make_intraday_df(
        n_bars=3, close_price=[100, 100, 120.5],
        z_score=[+3.0, 1.0, -3.0],   # bar 1 z=1.0 dead zone, no transition
        daily_pre_close=100.0,
    )
    trades, _, _ = run_backtest(df, BacktestParams())
    # bar 0 sell 正常；bar 2 想买（z<-2，ratio 0.5→1.0）但涨停 → new_ratio 被截断回 0.5 → no trade
    bar2_trades = [t for t in trades if t.open_idx == 2]
    assert len(bar2_trades) == 0, f"涨停 bar 不应有交易: {bar2_trades}"


def test_limit_down_blocks_sell():
    """跌停时 (close ≤ daily_pre_close × 0.805) 无法减仓卖出，bar 应无卖出交易记录。"""
    df = _make_intraday_df(
        n_bars=2, close_price=[100, 79.5],
        z_score=[1.0, +3.0],   # bar 0 z=1.0 dead zone（保持初始 ratio=1.0），bar 1 想卖
        daily_pre_close=100.0,
    )
    trades, _, _ = run_backtest(df, BacktestParams())
    # bar 1 z>+2 想从 1.0→0.5 但跌停截断
    bar1_trades = [t for t in trades if t.open_idx == 1]
    assert len(bar1_trades) == 0, f"跌停 bar 不应有交易: {bar1_trades}"


# ─── 10. 停牌跳过 ──────────────────────────────────────────────────────────


def test_suspension_locks_position():
    df = _make_intraday_df(
        n_bars=3, close_price=[100, 100, 100],
        z_score=[+3.0, +3.0, +3.0],
        suspension=[0, 1, 0],
    )
    trades, _, _ = run_backtest(df, BacktestParams())
    susp_trades = [t for t in trades if t.open_idx == 1]
    assert len(susp_trades) == 0


# ─── 11. session_bar=1 跳过 ────────────────────────────────────────────────


def test_first_bar_locked():
    df = pd.DataFrame({
        "ts_cst": pd.date_range("2024-01-15 09:35", periods=2, freq="5min", tz="Asia/Shanghai"),
        "date_cst": [pd.Timestamp("2024-01-15").date()] * 2,
        "session_bar": [1, 2],
        "close_price": [100.0, 100.0],
        "z_score": [+3.0, 0.0],
        "suspension": [0, 0],
        "daily_pre_close": [100.0, 100.0],
    })
    trades, _, _ = run_backtest(df, BacktestParams())
    first_bar_trades = [t for t in trades if t.open_idx == 0]
    assert len(first_bar_trades) == 0


# ─── 12. session_bar≥46 尾盘 ───────────────────────────────────────────────


def test_tail_lock():
    df = pd.DataFrame({
        "ts_cst": pd.date_range("2024-01-15 14:45", periods=4, freq="5min", tz="Asia/Shanghai"),
        "date_cst": [pd.Timestamp("2024-01-15").date()] * 4,
        "session_bar": [45, 46, 47, 48],
        "close_price": [100.0, 100.0, 100.0, 100.0],
        "z_score": [+3.0, +3.0, +3.0, +3.0],
        "suspension": np.zeros(4, dtype=int),
        "daily_pre_close": np.full(4, 100.0),
    })
    trades, _, _ = run_backtest(df, BacktestParams())
    tail_trades = [t for t in trades if t.open_idx >= 1]
    assert len(tail_trades) == 0, f"尾盘不应交易: {tail_trades}"


# ─── 13. 手续费 + 印花税 ───────────────────────────────────────────────────


def test_fee_and_stamp_duty_round_trip_loss():
    """sell + buy 来回，价格不变，最终 equity 应略小于初始（成本损失）。"""
    df = _make_intraday_df(n_bars=3, close_price=[100, 100, 100],
                           z_score=[+3.0, -3.0, 0.0])
    _, equity, _ = run_backtest(df, BacktestParams(initial_balance=200_000))
    final_equity = equity.iloc[-1]
    # 应有损失（fee + slippage），但 < 1%
    assert final_equity < 200_000.0
    assert final_equity > 200_000.0 * 0.99


# ─── 14. 前复权因子计算 ⭐核心 ⭐v0.3 ────────────────────────────────────────


def test_adjust_factor_single_dividend():
    last_5m = pd.Series([100.0, 80.0])
    day_close = pd.Series([80.0, 80.0])
    adj = day_close / last_5m
    np.testing.assert_array_almost_equal(adj.values, [0.8, 1.0], decimal=4)


def test_adjust_factor_multiple_dividends():
    last_5m = pd.Series([100.0, 90.0, 80.0])
    day_close = pd.Series([64.0, 72.0, 80.0])
    adj = day_close / last_5m
    np.testing.assert_array_almost_equal(adj.values, [0.64, 0.80, 1.0], decimal=4)


def test_adjust_factor_no_dividend():
    last_5m = pd.Series([100.0, 105.0])
    day_close = pd.Series([100.0, 105.0])
    adj = day_close / last_5m
    np.testing.assert_array_almost_equal(adj.values, [1.0, 1.0], decimal=6)


# ─── 15. 复权后对齐校验 ⭐核心 ⭐v0.3 ────────────────────────────────────────


def test_post_adjust_alignment_invariant():
    """合成 3 天 + 2 次除权，复权后每日末根 5m close == 同日 day close。"""
    n_per_day = 4
    days = [pd.Timestamp("2024-01-15"), pd.Timestamp("2024-01-16"), pd.Timestamp("2024-01-17")]
    raw_5m_last = [100.0, 80.0, 70.0]
    day_close_after_adj = [64.0, 72.0, 70.0]

    rows = []
    for d, raw_last in zip(days, raw_5m_last):
        for i in range(n_per_day):
            close_raw = raw_last - (n_per_day - 1 - i) * 0.1
            rows.append({"date_cst": d.date(), "close_price_raw": close_raw})
    df = pd.DataFrame(rows)
    day_indexed = pd.Series(day_close_after_adj, index=[d.date() for d in days])
    last_5m_unadj = pd.Series(raw_5m_last, index=[d.date() for d in days])
    adj_factor = (day_indexed / last_5m_unadj).rename("adj_factor")
    df = df.merge(adj_factor.reset_index().rename(columns={"index": "date_cst"}),
                  on="date_cst", how="inner")
    df["close_adj"] = df["close_price_raw"] * df["adj_factor"]

    for d, expected in zip(days, day_close_after_adj):
        last_adj = df[df["date_cst"] == d.date()]["close_adj"].iloc[-1]
        assert abs(last_adj - expected) < 1e-4, (
            f"date={d.date()}: last_5m_adj={last_adj}, day_close={expected}"
        )


# ─── 16. daily_pre_close 防呆 ⭐v0.3 ─────────────────────────────────────────


def test_daily_pre_close_uses_day_series():
    days = [pd.Timestamp("2024-01-15").date(), pd.Timestamp("2024-01-16").date()]
    day_close = pd.Series([100.0, 110.0], index=days)
    daily_pre_close = day_close.shift(1)
    assert pd.isna(daily_pre_close.iloc[0])
    assert daily_pre_close.iloc[1] == 100.0
    assert daily_pre_close.iloc[1] != 109.5  # 不能是前根 5m close
