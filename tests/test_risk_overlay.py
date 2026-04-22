"""Tests for add_risk_overlay — regime_resample extension.

对应 docs/GinkgoBrain/5min-ETH-趋势跟随-测试用例.md Level 1 测试 1–4。

设计契约：
- 新增 regime_resample: str = "1D" 可选参数
- 不传（或显式 "1D"）时行为与改造前字节一致
- 传 "1h" 时按 1h bar 做 gate resample；regime_ma_days 变为"bars of resample"语义
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.backtest_gbdt import add_risk_overlay


# ─────────────────────────────────────────── fixtures

def _make_4h_fixture(n_bars: int = 12 * 365, seed: int = 42) -> pd.DataFrame:
    """2 年合成 4h K 线（6 根/天 × 365 × 2 ≈ 4380 根的近似，实际 12/天=2×365.5）。"""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2022-01-01", periods=n_bars, freq="4h", tz="UTC")
    price = 60_000 + np.cumsum(rng.normal(0, 200, n_bars))
    price = np.maximum(price, 1_000)
    return pd.DataFrame({
        "timestamp": ts,
        "open":  price * rng.uniform(0.995, 1.0, n_bars),
        "high":  price * rng.uniform(1.0, 1.01, n_bars),
        "low":   price * rng.uniform(0.99, 1.0, n_bars),
        "close": price,
        "volume": rng.uniform(100, 1000, n_bars),
    })


def _make_5min_fixture(n_hours: int = 200, trend: str = "up") -> pd.DataFrame:
    """n_hours * 12 根 5min bar，close 单调（up / down / flat）+ 小噪声。"""
    rng = np.random.default_rng(0)
    n = n_hours * 12
    ts = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    slope = {"up": 5.0, "down": -5.0, "flat": 0.0}[trend]
    close = 3_000 + slope * np.arange(n) + rng.normal(0, 10, n)
    close = np.maximum(close, 100)
    return pd.DataFrame({
        "timestamp": ts,
        "open":  close,
        "high":  close * 1.001,
        "low":   close * 0.999,
        "close": close,
        "volume": rng.uniform(1, 10, n),
    })


# ─────────────────────────────────────────── 1. 向下兼容

def test_backward_compatible_default_matches_explicit_1D():
    """不传 regime_resample 与显式传 '1D' 结果完全一致 — 证明 BTC 4h 旧调用路径零影响。"""
    df = _make_4h_fixture()
    out_old_style = add_risk_overlay(df, "4h", vol_window=168, regime_ma_days=200)
    out_explicit  = add_risk_overlay(df, "4h", vol_window=168, regime_ma_days=200,
                                     regime_resample="1D")
    pd.testing.assert_frame_equal(out_old_style, out_explicit)


def test_default_output_columns_non_null_after_warmup():
    """默认路径产出 regime_ok 和 realized_vol_ann 两列，warmup 后非空。"""
    df = _make_4h_fixture()
    out = add_risk_overlay(df, "4h", vol_window=168, regime_ma_days=200)
    assert "regime_ok" in out.columns
    assert "realized_vol_ann" in out.columns
    # realized_vol_ann 经 bfill + fillna(0.6)，不应有 NaN
    assert out["realized_vol_ann"].notna().all()
    # regime_ok 为 int 值（0/1）
    assert out["regime_ok"].isin([0, 1]).all()


# ─────────────────────────────────────────── 2. 1h gate 路径

def test_1h_resample_uptrend_gate_all_on():
    """100 小时单调上涨（warmup 之外部分），1h close 始终 > SMA50，regime_ok=1 占主导。"""
    df = _make_5min_fixture(n_hours=200, trend="up")
    out = add_risk_overlay(df, "5min", vol_window=288, regime_ma_days=50,
                           regime_resample="1h")
    # 前 50 小时是 warmup（rolling 未填满），regime_ok 允许为 0
    # 后半段（warmup 之后）所有 5min bar 的 regime_ok 应全为 1
    post_warmup_cut = df["timestamp"].iloc[0] + pd.Timedelta("55h")
    post = out[out["timestamp"] >= post_warmup_cut]
    assert (post["regime_ok"] == 1).all(), \
        f"uptrend warmup 后 regime_ok 应全为 1，实际有 {(post['regime_ok']==0).sum()} 个 0"


def test_1h_resample_downtrend_gate_all_off():
    """100 小时单调下跌，warmup 后 regime_ok 全为 0。"""
    df = _make_5min_fixture(n_hours=200, trend="down")
    out = add_risk_overlay(df, "5min", vol_window=288, regime_ma_days=50,
                           regime_resample="1h")
    post_warmup_cut = df["timestamp"].iloc[0] + pd.Timedelta("55h")
    post = out[out["timestamp"] >= post_warmup_cut]
    assert (post["regime_ok"] == 0).all(), \
        f"downtrend warmup 后 regime_ok 应全为 0，实际有 {(post['regime_ok']==1).sum()} 个 1"


def test_1h_resample_warmup_period_zero():
    """warmup 段（前 50 小时未填满 SMA50）的 regime_ok 应为 0（NaN→ffill→0）。"""
    df = _make_5min_fixture(n_hours=200, trend="up")
    out = add_risk_overlay(df, "5min", vol_window=288, regime_ma_days=50,
                           regime_resample="1h")
    # 最早 5 分钟：第一根 1h bar 还没算出 SMA50，regime_ok 必然是 0
    first_bar = out.iloc[0]
    assert first_bar["regime_ok"] == 0


def test_1h_resample_same_regime_within_hour():
    """同一个 1h 内的 12 根 5min bar，regime_ok 必须相同（由 ffill 保证）。"""
    df = _make_5min_fixture(n_hours=200, trend="up")
    out = add_risk_overlay(df, "5min", vol_window=288, regime_ma_days=50,
                           regime_resample="1h")
    # 按小时 group，每组 regime_ok 应只有一个独特值
    out["hour"] = out["timestamp"].dt.floor("h")
    nunique_per_hour = out.groupby("hour")["regime_ok"].nunique()
    assert (nunique_per_hour == 1).all(), \
        f"同一小时内 regime_ok 出现多值的小时数：{(nunique_per_hour != 1).sum()}"


# ─────────────────────────────────────────── 3. resample 值边界

def test_resample_15min_with_ma_bars_20():
    """resample='15min' + regime_ma_days=20 应能跑通并产出合法 regime_ok。"""
    df = _make_5min_fixture(n_hours=100, trend="up")
    out = add_risk_overlay(df, "5min", vol_window=288, regime_ma_days=20,
                           regime_resample="15min")
    assert out["regime_ok"].isin([0, 1]).all()
    # warmup 后应为 1（上涨）
    post_warmup_cut = df["timestamp"].iloc[0] + pd.Timedelta("6h")  # 20*15min=5h
    post = out[out["timestamp"] >= post_warmup_cut]
    assert (post["regime_ok"] == 1).all()


def test_resample_invalid_raises():
    """非法 resample 字符串应让 pandas 抛 ValueError（不做额外包装）。"""
    df = _make_5min_fixture(n_hours=50)
    with pytest.raises(ValueError):
        add_risk_overlay(df, "5min", vol_window=288, regime_ma_days=50,
                         regime_resample="1x")
