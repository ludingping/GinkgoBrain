"""strategies/cn_a_trend_base — 成本、宇宙掩码、等权指数（2026-09-21）。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategies.cn_a_trend_base import costs
from strategies.cn_a_trend_base.ew_index import ew_index, summarize
from strategies.cn_a_trend_base.universe import listing_age, st_codes, universe_mask


def test_costs_stamp_switch_and_limits() -> None:
    assert costs.stamp_tax("2023-08-27") == pytest.approx(0.001)
    assert costs.stamp_tax("2023-08-28") == pytest.approx(0.0005)
    assert costs.sell_cost("2024-01-02") == pytest.approx(0.00025 + 0.001 + 0.0005)
    assert costs.limit_threshold("600519") == pytest.approx(0.098)
    assert costs.limit_threshold("300866") == pytest.approx(0.198)
    assert costs.limit_threshold("688981") == pytest.approx(0.198)
    op = pd.DataFrame({"600519": [110.0, 105.0], "300866": [119.0, 120.0]})
    pc = pd.DataFrame({"600519": [100.0, 100.0], "300866": [100.0, 100.0]})
    lu = costs.limit_up_open(op, pc)
    assert lu["600519"].tolist() == [True, False]
    assert lu["300866"].tolist() == [False, True]


def _frames():
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    close = pd.DataFrame(100.0, index=dates, columns=["A", "B", "NEW", "ST1"])
    close.loc[dates[:25], "NEW"] = np.nan                       # 上市晚
    amount = pd.DataFrame(1e8, index=dates, columns=close.columns)
    amount["B"] = 1e7                                           # 不够流动
    susp = pd.DataFrame(0, index=dates, columns=close.columns)
    susp.loc[dates[10], "A"] = 1
    amount.loc[dates[10], "A"] = 0.0
    return dates, close, amount, susp


def test_universe_mask_rules() -> None:
    dates, close, amount, susp = _frames()
    names = pd.Series({"A": "平安银行", "B": "贵州茅台", "NEW": "新股", "ST1": "*ST海润"})
    st = st_codes(names)
    assert st == {"ST1"}
    m = universe_mask(close, amount, susp, st, min_listing_days=3)
    assert m.loc[dates[0], "A"]                                  # 窗口起点已在市 → 视为已过次新期
    assert not m["B"].any()                                     # 流动性
    assert not m["ST1"].any()                                   # ST
    assert not m.loc[dates[10], "A"] and m.loc[dates[9], "A"]   # 停牌日剔除
    assert not m.loc[dates[26], "NEW"] and m.loc[dates[29], "NEW"]   # 上市第 4 个交易日起（age ≥ 3）
    age = listing_age(close, seasoned_at_start=None)
    assert np.isnan(age.loc[dates[0], "NEW"]) and age.loc[dates[25], "NEW"] == 0
    assert listing_age(close).loc[dates[0], "A"] == 250


def test_ew_index_daily_rebalance_math_and_turnover() -> None:
    dates = pd.date_range("2024-01-01", periods=4, freq="B")
    close = pd.DataFrame({"A": [100, 110, 110, 110], "B": [100, 90, 90, 90]}, index=dates, dtype=float)
    mask = pd.DataFrame(True, index=dates, columns=["A", "B"])
    mask.loc[dates[2]:, "B"] = False                            # B 从第 3 天起出宇宙
    out = ew_index(close, mask, rebalance=None)
    assert out["ret_gross"].iloc[0] == 0.0                      # 首日无持仓
    assert out["ret_gross"].iloc[1] == pytest.approx(0.0)       # 50/50 × (+10%, −10%)
    assert out["n"].tolist() == [0, 2, 2, 1]
    assert out["turnover"].iloc[1] == pytest.approx(1.0)        # 第 2 天开仓 50/50
    # 第 3 天：漂移 A 0.55 / B 0.45 → 目标 50/50 → 换手 0.10
    assert out["turnover"].iloc[2] == pytest.approx(0.10)
    # 第 4 天：目标 A 100% → 换手 1.0；成本 = 0.5 × (buy + sell)
    assert out["turnover"].iloc[3] == pytest.approx(1.0)
    assert out["cost"].iloc[3] == pytest.approx(0.5 * (costs.buy_cost() + costs.sell_cost(dates[3])))
    s = summarize(out["idx_gross"])
    assert set(s) >= {"total_return", "max_drawdown", "calmar", "sharpe"}


def test_ew_index_ignores_zero_prices() -> None:
    dates = pd.date_range("2024-01-01", periods=4, freq="B")
    close = pd.DataFrame({"A": [100, 0.0, 110, 110], "B": [100, 100, 100, 100]}, index=dates, dtype=float)
    mask = pd.DataFrame(True, index=dates, columns=["A", "B"])
    out = ew_index(close, mask)
    assert np.isfinite(out["idx_net"]).all()
    assert out["ret_gross"].iloc[1] == 0.0 and out["ret_gross"].iloc[2] == 0.0


def test_ew_index_monthly_rebalance_drifts_between() -> None:
    dates = pd.bdate_range("2024-01-29", "2024-02-06")          # 跨月：2 月首个交易日 = 02-01
    close = pd.DataFrame({"A": np.linspace(100, 150, len(dates)), "B": 100.0}, index=dates)
    mask = pd.DataFrame(True, index=dates, columns=["A", "B"])
    out = ew_index(close, mask, rebalance="M")
    rb_days = out.index[out["turnover"] > 0]
    assert list(rb_days) == [dates[1], pd.Timestamp("2024-02-01")]   # 首个可持仓日 + 月初
    # 月内不换手，权重漂移：净值 = 毛值（无成本）
    mid = out.loc[dates[2]:pd.Timestamp("2024-01-31")]
    assert out["n"].iloc[1] == 2                                  # 首次建仓不等月初
    assert (mid["turnover"] == 0).all() and (mid["cost"] == 0).all()


# --- xs_probe (E-A2) ---

def test_xs_probe_features_and_ic_sign() -> None:
    from strategies.cn_a_trend_base.xs_probe import (daily_rank_ic, decile_returns, fold_stats,
                                                     forward_open_return, momentum, nw_tstat, realized_vol, reversal, verdict)
    rng = np.random.default_rng(0)
    dates = pd.date_range("2017-01-01", periods=400, freq="B")
    n = 300
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.02, (len(dates), n)), axis=0)),
                         index=dates, columns=[f"{600000 + i}" for i in range(n)])
    open_px = close.shift(1).fillna(100.0)
    mask = pd.DataFrame(True, index=dates, columns=close.columns)
    fwd = forward_open_return(open_px, 5)
    assert fwd.iloc[0, 0] == pytest.approx(open_px.iloc[6, 0] / open_px.iloc[1, 0] - 1)
    # 特征 = 未来收益本身 → IC ≈ +1；取负 → −1
    ic = daily_rank_ic(fwd, fwd, mask)
    assert ic.dropna().min() > 0.99
    ic_neg = daily_rank_ic(-fwd, fwd, mask)
    st = fold_stats(ic_neg, mask.sum(axis=1), {"f": ("2017-01-01", "2018-06-30")}, lags=5)
    assert st[0].ic_mean < -0.99 and st[0].ic_t < -10
    assert verdict(st, expected_sign=-1)[0] and not verdict(st, expected_sign=+1)[0]
    assert abs(nw_tstat(pd.Series(rng.normal(0, 1, 500)), 5)) < 4
    m = momentum(close, 12); r = reversal(close, 20); v = realized_vol(close, 60)
    assert m.iloc[252, 0] == pytest.approx(close.iloc[231, 0] / close.iloc[0, 0] - 1)
    assert r.iloc[20, 0] == pytest.approx(close.iloc[20, 0] / close.iloc[0, 0] - 1)
    assert v.iloc[100].between(0.005, 0.05).all()
    dec = decile_returns(fwd, fwd, mask, every=20)
    assert (dec["D10"] > dec["D1"]).all()


# --- xs_portfolio (E-A3) ---

def test_long_portfolio_backtest_mechanics() -> None:
    from strategies.cn_a_trend_base.xs_portfolio import long_portfolio_backtest, select_bottom_decile
    dates = pd.date_range("2024-01-01", periods=12, freq="B")
    cols = [f"6{i:05d}" for i in range(60)]
    rng = np.random.default_rng(2)
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (12, 60)), axis=0)), index=dates, columns=cols)
    open_px = close * (1 + rng.normal(0, 0.002, close.shape))
    mask = pd.DataFrame(True, index=dates, columns=cols)
    feat = pd.DataFrame(rng.normal(size=close.shape), index=dates, columns=cols)
    sel = select_bottom_decile(feat.iloc[0], mask.iloc[0], q=10)
    assert len(sel) == 6 and feat.iloc[0][sel].max() <= feat.iloc[0].drop(sel).min()
    lu = pd.DataFrame(False, index=dates, columns=cols); lu.loc[dates[1], sel[0]] = True   # 首只涨停不可买
    out = long_portfolio_backtest(close, open_px, feat, mask, every=5, q=10, limit_up=lu)
    assert np.isfinite(out["ret_net"]).all()
    assert out["n"].iloc[1] == 5 and out["cash"].iloc[1] > 0                     # 6 选 5 买入，现金留存
    assert out["turnover"].iloc[1] > 0 and out["cost"].iloc[1] > 0
    assert (out["turnover"].iloc[2:5] == 0).all()                                 # 持有期不换手
    assert out["n"].iloc[6] == 6 and out["turnover"].iloc[6] <= 2.0              # 第二次换仓
    # 无成本对照：净收益 = 毛收益 − 成本
    np.testing.assert_allclose(out["ret_net"], out["ret_gross"] - out["cost"])


def test_blend_sleeves_constant_mix_and_reset() -> None:
    from strategies.cn_a_trend_base.ew_index import blend_sleeves
    dates = pd.bdate_range("2024-01-29", "2024-02-06")
    rets = pd.DataFrame({"bm": 0.0, "rev": 0.10}, index=dates)          # rev 每天 +10%，bm 0
    out = blend_sleeves(rets, {"bm": 0.7, "rev": 0.3}, reset="M", reset_cost=0.0)
    assert out["ret_net"].iloc[0] == pytest.approx(0.03)               # 首日 30% × 10%
    assert out["ret_net"].iloc[1] > 0.03                                # 漂移：rev 权重上升
    reset_day = pd.Timestamp("2024-02-01")
    assert out.loc[reset_day, "turnover"] > 0 and out.loc[reset_day, "ret_net"] == pytest.approx(0.03)
    assert (out.loc[out.index != reset_day, "turnover"] == 0).all()
