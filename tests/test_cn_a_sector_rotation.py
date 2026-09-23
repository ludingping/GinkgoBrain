"""strategies/cn_a_sector_rotation — 特征、槽位状态机、执行器、BM-S、指标与验收、IC 探针（2026-09-23）。

设计：GinkgoRoad docs/GinkgoBrain/A股日频-板块趋势轮动-设计.md；用例编号见同目录 ...-测试用例.md。
全部合成 fixture，不连 DB。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategies.cn_a_trend_base import costs
from strategies.cn_a_sector_rotation.features import in_trend, sma, strength
from strategies.cn_a_sector_rotation.metrics import acceptance, beats_random, perf, random_summary
from strategies.cn_a_sector_rotation.rotation import (
    RuleParams, hold_all_selector, random_selector, rule_selector, run_executor, run_rotation,
)


# ─── fixtures ────────────────────────────────────────────────────────────

def _flat(codes: list[str], n: int = 40, px: float = 100.0, start: str = "2024-01-01") -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range(start, periods=n, freq="B")
    close = pd.DataFrame(px, index=dates, columns=codes, dtype=float)
    return close, close.copy()


def _frames(close: pd.DataFrame, trend_true: dict[str, tuple[int, int]], strength_vals: dict[str, float]):
    """trend_true[code] = (起, 止) 行号区间为真；strength 常量。"""
    trend = pd.DataFrame(False, index=close.index, columns=close.columns)
    for c, (a, b) in trend_true.items():
        trend.iloc[a:b, trend.columns.get_loc(c)] = True
    s = pd.DataFrame({c: strength_vals.get(c, 0.0) for c in close.columns}, index=close.index)
    return trend.astype("boolean"), s


def _held_at(res, day: int) -> set[str]:
    return set(res.weights.iloc[day][res.weights.iloc[day] > 0].index)


# ─── T1/T2 特征数学（§3.1 / §3.2） ────────────────────────────────────────

def test_in_trend_and_strength_math() -> None:
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    up = pd.Series(np.arange(30, dtype=float) + 100.0, index=dates)          # 单调上行
    down = pd.Series(130.0 - np.arange(30, dtype=float), index=dates)        # 单调下行
    close = pd.DataFrame({"UP": up, "DN": down})
    tr = in_trend(close, n=5, slope_lookback=3)
    assert tr["UP"].iloc[:7].isna().all()                                    # 5 + 3 − 1 = 7 行历史不足 → NA
    assert bool(tr["UP"].iloc[7]) and bool(tr["UP"].iloc[-1])
    assert tr["DN"].iloc[7] is not pd.NA and not bool(tr["DN"].iloc[-1])
    # 价格在均线上但均线拐头向下 → 假
    v = pd.Series([100.0] * 15 + [70.0] * 4 + [90.0], index=dates[:20])      # SMA5: 100 → 74（向下），close 90 > 74
    tr2 = in_trend(pd.DataFrame({"V": v}), n=5, slope_lookback=5)
    assert v.iloc[19] > sma(pd.DataFrame({"V": v}), 5)["V"].iloc[19] and not bool(tr2["V"].iloc[19])
    st = strength(close, L=10)
    assert st["UP"].iloc[:10].isna().all()
    assert st["UP"].iloc[10] == pytest.approx(110.0 / 100.0 - 1.0)
    assert st["DN"].iloc[-1] < 0


# ─── T3 满仓时新趋势行业不进入（§3.3 进入） ────────────────────────────────

def test_full_slots_new_trend_sector_does_not_enter() -> None:
    close, open_px = _flat(["A", "B", "C", "D"])
    trend, s = _frames(close, {"A": (0, 40), "B": (0, 40), "C": (10, 40)}, {"A": 0.1, "B": 0.2, "C": 0.9})
    res = run_rotation(close, open_px, RuleParams(k=2, every=5), trend=trend, strength=s)
    assert _held_at(res, 1) == {"A", "B"}                                    # 第 0 日评估，第 1 日开盘成交
    assert all(_held_at(res, d) == {"A", "B"} for d in range(11, 40))       # C 强度最高但无空槽 → 不进
    assert "C" not in set(res.trades["code"])


# ─── T4 退出后同一评估空槽由最高强度者补入（§3.3） ─────────────────────────

def test_exit_frees_slot_filled_same_evaluation_by_top_strength() -> None:
    close, open_px = _flat(["A", "B", "C", "D"])
    trend, s = _frames(close, {"A": (0, 15), "B": (0, 40), "C": (0, 40), "D": (0, 40)},
                       {"A": 0.5, "B": 0.4, "C": 0.3, "D": 0.35})
    res = run_rotation(close, open_px, RuleParams(k=2, every=5), trend=trend, strength=s)
    assert _held_at(res, 14) == {"A", "B"}
    assert _held_at(res, 16) == {"B", "D"}                                   # 第 15 日 A 出趋势；D(0.35) > C(0.3)
    tr = res.trades.set_index("code")
    assert tr.loc["A", "exit_date"] == close.index[16] and tr.loc["D", "entry_date"] == close.index[16]
    assert tr.loc["A", "exit_reason"] == "trend"


# ─── T5 排名在 K 与 2K 之间的持仓不动；跌出 2K 退出（§3.3 退出 2） ────────────

def test_held_ranked_between_k_and_2k_stays_but_beyond_2k_exits() -> None:
    codes = list("ABCDEF")
    close, open_px = _flat(codes)
    trend, s = _frames(close, {c: (0, 40) for c in codes}, {"A": 0.9, "B": 0.8, "C": 0.7, "D": 0.6, "E": 0.5, "F": 0.4})
    s = s.copy()
    s.loc[close.index[10]:, "A"] = 0.65                                      # A 降到第 3 名（K=2 < 3 ≤ 2K=4）
    s.loc[close.index[20]:, "A"] = 0.45                                      # A 降到第 5 名（> 2K）
    res = run_rotation(close, open_px, RuleParams(k=2, every=5), trend=trend, strength=s)
    assert _held_at(res, 1) == {"A", "B"}
    assert _held_at(res, 16) == {"A", "B"}                                   # 第 10/15 日评估：A 第 3 名 → 不动
    assert _held_at(res, 21) == {"B", "C"}                                   # 第 20 日评估：A 第 5 名 → 退出，C 补入
    assert res.trades.set_index("code").loc["A", "exit_reason"] == "rank"


# ─── T6 全部行业出趋势 → 100% 现金（§3.3 全空仓） ─────────────────────────

def test_all_out_of_trend_goes_full_cash() -> None:
    close, open_px = _flat(["A", "B", "C"])
    trend, s = _frames(close, {"A": (0, 20), "B": (0, 20), "C": (0, 20)}, {"A": 0.3, "B": 0.2, "C": 0.1})
    res = run_rotation(close, open_px, RuleParams(k=2, every=5), trend=trend, strength=s)
    assert res.daily["n"].iloc[10] == 2
    assert (res.daily["n"].iloc[21:] == 0).all() and np.allclose(res.daily["cash"].iloc[21:], 1.0)
    assert (res.daily["ret_net"].iloc[22:] == 0).all()


# ─── T7 随机对照与主规则共用同一执行器（§3.5） ─────────────────────────────

def test_random_control_and_main_rule_share_executor() -> None:
    codes = list("ABCDEFGH")
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (60, 8)), axis=0)), index=dates, columns=codes)
    open_px = close.shift(1).fillna(100.0)
    p = RuleParams(k=3, n_in=10, L=5, every=5)
    main = run_rotation(close, open_px, p)
    same = run_rotation(close, open_px, p, selector=rule_selector(p))       # 显式传入主规则选择器 → 逐日相同
    pd.testing.assert_frame_equal(main.daily, same.daily)
    rnd = run_rotation(close, open_px, p, selector=random_selector(p), seed=1)
    assert list(rnd.daily.columns) == list(main.daily.columns)
    assert rnd.daily["n"].max() <= 3 and rnd.daily["n"].iloc[1] == 3        # 不看趋势：首次评估即满仓
    rnd2 = run_rotation(close, open_px, p, selector=random_selector(p), seed=1)
    pd.testing.assert_frame_equal(rnd.daily, rnd2.daily)                     # 种子可复现
    assert not rnd.daily["ret_net"].equals(main.daily["ret_net"])


# ─── T8 T+1 开盘成交与成本：一次往返 = buy_cost + sell_cost（§1.3） ─────────

def test_t_plus_1_open_execution_and_round_trip_cost() -> None:
    dates = pd.date_range("2024-01-01", periods=12, freq="B")
    close = pd.DataFrame({"A": [100.0, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110]}, index=dates)
    open_px = pd.DataFrame({"A": [100.0, 105, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110]}, index=dates)
    trend, s = _frames(close, {"A": (0, 5)}, {"A": 0.1})
    res = run_rotation(close, open_px, RuleParams(k=1, every=5), trend=trend, strength=s)
    d = res.daily
    assert d["n"].iloc[0] == 0 and d["n"].iloc[1] == 1                      # T 评估、T+1 成交
    assert d["ret_gross"].iloc[1] == pytest.approx(110.0 / 105.0 - 1.0)     # 开盘 105 买入，不是收盘 100
    assert d["cost"].iloc[1] == pytest.approx(costs.buy_cost())             # 满仓买入 1.0 × buy_cost
    assert d["turnover"].iloc[1] == pytest.approx(1.0)
    assert (d["turnover"].iloc[2:6] == 0).all()
    # 第 5 日评估：出趋势 → 第 6 日开盘全卖；卖出额 = 持仓市值 → 成本 = sell_cost × 市值/组合
    v_ratio = d["cash"].iloc[6]                                              # 卖后全现金，权重 1
    assert v_ratio == pytest.approx(1.0)
    assert d["cost"].iloc[6] == pytest.approx(costs.sell_cost(dates[6]) * (1.0 - d["cash"].iloc[5]), rel=1e-6)
    np.testing.assert_allclose(d["ret_net"], d["ret_gross"] - d["cost"])
    assert len(res.trades) == 1 and res.trades.iloc[0]["exit_date"] == dates[6]


# ─── T9 BM-S：每月首个交易日再平衡，月内不换手，年换手小（§1.1） ──────────

def test_bm_s_monthly_rebalance_turnover_small() -> None:
    from strategies.cn_a_sector_rotation.panel import bm_s
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2023-01-01", "2024-12-31")
    codes = [f"8010{i:02d}" for i in range(31)]
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.012, (len(dates), 31)), axis=0)), index=dates, columns=codes)
    open_px = close * (1 + rng.normal(0, 0.002, close.shape))
    bm = bm_s(close, open_px, codes)
    rb = bm.index[bm["turnover"] > 0]
    month_first = pd.Series(dates, index=dates).groupby(dates.to_period("M")).first()
    assert set(rb[1:]) <= set(month_first) and len(rb) >= 22                 # 首次建仓外只在月首交易日换仓
    assert bm["n"].iloc[1:].eq(31).all()
    turn_yr = bm["turnover"].sum() / 2.0 / (len(dates) / 244)
    assert 0 < turn_yr < 2.0                                                 # 单边年换手：漂移再平衡应很小
    assert bm["cost"].iloc[1] == pytest.approx(costs.buy_cost(), rel=1e-6)   # 首次建仓全买
    assert (bm["cash"].iloc[1:] < 0.01).all()


# ─── T10 周频评估只在评估日次日成交（§3.3 评估频率） ──────────────────────

def test_weekly_evaluation_trades_only_after_evaluation_days() -> None:
    codes = list("ABCDEFGH")
    rng = np.random.default_rng(5)
    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.02, (60, 8)), axis=0)), index=dates, columns=codes)
    open_px = close.shift(1).fillna(100.0)
    p = RuleParams(k=2, every=5)
    res = run_rotation(close, open_px, p, selector=random_selector(p), seed=7, start=dates[3])
    traded = np.flatnonzero(res.daily["turnover"].to_numpy() > 0)
    assert len(traded) > 3 and all((t - 1 - 3) % 5 == 0 for t in traded)     # 评估日 t=3,8,13,… → 成交 4,9,14,…
    daily = run_rotation(close, open_px, RuleParams(k=2, every=1), selector=random_selector(p), seed=7)
    assert (daily.daily["turnover"].iloc[1:] > 0).mean() > 0.5              # 每日评估 → 几乎每日换仓


# ─── T11 exit_trail：距持仓期最高点回撤恰好 15% 触发（§3.4） ──────────────

def test_exit_trail_triggers_at_exactly_15pct() -> None:
    def run(last: float):
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        px = [100.0, 100, 120, 110, 105, last, last, last, last, last]
        close = pd.DataFrame({"A": px}, index=dates); open_px = close.copy()
        trend, s = _frames(close, {"A": (0, 10)}, {"A": 0.1})
        return run_rotation(close, open_px, RuleParams(k=1, every=1, exit_trail=True), trend=trend, strength=s)
    hit = run(102.0)                                                          # 持仓期最高 120 → 102 = −15%
    assert hit.daily["n"].iloc[5] == 1 and hit.daily["n"].iloc[6] == 0
    assert hit.trades.iloc[0]["exit_reason"] == "trail"
    miss = run(102.5)                                                         # −14.6% → 不触发
    assert (miss.daily["n"].iloc[1:] == 1).all()
    # 位置高点从进入日起算：进入前的 100→120 之外的历史不算
    base = run(120.0)
    assert (base.daily["n"].iloc[1:] == 1).all()


# ─── T12 exit_fast：close < SMA(20) 退出，即便仍在 SMA(N) 趋势中（§3.4） ─────

def test_exit_fast_uses_sma20() -> None:
    dates = pd.date_range("2024-01-01", periods=40, freq="B")
    px = np.full(40, 100.0); px[30:] = 95.0                                   # 第 30 日起跌破 SMA20
    close = pd.DataFrame({"A": px, "B": 100.0}, index=dates); open_px = close.copy()
    trend, s = _frames(close, {"A": (0, 40), "B": (0, 40)}, {"A": 0.5, "B": 0.1})
    slow = run_rotation(close, open_px, RuleParams(k=1, every=1), trend=trend, strength=s)
    fast = run_rotation(close, open_px, RuleParams(k=1, every=1, exit_fast=True), trend=trend, strength=s)
    assert (slow.weights["A"].iloc[1:] > 0).all()                            # 主规则：趋势仍真 → 持有
    assert fast.weights["A"].iloc[31] == 0 and fast.weights["B"].iloc[31] > 0   # 快退出：卖 A 补 B
    assert fast.trades.iloc[0]["exit_reason"] == "fast"


# ─── T13 H-B1 模式：持有全部在趋势行业等权，每周再平衡（§5 H-B1） ───────────

def test_hold_all_in_trend_mode_rebalances_weekly() -> None:
    close, open_px = _flat(["A", "B", "C", "D"])
    trend, s = _frames(close, {"A": (0, 40), "B": (0, 40), "C": (0, 12)}, {})
    p = RuleParams(k=None, every=5)
    res = run_rotation(close, open_px, p, trend=trend, strength=s, selector=hold_all_selector(p))
    w = res.weights
    assert np.allclose(w.iloc[1][["A", "B", "C"]], 1 / 3, rtol=3e-3) and w.iloc[1]["A"] == w.iloc[1]["C"] and w.iloc[1]["D"] == 0
    assert np.allclose(w.iloc[16][["A", "B"]], 1 / 2, rtol=3e-3) and w.iloc[16]["C"] == 0   # 第 15 日评估 C 出趋势
    assert res.daily["turnover"].iloc[16] == pytest.approx(2 / 3, rel=1e-2)   # 卖 1/3、买 1/3（成本级误差）


# ─── T14 指标与验收评估器（§1.2 双栏 + no-harm + 随机对照） ───────────────

def test_metrics_and_acceptance_on_toy_numbers() -> None:
    dates = pd.date_range("2020-01-01", periods=488, freq="B")               # 2 年
    daily = pd.DataFrame({"ret_net": 0.001, "turnover": 0.0, "n": 4, "cash": 0.0}, index=dates)
    daily.loc[dates[::20], "turnover"] = 0.5
    m = perf(daily, "2020-01-01", "2021-12-31")
    assert m["years"] == pytest.approx(2.0, rel=0.02) and m["turnover_yr"] == pytest.approx(0.5 * 25 / 2 / m["years"], rel=0.05)
    base = {"total_return": 0.30, "max_drawdown": -0.30, "calmar": 0.4, "turnover_yr": 2.0}
    good_risk = {"total_return": 0.25, "max_drawdown": -0.19, "calmar": 0.6, "turnover_yr": 8.0}
    assert acceptance(good_risk, base, "risk") == "PASS"
    assert acceptance({**good_risk, "max_drawdown": -0.21}, base, "risk") == "FAIL(mdd)"
    assert acceptance({**good_risk, "total_return": 0.23}, base, "risk") == "FAIL(return)"
    good_ret = {"total_return": 0.34, "max_drawdown": -0.29, "calmar": 0.5, "turnover_yr": 11.9}
    assert acceptance(good_ret, base, "return") == "PASS"
    assert acceptance({**good_ret, "total_return": 0.32}, base, "return") == "FAIL(return)"
    assert acceptance({**good_ret, "turnover_yr": 12.1}, base, "return") == "FAIL(turnover)"
    assert acceptance({**good_ret, "calmar": 0.3, "total_return": 0.32}, base, "return") == "FAIL(return,calmar)"
    assert acceptance({"total_return": 0.24, "max_drawdown": -0.30, "calmar": 0.1, "turnover_yr": 1.0}, base, "noharm") == "PASS"
    assert acceptance({"total_return": 0.24, "max_drawdown": -0.31, "calmar": 0.1, "turnover_yr": 1.0}, base, "noharm") == "FAIL(mdd)"
    neg = {**base, "total_return": -0.10}                                    # BM 为负：收益只需高于 BM
    assert acceptance({**good_ret, "total_return": -0.05}, neg, "return").startswith("PASS")
    rs = random_summary([{"max_drawdown": -x, "total_return": x} for x in np.linspace(0.1, 0.5, 100)])
    assert rs["max_drawdown"]["median"] == pytest.approx(-0.3, abs=0.01)
    assert rs["max_drawdown"]["p5"] < rs["max_drawdown"]["median"] < rs["max_drawdown"]["p95"]
    assert beats_random({"max_drawdown": -0.25}, rs) and not beats_random({"max_drawdown": -0.35}, rs)


# ─── T15 IC 探针：周频 rank IC、五分位价差、成交额上下半区（§4 L1 / H-B3） ──

def test_ic_probe_weekly_ic_and_quintiles() -> None:
    from strategies.cn_a_sector_rotation.ic_probe import amount_split_masks, quantile_spread, weekly_rank_ic
    from strategies.cn_a_trend_base.xs_probe import forward_open_return
    rng = np.random.default_rng(0)
    dates = pd.date_range("2017-01-01", periods=300, freq="B")
    codes = [f"8010{i:02d}" for i in range(31)]
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.02, (300, 31)), axis=0)), index=dates, columns=codes)
    open_px = close.shift(1).fillna(100.0)
    mask = pd.DataFrame(True, index=dates, columns=codes)
    fwd = forward_open_return(open_px, 20)
    ic = weekly_rank_ic(fwd, fwd, mask, every=5)
    assert len(ic) == 60 and ic.dropna().min() > 0.99                        # 每 5 日一个截面，特征 = 未来收益 → IC≈1
    ic_neg = weekly_rank_ic(-fwd, fwd, mask, every=5)
    assert ic_neg.dropna().max() < -0.99
    q = quantile_spread(fwd, fwd, mask, every=5, q=5)
    assert (q["Q5"] > q["Q1"]).all() and list(q.columns) == ["Q1", "Q2", "Q3", "Q4", "Q5"]
    amount = pd.DataFrame(rng.lognormal(20, 1, close.shape), index=dates, columns=codes)
    up, lo = amount_split_masks(amount, window=20)
    assert (up & lo).sum().sum() == 0 and up.iloc[:19].sum().sum() == 0
    assert 14 <= up.iloc[-1].sum() <= 16 and 15 <= lo.iloc[-1].sum() <= 16   # 中位数切分 ≈ 各半


# ─── T16 面板：逐年覆盖、首日、坏数据报错不填零（§4 L0） ─────────────────

def test_panel_coverage_and_validation() -> None:
    from strategies.cn_a_sector_rotation.panel import coverage_by_year, first_dates, to_wide, validate_panel
    dates = pd.bdate_range("2020-01-01", "2021-12-31")
    long = pd.DataFrame({"trade_date": np.repeat(dates, 2), "index_code": ["801010", "801030"] * len(dates),
                         "close": 100.0})
    long = long[~((long["index_code"] == "801030") & (long["trade_date"] < "2021-01-01"))]     # 801030 2021 起
    wide = to_wide(long, "close")
    assert wide.shape == (len(dates), 2) and wide["801030"].loc[:"2020-12-31"].isna().all()
    fd = first_dates(wide)
    assert fd["801010"] == dates[0] and fd["801030"] == pd.Timestamp("2021-01-01")
    cov = coverage_by_year(wide, ["801010", "801030"])
    assert cov.loc[2020, "coverage"] == pytest.approx(0.5) and cov.loc[2021, "coverage"] == pytest.approx(1.0)
    assert cov.loc[2020, "gap_days"] == 0
    validate_panel(wide)
    bad = wide.copy(); bad.iloc[10, 0] = 0.0
    with pytest.raises(ValueError, match="801010"):
        validate_panel(bad)


# ─── T18 缺行日：不填零、收益 0，下一有效日补齐整段涨跌（L0"缺失不填零"） ─────

def test_missing_close_day_carries_price_and_catches_up() -> None:
    dates = pd.date_range("2024-01-01", periods=6, freq="B")
    close = pd.DataFrame({"A": [100.0, 100.0, np.nan, 120.0, 120.0, 120.0]}, index=dates)
    open_px = close.copy(); open_px.iloc[1] = 100.0
    trend, s = _frames(close, {"A": (0, 6)}, {"A": 0.1})
    res = run_rotation(close, open_px, RuleParams(k=1, every=5), trend=trend, strength=s)
    assert res.daily["ret_gross"].iloc[2] == 0.0                              # 缺行日收益 0
    assert res.daily["ret_gross"].iloc[3] == pytest.approx(0.20, rel=2e-3)   # 下一有效日补齐 100 → 120（成本级误差）
    assert res.daily["n"].iloc[2] == 1 and res.daily["n"].iloc[3] == 1       # 持仓不因缺行消失


# ─── T17 通用执行器：开盘缺价则跳过该笔、现金不足按比例缩买 ───────────────

def test_executor_skips_untradable_and_caps_buys_at_cash() -> None:
    dates = pd.date_range("2024-01-01", periods=6, freq="B")
    close = pd.DataFrame({"A": 100.0, "B": 100.0}, index=dates); open_px = close.copy()
    open_px.loc[dates[1], "B"] = np.nan                                      # B 第 1 日无开盘价
    plans = {0: {"A": 0.5, "B": 0.5}, 2: {"A": 0.0, "B": 0.5}}
    res = run_executor(close, open_px, lambda ctx: plans.get(ctx.t, {}), every=1)
    assert res.weights.iloc[1]["A"] == pytest.approx(0.5, rel=5e-3) and res.weights.iloc[1]["B"] == 0
    assert res.daily["cash"].iloc[1] == pytest.approx(0.5, rel=5e-3)
    assert res.weights.iloc[3]["A"] == 0 and res.weights.iloc[3]["B"] == pytest.approx(0.5, rel=5e-3)
    over = run_executor(close, open_px, lambda ctx: {"A": 0.7, "B": 0.7} if ctx.t == 2 else {}, every=1)
    assert over.daily["cash"].iloc[3] >= -5e-3 and over.weights.iloc[3].sum() == pytest.approx(1.0, rel=5e-3)   # 现金只欠成本
