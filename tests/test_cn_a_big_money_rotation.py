"""测试 A 股大单净流入横截面轮动策略 v0.1.

对应设计：GinkgoRoad/docs/GinkgoBrain/A股日频-大单净流入横截面轮动-设计.md
对应测试用例清单：同目录 -测试用例.md

测试编号 T1-T29 与测试用例清单一一对应。所有 Level 1 单测使用合成 fixture，
不连接真实 PG（参考 test_contract_db.py 用 SQLite monkey-patch 的惯例）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategies.cn_a_big_money_rotation.data import (
    ANOMALY_DEVIATION_THRESHOLD,
    compute_historical_float_mv,
    compute_money_flow_factors,
    recover_float_share,
)
from strategies.cn_a_big_money_rotation.universe import (
    LIQUIDITY_FLOOR_RMB,
    MIN_LISTING_DAYS,
    daily_universe,
    is_st_name,
    is_suspended,
    passes_liquidity,
    passes_listing_age,
)
from strategies.cn_a_big_money_rotation.signal import (
    TARGET_TOP_N,
    compute_signals,
    dual_signal_overlap,
    rank_cross_section,
    select_top_n,
)
from strategies.cn_a_big_money_rotation.portfolio import (
    CAP_BUY_RATIO,
    COMMISSION_RATE,
    HOLDING_MIN_DAYS,
    SLIPPAGE_RATE,
    STAMP_DUTY_RATE,
    CostParams,
    Order,
    Position,
    PortfolioState,
    Side,
    apply_orders_at_t1,
    buy_cost,
    capacity_cap_for_buy,
    decide_orders,
    holdings_in_lock,
    is_one_word_limit_down,
    is_one_word_limit_up,
    per_position_target,
    sell_cost,
)
from strategies.cn_a_big_money_rotation.evaluate import (
    CHURNING_RED_LINE,
    IC_MEAN_THRESHOLD,
    IC_TSTAT_THRESHOLD,
    DualSignalStatus,
    ICStats,
    StrategyMetrics,
    Verdict,
    annualize_return,
    classify_dual_signal,
    compute_ic,
    compute_ic_stats,
    compute_information_ratio,
    compute_max_drawdown,
    compute_sharpe,
    detect_unstable_window,
    render_verdict,
)


def test_smoke_package_importable():
    """M0 占位：验证策略包可被 import。后续 M1-M6 在此文件累加 T1-T29。"""
    import strategies.cn_a_big_money_rotation as pkg

    assert pkg.__version__.startswith("0.1")


# ============================================================================
# M1 数据层（T1-T4）
# ============================================================================


class TestT1FloatShareRecovery:
    """T1. snapshot 反推 float_share 一致性 ⭐核心 (§3.3)."""

    def test_t1_consistent(self):
        """自洽数据：主路径与互验路径都得 1e8 股，偏差 0%，不标异常."""
        row = dict(
            float_market_cap=1e9,
            current_price=10.0,
            total_volume=1e7,
            turnover_rate=10.0,
        )
        r = recover_float_share(row)
        assert r.share_main == pytest.approx(1e8)
        assert r.share_check == pytest.approx(1e8)
        assert r.deviation_pct == pytest.approx(0.0, abs=1e-12)
        assert r.share == pytest.approx(1e8)
        assert not r.anomaly
        assert r.notes == {}

    def test_t1_minor_deviation_no_anomaly(self):
        """偏差 ~4.2% < 5% → 不标异常."""
        row = dict(
            float_market_cap=1e9,
            current_price=10.0,
            total_volume=1e7,
            turnover_rate=9.6,
        )
        r = recover_float_share(row)
        # share_check = 1e7 / 0.096 ≈ 1.0417e8 → deviation ≈ 0.0417
        assert r.deviation_pct == pytest.approx(0.0417, abs=0.005)
        assert r.deviation_pct < ANOMALY_DEVIATION_THRESHOLD
        assert not r.anomaly

    def test_t1_large_deviation_marks_anomaly(self):
        """偏差 ~30% > 5% → anomaly=True 且标 share_recover_anomaly."""
        row = dict(
            float_market_cap=1e9,
            current_price=10.0,
            total_volume=1e7,
            turnover_rate=7.0,
        )
        r = recover_float_share(row)
        # share_check = 1e7 / 0.07 ≈ 1.4286e8 → deviation ≈ 0.4286
        assert r.deviation_pct > ANOMALY_DEVIATION_THRESHOLD
        assert r.anomaly
        assert r.notes.get("share_recover_anomaly") is True

    def test_t1_main_zero_division_fallback_to_check(self):
        """current_price=0 → 主路径不可用，回退互验，标 main_path_zero_division."""
        row = dict(
            float_market_cap=1e9,
            current_price=0.0,
            total_volume=1e7,
            turnover_rate=10.0,
        )
        r = recover_float_share(row)
        assert r.share_main is None
        assert r.share_check == pytest.approx(1e8)
        assert r.share == pytest.approx(1e8)  # 回退至 check
        assert r.notes.get("main_path_zero_division") is True

    def test_t1_check_path_zero_division_no_anomaly(self):
        """turnover_rate=0 → 互验跳过，输出主路径，不标异常（数据稀疏不是错误）."""
        row = dict(
            float_market_cap=1e9,
            current_price=10.0,
            total_volume=1e7,
            turnover_rate=0.0,
        )
        r = recover_float_share(row)
        assert r.share_main == pytest.approx(1e8)
        assert r.share_check is None
        assert r.share == pytest.approx(1e8)
        assert not r.anomaly

    def test_t1_both_paths_unavailable(self):
        """主+互验全不可用 → share=None, anomaly=True, both_paths_unavailable."""
        row = dict(
            float_market_cap=None,
            current_price=None,
            total_volume=None,
            turnover_rate=None,
        )
        r = recover_float_share(row)
        assert r.share is None
        assert r.anomaly
        assert r.notes.get("both_paths_unavailable") is True


class TestT2HistoricalFloatMV:
    """T2. 历史 float_mv = float_share × close_t (§3.3)."""

    def test_t2_multi_day_reconstruction(self):
        """4 日 close 序列重建 float_mv 序列."""
        close = pd.Series([10.0, 11.0, 12.0, 10.5])
        mv = compute_historical_float_mv(1e8, close)
        expected = pd.Series([1e9, 1.1e9, 1.2e9, 1.05e9])
        pd.testing.assert_series_equal(
            mv,
            expected,
            check_exact=False,
            rtol=1e-9,
            check_names=False,
        )

    def test_t2_close_null_propagates_nan(self):
        """close NULL → 当日 float_mv = NaN（不前向填充）."""
        close = pd.Series([10.0, np.nan, 12.0])
        mv = compute_historical_float_mv(1e8, close)
        assert mv.iloc[0] == pytest.approx(1e9)
        assert pd.isna(mv.iloc[1])
        assert mv.iloc[2] == pytest.approx(1.2e9)


class TestT3MoneyFlowSignalDerivation:
    """T3. money_flow 派生因子 (§3.4)."""

    @staticmethod
    def _default_mf_row(**overrides):
        """默认 cje fixture：8 档金额（仅大单族非零，方便 big_net_inflow 验证）."""
        base = dict(
            zmbtdcje=1e7, zmbddcje=2e7, zmbzdcje=0.0, zmbxdcje=0.0,
            zmstdcje=5e6, zmsddcje=8e6, zmszdcje=0.0, zmsxdcje=0.0,
        )
        base.update(overrides)
        return base

    def test_t3_big_net_inflow_basic(self):
        """big_net_inflow = (10e6+20e6) - (5e6+8e6) = 17e6."""
        df = pd.DataFrame([self._default_mf_row()])
        out = compute_money_flow_factors(df)
        assert out["big_net_inflow"].iloc[0] == pytest.approx(1.7e7)

    def test_t3_total_amount_full_8_fields(self):
        """8 档分别 1e6..8e6 → total = 36e6."""
        row = {f: (i + 1) * 1e6 for i, f in enumerate([
            "zmbtdcje", "zmbddcje", "zmbzdcje", "zmbxdcje",
            "zmstdcje", "zmsddcje", "zmszdcje", "zmsxdcje",
        ])}
        df = pd.DataFrame([row])
        out = compute_money_flow_factors(df)
        assert out["total_amount"].iloc[0] == pytest.approx(36e6)

    def test_t3_big_net_per_mv_main_signal(self):
        """signal_A = big_net_inflow / float_mv = 17e6/1e9 = 0.017."""
        df = pd.DataFrame([self._default_mf_row()])
        df["float_mv"] = 1e9
        out = compute_money_flow_factors(df)
        assert out["big_net_per_mv"].iloc[0] == pytest.approx(0.017)

    def test_t3_big_net_ratio_diagnostic_signal(self):
        """signal_B = big_net_inflow / total_amount.

        构造：大单买 10+10=20, 大单卖 1+2=3 → big_net = 17 (e6)
                中小买 5+3=8,  中小卖 3+2=5 → 中小 sum = 13
                total = 20 + 3 + 13 = 36 (e6) → ratio = 17/36 ≈ 0.472
        """
        row = dict(
            zmbtdcje=10e6, zmbddcje=10e6, zmbzdcje=5e6, zmbxdcje=3e6,
            zmstdcje=1e6,  zmsddcje=2e6, zmszdcje=3e6, zmsxdcje=2e6,
        )
        df = pd.DataFrame([row])
        out = compute_money_flow_factors(df)
        assert out["big_net_inflow"].iloc[0] == pytest.approx(17e6)
        assert out["total_amount"].iloc[0] == pytest.approx(36e6)
        assert out["big_net_ratio"].iloc[0] == pytest.approx(17 / 36, abs=1e-6)

    def test_t3_null_field_coalesced_to_zero(self):
        """单档 NULL → COALESCE 为 0.

        zmbtdcje=NULL（主买特大缺）+ default 其余 → big_net = 0+20-5-8 = 7e6
                                                   total = 0+20+5+8 = 33e6
        """
        row = self._default_mf_row(zmbtdcje=None)
        df = pd.DataFrame([row])
        out = compute_money_flow_factors(df)
        assert out["big_net_inflow"].iloc[0] == pytest.approx(7e6)
        assert out["total_amount"].iloc[0] == pytest.approx(33e6)

    def test_t3_total_zero_ratio_is_nan(self):
        """total_amount=0（全档为 0）→ big_net_ratio = NaN（不入截面排序）."""
        row = {f: 0.0 for f in [
            "zmbtdcje", "zmbddcje", "zmbzdcje", "zmbxdcje",
            "zmstdcje", "zmsddcje", "zmszdcje", "zmsxdcje",
        ]}
        df = pd.DataFrame([row])
        out = compute_money_flow_factors(df)
        assert out["total_amount"].iloc[0] == 0.0
        assert pd.isna(out["big_net_ratio"].iloc[0])


class TestT4MoneyFlowMissing:
    """T4. money_flow 行整体缺失处置（设计 §4 步骤 4）."""

    def test_t4_empty_input_returns_empty_with_derived_columns(self):
        """空 DataFrame 输入 → 输出 0 行 但派生列齐全（universe 层 INNER JOIN 自然剔除）."""
        df = pd.DataFrame(columns=[
            "zmbtdcje", "zmbddcje", "zmbzdcje", "zmbxdcje",
            "zmstdcje", "zmsddcje", "zmszdcje", "zmsxdcje",
        ])
        out = compute_money_flow_factors(df)
        assert len(out) == 0
        for col in ("big_net_inflow", "total_amount", "big_net_ratio"):
            assert col in out.columns


# ============================================================================
# M2 Universe 层（T5-T8）
# ============================================================================

from datetime import date, timedelta  # noqa: E402  (位置故意：分节边界)


class TestT5STPrefixFilter:
    """T5. ST/*ST/退 前缀过滤 (§4 步骤 2)."""

    @pytest.mark.parametrize("name,expected", [
        ("平安银行", False),       # 正常股
        ("ST 康美", True),         # ST
        ("*ST 海航", True),        # *ST
        ("退市某某", True),        # 退
        ("  ST 北农", True),       # 前导空白
        ("STAR", True),            # 大写 ST 开头（已知误伤；caller 责）
        ("", False),               # 空字符串
        (None, False),             # NULL
    ])
    def test_t5_st_prefix(self, name, expected):
        assert is_st_name(name) is expected


class TestT6ListingAgeFilter:
    """T6. 次新过滤：上市 < 250 日剔除 (§4 步骤 3)."""

    def test_t6_249_days_rejected(self):
        as_of = date(2026, 5, 11)
        listed = as_of - timedelta(days=249)
        assert not passes_listing_age(listed, as_of)

    def test_t6_250_days_accepted(self):
        as_of = date(2026, 5, 11)
        listed = as_of - timedelta(days=250)
        assert passes_listing_age(listed, as_of)

    def test_t6_null_listing_date_rejected(self):
        """上市日 NULL → 保守剔除."""
        assert not passes_listing_age(None, date(2026, 5, 11))

    def test_t6_uses_min_days_default(self):
        assert MIN_LISTING_DAYS == 250


class TestT7SuspensionFilter:
    """T7. 停牌过滤 (§4 步骤 4)."""

    def test_t7_zero_amount_suspended(self):
        assert is_suspended("000001", {"000001": 0.0}, {"000001"}) is True

    def test_t7_negative_amount_suspended(self):
        """成交额 NaN → 停牌."""
        assert is_suspended("000001", {"000001": float("nan")}, {"000001"}) is True

    def test_t7_missing_kline_row_suspended(self):
        assert is_suspended("000001", {}, {"000001"}) is True

    def test_t7_missing_money_flow_row_suspended(self):
        """kline 有但 money_flow 没有 → 剔除."""
        assert is_suspended("000001", {"000001": 1e8}, set()) is True

    def test_t7_normal_day_not_suspended(self):
        assert is_suspended("000001", {"000001": 1e8}, {"000001"}) is False


class TestT8LiquidityFloor:
    """T8. 流动性下限 ⭐ (§4 步骤 5)."""

    def test_t8_mean_4999w_rejected(self):
        """20 日均 = 4999 万 → 剔除（均值 < 5000 万）."""
        amts = pd.Series([4999e4] * 20)
        assert not passes_liquidity(amts)

    def test_t8_mean_5000w_accepted(self):
        """20 日均 = 5000 万 → 保留（边界等于条件）."""
        amts = pd.Series([5000e4] * 20)
        assert passes_liquidity(amts)

    def test_t8_same_day_below_mean_above_rejected(self):
        """当日 4000 万 + 20 日均 8000 万 → 剔除（双条件 AND，当日不足）."""
        amts = pd.Series([8000e4] * 19 + [4000e4])
        assert not passes_liquidity(amts)

    def test_t8_partial_window_above_floor_accepted(self):
        """不足 20 日：仅 100 日内的 5 日窗口，均 6000 万 → 保留（用现有窗口均值）."""
        amts = pd.Series([6000e4] * 5)
        assert passes_liquidity(amts)

    def test_t8_empty_series_rejected(self):
        assert not passes_liquidity(pd.Series(dtype=float))

    def test_t8_uses_default_floor(self):
        assert LIQUIDITY_FLOOR_RMB == 5e7


class TestDailyUniverseComposite:
    """组合测试：daily_universe 5 步联合（覆盖 T5-T8 综合）."""

    def _make_input(self):
        as_of = date(2026, 5, 11)
        sec_list = pd.DataFrame({
            "stock_code": ["000001", "ST0002", "300003", "600004", "688005"],
            "stock_name": ["平安银行", "ST 风险", "新股", "退市股", "正常 5"],
        })
        # 000001 上市 5 年；ST0002 上市 5 年但 ST；300003 上市 100 日；
        # 600004 上市 5 年但名称含"退"；688005 上市 5 年。
        old = as_of - timedelta(days=2000)
        new = as_of - timedelta(days=100)
        listing_dates = {
            "000001": old,
            "ST0002": old,
            "300003": new,
            "600004": old,
            "688005": old,
        }
        kline_today_amount = {
            "000001": 1e9,    # 10 亿
            "ST0002": 1e9,
            "300003": 1e9,
            "600004": 1e9,
            "688005": 1e9,
        }
        money_flow_today_codes = {"000001", "ST0002", "300003", "600004", "688005"}
        amount_history = {
            c: pd.Series([1e9] * 20)
            for c in ["000001", "ST0002", "300003", "600004", "688005"]
        }
        return (as_of, sec_list, listing_dates, kline_today_amount,
                money_flow_today_codes, amount_history)

    def test_daily_universe_filters_st_new_delisted(self):
        """只有 000001 和 688005 应通过（ST0002 ST 剔；300003 次新剔；600004 退 剔）."""
        args = self._make_input()
        universe = daily_universe(*args)
        assert universe == {"000001", "688005"}

    def test_daily_universe_suspension_excluded(self):
        """000001 当日停牌（amount=0）→ 被剔除."""
        as_of, sec, listing, kline, mf, hist = self._make_input()
        kline = {**kline, "000001": 0.0}
        universe = daily_universe(as_of, sec, listing, kline, mf, hist)
        assert "000001" not in universe
        assert universe == {"688005"}

    def test_daily_universe_money_flow_missing_excluded(self):
        """000001 当日缺 money_flow 行 → 被剔除."""
        as_of, sec, listing, kline, mf, hist = self._make_input()
        mf = mf - {"000001"}
        universe = daily_universe(as_of, sec, listing, kline, mf, hist)
        assert "000001" not in universe

    def test_daily_universe_low_liquidity_excluded(self):
        """000001 20 日均 4000 万 → 剔除."""
        as_of, sec, listing, kline, mf, hist = self._make_input()
        hist = {**hist, "000001": pd.Series([4000e4] * 20)}
        universe = daily_universe(as_of, sec, listing, kline, mf, hist)
        assert "000001" not in universe


# ============================================================================
# M3 信号层（T9-T12）
# ============================================================================


class TestT9CrossSectionRankDeterministic:
    """T9. 截面排序确定性."""

    def test_t9_same_input_same_output(self):
        scores = pd.Series([0.5, 0.3, 0.8, 0.1])
        codes = pd.Series(["000001", "300003", "600004", "688005"])
        r1 = rank_cross_section(scores, codes)
        r2 = rank_cross_section(scores, codes)
        pd.testing.assert_series_equal(r1, r2)

    def test_t9_descending_order(self):
        """大者 rank=1（默认 ascending=False）."""
        scores = pd.Series([0.5, 0.3, 0.8, 0.1])
        codes = pd.Series(["A", "B", "C", "D"])
        rank = rank_cross_section(scores, codes)
        # 0.8(C)=1, 0.5(A)=2, 0.3(B)=3, 0.1(D)=4
        assert list(rank) == [2, 3, 1, 4]

    def test_t9_nan_score_to_tail(self):
        """NaN score → rank 排到末尾."""
        scores = pd.Series([0.5, np.nan, 0.8, np.nan])
        codes = pd.Series(["A", "B", "C", "D"])
        rank = rank_cross_section(scores, codes)
        # 0.8(C)=1, 0.5(A)=2, B/D 排末尾（在 valid 中 mergesort 稳定，code "B" < "D" 先到）
        # 实现里 NaN 块按原 position 顺序，所以 B=3, D=4
        assert rank.iloc[2] == 1  # C
        assert rank.iloc[0] == 2  # A
        assert rank.iloc[1] in (3, 4)  # B
        assert rank.iloc[3] in (3, 4)  # D
        assert rank.iloc[1] != rank.iloc[3]


class TestT10TieBreakByStockCode:
    """T10. Score 相同时按 stock_code 字典序升序 tie-break ⭐."""

    def test_t10_three_tied_scores(self):
        """三只股 score 完全相同 → 字典序升序排."""
        scores = pd.Series([0.5, 0.5, 0.5])
        codes = pd.Series(["600000", "000001", "300001"])
        rank = rank_cross_section(scores, codes)
        # 字典序：000001 < 300001 < 600000 → rank 1, 2, 3
        # 但输入顺序是 [600000, 000001, 300001] → rank 应是 [3, 1, 2]
        assert list(rank) == [3, 1, 2]

    def test_t10_input_order_irrelevant(self):
        """不同 input 顺序，相同 score → 输出还原到原 position 后 rank 一致."""
        scores_a = pd.Series([0.5, 0.5, 0.5])
        codes_a = pd.Series(["600000", "000001", "300001"])
        # 倒序输入
        scores_b = pd.Series([0.5, 0.5, 0.5])
        codes_b = pd.Series(["300001", "000001", "600000"])

        rank_a = rank_cross_section(scores_a, codes_a)
        rank_b = rank_cross_section(scores_b, codes_b)

        # rank_a 中 600000 的 rank = 3
        # rank_b 中 600000 的 rank = 3（不论它出现在哪个 position）
        assert rank_a[codes_a == "600000"].iloc[0] == 3
        assert rank_b[codes_b == "600000"].iloc[0] == 3
        assert rank_a[codes_a == "000001"].iloc[0] == 1
        assert rank_b[codes_b == "000001"].iloc[0] == 1


class TestT11Top20Selection:
    """T11. Top20 唯一性 & 短样本退化."""

    def _make_signal_df(self, n_stocks: int, *, valid_n: int | None = None):
        """造 n_stocks 行 signal DataFrame，前 valid_n 个有效 score（valid_n=None 表全部有效）."""
        if valid_n is None:
            valid_n = n_stocks
        codes = [f"S{i:04d}" for i in range(n_stocks)]
        scores = [0.01 * (n_stocks - i) for i in range(n_stocks)]  # 降序，全 unique
        for i in range(valid_n, n_stocks):
            scores[i] = np.nan
        df = pd.DataFrame({
            "stock_code": codes,
            "score_A": scores,
        })
        df["rank_A"] = rank_cross_section(df["score_A"], df["stock_code"])
        return df

    def test_t11_universe_3500_exactly_20(self):
        df = self._make_signal_df(3500)
        sel = select_top_n(df, n=20)
        assert len(sel) == 20

    def test_t11_universe_15_returns_all(self):
        df = self._make_signal_df(15)
        sel = select_top_n(df, n=20)
        assert len(sel) == 15

    def test_t11_universe_20_exact(self):
        df = self._make_signal_df(20)
        sel = select_top_n(df, n=20)
        assert len(sel) == 20

    def test_t11_nan_scores_excluded(self):
        """30 只候选，仅前 10 有效 score → Top20 实际只返回 10."""
        df = self._make_signal_df(30, valid_n=10)
        sel = select_top_n(df, n=20)
        assert len(sel) == 10
        assert sel["score_A"].notna().all()


class TestT12DualSignalOutput:
    """T12. 双信号同步输出 ⭐ (§3.5)."""

    def _make_factor_df(self):
        return pd.DataFrame({
            "stock_code": ["000001", "300003", "600004"],
            "big_net_inflow": [1.7e7, 5e6, 3e7],
            "total_amount": [4.3e7, 2e7, 1e8],
            "float_mv": [1e9, 8e8, 5e9],
        })

    def test_t12_outputs_all_four_columns(self):
        df = self._make_factor_df()
        out = compute_signals(df)
        for col in ("score_A", "rank_A", "score_B", "rank_B"):
            assert col in out.columns

    def test_t12_score_A_main_signal(self):
        """score_A = big_net_inflow / float_mv."""
        df = self._make_factor_df()
        out = compute_signals(df)
        assert out["score_A"].iloc[0] == pytest.approx(1.7e7 / 1e9)
        assert out["score_A"].iloc[2] == pytest.approx(3e7 / 5e9)

    def test_t12_score_B_diagnostic_signal(self):
        """score_B = big_net_inflow / total_amount."""
        df = self._make_factor_df()
        out = compute_signals(df)
        assert out["score_B"].iloc[0] == pytest.approx(1.7e7 / 4.3e7)

    def test_t12_select_top_uses_rank_A_only(self):
        """select_top_n 默认按 rank_A 截取，不查 rank_B."""
        df = self._make_factor_df()
        out = compute_signals(df)
        sel = select_top_n(out, n=2)
        # 按 score_A: 000001=0.017, 300003=0.00625, 600004=0.006
        # 按 score_B: 000001=0.395, 300003=0.25, 600004=0.30
        # Top2 by rank_A → 000001, 300003
        assert set(sel["stock_code"]) == {"000001", "300003"}

    def test_t12_dual_signal_overlap_jaccard(self):
        assert dual_signal_overlap(["A", "B", "C"], ["B", "C", "D"]) == pytest.approx(2 / 4)
        assert dual_signal_overlap(["A"], ["A"]) == 1.0
        assert dual_signal_overlap([], []) == 0.0
        assert dual_signal_overlap(["A"], ["B"]) == 0.0

    def test_t12_target_top_n_default_20(self):
        assert TARGET_TOP_N == 20


# ============================================================================
# M4 组合/撮合层（T13-T21）
# ============================================================================


def _make_position(code: str, entry_step: int = 0, shares: float = 100.0,
                   cost_basis: float = 10000.0) -> Position:
    return Position(stock_code=code, entry_step=entry_step,
                    shares=shares, cost_basis=cost_basis)


class TestT13HoldingPeriodMin3Days:
    """T13. 持有期 ≥ 3 日约束 ⭐核心 (§6)."""

    def test_t13_step_1_locked(self):
        """T 买 (entry_step=0) → step=1 仍锁定."""
        positions = {"A": _make_position("A", entry_step=0)}
        locked = holdings_in_lock(positions, current_step=1)
        assert "A" in locked

    def test_t13_step_2_locked(self):
        positions = {"A": _make_position("A", entry_step=0)}
        assert "A" in holdings_in_lock(positions, current_step=2)

    def test_t13_step_3_still_locked(self):
        """step=3 仍锁定（持有期 = 3 个完整交易日：1/2/3 不可卖）."""
        positions = {"A": _make_position("A", entry_step=0)}
        assert "A" in holdings_in_lock(positions, current_step=3)

    def test_t13_step_4_unlocked(self):
        """step=4 才可卖."""
        positions = {"A": _make_position("A", entry_step=0)}
        assert "A" not in holdings_in_lock(positions, current_step=4)

    def test_t13_decide_orders_respects_lock(self):
        """持有期内即使不在 new_selected，也不发 SELL."""
        prev = {"A": _make_position("A", entry_step=0)}
        orders = decide_orders(
            prev_positions=prev,
            new_selected=set(),               # A 不在 new_selected
            holding_locked={"A"},             # 但仍在锁定
            cash_avail=0.0,
            total_assets=10000.0,
        )
        sells = [o for o in orders if o.side is Side.SELL]
        assert sells == []

    def test_t13_decide_orders_exits_after_lock(self):
        """已过持有期 + 不在 new_selected → 发 SELL."""
        prev = {"A": _make_position("A", entry_step=0)}
        orders = decide_orders(
            prev_positions=prev,
            new_selected=set(),
            holding_locked=set(),             # 已解锁
            cash_avail=0.0,
            total_assets=10000.0,
        )
        sells = [o for o in orders if o.side is Side.SELL]
        assert len(sells) == 1
        assert sells[0].stock_code == "A"

    def test_t13_holding_min_days_default_3(self):
        assert HOLDING_MIN_DAYS == 3


class TestT14NoDailyRebalance:
    """T14 Entry-only rebalance ⭐⭐⭐ 致命漏洞防线 (§6.1).

    review 阶段识别的核心漏洞：日度等权 rebalance 会在 1 月内吃光 alpha.
    """

    def test_t14_pool_40_to_42_only_2_buys_zero_trims(self):
        """持仓池 40 → 42 只 → 2 笔买单，0 trim 单（核心防线）."""
        prev_positions = {
            f"i{k:02d}": _make_position(f"i{k:02d}", entry_step=0, shares=100, cost_basis=1000)
            for k in range(1, 41)
        }
        # T+1：新 selected = {i03..i42}；i01/i02 退出 selected 但仍持有期锁定（40 只全锁）
        new_selected = {f"i{k:02d}" for k in range(3, 43)}
        holding_locked = set(prev_positions.keys())  # 全部锁定

        orders = decide_orders(
            prev_positions=prev_positions,
            new_selected=new_selected,
            holding_locked=holding_locked,
            cash_avail=100_000.0,
            total_assets=1_000_000.0,
        )
        buys = [o for o in orders if o.side is Side.BUY]
        sells = [o for o in orders if o.side is Side.SELL]
        trims = [o for o in orders if o.is_rebalance_trim]

        assert len(buys) == 2, f"应仅 2 笔新进场买单，实际 {len(buys)}"
        assert {o.stock_code for o in buys} == {"i41", "i42"}
        assert len(sells) == 0, "持有期锁定中不发卖单"
        assert len(trims) == 0, "持有期间绝对不发 trim 单"

    def test_t14_position_value_drift_no_trim(self):
        """某持仓涨 30% 致权重偏离 → 不发 trim（权重自然漂移）."""
        prev = {"A": _make_position("A", entry_step=0, shares=100, cost_basis=10000)}
        # 假设 close 涨 30% → 当前价值 = 13000；total = cash + 13000
        # 没有新 selected，也没有 unlocked，应该无任何订单
        orders = decide_orders(
            prev_positions=prev,
            new_selected={"A"},               # A 还在 selected → 不卖
            holding_locked={"A"},
            cash_avail=5000.0,
            total_assets=18000.0,             # 5000 cash + 13000 (A 涨 30%)
        )
        assert orders == []  # 0 trim, 0 buy, 0 sell

    def test_t14_total_order_count_equals_buys_plus_sells(self):
        """订单总数永远等于 buys + sells，没有第三种 trim 来源."""
        prev = {f"i{k:02d}": _make_position(f"i{k:02d}", entry_step=10) for k in range(1, 21)}
        # 5 只新进场 + 3 只退出（已解锁）
        new_selected = (set(prev.keys()) - {"i01", "i02", "i03"}) | {"x01", "x02", "x03", "x04", "x05"}
        orders = decide_orders(
            prev_positions=prev,
            new_selected=new_selected,
            holding_locked=set(),
            cash_avail=100_000.0,
            total_assets=1_000_000.0,
        )
        buys = [o for o in orders if o.side is Side.BUY]
        sells = [o for o in orders if o.side is Side.SELL]
        assert len(buys) == 5
        assert len(sells) == 3
        assert len(orders) == len(buys) + len(sells)
        assert all(not o.is_rebalance_trim for o in orders)


class TestT15EntryAllocationFormula:
    """T15. Entry 分配公式：min(cash/K, total/N) (§6.1)."""

    def test_t15_cash_per_below_top_limit(self):
        """cash=2e6, K=5, total=1e7, N=20 → per = min(4e5, 5e5) = 4e5."""
        per = per_position_target(cash_avail=2e6, k_new_entries=5, total_assets=1e7, n_target=20)
        assert per == pytest.approx(4e5)

    def test_t15_cash_per_above_top_limit_capped(self):
        """cash=2e6, K=2, total=1e7, N=20 → per = min(1e6, 5e5) = 5e5."""
        per = per_position_target(cash_avail=2e6, k_new_entries=2, total_assets=1e7, n_target=20)
        assert per == pytest.approx(5e5)

    def test_t15_zero_entries(self):
        per = per_position_target(cash_avail=2e6, k_new_entries=0, total_assets=1e7, n_target=20)
        assert per == 0.0


class TestT16LimitUpSkipBuy:
    """T16. T+1 一字涨停跳过买入 ⭐ (§7.1)."""

    def test_t16_limit_up_skip(self):
        state = PortfolioState(cash=1_000_000.0, current_step=1)
        orders = [Order(stock_code="A", side=Side.BUY, target_amount=10_000.0)]
        result = apply_orders_at_t1(
            state=state,
            orders=orders,
            t1_open={"A": 10.0},
            t1_limit_up_flag={"A": True},     # 一字涨停
            t1_limit_down_flag={"A": False},
            cap_buy_amount={"A": 100_000.0},
        )
        assert state.cash == pytest.approx(1_000_000.0)  # cash 未动
        assert "A" not in state.positions
        assert len(result.fills) == 0
        assert len(result.unfilled) == 1

    def test_t16_normal_open_buy_executes(self):
        state = PortfolioState(cash=1_000_000.0, current_step=1)
        orders = [Order(stock_code="A", side=Side.BUY, target_amount=10_000.0)]
        result = apply_orders_at_t1(
            state=state,
            orders=orders,
            t1_open={"A": 10.0},
            t1_limit_up_flag={"A": False},
            t1_limit_down_flag={"A": False},
            cap_buy_amount={"A": 100_000.0},  # cap 远大于 target
        )
        assert "A" in state.positions
        assert len(result.fills) == 1
        assert result.fills[0].fill_amount == pytest.approx(10_000.0)


class TestT17LimitDownPostponeSell:
    """T17. T+1 一字跌停顺延卖出 (§7.1)."""

    def test_t17_limit_down_keeps_position(self):
        state = PortfolioState(
            cash=0.0, current_step=4,
            positions={"A": _make_position("A", entry_step=0, shares=1000, cost_basis=10000)},
        )
        orders = [Order(stock_code="A", side=Side.SELL)]
        result = apply_orders_at_t1(
            state=state,
            orders=orders,
            t1_open={"A": 10.0},
            t1_limit_up_flag={"A": False},
            t1_limit_down_flag={"A": True},   # 一字跌停
            cap_buy_amount={},
        )
        assert "A" in state.positions
        assert state.cash == pytest.approx(0.0)
        assert len(result.fills) == 0
        assert len(result.unfilled) == 1

    def test_t17_one_word_limit_predicates(self):
        assert is_one_word_limit_up(11.0, 11.0, 11.0, 11.0) is True
        assert is_one_word_limit_up(11.0, 11.0, 10.9, 11.0) is False
        assert is_one_word_limit_down(9.0, 9.0, 9.0, 9.0) is True
        assert is_one_word_limit_down(9.0, 9.5, 9.0, 9.0) is False


class TestT18CapacityConstraintTruncate:
    """T18. 容量约束截断 ⭐ (§7.2)."""

    def test_t18_target_below_cap_full_fill(self):
        """target=10 万, cap=25 万 → fill=10 万, unfilled=0."""
        state = PortfolioState(cash=1_000_000.0, current_step=1)
        orders = [Order(stock_code="A", side=Side.BUY, target_amount=100_000.0)]
        result = apply_orders_at_t1(
            state=state,
            orders=orders,
            t1_open={"A": 10.0},
            t1_limit_up_flag={}, t1_limit_down_flag={},
            cap_buy_amount={"A": 250_000.0},
        )
        assert result.fills[0].fill_amount == pytest.approx(100_000.0)
        assert len(result.unfilled) == 0

    def test_t18_target_above_cap_truncated(self):
        """target=50 万, cap=12.5 万 → fill=12.5 万, unfilled=37.5 万."""
        state = PortfolioState(cash=1_000_000.0, current_step=1)
        orders = [Order(stock_code="A", side=Side.BUY, target_amount=500_000.0)]
        result = apply_orders_at_t1(
            state=state,
            orders=orders,
            t1_open={"A": 10.0},
            t1_limit_up_flag={}, t1_limit_down_flag={},
            cap_buy_amount={"A": 125_000.0},
        )
        assert result.fills[0].fill_amount == pytest.approx(125_000.0)
        assert len(result.unfilled) == 1
        assert result.unfilled[0].target_amount == pytest.approx(375_000.0)

    def test_t18_capacity_cap_formula(self):
        """avg_20d=5000 万 → cap = 0.25% × 5000 万 = 12.5 万."""
        assert capacity_cap_for_buy(5e7) == pytest.approx(125_000.0)
        assert capacity_cap_for_buy(1e8) == pytest.approx(250_000.0)

    def test_t18_zero_avg_amount_zero_cap(self):
        """流动性 0 → cap 0 → 全部 unfilled（universe 应已剔除，兜底）."""
        state = PortfolioState(cash=1_000_000.0, current_step=1)
        orders = [Order(stock_code="A", side=Side.BUY, target_amount=100_000.0)]
        result = apply_orders_at_t1(
            state=state,
            orders=orders,
            t1_open={"A": 10.0},
            t1_limit_up_flag={}, t1_limit_down_flag={},
            cap_buy_amount={"A": 0.0},
        )
        assert len(result.fills) == 0
        assert len(result.unfilled) == 1


class TestT19UnfilledCarryover:
    """T19. Unfilled 顺延逻辑 (§7.2)."""

    def test_t19_unfilled_amount_recorded(self):
        """target=40 万, cap=10 万 → fill=10 万, unfilled Order 的 target_amount=30 万."""
        state = PortfolioState(cash=1_000_000.0, current_step=1)
        orders = [Order(stock_code="A", side=Side.BUY, target_amount=400_000.0)]
        result = apply_orders_at_t1(
            state=state,
            orders=orders,
            t1_open={"A": 10.0},
            t1_limit_up_flag={}, t1_limit_down_flag={},
            cap_buy_amount={"A": 100_000.0},
        )
        assert result.fills[0].fill_amount == pytest.approx(100_000.0)
        assert result.unfilled[0].target_amount == pytest.approx(300_000.0)

    def test_t19_fill_position_holding_period_starts_at_fill_step(self):
        """fill 入仓的 entry_step = state.current_step（顺延后再 fill 时 entry 推到当时 step）."""
        state = PortfolioState(cash=1_000_000.0, current_step=5)
        orders = [Order(stock_code="A", side=Side.BUY, target_amount=10_000.0)]
        apply_orders_at_t1(
            state=state,
            orders=orders,
            t1_open={"A": 10.0},
            t1_limit_up_flag={}, t1_limit_down_flag={},
            cap_buy_amount={"A": 100_000.0},
        )
        assert state.positions["A"].entry_step == 5


class TestT20ExitFullClose:
    """T20. 卖出全清 (§6.1)."""

    def test_t20_sell_clears_full_shares(self):
        """卖出条件触发 → 一次性清空 position."""
        state = PortfolioState(
            cash=0.0, current_step=4,
            positions={"A": _make_position("A", entry_step=0, shares=1000, cost_basis=10000)},
        )
        orders = [Order(stock_code="A", side=Side.SELL)]
        result = apply_orders_at_t1(
            state=state,
            orders=orders,
            t1_open={"A": 12.0},   # 涨 20%
            t1_limit_up_flag={}, t1_limit_down_flag={},
            cap_buy_amount={},
        )
        assert "A" not in state.positions
        # 单笔卖单（不分批）
        sell_fills = [f for f in result.fills if f.side is Side.SELL]
        assert len(sell_fills) == 1
        assert sell_fills[0].fill_shares == pytest.approx(1000.0)
        # 现金回收（扣 sell_cost）
        gross = 1000 * 12.0  # 12000
        expected_cash = gross - sell_cost(gross)
        assert state.cash == pytest.approx(expected_cash)


class TestT21TransactionCost:
    """T21. 成本计算 (§7.1)."""

    def test_t21_buy_cost_75_per_100k(self):
        """买入 10 万 → 75 元 ≈ 0.075%（佣金 25 + 滑点 50）."""
        assert buy_cost(100_000.0) == pytest.approx(75.0)

    def test_t21_sell_cost_125_per_100k(self):
        """卖出 10 万 → 125 元 ≈ 0.125%（佣金 25 + 印花税 50 + 滑点 50）."""
        assert sell_cost(100_000.0) == pytest.approx(125.0)

    def test_t21_round_trip_200_per_100k(self):
        """往返 10 万 = 75 + 125 = 200 元 ≈ 0.20%."""
        assert buy_cost(100_000.0) + sell_cost(100_000.0) == pytest.approx(200.0)

    def test_t21_cost_rate_constants(self):
        assert COMMISSION_RATE == 0.00025
        assert STAMP_DUTY_RATE == 0.0005
        assert SLIPPAGE_RATE == 0.0005
        assert CAP_BUY_RATIO == pytest.approx(0.0025)


# ============================================================================
# M5 评估层（T22-T29）
# ============================================================================


def _baseline_metrics(**overrides) -> StrategyMetrics:
    """造一个"全部通过门槛"的 metrics fixture；按 overrides 改单项."""
    base = dict(
        ic_a=ICStats(mean=0.03, std=0.10, t_stat=4.84, n=260),
        ic_b=ICStats(mean=0.025, std=0.10, t_stat=4.0, n=260),
        cumulative_return=0.20,
        annualized_return=0.18,
        excess_return_bm1=0.06,
        excess_return_bm2=0.04,
        ir_bm1=0.5,
        ir_bm2=0.4,
        max_drawdown=-0.20,
        max_relative_dd_bm1=-0.10,
        monthly_turnover=0.40,
        sharpe=1.0,
        overlap_ab=0.7,
    )
    base.update(overrides)
    return StrategyMetrics(**base)


class TestT22ICComputation:
    """T22. IC 计算正确性 ⭐."""

    def test_t22_perfect_positive_correlation(self):
        scores = pd.Series([1, 2, 3, 4, 5])
        rets = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
        assert compute_ic(scores, rets) == pytest.approx(1.0)

    def test_t22_perfect_negative_correlation(self):
        scores = pd.Series([1, 2, 3, 4, 5])
        rets = pd.Series([0.5, 0.4, 0.3, 0.2, 0.1])
        assert compute_ic(scores, rets) == pytest.approx(-1.0)

    def test_t22_no_correlation_near_zero(self):
        """随机配对 → IC ≈ 0（容差大些 0.3，5 个样本噪声大）."""
        np.random.seed(42)
        scores = pd.Series(np.random.randn(100))
        rets = pd.Series(np.random.randn(100))
        ic = compute_ic(scores, rets)
        assert abs(ic) < 0.3

    def test_t22_tstat_260days(self):
        """260 日 IC mean=0.03 std=0.10 → t ≈ 0.03/(0.10/√260) ≈ 4.84."""
        daily = pd.Series([0.03] * 260)
        # mean = 0.03, std = 0 → 我们手动构造非零 std
        rng = np.random.default_rng(0)
        daily = pd.Series(0.03 + 0.10 * rng.standard_normal(260))
        stats = compute_ic_stats(daily)
        assert stats.n == 260
        assert stats.mean == pytest.approx(0.03, abs=0.02)
        # t-stat 应在 4.84 附近；容差给宽
        assert stats.t_stat == pytest.approx(stats.mean / (stats.std / np.sqrt(260)), rel=1e-9)

    def test_t22_nan_excluded(self):
        """某些日期 NaN → 不入均值."""
        daily = pd.Series([0.02, np.nan, 0.03, np.nan, 0.04])
        stats = compute_ic_stats(daily)
        assert stats.n == 3
        assert stats.mean == pytest.approx(0.03)

    def test_t22_empty_returns_nan(self):
        stats = compute_ic_stats(pd.Series([np.nan, np.nan]))
        assert stats.n == 0
        assert np.isnan(stats.mean)

    def test_t22_passes_method(self):
        passing = ICStats(mean=0.03, std=0.10, t_stat=4.0, n=260)
        failing_mean = ICStats(mean=0.015, std=0.10, t_stat=4.0, n=260)
        failing_t = ICStats(mean=0.03, std=0.10, t_stat=2.5, n=260)
        assert passing.passes()
        assert not failing_mean.passes()
        assert not failing_t.passes()


class TestT23DualBenchmarkExcessReturn:
    """T23. 双基准超额计算 ⭐."""

    def test_t23_annualize_geometric(self):
        """13 月（即 273 个交易日近似）累计 15% → 年化 ≈ 13.8%."""
        n = 273  # 13 months × 21
        # cumulative 15%
        ann = annualize_return(0.15, n_periods=n, periods_per_year=252)
        # base=1.15, exp = 252/273 ≈ 0.923 → 1.15^0.923 ≈ 1.1375 → 13.75%
        assert ann == pytest.approx(0.138, abs=0.005)

    def test_t23_information_ratio(self):
        strat = pd.Series([0.001] * 252)  # 日均 0.1%
        bench = pd.Series([0.0005] * 252)
        # diff = 0.0005，std = 0 → IR = 0 (设计安排零除返回 0)
        ir = compute_information_ratio(strat, bench)
        assert ir == 0.0

    def test_t23_information_ratio_with_volatility(self):
        """构造已知 mean / std 的 diff 序列：±X 围绕 Y 振荡 → 可算出 IR 闭式解."""
        # diff 序列: 在 Y=0.005 上下波动 ±0.01 → mean=0.005, std (ddof=1) 可算
        diff_array = np.array([0.015, -0.005] * 126)  # 252 days
        strat = pd.Series(diff_array)
        bench = pd.Series([0.0] * 252)
        ir = compute_information_ratio(strat, bench)
        # 闭式期望
        expected_mean = float(diff_array.mean())          # 0.005
        expected_std = float(diff_array.std(ddof=1))      # ≈ 0.01002
        expected_ir = expected_mean / expected_std * np.sqrt(252)
        assert ir == pytest.approx(expected_ir, rel=1e-6)
        assert expected_ir > 7.0  # 量级 sanity check

    def test_t23_max_drawdown(self):
        equity = pd.Series([1.0, 1.2, 0.9, 1.1, 0.8, 1.3])
        # 最大跌幅：1.2 → 0.8 = -33%
        assert compute_max_drawdown(equity) == pytest.approx(-1.0 / 3, abs=0.01)

    def test_t23_sharpe(self):
        rng = np.random.default_rng(2)
        daily = pd.Series(0.001 + 0.01 * rng.standard_normal(252))
        sharpe = compute_sharpe(daily)
        # mean ≈ 0.001, std ≈ 0.01 → annualized sharpe ≈ 0.001/0.01 × √252 ≈ 1.59
        assert sharpe == pytest.approx(1.59, abs=0.5)


class TestT24VerdictChurning:
    """T24. CHURNING 分支触发."""

    def test_t24_ic_pass_but_excess_below_3pct(self):
        m = _baseline_metrics(excess_return_bm1=0.025)
        v, notes = render_verdict(m)
        assert v is Verdict.CHURNING
        assert notes["reason"] == "ic_pass_but_cost_eats_alpha"

    def test_t24_red_line_constant(self):
        assert CHURNING_RED_LINE == 0.03


class TestT25VerdictSizeBetaOnly:
    """T25. SIZE-BETA-ONLY 分支触发 ⭐."""

    def test_t25_bm1_pass_bm2_negative(self):
        m = _baseline_metrics(excess_return_bm1=0.07, excess_return_bm2=-0.01)
        v, notes = render_verdict(m)
        assert v is Verdict.SIZE_BETA_ONLY

    def test_t25_bm2_zero_triggers_size_beta(self):
        """vs BM2 = 0 → 也算 SIZE-BETA-ONLY（≤ 0 触发）."""
        m = _baseline_metrics(excess_return_bm1=0.07, excess_return_bm2=0.0)
        v, _ = render_verdict(m)
        assert v is Verdict.SIZE_BETA_ONLY


class TestT26VerdictNoAlpha:
    """T26. NO-ALPHA 分支触发."""

    def test_t26_ic_mean_below_threshold(self):
        m = _baseline_metrics(ic_a=ICStats(mean=0.015, std=0.10, t_stat=4.0, n=260))
        v, notes = render_verdict(m)
        assert v is Verdict.NO_ALPHA
        assert notes["reason"] == "ic_a_failed"

    def test_t26_tstat_below_threshold(self):
        m = _baseline_metrics(ic_a=ICStats(mean=0.025, std=0.10, t_stat=2.5, n=260))
        v, _ = render_verdict(m)
        assert v is Verdict.NO_ALPHA

    def test_t26_threshold_constants(self):
        assert IC_MEAN_THRESHOLD == 0.02
        assert IC_TSTAT_THRESHOLD == 3.0


class TestT27VerdictPassPrelim:
    """T27. PASS-prelim 分支触发."""

    def test_t27_all_thresholds_pass(self):
        m = _baseline_metrics()  # 默认 fixture 全过门槛
        v, notes = render_verdict(m)
        assert v is Verdict.PASS_PRELIM
        assert notes["reason"] == "all_thresholds_pass"


class TestT28VerdictUnstable:
    """T28. UNSTABLE 检测."""

    def test_t28_consecutive_3m_loss_triggers_unstable(self):
        """3 连月 vs BM2 累计 -12% → UNSTABLE."""
        m = _baseline_metrics()
        monthly_excess = pd.Series([0.01, 0.02, -0.05, -0.05, -0.05, 0.01])  # 中间 3 月 -15%
        v, _ = render_verdict(m, monthly_excess_vs_bm2=monthly_excess)
        assert v is Verdict.UNSTABLE

    def test_t28_no_consecutive_loss_no_trigger(self):
        m = _baseline_metrics()
        monthly_excess = pd.Series([0.01, -0.03, 0.02, -0.03, 0.02, 0.01])
        v, _ = render_verdict(m, monthly_excess_vs_bm2=monthly_excess)
        assert v is Verdict.PASS_PRELIM

    def test_t28_detect_unstable_window_function(self):
        s = pd.Series([0.01, -0.04, -0.05, -0.05])  # 后 3 个累计 -14%
        assert detect_unstable_window(s) is True
        s2 = pd.Series([-0.04, 0.05, -0.04, 0.05])  # 任何 3 连无 < -10%
        assert detect_unstable_window(s2) is False


class TestT29DualSignalCrossCheck:
    """T29. 双信号交叉验证 ⭐ (§3.5)."""

    def _ic(self, mean: float, t_stat: float) -> ICStats:
        return ICStats(mean=mean, std=0.10, t_stat=t_stat, n=260)

    def test_t29_dual_pass_high_overlap(self):
        status = classify_dual_signal(
            self._ic(0.03, 4.0),
            self._ic(0.025, 3.5),
            overlap=0.65,
        )
        assert status is DualSignalStatus.DUAL_CONFIRMED

    def test_t29_a_pass_b_fail(self):
        status = classify_dual_signal(
            self._ic(0.03, 4.0),
            self._ic(0.01, 2.0),
            overlap=0.5,
        )
        assert status is DualSignalStatus.ALPHA_MAY_BE_BETA_DRIVEN

    def test_t29_a_fail_b_pass(self):
        status = classify_dual_signal(
            self._ic(0.01, 2.0),
            self._ic(0.03, 4.0),
            overlap=0.5,
        )
        assert status is DualSignalStatus.SWITCH_PRIMARY_SIGNAL

    def test_t29_dual_fail(self):
        status = classify_dual_signal(
            self._ic(0.01, 2.0),
            self._ic(0.005, 1.5),
            overlap=0.3,
        )
        assert status is DualSignalStatus.NO_ALPHA

    def test_t29_dual_pass_low_overlap_warns(self):
        """双 PASS 但重合度 < 60% → 保守标 may_be_beta_driven."""
        status = classify_dual_signal(
            self._ic(0.03, 4.0),
            self._ic(0.03, 4.0),
            overlap=0.40,
        )
        assert status is DualSignalStatus.ALPHA_MAY_BE_BETA_DRIVEN
