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
