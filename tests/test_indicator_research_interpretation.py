import pandas as pd

from indicator_research.interpretation import (
    describe_direction,
    describe_donchian,
    describe_momentum,
    describe_regime,
    describe_volatility,
    describe_volume_confirmation,
)


def make_frame(**columns):
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-09-28", periods=21, freq="B"),
            **columns,
        }
    )


def test_direction_description_combines_long_trend_and_macd():
    frame = make_frame(
        close=[90.0] * 21,
        sma200=[100.0] * 21,
        sma200_slope20=[-0.02] * 21,
        dif=[-0.5] * 21,
        dea=[-0.4] * 21,
        macd_hist=[-0.2] * 21,
    )

    result = describe_direction(frame)

    assert "长期空头参考" in result
    assert "零轴下方的负向动量" in result
    assert "同向偏弱" in result
    assert "截至 2026-10-26" in result


def test_donchian_description_uses_previous_channel_for_breakout():
    frame = make_frame(
        close=[100.0] * 20 + [111.0],
        donchian_upper20=[110.0] * 20 + [112.0],
        donchian_lower20=[90.0] * 21,
    )

    result = describe_donchian(frame)

    assert "向上突破观察" in result
    assert "前一交易日通道上轨 110.00" in result
    assert "区间的 95.5% 位置" in result


def test_momentum_description_applies_extreme_thresholds_without_reversal_claim():
    frame = make_frame(
        rsi14=[72.0] * 21,
        kdj_k=[85.0] * 21,
        kdj_d=[80.0] * 21,
        kdj_j=[95.0] * 21,
        roc20=[-2.0] * 21,
        cci20=[120.0] * 21,
    )

    result = describe_momentum(frame)

    assert "超买参考区（72.00）" in result
    assert "高位参考区" in result
    assert "+100 上方的强势 / 超买参考区" in result
    assert "分歧或转换阶段" in result
    assert "超买不等于立即下跌" in result


def test_volatility_description_distinguishes_squeeze_from_direction():
    frame = make_frame(
        close=[110.0] * 21,
        bb_pct_b=[1.1] * 21,
        bb_width=[0.08] * 21,
        bb_width_rank126=[0.05] * 21,
        natr14=[2.3] * 21,
    )

    result = describe_volatility(frame)

    assert "上轨之外" in result
    assert "低分位压缩（Squeeze 参考）" in result
    assert "边界试探" in result
    assert "不预判方向" in result


def test_volume_description_compares_price_and_obv_over_20_sessions():
    frame = make_frame(
        close=[100.0] * 20 + [110.0],
        obv=[0.0] * 20 + [1_000_000.0],
        mfi14=[85.0] * 21,
        volume_ratio20=[2.2] * 21,
        volume_direction=["bull"] * 21,
        obv_bullish_divergence=[False] * 20 + [True],
        obv_bearish_divergence=[False] * 21,
        mfi_bullish_divergence=[False] * 21,
        mfi_bearish_divergence=[False] * 20 + [True],
    )

    result = describe_volume_confirmation(frame)

    assert "量价同向偏强" in result
    assert "+10.00%" in result
    assert "超买参考区（85.00）" in result
    assert "显著放量" in result
    assert "图中牛熊区间：牛市参考" in result
    assert "OBV 底背离 1 次、顶背离 0 次" in result
    assert "MFI 底背离 0 次、顶背离 1 次" in result
    assert "标记在确认日" in result


def test_regime_description_combines_strength_chop_and_direction():
    frame = make_frame(
        adx14=[30.0] * 21,
        plus_di=[30.0] * 21,
        minus_di=[15.0] * 21,
        chop14=[35.0] * 21,
        er10=[0.4] * 21,
        vhf20=[0.6] * 21,
        reg_slope20=[0.001] * 21,
        reg_r2_20=[0.7] * 21,
    )

    result = describe_regime(frame)

    assert "有趋势参考（30.00）" in result
    assert "趋势参考（35.00）" in result
    assert "同向支持趋势状态" in result
    assert "方向参考一致" in result
    assert "线性拟合质量较高" in result
