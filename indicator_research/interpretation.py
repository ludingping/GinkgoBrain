"""Dynamic Chinese interpretations for the indicator research notebook."""
import pandas as pd


def _format_status(title: str, data: pd.DataFrame, items: list[str]) -> str:
    if "date" in data.columns:
        latest_date = pd.to_datetime(data["date"].iloc[-1]).date()
    else:
        latest_date = pd.to_datetime(data.index[-1]).date()
    bullets = "\n".join(f"- {item}" for item in items)
    return (
        f"**{title}｜截至 {latest_date}**\n\n"
        f"{bullets}\n\n"
        "*以上为指标状态说明，经典阈值仅作观察参考，不构成独立交易信号。*"
    )


def describe_direction(data: pd.DataFrame) -> str:
    latest = data.iloc[-1]
    close = latest["close"]
    sma200 = latest["sma200"]
    slope = latest["sma200_slope20"]
    dif = latest["dif"]
    dea = latest["dea"]
    hist = latest["macd_hist"]

    long_bullish = close >= sma200 and slope > 0
    long_bearish = close < sma200 and slope < 0
    if long_bullish:
        long_state = "长期多头参考"
        long_reason = "价格位于长均线上方，且长均线本身上行"
    elif long_bearish:
        long_state = "长期空头参考"
        long_reason = "价格位于长均线下方，且长均线本身下行"
    else:
        long_state = "长期方向分歧 / 过渡"
        long_reason = "价格位置与长均线斜率没有同向确认"

    if hist > 0:
        macd_state = "零轴上方的正向动量" if dif > 0 else "零轴下方的修复性动量"
        macd_reason = "DIF > DEA，短期动量改善"
    elif hist < 0:
        macd_state = "零轴下方的负向动量" if dif < 0 else "零轴上方的回落动量"
        macd_reason = "DIF < DEA，短期动量走弱"
    else:
        macd_state = "DIF 与 DEA 持平"
        macd_reason = "短期动量处于交叉临界"

    if long_bullish and hist > 0:
        combined = "长期方向与 MACD 同向偏强，趋势延续仍需量价确认。"
    elif long_bearish and hist < 0:
        combined = "长期方向与 MACD 同向偏弱，短期超卖不能单独视为趋势反转。"
    else:
        combined = "长期方向与 MACD 未形成同向确认，按过渡或分歧状态观察。"

    return _format_status("方向层状态解读", data, [
        (
            f"**长期方向：{long_state}。** 收盘价 ≥ SMA200 且 SMA200 的 20 日斜率 > 0 为多头参考；"
            f"收盘价 < SMA200 且斜率 < 0 为空头参考。当前收盘 {close:.2f}，"
            f"相对 SMA200 为 {close / sma200 - 1:+.2%}，斜率 {slope:+.2%}，{long_reason}。"
        ),
        (
            f"**MACD：{macd_state}。** DIF = {dif:.3f}，DEA = {dea:.3f}，"
            f"柱 = {hist:.3f}；{macd_reason}。DIF 的零轴位置用于区分趋势背景，"
            "DIF 与 DEA 的关系用于观察短期动量。"
        ),
        f"**综合解读：**{combined} SMA20/50/250 作为不同周期的价格位置参照，不额外投票。",
    ])


def describe_donchian(data: pd.DataFrame) -> str:
    latest = data.iloc[-1]
    previous = data.iloc[-2]
    close = latest["close"]
    upper = latest["donchian_upper20"]
    lower = latest["donchian_lower20"]
    previous_upper = previous["donchian_upper20"]
    previous_lower = previous["donchian_lower20"]

    if close > previous_upper:
        breakout_state = "向上突破观察"
        breakout_reason = f"收盘价高于前一交易日通道上轨 {previous_upper:.2f}"
    elif close < previous_lower:
        breakout_state = "向下突破观察"
        breakout_reason = f"收盘价低于前一交易日通道下轨 {previous_lower:.2f}"
    else:
        breakout_state = "通道内运行"
        breakout_reason = "收盘价尚未越出前一交易日的 20 日通道"

    width = upper - lower
    if width > 0:
        position = (close - lower) / width
        position_text = (
            f"当前通道为 {lower:.2f}—{upper:.2f}，收盘位于区间的 {position:.1%} 位置；"
            "0% 对应下轨，100% 对应上轨。"
        )
    else:
        position_text = f"当前上下轨均为 {upper:.2f}，通道宽度为零，无法计算区间位置。"

    if breakout_state == "通道内运行":
        interpretation = "当前主要是区间位置信息；接近上沿或下沿本身不保证突破或反转。"
    else:
        interpretation = "目前只满足单日收盘突破条件，仍需观察后续能否维持在边界外，并结合量能确认。"

    items = [
        (
            f"**当日突破状态：{breakout_state}。** 判据使用前一交易日通道，避免把当日高低点纳入比较基准；"
            f"收盘 > 前一日上轨为向上突破观察，收盘 < 前一日下轨为向下突破观察。{breakout_reason}。"
        ),
    ]
    if "donchian_direction" in data.columns:
        direction_state = latest["donchian_direction"]
        direction_label = {"bull": "牛市参考", "bear": "熊市参考"}.get(direction_state, "尚未形成方向")
        items.append(
            f"**图中方向区间：{direction_label}。** 背景状态在突破后持续至反向突破，"
            "与当日是否仍在通道内是两个不同维度；同向重复突破不重新标记起点。"
        )
    items.extend([
        f"**区间位置：**{position_text}",
        f"**解读：**{interpretation}通道描述近期价格区间，不单独预测趋势方向。",
    ])
    return _format_status("唐奇安通道状态解读", data, items)


def describe_momentum(data: pd.DataFrame) -> str:
    latest = data.iloc[-1]
    rsi = latest["rsi14"]
    k = latest["kdj_k"]
    d = latest["kdj_d"]
    j = latest["kdj_j"]
    roc = latest["roc20"]
    cci = latest["cci20"]

    if rsi >= 70:
        rsi_state = "超买参考区"
    elif rsi <= 30:
        rsi_state = "超卖参考区"
    elif rsi >= 50:
        rsi_state = "中性偏强"
    else:
        rsi_state = "中性偏弱"

    if k >= 80 and d >= 80:
        kdj_zone = "高位参考区"
    elif k <= 20 and d <= 20:
        kdj_zone = "低位参考区"
    else:
        kdj_zone = "非极端区"
    if k > d:
        kdj_direction = "K > D，短期动量向上"
    elif k < d:
        kdj_direction = "K < D，短期动量向下"
    else:
        kdj_direction = "K = D，短期方向持平"

    if roc > 0:
        roc_state = "正动量"
        roc_reason = "收盘价高于 20 个交易日前"
    elif roc < 0:
        roc_state = "负动量"
        roc_reason = "收盘价低于 20 个交易日前"
    else:
        roc_state = "零动量"
        roc_reason = "收盘价与 20 个交易日前持平"

    if cci >= 100:
        cci_state = "+100 上方的强势 / 超买参考区"
    elif cci <= -100:
        cci_state = "−100 下方的弱势 / 超卖参考区"
    elif cci > 0:
        cci_state = "常态区内偏强"
    elif cci < 0:
        cci_state = "常态区内偏弱"
    else:
        cci_state = "常态区中轴"

    if roc > 0 and k > d:
        combined = "ROC 为正且 K > D，短期动量偏改善。"
    elif roc < 0 and k < d:
        combined = "ROC 为负且 K < D，短期动量偏弱。"
    else:
        combined = "ROC 与 KDJ 的方向参考不一致，短期动量处于分歧或转换阶段。"

    in_range = 30 <= rsi <= 70 and 20 <= k <= 80 and 20 <= d <= 80 and -100 <= cci <= 100
    range_state = "当前处于" if in_range else "当前不处于"

    return _format_status("动量层状态解读", data, [
        (
            f"**RSI14：{rsi_state}（{rsi:.2f}）。** ≥ 70 为超买参考，≤ 30 为超卖参考；"
            "其余区间再用 50 中轴区分偏强或偏弱。极值表示动量温度，不自动意味着反转。"
        ),
        (
            f"**KDJ：{kdj_zone}；{kdj_direction}。** K = {k:.2f}，D = {d:.2f}，J = {j:.2f}；"
            "K、D 均 ≥ 80 为高位参考，均 ≤ 20 为低位参考，强趋势中可能持续钝化。"
        ),
        f"**ROC20：{roc_state}（{roc:+.2f}%）。** 以 0 为判据，{roc_reason}。",
        (
            f"**CCI20：{cci_state}（{cci:.2f}）。** ≥ +100 或 ≤ −100 为统计极端参考，"
            "−100 至 +100 内再用 0 轴观察强弱。"
        ),
        (
            f"**震荡观察：{range_state}动量震荡观察区。** 紫色区间要求 RSI14 在 30—70、"
            "K 与 D 在 20—80、CCI20 在 −100—+100 内同时成立；ROC 不参与判定，"
            "非极端动量不等同于价格必然横盘。"
        ),
        f"**综合解读：**{combined}超买不等于立即下跌，超卖也不等于立即上涨。",
    ])


def describe_volatility(data: pd.DataFrame) -> str:
    latest = data.iloc[-1]
    close = latest["close"]
    pct_b = latest["bb_pct_b"]
    width = latest["bb_width"]
    width_rank = latest["bb_width_rank126"]
    natr = latest["natr14"]

    if pct_b > 1:
        band_state = "上轨之外"
    elif pct_b < 0:
        band_state = "下轨之外"
    else:
        band_state = "布林带内"

    if width_rank <= 0.10:
        width_state = "低分位压缩（Squeeze 参考）"
    elif width_rank >= 0.90:
        width_state = "高分位扩张"
    else:
        width_state = "中间分位"

    if width_rank <= 0.10 and (pct_b > 1 or pct_b < 0):
        combined = "带宽仍在低分位而价格已到带外，属于压缩后的边界试探；是否转入扩张需看后续带宽，不预判方向。"
    elif width_rank <= 0.10:
        combined = "当前重点是波动压缩；未来可能扩张，但压缩本身不提供突破方向。"
    elif pct_b > 1 or pct_b < 0:
        combined = "价格相对 20 日波动区间处于极端位置，既可能趋势延续，也可能回归，不能仅凭带外逆向操作。"
    else:
        combined = "价格仍在带内，结合带宽分位观察波动扩张或收缩，不据此给出涨跌方向。"

    if "volatility_regime" in data.columns:
        regime = latest["volatility_regime"]
    else:
        expanding = width > data["bb_width"].iloc[-2]
        trending = (pct_b > 1 or pct_b < 0) and expanding
        ranging = 0 <= pct_b <= 1 and width_rank <= 0.10
        regime = "trend" if trending else "range" if ranging else "neutral"
    regime_labels = {"trend": "趋势扩张参考", "range": "震荡 / 压缩参考", "neutral": "中性 / 过渡"}
    regime_state = regime_labels.get(regime, "数据 warmup")

    return _format_status("波动层状态解读", data, [
        (
            f"**价格位置：{band_state}。** 收盘 {close:.2f}，%B = {pct_b:.3f}；"
            "%B > 1 表示收盘在上轨之外，%B < 0 表示在下轨之外，0—1 表示带内。"
        ),
        (
            f"**带宽状态：{width_state}。** BandWidth = {width:.2%}，126 日滚动分位 = {width_rank:.1%}；"
            "≤ 10% 为压缩参考，≥ 90% 为高分位扩张描述，其余为中间分位。"
        ),
        (
            f"**相对波动：NATR14 = {natr:.2f}%。** NATR = ATR14 / 收盘价；"
            "数值越高表示相对波动越大，没有固定牛熊阈值，也不指示方向。"
        ),
        (
            f"**图中波动状态：{regime_state}。** 带外且带宽扩大为橙色趋势扩张参考；"
            "带内且带宽分位 ≤ 10% 为蓝色震荡 / 压缩参考。圆点为起点，X 为状态失效的终点确认日；"
            "趋势状态不区分涨跌方向。"
        ),
        f"**综合解读：**{combined}",
    ])


def describe_volume_confirmation(data: pd.DataFrame) -> str:
    latest = data.iloc[-1]
    price_change20 = data["close"].pct_change(20).iloc[-1]
    obv_change20 = data["obv"].diff(20).iloc[-1]
    mfi = latest["mfi14"]
    volume_ratio = latest["volume_ratio20"]

    if price_change20 > 0 and obv_change20 > 0:
        confirmation = "量价同向偏强"
        confirmation_reason = "20 日价格上涨，OBV 也上升，累积量价方向相互确认"
    elif price_change20 < 0 and obv_change20 < 0:
        confirmation = "量价同向偏弱"
        confirmation_reason = "20 日价格下跌，OBV 也下降，弱势方向相互确认"
    elif price_change20 == 0 and obv_change20 == 0:
        confirmation = "量价均持平"
        confirmation_reason = "20 日价格与 OBV 均未发生变化"
    else:
        confirmation = "价格与 OBV 方向分歧"
        confirmation_reason = "20 日价格变化与 OBV 变化没有同向，后续确认弱于同向状态"

    if mfi >= 80:
        mfi_state = "超买参考区"
    elif mfi <= 20:
        mfi_state = "超卖参考区"
    elif mfi >= 50:
        mfi_state = "量价偏强区"
    else:
        mfi_state = "量价偏弱区"

    if volume_ratio >= 2:
        volume_state = "显著放量"
    elif volume_ratio >= 1:
        volume_state = "温和放量"
    else:
        volume_state = "相对缩量"

    items = [
        (
            f"**20 日量价配合：{confirmation}。** 价格变化 {price_change20:+.2%}，"
            f"OBV 相较 20 个交易日前{'上升' if obv_change20 > 0 else '下降' if obv_change20 < 0 else '持平'}；"
            f"{confirmation_reason}。OBV 的绝对水平不用于跨股票比较。"
        ),
        (
            f"**MFI14：{mfi_state}（{mfi:.2f}）。** ≥ 80 为超买参考，≤ 20 为超卖参考；"
            "其余区间以 50 观察量价强弱，不把 MFI 直接解释为确定性资金流入或流出。"
        ),
        (
            f"**量能状态：{volume_state}。** 当日成交量 / 20 日均量 = {volume_ratio:.2f}；"
            "≥ 2 为显著放量，1—2 为温和放量，< 1 为相对缩量。放量只表示活跃度提高，仍需结合价格方向。"
        ),
    ]
    if "volume_direction" in data.columns:
        direction_label = {
            "bull": "牛市参考",
            "bear": "熊市参考",
            "neutral": "量价分歧 / 过渡",
        }.get(latest["volume_direction"], "warmup")
        items.append(
            f"**图中牛熊区间：{direction_label}。** 红色要求 20 日价格与 OBV 同时上升，"
            "绿色要求两者同时下降，其余为灰色分歧/过渡；三角标记状态切换日，首段不补历史起点。"
        )
    divergence_summaries = []
    for prefix, label in [("obv", "OBV"), ("mfi", "MFI")]:
        bullish_column = f"{prefix}_bullish_divergence"
        bearish_column = f"{prefix}_bearish_divergence"
        if bullish_column in data.columns and bearish_column in data.columns:
            bullish_count = int(data[bullish_column].tail(20).sum())
            bearish_count = int(data[bearish_column].tail(20).sum())
            divergence_summaries.append(f"{label} 底背离 {bullish_count} 次、顶背离 {bearish_count} 次")
    if divergence_summaries:
        items.append(
            f"**近 20 个交易日新确认的背离：**{'；'.join(divergence_summaries)}。"
            "底背离为红色上三角，顶背离为绿色下三角；左右各 2 根 K 线确认摆动点，"
            "标记在确认日而非过去的摆动点，不预示必然反转。量比不单独判断背离。"
        )
    return _format_status("量价确认层状态解读", data, items)


def describe_regime(data: pd.DataFrame) -> str:
    latest = data.iloc[-1]
    adx = latest["adx14"]
    plus_di = latest["plus_di"]
    minus_di = latest["minus_di"]
    chop = latest["chop14"]
    er = latest["er10"]
    vhf = latest["vhf20"]
    slope = latest["reg_slope20"]
    r2 = latest["reg_r2_20"]

    if adx >= 40:
        adx_state = "强趋势参考"
    elif adx >= 25:
        adx_state = "有趋势参考"
    elif adx >= 20:
        adx_state = "趋势强度过渡区"
    else:
        adx_state = "弱趋势 / 震荡参考"

    if plus_di > minus_di:
        di_state = "+DI > −DI，方向运动偏多"
    elif plus_di < minus_di:
        di_state = "+DI < −DI，方向运动偏空"
    else:
        di_state = "+DI = −DI，方向运动持平"

    if chop > 61.8:
        chop_state = "震荡参考"
    elif chop < 38.2:
        chop_state = "趋势参考"
    else:
        chop_state = "趋势 / 震荡过渡区"

    slope_direction = "上升" if slope > 0 else "下降" if slope < 0 else "平坦"
    if r2 >= 0.5:
        r2_state = "较高"
    elif r2 >= 0.2:
        r2_state = "中等"
    else:
        r2_state = "较低"

    if adx >= 25 and chop < 38.2:
        regime_combined = "ADX 与 CHOP 同向支持趋势状态"
    elif adx < 20 and chop > 61.8:
        regime_combined = "ADX 与 CHOP 同向支持震荡状态"
    elif (adx >= 25 and chop > 61.8) or (adx < 20 and chop < 38.2):
        regime_combined = "ADX 与 CHOP 给出相反状态，按分歧处理"
    else:
        regime_combined = "ADX 与 CHOP 至少一项处于过渡区，暂不判为单一趋势或震荡状态"

    di_bullish = plus_di > minus_di
    di_bearish = plus_di < minus_di
    if (di_bullish and slope > 0) or (di_bearish and slope < 0):
        direction_combined = "DI 与回归斜率的方向参考一致"
    elif (di_bullish and slope < 0) or (di_bearish and slope > 0):
        direction_combined = "DI 与回归斜率的方向参考相反"
    else:
        direction_combined = "DI 或回归斜率至少一项方向持平"

    return _format_status("趋势 / 震荡状态解读", data, [
        (
            f"**ADX14：{adx_state}（{adx:.2f}）。** < 20 为弱趋势参考，20—25 为过渡区，"
            f"≥ 25 为有趋势参考，≥ 40 为强趋势参考；{di_state}（+DI {plus_di:.2f} / −DI {minus_di:.2f}）。"
        ),
        (
            f"**CHOP14：{chop_state}（{chop:.2f}）。** > 61.8 偏震荡，< 38.2 偏趋势，"
            "中间区域不做单一状态判断。"
        ),
        (
            f"**路径效率：ER10 = {er:.3f}，VHF20 = {vhf:.3f}。** 越高表示价格路径越顺畅，"
            "越低表示来回噪声越多；两者都不提供涨跌方向，本面板不设固定判级阈值。"
        ),
        (
            f"**回归趋势：**20 日 log 收盘价斜率 {slope * 10_000:+.1f} bp/day，方向{ slope_direction }；"
            f"R² = {r2:.3f}，线性拟合质量{r2_state}（≥ 0.5 较高，0.2—0.5 中等，< 0.2 较低）。"
        ),
        (
            f"**综合解读：**{regime_combined}；{direction_combined}。"
            "R² 只衡量线性拟合质量，趋势 / 震荡状态也不直接映射仓位。"
        ),
    ])
