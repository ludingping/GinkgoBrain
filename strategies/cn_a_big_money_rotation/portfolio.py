"""M4 组合/撮合层：持有期 + Entry-only rebalance + 容量截断（设计 §6, §7）.

⭐⭐⭐ 关键设计：仅在 entry/exit 两个时点发出订单；持有期间**不发任何 trim**.
此处是 review 阶段识别的致命漏洞防线（日度等权 rebalance 会在 1 月内吃光 alpha）.

成本（设计 §7.1）：
- 佣金 0.025%（双边）
- 印花税 0.05%（卖方）
- 滑点 0.05%（双边）
- 买入往返成本 ≈ 0.075% + 0.125% = 0.20%

容量约束（设计 §7.2）：
- v0.1 退化口径：cap_buy ≈ avg_20d_daily_amount × 0.25%
  （= 集合竞价占比 2.5% × 单笔容量上限 10%）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping


# ============================================================================
# 常量（设计 §6, §7）
# ============================================================================

HOLDING_MIN_DAYS = 3                # 持有期 ≥ 3 个完整交易日（设计 §6）
TARGET_N = 20                       # 目标 Top-N
COMMISSION_RATE = 0.00025           # 佣金 0.025%（双边）
STAMP_DUTY_RATE = 0.0005            # 印花税 0.05%（卖方）
SLIPPAGE_RATE = 0.0005              # 滑点 0.05%（双边，开盘集合竞价偏差）

# 容量约束：集合竞价成交额 ≈ 日成交额 × 2.5%；单笔买单上限 = 竞价成交额 × 10%
AUCTION_AMOUNT_RATIO = 0.025
CAPACITY_RATIO = 0.10
CAP_BUY_RATIO = AUCTION_AMOUNT_RATIO * CAPACITY_RATIO  # = 0.0025 = 0.25%


# ============================================================================
# 数据结构
# ============================================================================

class Side(Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Position:
    """单只持仓."""

    stock_code: str
    entry_step: int     # backtest step index（trade_date 在序列中的 0-based 位置）
    shares: float       # v0.1 不做 lot rounding
    cost_basis: float   # 含成本的累计本金（cash 流出额）


@dataclass
class Order:
    """订单（仅 entry/exit；持有期间不发 trim）."""

    stock_code: str
    side: Side
    target_amount: float = 0.0      # BUY: 期望成交金额；SELL: 0（按 state 全清）
    is_rebalance_trim: bool = False  # ⭐⭐⭐ 始终 False；T14 测试用于检测错误实现


@dataclass
class Fill:
    """单笔成交记录."""

    stock_code: str
    side: Side
    fill_amount: float    # 成交金额（不含成本）
    fill_shares: float
    fill_price: float     # 撮合用的开盘价（滑点反映在 cost 而非 price）
    cost: float           # 该笔成本总额（佣金 + 印花税 + 滑点）


@dataclass
class CostParams:
    commission_rate: float = COMMISSION_RATE
    stamp_duty_rate: float = STAMP_DUTY_RATE
    slippage_rate: float = SLIPPAGE_RATE


@dataclass
class FillResult:
    """T+1 撮合结果."""

    fills: list[Fill] = field(default_factory=list)
    unfilled: list[Order] = field(default_factory=list)


@dataclass
class PortfolioState:
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    current_step: int = 0

    def total_assets(self, last_close: Mapping[str, float]) -> float:
        v = self.cash
        for pos in self.positions.values():
            v += pos.shares * last_close.get(pos.stock_code, 0.0)
        return v


# ============================================================================
# 成本函数
# ============================================================================

def buy_cost(fill_amount: float, params: CostParams | None = None) -> float:
    """买入成本 = 金额 × (佣金 + 滑点)；≈ 0.075%."""
    p = params or CostParams()
    return fill_amount * (p.commission_rate + p.slippage_rate)


def sell_cost(fill_amount: float, params: CostParams | None = None) -> float:
    """卖出成本 = 金额 × (佣金 + 印花税 + 滑点)；≈ 0.125%."""
    p = params or CostParams()
    return fill_amount * (p.commission_rate + p.stamp_duty_rate + p.slippage_rate)


# ============================================================================
# 持有期 + 订单决策（设计 §6.1）
# ============================================================================

def holdings_in_lock(
    positions: Mapping[str, Position],
    current_step: int,
    min_days: int = HOLDING_MIN_DAYS,
) -> set[str]:
    """T 日仍在持有期锁定内的 stock_code 集合.

    设计 §6 口径："T 买、T+1/T+2/T+3 不可卖、T+4 才可卖" → 持有 3 个完整交易日.
    实现：(current_step - entry_step) <= min_days；即 step 差 ≤ 3 仍锁定，差 ≥ 4 解锁.
    """
    return {
        code for code, p in positions.items()
        if (current_step - p.entry_step) <= min_days
    }


def per_position_target(
    cash_avail: float,
    k_new_entries: int,
    total_assets: float,
    n_target: int = TARGET_N,
) -> float:
    """Entry 分配公式：min(cash/K, total/N).

    K=0 → 0；total/N 是单股目标权重上限.
    """
    if k_new_entries <= 0:
        return 0.0
    cash_per = cash_avail / k_new_entries
    top_limit = total_assets / max(1, n_target)
    return max(0.0, min(cash_per, top_limit))


def decide_orders(
    prev_positions: Mapping[str, Position],
    new_selected: set[str],
    holding_locked: set[str],
    cash_avail: float,
    total_assets: float,
    n_target: int = TARGET_N,
) -> list[Order]:
    """T+1 开盘订单决策（设计 §6.1）.

    ⭐⭐⭐ 仅产 entry (BUY) + exit (SELL) 两类订单；**绝不发出 trim 单**.
    持有期间各头寸权重随价格自然漂移.

    SELL：已过持有期 且 不在 new_selected → 全清（Order 不带金额，apply 时按 shares）.
    BUY：new_selected 中 ∉ prev_positions → 按 per_position_target 分配（升序字典序）.
    """
    held = set(prev_positions.keys())

    # ---- SELL ----
    to_sell = (held - new_selected) - holding_locked
    sell_orders: list[Order] = [
        Order(stock_code=code, side=Side.SELL, target_amount=0.0)
        for code in sorted(to_sell)
    ]

    # ---- BUY ----
    new_entries = new_selected - held
    K = len(new_entries)
    per_target = per_position_target(cash_avail, K, total_assets, n_target)
    buy_orders: list[Order] = [
        Order(stock_code=code, side=Side.BUY, target_amount=per_target)
        for code in sorted(new_entries)
    ]

    return sell_orders + buy_orders


# ============================================================================
# T+1 撮合（设计 §7）
# ============================================================================

def is_one_word_limit_up(open_p: float, high_p: float, low_p: float, limit_up: float) -> bool:
    """一字涨停判定：open == high == low == limit_up（容许微小浮点误差）."""
    eps = max(abs(limit_up), 1.0) * 1e-6
    return (
        abs(open_p - limit_up) < eps
        and abs(high_p - limit_up) < eps
        and abs(low_p - limit_up) < eps
    )


def is_one_word_limit_down(open_p: float, high_p: float, low_p: float, limit_down: float) -> bool:
    """一字跌停判定：open == high == low == limit_down."""
    eps = max(abs(limit_down), 1.0) * 1e-6
    return (
        abs(open_p - limit_down) < eps
        and abs(high_p - limit_down) < eps
        and abs(low_p - limit_down) < eps
    )


def capacity_cap_for_buy(avg_20d_amount: float) -> float:
    """v0.1 退化容量上限 = avg_20d_daily_amount × 0.25%（设计 §7.2）."""
    if avg_20d_amount is None or avg_20d_amount <= 0:
        return 0.0
    return float(avg_20d_amount) * CAP_BUY_RATIO


def apply_orders_at_t1(
    state: PortfolioState,
    orders: list[Order],
    t1_open: Mapping[str, float],
    t1_limit_up_flag: Mapping[str, bool],
    t1_limit_down_flag: Mapping[str, bool],
    cap_buy_amount: Mapping[str, float],
    cost_params: CostParams | None = None,
) -> FillResult:
    """T+1 开盘撮合（设计 §7）：mutates state.

    顺序：先 SELL 释放 cash → 再 BUY 用更新后的 cash.

    SELL：
    - 一字跌停 → unfilled（顺延）；position 保留
    - 否则按 shares × open 全清；扣 sell_cost；cash += net；删 position

    BUY：
    - 一字涨停 → unfilled（跳过）；cash 留存
    - 否则 fill = min(target, cap_buy)；扣 buy_cost；
      若 cash 不够覆盖 fill + cost，按可用 cash 回推 fill
    - target > cap 时把剩余 target - fill 作为 unfilled Order 输出（诊断用）

    Args:
        cap_buy_amount: stock_code → **已算好**的容量金额上限（元）.
                        由 caller 用 capacity_cap_for_buy(avg_20d) 预先算好.
    """
    cp = cost_params or CostParams()
    result = FillResult()

    # ---- SELL ----
    for o in orders:
        if o.side is not Side.SELL:
            continue
        if t1_limit_down_flag.get(o.stock_code, False):
            result.unfilled.append(o)
            continue
        pos = state.positions.get(o.stock_code)
        if pos is None or pos.shares <= 0:
            continue
        op = t1_open.get(o.stock_code)
        if op is None or op <= 0:
            result.unfilled.append(o)
            continue
        gross = pos.shares * float(op)
        cost = sell_cost(gross, cp)
        state.cash += gross - cost
        result.fills.append(Fill(
            stock_code=o.stock_code,
            side=Side.SELL,
            fill_amount=gross,
            fill_shares=pos.shares,
            fill_price=float(op),
            cost=cost,
        ))
        del state.positions[o.stock_code]

    # ---- BUY ----
    for o in orders:
        if o.side is not Side.BUY:
            continue
        if t1_limit_up_flag.get(o.stock_code, False):
            result.unfilled.append(o)
            continue
        op = t1_open.get(o.stock_code)
        if op is None or op <= 0:
            result.unfilled.append(o)
            continue
        cap = float(cap_buy_amount.get(o.stock_code, 0.0))
        fill_amt = min(o.target_amount, cap)
        if fill_amt <= 0:
            result.unfilled.append(o)
            continue
        # 现金不足覆盖 fill + 成本时，按现金回推 fill
        unit_cost_rate = cp.commission_rate + cp.slippage_rate
        max_fill_by_cash = state.cash / (1.0 + unit_cost_rate) if state.cash > 0 else 0.0
        fill_amt = min(fill_amt, max_fill_by_cash)
        if fill_amt <= 0:
            result.unfilled.append(o)
            continue
        cost = buy_cost(fill_amt, cp)
        total_cash_out = fill_amt + cost
        shares = fill_amt / float(op)
        state.cash -= total_cash_out
        state.positions[o.stock_code] = Position(
            stock_code=o.stock_code,
            entry_step=state.current_step,
            shares=shares,
            cost_basis=total_cash_out,
        )
        result.fills.append(Fill(
            stock_code=o.stock_code,
            side=Side.BUY,
            fill_amount=fill_amt,
            fill_shares=shares,
            fill_price=float(op),
            cost=cost,
        ))
        if o.target_amount > cap:
            result.unfilled.append(Order(
                stock_code=o.stock_code,
                side=Side.BUY,
                target_amount=o.target_amount - fill_amt,
            ))

    return result
