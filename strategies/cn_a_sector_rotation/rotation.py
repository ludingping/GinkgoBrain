"""§3.3–§3.5 规则引擎：通用执行器（T 收盘决策 → T+1 开盘成交、§1.3 成本、权重漂移）+ 三个选择器。

执行器 run_executor 只认"计划"：{代码: 目标占组合价值比例}，未列出的持仓保持不动（漂移）。
主规则（rule_selector）、随机对照（random_selector）、H-B1 持有全部（hold_all_selector）、BM-S 月度等权
（panel.equal_weight_selector）都经同一执行器，成本与撮合口径完全一致（设计 §3.5、§6 单测要求）。
约定：成本从现金扣（现金可为小负数 = 欠成本），与上一条线 xs_portfolio 一致；买入总额不超过卖后现金。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from strategies.cn_a_trend_base.costs import buy_cost, sell_cost

from .features import below_sma, in_trend
from .features import strength as strength_feature

EPS = 1e-12


@dataclass(frozen=True)
class RuleParams:
    """§3 参数。k=None → H-B1 模式（持有全部在趋势行业）。"""
    k: int | None = 4
    n_in: int = 120
    L: int = 60
    every: int = 5                 # 每 every 个交易日评估一次；1 = 每日
    slope_lookback: int = 20
    exit_fast: bool = False        # §3.4：close < SMA(fast_n) 退出
    exit_trail: bool = False       # §3.4：距持仓期最高收盘回撤 ≥ trail_dd 退出
    trail_dd: float = 0.15
    band: int = 2                  # 轮动退出带宽：排名跌出前 band×K
    fast_n: int = 20


@dataclass
class EvalContext:
    """T 日收盘时选择器可见的状态（全部 ≤ T 可知）。数组按列顺序。"""
    t: int
    date: pd.Timestamp
    codes: pd.Index
    held: np.ndarray               # bool：收盘时持仓
    available: np.ndarray          # bool：当日有收盘价且在宇宙掩码内
    trend: np.ndarray              # bool：in_trend（NA → False）∧ 掩码
    strength: np.ndarray           # float
    fast_exit: np.ndarray          # bool：close < SMA(20)
    dd_from_high: np.ndarray       # float：持仓期最高收盘以来的回撤（未持仓 → NaN）
    rng: np.random.Generator


Plan = dict[str, float]
Selector = Callable[[EvalContext], Plan]
DAILY_COLS = ["ret_net", "ret_gross", "turnover", "cost", "n", "cash"]
TRADE_COLS = ["code", "entry_date", "exit_date", "exit_reason", "ret"]


@dataclass
class RotationResult:
    daily: pd.DataFrame            # DAILY_COLS，index = 日期
    weights: pd.DataFrame          # 收盘持仓权重（日期 × 代码）
    trades: pd.DataFrame           # TRADE_COLS；未平仓 exit_date = NaT、exit_reason = "open"


def _safe_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        r = a / b - 1.0
    return np.where(np.isfinite(r), r, 0.0)


def _execute(plan_idx: dict[int, float], v_open: np.ndarray, cash: float, open_row: np.ndarray, date: pd.Timestamp
             ) -> tuple[np.ndarray, float, float, float]:
    """T+1 开盘：先卖后买；开盘缺价的代码跳过；买入总额不超过卖后现金（按比例缩）。返回 (v, cash, cost, 成交额)。"""
    port_open = cash + v_open.sum()
    tradable = np.isfinite(open_row) & (open_row > 0)
    v = v_open.copy()
    sell_amt = 0.0; buys: dict[int, float] = {}
    for i, frac in plan_idx.items():
        if not tradable[i]:
            continue
        target = frac * port_open
        delta = target - v[i]
        if delta < -EPS:
            sell_amt += -delta; v[i] = target if target > EPS else 0.0
        elif delta > EPS:
            buys[i] = delta
    cash += sell_amt
    total_buy = sum(buys.values())
    scale = 1.0 if total_buy <= EPS else (min(1.0, cash / total_buy) if cash > 0 else 0.0)
    buy_amt = sum(a * scale for a in buys.values())
    for i, amt in buys.items():
        v[i] += amt * scale
    cost = buy_amt * buy_cost() + sell_amt * sell_cost(date)
    return v, cash - buy_amt - cost, cost, sell_amt + buy_amt


def _eval_days(idx: pd.DatetimeIndex, every: int | pd.Series, start) -> np.ndarray:
    if isinstance(every, pd.Series):
        return every.reindex(idx).fillna(False).to_numpy(bool)
    start_i = int(idx.searchsorted(pd.Timestamp(start))) if start is not None else 0
    t = np.arange(len(idx))
    return (t >= start_i) & ((t - start_i) % every == 0)


def run_executor(close: pd.DataFrame, open_px: pd.DataFrame, planner: Selector, *, every: int | pd.Series = 5,
                 start=None, seed: int | None = None, mask: pd.DataFrame | None = None,
                 trend: pd.DataFrame | None = None, strength_df: pd.DataFrame | None = None,
                 fast_exit: pd.DataFrame | None = None,
                 reasons: Callable[[EvalContext, Plan], dict[str, str]] | None = None) -> RotationResult:
    """
    逐日：T 日收盘估值 → 评估日调用 planner 得计划 → T+1 开盘执行。组合起点 1.0。
    every：整数 = 每 every 个交易日（自 start 起）；bool Series = 指定评估日。
    trend / strength_df / fast_exit：选择器可见的特征（可空布尔 / 浮点，NA → False / NaN）。
    reasons(ctx, plan)：计划里清零代码的退出原因，写入 trades.exit_reason。
    """
    idx, cols = close.index, close.columns
    C, O = close.to_numpy(float), open_px.to_numpy(float)
    Cf = close.ffill().to_numpy(float)               # 估值用最后有效收盘：缺行日收益 0，下一有效日补齐整段涨跌（不填零）
    n_days, n_cols = C.shape
    M = mask.to_numpy(bool) if mask is not None else np.ones(C.shape, bool)
    TR = trend.fillna(False).to_numpy(bool) if trend is not None else np.zeros(C.shape, bool)
    S = strength_df.to_numpy(float) if strength_df is not None else np.full(C.shape, np.nan)
    FX = fast_exit.fillna(False).to_numpy(bool) if fast_exit is not None else np.zeros(C.shape, bool)
    eval_day = _eval_days(idx, every, start)
    rng = np.random.default_rng(seed)
    v = np.zeros(n_cols); cash = 1.0
    pos_high = np.full(n_cols, np.nan); entry_i = np.full(n_cols, -1)
    pending: dict[int, float] | None = None; pending_reasons: dict[int, str] = {}
    out = np.zeros((n_days, len(DAILY_COLS))); W = np.zeros((n_days, n_cols)); trades: list[dict] = []
    for t in range(n_days):
        port_prev = cash + v.sum()
        cost = 0.0; traded = 0.0
        if t > 0 and pending:
            v_open = v * (1.0 + _safe_ratio(O[t], Cf[t - 1]))
            was_held = v_open > EPS
            v, cash, cost, traded = _execute(pending, v_open, cash, O[t], idx[t])
            now_held = v > EPS
            for i in np.flatnonzero(was_held & ~now_held):
                trades.append({"code": cols[i], "entry_date": idx[entry_i[i]], "exit_date": idx[t],
                               "exit_reason": pending_reasons.get(i, "plan"), "ret": float(O[t, i] / O[entry_i[i], i] - 1.0)})
                entry_i[i] = -1
            for i in np.flatnonzero(~was_held & now_held):
                entry_i[i] = t; pos_high[i] = np.nan
            v = v * (1.0 + _safe_ratio(Cf[t], O[t]))
            pending = None
        elif t > 0:
            v = v * (1.0 + _safe_ratio(Cf[t], Cf[t - 1]))
        port_net = cash + v.sum()
        held = v > EPS
        pos_high = np.where(held, np.fmax(pos_high, C[t]), np.nan)
        ret_gross = (port_net + cost) / port_prev - 1.0
        out[t] = (ret_gross - cost / port_prev, ret_gross, traded / port_prev, cost / port_prev, held.sum(), cash / port_net)
        W[t] = v / port_net
        if eval_day[t]:
            with np.errstate(invalid="ignore"):
                dd = np.where(held, C[t] / pos_high - 1.0, np.nan)
            ctx = EvalContext(t, idx[t], cols, held, np.isfinite(C[t]) & M[t], TR[t] & M[t], S[t], FX[t], dd, rng)
            plan = planner(ctx)
            if plan:
                pending = {int(cols.get_loc(c)): f for c, f in plan.items()}
                pending_reasons = {int(cols.get_loc(c)): r for c, r in reasons(ctx, plan).items()} if reasons else {}
    for i in np.flatnonzero(entry_i >= 0):                        # 未平仓也记一笔
        trades.append({"code": cols[i], "entry_date": idx[entry_i[i]], "exit_date": pd.NaT, "exit_reason": "open", "ret": np.nan})
    daily = pd.DataFrame(out, index=idx, columns=DAILY_COLS)
    return RotationResult(daily, pd.DataFrame(W, index=idx, columns=cols), pd.DataFrame(trades, columns=TRADE_COLS))


# ─── 选择器（§3.3 / §3.5 / H-B1）。每个选择器把本次退出原因存到 .last_reasons ──────────

def _rank_desc(s: np.ndarray, cand: np.ndarray) -> np.ndarray:
    """候选内按 strength 降序名次（1 = 最强）；非候选 = +inf。并列按列顺序（稳定排序）。"""
    rank = np.full(len(s), np.inf)
    ci = np.flatnonzero(cand)
    order = ci[np.argsort(-s[ci], kind="stable")]
    rank[order] = np.arange(1, len(order) + 1)
    return rank


def _exit_flags(ctx: EvalContext, p: RuleParams, rank: np.ndarray) -> dict[int, str]:
    """持仓退出触发（§3.3 两条 + §3.4 变体），{列索引: 原因}；记录优先级 trend > rank > fast > trail。"""
    out: dict[int, str] = {}
    for i in np.flatnonzero(ctx.held):
        if not ctx.trend[i]:
            out[i] = "trend"
        elif p.k is not None and rank[i] > p.band * p.k:
            out[i] = "rank"
        elif p.exit_fast and ctx.fast_exit[i]:
            out[i] = "fast"
        elif p.exit_trail and np.isfinite(ctx.dd_from_high[i]) and ctx.dd_from_high[i] <= -p.trail_dd + EPS:
            out[i] = "trail"
    return out


def rule_selector(p: RuleParams) -> Selector:
    """§3.3 主规则：K 槽位；退出 = 出趋势 / 排名跌出前 band·K（/ fast / trail）；空槽按 strength 从高到低补入。"""
    if p.k is None:
        return hold_all_selector(p)

    def decide(ctx: EvalContext) -> Plan:
        cand = ctx.trend & np.isfinite(ctx.strength) & ctx.available
        exits = _exit_flags(ctx, p, _rank_desc(ctx.strength, cand))
        n_free = max(p.k - (int(ctx.held.sum()) - len(exits)), 0)
        pool = np.flatnonzero(cand & ~ctx.held)
        entries = pool[np.argsort(-ctx.strength[pool], kind="stable")][:n_free]
        decide.last_reasons = {ctx.codes[i]: r for i, r in exits.items()}
        return {**{ctx.codes[i]: 0.0 for i in exits}, **{ctx.codes[i]: 1.0 / p.k for i in entries}}
    decide.last_reasons = {}
    return decide


def hold_all_selector(p: RuleParams) -> Selector:
    """H-B1：持有全部 in_trend 行业等权，每次评估再平衡；无行业在趋势 → 全现金。"""
    def decide(ctx: EvalContext) -> Plan:
        keep = ctx.trend & ctx.available
        drop = np.flatnonzero(ctx.held & ~keep)
        n = int(keep.sum())
        decide.last_reasons = {ctx.codes[i]: "trend" for i in drop}
        return {**{ctx.codes[i]: 1.0 / n for i in np.flatnonzero(keep)}, **{ctx.codes[i]: 0.0 for i in drop}}
    decide.last_reasons = {}
    return decide


def random_selector(p: RuleParams) -> Selector:
    """§3.5 随机对照：每次评估在全部可用行业里随机抽 K 个（不看趋势），同 K、同频率、同成本。"""
    def decide(ctx: EvalContext) -> Plan:
        avail = np.flatnonzero(ctx.available)
        k = min(p.k or 0, len(avail))
        draw = set(ctx.rng.choice(avail, size=k, replace=False).tolist()) if k else set()
        exits = [i for i in np.flatnonzero(ctx.held) if i not in draw]
        decide.last_reasons = {ctx.codes[i]: "random" for i in exits}
        return {**{ctx.codes[i]: 0.0 for i in exits}, **{ctx.codes[i]: 1.0 / p.k for i in draw if not ctx.held[i]}}
    decide.last_reasons = {}
    return decide


def run_rotation(close: pd.DataFrame, open_px: pd.DataFrame, p: RuleParams, *, selector: Selector | None = None,
                 mask: pd.DataFrame | None = None, trend: pd.DataFrame | None = None,
                 strength: pd.DataFrame | None = None, start=None, seed: int | None = None) -> RotationResult:
    """按 §3 参数算特征（或用传入的 trend / strength 覆盖，便于状态机单测），经 run_executor 回测。"""
    tr = trend if trend is not None else in_trend(close, p.n_in, p.slope_lookback)
    st = strength if strength is not None else strength_feature(close, p.L)
    fx = below_sma(close, p.fast_n) if p.exit_fast else None
    sel = selector or rule_selector(p)
    reasons = (lambda ctx, plan: getattr(sel, "last_reasons", {})) if hasattr(sel, "last_reasons") else None
    return run_executor(close, open_px, sel, every=p.every, start=start, seed=seed, mask=mask, trend=tr,
                        strength_df=st, fast_exit=fx, reasons=reasons)


__all__ = ["RuleParams", "EvalContext", "RotationResult", "DAILY_COLS", "TRADE_COLS", "run_executor", "rule_selector",
           "hold_all_selector", "random_selector", "run_rotation"]
