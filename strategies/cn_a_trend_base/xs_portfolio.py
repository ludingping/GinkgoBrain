"""L2 横截面多头组合回测（日频估值、T+1 开盘成交、涨停不买/跌停不卖、§1.2 成本）。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .costs import buy_cost, sell_cost


def select_bottom_decile(feature_row: pd.Series, members: pd.Series, q: int = 10, bottom: bool = True) -> list[str]:
    """T 日截面：在宇宙成员里按特征分 q 组，取最低（bottom=True）或最高一组的代码。"""
    x = feature_row[members & feature_row.notna()]
    if len(x) < q * 5:
        return []
    ranks = x.rank(method="first")
    n = len(x)
    cut = np.ceil(n / q)
    chosen = ranks <= cut if bottom else ranks > n - cut
    return list(x.index[chosen])


def long_portfolio_backtest(close: pd.DataFrame, open_px: pd.DataFrame, feature: pd.DataFrame, mask: pd.DataFrame,
                            *, every: int = 20, q: int = 10, bottom: bool = True,
                            limit_up: pd.DataFrame | None = None, limit_down: pd.DataFrame | None = None,
                            start: str | None = None) -> pd.DataFrame:
    """
    每 `every` 个交易日的收盘（T）选组；T+1 开盘按等权换仓：不在新组且可卖的卖出（跌停不可卖 → 继续持有），
    新进且可买的买入（涨停不可买 → 跳过，现金保留）；持有期内权重随价格漂移。
    日收益按收盘估值：换仓日 = 开盘前漂移 × 开盘→收盘；成本 = 买入额×buy_cost + 卖出额×sell_cost(日期)。
    Returns: DataFrame(ret_net, ret_gross, turnover, cost, n, cash)，index = 日期。
    """
    idx = close.index
    cols = close.columns
    C, O = close.to_numpy(float), open_px.to_numpy(float)
    M = mask.to_numpy(bool)
    LU = limit_up.to_numpy(bool) if limit_up is not None else np.zeros(C.shape, bool)
    LD = limit_down.to_numpy(bool) if limit_down is not None else np.zeros(C.shape, bool)
    F = feature
    n_days = len(idx)
    w = np.zeros(len(cols))                 # 收盘时持仓权重（占组合价值）
    cash = 1.0
    pending: list[int] | None = None        # T 日选出的目标（列索引），T+1 执行
    out = np.zeros((n_days, 6))
    start_i = idx.searchsorted(pd.Timestamp(start)) if start else 0
    for t in range(n_days):
        ret_g = 0.0; cost = 0.0; turn = 0.0
        if t > 0:
            r_cc = C[t] / C[t - 1] - 1.0
            r_cc = np.where(np.isfinite(r_cc), r_cc, 0.0)
            r_oc = C[t] / O[t] - 1.0
            r_co = O[t] / C[t - 1] - 1.0
            r_oc = np.where(np.isfinite(r_oc), r_oc, 0.0); r_co = np.where(np.isfinite(r_co), r_co, 0.0)
            if pending is not None:
                # 开盘前：持仓漂移到开盘
                v_open = w * (1.0 + r_co)
                port_open = cash + v_open.sum()
                target_set = np.zeros(len(cols), bool); target_set[pending] = True
                keep = (v_open > 0) & (target_set | LD[t])                  # 跌停不可卖 → 被迫保留
                sell = (v_open > 0) & ~keep
                buy_new = target_set & ~(v_open > 0) & ~LU[t] & np.isfinite(O[t]) & (O[t] > 0)
                sell_amt = v_open[sell].sum()
                cash_after = cash + sell_amt
                cost += sell_amt * sell_cost(idx[t])
                # 目标：每个"槽位"（目标组成员 + 被迫保留者）等权；涨停买不到的槽位留现金
                n_slots = int((target_set | keep).sum())
                n_tgt = int(keep.sum() + buy_new.sum())
                if n_tgt > 0:
                    tgt_w = np.zeros(len(cols)); tgt_w[keep | buy_new] = port_open / n_slots
                    delta = tgt_w - np.where(keep, v_open, 0.0)
                    buys = delta[delta > 0].sum(); sells_adj = -delta[delta < 0].sum()
                    cost += buys * buy_cost() + sells_adj * sell_cost(idx[t])
                    turn = (sell_amt + buys + sells_adj) / port_open
                    v_open_new = tgt_w
                    cash_after = port_open - tgt_w.sum()
                else:
                    v_open_new = np.zeros(len(cols)); turn = sell_amt / port_open
                # 开盘 → 收盘
                v_close = v_open_new * (1.0 + r_oc)
                port_close = cash_after + v_close.sum()
                ret_g = port_close / (cash + w.sum()) - 1.0
                port_close_net = port_close - cost * (cash + w.sum())
                w = v_close / port_close_net * 1.0; cash = cash_after / port_close_net
                pending = None
            else:
                v_close = w * (1.0 + r_cc)
                port_close = cash + v_close.sum()
                ret_g = port_close - 1.0
                w = v_close / port_close; cash = cash / port_close
        ret_n = ret_g - cost
        # 收盘选组（T 日）
        if t >= start_i and (t - start_i) % every == 0:
            row = F.iloc[t]; members = pd.Series(M[t], index=cols)
            sel = select_bottom_decile(row, members, q=q, bottom=bottom)
            pending = [cols.get_loc(c) for c in sel]
        out[t] = (ret_n, ret_g, turn, cost, (w > 0).sum(), cash)
    return pd.DataFrame(out, index=idx, columns=["ret_net", "ret_gross", "turnover", "cost", "n", "cash"])
