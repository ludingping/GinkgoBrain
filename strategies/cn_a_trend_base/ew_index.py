"""自建"全 A 等权指数"（BM）：T−1 收盘宇宙等权，持有 T 日收益，日度再平衡；毛收益与含成本净收益。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .costs import turnover_cost


def rebalance_days(index: pd.DatetimeIndex, freq: str | None) -> pd.Series:
    """再平衡日（bool）。freq=None → 每日；"M" → 每月首个交易日；"W" → 每周首个交易日。"""
    if freq is None:
        return pd.Series(True, index=index)
    key = index.to_period(freq)
    return pd.Series(key != np.roll(key, 1), index=index).fillna(True) | pd.Series(index == index[0], index=index)


def ew_index(close: pd.DataFrame, mask: pd.DataFrame, rebalance: str | None = "M") -> pd.DataFrame:
    """
    Args:
        close: 日期 × 代码 的前复权收盘价（停牌日可为上一收盘或 NaN）。
        mask:  日期 × 代码 的宇宙 bool（T 日收盘后可知）。
        rebalance: None=每日等权（换手 ~40x/年，只作诊断）；"M"=每月首个交易日按 T−1 宇宙等权，
                   期间权重随价格漂移、不因成员退出而强制卖出（≈ 等权指数的做法）。
    Returns:
        DataFrame（index = 日期）: n、ret_gross、turnover（Σ|Δw|，仅再平衡日非零）、cost、ret_net、
        idx_gross、idx_net（起点 1.0）。第一天无持仓，收益 0。
    """
    if not close.index.equals(mask.index) or not close.columns.equals(mask.columns):
        raise ValueError("ew_index: close and mask must share index and columns "
                         f"(close {close.shape}, mask {mask.shape})")
    px = close.where(close > 0)                              # 0 / 负价是坏数据，不是收益
    ret = (px / px.shift(1) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)   # 缺价/停牌 → 0
    held = mask.shift(1).fillna(False).astype(bool)          # T 日可用的最新宇宙 = T−1 收盘
    rb = rebalance_days(close.index, rebalance)
    w = np.zeros(close.shape[1])
    W = np.empty(close.shape, dtype="float64")
    R = ret.to_numpy(); H = held.to_numpy()
    turnover = np.zeros(len(close)); ret_gross = np.zeros(len(close))
    for t in range(len(close)):
        if rb.iloc[t] or (w.sum() == 0.0 and H[t].any()):   # 再平衡日，或空仓且宇宙非空 → 首次建仓
            n_t = H[t].sum()
            target = H[t] / n_t if n_t else np.zeros_like(w)
            turnover[t] = np.abs(target - w).sum()
            w = target
        W[t] = w
        r = float((w * R[t]).sum()); ret_gross[t] = r
        w = w * (1.0 + R[t]) / (1.0 + r) if (1.0 + r) != 0 else w
    n = pd.Series((W > 0).sum(axis=1), index=close.index)
    turnover = pd.Series(turnover, index=close.index)
    cost = turnover_cost(turnover)
    ret_gross = pd.Series(ret_gross, index=close.index)
    ret_net = ret_gross - cost
    out = pd.DataFrame({"n": n, "ret_gross": ret_gross, "turnover": turnover, "cost": cost, "ret_net": ret_net})
    out["idx_gross"] = (1.0 + out["ret_gross"]).cumprod()
    out["idx_net"] = (1.0 + out["ret_net"]).cumprod()
    return out


def summarize(idx: pd.Series, periods_per_year: int = 244) -> dict:
    r = idx.pct_change().dropna()
    years = len(r) / periods_per_year
    total = idx.iloc[-1] / idx.iloc[0] - 1.0
    ann = (1.0 + total) ** (1.0 / years) - 1.0 if years > 0 and total > -1 else float("nan")
    mdd = float((idx / idx.cummax() - 1.0).min())
    return {"total_return": float(total), "ann_return": float(ann), "max_drawdown": mdd,
            "calmar": float(ann / abs(mdd)) if mdd < 0 else float("nan"),
            "sharpe": float(r.mean() / r.std() * np.sqrt(periods_per_year)) if r.std() > 0 else float("nan"),
            "years": years}


def blend_sleeves(rets: pd.DataFrame, weights: dict[str, float], reset: str | None = "M",
                  reset_cost: float = 0.0015) -> pd.DataFrame:
    """两条（或多条）腿的净日收益按固定比例混合，`reset` 频率把腿间比例复位到目标，期间随收益漂移。
    腿内换仓成本已含在各腿 ret 里；复位成本 = Σ|Δw| × reset_cost（平均单边成本近似）。
    Returns: DataFrame(ret_net, turnover, cost, idx_net)。"""
    cols = [c for c in weights if c in rets.columns]
    R = rets[cols].fillna(0.0).to_numpy(float)
    tgt = np.array([weights[c] for c in cols], float); tgt = tgt / tgt.sum()
    rb = rebalance_days(rets.index, reset)
    w = tgt.copy(); out = np.zeros((len(rets), 3))
    for t in range(len(rets)):
        turn = 0.0
        if rb.iloc[t] and t > 0:
            turn = np.abs(tgt - w).sum(); w = tgt.copy()
        cost = turn * reset_cost
        r = float((w * R[t]).sum())
        out[t] = (r - cost, turn, cost)
        w = w * (1.0 + R[t]) / (1.0 + r) if (1.0 + r) != 0 else w
    df = pd.DataFrame(out, index=rets.index, columns=["ret_net", "turnover", "cost"])
    df["idx_net"] = (1.0 + df["ret_net"]).cumprod()
    return df
