"""L1 横截面探针：逐日 rank IC、折内统计（重叠期 Newey–West t）、十分位价差。纯函数，输入宽表。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ─── 特征（T 日收盘后可知；只用 ≤ T 的数据） ──────────────────────────────

def momentum(close: pd.DataFrame, months: int, skip_months: int = 1, days_per_month: int = 21) -> pd.DataFrame:
    """close_{t−skip} / close_{t−months} − 1（跳过最近 skip 个月）。"""
    a, b = skip_months * days_per_month, months * days_per_month
    return close.shift(a) / close.shift(b) - 1.0


def reversal(close: pd.DataFrame, days: int) -> pd.DataFrame:
    """最近 days 日收益（登记方向：IC < 0）。"""
    return close / close.shift(days) - 1.0


def realized_vol(close: pd.DataFrame, days: int) -> pd.DataFrame:
    """最近 days 日对数收益标准差（登记方向：IC < 0）。"""
    return np.log(close).diff().rolling(days, min_periods=max(5, days // 2)).std()


def forward_open_return(open_px: pd.DataFrame, k: int) -> pd.DataFrame:
    """T 日信号 → T+1 开盘买入、T+1+k 开盘卖出的收益（对齐到 T 行）。"""
    return open_px.shift(-(1 + k)) / open_px.shift(-1) - 1.0


# ─── 逐日横截面 rank IC ─────────────────────────────────────────────────

def _rank_rows(x: np.ndarray) -> np.ndarray:
    """每行对非 NaN 值做平均秩，NaN 保持 NaN。"""
    out = np.full(x.shape, np.nan)
    for i in range(x.shape[0]):
        v = x[i]; m = np.isfinite(v)
        if m.sum() >= 2:
            out[i, m] = pd.Series(v[m]).rank().to_numpy()
    return out


def daily_rank_ic(feature: pd.DataFrame, fwd: pd.DataFrame, mask: pd.DataFrame, min_names: int = 200) -> pd.Series:
    """每日在 mask 成员上算 Spearman(feature, fwd)；成员不足 min_names 的日子为 NaN。"""
    f = feature.where(mask).to_numpy(dtype=float)
    r = fwd.where(mask).to_numpy(dtype=float)
    valid = np.isfinite(f) & np.isfinite(r)
    f = np.where(valid, f, np.nan); r = np.where(valid, r, np.nan)
    rf, rr = _rank_rows(f), _rank_rows(r)
    n = valid.sum(axis=1)
    ic = np.full(len(feature), np.nan)
    for i in range(len(feature)):
        if n[i] >= min_names:
            a, b = rf[i][valid[i]], rr[i][valid[i]]
            ic[i] = np.corrcoef(a, b)[0, 1]
    return pd.Series(ic, index=feature.index, name="ic")


def nw_tstat(x: pd.Series, lags: int) -> float:
    """均值的 t 统计，Newey–West（Bartlett）修正重叠期自相关。"""
    v = x.dropna().to_numpy(dtype=float)
    n = len(v)
    if n < 10:
        return float("nan")
    mu = v.mean(); e = v - mu
    s2 = (e @ e) / n
    for j in range(1, min(lags, n - 1) + 1):
        w = 1.0 - j / (lags + 1)
        s2 += 2.0 * w * (e[:-j] @ e[j:]) / n
    if s2 <= 0:                                   # 零方差（如恒等 IC）：均值非零即无穷大 t
        return float(np.sign(mu) * np.inf) if mu != 0 else float("nan")
    return float(mu / np.sqrt(s2 / n))


@dataclass(frozen=True)
class FoldStat:
    name: str
    start: str
    end: str
    n_days: int
    ic_mean: float
    ic_t: float
    mean_names: float


def fold_stats(ic: pd.Series, names: pd.Series, folds: dict[str, tuple[str, str | None]], lags: int) -> list[FoldStat]:
    out = []
    for name, (s, e) in folds.items():
        seg = ic.loc[s:e] if e else ic.loc[s:]
        nm = names.loc[s:e] if e else names.loc[s:]
        d = seg.dropna()
        out.append(FoldStat(name, str(seg.index[0].date()) if len(seg) else s, str(seg.index[-1].date()) if len(seg) else str(e),
                            int(len(d)), float(d.mean()) if len(d) else float("nan"), nw_tstat(d, lags),
                            float(nm.mean()) if len(nm) else float("nan")))
    return out


def verdict(stats: list[FoldStat], expected_sign: int, ic_min: float = 0.02, t_min: float = 2.0,
            n_min_days: int = 120) -> tuple[bool, str]:
    """三折：|IC| ≥ ic_min、|t| ≥ t_min、与登记方向同号、天数 ≥ n_min_days；最近折不反号（包含在同号里）。"""
    for f in stats:
        if f.n_days < n_min_days:
            return False, f"n<{n_min_days} on {f.name}"
        if not np.isfinite(f.ic_mean) or np.sign(f.ic_mean) != expected_sign:
            return False, f"sign on {f.name}"
        if abs(f.ic_mean) < ic_min:
            return False, f"|IC|<{ic_min} on {f.name}"
        if np.isnan(f.ic_t) or abs(f.ic_t) < t_min:      # ±inf（零方差）视为通过
            return False, f"|t|<{t_min} on {f.name}"
    return True, "ok"


def decile_returns(feature: pd.DataFrame, fwd: pd.DataFrame, mask: pd.DataFrame, every: int = 20,
                   q: int = 10) -> pd.DataFrame:
    """每 `every` 个交易日取一次截面，按特征分 q 组，返回 日期 × 组 的平均前向收益。"""
    rows = {}
    idx = feature.index[::every]
    f = feature.where(mask); r = fwd.where(mask)
    for d in idx:
        x, y = f.loc[d], r.loc[d]
        m = x.notna() & y.notna()
        if m.sum() < 10 * q:
            continue
        g = pd.qcut(x[m].rank(method="first"), q, labels=False)
        rows[d] = y[m].groupby(g).mean().reindex(range(q)).to_numpy()
    return pd.DataFrame.from_dict(rows, orient="index", columns=[f"D{i + 1}" for i in range(q)]).sort_index()
