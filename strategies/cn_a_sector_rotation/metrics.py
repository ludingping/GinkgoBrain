"""§1.2 指标与验收：分段绩效、对 BM-S 的比值、双栏验收 + test no-harm、随机对照汇总（§3.5）。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.cn_a_trend_base.ew_index import summarize

TURNOVER_MAX = 12.0                       # 年单边换手上限（两栏相同）
RISK_LAYER = {"mdd_mult": 2.0 / 3.0, "ret_mult": 0.8}        # H-B1 风控层
RETURN_LAYER = {"mdd_mult": 1.0, "ret_mult": 1.1}            # H-B2 收益层
NOHARM = {"mdd_mult": 1.0, "ret_mult": 0.8}                  # test 段
METRIC_KEYS = ("total_return", "ann_return", "max_drawdown", "calmar", "sharpe", "turnover_yr", "n_mean", "cash_mean")


def segment(x: pd.DataFrame | pd.Series, start: str | None, end: str | None):
    """闭区间切片；end=None → 到底。"""
    return x.loc[start:end] if end else x.loc[start:]


def perf(daily: pd.DataFrame, start: str | None = None, end: str | None = None, periods_per_year: int = 244) -> dict:
    """由逐日 DataFrame(ret_net, turnover, n, cash) 得分段指标；turnover_yr = 年单边换手（Σ turnover / 2 / 年数）。"""
    d = segment(daily, start, end)
    idx = (1.0 + d["ret_net"]).cumprod()
    idx = pd.concat([pd.Series([1.0], index=[d.index[0] - pd.Timedelta(days=1)]), idx])   # 起点 1.0，首日收益计入
    m = summarize(idx, periods_per_year)
    years = m["years"]
    return {**m, "turnover_yr": float(d["turnover"].sum() / 2.0 / years) if years > 0 else float("nan"),
            "n_mean": float(d["n"].mean()), "cash_mean": float(d["cash"].mean()),
            "start": d.index[0], "end": d.index[-1]}


def ratios(m: dict, base: dict) -> dict:
    """MDD 比与收益比（BM 收益 ≤ 0 时收益比为 NaN）。"""
    mdd = abs(m["max_drawdown"]) / abs(base["max_drawdown"]) if base["max_drawdown"] < 0 else float("nan")
    ret = m["total_return"] / base["total_return"] if base["total_return"] > 0 else float("nan")
    return {"mdd_ratio": float(mdd), "ret_ratio": float(ret)}


def _return_ok(m: dict, base: dict, mult: float) -> bool:
    if base["total_return"] > 0:
        return m["total_return"] >= mult * base["total_return"]
    return m["total_return"] > base["total_return"]        # BM 为负：只要求高于 BM


def acceptance(m: dict, base: dict, layer: str) -> str:
    """§1.2：layer ∈ {risk, return, noharm}。返回 "PASS" 或 "FAIL(a,b)"。"""
    spec = {"risk": RISK_LAYER, "return": RETURN_LAYER, "noharm": NOHARM}[layer]
    checks = {"mdd": abs(m["max_drawdown"]) <= spec["mdd_mult"] * abs(base["max_drawdown"]) + 1e-12,
              "return": _return_ok(m, base, spec["ret_mult"])}
    if layer != "noharm":
        checks["calmar"] = m["calmar"] > base["calmar"]
    checks["turnover"] = m["turnover_yr"] <= TURNOVER_MAX
    return "PASS" if all(checks.values()) else "FAIL(" + ",".join(k for k, v in checks.items() if not v) + ")"


def random_summary(ms: list[dict], keys: tuple[str, ...] = METRIC_KEYS) -> dict[str, dict[str, float]]:
    """§3.5：多次随机抽样的每个指标 → {median, p5, p95}。"""
    out = {}
    for k in keys:
        v = np.array([x[k] for x in ms if k in x], dtype=float)
        v = v[np.isfinite(v)]
        out[k] = ({"median": float(np.median(v)), "p5": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95))}
                  if len(v) else {"median": float("nan"), "p5": float("nan"), "p95": float("nan")})
    return out


def beats_random(m: dict, rs: dict[str, dict[str, float]]) -> bool:
    """策略 MDD 优于（浅于）随机对照中位数。"""
    return abs(m["max_drawdown"]) < abs(rs["max_drawdown"]["median"])


__all__ = ["TURNOVER_MAX", "RISK_LAYER", "RETURN_LAYER", "NOHARM", "METRIC_KEYS", "segment", "perf", "ratios",
           "acceptance", "random_summary", "beats_random"]
