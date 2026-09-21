"""
H-A1 — 指数层趋势门控：自建全 A 等权指数（BM）收盘 > SMA(N) → 持有 BM 组合，否则现金。
设计：GinkgoRoad docs/GinkgoBrain/A股日频-趋势底座与横截面增量-设计.md §5 H-A1，验收 §1.1 风控层。
执行：T 收盘出信号，T+1 生效（用 T+1 的日收益），整仓切换按 §1.2 成本计费；持仓期间承担 BM 的月度再平衡成本。
指数层不模拟涨跌停不可成交（见设计 §7）。
Usage:
    uv run python scripts/cn_backtest_gate.py [--cache data/cache/cn] [--sma 150 200 250] [--tag h_a1]
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from strategies.cn_a_trend_base.costs import buy_cost, sell_cost                 # noqa: E402
from strategies.cn_a_trend_base.ew_index import ew_index, summarize              # noqa: E402

PERIODS = {"pre_val": ("2017-01-01", "2021-12-31"), "val": ("2022-01-01", "2025-06-30"),
           "test": ("2025-07-01", None)}
MDD_RATIO, RET_RATIO, RET_RATIO_FALLBACK = 2.0 / 3.0, 0.80, 0.60


def gate_strategy(bm: pd.DataFrame, sma_days: int) -> pd.DataFrame:
    """bm: ew_index() 输出。返回逐日 pos / ret_net / idx_net / switches。"""
    idx = bm["idx_gross"]
    sma = idx.rolling(sma_days).mean()
    signal = ((idx > sma) & sma.notna()).astype(float)          # T 收盘可知
    pos = signal.shift(1).fillna(0.0)                            # T+1 生效
    switch = pos.diff().fillna(pos)
    switch_cost = pd.Series(0.0, index=bm.index)
    switch_cost[switch > 0] = buy_cost()
    switch_cost[switch < 0] = [sell_cost(d) for d in switch.index[switch < 0]]
    ret_net = pos * bm["ret_gross"] - pos * bm["cost"] - switch_cost
    out = pd.DataFrame({"pos": pos, "ret_net": ret_net, "switch": (switch != 0).astype(int)})
    out["idx_net"] = (1.0 + out["ret_net"]).cumprod()
    return out


def evaluate(strat: dict, base: dict, mode: str, ret_ratio: float = RET_RATIO) -> dict:
    checks = {"return": strat["total_return"] >= ret_ratio * base["total_return"] if base["total_return"] > 0
              else strat["total_return"] >= base["total_return"]}
    if mode == "improve":
        checks["mdd"] = abs(strat["max_drawdown"]) <= MDD_RATIO * abs(base["max_drawdown"])
        checks["calmar"] = strat["calmar"] > base["calmar"]
    else:
        checks["mdd"] = abs(strat["max_drawdown"]) <= abs(base["max_drawdown"]) + 1e-12
    return {**checks, "pass": all(checks.values())}


def seg(df: pd.DataFrame, s: str, e: str | None) -> pd.DataFrame:
    return df.loc[s:e] if e else df.loc[s:]


def row(label: str, r: pd.Series, base: dict | None, mode: str, switches_per_year: float) -> str:
    m = summarize((1 + r).cumprod())
    acc = ""
    if base is not None:
        a = evaluate(m, base, mode)
        acc = "PASS" if a["pass"] else "FAIL(" + ",".join(k for k, v in a.items() if k != "pass" and not v) + ")"
        if not a["pass"] and evaluate(m, base, mode, RET_RATIO_FALLBACK)["pass"]:
            acc += " / PASS@60%"
        ratio = f"{abs(m['max_drawdown']) / abs(base['max_drawdown']):.2f} | {m['total_return'] / base['total_return']:.2f}" \
            if base["total_return"] > 0 else f"{abs(m['max_drawdown']) / abs(base['max_drawdown']):.2f} | —"
    else:
        ratio = "— | —"
    return (f"| {label} | {m['total_return']:+.1%} | {m['ann_return']:+.1%} | {m['max_drawdown']:.1%} | {ratio} | "
            f"{m['calmar']:.2f} | {m['sharpe']:.2f} | {switches_per_year:.1f} | {acc} |")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, default=REPO_ROOT / "data" / "cache" / "cn")
    ap.add_argument("--sma", type=int, nargs="+", default=[150, 200, 250])
    ap.add_argument("--tag", default="h_a1")
    args = ap.parse_args()

    mask = pd.read_parquet(args.cache / "universe.parquet")
    close = pd.read_parquet(args.cache / "close.parquet").loc[mask.index, mask.columns]   # cache 可能比宇宙窗口长
    universes = {"全 A": mask,
                 "沪（6xx）": mask.loc[:, [c for c in mask.columns if c.startswith("6")]],
                 "深（0xx/3xx）": mask.loc[:, [c for c in mask.columns if not c.startswith("6")]]}
    lines = [f"# H-A1 指数趋势门控 — BM 收盘 > SMA(N) 持有等权组合，否则现金", "",
             f"- Generated: `{datetime.now().isoformat(timespec='seconds')}`; SMA ∈ {args.sma}; T+1 生效；"
             f"§1.2 成本；月度再平衡 BM", 
             f"- 验收（风控层）：MDD ≤ 2/3 × BM、收益 ≥ 80% × BM（备选 60%）、Calmar > BM；test 段 no-harm", ""]
    for uname, um in universes.items():
        bm = ew_index(close.loc[:, um.columns], um, rebalance="M")
        strats = {n: gate_strategy(bm, n) for n in args.sma}
        lines += [f"## 宇宙：{uname}（{um.shape[1]} 只代码）", ""]
        for pname, (s, e) in PERIODS.items():
            mode = "noharm" if pname == "test" else "improve"
            b = seg(bm, s, e)
            base = summarize((1 + b["ret_net"]).cumprod())
            years = base["years"]
            lines += [f"### {pname} — {b.index[0].date()} → {b.index[-1].date()}（{mode}）", "",
                      "| 策略 | 总收益 | 年化 | MDD | MDD 比 \\| 收益比 | Calmar | Sharpe | 切换/年 | 验收 |",
                      "|---|---:|---:|---:|---|---:|---:|---:|:--|",
                      row("BM 等权（净）", b["ret_net"], None, mode, b["turnover"].gt(0).sum() / years)]
            for n, st in strats.items():
                st_seg = seg(st, s, e)
                lines.append(row(f"gate SMA{n}", st_seg["ret_net"], base, mode, st_seg["switch"].sum() / years))
            lines.append("")
    rep = REPO_ROOT / "reports" / f"cn_backtest_gate_{args.tag}.md"
    rep.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines)); print(f"\nwrote {rep}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
