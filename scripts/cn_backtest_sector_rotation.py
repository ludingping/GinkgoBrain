"""
E-B3 — H-B1（行业绝对趋势过滤，风控层）与 H-B2（相对强度轮动，收益层）同批 L2 回测。
设计：GinkgoRoad docs/GinkgoBrain/A股日频-板块趋势轮动-设计.md §1.2 / §3 / §4 L2 / §5 / §6 E-B3。
宇宙 × 三段：BM-S 行、H-B1（N_in 60/120/200）、H-B2 主变体（K=4, N=120, L=60, 周频）+ 邻域（K 3/5、N 60/200、
L 40/120、每日评估、exit_fast、exit_trail）+ 随机对照（同 K、同频率、同成本；100 次中位数与 5%–95%）。
一次跑完整网格；--universes / --segments / --variants / --n-random 可子集。报告格式沿用 reports/cn_backtest_xs_h_a3.md。
Usage:
    uv run python scripts/cn_backtest_sector_rotation.py [--cache data/cache/cn_sector] [--universes L1 L2 L1-upper L1-lower]
                                                         [--variants 主 K3 ...] [--n-random 100] [--seed 0] [--tag x]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from strategies.cn_a_sector_rotation.features import in_trend, strength                        # noqa: E402
from strategies.cn_a_sector_rotation.ic_probe import amount_split_masks                         # noqa: E402
from strategies.cn_a_sector_rotation.metrics import acceptance, beats_random, perf, random_summary, ratios   # noqa: E402
from strategies.cn_a_sector_rotation.panel import BACKTEST_START, SEGMENTS, bm_s, load_panel   # noqa: E402
from strategies.cn_a_sector_rotation.rotation import RotationResult, RuleParams, random_selector, run_rotation   # noqa: E402

MAIN = RuleParams(k=4, n_in=120, L=60, every=5)
MAIN_NAME = "H-B2 K4 N120 L60 周频*"
# 名称 → (参数, 验收层)；* = 主变体。H-B1 走风控层，H-B2 走收益层（test 段一律 no-harm）
VARIANTS: dict[str, tuple[RuleParams, str]] = {
    "H-B1 N60": (RuleParams(k=None, n_in=60), "risk"),
    "H-B1 N120*": (RuleParams(k=None, n_in=120), "risk"),
    "H-B1 N200": (RuleParams(k=None, n_in=200), "risk"),
    MAIN_NAME: (MAIN, "return"),
    "H-B2 K3": (RuleParams(k=3, n_in=120, L=60, every=5), "return"),
    "H-B2 K5": (RuleParams(k=5, n_in=120, L=60, every=5), "return"),
    "H-B2 N60": (RuleParams(k=4, n_in=60, L=60, every=5), "return"),
    "H-B2 N200": (RuleParams(k=4, n_in=200, L=60, every=5), "return"),
    "H-B2 L40": (RuleParams(k=4, n_in=120, L=40, every=5), "return"),
    "H-B2 L120": (RuleParams(k=4, n_in=120, L=120, every=5), "return"),
    "H-B2 每日评估": (RuleParams(k=4, n_in=120, L=60, every=1), "return"),
    "H-B2 exit_fast": (RuleParams(k=4, n_in=120, L=60, every=5, exit_fast=True), "return"),
    "H-B2 exit_trail": (RuleParams(k=4, n_in=120, L=60, every=5, exit_trail=True), "return"),
}
HEADER = ["| 策略 | 总收益 | 年化 | MDD | MDD 比 \\| 收益比 | Calmar | Sharpe | 换手/年 | 持仓数 | 现金 | 验收 |",
          "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|:--|"]


def build_universes(close: pd.DataFrame, amount: pd.DataFrame, l1: list[str], l2: list[str]) -> dict[str, tuple[list[str], pd.DataFrame | None]]:
    """名称 → (列, 时变掩码或 None)。上 / 下半区按 20 日成交额中位数逐日切分，作用于 in_trend（持仓掉出半区 → 视为出趋势）。"""
    up, lo = amount_split_masks(amount.loc[:, l1], window=20)
    out = {"L1": (l1, None), "L2": (l2, None), "L1-upper": (l1, up), "L1-lower": (l1, lo)}
    return {k: v for k, v in out.items() if v[0]}


def run_variant(close, open_px, p: RuleParams, mask, start, *, selector=None, seed=None) -> RotationResult:
    """特征用全历史预热，回测自 start 起。"""
    tr = in_trend(close, p.n_in, p.slope_lookback); st = strength(close, p.L)
    sl = slice(start, None)
    return run_rotation(close.loc[sl], open_px.loc[sl], p, selector=selector, mask=mask.loc[sl] if mask is not None else None,
                        trend=tr.loc[sl], strength=st.loc[sl], seed=seed)


def row(name: str, m: dict, base: dict | None, verdict: str) -> str:
    if base is None:
        rr = "—"
    else:
        rt = ratios(m, base)
        rr = f"{rt['mdd_ratio']:.2f} | " + (f"{rt['ret_ratio']:.2f}" if pd.notna(rt["ret_ratio"]) else "—")
    return (f"| {name} | {m['total_return']:+.1%} | {m['ann_return']:+.1%} | {m['max_drawdown']:.1%} | {rr} | {m['calmar']:.2f} | "
            f"{m['sharpe']:.2f} | {m['turnover_yr']:.1f} | {m['n_mean']:.1f} | {m['cash_mean']:.0%} | {verdict} |")


def random_row(rs: dict, base: dict) -> str:
    def f(k: str, fmt: str) -> str:
        return f"{rs[k]['median']:{fmt}} [{rs[k]['p5']:{fmt}}, {rs[k]['p95']:{fmt}}]"
    mdd_r = abs(rs["max_drawdown"]["median"]) / abs(base["max_drawdown"]) if base["max_drawdown"] < 0 else float("nan")
    return (f"| 随机对照 K=4（中位 [5%, 95%]） | {f('total_return', '+.1%')} | {f('ann_return', '+.1%')} | {f('max_drawdown', '.1%')} | "
            f"{mdd_r:.2f} | — | {f('calmar', '.2f')} | {f('sharpe', '.2f')} | {f('turnover_yr', '.1f')} | {rs['n_mean']['median']:.1f} | "
            f"{rs['cash_mean']['median']:.0%} | — |")


def trade_lines(res: RotationResult, name: str) -> list[str]:
    tr = res.trades.dropna(subset=["exit_date"])
    if tr.empty:
        return [f"- {name}：无完整进入—退出"]
    days = (tr["exit_date"] - tr["entry_date"]).dt.days
    reasons = ", ".join(f"{k} {v}" for k, v in tr["exit_reason"].value_counts().items())
    first = tr.sort_values("entry_date").iloc[0]
    return [f"- {name}：完整往返 {len(tr)} 笔，平均持有 {days.mean():.0f} 天，平均收益 {tr['ret'].mean():+.1%}；退出原因 {reasons}；"
            f"首笔 {first['code']} {first['entry_date'].date()} → {first['exit_date'].date()}（{first['exit_reason']}，{first['ret']:+.1%}）"]


def segment_lines(sname: str, bm: pd.DataFrame, results: dict[str, RotationResult], variants: dict, rnd: list[pd.DataFrame],
                  out_json: dict, uname: str) -> list[str]:
    s, e = SEGMENTS[sname]
    base = perf(bm, s, e); noharm = sname == "test"
    rs = random_summary([perf(d, s, e) for d in rnd]) if rnd else None
    lines = [f"### {sname} — {base['start'].date()} → {base['end'].date()}（{'noharm' if noharm else 'improve'}）", "", *HEADER,
             row("BM-S 等权（净）", base, None, "")]
    for name, res in results.items():
        m = perf(res.daily, s, e)
        v = acceptance(m, base, "noharm" if noharm else variants[name][1])
        if name == MAIN_NAME and rs:
            v += f"；随机：{'优' if beats_random(m, rs) else '劣'}"
        lines.append(row(name, m, base, v))
        out_json[f"{uname}|{sname}|{name}"] = {**{k: val for k, val in m.items() if k not in ("start", "end")}, "verdict": v}
    if rs:
        lines.append(random_row(rs, base)); out_json[f"{uname}|{sname}|random"] = rs
    return lines + [""]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, default=REPO_ROOT / "data" / "cache" / "cn_sector")
    ap.add_argument("--universes", nargs="+", default=["L1", "L2", "L1-upper", "L1-lower"])
    ap.add_argument("--segments", nargs="+", default=list(SEGMENTS))
    ap.add_argument("--variants", nargs="+", default=None, help="子集：按 VARIANTS 名称子串匹配（如 '主' 'K3' 'H-B1'）")
    ap.add_argument("--n-random", type=int, default=100, help="0 = 跳过随机对照")
    ap.add_argument("--random-every", type=int, default=MAIN.every,
                    help="随机对照的评估频率（登记值 = 主规则的 5；每次评估整组重抽 → 换手 ≈ 40x/年，可改 20 观察成本效应）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", default=BACKTEST_START)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    panel = load_panel(args.cache, ("close", "open", "amount"))
    ind = pd.read_parquet(args.cache / "sw_industry.parquet")
    l1 = list(ind.loc[ind["level"] == 1, "index_code"]); l2 = list(ind.loc[ind["level"] == 2, "index_code"])
    close = panel["close"].where(panel["close"] > 0); open_px = panel["open"].where(panel["open"] > 0)
    bm = bm_s(close, open_px, l1, start=args.start)
    variants = {k: v for k, v in VARIANTS.items() if not args.variants or any(s in k for s in args.variants)}
    unis = {k: v for k, v in build_universes(close, panel["amount"], l1, l2).items() if k in args.universes}

    lines = ["# E-B3 H-B1 / H-B2 行业趋势轮动（L2）", "",
             f"- Generated: `{datetime.now().isoformat(timespec='seconds')}`; 回测自 {args.start}；T 收盘评估、T+1 开盘成交；§1.3 成本；"
             f"BM-S = 31 个一级行业等权月度再平衡（净）；随机对照 = 同 K、同成本，每 {args.random_every} 日评估时在可用行业中随机重抽 K 个（不看趋势）",
             "- 验收：H-B1 风控层 MDD ≤ 2/3 BM-S、收益 ≥ 0.8 BM-S、Calmar > BM-S、换手 ≤ 12；H-B2 收益层 MDD ≤ BM-S、收益 ≥ 1.1 BM-S、"
             "Calmar > BM-S、换手 ≤ 12；test no-harm；主变体另标「随机：优/劣」= MDD 浅于随机对照中位数", ""]
    out_json: dict = {}
    for uname, (cols, mask) in unis.items():
        t0 = datetime.now()
        c, o = close.loc[:, cols], open_px.loc[:, cols]
        results = {name: run_variant(c, o, p, mask, args.start) for name, (p, _) in variants.items()}
        rp = RuleParams(**{**MAIN.__dict__, "every": args.random_every})
        rnd = [run_variant(c, o, rp, mask, args.start, selector=random_selector(rp), seed=args.seed + i).daily
               for i in range(args.n_random)]
        print(f"[{uname}] {len(results)} variants + {len(rnd)} random runs in {(datetime.now() - t0).seconds}s", file=sys.stderr)
        lines += [f"## 宇宙：{uname}（{len(cols)} 个行业）", ""]
        for sname in args.segments:
            lines += segment_lines(sname, bm, results, variants, rnd, out_json, uname)
        lines += ["#### 持仓日志（全样本）", ""]
        lines += [ln for name in (MAIN_NAME, "H-B1 N120*") if name in results for ln in trade_lines(results[name], name)] + [""]
    tag = f"_{args.tag}" if args.tag else ""
    rep = REPO_ROOT / "reports" / f"cn_backtest_sector_rotation{tag}.md"
    rep.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (REPO_ROOT / "reports" / f"cn_backtest_sector_rotation{tag}.json").write_text(json.dumps(out_json, indent=1, default=str), encoding="utf-8")
    print("\n".join(lines)); print(f"\nwrote {rep}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
