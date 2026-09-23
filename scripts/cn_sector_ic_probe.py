"""
E-B2 — H-B3 行业相对强度 rank IC 探针（L1）。不加趋势过滤；周频截面；三段 + 子宇宙 + 五分位。
设计：GinkgoRoad docs/GinkgoBrain/A股日频-板块趋势轮动-设计.md §4 L1 / §5 H-B3 / §6 E-B2。
前向收益 = T+1 开盘 → T+1+k 开盘（与回测撮合一致）。t 统计：周频 IC 序列的 Newey–West t，lags = ceil(k/5)（重叠期）。
Usage:
    uv run python scripts/cn_sector_ic_probe.py [--cache data/cache/cn_sector] [--L 60 40 120] [--k 20 60] [--every 5]
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

from strategies.cn_a_sector_rotation.features import strength                                  # noqa: E402
from strategies.cn_a_sector_rotation.ic_probe import (                                          # noqa: E402
    IC_MIN, N_MIN_WEEKS, T_MIN, amount_split_masks, l1_verdict, quantile_spread, segment_stats, spread_by_segment,
    weekly_rank_ic,
)
from strategies.cn_a_sector_rotation.panel import BACKTEST_START, SEGMENTS, load_panel        # noqa: E402
from strategies.cn_a_trend_base.xs_probe import forward_open_return                             # noqa: E402


def universes(close: pd.DataFrame, amount: pd.DataFrame, l1: list[str], l2: list[str]) -> dict[str, pd.DataFrame]:
    """§2.4：一级全部；二级全部；一级按 20 日成交额中位数切上 / 下半区。掩码 = 当日有收盘价 ∧ 子宇宙。"""
    have = close.notna()
    up, lo = amount_split_masks(amount.loc[:, l1], window=20)
    out = {"L1": have.loc[:, l1], "L1-upper": have.loc[:, l1] & up, "L1-lower": have.loc[:, l1] & lo}
    if l2:
        out["L2"] = have.loc[:, l2]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, default=REPO_ROOT / "data" / "cache" / "cn_sector")
    ap.add_argument("--L", type=int, nargs="+", default=[60, 40, 120], help="强度窗口；第一个为主参数")
    ap.add_argument("--k", type=int, nargs="+", default=[20, 60])
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--tag", default="h_b3")
    args = ap.parse_args()

    panel = load_panel(args.cache, ("close", "open", "amount"))
    ind = pd.read_parquet(args.cache / "sw_industry.parquet")
    l1 = list(ind.loc[ind["level"] == 1, "index_code"]); l2 = list(ind.loc[ind["level"] == 2, "index_code"])
    close = panel["close"].where(panel["close"] > 0); open_px = panel["open"].where(panel["open"] > 0)
    subs = universes(close, panel["amount"], l1, l2)
    fwd = {k: forward_open_return(open_px, k) for k in args.k}
    feats = {L: strength(close, L) for L in args.L}
    seg_from = {n: (max(s, BACKTEST_START), e) for n, (s, e) in SEGMENTS.items()}

    lines = ["# E-B2 H-B3 行业相对强度 rank IC 探针（L1）", "",
             f"- Generated: `{datetime.now().isoformat(timespec='seconds')}`; strength = 过去 L 日收益，不加趋势过滤；"
             f"每 {args.every} 个交易日一个截面（周频），成员 ≥ 10；前向收益 T+1 开盘 → T+1+k 开盘",
             f"- 段：pre_val {SEGMENTS['pre_val'][0]}→{SEGMENTS['pre_val'][1]}、val {SEGMENTS['val'][0]}→{SEGMENTS['val'][1]}、"
             f"test {SEGMENTS['test'][0]}→；NW-t lags = ceil(k/every)（重叠期）",
             f"- 门槛（§4 L1）：三段 IC ≥ {IC_MIN}、t ≥ {T_MIN}、三段同号且最近段不反号、每段 ≥ {N_MIN_WEEKS} 个截面；"
             "五分位 Q5−Q1 > 0；二级行业同向", ""]
    out_json: dict = {}
    for L, feat in feats.items():
        for k in args.k:
            lines += [f"## L={L}{' (主)' if L == args.L[0] else ''} · k={k}", "",
                      "| 宇宙 | pre_val IC (t, n) | val IC (t, n) | test IC (t, n) | Q5−Q1 (bps, pre/val/test) | 判定 |",
                      "|---|---|---|---|---|:--|"]
            for uname, um in subs.items():
                f = feat.loc[:, um.columns]; r = fwd[k].loc[:, um.columns]
                ic = weekly_rank_ic(f, r, um, every=args.every).loc[BACKTEST_START:]
                st = segment_stats(ic, um.sum(axis=1), seg_from, k, args.every)
                ok, why = l1_verdict(st)
                sl = slice(BACKTEST_START, None)
                sp = spread_by_segment(quantile_spread(f.loc[sl], r.loc[sl], um.loc[sl], every=args.every, q=5), seg_from)
                cells = " | ".join(f"{s.ic_mean:+.3f} ({s.ic_t:.1f}, {s.n_days})" for s in st)
                spread = "/".join(f"{v * 1e4:+.0f}" if pd.notna(v) else "—" for v in sp.values())
                lines.append(f"| {uname} ({um.shape[1]}) | {cells} | {spread} | {'PASS' if ok else 'FAIL (' + why + ')'} |")
                out_json[f"L{L}|k{k}|{uname}"] = {"segments": [s.__dict__ for s in st], "spread": sp, "pass": ok, "why": why}
            lines.append("")
    rep = REPO_ROOT / "reports" / f"cn_{args.tag}_probe.md"
    rep.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (REPO_ROOT / "reports" / f"cn_{args.tag}_probe.json").write_text(json.dumps(out_json, indent=1, default=str), encoding="utf-8")
    print("\n".join(lines)); print(f"\nwrote {rep}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
