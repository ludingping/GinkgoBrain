"""
H-A3b — BM 等权 + H-A3 反转十分位 固定比例混合（事后登记）。
Usage: uv run python scripts/cn_backtest_blend.py [--rev-weights 0.3 0.2 0.4] [--tag h_a3b]
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from strategies.cn_a_trend_base.costs import limit_down_open, limit_up_open          # noqa: E402
from strategies.cn_a_trend_base.ew_index import blend_sleeves, ew_index, summarize   # noqa: E402
from strategies.cn_a_trend_base.xs_portfolio import long_portfolio_backtest         # noqa: E402
from strategies.cn_a_trend_base.xs_probe import reversal                             # noqa: E402
from scripts.cn_backtest_xs import PERIODS, accept, seg                              # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, default=REPO_ROOT / "data" / "cache" / "cn")
    ap.add_argument("--rev-weights", type=float, nargs="+", default=[0.3, 0.2, 0.4])
    ap.add_argument("--tag", default="h_a3b")
    args = ap.parse_args()
    mask = pd.read_parquet(args.cache / "universe.parquet")
    close = pd.read_parquet(args.cache / "close.parquet").loc[mask.index, mask.columns].where(lambda x: x > 0)
    open_px = pd.read_parquet(args.cache / "open.parquet").loc[mask.index, mask.columns].where(lambda x: x > 0)
    pre_close = pd.read_parquet(args.cache / "pre_close.parquet").loc[mask.index, mask.columns]
    lu, ld = limit_up_open(open_px, pre_close), limit_down_open(open_px, pre_close)
    subs = {"全 A": list(mask.columns), "沪": [c for c in mask.columns if c.startswith("6")], "深": [c for c in mask.columns if not c.startswith("6")]}
    lines = [f"# H-A3b BM 等权 + 反转十分位混合（事后登记）", "",
             f"- Generated: `{datetime.now().isoformat(timespec='seconds')}`; 反转腿 = rev_20 最低十分位（每 20 日、T+1 开盘、涨跌停规则）；"
             f"腿间比例每月首个交易日复位；反转占比 {args.rev_weights}",
             f"- 验收（收益层）：MDD ≤ BM、收益 ≥ 1.1 × BM、Calmar > BM、换手 ≤ 12；test no-harm", ""]
    for uname, cols in subs.items():
        um = mask.loc[:, cols]; c = close.loc[:, cols]; o = open_px.loc[:, cols]
        bm = ew_index(c, um, "M")
        rev = long_portfolio_backtest(c, o, reversal(c, 20), um, every=20, q=10, bottom=True,
                                      limit_up=lu.loc[:, cols], limit_down=ld.loc[:, cols], start="2016-12-01")
        rets = pd.DataFrame({"bm": bm["ret_net"], "rev": rev["ret_net"]})
        blends = {w: blend_sleeves(rets, {"bm": 1 - w, "rev": w}, "M") for w in args.rev_weights}
        lines += [f"## 宇宙：{uname}", ""]
        for pname, (s, e) in PERIODS.items():
            mode = "noharm" if pname == "test" else "improve"
            b = seg(bm, s, e); base = summarize((1 + b["ret_net"]).cumprod())
            lines += [f"### {pname} — {b.index[0].date()} → {b.index[-1].date()}（{mode}）", "",
                      "| 策略 | 总收益 | 年化 | MDD | MDD 比 \\| 收益比 | Calmar | Sharpe | 换手/年 | 验收 |", "|---|---:|---:|---:|---|---:|---:|---:|:--|",
                      f"| BM 等权（净） | {base['total_return']:+.1%} | {base['ann_return']:+.1%} | {base['max_drawdown']:.1%} | — | {base['calmar']:.2f} | {base['sharpe']:.2f} | {b['turnover'].sum() / base['years']:.1f} | |"]
            for w, bl in blends.items():
                bs = seg(bl, s, e); m = summarize((1 + bs["ret_net"]).cumprod())
                rs = seg(rev, s, e)
                turn_yr = (w * rs["turnover"].sum() / 2.0 + (1 - w) * seg(b, s, e)["turnover"].sum() + bs["turnover"].sum()) / m["years"]
                ratio = f"{abs(m['max_drawdown']) / abs(base['max_drawdown']):.2f} | " + (f"{m['total_return'] / base['total_return']:.2f}" if base["total_return"] > 0 else "—")
                lines.append(f"| 混合 反转 {w:.0%} | {m['total_return']:+.1%} | {m['ann_return']:+.1%} | {m['max_drawdown']:.1%} | {ratio} | "
                             f"{m['calmar']:.2f} | {m['sharpe']:.2f} | {turn_yr:.1f} | {accept(m, base, mode, turn_yr)} |")
            lines.append("")
    rep = REPO_ROOT / "reports" / f"cn_backtest_blend_{args.tag}.md"
    rep.write_text("\n".join(lines) + "\n", encoding="utf-8"); print("\n".join(lines)); print(f"\nwrote {rep}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
