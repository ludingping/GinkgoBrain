"""
E-A3 — H-A3 反转多头组合（L2）：每 20 个交易日按最近 20 日收益取最低十分位等权，T+1 开盘成交，
涨停不买/跌停不卖，§1.2 成本；对 BM（月度再平衡等权，净）按 §1.1 收益层验收。
邻域：rev_10 / rev_40；子宇宙：沪 / 深。
Usage: uv run python scripts/cn_backtest_xs.py [--cache data/cache/cn] [--tag h_a3]
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
from strategies.cn_a_trend_base.ew_index import ew_index, summarize                  # noqa: E402
from strategies.cn_a_trend_base.xs_portfolio import long_portfolio_backtest         # noqa: E402
from strategies.cn_a_trend_base.xs_probe import reversal                             # noqa: E402

PERIODS = {"pre_val": ("2017-01-01", "2021-12-31"), "val": ("2022-01-01", "2025-06-30"), "test": ("2025-07-01", None)}
RETURN_LAYER = {"ret_mult": 1.10, "turnover_max": 12.0}


def seg(x, s, e):
    return x.loc[s:e] if e else x.loc[s:]


def accept(m: dict, base: dict, mode: str, turn_yr: float) -> str:
    if mode == "improve":
        checks = {"mdd": abs(m["max_drawdown"]) <= abs(base["max_drawdown"]) + 1e-12,
                  "return": m["total_return"] >= RETURN_LAYER["ret_mult"] * base["total_return"] if base["total_return"] > 0
                  else m["total_return"] > base["total_return"],
                  "calmar": m["calmar"] > base["calmar"], "turnover": turn_yr <= RETURN_LAYER["turnover_max"]}
    else:
        checks = {"mdd": abs(m["max_drawdown"]) <= abs(base["max_drawdown"]) + 1e-12,
                  "return": m["total_return"] >= 0.8 * base["total_return"], "turnover": turn_yr <= RETURN_LAYER["turnover_max"]}
    return "PASS" if all(checks.values()) else "FAIL(" + ",".join(k for k, v in checks.items() if not v) + ")"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, default=REPO_ROOT / "data" / "cache" / "cn")
    ap.add_argument("--tag", default="h_a3")
    ap.add_argument("--windows", type=int, nargs="+", default=[20, 10, 40])
    args = ap.parse_args()
    mask = pd.read_parquet(args.cache / "universe.parquet")
    close = pd.read_parquet(args.cache / "close.parquet").loc[mask.index, mask.columns]
    open_px = pd.read_parquet(args.cache / "open.parquet").loc[mask.index, mask.columns]
    pre_close = pd.read_parquet(args.cache / "pre_close.parquet").loc[mask.index, mask.columns]
    close = close.where(close > 0); open_px = open_px.where(open_px > 0)
    lu, ld = limit_up_open(open_px, pre_close), limit_down_open(open_px, pre_close)
    subs = {"全 A": mask.columns, "沪": [c for c in mask.columns if c.startswith("6")], "深": [c for c in mask.columns if not c.startswith("6")]}

    lines = [f"# E-A3 H-A3 反转多头组合（L2）", "",
             f"- Generated: `{datetime.now().isoformat(timespec='seconds')}`; 每 20 日选最近 N 日收益最低十分位等权；T+1 开盘成交；"
             f"涨停不买、跌停不卖；§1.2 成本；BM = 月度再平衡等权（净）",
             f"- 验收（收益层）：MDD 不深于 BM、收益 ≥ 1.1 × BM、Calmar > BM、年单边换手 ≤ 12；test no-harm", ""]
    for uname, cols in subs.items():
        um = mask.loc[:, cols]; c = close.loc[:, cols]; o = open_px.loc[:, cols]
        bm = ew_index(c, um, "M")
        lines += [f"## 宇宙：{uname}（{len(cols)} 只）", ""]
        results = {}
        for n in args.windows:
            feat = reversal(c, n)
            results[f"rev_{n}"] = long_portfolio_backtest(c, o, feat, um, every=20, q=10, bottom=True,
                                                          limit_up=lu.loc[:, cols], limit_down=ld.loc[:, cols], start="2016-12-01")
        for pname, (s, e) in PERIODS.items():
            mode = "noharm" if pname == "test" else "improve"
            b = seg(bm, s, e); base = summarize((1 + b["ret_net"]).cumprod())
            lines += [f"### {pname} — {b.index[0].date()} → {b.index[-1].date()}（{mode}）", "",
                      "| 策略 | 总收益 | 年化 | MDD | MDD 比 \\| 收益比 | Calmar | Sharpe | 换手/年 | 持仓数 | 验收 |",
                      "|---|---:|---:|---:|---|---:|---:|---:|---:|:--|",
                      f"| BM 等权（净） | {base['total_return']:+.1%} | {base['ann_return']:+.1%} | {base['max_drawdown']:.1%} | — | "
                      f"{base['calmar']:.2f} | {base['sharpe']:.2f} | {b['turnover'].sum() / base['years']:.1f} | {b['n'].mean():.0f} | |"]
            for rname, r in results.items():
                rs = seg(r, s, e); m = summarize((1 + rs["ret_net"]).cumprod())
                turn_yr = rs["turnover"].sum() / m["years"] / 2.0          # 单边
                ratio = f"{abs(m['max_drawdown']) / abs(base['max_drawdown']):.2f} | " + (f"{m['total_return'] / base['total_return']:.2f}" if base["total_return"] > 0 else "—")
                lines.append(f"| {rname} 最低十分位 | {m['total_return']:+.1%} | {m['ann_return']:+.1%} | {m['max_drawdown']:.1%} | {ratio} | "
                             f"{m['calmar']:.2f} | {m['sharpe']:.2f} | {turn_yr:.1f} | {rs['n'].mean():.0f} | {accept(m, base, mode, turn_yr)} |")
            lines.append("")
    rep = REPO_ROOT / "reports" / f"cn_backtest_xs_{args.tag}.md"
    rep.write_text("\n".join(lines) + "\n", encoding="utf-8"); print("\n".join(lines)); print(f"\nwrote {rep}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
