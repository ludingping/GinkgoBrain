"""
E-A2 — A 股横截面 IC 探针（L1）。H-A2 动量 / H-A3 反转 / H-A4 低波动，一批登记一批报告。
设计：GinkgoRoad docs/GinkgoBrain/A股日频-趋势底座与横截面增量-设计.md §3 L1 / §5。
折：f1 2017→2019、f2 2020→2022、f3 2023→2025-06；test 2025-07→ 留出单列。前向收益 = T+1 开盘 → T+1+k 开盘。
Usage:
    uv run python scripts/cn_xs_ic_probe.py [--cache data/cache/cn] [--horizons 5 20] [--tag h_a2_a4]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from strategies.cn_a_trend_base.xs_probe import (                                # noqa: E402
    daily_rank_ic, decile_returns, fold_stats, forward_open_return, momentum, realized_vol, reversal, verdict,
)

FOLDS = {"f1": ("2017-01-01", "2019-12-31"), "f2": ("2020-01-01", "2022-12-31"), "f3": ("2023-01-01", "2025-06-30")}
TEST = {"test": ("2025-07-01", None)}
# 登记的特征与方向（设计 §5）；* = 主变体
FEATURES = {
    "H-A2 动量": {"sign": +1, "items": {"mom_12_1*": lambda c: momentum(c, 12), "mom_9_1": lambda c: momentum(c, 9),
                                        "mom_6_1": lambda c: momentum(c, 6)}},
    "H-A3 反转": {"sign": -1, "items": {"rev_20*": lambda c: reversal(c, 20), "rev_10": lambda c: reversal(c, 10),
                                        "rev_40": lambda c: reversal(c, 40)}},
    "H-A4 低波动": {"sign": -1, "items": {"vol_60*": lambda c: realized_vol(c, 60), "vol_40": lambda c: realized_vol(c, 40),
                                          "vol_120": lambda c: realized_vol(c, 120)}},
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, default=REPO_ROOT / "data" / "cache" / "cn")
    ap.add_argument("--horizons", type=int, nargs="+", default=[5, 20])
    ap.add_argument("--tag", default="h_a2_a4")
    args = ap.parse_args()

    mask = pd.read_parquet(args.cache / "universe.parquet")
    close = pd.read_parquet(args.cache / "close.parquet").loc[mask.index, mask.columns]
    open_px = pd.read_parquet(args.cache / "open.parquet").loc[mask.index, mask.columns]
    close = close.where(close > 0); open_px = open_px.where(open_px > 0)
    sh = [c for c in mask.columns if c.startswith("6")]
    sz = [c for c in mask.columns if not c.startswith("6")]
    subs = {"全 A": mask, "沪": mask.loc[:, sh], "深": mask.loc[:, sz]}   # 列子集；feature.where(mask) 按列对齐
    fwd = {k: forward_open_return(open_px, k) for k in args.horizons}
    names_all = mask.sum(axis=1)

    lines = [f"# E-A2 横截面 IC 探针 — H-A2 / H-A3 / H-A4", "",
             f"- Generated: `{datetime.now().isoformat(timespec='seconds')}`; horizons {args.horizons} 交易日；"
             f"前向收益 T+1 开盘 → T+1+k 开盘；逐日 Spearman，宇宙成员 ≥ 200",
             f"- 折：f1 2017→2019、f2 2020→2022、f3 2023→2025-06；test 2025-07→ 留出（只报告）",
             f"- 门槛：每折 |IC| ≥ 0.02、NW-t ≥ 2、与登记方向同号、≥ 120 天；十分位价差同向；沪/深同向", ""]
    out_json: dict = {}
    for hyp, spec in FEATURES.items():
        sign = spec["sign"]
        lines += [f"## {hyp}（登记方向 IC {'>' if sign > 0 else '<'} 0）", ""]
        for fname, fn in spec["items"].items():
            feat = fn(close)
            for k in args.horizons:
                lines += [f"### {fname} · k={k}", "",
                          "| 宇宙 | f1 IC (t, n) | f2 IC (t, n) | f3 IC (t, n) | test IC (t, n) | D10−D1 (k 日, f1/f2/f3) | 判定 |",
                          "|---|---|---|---|---|---|:--|"]
                for uname, um in subs.items():
                    ic = daily_rank_ic(feat.loc[:, um.columns], fwd[k].loc[:, um.columns], um)
                    st = fold_stats(ic, um.sum(axis=1), FOLDS, lags=k)
                    te = fold_stats(ic, um.sum(axis=1), TEST, lags=k)[0]
                    ok, why = verdict(st, sign)
                    dec = decile_returns(feat.loc[:, um.columns], fwd[k].loc[:, um.columns], um, every=k if k >= 5 else 5)
                    spread = []
                    for _, (s, e) in FOLDS.items():
                        d = dec.loc[s:e]
                        spread.append(f"{(d['D10'] - d['D1']).mean() * 1e4:+.0f}" if len(d) else "—")
                    cells = " | ".join(f"{f.ic_mean:+.3f} ({f.ic_t:.1f}, {f.n_days})" for f in st)
                    lines.append(f"| {uname} | {cells} | {te.ic_mean:+.3f} ({te.ic_t:.1f}, {te.n_days}) | "
                                 f"{'/'.join(spread)} bps | {'PASS' if ok else 'FAIL (' + why + ')'} |")
                    out_json[f"{fname}|k{k}|{uname}"] = {"folds": [f.__dict__ for f in st], "test": te.__dict__,
                                                        "spread_bps": spread, "pass": ok, "why": why}
                lines.append("")
    rep = REPO_ROOT / "reports" / f"cn_xs_ic_probe_{args.tag}.md"
    rep.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (REPO_ROOT / "reports" / f"cn_xs_ic_probe_{args.tag}.json").write_text(json.dumps(out_json, indent=1, default=str), encoding="utf-8")
    print("\n".join(lines)); print(f"\nwrote {rep}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
