"""
E-A1 — 自建"全 A 等权指数"(BM) + 宇宙日表。
设计：GinkgoRoad docs/GinkgoBrain/A股日频-趋势底座与横截面增量-设计.md §1.1 / §2.3 / §4。
Usage:
    uv run python scripts/cn_build_ew_index.py --start 2015-12-01 [--end 2026-09-30] [--out data/cache/cn]
Outputs (parquet, gitignored):
    <out>/close.parquet / open.parquet / pre_close.parquet / amount.parquet / suspension.parquet  (日期 × 代码)
    <out>/universe.parquet   (日期 × 代码 bool；T 日收盘后可知)
    <out>/ew_index.parquet   (n, ret_gross, turnover, cost, ret_net, idx_gross, idx_net)
and reports/cn_ew_index_build.md (逐年宇宙数 / 停牌率 / 涨停不可买占比 / BM 分段指标).
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import text

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from strategies.cn_a_trend_base.costs import limit_up_open                       # noqa: E402
from strategies.cn_a_trend_base.ew_index import ew_index, summarize              # noqa: E402
from strategies.cn_a_trend_base.universe import st_codes, universe_mask, yearly_stats  # noqa: E402
from utils import db                                                             # noqa: E402

PERIODS = {"pre_val": ("2017-01-01", "2021-12-31"), "val": ("2022-01-01", "2025-06-30"),
           "test": ("2025-07-01", None)}


def load_long(start: str, end: str | None) -> pd.DataFrame:
    where = "trade_time >= :start" + (" AND trade_time < :end" if end else "")
    q = text(f"SELECT trade_time::date AS d, stock_code AS code, open_price, close_price, pre_close, "
             f"amount, suspension FROM cnstock_kline_day WHERE {where}")
    params = {"start": start, **({"end": end} if end else {})}
    with db.get_contract_engine().connect() as conn:
        df = pd.read_sql(q, conn, params=params, parse_dates=["d"])
    for c in ("open_price", "close_price", "pre_close", "amount"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    df["suspension"] = df["suspension"].fillna(1).astype("int8")
    return df


def to_wide(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return df.pivot(index="d", columns="code", values=col).sort_index()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2016-01-01", help="全市场日线自 2016-01-04 起（2015-12 仅 6 只）")
    ap.add_argument("--end", default=None)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "cache" / "cn")
    ap.add_argument("--from-cache", action="store_true", help="reuse <out>/*.parquet instead of querying the DB")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cols = ("open_price", "close_price", "pre_close", "amount", "suspension")
    if args.from_cache:
        wide = {c: pd.read_parquet(args.out / f"{c.replace('_price', '')}.parquet").loc[args.start:args.end] for c in cols}
        print(f"loaded wide frames from {args.out}: {wide['close_price'].shape} (sliced {args.start} → {args.end or 'latest'})")
    else:
        t0 = datetime.now()
        long = load_long(args.start, args.end)
        print(f"loaded {len(long):,} rows, {long['code'].nunique()} codes, "
              f"{long['d'].min().date()} → {long['d'].max().date()} in {(datetime.now() - t0).seconds}s")
        wide = {c: to_wide(long, c) for c in cols}
        del long
    names = db.execute("SELECT stock_code, stock_name FROM cnstock_security_list").set_index("stock_code")["stock_name"]
    st = st_codes(names)
    mask = universe_mask(wide["close_price"], wide["amount"], wide["suspension"], st)
    lu = limit_up_open(wide["open_price"], wide["pre_close"])
    bm = ew_index(wide["close_price"], mask)
    stats = yearly_stats(mask, wide["suspension"], wide["close_price"].notna(), lu)

    if not args.from_cache:
        for k, v in wide.items():
            v.to_parquet(args.out / f"{k.replace('_price', '')}.parquet")
    mask.to_parquet(args.out / "universe.parquet")
    bm.to_parquet(args.out / "ew_index.parquet")

    lines = ["# E-A1 自建全 A 等权指数（BM）构建报告", "",
             f"- Generated: `{datetime.now().isoformat(timespec='seconds')}`; window {args.start} → {args.end or 'latest'}",
             f"- Codes: {mask.shape[1]}; trading days: {len(mask)}; ST（当前名称）剔除 {len(st)} 只",
             f"- 成本：佣金 0.025% 双边、滑点 0.1% 单边、印花税 0.1%→0.05%（2023-08-28）；日度再平衡按 Σ|Δw| 计费", "",
             "## 逐年宇宙", "", "| 年 | 宇宙均值 | 停牌率 | 宇宙内开盘涨停占比 |", "|---|---:|---:|---:|"]
    for y, r in stats.iterrows():
        lines.append(f"| {y} | {r['universe_mean']:.0f} | {r['suspended_rate']:.2%} | {r['limit_up_in_universe']:.2%} |")
    lines += ["", "## BM 分段指标（idx_gross / idx_net）", "",
              "| 段 | 窗口 | 毛收益 | 净收益 | 净 MDD | 净 Calmar | 净 Sharpe | 年均换手 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for name, (s, e) in PERIODS.items():
        seg = bm.loc[s:e] if e else bm.loc[s:]
        if seg.empty:
            continue
        g, n = summarize(seg["idx_gross"]), summarize(seg["idx_net"])
        turn = seg["turnover"].sum() / n["years"] if n["years"] else float("nan")
        lines.append(f"| {name} | {seg.index[0].date()} → {seg.index[-1].date()} | {g['total_return']:+.1%} | "
                     f"{n['total_return']:+.1%} | {n['max_drawdown']:.1%} | {n['calmar']:.2f} | {n['sharpe']:.2f} | {turn:.1f}x |")
    yr = bm["ret_net"].groupby(bm.index.year).apply(lambda r: (1 + r).prod() - 1)
    lines += ["", "## BM 逐年净收益", "", "| 年 | 净收益 |", "|---|---:|"] + [f"| {y} | {v:+.1%} |" for y, v in yr.items()]
    rep = REPO_ROOT / "reports" / "cn_ew_index_build.md"
    rep.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {args.out}/ and {rep}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
