"""
E-B1 — 申万行业指数面板缓存 + BM-S + 逐年覆盖报告（L0）。
设计：GinkgoRoad docs/GinkgoBrain/A股日频-板块趋势轮动-设计.md §1.1 / §2.4 / §4 L0 / §6 E-B1。
Usage:
    uv run python scripts/cn_build_sector_panel.py [--start 1999-12-01] [--end 2026-09-30] [--out data/cache/cn_sector]
    uv run python scripts/cn_build_sector_panel.py --from-cache           # 只重算 BM-S 与报告
Outputs (parquet, gitignored):
    <out>/close.parquet / open.parquet / high.parquet / low.parquet / amount.parquet   (日期 × 代码，全部申万代码)
    <out>/sw_industry.parquet   (index_code, index_name, level)
    <out>/bm_s.parquet          (ret_net, ret_gross, turnover, cost, n, cash, idx_net；2005-01-04 起建仓)
    <out>/ref_000985.parquet    (close, open；中证全指，副基准只报告)
and reports/cn_sector_panel_build.md.
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

from strategies.cn_a_sector_rotation.metrics import perf                                       # noqa: E402
from strategies.cn_a_sector_rotation.panel import (                                             # noqa: E402
    BACKTEST_START, PANEL_FIELDS, SEGMENTS, bm_s, coverage_by_year, first_dates, gap_dates, load_panel, save_panel,
    sw_levels, to_wide, validate_panel,
)
from strategies.cn_a_trend_base.ew_index import summarize                                      # noqa: E402
from utils import db                                                                           # noqa: E402


def load_sw_long(start: str, end: str | None) -> pd.DataFrame:
    where = "source = 'sw' AND trade_date >= :start" + (" AND trade_date <= :end" if end else "")
    q = text(f"SELECT trade_date, index_code, index_name, open, high, low, close, volume, amount "
             f"FROM cnstock_index_kline_day WHERE {where}")
    with db.get_contract_engine().connect() as conn:
        df = pd.read_sql(q, conn, params={"start": start, **({"end": end} if end else {})}, parse_dates=["trade_date"])
    for c in PANEL_FIELDS + ("volume",):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    return df


def load_sw_industry() -> pd.DataFrame | None:
    try:
        with db.get_contract_engine().connect() as conn:
            return pd.read_sql(text("SELECT index_code, index_name, level, parent_code FROM cnstock_sw_industry"), conn)
    except Exception as e:                                        # 表未建：退回 SW_L1_CODES 常量（panel.sw_levels）
        print(f"warning: cnstock_sw_industry unavailable ({type(e).__name__}); falling back to SW_L1_CODES")
        return None


def load_ref(code: str) -> pd.DataFrame:
    q = text("SELECT trade_date, close, open FROM cnstock_index_kline_day WHERE source = 'csindex' AND index_code = :c")
    with db.get_contract_engine().connect() as conn:
        df = pd.read_sql(q, conn, params={"c": code}, parse_dates=["trade_date"]).set_index("trade_date").sort_index()
    return df.astype("float64")


def coverage_lines(close: pd.DataFrame, l1: list[str], l2: list[str], names: pd.Series) -> list[str]:
    cov1 = coverage_by_year(close.loc[BACKTEST_START:], l1)
    lines = [f"## 逐年覆盖（一级 {len(l1)} 个，{BACKTEST_START} 起；验收：覆盖 ≥ 95%，缺口不填零）", "",
             "| 年 | 日均可用 | 最少可用 | 覆盖率 | 缺口天数 | L0 |", "|---|---:|---:|---:|---:|:--|"]
    for y, r in cov1.iterrows():
        lines.append(f"| {y} | {r['mean_available']:.1f} | {int(r['min_available'])} | {r['coverage']:.1%} | "
                     f"{int(r['gap_days'])} | {'ok' if r['ok'] else 'FAIL'} |")
    gd = gap_dates(close.loc[BACKTEST_START:], l1)
    lines += ["", f"缺行日期（已有数据的一级行业中过半缺行；共 {len(gd)} 天，应回报 Spider D-A3 补采，不填零）："
              + ("、".join(f"{d.date()}({int(n)} 缺)" for d, n in gd.head(20).items()) if len(gd) else "无")]
    fd = first_dates(close.loc[:, l1]).sort_values()
    lines += ["", "## 一级行业首个交易日", "", "| 代码 | 名称 | 首日 |", "|---|---|---|"]
    lines += [f"| {c} | {names.get(c, '')} | {d.date() if pd.notna(d) else '—'} |" for c, d in fd.items()]
    if l2:
        cov2 = coverage_by_year(close.loc[BACKTEST_START:], l2)
        lines += ["", f"## 二级行业覆盖（{len(l2)} 个）", "", "| 年 | 日均可用 | 覆盖率 | 缺口天数 |", "|---|---:|---:|---:|"]
        lines += [f"| {y} | {r['mean_available']:.1f} | {r['coverage']:.1%} | {int(r['gap_days'])} |" for y, r in cov2.iterrows()]
    return lines


def bm_lines(bm: pd.DataFrame, ref: pd.DataFrame | None, ref_code: str) -> list[str]:
    lines = ["", "## BM-S 分段（31 个一级行业等权、每月首个交易日再平衡、T+1 开盘、§1.3 成本；净）", "",
             f"| 段 | 窗口 | 净收益 | 年化 | MDD | Calmar | Sharpe | 单边换手/年 | {ref_code} 收益 | {ref_code} MDD |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, (s, e) in SEGMENTS.items():
        seg = bm.loc[s:e] if e else bm.loc[s:]
        if seg.empty:
            continue
        m = perf(bm, s, e)
        r = summarize((ref["close"].loc[s:e] if e else ref["close"].loc[s:])) if ref is not None and len(ref) else None
        rc = f"{r['total_return']:+.1%} | {r['max_drawdown']:.1%}" if r else "— | —"
        lines.append(f"| {name} | {seg.index[0].date()} → {seg.index[-1].date()} | {m['total_return']:+.1%} | {m['ann_return']:+.1%} | "
                     f"{m['max_drawdown']:.1%} | {m['calmar']:.2f} | {m['sharpe']:.2f} | {m['turnover_yr']:.2f} | {rc} |")
    r = bm["ret_net"].loc[BACKTEST_START:]
    yr = r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1)
    lines += ["", "## BM-S 逐年净收益", "", "| 年 | 净收益 |", "|---|---:|"] + [f"| {y} | {v:+.1%} |" for y, v in yr.items()]
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="1999-12-01", help="全部历史（SMA 预热用）；回测/BM 自 2005-01-04")
    ap.add_argument("--end", default=None)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "cache" / "cn_sector")
    ap.add_argument("--ref-code", default="000985")
    ap.add_argument("--from-cache", action="store_true", help="reuse <out>/*.parquet instead of querying the DB")
    args = ap.parse_args()

    if args.from_cache:
        wide = load_panel(args.out)
        industry = pd.read_parquet(args.out / "sw_industry.parquet")
        ref = pd.read_parquet(args.out / f"ref_{args.ref_code}.parquet")
        print(f"loaded panel from {args.out}: {wide['close'].shape}")
    else:
        t0 = datetime.now()
        long = load_sw_long(args.start, args.end)
        if long.empty:
            print("no rows with source='sw' in cnstock_index_kline_day — D-A3 not ingested yet; nothing to build")
            return 1
        print(f"loaded {len(long):,} sw rows, {long['index_code'].nunique()} codes, "
              f"{long['trade_date'].min().date()} → {long['trade_date'].max().date()} in {(datetime.now() - t0).seconds}s")
        wide = {c: to_wide(long, c) for c in PANEL_FIELDS}
        names = long.drop_duplicates("index_code").set_index("index_code")["index_name"]
        l1, l2 = sw_levels(load_sw_industry(), list(wide["close"].columns))
        industry = pd.DataFrame({"index_code": l1 + l2, "index_name": [names.get(c, "") for c in l1 + l2],
                                 "level": [1] * len(l1) + [2] * len(l2)})
        ref = load_ref(args.ref_code)
        save_panel(args.out, wide)
        industry.to_parquet(args.out / "sw_industry.parquet")
        ref.to_parquet(args.out / f"ref_{args.ref_code}.parquet")
    validate_panel(wide["close"])
    names = industry.set_index("index_code")["index_name"]
    l1 = list(industry.loc[industry["level"] == 1, "index_code"]); l2 = list(industry.loc[industry["level"] == 2, "index_code"])
    if len(l1) != 31:
        print(f"warning: expected 31 SW L1 codes, got {len(l1)}: {l1}")
    bm = bm_s(wide["close"], wide["open"], l1, start=BACKTEST_START)
    bm.to_parquet(args.out / "bm_s.parquet")

    lines = ["# E-B1 申万行业指数面板 + BM-S 构建报告", "",
             f"- Generated: `{datetime.now().isoformat(timespec='seconds')}`; panel {wide['close'].index[0].date()} → "
             f"{wide['close'].index[-1].date()}，{len(l1)} 个一级 / {len(l2)} 个二级；BM-S 自 {BACKTEST_START} 建仓",
             "- 成本：佣金 0.025% 双边、滑点 0.1% 单边、印花税 0.1%→0.05%（2023-08-28）；T 收盘决策、T+1 开盘成交", ""]
    lines += coverage_lines(wide["close"], l1, l2, names)
    lines += bm_lines(bm, ref, args.ref_code)
    rep = REPO_ROOT / "reports" / "cn_sector_panel_build.md"
    rep.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines)); print(f"\nwrote {args.out}/ and {rep}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
