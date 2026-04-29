"""
A 股持仓股票每日信号扫描器（盘后 15:30 跑，输出 T+1 交易决策参考）。

工作流程：
1. 从 public.positions 拉当前 OPEN 的 A 股持仓
2. 对每只股票：
   - 拉最近的 day K（含目标日）
   - 计算 5 个 indicator regime 的当前状态
   - 检测目标日是否为某 regime 的"首日"（事件触发）
   - 对照 notebooks/cnstock_<code>_signals.csv，看命中哪些已知信号
3. 生成 markdown 报告 → reports/daily_signals/YYYYMMDD.md

Usage:
    uv run python scripts/cnstock_daily_signals.py
    uv run python scripts/cnstock_daily_signals.py --as-of 2026-04-22
    uv run python scripts/cnstock_daily_signals.py --codes 300866,002594

Cron 示例（每个交易日 15:30）：
    30 15 * * 1-5  cd /path/to/GinkgoBrain && uv run python scripts/cnstock_daily_signals.py
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.db import get_contract_engine, read_cnstock_kline_day  # noqa: E402

# ─── Defaults ───────────────────────────────────────────────────────────────

DEFAULT_OUT_DIR = REPO_ROOT / "reports" / "daily_signals"
DEFAULT_SIG_DIR = REPO_ROOT / "notebooks"
TRADING_DAYS_PER_YEAR = 252

# A 股代码正则：6xx (沪)、0xx (深主板)、3xx (创业板)、688 (科创板)；排除 8/4xxx (北交所)
A_SHARE_REGEX = r"^[036]\d{5}$"

INDICATORS = ["vol_21d", "trend", "drawdown_60d", "volume_z_21d", "rsi_14"]
INDICATOR_TO_REGIME_COL = {
    "vol_21d":      "regime_vol",
    "trend":        "regime_trend",
    "drawdown_60d": "regime_dd",
    "volume_z_21d": "regime_vol_z",
    "rsi_14":       "regime_rsi",
}


# ─── Data: positions + day K ────────────────────────────────────────────────


@dataclass
class HeldStock:
    code: str
    name: str
    quantity: float
    cost_price: float
    current_price: float
    strategy: str | None
    status: str


def fetch_held_stocks() -> list[HeldStock]:
    """从 positions 表拉当前 OPEN 的 A 股持仓。"""
    with get_contract_engine().connect() as conn:
        rows = conn.execute(text(f"""
            SELECT security_code, security_name, quantity,
                   cost_price, current_price, strategy, status
            FROM public.positions
            WHERE (status IS NULL OR status != 'closed')
              AND quantity > 0
              AND security_code ~ '{A_SHARE_REGEX}'
            ORDER BY security_code
        """)).fetchall()
    by_code: dict[str, HeldStock] = {}
    for r in rows:
        code = r[0]
        if code in by_code:
            prev = by_code[code]
            new_qty = prev.quantity + (r[2] or 0)
            new_cost = (prev.quantity * prev.cost_price + (r[2] or 0) * (r[3] or 0)) / max(new_qty, 1)
            by_code[code] = HeldStock(
                code=code, name=r[1] or "?", quantity=new_qty, cost_price=new_cost,
                current_price=r[4] or 0.0, strategy=r[5], status=r[6],
            )
        else:
            by_code[code] = HeldStock(
                code=code, name=r[1] or "?", quantity=r[2] or 0,
                cost_price=r[3] or 0, current_price=r[4] or 0,
                strategy=r[5], status=r[6],
            )
    return list(by_code.values())


def load_day(code: str, end_date: datetime) -> pd.DataFrame:
    """拉到 end_date（含）为止的日线，回溯 300 自然日 ≈ 200 交易日 保证 indicator 有效。"""
    start = (end_date - timedelta(days=300)).strftime("%Y-%m-%d")
    end   = (end_date + timedelta(days=1)).strftime("%Y-%m-%d")  # exclusive upper
    df = read_cnstock_kline_day(code, start=start, end=end)
    if df.empty:
        return df
    df = df.copy()
    df["ts_cst"] = df["trade_time"].dt.tz_convert("Asia/Shanghai")
    df["date"]   = pd.to_datetime(df["ts_cst"].dt.date)
    return df.sort_values("date").reset_index(drop=True)


# ─── Indicator computation (与 multi_stock 脚本同) ──────────────────────────


def build_regime_columns(df_day: pd.DataFrame) -> pd.DataFrame:
    df = df_day.copy()
    df["ret"]     = df["close_price"].pct_change()
    df["vol_21d"] = df["ret"].rolling(21).std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    vp = df["vol_21d"] * 100
    lo, hi = vp.quantile(0.33), vp.quantile(0.67)
    df["regime_vol"] = pd.cut(vp, bins=[-np.inf, lo, hi, np.inf], labels=["低 vol", "中 vol", "高 vol"])

    df["ma_60"]       = df["close_price"].rolling(60).mean()
    df["ma_60_slope"] = df["ma_60"].diff(5)

    def _trend(row):
        if pd.isna(row["ma_60"]) or pd.isna(row["ma_60_slope"]):
            return np.nan
        above = row["close_price"] > row["ma_60"]
        up    = row["ma_60_slope"] > 0
        if above and up:           return "1.强势(上+涨)"
        if above and not up:       return "2.顶背离(上+跌)"
        if not above and up:       return "3.反弹(下+涨)"
        return "4.弱势(下+跌)"

    df["regime_trend"] = df.apply(_trend, axis=1)

    rh = df["close_price"].rolling(60, min_periods=20).max()
    df["dd_60d"] = df["close_price"] / rh - 1

    def _dd(d):
        if pd.isna(d):  return np.nan
        if d >= -0.05:  return "A.高点±5%"
        if d >= -0.15:  return "B.回撤5-15%"
        if d >= -0.25:  return "C.回撤15-25%"
        return                 "D.深回撤>25%"

    df["regime_dd"] = df["dd_60d"].apply(_dd)

    log_vol = np.log(df["volume"].replace(0, np.nan))
    df["vol_z_21d"] = (log_vol - log_vol.rolling(21).mean()) / log_vol.rolling(21).std()
    df["regime_vol_z"] = pd.cut(
        df["vol_z_21d"],
        bins=[-np.inf, -1.0, -0.5, 0.5, 1.0, np.inf],
        labels=["A.缩量<-1σ", "B.偏缩-1~-0.5σ", "C.中性±0.5σ", "D.偏放0.5~1σ", "E.放量>1σ"],
    )

    delta = df["close_price"].diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_g = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_l = loss.ewm(alpha=1/14, adjust=False).mean()
    df["rsi_14"] = 100 - 100 / (1 + avg_g / avg_l.replace(0, np.nan))
    df["regime_rsi"] = pd.cut(
        df["rsi_14"],
        bins=[0, 30, 50, 70, 100],
        labels=["A.超卖<30", "B.弱30-50", "C.强50-70", "D.超买>70"],
        include_lowest=True,
    )

    return df


# ─── Trigger detection ──────────────────────────────────────────────────────


@dataclass
class TriggeredSignal:
    signal_id: str
    indicator: str
    regime: str
    h: int
    direction: str
    tier: str
    size_pct: int
    n: int
    mean_pct: float
    win_pct: float
    t: float


def detect_triggers(
    df: pd.DataFrame,
    signal_lib: pd.DataFrame,
    target_date: pd.Timestamp,
) -> tuple[dict[str, str], list[TriggeredSignal]]:
    """
    返回 (今日 5 个 regime 状态, 今日触发的信号列表)。

    "触发" = 目标日的 regime 与昨日不同（即 first day of regime）。
    然后到该股票的 signal_lib 里找 (indicator × today's regime) 同时存在的条目。
    """
    if target_date not in df["date"].values:
        return {}, []

    idx_today = df.index[df["date"] == target_date][0]
    if idx_today == 0:
        return {}, []
    row_today = df.iloc[idx_today]
    row_yesterday = df.iloc[idx_today - 1]

    states: dict[str, str] = {}
    triggered: list[TriggeredSignal] = []

    for indicator in INDICATORS:
        regime_col = INDICATOR_TO_REGIME_COL[indicator]
        today_val   = row_today[regime_col]
        yest_val    = row_yesterday[regime_col]
        if pd.isna(today_val):
            states[indicator] = "—"
            continue
        states[indicator] = str(today_val)

        is_event = (today_val != yest_val) and not pd.isna(today_val)
        if not is_event:
            continue

        matches = signal_lib[
            (signal_lib["indicator"] == indicator) &
            (signal_lib["regime"] == str(today_val))
        ]
        for _, sig in matches.iterrows():
            triggered.append(TriggeredSignal(
                signal_id=sig["signal_id"],
                indicator=sig["indicator"],
                regime=sig["regime"],
                h=int(sig["h"]),
                direction=sig["direction"],
                tier=sig["tier"],
                size_pct=int(sig["size_pct"]),
                n=int(sig["n"]),
                mean_pct=float(sig["mean_pct"]),
                win_pct=float(sig["win_pct"]),
                t=float(sig["t"]),
            ))
    return states, triggered


def state_context(df: pd.DataFrame, target_date: pd.Timestamp) -> dict:
    """除了 regime label 还想看的一些数值，让报告更有 context。"""
    if target_date not in df["date"].values:
        return {}
    row = df.iloc[df.index[df["date"] == target_date][0]]
    return {
        "close":     float(row["close_price"]) if not pd.isna(row["close_price"]) else None,
        "vol_21d":   float(row["vol_21d"] * 100) if not pd.isna(row["vol_21d"]) else None,
        "dd_60d":    float(row["dd_60d"] * 100) if not pd.isna(row["dd_60d"]) else None,
        "rsi_14":    float(row["rsi_14"]) if not pd.isna(row["rsi_14"]) else None,
        "vol_z_21d": float(row["vol_z_21d"]) if not pd.isna(row["vol_z_21d"]) else None,
    }


# ─── Reporting ──────────────────────────────────────────────────────────────


TIER_EMOJI = {"high": "🟢", "medium": "🔵", "low": "🟡", "trial": "🟠"}


def render_stock_section(
    stock: HeldStock,
    states: dict[str, str],
    triggered: list[TriggeredSignal],
    ctx: dict,
    n_signals_in_lib: int,
) -> str:
    pnl_pct = (stock.current_price / stock.cost_price - 1) * 100 if stock.cost_price > 0 else 0
    lines = [
        f"## {stock.code} {stock.name}",
        f"",
        f"**持仓**: qty={stock.quantity:.0f}  cost={stock.cost_price:.2f}  curr={stock.current_price:.2f}  "
        f"P&L={pnl_pct:+.1f}%  策略={stock.strategy or '—'}",
        f"**信号库**: {n_signals_in_lib} 条 (`notebooks/cnstock_{stock.code}_signals.csv`)",
        f"",
        f"**今日 regime 状态**:",
    ]

    fmt_ctx = []
    if ctx.get("vol_21d") is not None:
        fmt_ctx.append(f"vol_21d={ctx['vol_21d']:.1f}%")
    if ctx.get("dd_60d") is not None:
        fmt_ctx.append(f"dd_60d={ctx['dd_60d']:+.1f}%")
    if ctx.get("rsi_14") is not None:
        fmt_ctx.append(f"rsi_14={ctx['rsi_14']:.1f}")
    if ctx.get("vol_z_21d") is not None:
        fmt_ctx.append(f"vol_z={ctx['vol_z_21d']:+.2f}")
    ctx_line = "  ".join(fmt_ctx)

    lines += [
        f"- vol regime: **{states.get('vol_21d', '—')}**",
        f"- trend: **{states.get('trend', '—')}**",
        f"- drawdown: **{states.get('drawdown_60d', '—')}**",
        f"- volume_z: **{states.get('volume_z_21d', '—')}**",
        f"- rsi_14: **{states.get('rsi_14', '—')}**",
        f"",
        f"({ctx_line})",
        f"",
    ]

    if not triggered:
        lines += [
            f"**今日触发**: 无",
            f"",
            f"→ 建议：维持现有仓位，无入场/出场信号。",
            f"",
        ]
        return "\n".join(lines)

    lines.append(f"**今日触发** ({len(triggered)} 个):")
    lines.append(f"")
    for sig in triggered:
        emoji = TIER_EMOJI.get(sig.tier, "")
        action = "买入 / 加仓" if sig.direction == "long" else "卖出 / 减仓 / 对冲做空"
        lines += [
            f"- {emoji} **{sig.tier.upper()}** `{sig.signal_id}` ({sig.direction}, h={sig.h} 日)",
            f"  - 历史: n={sig.n}, mean={sig.mean_pct:+.2f}%, win={sig.win_pct:.0f}%, t={sig.t:+.2f}",
            f"  - **建议**: T+1 {action}，目标 size **{sig.size_pct}%**，hold {sig.h} 日",
            f"",
        ]
    return "\n".join(lines)


def render_summary_table(per_stock: list[dict]) -> str:
    lines = [
        f"| 代码 | 名称 | P&L | 触发数 | 净方向 | 推荐仓位变化 |",
        f"|------|------|-----|--------|--------|--------------|",
    ]
    for ps in per_stock:
        n_long  = sum(1 for s in ps["triggered"] if s.direction == "long")
        n_short = sum(1 for s in ps["triggered"] if s.direction == "short")
        if n_long > n_short:
            net = f"LONG ({n_long}↑)"
            best = max((s for s in ps["triggered"] if s.direction == "long"),
                       key=lambda s: s.size_pct, default=None)
            rec = f"加仓 ~{best.size_pct}%" if best else "—"
        elif n_short > n_long:
            net = f"SHORT ({n_short}↓)"
            best = max((s for s in ps["triggered"] if s.direction == "short"),
                       key=lambda s: s.size_pct, default=None)
            rec = f"减仓/对冲 ~{best.size_pct}%" if best else "—"
        elif n_long == n_short and n_long > 0:
            net = "MIXED"
            rec = "信号冲突，观望"
        else:
            net = "—"
            rec = "维持"

        stock = ps["stock"]
        pnl = (stock.current_price / stock.cost_price - 1) * 100 if stock.cost_price > 0 else 0
        lines.append(
            f"| {stock.code} | {stock.name} | {pnl:+.1f}% | {len(ps['triggered'])} | {net} | {rec} |"
        )
    return "\n".join(lines)


# ─── Main ───────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="A 股持仓股票每日信号扫描器")
    parser.add_argument("--as-of",   default=None,
                        help="目标日期 YYYY-MM-DD（默认 = 数据库中最新交易日）")
    parser.add_argument("--codes",   default=None,
                        help="逗号分隔股票代码（默认 = 当前持仓 A 股）")
    parser.add_argument("--sig-dir", default=str(DEFAULT_SIG_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    sig_dir = Path(args.sig_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.codes:
        codes = [c.strip() for c in args.codes.split(",") if c.strip()]
        all_held = {h.code: h for h in fetch_held_stocks()}
        held = [
            all_held.get(code) or HeldStock(
                code=code, name="?", quantity=0, cost_price=0, current_price=0,
                strategy=None, status="N/A",
            )
            for code in codes
        ]
    else:
        held = fetch_held_stocks()

    if not held:
        print("⚠️ 没有 OPEN 状态的 A 股持仓。")
        return

    print(f"=== 每日信号扫描 ===")
    print(f"持仓股票: {len(held)} 只")
    for s in held:
        print(f"  {s.code} {s.name}")

    if args.as_of:
        as_of = pd.Timestamp(args.as_of)
    else:
        for s in held:
            tmp = load_day(s.code, datetime.now())
            if not tmp.empty:
                as_of = tmp["date"].max()
                break
        else:
            print("⚠️ 所有股票都没有最近的日线数据。")
            return

    print(f"\n目标日期 (as-of): {as_of.date()}")

    per_stock: list[dict] = []
    for stock in held:
        sig_csv = sig_dir / f"cnstock_{stock.code}_signals.csv"
        if not sig_csv.exists():
            print(f"\n--- {stock.code} {stock.name}: 跳过（无 signal CSV at {sig_csv}）---")
            continue

        df_day = load_day(stock.code, as_of.to_pydatetime())
        if df_day.empty or as_of not in df_day["date"].values:
            print(f"\n--- {stock.code} {stock.name}: 跳过（无 {as_of.date()} 当日数据）---")
            continue

        df = build_regime_columns(df_day)
        signal_lib = pd.read_csv(sig_csv)
        states, triggered = detect_triggers(df, signal_lib, as_of)
        ctx = state_context(df, as_of)

        per_stock.append({
            "stock":        stock,
            "states":       states,
            "triggered":    triggered,
            "ctx":          ctx,
            "n_lib":        len(signal_lib),
        })

        n_trig = len(triggered)
        marker = "🚨" if n_trig else "  "
        print(f"\n--- {stock.code} {stock.name} ---  {marker}今日触发 {n_trig}")
        for s in triggered:
            print(f"   {TIER_EMOJI.get(s.tier, '')} {s.tier} {s.signal_id} ({s.direction} h={s.h} size={s.size_pct}%)")

    if not per_stock:
        print("\n⚠️ 没有可扫描的股票（缺少 signal CSV 或当日无数据）。")
        return

    n_total_trig = sum(len(ps["triggered"]) for ps in per_stock)
    n_long  = sum(1 for ps in per_stock for s in ps["triggered"] if s.direction == "long")
    n_short = sum(1 for ps in per_stock for s in ps["triggered"] if s.direction == "short")

    md_parts = [
        f"# {as_of.date()} A 股持仓每日信号扫描",
        f"",
        f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**目标日期**: {as_of.date()} (盘后)",
        f"**扫描股票数**: {len(per_stock)}",
        f"**今日触发总数**: {n_total_trig}（{n_long} long / {n_short} short）",
        f"",
        f"## 摘要",
        f"",
        render_summary_table(per_stock),
        f"",
        f"---",
        f"",
    ]

    for ps in per_stock:
        md_parts.append(render_stock_section(
            ps["stock"], ps["states"], ps["triggered"], ps["ctx"], ps["n_lib"],
        ))
        md_parts.append("---")
        md_parts.append("")

    md_parts += [
        f"## 使用说明",
        f"",
        f"- 触发判定 = 今日 regime ≠ 昨日 regime（事件首日），命中预 computed 信号库",
        f"- 仓位百分比是研究 tier 推荐：high 90% / medium 70% / low 50% / trial 25%",
        f"- 持有期 h 个交易日后退出（无止损时）",
        f"- ⚠️ 信号库基于历史样本，无法预知未来；trial tier 建议半仓试水",
        f"",
        f"## 重新生成信号库",
        f"",
        f"```bash",
        f"uv run python scripts/cnstock_multi_stock_meta_label.py",
        f"```",
        f"",
    ]

    md = "\n".join(md_parts)
    out_path = out_dir / f"{as_of.strftime('%Y%m%d')}.md"
    out_path.write_text(md, encoding="utf-8")
    print(f"\n📝 报告写入 {out_path}")
    print(f"   {n_total_trig} 个触发（{n_long} long / {n_short} short）")


if __name__ == "__main__":
    main()
