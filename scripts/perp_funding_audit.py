"""
BTC 4h PPO Perp 资金费率审计 — 离线 verdict 工具.

设计：docs/GinkgoBrain/BTC4h-Perp资金费率审计-设计.md（v0.2）
测试：docs/GinkgoBrain/BTC4h-Perp资金费率审计-测试用例.md

核心修正（v0.2）：
- funding-then-rebalance 顺序：funding 用进入循环时的 pos（前根 bar 末仓位），不是 tgt
- 原因：Binance funding 严格在 hh:00:00.000 快照；订单到达撮合引擎已是 hh:00:01+
- v0.1 错误用 tgt 会漏算"满仓避费"成本 → 回测虚高 / 实盘 alpha 蒸发

Usage:
    uv run python scripts/perp_funding_audit.py \\
        --model models/saved/BTCUSDT_ppo_4h_signal_v3/best_model.zip \\
        --config config/stage2_4h_signal.yaml \\
        --regime-gate --regime-ma-days 200 \\
        --vol-target 0.01
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from stable_baselines3 import PPO

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from envs.signal_layered_env import SignalLayeredEnv  # noqa: E402
from scripts.backtest_ppo_overlay import (  # noqa: E402
    collect_actions,
    load_data,
    load_signal_list,
    compute_metrics,
)
from utils.db import read_funding  # noqa: E402

BARS_PER_DAY_4H = 6
ACTION_TO_RATIO = {0: 0.0, 1: 0.25, 2: 0.5, 3: 0.75, 4: 1.0}


# ─── Funding loading & alignment ─────────────────────────────────────────────


def load_funding_aligned(
    symbol: str,
    start: str,
    end: str,
    timestamps: pd.Series,
    source: str = "settled",
) -> tuple[np.ndarray, dict]:
    """
    Load funding from DB and align to a 4h bar timestamp series.

    For each timestamp in `timestamps`, look up funding_rate iff hour % 8 == 0.
    Returns:
        rates: np.ndarray of len(timestamps), 0 except at funding boundaries
        stats: dict with coverage/integrity diagnostics
    """
    funding = read_funding(symbol, start=start, end=end, source=source)
    if funding.empty:
        raise SystemExit(
            f"No funding records for {symbol} {source} in [{start}, {end}). "
            f"Check Spider collector / quant_db."
        )

    funding = funding.copy()
    funding["timestamp"] = pd.to_datetime(funding["timestamp"], utc=True)
    funding_lookup = funding.set_index("timestamp")["funding_rate"].to_dict()

    n = len(timestamps)
    rates = np.zeros(n)
    n_boundaries = 0
    n_matched = 0
    for i, ts in enumerate(timestamps):
        ts_ts = pd.Timestamp(ts)
        ts_utc = ts_ts.tz_convert("UTC") if ts_ts.tz is not None else ts_ts.tz_localize("UTC")
        if ts_utc.hour % 8 == 0:
            n_boundaries += 1
            if ts_utc in funding_lookup:
                rates[i] = float(funding_lookup[ts_utc])
                n_matched += 1

    coverage = n_matched / n_boundaries if n_boundaries > 0 else 0.0
    pos_count = int((rates > 0).sum())
    neg_count = int((rates < 0).sum())
    nonzero_rates = rates[rates != 0.0]
    stats = {
        "n_funding_records": len(funding),
        "n_boundaries_in_bars": n_boundaries,
        "n_matched": n_matched,
        "coverage": coverage,
        "pos_count": pos_count,
        "neg_count": neg_count,
        "pos_share": pos_count / max(n_matched, 1),
        "mean_rate": float(nonzero_rates.mean()) if len(nonzero_rates) > 0 else 0.0,
        "max_rate": float(nonzero_rates.max()) if len(nonzero_rates) > 0 else 0.0,
        "min_rate": float(nonzero_rates.min()) if len(nonzero_rates) > 0 else 0.0,
        "annualized_long_cost": float(nonzero_rates.mean()) * 3 * 365 if len(nonzero_rates) > 0 else 0.0,
    }
    return rates, stats


# ─── Simulation ──────────────────────────────────────────────────────────────


def simulate_with_funding(
    closes: np.ndarray,
    raw_ratios: np.ndarray,
    realized_vol: np.ndarray,
    regime_ok: np.ndarray,
    funding_rates: np.ndarray,
    vol_target: float,
    use_regime_gate: bool,
    rebalance_threshold: float,
    commission: float,
    use_funding: bool = True,
    initial_balance: float = 10_000.0,
    initial_pos: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Portfolio simulation with funding-then-rebalance ordering (design §4.1).

    Order per bar i:
        1. Funding cost using PRE-rebalance pos (= initial_pos at i=0,
           else final pos from i-1)
        2. Compute target ratio (vol-target × regime-gate × raw)
        3. Rebalance (commission)
        4. Mark to market using POST-rebalance pos

    Returns (equity, funding_paid, n_rebalances).
    """
    n = len(closes)
    portfolio = float(initial_balance)
    pos = float(initial_pos)
    equity = np.zeros(n)
    funding_paid = np.zeros(n)
    n_rebalances = 0

    for i in range(n):
        # 1. Funding (pre-rebalance) — design §2.4 critical path
        if use_funding and pos != 0.0 and funding_rates[i] != 0.0:
            cost = portfolio * pos * funding_rates[i]
            portfolio -= cost
            funding_paid[i] = cost  # >0 paid (long pays); <0 received

        # 2. Compute target ratio
        vol_scale = (
            min(vol_target / max(float(realized_vol[i]), 1e-4), 1.0)
            if vol_target > 0.0 else 1.0
        )
        regime_scale = 1.0 if (not use_regime_gate or regime_ok[i] == 1) else 0.0
        tgt = float(raw_ratios[i]) * vol_scale * regime_scale

        # 3. Rebalance
        delta = abs(tgt - pos)
        if delta >= rebalance_threshold:
            portfolio -= portfolio * delta * commission
            pos = tgt
            n_rebalances += 1

        # 4. Mark to market (post-rebalance pos)
        ret = float(closes[i + 1] / closes[i] - 1) if i < n - 1 else 0.0
        portfolio *= 1.0 + pos * ret
        equity[i] = portfolio

    return equity, funding_paid, n_rebalances


# ─── Plotting ────────────────────────────────────────────────────────────────


def plot_diagnostics(
    timestamps: pd.Series,
    closes: np.ndarray,
    funding_rates: np.ndarray,
    funding_paid_cum: np.ndarray,
    equity_no_funding: np.ndarray,
    equity_with_funding: np.ndarray,
    output_dir: Path,
    output_stem: str,
) -> list[Path]:
    """Generate 4 PNGs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    # 1. funding rate time series + cumulative
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    nonzero_mask = funding_rates != 0.0
    axes[0].scatter(timestamps[nonzero_mask], funding_rates[nonzero_mask] * 100, s=2, alpha=0.5)
    axes[0].axhline(0, color="gray", lw=0.5)
    axes[0].set_ylabel("Funding rate (%/8h)")
    axes[0].grid(alpha=0.3)
    axes[1].plot(timestamps, funding_paid_cum, color="C3", lw=1.0)
    axes[1].axhline(0, color="gray", lw=0.5)
    axes[1].set_ylabel("Cumulative funding paid (USD)")
    axes[1].set_xlabel("Time")
    axes[1].grid(alpha=0.3)
    fig.suptitle("Funding rate + cumulative cost (long position)")
    fig.tight_layout()
    p = output_dir / f"{output_stem}_funding_timeseries.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(p)

    # 2. equity comparison
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(timestamps, equity_no_funding, label="No funding (baseline)", color="C0", lw=1.0)
    ax.plot(timestamps, equity_with_funding, label="With funding (perp real)", color="C3", lw=1.0)
    btc_norm = closes / closes[0] * equity_no_funding[0]
    ax.plot(timestamps, btc_norm, label="BTC buy-and-hold", color="orange", lw=0.6, alpha=0.6)
    ax.set_ylabel("Portfolio value (USD)")
    ax.set_xlabel("Time")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.suptitle("Equity: with vs without funding cost")
    fig.tight_layout()
    p = output_dir / f"{output_stem}_equity_compare.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(p)

    # 3. monthly funding decomposition
    df = pd.DataFrame({"ts": timestamps.values, "paid": np.diff(np.concatenate([[0], funding_paid_cum]))})
    df["month"] = pd.to_datetime(df["ts"]).dt.to_period("M").astype(str)
    monthly = df.groupby("month")["paid"].sum()
    fig, ax = plt.subplots(figsize=(12, 4))
    colors = ["C3" if v > 0 else "C2" for v in monthly.values]
    ax.bar(monthly.index, monthly.values, color=colors)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("Monthly funding paid (USD)")
    ax.set_xlabel("Month")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Monthly funding cost (red=paid / green=received)")
    fig.tight_layout()
    p = output_dir / f"{output_stem}_monthly_funding.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(p)

    # 4. Drawdown comparison
    def _dd(eq):
        peak = np.maximum.accumulate(eq)
        return (eq - peak) / peak * 100

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(timestamps, _dd(equity_no_funding), 0, alpha=0.4, label="No funding", color="C0")
    ax.fill_between(timestamps, _dd(equity_with_funding), 0, alpha=0.4, label="With funding", color="C3")
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Time")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.suptitle("Drawdown comparison")
    fig.tight_layout()
    p = output_dir / f"{output_stem}_drawdown.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(p)

    return paths


# ─── Recap ───────────────────────────────────────────────────────────────────


def _verdict(net_sharpe: float, net_calmar: float, funding_pct_pnl: float) -> tuple[str, str]:
    """3-tier verdict per design §9."""
    sharpe_status = "✅" if net_sharpe > 0.7 else ("⚠️" if net_sharpe >= 0.3 else "❌")
    calmar_status = "✅" if net_calmar > 0.5 else ("⚠️" if net_calmar >= 0.2 else "❌")
    funding_status = "✅" if funding_pct_pnl < 50 else ("⚠️" if funding_pct_pnl <= 80 else "❌")
    statuses = [sharpe_status, calmar_status, funding_status]
    if all(s == "✅" for s in statuses):
        return "✅ 通过", "进 perp 实盘接入设计"
    if any(s == "❌" for s in statuses):
        return "❌ 不通过", "进 §9.3 fail 分支：(a) 1d 低频 / (b) spot 执行 / (c) 换品种"
    return "⚠️ 边缘", "进 funding-aware overlay 设计"


def _calmar(m: dict) -> float:
    if m["max_drawdown_pct"] >= 0:
        return float("inf")
    return abs(m["total_return_pct"] / m["max_drawdown_pct"])


def write_recap(
    args,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    funding_stats: dict,
    metrics_no_funding: dict,
    metrics_with_funding: dict,
    funding_paid_total: float,
    initial_balance: float,
    n_rebalances: int,
    png_paths: list[Path],
    output_path: Path,
) -> bool:
    """Write recap markdown; returns True if verdict pass (3-tier)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pnl_no_fund_usd = (metrics_no_funding["total_return_pct"] / 100) * initial_balance
    funding_pct_pnl = (
        abs(funding_paid_total) / abs(pnl_no_fund_usd) * 100
        if abs(pnl_no_fund_usd) > 1e-6 else 0.0
    )

    net_calmar = _calmar(metrics_with_funding)
    overall, next_step = _verdict(metrics_with_funding["sharpe"], net_calmar, funding_pct_pnl)
    pass_verdict = overall.startswith("✅")

    summary = (
        f"**Verdict**: {overall} — "
        f"净 Sharpe={metrics_with_funding['sharpe']:.3f}（baseline {metrics_no_funding['sharpe']:.3f}），"
        f"funding 占 |PnL|={funding_pct_pnl:.1f}%，"
        f"累计 funding {funding_paid_total:+.0f} USD。"
    )

    def relpath(p: Path) -> str:
        try:
            return str(p.relative_to(output_path.parent))
        except ValueError:
            return str(p)

    img_md = "\n".join(f"![{p.stem}]({relpath(p)})" for p in png_paths)

    md = f"""# BTC 4h PPO Perp 资金费率审计 — 复盘报告

> **日期**：2026-04-29（脚本运行）
> **设计**：[../BTC4h-Perp资金费率审计-设计.md](../BTC4h-Perp资金费率审计-设计.md)（v0.2）
> **测试**：[../BTC4h-Perp资金费率审计-测试用例.md](../BTC4h-Perp资金费率审计-测试用例.md)

## 1. 执行摘要

{summary}

**下一步**：{next_step}

## 2. 运行参数

| 参数 | 值 |
|---|---|
| 模型 | `{args.model}` |
| 配置 | `{args.config}` |
| OOS 段 | {test_start} ~ {test_end} |
| Funding source | {args.funding_source} |
| Vol target | {args.vol_target} |
| Regime gate | {args.regime_gate} (ma_days={args.regime_ma_days}) |
| Rebalance threshold | {args.rebalance_threshold} |
| Commission | {args.commission*100:.3f}% |
| Initial balance | {initial_balance:.0f} USD |

## 3. Funding 数据完整性

| 指标 | 值 |
|---|---|
| 总 funding records | {funding_stats['n_funding_records']:,} |
| 4h bar 边界数 | {funding_stats['n_boundaries_in_bars']:,} |
| 边界匹配数 | {funding_stats['n_matched']:,} |
| 覆盖率 | {funding_stats['coverage']*100:.2f}% |
| 正向（多头付） | {funding_stats['pos_count']} ({funding_stats['pos_share']*100:.1f}%) |
| 负向（多头收） | {funding_stats['neg_count']} |
| 平均 rate per 8h | {funding_stats['mean_rate']*100:.4f}% |
| 极端最大 | {funding_stats['max_rate']*100:.4f}% |
| 极端最小 | {funding_stats['min_rate']*100:.4f}% |
| 年化等价（持仓 100% 多头） | {funding_stats['annualized_long_cost']*100:.2f}% |

## 4. 业绩对比

| 指标 | No Funding (baseline) | With Funding (perp 真实) | Δ |
|---|---|---|---|
| Sharpe | {metrics_no_funding['sharpe']:.3f} | {metrics_with_funding['sharpe']:.3f} | {metrics_with_funding['sharpe']-metrics_no_funding['sharpe']:+.3f} |
| Total Return | {metrics_no_funding['total_return_pct']:+.2f}% | {metrics_with_funding['total_return_pct']:+.2f}% | {metrics_with_funding['total_return_pct']-metrics_no_funding['total_return_pct']:+.2f}% |
| Max Drawdown | {metrics_no_funding['max_drawdown_pct']:.2f}% | {metrics_with_funding['max_drawdown_pct']:.2f}% | {metrics_with_funding['max_drawdown_pct']-metrics_no_funding['max_drawdown_pct']:+.2f}% |
| Calmar | {_calmar(metrics_no_funding):.3f} | {net_calmar:.3f} | — |
| BTC buy-and-hold | {metrics_no_funding['buy_hold_return_pct']:+.2f}% | — | — |
| Trades (rebalances) | {n_rebalances} | {n_rebalances} | 0 |
| Funding paid total | — | {funding_paid_total:+,.0f} USD | — |
| Funding 占 \\|PnL\\| 比 | — | {funding_pct_pnl:.1f}% | — |

## 5. Done 判定（设计 §9）

| 指标 | 通过 | 边缘 | Fail | 实测 | 结果 |
|---|---|---|---|---|---|
| 净 Sharpe | > 0.7 | [0.3, 0.7] | < 0.3 | {metrics_with_funding['sharpe']:.3f} | {'✅' if metrics_with_funding['sharpe'] > 0.7 else '⚠️' if metrics_with_funding['sharpe'] >= 0.3 else '❌'} |
| 净 Calmar | > 0.5 | [0.2, 0.5] | < 0.2 | {net_calmar:.3f} | {'✅' if net_calmar > 0.5 else '⚠️' if net_calmar >= 0.2 else '❌'} |
| Funding 占 \\|PnL\\| | < 50% | [50%, 80%] | > 80% | {funding_pct_pnl:.1f}% | {'✅' if funding_pct_pnl < 50 else '⚠️' if funding_pct_pnl <= 80 else '❌'} |

**Verdict: {overall}**

## 6. 可视化

{img_md}

## 7. 下一步

{next_step}

具体动作：
- **如通过**：写 `BTC4h-Perp实盘接入-设计.md`（GinkgoBole trade module 集成 / API 鉴权 / 风控熔断 / 小资金 ramp）
- **如边缘**：写 `BTC4h-Funding感知-Overlay-设计.md`（funding>X% 降仓 / settled 前避费 / 改用 spot 执行）
- **如 fail**：选 §9.3 的 a/b/c：(a) 1d 低频 / (b) spot 执行无 funding / (c) 试 ETH-perp
"""
    output_path.write_text(md)
    print(f"\n✓ 复盘报告写入: {output_path}")
    return pass_verdict


# ─── Main ────────────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--test-start", default=None)
    p.add_argument("--test-end", default=None)
    p.add_argument("--vol-target", type=float, default=0.01)
    p.add_argument("--vol-window", type=int, default=168)
    p.add_argument("--regime-gate", action="store_true")
    p.add_argument("--regime-ma-days", type=int, default=200)
    p.add_argument("--regime-resample", default="1D")
    p.add_argument("--rebalance-threshold", type=float, default=0.01)
    p.add_argument("--commission", type=float, default=0.001)
    p.add_argument("--funding-source", default="settled")
    p.add_argument(
        "--output",
        default="docs/GinkgoBrain/复盘/2026-04-30_btc4h_funding_cost_audit.md",
        help="复盘报告路径（相对 GinkgoRoad）",
    )
    args = p.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    timeframe = cfg["crypto"].get("timeframe", "4h").lower()
    symbol = cfg["crypto"]["symbol"]
    if timeframe != "4h":
        print(f"WARNING: 本审计为 4h 模型设计，当前 timeframe={timeframe}")

    signal_cols = load_signal_list(cfg["signals"]["config_path"])
    print(f"Loaded {len(signal_cols)} signals")
    needs_contracts = any(s.startswith(("sig_funding_", "sig_oi_", "sig_liq_")) for s in signal_cols)
    bt_df = load_data(
        cfg, args.test_start, args.test_end,
        vol_window=args.vol_window, regime_ma_days=args.regime_ma_days,
        regime_resample=args.regime_resample,
        with_contracts=needs_contracts,
    )
    print(f"BT df: {len(bt_df):,} rows, {bt_df['timestamp'].min()} → {bt_df['timestamp'].max()}")

    _ALLOWED = {"window_size", "initial_balance", "commission",
                "risk_aversion_coef", "excess_return_coef",
                "stop_atr_mult", "stop_cooldown_steps"}
    env_cfg = {k: v for k, v in cfg["env"].items() if k in _ALLOWED}
    env_cfg["random_start"] = False
    env = SignalLayeredEnv(bt_df, signal_cols, **env_cfg)

    print(f"Loading PPO model from {args.model}")
    model = PPO.load(args.model, device="cpu")
    actions, step_indices = collect_actions(model, env)
    print(f"Collected {len(actions)} actions, steps {step_indices[0]}..{step_indices[-1]}")

    sub = bt_df.iloc[step_indices].reset_index(drop=True)
    closes = sub["close"].to_numpy()
    realized_vol = sub["realized_vol_ann"].to_numpy()
    regime_ok = sub["regime_ok"].to_numpy().astype(int)
    raw_ratios = np.array([ACTION_TO_RATIO[int(a)] for a in actions], dtype=float)

    test_start_ts = sub["timestamp"].iloc[0]
    test_end_ts = sub["timestamp"].iloc[-1]
    print(f"\n=== Loading {args.funding_source} funding for {symbol} ===")
    funding_rates, funding_stats = load_funding_aligned(
        symbol, str(test_start_ts), str(test_end_ts + pd.Timedelta(hours=4)),
        sub["timestamp"], source=args.funding_source,
    )
    print(f"  funding records: {funding_stats['n_funding_records']:,}")
    print(f"  4h bar boundaries: {funding_stats['n_boundaries_in_bars']:,}")
    print(f"  matched: {funding_stats['n_matched']:,} ({funding_stats['coverage']*100:.2f}%)")
    print(f"  正向比例: {funding_stats['pos_share']*100:.1f}%")
    print(f"  平均 rate/8h: {funding_stats['mean_rate']*100:.4f}%")
    print(f"  年化等价（100% 多头）: {funding_stats['annualized_long_cost']*100:.2f}%")

    if funding_stats["coverage"] < 0.95:
        print(f"\n⚠️  funding 边界覆盖率 {funding_stats['coverage']*100:.1f}% < 95%，回测可能不可信")

    sim_kwargs = dict(
        closes=closes, raw_ratios=raw_ratios, realized_vol=realized_vol,
        regime_ok=regime_ok, vol_target=args.vol_target,
        use_regime_gate=args.regime_gate, rebalance_threshold=args.rebalance_threshold,
        commission=args.commission,
    )
    print("\n=== Simulating without funding (baseline) ===")
    eq_no, _, n_rebal = simulate_with_funding(
        funding_rates=np.zeros(len(closes)), use_funding=False, **sim_kwargs,
    )
    print("=== Simulating with funding (perp 真实) ===")
    eq_with, funding_paid, _ = simulate_with_funding(
        funding_rates=funding_rates, use_funding=True, **sim_kwargs,
    )

    metrics_no = compute_metrics(eq_no, closes, timeframe)
    metrics_with = compute_metrics(eq_with, closes, timeframe)
    funding_total = float(funding_paid.sum())

    print("\n=== Metrics ===")
    print(f"  No funding   : Sharpe={metrics_no['sharpe']:.3f}, "
          f"Return={metrics_no['total_return_pct']:+.2f}%, "
          f"DD={metrics_no['max_drawdown_pct']:.2f}%")
    print(f"  With funding : Sharpe={metrics_with['sharpe']:.3f}, "
          f"Return={metrics_with['total_return_pct']:+.2f}%, "
          f"DD={metrics_with['max_drawdown_pct']:.2f}%")
    print(f"  Funding paid total: {funding_total:+,.2f} USD")
    pnl_no_usd = (metrics_no['total_return_pct'] / 100) * env_cfg.get('initial_balance', 10000)
    funding_pct_pnl = abs(funding_total) / abs(pnl_no_usd) * 100 if abs(pnl_no_usd) > 1e-6 else 0
    print(f"  Funding 占 |PnL| 比: {funding_pct_pnl:.1f}%")

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path("/home/davidlyu/projects/GinkgoRoad") / args.output
    output_dir = output_path.parent

    funding_paid_cum = np.cumsum(funding_paid)
    png_paths = plot_diagnostics(
        sub["timestamp"], closes, funding_rates, funding_paid_cum,
        eq_no, eq_with, output_dir, output_path.stem,
    )

    initial_balance = env_cfg.get("initial_balance", 10000.0)
    pass_verdict = write_recap(
        args, test_start_ts, test_end_ts, funding_stats,
        metrics_no, metrics_with, funding_total, initial_balance,
        n_rebal, png_paths, output_path,
    )

    print(f"\n[verdict: {'✓ 通过' if pass_verdict else '✗ 不通过 / 边缘'} done 判定]")
    return 0 if pass_verdict else 1


if __name__ == "__main__":
    sys.exit(main())
