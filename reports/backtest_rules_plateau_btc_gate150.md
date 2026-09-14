# Rule backtest — stage2_4h_signal_v2.yaml

- Generated: `2026-09-14T12:10:03`
- Symbol: `BTC/USDT`; gate = UTC daily close > SMA150
- Rules: `gate_only`; `gate_reduce:signal=dist_sma50,thr=0,level=0,side=below`
- Env: commission=0.001, min_hold=3, stop_atr_mult=0; Sharpe annualised by 2190
- Acceptance vs gate_only — improve: MDD ≤ 0.67×, return ≥ 80%×, Calmar >, trades/yr ≤ 30; noharm (test): MDD not deeper, return ≥ 80%×, trades/yr ≤ 30

## pre_val — 2021-08-01 00:00:00+08:00 → 2024-09-30 20:00:00+08:00

| Strategy | Total | Ann. | Max DD | Calmar | Sharpe | Trades | Trades/yr | Accept (improve) |
|---|---:|---:|---:|---:|---:|---:|---:|:--|
| buy_and_hold | +53.73% | +14.53% | -77.04% | 0.19 | 0.25 | 1 | 0.3 | — |
| gate_only | +109.18% | +26.22% | -37.39% | 0.70 | 0.63 | 30 | 9.5 | — |
| gate_reduce:signal=dist_sma50,thr=0,level=0,side=below | +132.78% | +30.55% | -24.18% | 1.26 | 0.85 | 46 | 14.5 | PASS |

## val — 2024-10-01 00:00:00+08:00 → 2026-04-12 20:00:00+08:00

| Strategy | Total | Ann. | Max DD | Calmar | Sharpe | Trades | Trades/yr | Accept (improve) |
|---|---:|---:|---:|---:|---:|---:|---:|:--|
| buy_and_hold | +11.55% | +7.40% | -49.84% | 0.15 | 0.16 | 1 | 0.7 | — |
| gate_only | +27.82% | +17.39% | -27.08% | 0.64 | 0.51 | 15 | 9.8 | — |
| gate_reduce:signal=dist_sma50,thr=0,level=0,side=below | +39.69% | +24.40% | -19.01% | 1.28 | 0.79 | 25 | 16.3 | FAIL(mdd) |

## test — 2026-04-13 00:00:00+08:00 → 2026-09-03 20:00:00+08:00

| Strategy | Total | Ann. | Max DD | Calmar | Sharpe | Trades | Trades/yr | Accept (noharm) |
|---|---:|---:|---:|---:|---:|---:|---:|:--|
| buy_and_hold | +14.32% | +40.46% | -29.25% | 1.38 | 0.90 | 1 | 2.5 | — |
| gate_only | +9.41% | +25.65% | -11.17% | 2.30 | 1.24 | 6 | 15.2 | — |
| gate_reduce:signal=dist_sma50,thr=0,level=0,side=below | +9.41% | +25.65% | -11.17% | 2.30 | 1.24 | 6 | 15.2 | PASS |
