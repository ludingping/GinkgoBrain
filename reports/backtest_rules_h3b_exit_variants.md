# Rule backtest — stage2_4h_signal_v2.yaml

- Generated: `2026-09-07T09:20:48`
- Rules: `gate_only`; `gate_reduce:signal=dist_sma50,thr=0,level=2,side=below`; `gate_reduce:signal=dist_sma50,thr=0,level=0,side=below`; `gate_reduce:signal=dist_sma20,thr=0,level=2,side=below`; `gate_reduce:signal=dist_sma100,thr=0,level=2,side=below`
- Env: commission=0.001, min_hold=3, stop_atr_mult=0; Sharpe annualised by 2190
- Acceptance vs gate_only — improve: MDD ≤ 0.67×, return ≥ 80%×, Calmar >, trades/yr ≤ 30; noharm (test): MDD not deeper, return ≥ 80%×, trades/yr ≤ 30

## val — 2024-10-01 00:00:00+08:00 → 2026-04-12 20:00:00+08:00

| Strategy | Total | Ann. | Max DD | Calmar | Sharpe | Trades | Trades/yr | Accept (improve) |
|---|---:|---:|---:|---:|---:|---:|---:|:--|
| buy_and_hold | +11.55% | +7.40% | -49.84% | 0.15 | 0.16 | 1 | 0.7 | — |
| gate_only | +17.60% | +11.17% | -33.95% | 0.33 | 0.30 | 19 | 12.4 | — |
| gate_reduce:signal=dist_sma50,thr=0,level=2,side=below | +29.15% | +18.18% | -26.03% | 0.70 | 0.57 | 41 | 26.8 | FAIL(mdd) |
| gate_reduce:signal=dist_sma50,thr=0,level=0,side=below | +40.04% | +24.60% | -19.01% | 1.29 | 0.81 | 25 | 16.3 | PASS |
| gate_reduce:signal=dist_sma20,thr=0,level=2,side=below | +16.42% | +10.44% | -28.28% | 0.37 | 0.35 | 54 | 35.3 | FAIL(turnover,mdd) |
| gate_reduce:signal=dist_sma100,thr=0,level=2,side=below | +32.85% | +20.38% | -25.89% | 0.79 | 0.60 | 37 | 24.2 | FAIL(mdd) |

## test — 2026-04-13 00:00:00+08:00 → 2026-09-03 20:00:00+08:00

| Strategy | Total | Ann. | Max DD | Calmar | Sharpe | Trades | Trades/yr | Accept (noharm) |
|---|---:|---:|---:|---:|---:|---:|---:|:--|
| buy_and_hold | +14.32% | +40.46% | -29.25% | 1.38 | 0.90 | 1 | 2.5 | — |
| gate_only | +17.21% | +49.62% | -4.55% | 10.92 | 2.93 | 0 | 0.0 | — |
| gate_reduce:signal=dist_sma50,thr=0,level=2,side=below | +17.21% | +49.62% | -4.55% | 10.92 | 2.93 | 0 | 0.0 | PASS |
| gate_reduce:signal=dist_sma50,thr=0,level=0,side=below | +17.21% | +49.62% | -4.55% | 10.92 | 2.93 | 0 | 0.0 | PASS |
| gate_reduce:signal=dist_sma20,thr=0,level=2,side=below | +17.21% | +49.62% | -4.55% | 10.92 | 2.93 | 0 | 0.0 | PASS |
| gate_reduce:signal=dist_sma100,thr=0,level=2,side=below | +17.21% | +49.62% | -4.55% | 10.92 | 2.93 | 0 | 0.0 | PASS |
