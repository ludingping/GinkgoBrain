# Rule backtest — stage2_4h_signal_v2.yaml

- Generated: `2026-09-06T21:52:12`
- Rules: `gate_only`; `gate_reduce:signal=dist_sma50,thr=0,level=2,side=below`; `gate_reduce:signal=dd20_atr,thr=-2,level=2,side=below`; `gate_reduce:signal=dist_low20,thr=1e-07,level=2,side=below`
- Env: commission=0.001, min_hold=3, stop_atr_mult=0; Sharpe annualised by 2190
- Acceptance vs gate_only: MDD ≤ 0.67×, return ≥ 80%×, Calmar >, trades/yr ≤ 30

## val — 2024-10-01 00:00:00+08:00 → 2026-04-12 20:00:00+08:00

| Strategy | Total | Ann. | Max DD | Calmar | Sharpe | Trades | Trades/yr | Accept |
|---|---:|---:|---:|---:|---:|---:|---:|:--|
| buy_and_hold | +11.55% | +7.40% | -49.84% | 0.15 | 0.16 | 1 | 0.7 | — |
| gate_only | +17.60% | +11.17% | -33.95% | 0.33 | 0.30 | 19 | 12.4 | — |
| gate_reduce:signal=dist_sma50,thr=0,level=2,side=below | +29.15% | +18.18% | -26.03% | 0.70 | 0.57 | 41 | 26.8 | FAIL(mdd) |
| gate_reduce:signal=dd20_atr,thr=-2,level=2,side=below | -3.01% | -1.98% | -40.87% | -0.05 | -0.06 | 63 | 41.1 | FAIL(mdd,return,calmar,turnover) |
| gate_reduce:signal=dist_low20,thr=1e-07,level=2,side=below | +18.07% | +11.46% | -31.33% | 0.37 | 0.31 | 31 | 20.2 | FAIL(mdd) |

## test — 2026-04-13 00:00:00+08:00 → 2026-09-03 20:00:00+08:00

| Strategy | Total | Ann. | Max DD | Calmar | Sharpe | Trades | Trades/yr | Accept |
|---|---:|---:|---:|---:|---:|---:|---:|:--|
| buy_and_hold | +14.32% | +40.46% | -29.25% | 1.38 | 0.90 | 1 | 2.5 | — |
| gate_only | +17.21% | +49.62% | -4.55% | 10.92 | 2.93 | 0 | 0.0 | — |
| gate_reduce:signal=dist_sma50,thr=0,level=2,side=below | +17.21% | +49.62% | -4.55% | 10.92 | 2.93 | 0 | 0.0 | FAIL(mdd,calmar) |
| gate_reduce:signal=dd20_atr,thr=-2,level=2,side=below | +17.21% | +49.62% | -4.55% | 10.92 | 2.93 | 0 | 0.0 | FAIL(mdd,calmar) |
| gate_reduce:signal=dist_low20,thr=1e-07,level=2,side=below | +17.21% | +49.62% | -4.55% | 10.92 | 2.93 | 0 | 0.0 | FAIL(mdd,calmar) |
