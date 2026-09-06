# Rule backtest — stage2_4h_signal_v2.yaml

- Generated: `2026-09-06T19:14:18`
- Rules: `gate_only`; `gate_reduce:signal=oi_level_90d_z,thr=0.667,level=2,side=above`; `gate_reduce:signal=oi_level_90d_z,thr=0.5,level=2,side=above`; `gate_reduce:signal=oi_level_90d_z,thr=0.667,level=0,side=above`
- Env: commission=0.001, min_hold=3, stop_atr_mult=0; Sharpe annualised by 2190
- Acceptance vs gate_only: MDD ≤ 0.67×, return ≥ 80%×, Calmar >, trades/yr ≤ 30

## val — 2024-10-01 00:00:00+08:00 → 2026-04-12 20:00:00+08:00

| Strategy | Total | Ann. | Max DD | Calmar | Sharpe | Trades | Trades/yr | Accept |
|---|---:|---:|---:|---:|---:|---:|---:|:--|
| buy_and_hold | +11.55% | +7.40% | -49.84% | 0.15 | 0.16 | 1 | 0.7 | — |
| gate_only | +17.60% | +11.17% | -33.95% | 0.33 | 0.30 | 19 | 12.4 | — |
| gate_reduce:signal=oi_level_90d_z,thr=0.667,level=2,side=above | +20.24% | +12.79% | -33.86% | 0.38 | 0.35 | 48 | 31.4 | FAIL(mdd,turnover) |
| gate_reduce:signal=oi_level_90d_z,thr=0.5,level=2,side=above | +26.07% | +16.33% | -33.96% | 0.48 | 0.45 | 66 | 43.1 | FAIL(mdd,turnover) |
| gate_reduce:signal=oi_level_90d_z,thr=0.667,level=0,side=above | +22.80% | +14.36% | -33.78% | 0.43 | 0.39 | 47 | 30.7 | FAIL(mdd,turnover) |

## test — 2026-04-13 00:00:00+08:00 → 2026-09-03 20:00:00+08:00

| Strategy | Total | Ann. | Max DD | Calmar | Sharpe | Trades | Trades/yr | Accept |
|---|---:|---:|---:|---:|---:|---:|---:|:--|
| buy_and_hold | +14.32% | +40.46% | -29.25% | 1.38 | 0.90 | 1 | 2.5 | — |
| gate_only | +17.21% | +49.62% | -4.55% | 10.92 | 2.93 | 0 | 0.0 | — |
| gate_reduce:signal=oi_level_90d_z,thr=0.667,level=2,side=above | +17.21% | +49.62% | -4.55% | 10.92 | 2.93 | 0 | 0.0 | FAIL(mdd,calmar) |
| gate_reduce:signal=oi_level_90d_z,thr=0.5,level=2,side=above | +13.33% | +37.36% | -4.55% | 8.22 | 2.48 | 3 | 7.6 | FAIL(mdd,return,calmar) |
| gate_reduce:signal=oi_level_90d_z,thr=0.667,level=0,side=above | +17.21% | +49.62% | -4.55% | 10.92 | 2.93 | 0 | 0.0 | FAIL(mdd,calmar) |
