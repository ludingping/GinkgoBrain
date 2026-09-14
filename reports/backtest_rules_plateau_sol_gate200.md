# Rule backtest — stage2_4h_signal_v2.yaml

- Generated: `2026-09-14T12:16:13`
- Symbol: `SOL/USDT`; gate = UTC daily close > SMA200
- Rules: `gate_only`; `gate_reduce:signal=dist_sma30,thr=0,level=0,side=below`; `gate_reduce:signal=dist_sma50,thr=0,level=0,side=below`; `gate_reduce:signal=dist_sma100,thr=0,level=0,side=below`
- Env: commission=0.001, min_hold=3, stop_atr_mult=0; Sharpe annualised by 2190
- Acceptance vs gate_only — improve: MDD ≤ 0.67×, return ≥ 80%×, Calmar >, trades/yr ≤ 30; noharm (test): MDD not deeper, return ≥ 80%×, trades/yr ≤ 30

## pre_val — 2021-08-01 00:00:00+08:00 → 2024-09-30 20:00:00+08:00

| Strategy | Total | Ann. | Max DD | Calmar | Sharpe | Trades | Trades/yr | Accept (improve) |
|---|---:|---:|---:|---:|---:|---:|---:|:--|
| buy_and_hold | +347.95% | +60.50% | -96.60% | 0.63 | 0.44 | 1 | 0.3 | — |
| gate_only | +971.55% | +111.34% | -66.25% | 1.68 | 1.00 | 30 | 9.5 | — |
| gate_reduce:signal=dist_sma30,thr=0,level=0,side=below | +776.67% | +98.37% | -64.95% | 1.51 | 1.09 | 72 | 22.7 | FAIL(return,mdd,calmar) |
| gate_reduce:signal=dist_sma50,thr=0,level=0,side=below | +1546.00% | +141.99% | -51.77% | 2.74 | 1.31 | 30 | 9.5 | FAIL(mdd) |
| gate_reduce:signal=dist_sma100,thr=0,level=0,side=below | +814.50% | +101.03% | -64.46% | 1.57 | 1.01 | 48 | 15.1 | FAIL(mdd,calmar) |

## val — 2024-10-01 00:00:00+08:00 → 2026-04-12 20:00:00+08:00

| Strategy | Total | Ann. | Max DD | Calmar | Sharpe | Trades | Trades/yr | Accept (improve) |
|---|---:|---:|---:|---:|---:|---:|---:|:--|
| buy_and_hold | -47.43% | -34.30% | -73.20% | -0.47 | -0.52 | 1 | 0.7 | — |
| gate_only | -0.73% | -0.48% | -45.32% | -0.01 | -0.01 | 11 | 7.2 | — |
| gate_reduce:signal=dist_sma30,thr=0,level=0,side=below | +13.20% | +8.43% | -38.46% | 0.22 | 0.20 | 25 | 16.3 | FAIL(mdd) |
| gate_reduce:signal=dist_sma50,thr=0,level=0,side=below | -0.55% | -0.36% | -40.52% | -0.01 | -0.01 | 21 | 13.7 | FAIL(mdd) |
| gate_reduce:signal=dist_sma100,thr=0,level=0,side=below | +0.17% | +0.11% | -46.00% | 0.00 | 0.00 | 25 | 16.3 | FAIL(mdd) |

## test — 2026-04-13 00:00:00+08:00 → 2026-09-03 20:00:00+08:00

| Strategy | Total | Ann. | Max DD | Calmar | Sharpe | Trades | Trades/yr | Accept (noharm) |
|---|---:|---:|---:|---:|---:|---:|---:|:--|
| buy_and_hold | +28.14% | +87.63% | -37.22% | 2.35 | 1.16 | 1 | 2.5 | — |
| gate_only | +23.06% | +69.30% | -10.32% | 6.72 | 2.41 | 0 | 0.0 | — |
| gate_reduce:signal=dist_sma30,thr=0,level=0,side=below | +23.06% | +69.30% | -10.32% | 6.72 | 2.41 | 0 | 0.0 | PASS |
| gate_reduce:signal=dist_sma50,thr=0,level=0,side=below | +23.06% | +69.30% | -10.32% | 6.72 | 2.41 | 0 | 0.0 | PASS |
| gate_reduce:signal=dist_sma100,thr=0,level=0,side=below | +23.06% | +69.30% | -10.32% | 6.72 | 2.41 | 0 | 0.0 | PASS |
