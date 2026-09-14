# Rule backtest — stage2_4h_signal_v2.yaml

- Generated: `2026-09-14T12:07:30`
- Symbol: `BTC/USDT`; gate = UTC daily close > SMA200
- Rules: `gate_only`; `gate_reduce:signal=dist_sma30,thr=0,level=0,side=below`; `gate_reduce:signal=dist_sma40,thr=0,level=0,side=below`; `gate_reduce:signal=dist_sma50,thr=0,level=0,side=below`; `gate_reduce:signal=dist_sma60,thr=0,level=0,side=below`; `gate_reduce:signal=dist_sma80,thr=0,level=0,side=below`; `gate_reduce:signal=dist_sma100,thr=0,level=0,side=below`
- Env: commission=0.001, min_hold=3, stop_atr_mult=0; Sharpe annualised by 2190
- Acceptance vs gate_only — improve: MDD ≤ 0.67×, return ≥ 80%×, Calmar >, trades/yr ≤ 30; noharm (test): MDD not deeper, return ≥ 80%×, trades/yr ≤ 30

## pre_val — 2021-08-01 00:00:00+08:00 → 2024-09-30 20:00:00+08:00

| Strategy | Total | Ann. | Max DD | Calmar | Sharpe | Trades | Trades/yr | Accept (improve) |
|---|---:|---:|---:|---:|---:|---:|---:|:--|
| buy_and_hold | +53.73% | +14.53% | -77.04% | 0.19 | 0.25 | 1 | 0.3 | — |
| gate_only | +70.44% | +18.32% | -37.64% | 0.49 | 0.45 | 28 | 8.8 | — |
| gate_reduce:signal=dist_sma30,thr=0,level=0,side=below | +62.86% | +16.64% | -33.43% | 0.50 | 0.53 | 66 | 20.8 | FAIL(mdd) |
| gate_reduce:signal=dist_sma40,thr=0,level=0,side=below | +169.14% | +36.67% | -19.14% | 1.92 | 1.04 | 44 | 13.9 | PASS |
| gate_reduce:signal=dist_sma50,thr=0,level=0,side=below | +110.89% | +26.55% | -25.58% | 1.04 | 0.76 | 40 | 12.6 | FAIL(mdd) |
| gate_reduce:signal=dist_sma60,thr=0,level=0,side=below | +61.15% | +16.25% | -34.85% | 0.47 | 0.48 | 46 | 14.5 | FAIL(mdd,calmar) |
| gate_reduce:signal=dist_sma80,thr=0,level=0,side=below | +54.54% | +14.72% | -34.50% | 0.43 | 0.41 | 44 | 13.9 | FAIL(return,mdd,calmar) |
| gate_reduce:signal=dist_sma100,thr=0,level=0,side=below | +49.80% | +13.60% | -36.52% | 0.37 | 0.37 | 42 | 13.3 | FAIL(return,mdd,calmar) |

## val — 2024-10-01 00:00:00+08:00 → 2026-04-12 20:00:00+08:00

| Strategy | Total | Ann. | Max DD | Calmar | Sharpe | Trades | Trades/yr | Accept (improve) |
|---|---:|---:|---:|---:|---:|---:|---:|:--|
| buy_and_hold | +11.55% | +7.40% | -49.84% | 0.15 | 0.16 | 1 | 0.7 | — |
| gate_only | +17.60% | +11.17% | -33.95% | 0.33 | 0.30 | 19 | 12.4 | — |
| gate_reduce:signal=dist_sma30,thr=0,level=0,side=below | +19.83% | +12.54% | -25.96% | 0.48 | 0.47 | 39 | 25.5 | FAIL(mdd) |
| gate_reduce:signal=dist_sma40,thr=0,level=0,side=below | +34.48% | +21.35% | -21.74% | 0.98 | 0.74 | 31 | 20.2 | PASS |
| gate_reduce:signal=dist_sma50,thr=0,level=0,side=below | +40.04% | +24.60% | -19.01% | 1.29 | 0.81 | 25 | 16.3 | PASS |
| gate_reduce:signal=dist_sma60,thr=0,level=0,side=below | +40.43% | +24.83% | -18.17% | 1.37 | 0.80 | 21 | 13.7 | PASS |
| gate_reduce:signal=dist_sma80,thr=0,level=0,side=below | +52.59% | +31.79% | -19.37% | 1.64 | 0.96 | 15 | 9.8 | PASS |
| gate_reduce:signal=dist_sma100,thr=0,level=0,side=below | +48.57% | +29.51% | -17.85% | 1.65 | 0.87 | 21 | 13.7 | PASS |

## test — 2026-04-13 00:00:00+08:00 → 2026-09-03 20:00:00+08:00

| Strategy | Total | Ann. | Max DD | Calmar | Sharpe | Trades | Trades/yr | Accept (noharm) |
|---|---:|---:|---:|---:|---:|---:|---:|:--|
| buy_and_hold | +14.32% | +40.46% | -29.25% | 1.38 | 0.90 | 1 | 2.5 | — |
| gate_only | +17.21% | +49.62% | -4.55% | 10.92 | 2.93 | 0 | 0.0 | — |
| gate_reduce:signal=dist_sma30,thr=0,level=0,side=below | +17.21% | +49.62% | -4.55% | 10.92 | 2.93 | 0 | 0.0 | PASS |
| gate_reduce:signal=dist_sma40,thr=0,level=0,side=below | +17.21% | +49.62% | -4.55% | 10.92 | 2.93 | 0 | 0.0 | PASS |
| gate_reduce:signal=dist_sma50,thr=0,level=0,side=below | +17.21% | +49.62% | -4.55% | 10.92 | 2.93 | 0 | 0.0 | PASS |
| gate_reduce:signal=dist_sma60,thr=0,level=0,side=below | +17.21% | +49.62% | -4.55% | 10.92 | 2.93 | 0 | 0.0 | PASS |
| gate_reduce:signal=dist_sma80,thr=0,level=0,side=below | +17.21% | +49.62% | -4.55% | 10.92 | 2.93 | 0 | 0.0 | PASS |
| gate_reduce:signal=dist_sma100,thr=0,level=0,side=below | +17.21% | +49.62% | -4.55% | 10.92 | 2.93 | 0 | 0.0 | PASS |
