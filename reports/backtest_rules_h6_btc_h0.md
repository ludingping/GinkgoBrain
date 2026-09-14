# Rule backtest — stage2_4h_signal_v2.yaml

- Generated: `2026-09-14T12:30:10`
- Symbol: `BTC/USDT`; gate = UTC daily close > SMA200
- Rules: `gate_only`; `gate_reduce:signal=dist_sma50,thr=0,level=0,side=below`; `gate_vol:target=0.3,window=10,hyst=0,floor=1`; `gate_vol:target=0.3,window=20,hyst=0,floor=1`; `gate_vol:target=0.3,window=30,hyst=0,floor=1`; `gate_vol:target=0.4,window=10,hyst=0,floor=1`; `gate_vol:target=0.4,window=20,hyst=0,floor=1`; `gate_vol:target=0.4,window=30,hyst=0,floor=1`; `gate_vol:target=0.5,window=10,hyst=0,floor=1`; `gate_vol:target=0.5,window=20,hyst=0,floor=1`; `gate_vol:target=0.5,window=30,hyst=0,floor=1`
- Env: commission=0.001, min_hold=3, stop_atr_mult=0; Sharpe annualised by 2190
- Acceptance vs gate_only — improve: MDD ≤ 0.67×, return ≥ 80%×, Calmar >, trades/yr ≤ 30; noharm (test): MDD not deeper, return ≥ 80%×, trades/yr ≤ 30

## pre_val — 2021-08-01 00:00:00+08:00 → 2024-09-30 20:00:00+08:00

| Strategy | Total | Ann. | Max DD | Calmar | Sharpe | Trades | Trades/yr | Accept (improve) |
|---|---:|---:|---:|---:|---:|---:|---:|:--|
| buy_and_hold | +53.73% | +14.53% | -77.04% | 0.19 | 0.25 | 1 | 0.3 | — |
| gate_only | +70.44% | +18.32% | -37.64% | 0.49 | 0.45 | 28 | 8.8 | — |
| gate_reduce:signal=dist_sma50,thr=0,level=0,side=below | +110.89% | +26.55% | -25.58% | 1.04 | 0.76 | 40 | 12.6 | FAIL(mdd) |
| gate_vol:target=0.3,window=10,hyst=0,floor=1 | +37.24% | +10.50% | -28.45% | 0.37 | 0.42 | 126 | 39.8 | FAIL(return,turnover,mdd,calmar) |
| gate_vol:target=0.3,window=20,hyst=0,floor=1 | +37.49% | +10.57% | -24.09% | 0.44 | 0.44 | 84 | 26.5 | FAIL(return,calmar) |
| gate_vol:target=0.3,window=30,hyst=0,floor=1 | +55.95% | +15.05% | -20.14% | 0.75 | 0.62 | 71 | 22.4 | FAIL(return) |
| gate_vol:target=0.4,window=10,hyst=0,floor=1 | +50.42% | +13.75% | -27.01% | 0.51 | 0.45 | 116 | 36.6 | FAIL(return,turnover,mdd) |
| gate_vol:target=0.4,window=20,hyst=0,floor=1 | +50.95% | +13.87% | -25.92% | 0.54 | 0.46 | 76 | 24.0 | FAIL(return,mdd) |
| gate_vol:target=0.4,window=30,hyst=0,floor=1 | +76.37% | +19.60% | -26.09% | 0.75 | 0.63 | 63 | 19.9 | FAIL(mdd) |
| gate_vol:target=0.5,window=10,hyst=0,floor=1 | +44.52% | +12.32% | -33.12% | 0.37 | 0.36 | 99 | 31.2 | FAIL(return,turnover,mdd,calmar) |
| gate_vol:target=0.5,window=20,hyst=0,floor=1 | +69.19% | +18.05% | -32.26% | 0.56 | 0.50 | 58 | 18.3 | FAIL(mdd) |
| gate_vol:target=0.5,window=30,hyst=0,floor=1 | +66.01% | +17.34% | -34.39% | 0.50 | 0.49 | 47 | 14.8 | FAIL(mdd) |

## val — 2024-10-01 00:00:00+08:00 → 2026-04-12 20:00:00+08:00

| Strategy | Total | Ann. | Max DD | Calmar | Sharpe | Trades | Trades/yr | Accept (improve) |
|---|---:|---:|---:|---:|---:|---:|---:|:--|
| buy_and_hold | +11.55% | +7.40% | -49.84% | 0.15 | 0.16 | 1 | 0.7 | — |
| gate_only | +17.60% | +11.17% | -33.95% | 0.33 | 0.30 | 19 | 12.4 | — |
| gate_reduce:signal=dist_sma50,thr=0,level=0,side=below | +40.04% | +24.60% | -19.01% | 1.29 | 0.81 | 25 | 16.3 | PASS |
| gate_vol:target=0.3,window=10,hyst=0,floor=1 | +16.96% | +10.77% | -23.68% | 0.45 | 0.40 | 73 | 47.7 | FAIL(turnover,mdd) |
| gate_vol:target=0.3,window=20,hyst=0,floor=1 | +20.10% | +12.71% | -22.92% | 0.55 | 0.45 | 56 | 36.6 | FAIL(turnover,mdd) |
| gate_vol:target=0.3,window=30,hyst=0,floor=1 | +16.69% | +10.61% | -21.86% | 0.49 | 0.39 | 45 | 29.4 | PASS |
| gate_vol:target=0.4,window=10,hyst=0,floor=1 | +14.93% | +9.51% | -28.47% | 0.33 | 0.30 | 45 | 29.4 | FAIL(mdd) |
| gate_vol:target=0.4,window=20,hyst=0,floor=1 | +23.70% | +14.90% | -25.79% | 0.58 | 0.45 | 39 | 25.5 | FAIL(mdd) |
| gate_vol:target=0.4,window=30,hyst=0,floor=1 | +20.27% | +12.81% | -26.41% | 0.48 | 0.38 | 37 | 24.2 | FAIL(mdd) |
| gate_vol:target=0.5,window=10,hyst=0,floor=1 | +23.52% | +14.79% | -28.00% | 0.53 | 0.42 | 29 | 18.9 | FAIL(mdd) |
| gate_vol:target=0.5,window=20,hyst=0,floor=1 | +19.22% | +12.17% | -31.05% | 0.39 | 0.34 | 24 | 15.7 | FAIL(mdd) |
| gate_vol:target=0.5,window=30,hyst=0,floor=1 | +19.25% | +12.19% | -31.00% | 0.39 | 0.34 | 28 | 18.3 | FAIL(mdd) |

## test — 2026-04-13 00:00:00+08:00 → 2026-09-03 20:00:00+08:00

| Strategy | Total | Ann. | Max DD | Calmar | Sharpe | Trades | Trades/yr | Accept (noharm) |
|---|---:|---:|---:|---:|---:|---:|---:|:--|
| buy_and_hold | +14.32% | +40.46% | -29.25% | 1.38 | 0.90 | 1 | 2.5 | — |
| gate_only | +17.21% | +49.62% | -4.55% | 10.92 | 2.93 | 0 | 0.0 | — |
| gate_reduce:signal=dist_sma50,thr=0,level=0,side=below | +17.21% | +49.62% | -4.55% | 10.92 | 2.93 | 0 | 0.0 | PASS |
| gate_vol:target=0.3,window=10,hyst=0,floor=1 | +11.59% | +32.10% | -3.45% | 9.31 | 2.84 | 2 | 5.1 | FAIL(return) |
| gate_vol:target=0.3,window=20,hyst=0,floor=1 | +11.05% | +30.47% | -3.34% | 9.13 | 2.93 | 5 | 12.7 | FAIL(return) |
| gate_vol:target=0.3,window=30,hyst=0,floor=1 | +14.19% | +40.02% | -3.49% | 11.46 | 3.06 | 1 | 2.5 | PASS |
| gate_vol:target=0.4,window=10,hyst=0,floor=1 | +13.55% | +38.06% | -4.17% | 9.13 | 2.83 | 3 | 7.6 | FAIL(return) |
| gate_vol:target=0.4,window=20,hyst=0,floor=1 | +16.10% | +46.05% | -3.43% | 13.41 | 3.28 | 1 | 2.5 | PASS |
| gate_vol:target=0.4,window=30,hyst=0,floor=1 | +17.21% | +49.62% | -4.55% | 10.92 | 2.93 | 0 | 0.0 | PASS |
| gate_vol:target=0.5,window=10,hyst=0,floor=1 | +17.19% | +49.56% | -3.90% | 12.72 | 3.18 | 2 | 5.1 | PASS |
| gate_vol:target=0.5,window=20,hyst=0,floor=1 | +17.21% | +49.62% | -4.55% | 10.92 | 2.93 | 0 | 0.0 | PASS |
| gate_vol:target=0.5,window=30,hyst=0,floor=1 | +17.21% | +49.62% | -4.55% | 10.92 | 2.93 | 0 | 0.0 | PASS |
