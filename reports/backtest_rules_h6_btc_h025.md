# Rule backtest — stage2_4h_signal_v2.yaml

- Generated: `2026-09-14T12:32:49`
- Symbol: `BTC/USDT`; gate = UTC daily close > SMA200
- Rules: `gate_only`; `gate_vol:target=0.3,window=10,hyst=0.25,floor=1`; `gate_vol:target=0.3,window=20,hyst=0.25,floor=1`; `gate_vol:target=0.3,window=30,hyst=0.25,floor=1`; `gate_vol:target=0.4,window=10,hyst=0.25,floor=1`; `gate_vol:target=0.4,window=20,hyst=0.25,floor=1`; `gate_vol:target=0.4,window=30,hyst=0.25,floor=1`; `gate_vol:target=0.5,window=10,hyst=0.25,floor=1`; `gate_vol:target=0.5,window=20,hyst=0.25,floor=1`; `gate_vol:target=0.5,window=30,hyst=0.25,floor=1`
- Env: commission=0.001, min_hold=3, stop_atr_mult=0; Sharpe annualised by 2190
- Acceptance vs gate_only — improve: MDD ≤ 0.67×, return ≥ 80%×, Calmar >, trades/yr ≤ 30; noharm (test): MDD not deeper, return ≥ 80%×, trades/yr ≤ 30

## pre_val — 2021-08-01 00:00:00+08:00 → 2024-09-30 20:00:00+08:00

| Strategy | Total | Ann. | Max DD | Calmar | Sharpe | Trades | Trades/yr | Accept (improve) |
|---|---:|---:|---:|---:|---:|---:|---:|:--|
| buy_and_hold | +53.73% | +14.53% | -77.04% | 0.19 | 0.25 | 1 | 0.3 | — |
| gate_only | +70.44% | +18.32% | -37.64% | 0.49 | 0.45 | 28 | 8.8 | — |
| gate_vol:target=0.3,window=10,hyst=0.25,floor=1 | +53.93% | +14.58% | -24.30% | 0.60 | 0.58 | 83 | 26.2 | FAIL(return) |
| gate_vol:target=0.3,window=20,hyst=0.25,floor=1 | +45.90% | +12.66% | -25.27% | 0.50 | 0.51 | 52 | 16.4 | FAIL(return,mdd) |
| gate_vol:target=0.3,window=30,hyst=0.25,floor=1 | +49.18% | +13.45% | -21.33% | 0.63 | 0.54 | 43 | 13.6 | FAIL(return) |
| gate_vol:target=0.4,window=10,hyst=0.25,floor=1 | +54.68% | +14.75% | -27.19% | 0.54 | 0.47 | 76 | 24.0 | FAIL(return,mdd) |
| gate_vol:target=0.4,window=20,hyst=0.25,floor=1 | +66.83% | +17.53% | -25.44% | 0.69 | 0.57 | 55 | 17.4 | FAIL(mdd) |
| gate_vol:target=0.4,window=30,hyst=0.25,floor=1 | +79.82% | +20.34% | -27.57% | 0.74 | 0.64 | 47 | 14.8 | FAIL(mdd) |
| gate_vol:target=0.5,window=10,hyst=0.25,floor=1 | +58.68% | +15.68% | -33.24% | 0.47 | 0.44 | 66 | 20.8 | FAIL(mdd,calmar) |
| gate_vol:target=0.5,window=20,hyst=0.25,floor=1 | +61.16% | +16.25% | -33.92% | 0.48 | 0.45 | 40 | 12.6 | FAIL(mdd,calmar) |
| gate_vol:target=0.5,window=30,hyst=0.25,floor=1 | +58.83% | +15.72% | -34.34% | 0.46 | 0.44 | 35 | 11.0 | FAIL(mdd,calmar) |

## val — 2024-10-01 00:00:00+08:00 → 2026-04-12 20:00:00+08:00

| Strategy | Total | Ann. | Max DD | Calmar | Sharpe | Trades | Trades/yr | Accept (improve) |
|---|---:|---:|---:|---:|---:|---:|---:|:--|
| buy_and_hold | +11.55% | +7.40% | -49.84% | 0.15 | 0.16 | 1 | 0.7 | — |
| gate_only | +17.60% | +11.17% | -33.95% | 0.33 | 0.30 | 19 | 12.4 | — |
| gate_vol:target=0.3,window=10,hyst=0.25,floor=1 | +21.36% | +13.48% | -22.74% | 0.59 | 0.47 | 51 | 33.3 | FAIL(turnover,mdd) |
| gate_vol:target=0.3,window=20,hyst=0.25,floor=1 | +18.03% | +11.43% | -23.43% | 0.49 | 0.42 | 31 | 20.2 | FAIL(mdd) |
| gate_vol:target=0.3,window=30,hyst=0.25,floor=1 | +7.31% | +4.72% | -26.53% | 0.18 | 0.17 | 29 | 18.9 | FAIL(return,mdd,calmar) |
| gate_vol:target=0.4,window=10,hyst=0.25,floor=1 | +12.13% | +7.76% | -27.95% | 0.28 | 0.24 | 32 | 20.9 | FAIL(return,mdd,calmar) |
| gate_vol:target=0.4,window=20,hyst=0.25,floor=1 | +23.15% | +14.57% | -27.59% | 0.53 | 0.42 | 27 | 17.6 | FAIL(mdd) |
| gate_vol:target=0.4,window=30,hyst=0.25,floor=1 | +21.18% | +13.37% | -27.20% | 0.49 | 0.38 | 23 | 15.0 | FAIL(mdd) |
| gate_vol:target=0.5,window=10,hyst=0.25,floor=1 | +19.32% | +12.23% | -28.59% | 0.43 | 0.35 | 23 | 15.0 | FAIL(mdd) |
| gate_vol:target=0.5,window=20,hyst=0.25,floor=1 | +19.55% | +12.37% | -30.79% | 0.40 | 0.35 | 23 | 15.0 | FAIL(mdd) |
| gate_vol:target=0.5,window=30,hyst=0.25,floor=1 | +19.16% | +12.13% | -32.15% | 0.38 | 0.33 | 21 | 13.7 | FAIL(mdd) |

## test — 2026-04-13 00:00:00+08:00 → 2026-09-03 20:00:00+08:00

| Strategy | Total | Ann. | Max DD | Calmar | Sharpe | Trades | Trades/yr | Accept (noharm) |
|---|---:|---:|---:|---:|---:|---:|---:|:--|
| buy_and_hold | +14.32% | +40.46% | -29.25% | 1.38 | 0.90 | 1 | 2.5 | — |
| gate_only | +17.21% | +49.62% | -4.55% | 10.92 | 2.93 | 0 | 0.0 | — |
| gate_vol:target=0.3,window=10,hyst=0.25,floor=1 | +13.60% | +38.22% | -3.45% | 11.07 | 3.13 | 2 | 5.1 | FAIL(return) |
| gate_vol:target=0.3,window=20,hyst=0.25,floor=1 | +14.19% | +40.02% | -3.49% | 11.46 | 3.06 | 1 | 2.5 | PASS |
| gate_vol:target=0.3,window=30,hyst=0.25,floor=1 | +14.19% | +40.02% | -3.49% | 11.46 | 3.06 | 1 | 2.5 | PASS |
| gate_vol:target=0.4,window=10,hyst=0.25,floor=1 | +15.08% | +42.84% | -4.03% | 10.62 | 2.98 | 2 | 5.1 | PASS |
| gate_vol:target=0.4,window=20,hyst=0.25,floor=1 | +15.92% | +45.48% | -4.25% | 10.69 | 2.95 | 1 | 2.5 | PASS |
| gate_vol:target=0.4,window=30,hyst=0.25,floor=1 | +17.21% | +49.62% | -4.55% | 10.92 | 2.93 | 0 | 0.0 | PASS |
| gate_vol:target=0.5,window=10,hyst=0.25,floor=1 | +16.92% | +48.68% | -3.74% | 13.01 | 3.13 | 2 | 5.1 | PASS |
| gate_vol:target=0.5,window=20,hyst=0.25,floor=1 | +17.21% | +49.62% | -4.55% | 10.92 | 2.93 | 0 | 0.0 | PASS |
| gate_vol:target=0.5,window=30,hyst=0.25,floor=1 | +17.21% | +49.62% | -4.55% | 10.92 | 2.93 | 0 | 0.0 | PASS |
