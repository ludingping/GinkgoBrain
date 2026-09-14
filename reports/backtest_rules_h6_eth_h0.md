# Rule backtest — stage2_4h_signal_v2.yaml

- Generated: `2026-09-14T12:30:12`
- Symbol: `ETH/USDT`; gate = UTC daily close > SMA200
- Rules: `gate_only`; `gate_vol:target=0.3,window=10,hyst=0,floor=1`; `gate_vol:target=0.3,window=20,hyst=0,floor=1`; `gate_vol:target=0.3,window=30,hyst=0,floor=1`; `gate_vol:target=0.4,window=10,hyst=0,floor=1`; `gate_vol:target=0.4,window=20,hyst=0,floor=1`; `gate_vol:target=0.4,window=30,hyst=0,floor=1`; `gate_vol:target=0.5,window=10,hyst=0,floor=1`; `gate_vol:target=0.5,window=20,hyst=0,floor=1`; `gate_vol:target=0.5,window=30,hyst=0,floor=1`
- Env: commission=0.001, min_hold=3, stop_atr_mult=0; Sharpe annualised by 2190
- Acceptance vs gate_only — improve: MDD ≤ 0.67×, return ≥ 80%×, Calmar >, trades/yr ≤ 30; noharm (test): MDD not deeper, return ≥ 80%×, trades/yr ≤ 30

## pre_val — 2021-08-01 00:00:00+08:00 → 2024-09-30 20:00:00+08:00

| Strategy | Total | Ann. | Max DD | Calmar | Sharpe | Trades | Trades/yr | Accept (improve) |
|---|---:|---:|---:|---:|---:|---:|---:|:--|
| buy_and_hold | +5.57% | +1.72% | -81.12% | 0.02 | 0.03 | 1 | 0.3 | — |
| gate_only | +148.47% | +33.27% | -37.81% | 0.88 | 0.63 | 13 | 4.1 | — |
| gate_vol:target=0.3,window=10,hyst=0,floor=1 | +81.96% | +20.79% | -25.22% | 0.82 | 0.75 | 119 | 37.5 | FAIL(return,turnover,mdd,calmar) |
| gate_vol:target=0.3,window=20,hyst=0,floor=1 | +79.89% | +20.35% | -22.91% | 0.89 | 0.75 | 73 | 23.0 | FAIL(return) |
| gate_vol:target=0.3,window=30,hyst=0,floor=1 | +90.12% | +22.47% | -20.67% | 1.09 | 0.85 | 57 | 18.0 | FAIL(return) |
| gate_vol:target=0.4,window=10,hyst=0,floor=1 | +127.88% | +29.68% | -31.63% | 0.94 | 0.80 | 119 | 37.5 | FAIL(turnover,mdd) |
| gate_vol:target=0.4,window=20,hyst=0,floor=1 | +102.35% | +24.91% | -27.31% | 0.91 | 0.70 | 74 | 23.3 | FAIL(return,mdd) |
| gate_vol:target=0.4,window=30,hyst=0,floor=1 | +113.80% | +27.09% | -22.72% | 1.19 | 0.78 | 52 | 16.4 | FAIL(return) |
| gate_vol:target=0.5,window=10,hyst=0,floor=1 | +104.96% | +25.41% | -34.26% | 0.74 | 0.62 | 94 | 29.7 | FAIL(return,mdd,calmar) |
| gate_vol:target=0.5,window=20,hyst=0,floor=1 | +127.67% | +29.64% | -34.36% | 0.86 | 0.70 | 58 | 18.3 | FAIL(mdd,calmar) |
| gate_vol:target=0.5,window=30,hyst=0,floor=1 | +132.33% | +30.47% | -30.72% | 0.99 | 0.72 | 47 | 14.8 | FAIL(mdd) |

## val — 2024-10-01 00:00:00+08:00 → 2026-04-12 20:00:00+08:00

| Strategy | Total | Ann. | Max DD | Calmar | Sharpe | Trades | Trades/yr | Accept (improve) |
|---|---:|---:|---:|---:|---:|---:|---:|:--|
| buy_and_hold | -15.84% | -10.65% | -65.12% | -0.16 | -0.16 | 1 | 0.7 | — |
| gate_only | +9.76% | +6.27% | -38.27% | 0.16 | 0.14 | 13 | 8.5 | — |
| gate_vol:target=0.3,window=10,hyst=0,floor=1 | +2.16% | +1.41% | -24.48% | 0.06 | 0.07 | 47 | 30.7 | FAIL(return,turnover,calmar) |
| gate_vol:target=0.3,window=20,hyst=0,floor=1 | +4.64% | +3.01% | -24.49% | 0.12 | 0.15 | 29 | 18.9 | FAIL(return,calmar) |
| gate_vol:target=0.3,window=30,hyst=0,floor=1 | +11.66% | +7.47% | -18.99% | 0.39 | 0.35 | 21 | 13.7 | PASS |
| gate_vol:target=0.4,window=10,hyst=0,floor=1 | +2.40% | +1.56% | -29.31% | 0.05 | 0.06 | 55 | 35.9 | FAIL(return,turnover,mdd,calmar) |
| gate_vol:target=0.4,window=20,hyst=0,floor=1 | +0.58% | +0.38% | -31.72% | 0.01 | 0.01 | 33 | 21.6 | FAIL(return,mdd,calmar) |
| gate_vol:target=0.4,window=30,hyst=0,floor=1 | +16.45% | +10.46% | -24.81% | 0.42 | 0.40 | 27 | 17.6 | PASS |
| gate_vol:target=0.5,window=10,hyst=0,floor=1 | +7.43% | +4.79% | -33.10% | 0.14 | 0.14 | 42 | 27.4 | FAIL(return,mdd,calmar) |
| gate_vol:target=0.5,window=20,hyst=0,floor=1 | +11.77% | +7.54% | -32.15% | 0.23 | 0.23 | 40 | 26.1 | FAIL(mdd) |
| gate_vol:target=0.5,window=30,hyst=0,floor=1 | +8.77% | +5.65% | -31.00% | 0.18 | 0.17 | 25 | 16.3 | FAIL(mdd) |

## test — 2026-04-13 00:00:00+08:00 → 2026-09-03 20:00:00+08:00

| Strategy | Total | Ann. | Max DD | Calmar | Sharpe | Trades | Trades/yr | Accept (noharm) |
|---|---:|---:|---:|---:|---:|---:|---:|:--|
| buy_and_hold | +14.36% | +40.57% | -36.73% | 1.10 | 0.68 | 1 | 2.5 | — |
| gate_only | +11.59% | +32.09% | -5.54% | 5.79 | 1.70 | 0 | 0.0 | — |
| gate_vol:target=0.3,window=10,hyst=0,floor=1 | +3.97% | +10.39% | -3.34% | 3.11 | 1.32 | 2 | 5.1 | FAIL(return) |
| gate_vol:target=0.3,window=20,hyst=0,floor=1 | +5.01% | +13.19% | -2.53% | 5.21 | 1.78 | 1 | 2.5 | FAIL(return) |
| gate_vol:target=0.3,window=30,hyst=0,floor=1 | +5.79% | +15.36% | -2.77% | 5.54 | 1.74 | 0 | 0.0 | FAIL(return) |
| gate_vol:target=0.4,window=10,hyst=0,floor=1 | +6.00% | +15.95% | -5.16% | 3.09 | 1.36 | 7 | 17.8 | FAIL(return) |
| gate_vol:target=0.4,window=20,hyst=0,floor=1 | +5.79% | +15.36% | -2.77% | 5.54 | 1.74 | 0 | 0.0 | FAIL(return) |
| gate_vol:target=0.4,window=30,hyst=0,floor=1 | +6.54% | +17.43% | -2.77% | 6.28 | 1.88 | 1 | 2.5 | FAIL(return) |
| gate_vol:target=0.5,window=10,hyst=0,floor=1 | +6.90% | +18.44% | -4.55% | 4.05 | 1.50 | 2 | 5.1 | FAIL(return) |
| gate_vol:target=0.5,window=20,hyst=0,floor=1 | +7.89% | +21.25% | -3.89% | 5.46 | 1.75 | 1 | 2.5 | FAIL(return) |
| gate_vol:target=0.5,window=30,hyst=0,floor=1 | +8.70% | +23.57% | -4.16% | 5.67 | 1.71 | 0 | 0.0 | FAIL(return) |
