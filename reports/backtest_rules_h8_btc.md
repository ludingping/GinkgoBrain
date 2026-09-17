# Rule backtest — stage2_4h_signal_v2.yaml

- Generated: `2026-09-17T12:06:25`
- Symbol: `BTC/USDT`; gate = UTC daily close > SMA200
- Rules: `gate_only`; `gate_reduce:signal=dist_sma50,thr=0,level=0,side=below`; `gate_exit_chop:exit=50,lookback=40,max_cross=2`; `gate_exit_chop:exit=50,lookback=40,max_cross=3`; `gate_exit_chop:exit=50,lookback=40,max_cross=4`; `gate_exit_chop:exit=50,lookback=60,max_cross=2`; `gate_exit_chop:exit=50,lookback=60,max_cross=3`; `gate_exit_chop:exit=50,lookback=60,max_cross=4`; `gate_exit_chop:exit=50,lookback=90,max_cross=2`; `gate_exit_chop:exit=50,lookback=90,max_cross=3`; `gate_exit_chop:exit=50,lookback=90,max_cross=4`
- Env: commission=0.001, min_hold=3, stop_atr_mult=0; Sharpe annualised by 2190
- Acceptance vs gate_only — improve: MDD ≤ 0.67×, return ≥ 80%×, Calmar >, trades/yr ≤ 30; noharm (test): MDD not deeper, return ≥ 80%×, trades/yr ≤ 30

## pre_val — 2021-08-01 00:00:00+08:00 → 2024-09-30 20:00:00+08:00

| Strategy | Total | Ann. | Max DD | Calmar | Sharpe | Trades | Trades/yr | Accept (improve) |
|---|---:|---:|---:|---:|---:|---:|---:|:--|
| buy_and_hold | +53.73% | +14.53% | -77.04% | 0.19 | 0.25 | 1 | 0.3 | — |
| gate_only | +70.44% | +18.32% | -37.64% | 0.49 | 0.45 | 28 | 8.8 | — |
| gate_reduce:signal=dist_sma50,thr=0,level=0,side=below | +110.89% | +26.55% | -25.58% | 1.04 | 0.76 | 40 | 12.6 | FAIL(mdd) |
| gate_exit_chop:exit=50,lookback=40,max_cross=2 | +79.26% | +20.22% | -26.41% | 0.77 | 0.57 | 36 | 11.4 | FAIL(mdd) |
| gate_exit_chop:exit=50,lookback=40,max_cross=3 | +116.83% | +27.66% | -25.58% | 1.08 | 0.79 | 36 | 11.4 | FAIL(mdd) |
| gate_exit_chop:exit=50,lookback=40,max_cross=4 | +116.83% | +27.66% | -25.58% | 1.08 | 0.79 | 36 | 11.4 | FAIL(mdd) |
| gate_exit_chop:exit=50,lookback=60,max_cross=2 | +102.82% | +25.00% | -25.58% | 0.98 | 0.69 | 32 | 10.1 | FAIL(mdd) |
| gate_exit_chop:exit=50,lookback=60,max_cross=3 | +86.42% | +21.71% | -25.58% | 0.85 | 0.63 | 36 | 11.4 | FAIL(mdd) |
| gate_exit_chop:exit=50,lookback=60,max_cross=4 | +116.83% | +27.66% | -25.58% | 1.08 | 0.79 | 36 | 11.4 | FAIL(mdd) |
| gate_exit_chop:exit=50,lookback=90,max_cross=2 | +38.20% | +10.75% | -35.13% | 0.31 | 0.29 | 32 | 10.1 | FAIL(return,mdd,calmar) |
| gate_exit_chop:exit=50,lookback=90,max_cross=3 | +30.94% | +8.88% | -32.12% | 0.28 | 0.25 | 38 | 12.0 | FAIL(return,mdd,calmar) |
| gate_exit_chop:exit=50,lookback=90,max_cross=4 | +70.13% | +18.25% | -32.12% | 0.57 | 0.49 | 36 | 11.4 | FAIL(mdd) |

## val — 2024-10-01 00:00:00+08:00 → 2026-04-12 20:00:00+08:00

| Strategy | Total | Ann. | Max DD | Calmar | Sharpe | Trades | Trades/yr | Accept (improve) |
|---|---:|---:|---:|---:|---:|---:|---:|:--|
| buy_and_hold | +11.55% | +7.40% | -49.84% | 0.15 | 0.16 | 1 | 0.7 | — |
| gate_only | +17.60% | +11.17% | -33.95% | 0.33 | 0.30 | 19 | 12.4 | — |
| gate_reduce:signal=dist_sma50,thr=0,level=0,side=below | +40.04% | +24.60% | -19.01% | 1.29 | 0.81 | 25 | 16.3 | PASS |
| gate_exit_chop:exit=50,lookback=40,max_cross=2 | +15.74% | +10.02% | -30.16% | 0.33 | 0.28 | 19 | 12.4 | FAIL(mdd) |
| gate_exit_chop:exit=50,lookback=40,max_cross=3 | +26.26% | +16.45% | -22.52% | 0.73 | 0.51 | 23 | 15.0 | PASS |
| gate_exit_chop:exit=50,lookback=40,max_cross=4 | +27.09% | +16.95% | -25.71% | 0.66 | 0.53 | 27 | 17.6 | FAIL(mdd) |
| gate_exit_chop:exit=50,lookback=60,max_cross=2 | +11.68% | +7.48% | -35.08% | 0.21 | 0.21 | 23 | 15.0 | FAIL(return,mdd,calmar) |
| gate_exit_chop:exit=50,lookback=60,max_cross=3 | +2.89% | +1.88% | -36.87% | 0.05 | 0.06 | 25 | 16.3 | FAIL(return,mdd,calmar) |
| gate_exit_chop:exit=50,lookback=60,max_cross=4 | +12.09% | +7.74% | -31.22% | 0.25 | 0.23 | 23 | 15.0 | FAIL(return,mdd,calmar) |
| gate_exit_chop:exit=50,lookback=90,max_cross=2 | +17.60% | +11.17% | -33.95% | 0.33 | 0.30 | 19 | 12.4 | FAIL(mdd,calmar) |
| gate_exit_chop:exit=50,lookback=90,max_cross=3 | +4.88% | +3.16% | -40.33% | 0.08 | 0.09 | 23 | 15.0 | FAIL(return,mdd,calmar) |
| gate_exit_chop:exit=50,lookback=90,max_cross=4 | +9.87% | +6.34% | -37.49% | 0.17 | 0.18 | 23 | 15.0 | FAIL(return,mdd,calmar) |

## test — 2026-04-13 00:00:00+08:00 → 2026-09-03 20:00:00+08:00

| Strategy | Total | Ann. | Max DD | Calmar | Sharpe | Trades | Trades/yr | Accept (noharm) |
|---|---:|---:|---:|---:|---:|---:|---:|:--|
| buy_and_hold | +14.32% | +40.46% | -29.25% | 1.38 | 0.90 | 1 | 2.5 | — |
| gate_only | +17.21% | +49.62% | -4.55% | 10.92 | 2.93 | 0 | 0.0 | — |
| gate_reduce:signal=dist_sma50,thr=0,level=0,side=below | +17.21% | +49.62% | -4.55% | 10.92 | 2.93 | 0 | 0.0 | PASS |
| gate_exit_chop:exit=50,lookback=40,max_cross=2 | +17.21% | +49.62% | -4.55% | 10.92 | 2.93 | 0 | 0.0 | PASS |
| gate_exit_chop:exit=50,lookback=40,max_cross=3 | +17.21% | +49.62% | -4.55% | 10.92 | 2.93 | 0 | 0.0 | PASS |
| gate_exit_chop:exit=50,lookback=40,max_cross=4 | +17.21% | +49.62% | -4.55% | 10.92 | 2.93 | 0 | 0.0 | PASS |
| gate_exit_chop:exit=50,lookback=60,max_cross=2 | +17.21% | +49.62% | -4.55% | 10.92 | 2.93 | 0 | 0.0 | PASS |
| gate_exit_chop:exit=50,lookback=60,max_cross=3 | +17.21% | +49.62% | -4.55% | 10.92 | 2.93 | 0 | 0.0 | PASS |
| gate_exit_chop:exit=50,lookback=60,max_cross=4 | +17.21% | +49.62% | -4.55% | 10.92 | 2.93 | 0 | 0.0 | PASS |
| gate_exit_chop:exit=50,lookback=90,max_cross=2 | +17.21% | +49.62% | -4.55% | 10.92 | 2.93 | 0 | 0.0 | PASS |
| gate_exit_chop:exit=50,lookback=90,max_cross=3 | +17.21% | +49.62% | -4.55% | 10.92 | 2.93 | 0 | 0.0 | PASS |
| gate_exit_chop:exit=50,lookback=90,max_cross=4 | +17.21% | +49.62% | -4.55% | 10.92 | 2.93 | 0 | 0.0 | PASS |
