# Rule backtest — stage2_4h_signal_v2.yaml

- Generated: `2026-09-17T12:06:32`
- Symbol: `ETH/USDT`; gate = UTC daily close > SMA200
- Rules: `gate_only`; `gate_reduce:signal=dist_sma50,thr=0,level=0,side=below`; `gate_exit_chop:exit=50,lookback=40,max_cross=2`; `gate_exit_chop:exit=50,lookback=40,max_cross=3`; `gate_exit_chop:exit=50,lookback=40,max_cross=4`; `gate_exit_chop:exit=50,lookback=60,max_cross=2`; `gate_exit_chop:exit=50,lookback=60,max_cross=3`; `gate_exit_chop:exit=50,lookback=60,max_cross=4`; `gate_exit_chop:exit=50,lookback=90,max_cross=2`; `gate_exit_chop:exit=50,lookback=90,max_cross=3`; `gate_exit_chop:exit=50,lookback=90,max_cross=4`
- Env: commission=0.001, min_hold=3, stop_atr_mult=0; Sharpe annualised by 2190
- Acceptance vs gate_only — improve: MDD ≤ 0.67×, return ≥ 80%×, Calmar >, trades/yr ≤ 30; noharm (test): MDD not deeper, return ≥ 80%×, trades/yr ≤ 30

## pre_val — 2021-08-01 00:00:00+08:00 → 2024-09-30 20:00:00+08:00

| Strategy | Total | Ann. | Max DD | Calmar | Sharpe | Trades | Trades/yr | Accept (improve) |
|---|---:|---:|---:|---:|---:|---:|---:|:--|
| buy_and_hold | +5.57% | +1.72% | -81.12% | 0.02 | 0.03 | 1 | 0.3 | — |
| gate_only | +148.47% | +33.27% | -37.81% | 0.88 | 0.63 | 13 | 4.1 | — |
| gate_reduce:signal=dist_sma50,thr=0,level=0,side=below | +63.58% | +16.80% | -37.64% | 0.45 | 0.41 | 49 | 15.5 | FAIL(return,mdd,calmar) |
| gate_exit_chop:exit=50,lookback=40,max_cross=2 | +32.02% | +9.16% | -48.56% | 0.19 | 0.20 | 29 | 9.1 | FAIL(return,mdd,calmar) |
| gate_exit_chop:exit=50,lookback=40,max_cross=3 | +35.15% | +9.97% | -44.80% | 0.22 | 0.23 | 39 | 12.3 | FAIL(return,mdd,calmar) |
| gate_exit_chop:exit=50,lookback=40,max_cross=4 | +35.15% | +9.97% | -44.80% | 0.22 | 0.23 | 39 | 12.3 | FAIL(return,mdd,calmar) |
| gate_exit_chop:exit=50,lookback=60,max_cross=2 | +83.35% | +21.08% | -37.47% | 0.56 | 0.44 | 21 | 6.6 | FAIL(return,mdd,calmar) |
| gate_exit_chop:exit=50,lookback=60,max_cross=3 | +44.15% | +12.23% | -41.26% | 0.30 | 0.28 | 31 | 9.8 | FAIL(return,mdd,calmar) |
| gate_exit_chop:exit=50,lookback=60,max_cross=4 | +37.52% | +10.58% | -44.80% | 0.24 | 0.25 | 35 | 11.0 | FAIL(return,mdd,calmar) |
| gate_exit_chop:exit=50,lookback=90,max_cross=2 | +85.15% | +21.45% | -36.75% | 0.58 | 0.44 | 19 | 6.0 | FAIL(return,mdd,calmar) |
| gate_exit_chop:exit=50,lookback=90,max_cross=3 | +34.67% | +9.85% | -45.69% | 0.22 | 0.22 | 27 | 8.5 | FAIL(return,mdd,calmar) |
| gate_exit_chop:exit=50,lookback=90,max_cross=4 | +46.29% | +12.75% | -41.26% | 0.31 | 0.28 | 29 | 9.1 | FAIL(return,mdd,calmar) |

## val — 2024-10-01 00:00:00+08:00 → 2026-04-12 20:00:00+08:00

| Strategy | Total | Ann. | Max DD | Calmar | Sharpe | Trades | Trades/yr | Accept (improve) |
|---|---:|---:|---:|---:|---:|---:|---:|:--|
| buy_and_hold | -15.84% | -10.65% | -65.12% | -0.16 | -0.16 | 1 | 0.7 | — |
| gate_only | +9.76% | +6.27% | -38.27% | 0.16 | 0.14 | 13 | 8.5 | — |
| gate_reduce:signal=dist_sma50,thr=0,level=0,side=below | +52.24% | +31.59% | -31.06% | 1.02 | 0.84 | 13 | 8.5 | FAIL(mdd) |
| gate_exit_chop:exit=50,lookback=40,max_cross=2 | +26.91% | +16.84% | -36.17% | 0.47 | 0.39 | 11 | 7.2 | FAIL(mdd) |
| gate_exit_chop:exit=50,lookback=40,max_cross=3 | +29.20% | +18.22% | -41.49% | 0.44 | 0.45 | 13 | 8.5 | FAIL(mdd) |
| gate_exit_chop:exit=50,lookback=40,max_cross=4 | +46.21% | +28.16% | -34.57% | 0.81 | 0.69 | 13 | 8.5 | FAIL(mdd) |
| gate_exit_chop:exit=50,lookback=60,max_cross=2 | +2.57% | +1.67% | -42.31% | 0.04 | 0.04 | 15 | 9.8 | FAIL(return,mdd,calmar) |
| gate_exit_chop:exit=50,lookback=60,max_cross=3 | +46.86% | +28.53% | -33.49% | 0.85 | 0.67 | 9 | 5.9 | FAIL(mdd) |
| gate_exit_chop:exit=50,lookback=60,max_cross=4 | +36.53% | +22.55% | -38.17% | 0.59 | 0.54 | 11 | 7.2 | FAIL(mdd) |
| gate_exit_chop:exit=50,lookback=90,max_cross=2 | +9.76% | +6.27% | -38.27% | 0.16 | 0.14 | 13 | 8.5 | FAIL(mdd,calmar) |
| gate_exit_chop:exit=50,lookback=90,max_cross=3 | +48.08% | +29.23% | -33.08% | 0.88 | 0.67 | 9 | 5.9 | FAIL(mdd) |
| gate_exit_chop:exit=50,lookback=90,max_cross=4 | +50.01% | +30.33% | -33.08% | 0.92 | 0.70 | 9 | 5.9 | FAIL(mdd) |

## test — 2026-04-13 00:00:00+08:00 → 2026-09-03 20:00:00+08:00

| Strategy | Total | Ann. | Max DD | Calmar | Sharpe | Trades | Trades/yr | Accept (noharm) |
|---|---:|---:|---:|---:|---:|---:|---:|:--|
| buy_and_hold | +14.36% | +40.57% | -36.73% | 1.10 | 0.68 | 1 | 2.5 | — |
| gate_only | +11.59% | +32.09% | -5.54% | 5.79 | 1.70 | 0 | 0.0 | — |
| gate_reduce:signal=dist_sma50,thr=0,level=0,side=below | +11.59% | +32.09% | -5.54% | 5.79 | 1.70 | 0 | 0.0 | PASS |
| gate_exit_chop:exit=50,lookback=40,max_cross=2 | +11.59% | +32.09% | -5.54% | 5.79 | 1.70 | 0 | 0.0 | PASS |
| gate_exit_chop:exit=50,lookback=40,max_cross=3 | +11.59% | +32.09% | -5.54% | 5.79 | 1.70 | 0 | 0.0 | PASS |
| gate_exit_chop:exit=50,lookback=40,max_cross=4 | +11.59% | +32.09% | -5.54% | 5.79 | 1.70 | 0 | 0.0 | PASS |
| gate_exit_chop:exit=50,lookback=60,max_cross=2 | +11.59% | +32.09% | -5.54% | 5.79 | 1.70 | 0 | 0.0 | PASS |
| gate_exit_chop:exit=50,lookback=60,max_cross=3 | +11.59% | +32.09% | -5.54% | 5.79 | 1.70 | 0 | 0.0 | PASS |
| gate_exit_chop:exit=50,lookback=60,max_cross=4 | +11.59% | +32.09% | -5.54% | 5.79 | 1.70 | 0 | 0.0 | PASS |
| gate_exit_chop:exit=50,lookback=90,max_cross=2 | +11.59% | +32.09% | -5.54% | 5.79 | 1.70 | 0 | 0.0 | PASS |
| gate_exit_chop:exit=50,lookback=90,max_cross=3 | +11.59% | +32.09% | -5.54% | 5.79 | 1.70 | 0 | 0.0 | PASS |
| gate_exit_chop:exit=50,lookback=90,max_cross=4 | +11.59% | +32.09% | -5.54% | 5.79 | 1.70 | 0 | 0.0 | PASS |
