# Rule backtest — stage2_4h_signal_v2.yaml

- Generated: `2026-09-14T12:14:09`
- Symbol: `ETH/USDT`; gate = UTC daily close > SMA200
- Rules: `gate_only`; `gate_reduce:signal=dist_sma30,thr=0,level=0,side=below`; `gate_reduce:signal=dist_sma50,thr=0,level=0,side=below`; `gate_reduce:signal=dist_sma100,thr=0,level=0,side=below`
- Env: commission=0.001, min_hold=3, stop_atr_mult=0; Sharpe annualised by 2190
- Acceptance vs gate_only — improve: MDD ≤ 0.67×, return ≥ 80%×, Calmar >, trades/yr ≤ 30; noharm (test): MDD not deeper, return ≥ 80%×, trades/yr ≤ 30

## pre_val — 2021-08-01 00:00:00+08:00 → 2024-09-30 20:00:00+08:00

| Strategy | Total | Ann. | Max DD | Calmar | Sharpe | Trades | Trades/yr | Accept (improve) |
|---|---:|---:|---:|---:|---:|---:|---:|:--|
| buy_and_hold | +5.57% | +1.72% | -81.12% | 0.02 | 0.03 | 1 | 0.3 | — |
| gate_only | +148.47% | +33.27% | -37.81% | 0.88 | 0.63 | 13 | 4.1 | — |
| gate_reduce:signal=dist_sma30,thr=0,level=0,side=below | +8.30% | +2.55% | -49.21% | 0.05 | 0.07 | 67 | 21.1 | FAIL(return,mdd,calmar) |
| gate_reduce:signal=dist_sma50,thr=0,level=0,side=below | +63.58% | +16.80% | -37.64% | 0.45 | 0.41 | 49 | 15.5 | FAIL(return,mdd,calmar) |
| gate_reduce:signal=dist_sma100,thr=0,level=0,side=below | +61.96% | +16.43% | -41.70% | 0.39 | 0.36 | 47 | 14.8 | FAIL(return,mdd,calmar) |

## val — 2024-10-01 00:00:00+08:00 → 2026-04-12 20:00:00+08:00

| Strategy | Total | Ann. | Max DD | Calmar | Sharpe | Trades | Trades/yr | Accept (improve) |
|---|---:|---:|---:|---:|---:|---:|---:|:--|
| buy_and_hold | -15.84% | -10.65% | -65.12% | -0.16 | -0.16 | 1 | 0.7 | — |
| gate_only | +9.76% | +6.27% | -38.27% | 0.16 | 0.14 | 13 | 8.5 | — |
| gate_reduce:signal=dist_sma30,thr=0,level=0,side=below | +59.48% | +35.64% | -29.06% | 1.23 | 1.01 | 23 | 15.0 | FAIL(mdd) |
| gate_reduce:signal=dist_sma50,thr=0,level=0,side=below | +52.24% | +31.59% | -31.06% | 1.02 | 0.84 | 13 | 8.5 | FAIL(mdd) |
| gate_reduce:signal=dist_sma100,thr=0,level=0,side=below | +31.61% | +19.65% | -34.67% | 0.57 | 0.47 | 17 | 11.1 | FAIL(mdd) |

## test — 2026-04-13 00:00:00+08:00 → 2026-09-03 20:00:00+08:00

| Strategy | Total | Ann. | Max DD | Calmar | Sharpe | Trades | Trades/yr | Accept (noharm) |
|---|---:|---:|---:|---:|---:|---:|---:|:--|
| buy_and_hold | +14.36% | +40.57% | -36.73% | 1.10 | 0.68 | 1 | 2.5 | — |
| gate_only | +11.59% | +32.09% | -5.54% | 5.79 | 1.70 | 0 | 0.0 | — |
| gate_reduce:signal=dist_sma30,thr=0,level=0,side=below | +11.59% | +32.09% | -5.54% | 5.79 | 1.70 | 0 | 0.0 | PASS |
| gate_reduce:signal=dist_sma50,thr=0,level=0,side=below | +11.59% | +32.09% | -5.54% | 5.79 | 1.70 | 0 | 0.0 | PASS |
| gate_reduce:signal=dist_sma100,thr=0,level=0,side=below | +11.59% | +32.09% | -5.54% | 5.79 | 1.70 | 0 | 0.0 | PASS |
