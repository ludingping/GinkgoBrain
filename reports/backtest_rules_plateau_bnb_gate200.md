# Rule backtest — stage2_4h_signal_v2.yaml

- Generated: `2026-09-14T12:18:13`
- Symbol: `BNB/USDT`; gate = UTC daily close > SMA200
- Rules: `gate_only`; `gate_reduce:signal=dist_sma30,thr=0,level=0,side=below`; `gate_reduce:signal=dist_sma50,thr=0,level=0,side=below`; `gate_reduce:signal=dist_sma100,thr=0,level=0,side=below`
- Env: commission=0.001, min_hold=3, stop_atr_mult=0; Sharpe annualised by 2190
- Acceptance vs gate_only — improve: MDD ≤ 0.67×, return ≥ 80%×, Calmar >, trades/yr ≤ 30; noharm (test): MDD not deeper, return ≥ 80%×, trades/yr ≤ 30

## pre_val — 2021-08-01 00:00:00+08:00 → 2024-09-30 20:00:00+08:00

| Strategy | Total | Ann. | Max DD | Calmar | Sharpe | Trades | Trades/yr | Accept (improve) |
|---|---:|---:|---:|---:|---:|---:|---:|:--|
| buy_and_hold | +72.45% | +18.76% | -71.59% | 0.26 | 0.27 | 1 | 0.3 | — |
| gate_only | +24.61% | +7.19% | -63.09% | 0.11 | 0.15 | 32 | 10.1 | — |
| gate_reduce:signal=dist_sma30,thr=0,level=0,side=below | +32.63% | +9.32% | -63.66% | 0.15 | 0.23 | 82 | 25.9 | FAIL(mdd) |
| gate_reduce:signal=dist_sma50,thr=0,level=0,side=below | +76.52% | +19.64% | -43.76% | 0.45 | 0.44 | 58 | 18.3 | FAIL(mdd) |
| gate_reduce:signal=dist_sma100,thr=0,level=0,side=below | +76.52% | +19.64% | -49.72% | 0.39 | 0.42 | 42 | 13.3 | FAIL(mdd) |

## val — 2024-10-01 00:00:00+08:00 → 2026-04-12 20:00:00+08:00

| Strategy | Total | Ann. | Max DD | Calmar | Sharpe | Trades | Trades/yr | Accept (improve) |
|---|---:|---:|---:|---:|---:|---:|---:|:--|
| buy_and_hold | +3.22% | +2.09% | -57.28% | 0.04 | 0.04 | 1 | 0.7 | — |
| gate_only | -17.74% | -11.98% | -45.89% | -0.26 | -0.27 | 41 | 26.8 | — |
| gate_reduce:signal=dist_sma30,thr=0,level=0,side=below | +7.45% | +4.80% | -43.49% | 0.11 | 0.12 | 53 | 34.6 | FAIL(turnover,mdd) |
| gate_reduce:signal=dist_sma50,thr=0,level=0,side=below | +29.55% | +18.42% | -37.98% | 0.49 | 0.43 | 45 | 29.4 | FAIL(mdd) |
| gate_reduce:signal=dist_sma100,thr=0,level=0,side=below | +29.20% | +18.21% | -32.82% | 0.55 | 0.41 | 25 | 16.3 | FAIL(mdd) |

## test — 2026-04-13 00:00:00+08:00 → 2026-09-03 20:00:00+08:00

| Strategy | Total | Ann. | Max DD | Calmar | Sharpe | Trades | Trades/yr | Accept (noharm) |
|---|---:|---:|---:|---:|---:|---:|---:|:--|
| buy_and_hold | +22.70% | +68.06% | -26.67% | 2.55 | 1.37 | 1 | 2.5 | — |
| gate_only | +15.83% | +45.19% | -5.32% | 8.49 | 2.56 | 0 | 0.0 | — |
| gate_reduce:signal=dist_sma30,thr=0,level=0,side=below | +15.83% | +45.19% | -5.32% | 8.49 | 2.56 | 0 | 0.0 | PASS |
| gate_reduce:signal=dist_sma50,thr=0,level=0,side=below | +15.83% | +45.19% | -5.32% | 8.49 | 2.56 | 0 | 0.0 | PASS |
| gate_reduce:signal=dist_sma100,thr=0,level=0,side=below | +15.83% | +45.19% | -5.32% | 8.49 | 2.56 | 0 | 0.0 | PASS |
