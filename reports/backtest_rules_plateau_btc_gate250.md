# Rule backtest — stage2_4h_signal_v2.yaml

- Generated: `2026-09-14T12:12:07`
- Symbol: `BTC/USDT`; gate = UTC daily close > SMA250
- Rules: `gate_only`; `gate_reduce:signal=dist_sma50,thr=0,level=0,side=below`
- Env: commission=0.001, min_hold=3, stop_atr_mult=0; Sharpe annualised by 2190
- Acceptance vs gate_only — improve: MDD ≤ 0.67×, return ≥ 80%×, Calmar >, trades/yr ≤ 30; noharm (test): MDD not deeper, return ≥ 80%×, trades/yr ≤ 30

## pre_val — 2021-08-01 00:00:00+08:00 → 2024-09-30 20:00:00+08:00

| Strategy | Total | Ann. | Max DD | Calmar | Sharpe | Trades | Trades/yr | Accept (improve) |
|---|---:|---:|---:|---:|---:|---:|---:|:--|
| buy_and_hold | +53.73% | +14.53% | -77.04% | 0.19 | 0.25 | 1 | 0.3 | — |
| gate_only | +53.46% | +14.47% | -45.90% | 0.32 | 0.36 | 34 | 10.7 | — |
| gate_reduce:signal=dist_sma50,thr=0,level=0,side=below | +116.57% | +27.61% | -25.95% | 1.06 | 0.82 | 42 | 13.3 | PASS |

## val — 2024-10-01 00:00:00+08:00 → 2026-04-12 20:00:00+08:00

| Strategy | Total | Ann. | Max DD | Calmar | Sharpe | Trades | Trades/yr | Accept (improve) |
|---|---:|---:|---:|---:|---:|---:|---:|:--|
| buy_and_hold | +11.55% | +7.40% | -49.84% | 0.15 | 0.16 | 1 | 0.7 | — |
| gate_only | +25.07% | +15.73% | -34.80% | 0.45 | 0.39 | 15 | 9.8 | — |
| gate_reduce:signal=dist_sma50,thr=0,level=0,side=below | +46.57% | +28.37% | -21.37% | 1.33 | 0.90 | 33 | 21.6 | PASS |

## test — 2026-04-13 00:00:00+08:00 → 2026-09-03 20:00:00+08:00

| Strategy | Total | Ann. | Max DD | Calmar | Sharpe | Trades | Trades/yr | Accept (noharm) |
|---|---:|---:|---:|---:|---:|---:|---:|:--|
| buy_and_hold | +14.32% | +40.46% | -29.25% | 1.38 | 0.90 | 1 | 2.5 | — |
| gate_only | +3.74% | +9.76% | -4.55% | 2.15 | 0.82 | 0 | 0.0 | — |
| gate_reduce:signal=dist_sma50,thr=0,level=0,side=below | +3.74% | +9.76% | -4.55% | 2.15 | 0.82 | 0 | 0.0 | PASS |
