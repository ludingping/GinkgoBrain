# Rule backtest — stage2_4h_signal_v2.yaml

- Generated: `2026-09-14T12:32:49`
- Symbol: `SOL/USDT`; gate = UTC daily close > SMA200
- Rules: `gate_only`; `gate_vol:target=0.4,window=20,hyst=0.0,floor=1`
- Env: commission=0.001, min_hold=3, stop_atr_mult=0; Sharpe annualised by 2190
- Acceptance vs gate_only — improve: MDD ≤ 0.67×, return ≥ 80%×, Calmar >, trades/yr ≤ 30; noharm (test): MDD not deeper, return ≥ 80%×, trades/yr ≤ 30

## pre_val — 2021-08-01 00:00:00+08:00 → 2024-09-30 20:00:00+08:00

| Strategy | Total | Ann. | Max DD | Calmar | Sharpe | Trades | Trades/yr | Accept (improve) |
|---|---:|---:|---:|---:|---:|---:|---:|:--|
| buy_and_hold | +347.95% | +60.50% | -96.60% | 0.63 | 0.44 | 1 | 0.3 | — |
| gate_only | +971.55% | +111.34% | -66.25% | 1.68 | 1.00 | 30 | 9.5 | — |
| gate_vol:target=0.4,window=20,hyst=0.0,floor=1 | +168.90% | +36.63% | -41.32% | 0.89 | 0.95 | 71 | 22.4 | FAIL(return,calmar) |

## val — 2024-10-01 00:00:00+08:00 → 2026-04-12 20:00:00+08:00

| Strategy | Total | Ann. | Max DD | Calmar | Sharpe | Trades | Trades/yr | Accept (improve) |
|---|---:|---:|---:|---:|---:|---:|---:|:--|
| buy_and_hold | -47.43% | -34.30% | -73.20% | -0.47 | -0.52 | 1 | 0.7 | — |
| gate_only | -0.73% | -0.48% | -45.32% | -0.01 | -0.01 | 11 | 7.2 | — |
| gate_vol:target=0.4,window=20,hyst=0.0,floor=1 | +6.48% | +4.18% | -24.56% | 0.17 | 0.15 | 28 | 18.3 | PASS |

## test — 2026-04-13 00:00:00+08:00 → 2026-09-03 20:00:00+08:00

| Strategy | Total | Ann. | Max DD | Calmar | Sharpe | Trades | Trades/yr | Accept (noharm) |
|---|---:|---:|---:|---:|---:|---:|---:|:--|
| buy_and_hold | +28.14% | +87.63% | -37.22% | 2.35 | 1.16 | 1 | 2.5 | — |
| gate_only | +23.06% | +69.30% | -10.32% | 6.72 | 2.41 | 0 | 0.0 | — |
| gate_vol:target=0.4,window=20,hyst=0.0,floor=1 | +16.87% | +48.54% | -6.42% | 7.56 | 2.61 | 1 | 2.5 | FAIL(return) |
