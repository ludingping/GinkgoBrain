# Rule backtest — stage2_4h_signal_v2.yaml

- Generated: `2026-09-06T17:28:50`
- Rules: `gate_only`
- Env: commission=0.001, min_hold=3, stop_atr_mult=0; Sharpe annualised by 2190
- Acceptance vs gate_only: MDD ≤ 0.67×, return ≥ 80%×, Calmar >, trades/yr ≤ 30

## val — 2024-10-01 00:00:00+08:00 → 2026-04-12 20:00:00+08:00

| Strategy | Total | Ann. | Max DD | Calmar | Sharpe | Trades | Trades/yr | Accept |
|---|---:|---:|---:|---:|---:|---:|---:|:--|
| buy_and_hold | +11.55% | +7.40% | -49.84% | 0.15 | 0.16 | 1 | 0.7 | — |
| gate_only | +17.60% | +11.17% | -33.95% | 0.33 | 0.30 | 19 | 12.4 | — |

## test — 2026-04-13 00:00:00+08:00 → 2026-09-03 20:00:00+08:00

| Strategy | Total | Ann. | Max DD | Calmar | Sharpe | Trades | Trades/yr | Accept |
|---|---:|---:|---:|---:|---:|---:|---:|:--|
| buy_and_hold | +14.32% | +40.46% | -29.25% | 1.38 | 0.90 | 1 | 2.5 | — |
| gate_only | +17.21% | +49.62% | -4.55% | 10.92 | 2.93 | 0 | 0.0 | — |
