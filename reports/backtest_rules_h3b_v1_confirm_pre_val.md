# Rule backtest — stage2_4h_signal_v2.yaml

- Generated: `2026-09-07T09:22:39`
- Rules: `gate_only`; `gate_reduce:signal=dist_sma50,thr=0,level=0,side=below`; `gate_reduce:signal=dist_sma50,thr=0,level=2,side=below`
- Env: commission=0.001, min_hold=3, stop_atr_mult=0; Sharpe annualised by 2190
- Acceptance vs gate_only — improve: MDD ≤ 0.67×, return ≥ 80%×, Calmar >, trades/yr ≤ 30; noharm (—): MDD not deeper, return ≥ 80%×, trades/yr ≤ 30

## pre_val — 2021-08-01 00:00:00+08:00 → 2024-09-30 20:00:00+08:00

| Strategy | Total | Ann. | Max DD | Calmar | Sharpe | Trades | Trades/yr | Accept (improve) |
|---|---:|---:|---:|---:|---:|---:|---:|:--|
| buy_and_hold | +53.73% | +14.53% | -77.04% | 0.19 | 0.25 | 1 | 0.3 | — |
| gate_only | +70.44% | +18.32% | -37.64% | 0.49 | 0.45 | 28 | 8.8 | — |
| gate_reduce:signal=dist_sma50,thr=0,level=0,side=below | +110.89% | +26.55% | -25.58% | 1.04 | 0.76 | 40 | 12.6 | FAIL(mdd) |
| gate_reduce:signal=dist_sma50,thr=0,level=2,side=below | +92.88% | +23.03% | -27.40% | 0.84 | 0.64 | 51 | 16.1 | FAIL(mdd) |
