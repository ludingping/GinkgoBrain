# H7 portfolio gate_only — stage2_4h_signal_v2.yaml

- Generated: `2026-09-14T16:47:33`
- Legs: BTC/USDT, ETH/USDT, SOL/USDT, BNB/USDT each `gate_only` (UTC daily close > SMA200); bar-level rebalance to fixed weights, rebalance cost not modelled
- Env per leg: commission=0.001, min_hold=3; Sharpe annualised by 2190
- Acceptance vs BTC gate_only — improve: MDD ≤ 0.67×, return ≥ 80%×, Calmar >; noharm (test): MDD not deeper, return ≥ 80%×

## pre_val — 2021-07-31 16:00:00+00:00 → 2024-09-30 08:00:00+00:00  (6941 bars)

| Strategy | Total | Ann. | Max DD | MDD ratio | Calmar | Sharpe | Accept (improve) |
|---|---:|---:|---:|---:|---:|---:|:--|
| BTC gate_only | +70.44% | +18.32% | -37.64% | 1.00 | 0.49 | 0.45 |  |
| ETH gate_only | +148.47% | +33.27% | -37.81% | 1.00 | 0.88 | 0.63 |  |
| SOL gate_only | +971.55% | +111.34% | -66.25% | 1.76 | 1.68 | 1.00 |  |
| BNB gate_only | +24.61% | +7.19% | -63.09% | 1.68 | 0.11 | 0.15 |  |
| P-EW4 | +222.79% | +44.73% | -42.69% | 1.13 | 1.05 | 0.88 | FAIL(mdd) |
| P-EW2 | +113.06% | +26.96% | -37.29% | 0.99 | 0.72 | 0.61 | FAIL(mdd) |
| P-BTC50 | +166.41% | +36.23% | -40.82% | 1.08 | 0.89 | 0.80 | FAIL(mdd) |

Pairwise correlation of per-bar gate_only log returns:

| | BTC | ETH | SOL | BNB |
|---|---:|---:|---:|---:|
| BTC | 1.00 | 0.77 | 0.51 | 0.58 |
| ETH | 0.77 | 1.00 | 0.57 | 0.62 |
| SOL | 0.51 | 0.57 | 1.00 | 0.45 |
| BNB | 0.58 | 0.62 | 0.45 | 1.00 |

## val — 2024-09-30 16:00:00+00:00 → 2026-04-12 08:00:00+00:00  (3353 bars)

| Strategy | Total | Ann. | Max DD | MDD ratio | Calmar | Sharpe | Accept (improve) |
|---|---:|---:|---:|---:|---:|---:|:--|
| BTC gate_only | +17.60% | +11.17% | -33.95% | 1.00 | 0.33 | 0.30 |  |
| ETH gate_only | +9.76% | +6.27% | -38.27% | 1.13 | 0.16 | 0.14 |  |
| SOL gate_only | -0.73% | -0.48% | -45.32% | 1.34 | -0.01 | -0.01 |  |
| BNB gate_only | -17.74% | -11.98% | -45.89% | 1.35 | -0.26 | -0.27 |  |
| P-EW4 | +6.72% | +4.34% | -32.20% | 0.95 | 0.13 | 0.12 | FAIL(return,mdd,calmar) |
| P-EW2 | +16.65% | +10.58% | -30.51% | 0.90 | 0.35 | 0.30 | FAIL(mdd) |
| P-BTC50 | +11.33% | +7.26% | -29.66% | 0.87 | 0.24 | 0.21 | FAIL(return,mdd,calmar) |

Pairwise correlation of per-bar gate_only log returns:

| | BTC | ETH | SOL | BNB |
|---|---:|---:|---:|---:|
| BTC | 1.00 | 0.55 | 0.57 | 0.50 |
| ETH | 0.55 | 1.00 | 0.63 | 0.57 |
| SOL | 0.57 | 0.63 | 1.00 | 0.53 |
| BNB | 0.50 | 0.57 | 0.53 | 1.00 |

## test — 2026-04-12 16:00:00+00:00 → 2026-09-03 08:00:00+00:00  (863 bars)

| Strategy | Total | Ann. | Max DD | MDD ratio | Calmar | Sharpe | Accept (noharm) |
|---|---:|---:|---:|---:|---:|---:|:--|
| BTC gate_only | +17.21% | +49.62% | -4.55% | 1.00 | 10.92 | 2.93 |  |
| ETH gate_only | +11.59% | +32.09% | -5.54% | 1.22 | 5.79 | 1.70 |  |
| SOL gate_only | +23.06% | +69.30% | -10.32% | 2.27 | 6.72 | 2.41 |  |
| BNB gate_only | +15.83% | +45.19% | -5.32% | 1.17 | 8.49 | 2.56 |  |
| P-EW4 | +16.99% | +48.91% | -5.87% | 1.29 | 8.33 | 2.64 | FAIL(mdd) |
| P-EW2 | +14.41% | +40.72% | -4.93% | 1.08 | 8.26 | 2.36 | FAIL(mdd) |
| P-BTC50 | +17.08% | +49.21% | -5.34% | 1.17 | 9.22 | 2.79 | FAIL(mdd) |

Pairwise correlation of per-bar gate_only log returns:

| | BTC | ETH | SOL | BNB |
|---|---:|---:|---:|---:|
| BTC | 1.00 | 0.84 | 0.74 | 0.75 |
| ETH | 0.84 | 1.00 | 0.74 | 0.70 |
| SOL | 0.74 | 0.74 | 1.00 | 0.76 |
| BNB | 0.75 | 0.70 | 0.76 | 1.00 |
