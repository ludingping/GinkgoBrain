# Signal IC probe — stage2_4h_signal_v2.yaml

- Generated: `2026-09-06T17:40:18`
- Timeframe **4h**, bars 12,397, 2021-01-06 20:00:00+08:00 → 2026-09-03 20:00:00+08:00
- Folds: 3 expanding time folds (validation windows only; no fitting)
- Rows: gate-on bars only (5927)
- Pass: |IC| ≥ 0.03 on every fold, same sign, latest fold agrees. k-bar targets overlap, so IC magnitude is the criterion, not p-values.

## horizon k = 6 bars

Fold windows: f1 2022-06-07→2023-11-06; f2 2023-11-06→2025-04-05; f3 2025-04-05→2026-09-04

| signal | IC f1 | IC f2 | IC f3 | IC mean | AUC last | spread last (bps) | verdict |
|---|---:|---:|---:|---:|---:|---:|:--|
| `sig_funding_current` | +0.0017 | +0.0136 | -0.0218 | -0.0022 | 0.4830 | -15.7 | FAIL (sign flips) |
| `sig_funding_trend` | +0.0110 | +0.0110 | -0.0113 | +0.0036 | 0.4878 | +7.2 | FAIL (sign flips) |
| `funding_cum_3d` | -0.0507 | +0.0647 | -0.0844 | -0.0235 | 0.4672 | -10.4 | FAIL (sign flips) |
| `funding_cum_7d` | -0.0810 | +0.0648 | -0.1068 | -0.0410 | 0.4589 | -49.8 | FAIL (sign flips) |
| `funding_rank_90d` | -0.0046 | +0.0494 | -0.0410 | +0.0013 | 0.4774 | -6.3 | FAIL (sign flips) |

## horizon k = 18 bars

Fold windows: f1 2022-06-07→2023-11-06; f2 2023-11-06→2025-04-05; f3 2025-04-05→2026-09-04

| signal | IC f1 | IC f2 | IC f3 | IC mean | AUC last | spread last (bps) | verdict |
|---|---:|---:|---:|---:|---:|---:|:--|
| `sig_funding_current` | +0.0357 | +0.0515 | -0.0071 | +0.0267 | 0.4798 | +5.5 | FAIL (sign flips) |
| `sig_funding_trend` | +0.0216 | +0.0458 | -0.0349 | +0.0108 | 0.4547 | -10.8 | FAIL (sign flips) |
| `funding_cum_3d` | -0.1100 | +0.1150 | -0.2132 | -0.0694 | 0.3855 | -109.5 | FAIL (sign flips) |
| `funding_cum_7d` | -0.1207 | +0.1290 | -0.2097 | -0.0671 | 0.3972 | -233.3 | FAIL (sign flips) |
| `funding_rank_90d` | +0.0143 | +0.0982 | -0.0468 | +0.0219 | 0.4629 | +20.8 | FAIL (sign flips) |
