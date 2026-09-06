# Signal IC probe — stage2_4h_signal_v2.yaml

- Generated: `2026-09-06T17:36:28`
- Timeframe **4h**, bars 12,397, 2021-01-06 20:00:00+08:00 → 2026-09-03 20:00:00+08:00
- Folds: 3 expanding time folds (validation windows only; no fitting)
- Pass: |IC| ≥ 0.03 on every fold, same sign, latest fold agrees. k-bar targets overlap, so IC magnitude is the criterion, not p-values.

## horizon k = 6 bars

Fold windows: f1 2022-06-07→2023-11-06; f2 2023-11-06→2025-04-05; f3 2025-04-05→2026-09-04

| signal | IC f1 | IC f2 | IC f3 | IC mean | AUC last | spread last (bps) | verdict |
|---|---:|---:|---:|---:|---:|---:|:--|
| `sig_funding_current` | -0.0237 | +0.0028 | -0.0142 | -0.0117 | 0.4944 | -4.6 | FAIL (sign flips) |
| `sig_funding_trend` | -0.0040 | -0.0088 | -0.0246 | -0.0125 | 0.4909 | -3.8 | FAIL (|IC|<0.03 on fold [1, 2, 3]) |
| `funding_cum_3d` | -0.0625 | +0.0238 | -0.0695 | -0.0360 | 0.4756 | -30.0 | FAIL (sign flips) |
| `funding_cum_7d` | -0.0493 | +0.0274 | -0.0723 | -0.0314 | 0.4758 | -56.5 | FAIL (sign flips) |
| `funding_rank_90d` | -0.0217 | +0.0263 | -0.0222 | -0.0059 | 0.4931 | -6.0 | FAIL (sign flips) |

## horizon k = 18 bars

Fold windows: f1 2022-06-07→2023-11-06; f2 2023-11-06→2025-04-05; f3 2025-04-05→2026-09-04

| signal | IC f1 | IC f2 | IC f3 | IC mean | AUC last | spread last (bps) | verdict |
|---|---:|---:|---:|---:|---:|---:|:--|
| `sig_funding_current` | -0.0070 | +0.0322 | -0.0256 | -0.0002 | 0.4787 | -30.9 | FAIL (sign flips) |
| `sig_funding_trend` | -0.0178 | +0.0192 | -0.0582 | -0.0189 | 0.4574 | -33.4 | FAIL (sign flips) |
| `funding_cum_3d` | -0.0950 | +0.0382 | -0.1635 | -0.0734 | 0.4190 | -171.1 | FAIL (sign flips) |
| `funding_cum_7d` | -0.0651 | +0.0486 | -0.1467 | -0.0544 | 0.4376 | -154.1 | FAIL (sign flips) |
| `funding_rank_90d` | -0.0172 | +0.0571 | -0.0478 | -0.0026 | 0.4704 | -35.7 | FAIL (sign flips) |
