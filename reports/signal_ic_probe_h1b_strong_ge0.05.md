# Signal IC probe — stage2_4h_signal_v2.yaml

- Generated: `2026-09-06T18:26:26`
- Timeframe **4h**, bars 12,397, 2021-01-06 20:00:00+08:00 → 2026-09-03 20:00:00+08:00
- Folds: 3 expanding time folds (validation windows only; no fitting)
- Rows: filtered (5411) — gate_on=True where=dist_sma200>=0.05
- Pass: |IC| ≥ 0.03 on every fold, same sign, latest fold agrees. k-bar targets overlap, so IC magnitude is the criterion, not p-values.

## horizon k = 18 bars

Fold windows: f1 2022-06-07→2023-11-06; f2 2023-11-06→2025-04-05; f3 2025-04-05→2026-09-04

| signal | IC f1 | IC f2 | IC f3 | IC mean | AUC last | spread last (bps) | verdict |
|---|---:|---:|---:|---:|---:|---:|:--|
| `funding_cum_3d` | -0.0598 | +0.0876 | -0.1699 | -0.0474 | 0.3980 | -19.9 | FAIL (sign flips) |
| `funding_cum_7d` | -0.0650 | +0.0982 | -0.1531 | -0.0400 | 0.4155 | -92.5 | FAIL (sign flips) |
| `funding_cum_3d_z` | -0.0493 | +0.0569 | -0.0425 | -0.0117 | 0.4485 | +46.7 | FAIL (sign flips) |

## horizon k = 6 bars

Fold windows: f1 2022-06-07→2023-11-06; f2 2023-11-06→2025-04-05; f3 2025-04-05→2026-09-04

| signal | IC f1 | IC f2 | IC f3 | IC mean | AUC last | spread last (bps) | verdict |
|---|---:|---:|---:|---:|---:|---:|:--|
| `funding_cum_3d` | -0.0191 | +0.0626 | -0.0806 | -0.0124 | 0.4762 | -25.6 | FAIL (sign flips) |
| `funding_cum_7d` | -0.0490 | +0.0590 | -0.0946 | -0.0282 | 0.4732 | -16.2 | FAIL (sign flips) |
| `funding_cum_3d_z` | +0.0105 | +0.0322 | +0.0043 | +0.0157 | 0.5041 | +3.6 | FAIL (|IC|<0.03 on fold [1, 3]) |
