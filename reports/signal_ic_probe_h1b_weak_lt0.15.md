# Signal IC probe — stage2_4h_signal_v2.yaml

- Generated: `2026-09-06T18:29:01`
- Timeframe **4h**, bars 12,397, 2021-01-06 20:00:00+08:00 → 2026-09-03 20:00:00+08:00
- Folds: 3 expanding time folds (validation windows only; no fitting)
- Rows: filtered (2261) — gate_on=True where=dist_sma200<0.15
- Pass: |IC| ≥ 0.03 on every fold, same sign, latest fold agrees, n ≥ 300 per fold. k-bar targets overlap, so IC magnitude is the criterion, not p-values.

## horizon k = 18 bars

Fold windows: f1 2022-06-07→2023-11-06; f2 2023-11-06→2025-04-05; f3 2025-04-05→2026-09-04

| signal | IC f1 (n) | IC f2 (n) | IC f3 (n) | IC mean | AUC last | spread last (bps) | verdict |
|---|---:|---:|---:|---:|---:|---:|:--|
| `funding_cum_3d` | -0.1542 (438) | +0.0947 (540) | -0.0894 (905) | -0.0496 | 0.4551 | -145.3 | FAIL (sign flips) |
| `funding_cum_7d` | -0.1326 (438) | +0.1449 (540) | -0.0502 (905) | -0.0126 | 0.4910 | -177.6 | FAIL (sign flips) |
| `funding_cum_3d_z` | -0.1038 (438) | +0.0257 (540) | +0.0470 (905) | -0.0104 | 0.4994 | +83.3 | FAIL (sign flips) |

## horizon k = 6 bars

Fold windows: f1 2022-06-07→2023-11-06; f2 2023-11-06→2025-04-05; f3 2025-04-05→2026-09-04

| signal | IC f1 (n) | IC f2 (n) | IC f3 (n) | IC mean | AUC last | spread last (bps) | verdict |
|---|---:|---:|---:|---:|---:|---:|:--|
| `funding_cum_3d` | -0.1001 (438) | -0.0228 (540) | -0.0280 (917) | -0.0503 | 0.4820 | +10.9 | FAIL (|IC|<0.03 on fold [2, 3]) |
| `funding_cum_7d` | -0.0806 (438) | +0.0488 (540) | -0.0158 (917) | -0.0159 | 0.4991 | -17.1 | FAIL (sign flips) |
| `funding_cum_3d_z` | -0.0753 (438) | -0.0605 (540) | +0.0454 (917) | -0.0301 | 0.5035 | +68.6 | FAIL (sign flips) |
