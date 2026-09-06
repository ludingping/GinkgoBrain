# Signal IC probe — stage2_4h_signal_v2.yaml

- Generated: `2026-09-06T18:29:55`
- Timeframe **4h**, bars 12,397, 2021-01-06 20:00:00+08:00 → 2026-09-03 20:00:00+08:00
- Folds: 3 expanding time folds (validation windows only; no fitting)
- Rows: filtered (3666) — gate_on=True where=dist_sma200>=0.15
- Pass: |IC| ≥ 0.03 on every fold, same sign, latest fold agrees, n ≥ 300 per fold. k-bar targets overlap, so IC magnitude is the criterion, not p-values.

## horizon k = 18 bars

Fold windows: f1 2022-06-07→2023-11-06; f2 2023-11-06→2025-04-05; f3 2025-04-05→2026-09-04

| signal | IC f1 (n) | IC f2 (n) | IC f3 (n) | IC mean | AUC last | spread last (bps) | verdict |
|---|---:|---:|---:|---:|---:|---:|:--|
| `funding_cum_3d` | -0.0723 (983) | +0.0405 (2017) | -0.0542 (312) | -0.0287 | 0.3916 | -106.6 | FAIL (sign flips) |
| `funding_cum_7d` | -0.1103 (983) | +0.0448 (2017) | -0.1187 (312) | -0.0614 | 0.3689 | -70.0 | FAIL (sign flips) |
| `funding_cum_3d_z` | -0.0630 (983) | +0.0487 (2017) | -0.0212 (312) | -0.0119 | 0.4184 | +13.6 | FAIL (sign flips) |

## horizon k = 6 bars

Fold windows: f1 2022-06-07→2023-11-06; f2 2023-11-06→2025-04-05; f3 2025-04-05→2026-09-04

| signal | IC f1 (n) | IC f2 (n) | IC f3 (n) | IC mean | AUC last | spread last (bps) | verdict |
|---|---:|---:|---:|---:|---:|---:|:--|
| `funding_cum_3d` | -0.0166 (983) | +0.0376 (2017) | +0.0005 (312) | +0.0072 | 0.5204 | -88.3 | FAIL (sign flips) |
| `funding_cum_7d` | -0.0678 (983) | +0.0238 (2017) | -0.1568 (312) | -0.0669 | 0.4311 | -81.9 | FAIL (sign flips) |
| `funding_cum_3d_z` | +0.0177 (983) | +0.0330 (2017) | +0.0706 (312) | +0.0404 | 0.5628 | +68.9 | FAIL (|IC|<0.03 on fold [1]) |
