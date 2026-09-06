# Signal IC probe — stage2_4h_signal_v2.yaml

- Generated: `2026-09-06T21:52:08`
- Timeframe **4h**, bars 12,397, 2021-01-06 20:00:00+08:00 → 2026-09-03 20:00:00+08:00
- Folds: 3 expanding time folds (validation windows only; no fitting)
- Rows: filtered (5927) — gate_on=True where=
- Pass: |IC| ≥ 0.03 on every fold, same sign, latest fold agrees, n ≥ 300 per fold. k-bar targets overlap, so IC magnitude is the criterion, not p-values.

## horizon k = 18 bars

Fold windows: f1 2022-06-07→2023-11-06; f2 2023-11-06→2025-04-05; f3 2025-04-05→2026-09-04

| signal | IC f1 (n) | IC f2 (n) | IC f3 (n) | IC mean | AUC last | spread last (bps) | verdict |
|---|---:|---:|---:|---:|---:|---:|:--|
| `dist_sma50` | +0.0408 (1421) | +0.0966 (2557) | -0.0284 (1217) | +0.0363 | 0.4703 | -93.7 | FAIL (sign flips) |
| `dd20_atr` | +0.0366 (1421) | +0.0246 (2557) | -0.0403 (1217) | +0.0069 | 0.4609 | -78.8 | FAIL (sign flips) |
| `dist_low20` | +0.0598 (1421) | +0.0125 (2557) | -0.0913 (1217) | -0.0063 | 0.4466 | -124.1 | FAIL (sign flips) |

## horizon k = 6 bars

Fold windows: f1 2022-06-07→2023-11-06; f2 2023-11-06→2025-04-05; f3 2025-04-05→2026-09-04

| signal | IC f1 (n) | IC f2 (n) | IC f3 (n) | IC mean | AUC last | spread last (bps) | verdict |
|---|---:|---:|---:|---:|---:|---:|:--|
| `dist_sma50` | -0.0096 (1421) | +0.0495 (2557) | -0.0024 (1229) | +0.0125 | 0.4925 | -41.6 | FAIL (sign flips) |
| `dd20_atr` | -0.0272 (1421) | -0.0031 (2557) | +0.0074 (1229) | -0.0076 | 0.4886 | -47.4 | FAIL (sign flips) |
| `dist_low20` | +0.0215 (1421) | +0.0005 (2557) | -0.0202 (1229) | +0.0006 | 0.4843 | -54.1 | FAIL (sign flips) |
