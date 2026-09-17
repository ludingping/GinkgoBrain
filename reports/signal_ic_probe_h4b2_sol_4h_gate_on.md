# Signal IC probe — cfg_sol_4h.yaml

- Generated: `2026-09-17T19:02:27`
- Timeframe **4h**, bars 11,493, 2021-06-06 12:00:00+08:00 → 2026-09-03 20:00:00+08:00
- Folds: 3 expanding time folds (validation windows only; no fitting)
- Rows: filtered (4169) — gate_on=True where=
- Pass: |IC| ≥ 0.03 on every fold, same sign, latest fold agrees, n ≥ 300 per fold. k-bar targets overlap, so IC magnitude is the criterion, not p-values.

## horizon k = 18 bars

Fold windows: f1 2022-09-28→2024-01-20; f2 2024-01-20→2025-05-13; f3 2025-05-13→2026-09-04

| signal | IC f1 (n) | IC f2 (n) | IC f3 (n) | IC mean | AUC last | spread last (bps) | verdict |
|---|---:|---:|---:|---:|---:|---:|:--|
| `retail_minus_top` | -0.1013 (1291) | +0.0010 (1959) | +0.0184 (731) | -0.0273 | 0.5407 | -211.6 | FAIL (sign flips) |

## horizon k = 42 bars

Fold windows: f1 2022-09-28→2024-01-20; f2 2024-01-20→2025-05-13; f3 2025-05-13→2026-09-04

| signal | IC f1 (n) | IC f2 (n) | IC f3 (n) | IC mean | AUC last | spread last (bps) | verdict |
|---|---:|---:|---:|---:|---:|---:|:--|
| `retail_minus_top` | -0.1572 (1291) | +0.0654 (1959) | +0.1828 (707) | +0.0303 | 0.5843 | +509.1 | FAIL (sign flips) |
