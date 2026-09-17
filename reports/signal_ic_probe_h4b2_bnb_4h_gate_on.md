# Signal IC probe — cfg_bnb_4h.yaml

- Generated: `2026-09-17T19:39:38`
- Timeframe **4h**, bars 11,493, 2021-06-06 12:00:00+08:00 → 2026-09-03 20:00:00+08:00
- Folds: 3 expanding time folds (validation windows only; no fitting)
- Rows: filtered (5063) — gate_on=True where=
- Pass: |IC| ≥ 0.03 on every fold, same sign, latest fold agrees, n ≥ 300 per fold. k-bar targets overlap, so IC magnitude is the criterion, not p-values.

## horizon k = 18 bars

Fold windows: f1 2022-09-28→2024-01-20; f2 2024-01-20→2025-05-13; f3 2025-05-13→2026-09-04

| signal | IC f1 (n) | IC f2 (n) | IC f3 (n) | IC mean | AUC last | spread last (bps) | verdict |
|---|---:|---:|---:|---:|---:|---:|:--|
| `retail_minus_top` | +0.0499 (1171) | -0.0381 (2145) | -0.1653 (1379) | -0.0512 | 0.4053 | -311.1 | FAIL (sign flips) |

## horizon k = 42 bars

Fold windows: f1 2022-09-28→2024-01-20; f2 2024-01-20→2025-05-13; f3 2025-05-13→2026-09-04

| signal | IC f1 (n) | IC f2 (n) | IC f3 (n) | IC mean | AUC last | spread last (bps) | verdict |
|---|---:|---:|---:|---:|---:|---:|:--|
| `retail_minus_top` | +0.0580 (1171) | +0.0460 (2145) | -0.2308 (1355) | -0.0423 | 0.3902 | -476.7 | FAIL (sign flips) |
