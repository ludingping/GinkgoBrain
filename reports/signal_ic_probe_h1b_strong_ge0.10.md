# Signal IC probe — stage2_4h_signal_v2.yaml

- Generated: `2026-09-06T18:28:09`
- Timeframe **4h**, bars 12,397, 2021-01-06 20:00:00+08:00 → 2026-09-03 20:00:00+08:00
- Folds: 3 expanding time folds (validation windows only; no fitting)
- Rows: filtered (4511) — gate_on=True where=dist_sma200>=0.10
- Pass: |IC| ≥ 0.03 on every fold, same sign, latest fold agrees, n ≥ 300 per fold. k-bar targets overlap, so IC magnitude is the criterion, not p-values.

## horizon k = 18 bars

Fold windows: f1 2022-06-07→2023-11-06; f2 2023-11-06→2025-04-05; f3 2025-04-05→2026-09-04

| signal | IC f1 (n) | IC f2 (n) | IC f3 (n) | IC mean | AUC last | spread last (bps) | verdict |
|---|---:|---:|---:|---:|---:|---:|:--|
| `funding_cum_3d` | -0.0474 (1193) | +0.0872 (2125) | -0.1575 (803) | -0.0392 | 0.4070 | +10.2 | FAIL (sign flips) |
| `funding_cum_7d` | -0.0637 (1193) | +0.0880 (2125) | -0.1739 (803) | -0.0499 | 0.4112 | +21.9 | FAIL (sign flips) |
| `funding_cum_3d_z` | -0.0597 (1193) | +0.0657 (2125) | -0.0169 (803) | -0.0036 | 0.4694 | +53.2 | FAIL (sign flips) |

## horizon k = 6 bars

Fold windows: f1 2022-06-07→2023-11-06; f2 2023-11-06→2025-04-05; f3 2025-04-05→2026-09-04

| signal | IC f1 (n) | IC f2 (n) | IC f3 (n) | IC mean | AUC last | spread last (bps) | verdict |
|---|---:|---:|---:|---:|---:|---:|:--|
| `funding_cum_3d` | -0.0185 (1193) | +0.0608 (2125) | -0.0738 (815) | -0.0105 | 0.4799 | -0.6 | FAIL (sign flips) |
| `funding_cum_7d` | -0.0496 (1193) | +0.0468 (2125) | -0.1123 (815) | -0.0384 | 0.4590 | -52.6 | FAIL (sign flips) |
| `funding_cum_3d_z` | +0.0054 (1193) | +0.0410 (2125) | +0.0051 (815) | +0.0172 | 0.5088 | +17.4 | FAIL (|IC|<0.03 on fold [1, 3]) |
