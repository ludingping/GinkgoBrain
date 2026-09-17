# Signal IC probe — stage2_4h_signal_v2.yaml

- Generated: `2026-09-17T16:59:21`
- Timeframe **4h**, bars 12,397, 2021-01-06 20:00:00+08:00 → 2026-09-03 20:00:00+08:00
- Folds: 3 expanding time folds (validation windows only; no fitting)
- Rows: filtered (5927) — gate_on=True where=
- Pass: |IC| ≥ 0.03 on every fold, same sign, latest fold agrees, n ≥ 300 per fold. k-bar targets overlap, so IC magnitude is the criterion, not p-values.

## horizon k = 6 bars

Fold windows: f1 2022-06-07→2023-11-06; f2 2023-11-06→2025-04-05; f3 2025-04-05→2026-09-04

| signal | IC f1 (n) | IC f2 (n) | IC f3 (n) | IC mean | AUC last | spread last (bps) | verdict |
|---|---:|---:|---:|---:|---:|---:|:--|
| `top_position_90d_z` | +0.0105 (1064) | +0.0590 (2016) | -0.0195 (1229) | +0.0167 | 0.5007 | +32.6 | FAIL (sign flips) |
| `retail_minus_top` | -0.0626 (1421) | +0.0263 (2555) | +0.0459 (1229) | +0.0032 | 0.5170 | +34.8 | FAIL (sign flips) |
| `retail_minus_top_90d_z` | -0.0593 (1064) | -0.0472 (2016) | +0.0179 (1229) | -0.0296 | 0.5013 | +7.2 | FAIL (sign flips) |
| `taker_dev_z30d` | -0.0177 (1421) | -0.0122 (2376) | -0.0113 (1229) | -0.0137 | 0.4885 | -37.2 | FAIL (|IC|<0.03 on fold [1, 2, 3]) |

## horizon k = 18 bars

Fold windows: f1 2022-06-07→2023-11-06; f2 2023-11-06→2025-04-05; f3 2025-04-05→2026-09-04

| signal | IC f1 (n) | IC f2 (n) | IC f3 (n) | IC mean | AUC last | spread last (bps) | verdict |
|---|---:|---:|---:|---:|---:|---:|:--|
| `top_position_90d_z` | +0.0120 (1064) | +0.0944 (2016) | -0.0837 (1217) | +0.0076 | 0.4533 | -24.2 | FAIL (sign flips) |
| `retail_minus_top` | -0.1261 (1421) | +0.0884 (2555) | +0.1355 (1217) | +0.0326 | 0.5685 | +170.6 | FAIL (sign flips) |
| `retail_minus_top_90d_z` | -0.1085 (1064) | -0.0557 (2016) | +0.0948 (1217) | -0.0231 | 0.5516 | +98.3 | FAIL (sign flips) |
| `taker_dev_z30d` | -0.0173 (1421) | -0.0388 (2376) | -0.0064 (1217) | -0.0209 | 0.5097 | -19.6 | FAIL (|IC|<0.03 on fold [1, 3]) |

## horizon k = 42 bars

Fold windows: f1 2022-06-07→2023-11-06; f2 2023-11-06→2025-04-05; f3 2025-04-05→2026-09-04

| signal | IC f1 (n) | IC f2 (n) | IC f3 (n) | IC mean | AUC last | spread last (bps) | verdict |
|---|---:|---:|---:|---:|---:|---:|:--|
| `top_position_90d_z` | +0.0084 (1064) | +0.1018 (2016) | -0.1759 (1193) | -0.0219 | 0.4051 | -266.6 | FAIL (sign flips) |
| `retail_minus_top` | -0.0967 (1421) | +0.1967 (2555) | +0.3066 (1193) | +0.1355 | 0.6420 | +438.7 | FAIL (sign flips) |
| `retail_minus_top_90d_z` | -0.1028 (1064) | -0.0120 (2016) | +0.2698 (1193) | +0.0517 | 0.6252 | +425.7 | FAIL (sign flips) |
| `taker_dev_z30d` | -0.0992 (1421) | -0.0505 (2376) | +0.0630 (1193) | -0.0289 | 0.5458 | +29.8 | FAIL (sign flips) |
