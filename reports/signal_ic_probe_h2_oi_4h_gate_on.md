# Signal IC probe — stage2_4h_signal_v2.yaml

- Generated: `2026-09-06T19:13:51`
- Timeframe **4h**, bars 12,397, 2021-01-06 20:00:00+08:00 → 2026-09-03 20:00:00+08:00
- Folds: 3 expanding time folds (validation windows only; no fitting)
- Rows: filtered (5927) — gate_on=True where=
- Pass: |IC| ≥ 0.03 on every fold, same sign, latest fold agrees, n ≥ 300 per fold. k-bar targets overlap, so IC magnitude is the criterion, not p-values.

## horizon k = 6 bars

Fold windows: f1 2022-06-07→2023-11-06; f2 2023-11-06→2025-04-05; f3 2025-04-05→2026-09-04

| signal | IC f1 (n) | IC f2 (n) | IC f3 (n) | IC mean | AUC last | spread last (bps) | verdict |
|---|---:|---:|---:|---:|---:|---:|:--|
| `oi_level_90d_z` | -0.0662 (1421) | -0.0195 (2557) | -0.0534 (1229) | -0.0464 | 0.4734 | -24.9 | FAIL (|IC|<0.03 on fold [2]) |
| `oi_chg_1d` | -0.0209 (1421) | -0.0523 (2547) | +0.0216 (1229) | -0.0172 | 0.4904 | +5.0 | FAIL (sign flips) |
| `oi_capitulation_1d` | +0.0745 (1421) | +0.0547 (2557) | -0.0495 (1229) | +0.0266 | 0.4910 | -29.4 | FAIL (sign flips) |
| `oi_new_longs_1d` | -0.0249 (1421) | -0.0297 (2557) | +0.0088 (1229) | -0.0152 | 0.4938 | +31.3 | FAIL (sign flips) |
| `sig_oi_change_zscore` | -0.0028 (1421) | -0.0396 (2557) | +0.0131 (1229) | -0.0097 | 0.5099 | -11.7 | FAIL (sign flips) |

## horizon k = 18 bars

Fold windows: f1 2022-06-07→2023-11-06; f2 2023-11-06→2025-04-05; f3 2025-04-05→2026-09-04

| signal | IC f1 (n) | IC f2 (n) | IC f3 (n) | IC mean | AUC last | spread last (bps) | verdict |
|---|---:|---:|---:|---:|---:|---:|:--|
| `oi_level_90d_z` | -0.0978 (1421) | -0.0335 (2557) | -0.1444 (1217) | -0.0919 | 0.4266 | +52.5 | **PASS** |
| `oi_chg_1d` | +0.0019 (1421) | -0.0017 (2547) | +0.0056 (1217) | +0.0019 | 0.4791 | +39.3 | FAIL (sign flips) |
| `oi_capitulation_1d` | -0.0015 (1421) | -0.0218 (2557) | -0.0063 (1217) | -0.0099 | 0.5099 | -17.7 | FAIL (|IC|<0.03 on fold [1, 2, 3]) |
| `oi_new_longs_1d` | +0.0560 (1421) | -0.0212 (2557) | +0.0138 (1217) | +0.0162 | 0.4810 | +46.4 | FAIL (sign flips) |
| `sig_oi_change_zscore` | +0.0112 (1421) | -0.0222 (2557) | +0.0173 (1217) | +0.0021 | 0.5073 | -31.9 | FAIL (sign flips) |
