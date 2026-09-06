# Signal IC probe — stage2_4h_signal_v2.yaml

- Generated: `2026-09-06T19:12:09`
- Timeframe **4h**, bars 12,397, 2021-01-06 20:00:00+08:00 → 2026-09-03 20:00:00+08:00
- Folds: 3 expanding time folds (validation windows only; no fitting)
- Rows: all bars
- Pass: |IC| ≥ 0.03 on every fold, same sign, latest fold agrees, n ≥ 300 per fold. k-bar targets overlap, so IC magnitude is the criterion, not p-values.

## horizon k = 6 bars

Fold windows: f1 2022-06-07→2023-11-06; f2 2023-11-06→2025-04-05; f3 2025-04-05→2026-09-04

| signal | IC f1 (n) | IC f2 (n) | IC f3 (n) | IC mean | AUC last | spread last (bps) | verdict |
|---|---:|---:|---:|---:|---:|---:|:--|
| `oi_level_90d_z` | -0.0618 (3099) | -0.0218 (3099) | -0.0160 (3093) | -0.0332 | 0.5017 | -41.7 | FAIL (|IC|<0.03 on fold [2, 3]) |
| `oi_chg_1d` | +0.0042 (3099) | -0.0728 (3088) | +0.0003 (3092) | -0.0227 | 0.4987 | -4.3 | FAIL (sign flips) |
| `oi_capitulation_1d` | -0.0049 (3099) | +0.0725 (3099) | -0.0216 (3093) | +0.0154 | 0.4908 | -14.2 | FAIL (sign flips) |
| `oi_new_longs_1d` | -0.0172 (3099) | -0.0510 (3099) | -0.0121 (3093) | -0.0268 | 0.4947 | +7.3 | FAIL (|IC|<0.03 on fold [1, 3]) |
| `sig_oi_change_zscore` | +0.0233 (3099) | -0.0477 (3099) | +0.0081 (3093) | -0.0054 | 0.5113 | -6.2 | FAIL (sign flips) |

## horizon k = 18 bars

Fold windows: f1 2022-06-07→2023-11-06; f2 2023-11-06→2025-04-05; f3 2025-04-05→2026-09-04

| signal | IC f1 (n) | IC f2 (n) | IC f3 (n) | IC mean | AUC last | spread last (bps) | verdict |
|---|---:|---:|---:|---:|---:|---:|:--|
| `oi_level_90d_z` | -0.0994 (3099) | -0.0175 (3099) | -0.0809 (3081) | -0.0659 | 0.4570 | -20.3 | FAIL (|IC|<0.03 on fold [2]) |
| `oi_chg_1d` | -0.0068 (3099) | -0.0187 (3088) | -0.0000 (3080) | -0.0085 | 0.4968 | -6.2 | FAIL (|IC|<0.03 on fold [1, 2, 3]) |
| `oi_capitulation_1d` | -0.0258 (3099) | -0.0026 (3099) | +0.0066 (3081) | -0.0072 | 0.5064 | +5.8 | FAIL (sign flips) |
| `oi_new_longs_1d` | +0.0081 (3099) | -0.0322 (3099) | -0.0043 (3081) | -0.0095 | 0.4867 | +26.0 | FAIL (sign flips) |
| `sig_oi_change_zscore` | +0.0172 (3099) | -0.0349 (3099) | +0.0037 (3081) | -0.0047 | 0.5115 | +1.4 | FAIL (sign flips) |
