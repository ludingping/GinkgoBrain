# Signal IC probe — stage2_4h_signal_v2.yaml

- Generated: `2026-09-17T17:00:41`
- Timeframe **4h**, bars 12,397, 2021-01-06 20:00:00+08:00 → 2026-09-03 20:00:00+08:00
- Folds: 3 expanding time folds (validation windows only; no fitting)
- Rows: all bars
- Pass: |IC| ≥ 0.03 on every fold, same sign, latest fold agrees, n ≥ 300 per fold. k-bar targets overlap, so IC magnitude is the criterion, not p-values.

## horizon k = 6 bars

Fold windows: f1 2022-06-07→2023-11-06; f2 2023-11-06→2025-04-05; f3 2025-04-05→2026-09-04

| signal | IC f1 (n) | IC f2 (n) | IC f3 (n) | IC mean | AUC last | spread last (bps) | verdict |
|---|---:|---:|---:|---:|---:|---:|:--|
| `top_position_90d_z` | +0.0160 (1418) | +0.0293 (2558) | -0.0316 (3093) | +0.0046 | 0.4905 | -39.7 | FAIL (sign flips) |
| `retail_minus_top` | -0.0652 (2066) | -0.0009 (3097) | +0.0324 (3093) | -0.0112 | 0.5223 | +30.1 | FAIL (sign flips) |
| `retail_minus_top_90d_z` | -0.0225 (1418) | -0.0603 (2558) | +0.0067 (3093) | -0.0254 | 0.5031 | +3.6 | FAIL (sign flips) |
| `taker_dev_z30d` | -0.0075 (3082) | -0.0349 (2918) | +0.0122 (3093) | -0.0101 | 0.5091 | -5.0 | FAIL (sign flips) |

## horizon k = 18 bars

Fold windows: f1 2022-06-07→2023-11-06; f2 2023-11-06→2025-04-05; f3 2025-04-05→2026-09-04

| signal | IC f1 (n) | IC f2 (n) | IC f3 (n) | IC mean | AUC last | spread last (bps) | verdict |
|---|---:|---:|---:|---:|---:|---:|:--|
| `top_position_90d_z` | -0.0223 (1418) | +0.0335 (2558) | -0.0866 (3081) | -0.0251 | 0.4638 | -116.3 | FAIL (sign flips) |
| `retail_minus_top` | -0.0849 (2066) | +0.0387 (3097) | +0.0789 (3081) | +0.0109 | 0.5340 | +78.8 | FAIL (sign flips) |
| `retail_minus_top_90d_z` | -0.0427 (1418) | -0.0824 (2558) | +0.0479 (3081) | -0.0257 | 0.5192 | +39.4 | FAIL (sign flips) |
| `taker_dev_z30d` | +0.0260 (3082) | -0.0412 (2918) | +0.0142 (3081) | -0.0004 | 0.5151 | +1.6 | FAIL (sign flips) |

## horizon k = 42 bars

Fold windows: f1 2022-06-07→2023-11-06; f2 2023-11-06→2025-04-05; f3 2025-04-05→2026-09-04

| signal | IC f1 (n) | IC f2 (n) | IC f3 (n) | IC mean | AUC last | spread last (bps) | verdict |
|---|---:|---:|---:|---:|---:|---:|:--|
| `top_position_90d_z` | -0.0658 (1418) | +0.0520 (2558) | -0.1251 (3057) | -0.0463 | 0.4338 | -340.8 | FAIL (sign flips) |
| `retail_minus_top` | -0.0404 (2066) | +0.1444 (3097) | +0.1376 (3057) | +0.0805 | 0.5619 | +110.5 | FAIL (sign flips) |
| `retail_minus_top_90d_z` | -0.0331 (1418) | -0.0450 (2558) | +0.1197 (3057) | +0.0139 | 0.5517 | +180.9 | FAIL (sign flips) |
| `taker_dev_z30d` | -0.0347 (3082) | -0.0517 (2918) | +0.0625 (3057) | -0.0080 | 0.5327 | +101.7 | FAIL (sign flips) |
