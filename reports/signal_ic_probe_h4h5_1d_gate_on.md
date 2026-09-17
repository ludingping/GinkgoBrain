# Signal IC probe — stage1_daily.yaml

- Generated: `2026-09-17T17:04:42`
- Timeframe **1d**, bars 1,895, 2021-02-03 00:00:00+08:00 → 2026-04-12 00:00:00+08:00
- Folds: 3 expanding time folds (validation windows only; no fitting)
- Rows: filtered (969) — gate_on=True where=
- Pass: |IC| ≥ 0.03 on every fold, same sign, latest fold agrees, n ≥ 300 per fold. k-bar targets overlap, so IC magnitude is the criterion, not p-values.

## horizon k = 1 bars

Fold windows: f1 2022-05-25→2023-09-10; f2 2023-09-10→2024-12-26; f3 2024-12-26→2026-04-13

| signal | IC f1 (n) | IC f2 (n) | IC f3 (n) | IC mean | AUC last | spread last (bps) | verdict |
|---|---:|---:|---:|---:|---:|---:|:--|
| `top_position_90d_z` | +0.0839 (158) | -0.0040 (364) | -0.0277 (275) | +0.0174 | 0.5056 | -14.0 | FAIL (n<300 on fold [1, 3]) |
| `retail_minus_top` | -0.1140 (216) | +0.0299 (364) | +0.0527 (275) | -0.0105 | 0.5205 | +25.3 | FAIL (n<300 on fold [1, 3]) |
| `retail_minus_top_90d_z` | -0.1444 (158) | -0.0125 (364) | +0.0412 (275) | -0.0386 | 0.5036 | -40.1 | FAIL (n<300 on fold [1, 3]) |
| `taker_dev_z30d` | -0.0483 (216) | -0.0063 (364) | -0.0827 (275) | -0.0457 | 0.4484 | -48.9 | FAIL (n<300 on fold [1, 3]) |

## horizon k = 3 bars

Fold windows: f1 2022-05-25→2023-09-10; f2 2023-09-10→2024-12-26; f3 2024-12-26→2026-04-13

| signal | IC f1 (n) | IC f2 (n) | IC f3 (n) | IC mean | AUC last | spread last (bps) | verdict |
|---|---:|---:|---:|---:|---:|---:|:--|
| `top_position_90d_z` | +0.0537 (158) | +0.0149 (364) | -0.0264 (275) | +0.0141 | 0.4774 | -25.6 | FAIL (n<300 on fold [1, 3]) |
| `retail_minus_top` | -0.0682 (216) | +0.0486 (364) | +0.1806 (275) | +0.0536 | 0.5897 | +196.3 | FAIL (n<300 on fold [1, 3]) |
| `retail_minus_top_90d_z` | -0.0322 (158) | -0.0650 (364) | +0.1531 (275) | +0.0187 | 0.5821 | +29.0 | FAIL (n<300 on fold [1, 3]) |
| `taker_dev_z30d` | -0.0167 (216) | +0.0031 (364) | +0.0046 (275) | -0.0030 | 0.5223 | +20.1 | FAIL (n<300 on fold [1, 3]) |

## horizon k = 7 bars

Fold windows: f1 2022-05-25→2023-09-10; f2 2023-09-10→2024-12-26; f3 2024-12-26→2026-04-13

| signal | IC f1 (n) | IC f2 (n) | IC f3 (n) | IC mean | AUC last | spread last (bps) | verdict |
|---|---:|---:|---:|---:|---:|---:|:--|
| `top_position_90d_z` | -0.0159 (158) | +0.0148 (364) | -0.1129 (275) | -0.0380 | 0.4227 | -170.7 | FAIL (n<300 on fold [1, 3]) |
| `retail_minus_top` | +0.0041 (216) | +0.1338 (364) | +0.3208 (275) | +0.1529 | 0.6368 | +563.9 | FAIL (n<300 on fold [1, 3]) |
| `retail_minus_top_90d_z` | +0.0850 (158) | -0.0266 (364) | +0.3331 (275) | +0.1305 | 0.6509 | +397.0 | FAIL (n<300 on fold [1, 3]) |
| `taker_dev_z30d` | -0.1055 (216) | -0.0711 (364) | +0.0322 (275) | -0.0481 | 0.5311 | +96.6 | FAIL (n<300 on fold [1, 3]) |
