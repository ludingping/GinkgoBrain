# Signal IC probe — stage1_daily.yaml

- Generated: `2026-09-17T17:05:41`
- Timeframe **1d**, bars 1,895, 2021-02-03 00:00:00+08:00 → 2026-04-12 00:00:00+08:00
- Folds: 3 expanding time folds (validation windows only; no fitting)
- Rows: all bars
- Pass: |IC| ≥ 0.03 on every fold, same sign, latest fold agrees, n ≥ 300 per fold. k-bar targets overlap, so IC magnitude is the criterion, not p-values.

## horizon k = 1 bars

Fold windows: f1 2022-05-25→2023-09-10; f2 2023-09-10→2024-12-26; f3 2024-12-26→2026-04-13

| signal | IC f1 (n) | IC f2 (n) | IC f3 (n) | IC mean | AUC last | spread last (bps) | verdict |
|---|---:|---:|---:|---:|---:|---:|:--|
| `top_position_90d_z` | +0.0881 (181) | -0.0173 (473) | -0.0197 (472) | +0.0171 | 0.5007 | -27.5 | FAIL (n<300 on fold [1]) |
| `retail_minus_top` | -0.0978 (301) | -0.0254 (473) | +0.0493 (472) | -0.0246 | 0.5251 | +24.0 | FAIL (sign flips) |
| `retail_minus_top_90d_z` | -0.1046 (181) | -0.0496 (473) | +0.0264 (472) | -0.0426 | 0.4954 | -35.9 | FAIL (n<300 on fold [1]) |
| `taker_dev_z30d` | +0.0214 (460) | +0.0046 (473) | -0.0271 (472) | -0.0004 | 0.4793 | -26.9 | FAIL (sign flips) |

## horizon k = 3 bars

Fold windows: f1 2022-05-25→2023-09-10; f2 2023-09-10→2024-12-26; f3 2024-12-26→2026-04-13

| signal | IC f1 (n) | IC f2 (n) | IC f3 (n) | IC mean | AUC last | spread last (bps) | verdict |
|---|---:|---:|---:|---:|---:|---:|:--|
| `top_position_90d_z` | +0.0394 (181) | -0.0090 (473) | -0.0311 (470) | -0.0002 | 0.4770 | +5.7 | FAIL (n<300 on fold [1]) |
| `retail_minus_top` | -0.0127 (301) | -0.0087 (473) | +0.1124 (470) | +0.0303 | 0.5556 | +174.9 | FAIL (sign flips) |
| `retail_minus_top_90d_z` | +0.0239 (181) | -0.0752 (473) | +0.0851 (470) | +0.0113 | 0.5467 | +32.3 | FAIL (n<300 on fold [1]) |
| `taker_dev_z30d` | +0.0850 (460) | -0.0002 (473) | +0.0446 (470) | +0.0431 | 0.5359 | +132.3 | FAIL (sign flips) |

## horizon k = 7 bars

Fold windows: f1 2022-05-25→2023-09-10; f2 2023-09-10→2024-12-26; f3 2024-12-26→2026-04-13

| signal | IC f1 (n) | IC f2 (n) | IC f3 (n) | IC mean | AUC last | spread last (bps) | verdict |
|---|---:|---:|---:|---:|---:|---:|:--|
| `top_position_90d_z` | -0.0603 (181) | +0.0065 (473) | -0.0534 (466) | -0.0358 | 0.4592 | -23.3 | FAIL (n<300 on fold [1]) |
| `retail_minus_top` | +0.0323 (301) | +0.0741 (473) | +0.2161 (466) | +0.1075 | 0.6037 | +528.9 | **PASS** |
| `retail_minus_top_90d_z` | +0.1170 (181) | -0.0238 (473) | +0.2169 (466) | +0.1034 | 0.5863 | +292.3 | FAIL (n<300 on fold [1]) |
| `taker_dev_z30d` | -0.0085 (460) | -0.0559 (473) | +0.0717 (466) | +0.0024 | 0.5425 | +154.4 | FAIL (sign flips) |
