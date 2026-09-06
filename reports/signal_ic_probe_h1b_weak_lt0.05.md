# Signal IC probe — stage2_4h_signal_v2.yaml

- Generated: `2026-09-06T18:25:34`
- Timeframe **4h**, bars 12,397, 2021-01-06 20:00:00+08:00 → 2026-09-03 20:00:00+08:00
- Folds: 3 expanding time folds (validation windows only; no fitting)
- Rows: filtered (516) — gate_on=True where=dist_sma200<0.05
- Pass: |IC| ≥ 0.03 on every fold, same sign, latest fold agrees. k-bar targets overlap, so IC magnitude is the criterion, not p-values.

## horizon k = 18 bars

Fold windows: f1 2022-06-07→2023-11-06; f2 2023-11-06→2025-04-05; f3 2025-04-05→2026-09-04

| signal | IC f1 | IC f2 | IC f3 | IC mean | AUC last | spread last (bps) | verdict |
|---|---:|---:|---:|---:|---:|---:|:--|
| `funding_cum_3d` | -0.2151 | +0.0796 | -0.4320 | -0.1891 | 0.3115 | -590.7 | FAIL (sign flips) |
| `funding_cum_7d` | -0.4888 | +0.2425 | -0.5139 | -0.2534 | 0.2829 | -652.3 | FAIL (sign flips) |
| `funding_cum_3d_z` | -0.1632 | -0.2253 | -0.3019 | -0.2301 | 0.3694 | +84.3 | **PASS** |

## horizon k = 6 bars

Fold windows: f1 2022-06-07→2023-11-06; f2 2023-11-06→2025-04-05; f3 2025-04-05→2026-09-04

| signal | IC f1 | IC f2 | IC f3 | IC mean | AUC last | spread last (bps) | verdict |
|---|---:|---:|---:|---:|---:|---:|:--|
| `funding_cum_3d` | -0.5248 | -0.1433 | -0.0355 | -0.2345 | 0.4336 | +72.4 | **PASS** |
| `funding_cum_7d` | -0.6326 | +0.0188 | -0.1610 | -0.2582 | 0.3592 | -110.1 | FAIL (sign flips) |
| `funding_cum_3d_z` | -0.3137 | -0.3731 | +0.0974 | -0.1965 | 0.4851 | +415.7 | FAIL (sign flips) |
