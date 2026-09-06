# Signal IC probe — stage2_4h_signal_v2.yaml

- Generated: `2026-09-06T18:27:17`
- Timeframe **4h**, bars 12,397, 2021-01-06 20:00:00+08:00 → 2026-09-03 20:00:00+08:00
- Folds: 3 expanding time folds (validation windows only; no fitting)
- Rows: filtered (1416) — gate_on=True where=dist_sma200<0.10
- Pass: |IC| ≥ 0.03 on every fold, same sign, latest fold agrees, n ≥ 300 per fold. k-bar targets overlap, so IC magnitude is the criterion, not p-values.

## horizon k = 18 bars

Fold windows: f1 2022-06-07→2023-11-06; f2 2023-11-06→2025-04-05; f3 2025-04-05→2026-09-04

| signal | IC f1 (n) | IC f2 (n) | IC f3 (n) | IC mean | AUC last | spread last (bps) | verdict |
|---|---:|---:|---:|---:|---:|---:|:--|
| `funding_cum_3d` | -0.3823 (228) | +0.1180 (432) | -0.0970 (414) | -0.1204 | 0.4299 | -76.8 | FAIL (n<300 on fold [1]) |
| `funding_cum_7d` | -0.3515 (228) | +0.2333 (432) | -0.0587 (414) | -0.0590 | 0.4660 | -368.6 | FAIL (n<300 on fold [1]) |
| `funding_cum_3d_z` | -0.2780 (228) | -0.0297 (432) | -0.0006 (414) | -0.1028 | 0.4404 | +15.0 | FAIL (n<300 on fold [1]) |

## horizon k = 6 bars

Fold windows: f1 2022-06-07→2023-11-06; f2 2023-11-06→2025-04-05; f3 2025-04-05→2026-09-04

| signal | IC f1 (n) | IC f2 (n) | IC f3 (n) | IC mean | AUC last | spread last (bps) | verdict |
|---|---:|---:|---:|---:|---:|---:|:--|
| `funding_cum_3d` | -0.1857 (228) | -0.0444 (432) | +0.0283 (414) | -0.0673 | 0.5035 | +85.0 | FAIL (n<300 on fold [1]) |
| `funding_cum_7d` | -0.1661 (228) | +0.0559 (432) | +0.0239 (414) | -0.0288 | 0.5177 | -53.2 | FAIL (n<300 on fold [1]) |
| `funding_cum_3d_z` | -0.1573 (228) | -0.1254 (432) | +0.0986 (414) | -0.0614 | 0.5157 | +123.7 | FAIL (n<300 on fold [1]) |
