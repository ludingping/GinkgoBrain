# Signal Elimination Log — TC-A7

- Generated: `2026-04-27T20:42:24`  
- Sharpe gate: `< -0.2`  
- Correlation gate: `|ρ| > 0.95`  
- State-only signals exempt from Sharpe gate: `['sig_regime_drawdown', 'sig_vol_atr_pct', 'sig_vol_bb_position', 'sig_vol_bb_width']`

## Kept signals

- `sig_trend_ema_cross` — Sharpe `+0.400`
- `sig_trend_macd_hist` — Sharpe `+0.008`
- `sig_mom_rsi_trend` — Sharpe `+0.227`
- `sig_mom_stoch_state` — Sharpe `+0.241`
- `sig_mom_roc_zscore` — Sharpe `+0.441`
- `sig_vol_atr_pct` — Sharpe `+0.284` *(state-only)*
- `sig_vol_bb_width` — Sharpe `+0.358` *(state-only)*
- `sig_vol_bb_position` — Sharpe `+0.510` *(state-only)*
- `sig_regime_drawdown` — Sharpe `+0.000` *(state-only)*
- `sig_regime_return_60` — Sharpe `+0.311`
- `sig_mtf_4h_trend` — Sharpe `+0.011`
- `sig_mtf_1d_regime` — Sharpe `+0.587`
- `sig_hf_body_ratio` — Sharpe `+0.162`
- `sig_hf_vol_zscore_24` — Sharpe `-0.015`
- `sig_oi_change_zscore` — Sharpe `-0.129`
- `sig_liq_long_zscore` — Sharpe `-0.013`
- `sig_liq_short_zscore` — Sharpe `-0.013`

## Eliminated signals

| Signal | Sharpe | Reason | Detail |
|--------|--------|--------|--------|
| `sig_mom_rsi_zone` | -0.275 | sharpe_below_threshold | Sharpe=-0.275 < -0.2 |
| `sig_volume_obv_slope` | -0.979 | sharpe_below_threshold | Sharpe=-0.979 < -0.2 |
| `sig_hf_range_position` | -0.614 | sharpe_below_threshold | Sharpe=-0.614 < -0.2 |
| `sig_mr_rsi_extreme` | -0.270 | sharpe_below_threshold | Sharpe=-0.270 < -0.2 |
| `sig_funding_current` | -0.525 | sharpe_below_threshold | Sharpe=-0.525 < -0.2 |
| `sig_funding_trend` | -0.283 | sharpe_below_threshold | Sharpe=-0.283 < -0.2 |
| `sig_liq_imbalance` | -0.279 | sharpe_below_threshold | Sharpe=-0.279 < -0.2 |
| `sig_mr_deviation_atr` | -0.091 | correlation_redundancy | |ρ|=0.997 (+0.997) with sig_trend_price_above_ma (kept: Sharpe=+0.024 vs -0.091) |
| `sig_trend_price_above_ma` | +0.024 | correlation_redundancy | |ρ|=0.983 (+0.983) with sig_mom_rsi_trend (kept: Sharpe=+0.227 vs +0.024) |
| `sig_trend_slope_21` | +0.217 | correlation_redundancy | |ρ|=0.978 (+0.978) with sig_trend_ema_cross (kept: Sharpe=+0.400 vs +0.217) |
| `sig_volume_ratio` | -0.057 | correlation_redundancy | |ρ|=0.958 (+0.958) with sig_hf_vol_zscore_24 (kept: Sharpe=-0.015 vs -0.057) |