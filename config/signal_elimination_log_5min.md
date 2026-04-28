# Signal Elimination Log — TC-A7

- Generated: `2026-04-26T11:22:40`  
- Sharpe gate: `< -0.2`  
- Correlation gate: `|ρ| > 0.95`  
- State-only signals exempt from Sharpe gate: `['sig_regime_drawdown', 'sig_vol_atr_pct', 'sig_vol_bb_position', 'sig_vol_bb_width']`

## Kept signals

- `sig_trend_macd_hist` — Sharpe `+0.406`
- `sig_trend_price_above_ma` — Sharpe `+0.732`
- `sig_trend_slope_21` — Sharpe `+0.366`
- `sig_mom_stoch_state` — Sharpe `+0.216`
- `sig_vol_atr_pct` — Sharpe `+0.332` *(state-only)*
- `sig_vol_bb_width` — Sharpe `+0.408` *(state-only)*
- `sig_vol_bb_position` — Sharpe `+0.721` *(state-only)*
- `sig_volume_obv_slope` — Sharpe `+0.307`
- `sig_volume_ratio` — Sharpe `+0.412`
- `sig_regime_drawdown` — Sharpe `+0.000` *(state-only)*
- `sig_regime_return_60` — Sharpe `+0.138`
- `sig_mtf_4h_trend` — Sharpe `+0.247`
- `sig_mtf_1d_regime` — Sharpe `+0.338`
- `sig_hf_vol_zscore_24` — Sharpe `+0.662`
- `sig_mr_rsi_extreme` — Sharpe `-0.029`
- `sig_funding_trend` — Sharpe `-0.029`
- `sig_oi_change_zscore` — Sharpe `+0.885`
- `sig_liq_long_zscore` — Sharpe `-0.064`
- `sig_liq_short_zscore` — Sharpe `-0.031`

## Eliminated signals

| Signal | Sharpe | Reason | Detail |
|--------|--------|--------|--------|
| `sig_mom_rsi_zone` | -0.492 | sharpe_below_threshold | Sharpe=-0.492 < -0.2 |
| `sig_mom_roc_zscore` | -0.224 | sharpe_below_threshold | Sharpe=-0.224 < -0.2 |
| `sig_hf_range_position` | -0.554 | sharpe_below_threshold | Sharpe=-0.554 < -0.2 |
| `sig_hf_body_ratio` | -0.389 | sharpe_below_threshold | Sharpe=-0.389 < -0.2 |
| `sig_funding_current` | -0.268 | sharpe_below_threshold | Sharpe=-0.268 < -0.2 |
| `sig_liq_imbalance` | -0.961 | sharpe_below_threshold | Sharpe=-0.961 < -0.2 |
| `sig_mr_deviation_atr` | +0.719 | correlation_redundancy | |ρ|=0.997 (+0.997) with sig_trend_price_above_ma (kept: Sharpe=+0.732 vs +0.719) |
| `sig_mom_rsi_trend` | +0.118 | correlation_redundancy | |ρ|=0.980 (+0.980) with sig_trend_price_above_ma (kept: Sharpe=+0.732 vs +0.118) |
| `sig_trend_ema_cross` | +0.354 | correlation_redundancy | |ρ|=0.977 (+0.977) with sig_trend_slope_21 (kept: Sharpe=+0.366 vs +0.354) |