# Signal Elimination Log — TC-A7

- Generated: `2026-09-05T23:03:25`  
- Sharpe gate: `< -0.2`  
- Correlation gate: `|ρ| > 0.95`  
- State-only signals exempt from Sharpe gate: `['sig_regime_drawdown', 'sig_vol_atr_pct', 'sig_vol_bb_position', 'sig_vol_bb_width']`

## Kept signals

- `sig_trend_macd_hist` — Sharpe `-0.065`
- `sig_trend_slope_21` — Sharpe `+0.850`
- `sig_mom_rsi_trend` — Sharpe `+0.649`
- `sig_mom_roc_zscore` — Sharpe `+0.128`
- `sig_vol_atr_pct` — Sharpe `+0.252` *(state-only)*
- `sig_vol_bb_width` — Sharpe `+0.300` *(state-only)*
- `sig_vol_bb_position` — Sharpe `+0.352` *(state-only)*
- `sig_volume_obv_slope` — Sharpe `+0.095`
- `sig_volume_ratio` — Sharpe `+0.396`
- `sig_regime_return_60` — Sharpe `+0.437`

## Eliminated signals

| Signal | Sharpe | Reason | Detail |
|--------|--------|--------|--------|
| `sig_mom_stoch_state` | -0.642 | sharpe_below_threshold | Sharpe=-0.642 < -0.2 |
| `sig_hf_range_position` | -0.571 | sharpe_below_threshold | Sharpe=-0.571 < -0.2 |
| `sig_hf_body_ratio` | -0.414 | sharpe_below_threshold | Sharpe=-0.414 < -0.2 |
| `sig_mr_rsi_extreme` | -0.285 | sharpe_below_threshold | Sharpe=-0.285 < -0.2 |
| `sig_mr_deviation_atr` | +0.518 | correlation_redundancy | |ρ|=0.997 (+0.997) with sig_trend_price_above_ma (kept: Sharpe=+0.592 vs +0.518) |
| `sig_trend_price_above_ma` | +0.592 | correlation_redundancy | |ρ|=0.982 (+0.982) with sig_mom_rsi_trend (kept: Sharpe=+0.649 vs +0.592) |
| `sig_trend_ema_cross` | +0.793 | correlation_redundancy | |ρ|=0.978 (+0.978) with sig_trend_slope_21 (kept: Sharpe=+0.850 vs +0.793) |
| `sig_hf_vol_zscore_24` | -0.020 | correlation_redundancy | |ρ|=0.961 (+0.961) with sig_volume_ratio (kept: Sharpe=+0.396 vs -0.020) |
| `sig_mom_rsi_zone` | +0.138 | correlation_redundancy | |ρ|=0.951 (-0.951) with sig_mom_rsi_trend (kept: Sharpe=+0.649 vs +0.138) |
| `sig_regime_drawdown` | +nan | manual_exclude | --exclude sig_regime_drawdown,sig_mtf_* |
| `sig_mtf_4h_trend` | +nan | manual_exclude | --exclude sig_regime_drawdown,sig_mtf_* |
| `sig_mtf_1d_regime` | +nan | manual_exclude | --exclude sig_regime_drawdown,sig_mtf_* |
| `sig_funding_current` | +nan | zero_variance | constant column in this sample |
| `sig_funding_trend` | +nan | zero_variance | constant column in this sample |
| `sig_oi_change_zscore` | +nan | zero_variance | constant column in this sample |
| `sig_liq_long_zscore` | +nan | zero_variance | constant column in this sample |
| `sig_liq_short_zscore` | +nan | zero_variance | constant column in this sample |
| `sig_liq_imbalance` | +nan | zero_variance | constant column in this sample |