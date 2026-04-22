# Signal Elimination Log — TC-A7

- Generated: `2026-04-19T11:37:58`  
- Sharpe gate: `< -0.2`  
- Correlation gate: `|ρ| > 0.95`  
- State-only signals exempt from Sharpe gate: `['sig_regime_drawdown', 'sig_vol_atr_pct', 'sig_vol_bb_position', 'sig_vol_bb_width']`

## Kept signals

- `sig_trend_ema_cross` — Sharpe `+0.320`
- `sig_trend_macd_hist` — Sharpe `+0.559`
- `sig_trend_price_above_ma` — Sharpe `+0.432`
- `sig_mom_stoch_state` — Sharpe `-0.188`
- `sig_mom_roc_zscore` — Sharpe `+0.494`
- `sig_vol_atr_pct` — Sharpe `+0.852` *(state-only)*
- `sig_vol_bb_width` — Sharpe `+0.740` *(state-only)*
- `sig_vol_bb_position` — Sharpe `+0.435` *(state-only)*
- `sig_volume_obv_slope` — Sharpe `-0.000`
- `sig_regime_drawdown` — Sharpe `+0.000` *(state-only)*
- `sig_regime_return_60` — Sharpe `+0.167`
- `sig_mtf_4h_trend` — Sharpe `+2.307`
- `sig_mtf_1d_regime` — Sharpe `+0.967`
- `sig_hf_vol_zscore_24` — Sharpe `+0.029`
- `sig_mr_rsi_extreme` — Sharpe `-0.002`

## Eliminated signals

| Signal | Sharpe | Reason | Detail |
|--------|--------|--------|--------|
| `sig_mom_rsi_zone` | -0.337 | sharpe_below_threshold | Sharpe=-0.337 < -0.2 |
| `sig_hf_range_position` | -1.157 | sharpe_below_threshold | Sharpe=-1.157 < -0.2 |
| `sig_hf_body_ratio` | -0.507 | sharpe_below_threshold | Sharpe=-0.507 < -0.2 |
| `sig_mr_deviation_atr` | +0.367 | correlation_redundancy | |ρ|=0.997 (+0.997) with sig_trend_price_above_ma (kept: Sharpe=+0.432 vs +0.367) |
| `sig_mom_rsi_trend` | +0.260 | correlation_redundancy | |ρ|=0.982 (+0.982) with sig_trend_price_above_ma (kept: Sharpe=+0.432 vs +0.260) |
| `sig_trend_slope_21` | +0.096 | correlation_redundancy | |ρ|=0.978 (+0.978) with sig_trend_ema_cross (kept: Sharpe=+0.320 vs +0.096) |
| `sig_volume_ratio` | +0.008 | correlation_redundancy | |ρ|=0.955 (+0.955) with sig_hf_vol_zscore_24 (kept: Sharpe=+0.029 vs +0.008) |