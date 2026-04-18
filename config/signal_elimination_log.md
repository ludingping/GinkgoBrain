# Signal Elimination Log — TC-A7

- Generated: `2026-04-18T13:43:54`  
- Sharpe gate: `< -0.2`  
- Correlation gate: `|ρ| > 0.95`  
- State-only signals exempt from Sharpe gate: `['sig_regime_drawdown', 'sig_vol_atr_pct', 'sig_vol_bb_position', 'sig_vol_bb_width']`

## Kept signals

- `sig_trend_macd_hist` — Sharpe `-0.167`
- `sig_trend_slope_21` — Sharpe `+0.745`
- `sig_mom_rsi_trend` — Sharpe `+0.693`
- `sig_mom_roc_zscore` — Sharpe `+0.074`
- `sig_vol_atr_pct` — Sharpe `+0.241` *(state-only)*
- `sig_vol_bb_width` — Sharpe `+0.360` *(state-only)*
- `sig_vol_bb_position` — Sharpe `+0.310` *(state-only)*
- `sig_volume_obv_slope` — Sharpe `+0.111`
- `sig_volume_ratio` — Sharpe `+0.460`
- `sig_regime_drawdown` — Sharpe `+0.000` *(state-only)*
- `sig_regime_return_60` — Sharpe `+0.391`

## Eliminated signals

| Signal | Sharpe | Reason | Detail |
|--------|--------|--------|--------|
| `sig_mom_stoch_state` | -0.594 | sharpe_below_threshold | Sharpe=-0.594 < -0.2 |
| `sig_trend_price_above_ma` | +0.528 | correlation_redundancy | |ρ|=0.982 (+0.982) with sig_mom_rsi_trend (kept: Sharpe=+0.693 vs +0.528) |
| `sig_trend_ema_cross` | +0.682 | correlation_redundancy | |ρ|=0.978 (+0.978) with sig_trend_slope_21 (kept: Sharpe=+0.745 vs +0.682) |
| `sig_mom_rsi_zone` | +0.093 | correlation_redundancy | |ρ|=0.952 (-0.952) with sig_mom_rsi_trend (kept: Sharpe=+0.693 vs +0.093) |