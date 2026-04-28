# Signal Elimination Log — TC-A7

- Generated: `2026-04-28T10:04:13`  
- Sharpe gate: `< -0.2`  
- Correlation gate: `|ρ| > 0.95`  
- State-only signals exempt from Sharpe gate: `['sig_regime_drawdown', 'sig_vol_atr_pct', 'sig_vol_bb_position', 'sig_vol_bb_width']`

## Kept signals

- `sig_trend_macd_hist` — Sharpe `+0.143`
- `sig_trend_price_above_ma` — Sharpe `+0.047`
- `sig_trend_slope_21` — Sharpe `+0.308`
- `sig_mom_rsi_trend` — Sharpe `+0.147`
- `sig_mom_stoch_state` — Sharpe `+0.303`
- `sig_mom_roc_zscore` — Sharpe `+0.135`
- `sig_vol_atr_pct` — Sharpe `+0.012` *(state-only)*
- `sig_vol_bb_width` — Sharpe `-0.287` *(state-only)*
- `sig_vol_bb_position` — Sharpe `+0.424` *(state-only)*
- `sig_volume_ratio` — Sharpe `-0.034`
- `sig_regime_drawdown` — Sharpe `+0.000` *(state-only)*
- `sig_regime_return_60` — Sharpe `+0.258`
- `sig_mtf_4h_trend` — Sharpe `+0.117`
- `sig_mtf_1d_regime` — Sharpe `+0.539`
- `sig_hf_range_position` — Sharpe `-0.113`
- `sig_hf_body_ratio` — Sharpe `+0.357`
- `sig_hf_vol_zscore_24` — Sharpe `+0.236`
- `sig_mr_deviation_atr` — Sharpe `+0.057`
- `sig_funding_trend` — Sharpe `-0.071`
- `sig_oi_change_zscore` — Sharpe `+0.000`
- `sig_liq_long_zscore` — Sharpe `+0.031`
- `sig_liq_short_zscore` — Sharpe `-0.132`

## Eliminated signals

| Signal | Sharpe | Reason | Detail |
|--------|--------|--------|--------|
| `sig_mom_rsi_zone` | -0.370 | sharpe_below_threshold | Sharpe=-0.370 < -0.2 |
| `sig_volume_obv_slope` | -0.430 | sharpe_below_threshold | Sharpe=-0.430 < -0.2 |
| `sig_mr_rsi_extreme` | -0.476 | sharpe_below_threshold | Sharpe=-0.476 < -0.2 |
| `sig_funding_current` | -0.722 | sharpe_below_threshold | Sharpe=-0.722 < -0.2 |
| `sig_liq_imbalance` | -0.277 | sharpe_below_threshold | Sharpe=-0.277 < -0.2 |
| `sig_trend_ema_cross` | +0.123 | correlation_redundancy | |ρ|=0.977 (+0.977) with sig_trend_slope_21 (kept: Sharpe=+0.308 vs +0.123) |