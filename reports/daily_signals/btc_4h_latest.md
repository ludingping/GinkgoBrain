# BTC/USDT 4h PPO signal

- As of bar: `2026-09-05 00:00:00+08:00` (close 79,793.34)
- Generated: `2026-09-05T23:49:09`
- Model: `artifacts/btc_4h_ppo_gate_v2.zip`
- Replay anchor: `2026-04-13 00:00:00+08:00` (870 bars replayed)

## Decision

| | |
|---|---|
| **Target position** | **0%** — 维持 0% |
| Current position (replayed book) | 0% |
| Unrealised P&L | +0.00% |
| Action lock | free |
| Steps since last switch | 3 |
| Steps since stop | 24 |
| Action probs (0/25/50/75/100%) | 0.24 / 0.23 / 0.20 / 0.18 / 0.15 |

## Signals now (by |value|)

| Signal | Value | Read |
|--------|------:|------|
| `sig_vol_bb_width` | +0.520 | ↑ bullish |
| `sig_regime_return_60` | -0.422 | ↓ bearish |
| `sig_mom_roc_zscore` | -0.367 | ↓ bearish |
| `sig_volume_obv_slope` | -0.334 | ↓ bearish |
| `sig_vol_bb_position` | +0.289 | ↑ bullish |
| `sig_volume_ratio` | -0.149 | ≈ neutral |
| `sig_trend_macd_hist` | +0.118 | ≈ neutral |
| `sig_mom_rsi_trend` | +0.094 | ≈ neutral |

## Last 24 bars

| Bar | Close | Executed | Target | Position | Stop |
|-----|------:|---------:|-------:|---------:|:----:|
| 2026-09-01 00:00 | 78,898 | 0 | 0% | 0% |  |
| 2026-09-01 04:00 | 78,581 | 0 | 0% | 0% |  |
| 2026-09-01 08:00 | 78,679 | 0 | 0% | 0% |  |
| 2026-09-01 12:00 | 78,632 | 0 | 0% | 0% |  |
| 2026-09-01 16:00 | 78,099 | 0 | 0% | 0% |  |
| 2026-09-01 20:00 | 77,919 | 0 | 0% | 0% |  |
| 2026-09-02 00:00 | 77,312 | 0 | 0% | 0% |  |
| 2026-09-02 04:00 | 77,439 | 0 | 0% | 0% |  |
| 2026-09-02 08:00 | 77,564 | 0 | 0% | 0% |  |
| 2026-09-02 12:00 | 77,474 | 0 | 0% | 0% |  |
| 2026-09-02 16:00 | 76,829 | 0 | 0% | 0% |  |
| 2026-09-02 20:00 | 77,289 | 0 | 0% | 0% |  |
| 2026-09-03 00:00 | 77,347 | 0 | 0% | 0% |  |
| 2026-09-03 04:00 | 77,340 | 0 | 0% | 0% |  |
| 2026-09-03 08:00 | 77,708 | 0 | 0% | 0% |  |
| 2026-09-03 12:00 | 77,663 | 0 | 0% | 0% |  |
| 2026-09-03 16:00 | 77,948 | 0 | 0% | 0% |  |
| 2026-09-03 20:00 | 81,348 | 0 | 0% | 0% |  |
| 2026-09-04 00:00 | 81,755 | 0 | 0% | 0% |  |
| 2026-09-04 04:00 | 81,270 | 0 | 0% | 0% |  |
| 2026-09-04 08:00 | 80,850 | 0 | 0% | 0% |  |
| 2026-09-04 12:00 | 80,657 | 0 | 0% | 0% |  |
| 2026-09-04 16:00 | 81,225 | 0 | 0% | 0% |  |
| 2026-09-04 20:00 | 79,429 | 0 | 0% | 0% |  |

## Replay since anchor (2026-04-13 → 2026-09-05)

| Total return | Max DD | Sharpe (ann.) | Trades | Stops |
|-------------:|-------:|--------------:|-------:|------:|
| +1.14% | -1.12% | 0.782 | 29 | 0 |

> Positions are *targets* for a spot long-only book; the replayed book starts flat at the anchor with the config's initial balance.