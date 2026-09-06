# BTC/USDT 1h PPO signal

- As of bar: `2026-09-05 05:00:00+08:00` (close 79,624.00)
- Generated: `2026-09-05T22:03:13`
- Model: `models/saved/BTCUSDT_signal_layered_1h_v3/best_model.zip`
- Replay anchor: `2026-04-13 00:00:00+08:00` (3,485 bars replayed)

## Decision

| | |
|---|---|
| **Target position** | **25%** — 维持 25% |
| Current position (replayed book) | 25% |
| Unrealised P&L | -0.21% |
| Action lock | 🔒 min_hold / cooldown active |
| Steps since last switch | 1 |
| Steps since stop | 96 |
| Action probs (0/25/50/75/100%) | 0.00 / 1.00 / 0.00 / 0.00 / 0.00 |

## Signals now (by |value|)

| Signal | Value | Read |
|--------|------:|------|
| `sig_mtf_4h_trend` | +1.000 | ↑ bullish |
| `sig_vol_bb_width` | +0.620 | ↑ bullish |
| `sig_mtf_1d_regime` | +0.587 | ↑ bullish |
| `sig_vol_atr_pct` | +0.540 | ↑ bullish |
| `sig_regime_return_60` | +0.439 | ↑ bullish |
| `sig_vol_bb_position` | -0.408 | ↓ bearish |
| `sig_trend_macd_hist` | -0.320 | ↓ bearish |
| `sig_trend_ema_cross` | -0.274 | ↓ bearish |

## Last 24 bars

| Bar | Close | Executed | Target | Position | Stop |
|-----|------:|---------:|-------:|---------:|:----:|
| 2026-09-04 05:00 | 81,590 | 3 | 75% | 75% |  |
| 2026-09-04 06:00 | 81,140 | 3 | 75% | 75% |  |
| 2026-09-04 07:00 | 81,270 | 3 | 75% | 75% |  |
| 2026-09-04 08:00 | 80,972 | 3 | 75% | 75% |  |
| 2026-09-04 09:00 | 80,968 | 3 | 75% | 75% |  |
| 2026-09-04 10:00 | 80,832 | 3 | 75% | 75% |  |
| 2026-09-04 11:00 | 80,850 | 3 | 75% | 75% |  |
| 2026-09-04 12:00 | 81,071 | 3 | 75% | 75% |  |
| 2026-09-04 13:00 | 80,996 | 3 | 75% | 75% |  |
| 2026-09-04 14:00 | 80,797 | 3 | 75% | 75% |  |
| 2026-09-04 15:00 | 80,657 | 3 | 75% | 75% |  |
| 2026-09-04 16:00 | 81,147 | 3 | 75% | 75% |  |
| 2026-09-04 17:00 | 81,057 | 0 | 0% | 0% |  |
| 2026-09-04 18:00 | 81,180 | 0 | 0% | 0% |  |
| 2026-09-04 19:00 | 81,225 | 0 | 0% | 0% |  |
| 2026-09-04 20:00 | 79,454 | 0 | 0% | 0% |  |
| 2026-09-04 21:00 | 79,379 | 0 | 0% | 0% |  |
| 2026-09-04 22:00 | 78,916 | 0 | 0% | 0% |  |
| 2026-09-04 23:00 | 79,429 | 0 | 0% | 0% |  |
| 2026-09-05 00:00 | 79,746 | 0 | 0% | 0% |  |
| 2026-09-05 01:00 | 79,472 | 0 | 0% | 0% |  |
| 2026-09-05 02:00 | 79,618 | 0 | 0% | 0% |  |
| 2026-09-05 03:00 | 79,793 | 1 | 25% | 25% |  |
| 2026-09-05 04:00 | 79,697 | 1 | 25% | 25% |  |

## Replay since anchor (2026-04-13 → 2026-09-05)

| Total return | Max DD | Sharpe (ann.) | Trades | Stops |
|-------------:|-------:|--------------:|-------:|------:|
| -20.21% | -27.62% | -2.858 | 393 | 14 |

> Positions are *targets* for a spot long-only book; the replayed book starts flat at the anchor with the config's initial balance.