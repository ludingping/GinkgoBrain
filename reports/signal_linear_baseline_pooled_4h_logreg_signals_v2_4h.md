# Pooled Multi-Asset Linear Baseline

- Generated: `2026-09-06T10:51:33`  
- Signals config: `config/signals_v2_4h.yaml` (10 features)  
- Timeframe: **4h**, horizon **k = 6 bars**, model **logreg** on standardised features  
- Assets: BTC/USDT, ETH/USDT, BNB/USDT, SOL/USDT; target: **BTC/USDT**  
- Bars after warmup: BTC/USDT 12,291, ETH/USDT 12,291, BNB/USDT 12,291, SOL/USDT 12,291  
- CV: 3 expanding time folds, embargo = 6 bars

## AUC by arm (mean over folds)

| eval on | single (own history) | pooled (all assets) | transfer (others only) | Δ pooled − single |
|---|---|---|---|---|
| BTC/USDT | 0.5256 | 0.5287 | 0.5273 | +0.0031 |
| ETH/USDT | 0.5265 | 0.5297 | 0.5295 | +0.0032 |
| BNB/USDT | 0.5275 | 0.5256 | 0.5249 | -0.0019 |
| SOL/USDT | 0.5158 | 0.5087 | 0.5064 | -0.0071 |
| ALL | — | 0.5228 | — | — |

## Per-fold detail

| fold | val window | eval on | arm | n_train | n_val | AUC |
|---|---|---|---|---|---|---|
| 1 | 2022-06-20 → 2023-11-14 | BTC/USDT | single | 3,069 | 3,072 | 0.5386 |
| 1 | 2022-06-20 → 2023-11-14 | BTC/USDT | pooled | 12,276 | 3,072 | 0.5365 |
| 1 | 2022-06-20 → 2023-11-14 | BTC/USDT | transfer | 9,207 | 3,072 | 0.5329 |
| 1 | 2022-06-20 → 2023-11-14 | ETH/USDT | single | 3,069 | 3,072 | 0.5304 |
| 1 | 2022-06-20 → 2023-11-14 | ETH/USDT | pooled | 12,276 | 3,072 | 0.5456 |
| 1 | 2022-06-20 → 2023-11-14 | ETH/USDT | transfer | 9,207 | 3,072 | 0.5496 |
| 1 | 2022-06-20 → 2023-11-14 | BNB/USDT | single | 3,069 | 3,072 | 0.5474 |
| 1 | 2022-06-20 → 2023-11-14 | BNB/USDT | pooled | 12,276 | 3,072 | 0.5440 |
| 1 | 2022-06-20 → 2023-11-14 | BNB/USDT | transfer | 9,207 | 3,072 | 0.5412 |
| 1 | 2022-06-20 → 2023-11-14 | SOL/USDT | single | 3,069 | 3,072 | 0.5272 |
| 1 | 2022-06-20 → 2023-11-14 | SOL/USDT | pooled | 12,276 | 3,072 | 0.5107 |
| 1 | 2022-06-20 → 2023-11-14 | SOL/USDT | transfer | 9,207 | 3,072 | 0.5037 |
| 1 | 2022-06-20 → 2023-11-14 | ALL | pooled | 12,276 | 12,288 | 0.5340 |
| 2 | 2023-11-14 → 2025-04-09 | BTC/USDT | single | 6,141 | 3,072 | 0.5285 |
| 2 | 2023-11-14 → 2025-04-09 | BTC/USDT | pooled | 24,564 | 3,072 | 0.5437 |
| 2 | 2023-11-14 → 2025-04-09 | BTC/USDT | transfer | 18,423 | 3,072 | 0.5443 |
| 2 | 2023-11-14 → 2025-04-09 | ETH/USDT | single | 6,141 | 3,072 | 0.5459 |
| 2 | 2023-11-14 → 2025-04-09 | ETH/USDT | pooled | 24,564 | 3,072 | 0.5429 |
| 2 | 2023-11-14 → 2025-04-09 | ETH/USDT | transfer | 18,423 | 3,072 | 0.5395 |
| 2 | 2023-11-14 → 2025-04-09 | BNB/USDT | single | 6,141 | 3,072 | 0.5242 |
| 2 | 2023-11-14 → 2025-04-09 | BNB/USDT | pooled | 24,564 | 3,072 | 0.5291 |
| 2 | 2023-11-14 → 2025-04-09 | BNB/USDT | transfer | 18,423 | 3,072 | 0.5316 |
| 2 | 2023-11-14 → 2025-04-09 | SOL/USDT | single | 6,141 | 3,072 | 0.5203 |
| 2 | 2023-11-14 → 2025-04-09 | SOL/USDT | pooled | 24,564 | 3,072 | 0.5149 |
| 2 | 2023-11-14 → 2025-04-09 | SOL/USDT | transfer | 18,423 | 3,072 | 0.5125 |
| 2 | 2023-11-14 → 2025-04-09 | ALL | pooled | 24,564 | 12,288 | 0.5320 |
| 3 | 2025-04-09 → 2026-09-03 | BTC/USDT | single | 9,213 | 3,072 | 0.5096 |
| 3 | 2025-04-09 → 2026-09-03 | BTC/USDT | pooled | 36,852 | 3,072 | 0.5059 |
| 3 | 2025-04-09 → 2026-09-03 | BTC/USDT | transfer | 27,639 | 3,072 | 0.5048 |
| 3 | 2025-04-09 → 2026-09-03 | ETH/USDT | single | 9,213 | 3,072 | 0.5032 |
| 3 | 2025-04-09 → 2026-09-03 | ETH/USDT | pooled | 36,852 | 3,072 | 0.5006 |
| 3 | 2025-04-09 → 2026-09-03 | ETH/USDT | transfer | 27,639 | 3,072 | 0.4993 |
| 3 | 2025-04-09 → 2026-09-03 | BNB/USDT | single | 9,213 | 3,072 | 0.5110 |
| 3 | 2025-04-09 → 2026-09-03 | BNB/USDT | pooled | 36,852 | 3,072 | 0.5038 |
| 3 | 2025-04-09 → 2026-09-03 | BNB/USDT | transfer | 27,639 | 3,072 | 0.5019 |
| 3 | 2025-04-09 → 2026-09-03 | SOL/USDT | single | 9,213 | 3,072 | 0.4998 |
| 3 | 2025-04-09 → 2026-09-03 | SOL/USDT | pooled | 36,852 | 3,072 | 0.5004 |
| 3 | 2025-04-09 → 2026-09-03 | SOL/USDT | transfer | 27,639 | 3,072 | 0.5030 |
| 3 | 2025-04-09 → 2026-09-03 | ALL | pooled | 36,852 | 12,288 | 0.5022 |
