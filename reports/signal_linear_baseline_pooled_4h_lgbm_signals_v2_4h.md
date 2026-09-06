# Pooled Multi-Asset Linear Baseline

- Generated: `2026-09-06T10:52:04`  
- Signals config: `config/signals_v2_4h.yaml` (10 features)  
- Timeframe: **4h**, horizon **k = 6 bars**, model **lgbm** on standardised features  
- Assets: BTC/USDT, ETH/USDT, BNB/USDT, SOL/USDT; target: **BTC/USDT**  
- Bars after warmup: BTC/USDT 12,291, ETH/USDT 12,291, BNB/USDT 12,291, SOL/USDT 12,291  
- CV: 3 expanding time folds, embargo = 6 bars

## AUC by arm (mean over folds)

| eval on | single (own history) | pooled (all assets) | transfer (others only) | Δ pooled − single |
|---|---|---|---|---|
| BTC/USDT | 0.5138 | 0.5244 | 0.5286 | +0.0105 |
| ETH/USDT | 0.5296 | 0.5336 | 0.5322 | +0.0041 |
| BNB/USDT | 0.5179 | 0.5225 | 0.5171 | +0.0047 |
| SOL/USDT | 0.5215 | 0.5207 | 0.5144 | -0.0008 |
| ALL | — | 0.5249 | — | — |

## Per-fold detail

| fold | val window | eval on | arm | n_train | n_val | AUC |
|---|---|---|---|---|---|---|
| 1 | 2022-06-20 → 2023-11-14 | BTC/USDT | single | 3,069 | 3,072 | 0.5056 |
| 1 | 2022-06-20 → 2023-11-14 | BTC/USDT | pooled | 12,276 | 3,072 | 0.5244 |
| 1 | 2022-06-20 → 2023-11-14 | BTC/USDT | transfer | 9,207 | 3,072 | 0.5313 |
| 1 | 2022-06-20 → 2023-11-14 | ETH/USDT | single | 3,069 | 3,072 | 0.5379 |
| 1 | 2022-06-20 → 2023-11-14 | ETH/USDT | pooled | 12,276 | 3,072 | 0.5417 |
| 1 | 2022-06-20 → 2023-11-14 | ETH/USDT | transfer | 9,207 | 3,072 | 0.5406 |
| 1 | 2022-06-20 → 2023-11-14 | BNB/USDT | single | 3,069 | 3,072 | 0.5178 |
| 1 | 2022-06-20 → 2023-11-14 | BNB/USDT | pooled | 12,276 | 3,072 | 0.5498 |
| 1 | 2022-06-20 → 2023-11-14 | BNB/USDT | transfer | 9,207 | 3,072 | 0.5485 |
| 1 | 2022-06-20 → 2023-11-14 | SOL/USDT | single | 3,069 | 3,072 | 0.5329 |
| 1 | 2022-06-20 → 2023-11-14 | SOL/USDT | pooled | 12,276 | 3,072 | 0.5294 |
| 1 | 2022-06-20 → 2023-11-14 | SOL/USDT | transfer | 9,207 | 3,072 | 0.5196 |
| 1 | 2022-06-20 → 2023-11-14 | ALL | pooled | 12,276 | 12,288 | 0.5354 |
| 2 | 2023-11-14 → 2025-04-09 | BTC/USDT | single | 6,141 | 3,072 | 0.5241 |
| 2 | 2023-11-14 → 2025-04-09 | BTC/USDT | pooled | 24,564 | 3,072 | 0.5413 |
| 2 | 2023-11-14 → 2025-04-09 | BTC/USDT | transfer | 18,423 | 3,072 | 0.5503 |
| 2 | 2023-11-14 → 2025-04-09 | ETH/USDT | single | 6,141 | 3,072 | 0.5272 |
| 2 | 2023-11-14 → 2025-04-09 | ETH/USDT | pooled | 24,564 | 3,072 | 0.5381 |
| 2 | 2023-11-14 → 2025-04-09 | ETH/USDT | transfer | 18,423 | 3,072 | 0.5368 |
| 2 | 2023-11-14 → 2025-04-09 | BNB/USDT | single | 6,141 | 3,072 | 0.5199 |
| 2 | 2023-11-14 → 2025-04-09 | BNB/USDT | pooled | 24,564 | 3,072 | 0.5216 |
| 2 | 2023-11-14 → 2025-04-09 | BNB/USDT | transfer | 18,423 | 3,072 | 0.5155 |
| 2 | 2023-11-14 → 2025-04-09 | SOL/USDT | single | 6,141 | 3,072 | 0.5345 |
| 2 | 2023-11-14 → 2025-04-09 | SOL/USDT | pooled | 24,564 | 3,072 | 0.5292 |
| 2 | 2023-11-14 → 2025-04-09 | SOL/USDT | transfer | 18,423 | 3,072 | 0.5197 |
| 2 | 2023-11-14 → 2025-04-09 | ALL | pooled | 24,564 | 12,288 | 0.5324 |
| 3 | 2025-04-09 → 2026-09-03 | BTC/USDT | single | 9,213 | 3,072 | 0.5118 |
| 3 | 2025-04-09 → 2026-09-03 | BTC/USDT | pooled | 36,852 | 3,072 | 0.5075 |
| 3 | 2025-04-09 → 2026-09-03 | BTC/USDT | transfer | 27,639 | 3,072 | 0.5041 |
| 3 | 2025-04-09 → 2026-09-03 | ETH/USDT | single | 9,213 | 3,072 | 0.5236 |
| 3 | 2025-04-09 → 2026-09-03 | ETH/USDT | pooled | 36,852 | 3,072 | 0.5212 |
| 3 | 2025-04-09 → 2026-09-03 | ETH/USDT | transfer | 27,639 | 3,072 | 0.5191 |
| 3 | 2025-04-09 → 2026-09-03 | BNB/USDT | single | 9,213 | 3,072 | 0.5159 |
| 3 | 2025-04-09 → 2026-09-03 | BNB/USDT | pooled | 36,852 | 3,072 | 0.4962 |
| 3 | 2025-04-09 → 2026-09-03 | BNB/USDT | transfer | 27,639 | 3,072 | 0.4873 |
| 3 | 2025-04-09 → 2026-09-03 | SOL/USDT | single | 9,213 | 3,072 | 0.4971 |
| 3 | 2025-04-09 → 2026-09-03 | SOL/USDT | pooled | 36,852 | 3,072 | 0.5036 |
| 3 | 2025-04-09 → 2026-09-03 | SOL/USDT | transfer | 27,639 | 3,072 | 0.5040 |
| 3 | 2025-04-09 → 2026-09-03 | ALL | pooled | 36,852 | 12,288 | 0.5068 |
