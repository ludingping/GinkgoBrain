# Pooled Multi-Asset Linear Baseline

- Generated: `2026-09-06T10:51:51`  
- Signals config: `config/signals_4h_probe19.yaml` (19 features)  
- Timeframe: **4h**, horizon **k = 6 bars**, model **logreg** on standardised features  
- Assets: BTC/USDT, ETH/USDT, BNB/USDT, SOL/USDT; target: **BTC/USDT**  
- Bars after warmup: BTC/USDT 12,291, ETH/USDT 12,291, BNB/USDT 12,291, SOL/USDT 12,291  
- CV: 3 expanding time folds, embargo = 6 bars

## AUC by arm (mean over folds)

| eval on | single (own history) | pooled (all assets) | transfer (others only) | Δ pooled − single |
|---|---|---|---|---|
| BTC/USDT | 0.5235 | 0.5271 | 0.5248 | +0.0036 |
| ETH/USDT | 0.5288 | 0.5271 | 0.5254 | -0.0017 |
| BNB/USDT | 0.5181 | 0.5226 | 0.5226 | +0.0045 |
| SOL/USDT | 0.5179 | 0.5142 | 0.5107 | -0.0037 |
| ALL | — | 0.5222 | — | — |

## Per-fold detail

| fold | val window | eval on | arm | n_train | n_val | AUC |
|---|---|---|---|---|---|---|
| 1 | 2022-06-20 → 2023-11-14 | BTC/USDT | single | 3,069 | 3,072 | 0.5363 |
| 1 | 2022-06-20 → 2023-11-14 | BTC/USDT | pooled | 12,276 | 3,072 | 0.5426 |
| 1 | 2022-06-20 → 2023-11-14 | BTC/USDT | transfer | 9,207 | 3,072 | 0.5386 |
| 1 | 2022-06-20 → 2023-11-14 | ETH/USDT | single | 3,069 | 3,072 | 0.5470 |
| 1 | 2022-06-20 → 2023-11-14 | ETH/USDT | pooled | 12,276 | 3,072 | 0.5587 |
| 1 | 2022-06-20 → 2023-11-14 | ETH/USDT | transfer | 9,207 | 3,072 | 0.5584 |
| 1 | 2022-06-20 → 2023-11-14 | BNB/USDT | single | 3,069 | 3,072 | 0.5220 |
| 1 | 2022-06-20 → 2023-11-14 | BNB/USDT | pooled | 12,276 | 3,072 | 0.5387 |
| 1 | 2022-06-20 → 2023-11-14 | BNB/USDT | transfer | 9,207 | 3,072 | 0.5421 |
| 1 | 2022-06-20 → 2023-11-14 | SOL/USDT | single | 3,069 | 3,072 | 0.5243 |
| 1 | 2022-06-20 → 2023-11-14 | SOL/USDT | pooled | 12,276 | 3,072 | 0.5110 |
| 1 | 2022-06-20 → 2023-11-14 | SOL/USDT | transfer | 9,207 | 3,072 | 0.5010 |
| 1 | 2022-06-20 → 2023-11-14 | ALL | pooled | 12,276 | 12,288 | 0.5374 |
| 2 | 2023-11-14 → 2025-04-09 | BTC/USDT | single | 6,141 | 3,072 | 0.5308 |
| 2 | 2023-11-14 → 2025-04-09 | BTC/USDT | pooled | 24,564 | 3,072 | 0.5387 |
| 2 | 2023-11-14 → 2025-04-09 | BTC/USDT | transfer | 18,423 | 3,072 | 0.5364 |
| 2 | 2023-11-14 → 2025-04-09 | ETH/USDT | single | 6,141 | 3,072 | 0.5401 |
| 2 | 2023-11-14 → 2025-04-09 | ETH/USDT | pooled | 24,564 | 3,072 | 0.5355 |
| 2 | 2023-11-14 → 2025-04-09 | ETH/USDT | transfer | 18,423 | 3,072 | 0.5337 |
| 2 | 2023-11-14 → 2025-04-09 | BNB/USDT | single | 6,141 | 3,072 | 0.5315 |
| 2 | 2023-11-14 → 2025-04-09 | BNB/USDT | pooled | 24,564 | 3,072 | 0.5401 |
| 2 | 2023-11-14 → 2025-04-09 | BNB/USDT | transfer | 18,423 | 3,072 | 0.5404 |
| 2 | 2023-11-14 → 2025-04-09 | SOL/USDT | single | 6,141 | 3,072 | 0.5380 |
| 2 | 2023-11-14 → 2025-04-09 | SOL/USDT | pooled | 24,564 | 3,072 | 0.5365 |
| 2 | 2023-11-14 → 2025-04-09 | SOL/USDT | transfer | 18,423 | 3,072 | 0.5333 |
| 2 | 2023-11-14 → 2025-04-09 | ALL | pooled | 24,564 | 12,288 | 0.5370 |
| 3 | 2025-04-09 → 2026-09-03 | BTC/USDT | single | 9,213 | 3,072 | 0.5036 |
| 3 | 2025-04-09 → 2026-09-03 | BTC/USDT | pooled | 36,852 | 3,072 | 0.5000 |
| 3 | 2025-04-09 → 2026-09-03 | BTC/USDT | transfer | 27,639 | 3,072 | 0.4995 |
| 3 | 2025-04-09 → 2026-09-03 | ETH/USDT | single | 9,213 | 3,072 | 0.4992 |
| 3 | 2025-04-09 → 2026-09-03 | ETH/USDT | pooled | 36,852 | 3,072 | 0.4869 |
| 3 | 2025-04-09 → 2026-09-03 | ETH/USDT | transfer | 27,639 | 3,072 | 0.4842 |
| 3 | 2025-04-09 → 2026-09-03 | BNB/USDT | single | 9,213 | 3,072 | 0.5008 |
| 3 | 2025-04-09 → 2026-09-03 | BNB/USDT | pooled | 36,852 | 3,072 | 0.4889 |
| 3 | 2025-04-09 → 2026-09-03 | BNB/USDT | transfer | 27,639 | 3,072 | 0.4854 |
| 3 | 2025-04-09 → 2026-09-03 | SOL/USDT | single | 9,213 | 3,072 | 0.4913 |
| 3 | 2025-04-09 → 2026-09-03 | SOL/USDT | pooled | 36,852 | 3,072 | 0.4950 |
| 3 | 2025-04-09 → 2026-09-03 | SOL/USDT | transfer | 27,639 | 3,072 | 0.4978 |
| 3 | 2025-04-09 → 2026-09-03 | ALL | pooled | 36,852 | 12,288 | 0.4923 |
