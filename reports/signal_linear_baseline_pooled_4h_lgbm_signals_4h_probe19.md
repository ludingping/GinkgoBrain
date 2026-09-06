# Pooled Multi-Asset Linear Baseline

- Generated: `2026-09-06T10:52:19`  
- Signals config: `config/signals_4h_probe19.yaml` (19 features)  
- Timeframe: **4h**, horizon **k = 6 bars**, model **lgbm** on standardised features  
- Assets: BTC/USDT, ETH/USDT, BNB/USDT, SOL/USDT; target: **BTC/USDT**  
- Bars after warmup: BTC/USDT 12,291, ETH/USDT 12,291, BNB/USDT 12,291, SOL/USDT 12,291  
- CV: 3 expanding time folds, embargo = 6 bars

## AUC by arm (mean over folds)

| eval on | single (own history) | pooled (all assets) | transfer (others only) | Δ pooled − single |
|---|---|---|---|---|
| BTC/USDT | 0.5148 | 0.5260 | 0.5275 | +0.0112 |
| ETH/USDT | 0.5330 | 0.5410 | 0.5402 | +0.0080 |
| BNB/USDT | 0.5232 | 0.5251 | 0.5208 | +0.0020 |
| SOL/USDT | 0.5274 | 0.5249 | 0.5207 | -0.0026 |
| ALL | — | 0.5288 | — | — |

## Per-fold detail

| fold | val window | eval on | arm | n_train | n_val | AUC |
|---|---|---|---|---|---|---|
| 1 | 2022-06-20 → 2023-11-14 | BTC/USDT | single | 3,069 | 3,072 | 0.5037 |
| 1 | 2022-06-20 → 2023-11-14 | BTC/USDT | pooled | 12,276 | 3,072 | 0.5206 |
| 1 | 2022-06-20 → 2023-11-14 | BTC/USDT | transfer | 9,207 | 3,072 | 0.5243 |
| 1 | 2022-06-20 → 2023-11-14 | ETH/USDT | single | 3,069 | 3,072 | 0.5388 |
| 1 | 2022-06-20 → 2023-11-14 | ETH/USDT | pooled | 12,276 | 3,072 | 0.5573 |
| 1 | 2022-06-20 → 2023-11-14 | ETH/USDT | transfer | 9,207 | 3,072 | 0.5571 |
| 1 | 2022-06-20 → 2023-11-14 | BNB/USDT | single | 3,069 | 3,072 | 0.5263 |
| 1 | 2022-06-20 → 2023-11-14 | BNB/USDT | pooled | 12,276 | 3,072 | 0.5520 |
| 1 | 2022-06-20 → 2023-11-14 | BNB/USDT | transfer | 9,207 | 3,072 | 0.5498 |
| 1 | 2022-06-20 → 2023-11-14 | SOL/USDT | single | 3,069 | 3,072 | 0.5426 |
| 1 | 2022-06-20 → 2023-11-14 | SOL/USDT | pooled | 12,276 | 3,072 | 0.5288 |
| 1 | 2022-06-20 → 2023-11-14 | SOL/USDT | transfer | 9,207 | 3,072 | 0.5223 |
| 1 | 2022-06-20 → 2023-11-14 | ALL | pooled | 12,276 | 12,288 | 0.5388 |
| 2 | 2023-11-14 → 2025-04-09 | BTC/USDT | single | 6,141 | 3,072 | 0.5259 |
| 2 | 2023-11-14 → 2025-04-09 | BTC/USDT | pooled | 24,564 | 3,072 | 0.5478 |
| 2 | 2023-11-14 → 2025-04-09 | BTC/USDT | transfer | 18,423 | 3,072 | 0.5547 |
| 2 | 2023-11-14 → 2025-04-09 | ETH/USDT | single | 6,141 | 3,072 | 0.5378 |
| 2 | 2023-11-14 → 2025-04-09 | ETH/USDT | pooled | 24,564 | 3,072 | 0.5468 |
| 2 | 2023-11-14 → 2025-04-09 | ETH/USDT | transfer | 18,423 | 3,072 | 0.5485 |
| 2 | 2023-11-14 → 2025-04-09 | BNB/USDT | single | 6,141 | 3,072 | 0.5242 |
| 2 | 2023-11-14 → 2025-04-09 | BNB/USDT | pooled | 24,564 | 3,072 | 0.5262 |
| 2 | 2023-11-14 → 2025-04-09 | BNB/USDT | transfer | 18,423 | 3,072 | 0.5206 |
| 2 | 2023-11-14 → 2025-04-09 | SOL/USDT | single | 6,141 | 3,072 | 0.5404 |
| 2 | 2023-11-14 → 2025-04-09 | SOL/USDT | pooled | 24,564 | 3,072 | 0.5318 |
| 2 | 2023-11-14 → 2025-04-09 | SOL/USDT | transfer | 18,423 | 3,072 | 0.5245 |
| 2 | 2023-11-14 → 2025-04-09 | ALL | pooled | 24,564 | 12,288 | 0.5379 |
| 3 | 2025-04-09 → 2026-09-03 | BTC/USDT | single | 9,213 | 3,072 | 0.5149 |
| 3 | 2025-04-09 → 2026-09-03 | BTC/USDT | pooled | 36,852 | 3,072 | 0.5096 |
| 3 | 2025-04-09 → 2026-09-03 | BTC/USDT | transfer | 27,639 | 3,072 | 0.5035 |
| 3 | 2025-04-09 → 2026-09-03 | ETH/USDT | single | 9,213 | 3,072 | 0.5222 |
| 3 | 2025-04-09 → 2026-09-03 | ETH/USDT | pooled | 36,852 | 3,072 | 0.5187 |
| 3 | 2025-04-09 → 2026-09-03 | ETH/USDT | transfer | 27,639 | 3,072 | 0.5149 |
| 3 | 2025-04-09 → 2026-09-03 | BNB/USDT | single | 9,213 | 3,072 | 0.5190 |
| 3 | 2025-04-09 → 2026-09-03 | BNB/USDT | pooled | 36,852 | 3,072 | 0.4972 |
| 3 | 2025-04-09 → 2026-09-03 | BNB/USDT | transfer | 27,639 | 3,072 | 0.4921 |
| 3 | 2025-04-09 → 2026-09-03 | SOL/USDT | single | 9,213 | 3,072 | 0.4994 |
| 3 | 2025-04-09 → 2026-09-03 | SOL/USDT | pooled | 36,852 | 3,072 | 0.5141 |
| 3 | 2025-04-09 → 2026-09-03 | SOL/USDT | transfer | 27,639 | 3,072 | 0.5154 |
| 3 | 2025-04-09 → 2026-09-03 | ALL | pooled | 36,852 | 12,288 | 0.5096 |
