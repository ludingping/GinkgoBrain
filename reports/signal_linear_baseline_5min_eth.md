# Linear Baseline Report — TC-A6

- Generated: `2026-04-22T15:20:20`  
- Signals config: `config/signals_v1_1h.yaml` (15 features)  
- Data source: `config:stage2_5min_eth_signal.yaml (tf=5min)`  
- Target horizon: **k = 288 bars**  
- CV: TimeSeriesSplit(3 folds, no shuffle)  
- Iron gate: AUC mean > **0.52** AND min > **0.5**  
- R² diagnostic: mean > **-0.1** (catastrophe check, not a positive-predictive gate)

## Verdict: ✅ PASS

| Metric | Rule | Min | Mean | Max | Pass |
|--------|------|-----|------|-----|------|
| ROC-AUC (LogReg, iron gate) | mean > 0.52 AND min > 0.5 | 0.5214 | 0.5343 | 0.5481 | ✅ |
| R² (LinReg, diagnostic) | mean > -0.1 | -0.0126 | +0.0050 | +0.0160 | ✅ |

## Per-fold metrics

| Fold | n_train | n_val | ROC-AUC | R² |
|------|---------|-------|---------|-----|
| 1 | 138,632 | 138,631 | 0.5214 | -0.0126 |
| 2 | 277,263 | 138,631 | 0.5481 | +0.0160 |
| 3 | 415,894 | 138,631 | 0.5335 | +0.0114 |

## Diagnostics (last fold)

![confusion matrix](signal_linear_baseline_5min_artifacts/confusion_matrix.png)

![residuals](signal_linear_baseline_5min_artifacts/residuals.png)

## Feature weights (last fold)

Positive = bullish contribution; negative = bearish. Standardized features.

![coefficients](signal_linear_baseline_5min_artifacts/coefficients.png)
