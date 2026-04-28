# Linear Baseline Report — TC-A6

- Generated: `2026-04-26T11:33:25`  
- Signals config: `config/signals_v1_5min_eth.yaml` (19 features)  
- Data source: `config:stage2_5min_eth_signal.yaml (tf=5min)`  
- Target horizon: **k = 288 bars**  
- CV: TimeSeriesSplit(3 folds, no shuffle)  
- Iron gate: AUC mean > **0.52** AND min > **0.5**  
- R² diagnostic: mean > **-0.1** (catastrophe check, not a positive-predictive gate)

## Verdict: ❌ FAIL — Phase 2 frozen

| Metric | Rule | Min | Mean | Max | Pass |
|--------|------|-----|------|-----|------|
| ROC-AUC (LogReg, iron gate) | mean > 0.52 AND min > 0.5 | 0.4996 | 0.5146 | 0.5307 | ❌ |
| R² (LinReg, diagnostic) | mean > -0.1 | -0.0278 | -0.0084 | +0.0026 | ✅ |

## Per-fold metrics

| Fold | n_train | n_val | ROC-AUC | R² |
|------|---------|-------|---------|-----|
| 1 | 138,632 | 138,631 | 0.4996 | -0.0278 |
| 2 | 277,263 | 138,631 | 0.5307 | +0.0026 |
| 3 | 415,894 | 138,631 | 0.5133 | -0.0001 |

## Diagnostics (last fold)

![confusion matrix](signal_linear_baseline_5min_artifacts/confusion_matrix.png)

![residuals](signal_linear_baseline_5min_artifacts/residuals.png)

## Feature weights (last fold)

Positive = bullish contribution; negative = bearish. Standardized features.

![coefficients](signal_linear_baseline_5min_artifacts/coefficients.png)

## Next steps (iron-gate failure)

- Return to §4 signal design — not a hyperparameter issue.
- Check whether any sig_* is degenerate post-warmup (flat, near-zero std).
- Consider adding non-linear transformations that linear models miss (interaction terms, regime-conditioned features).
- Do NOT proceed to Phase 2 PPO training until this gate passes.
