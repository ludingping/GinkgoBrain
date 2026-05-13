# v0.1 PRELIMINARY Verdict — NO-ALPHA

**run dir**: `/home/davidlyu/projects/GinkgoBrain/reports/v0.1_20260513_222337`
**verdict notes**: {'ic_a_pass': False, 'reason': 'ic_a_failed'}
**extra notes**: {'bm1_unavailable': True, 'dual_signal_status': 'no_alpha', 'n_trade_dates': 267, 'final_equity': 14584944.591867475, 'cumulative_return': 0.45849445918674747, 'n_ic_observations': 263}

## Metrics
- IC_A mean=-0.0048, t-stat=-0.98, n=239
- IC_B mean=nan, t-stat=nan, n=0
- cum return: 45.85%
- annualized: 42.98%
- excess vs BM1 (fallback vs 0% cash): 42.98%
- excess vs BM2 (等权 universe): -12.76%
- max drawdown: -9.15%
- sharpe (扣成本): 2.16
- IR vs BM2: -0.36
- dual signal overlap (A vs B Top-N): 6.3%

## 后续
- 若 verdict ∈ {PASS-prelim} → 数据攒到 ≥18 月后 v0.1.5 复跑（前 ⅔ IS + 后 ⅓ OOS）
- 若 verdict ∈ {CHURNING, NO-ALPHA, SIZE-BETA-ONLY} → 关闭策略 + 归档 lesson