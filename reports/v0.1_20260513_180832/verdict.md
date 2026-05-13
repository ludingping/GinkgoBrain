# v0.1 PRELIMINARY Verdict — NO-ALPHA

**run dir**: `/home/davidlyu/projects/GinkgoBrain/reports/v0.1_20260513_180832`
**verdict notes**: {'ic_a_pass': False, 'reason': 'ic_a_failed'}
**extra notes**: {'bm1_unavailable': True, 'dual_signal_status': 'no_alpha', 'n_trade_dates': 267, 'final_equity': 10511712.781634836, 'cumulative_return': 0.05117127816348366, 'n_ic_observations': 263}

## Metrics
- IC_A mean=0.0017, t-stat=0.23, n=263
- IC_B mean=-0.0034, t-stat=-0.45, n=263
- cum return: 5.12%
- annualized: 4.84%
- excess vs BM1 (fallback vs 0% cash): 4.84%
- excess vs BM2 (等权 universe): -40.58%
- max drawdown: -20.42%
- sharpe (扣成本): 0.31
- IR vs BM2: -1.60
- dual signal overlap (A vs B Top-N): 13.6%

## 后续
- 若 verdict ∈ {PASS-prelim} → 数据攒到 ≥18 月后 v0.1.5 复跑（前 ⅔ IS + 后 ⅓ OOS）
- 若 verdict ∈ {CHURNING, NO-ALPHA, SIZE-BETA-ONLY} → 关闭策略 + 归档 lesson