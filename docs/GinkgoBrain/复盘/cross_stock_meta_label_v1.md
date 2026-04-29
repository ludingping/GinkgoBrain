# A 股多股票 meta-label 横向稳健性 v1

**生成时间**: 2026-04-29 15:03
**窗口**: 2023-04-04 → 2026-04-22
**股票数**: 7

## 方法
对每只股票：
1. 加载日线（已前复权）
2. 计算 5 个 day 级 indicator regime（vol / trend / drawdown / volume_z / RSI）
3. 事件去重后跑 `regime_score(h ∈ {1,5,21})` → 生成 event scoreboard
4. 按 |t|≥2 + (n, t) 分级 → 生成 per-stock 信号库

横向聚合：把每只股票的 scoreboard 按 `(indicator, regime, h)` 拼成 hypothesis 矩阵，
统计每个 hypothesis 在多少只股票上同向 |t|≥2，分四档：

| Tier | 标准 |
|------|------|
| A | ≥3/4 stocks 同向 \|t\|≥2 |
| B | 2/4 stocks 同向 \|t\|≥2 |
| C | 仅 1 只股票 \|t\|≥2 |
| F | 0 只 OR 不同股票方向反转 |

## 数据覆盖

| 股票 | 名称 | 交易日数 | 候选信号数 |
|------|------|---------|------------|
| 300866 | 安克创新 | 736 | 8 |
| 002594 | 比亚迪 | 736 | 4 |
| 600765 | 中航重机 | 736 | 3 |
| 600893 | 航发动力 | 736 | 7 |
| 300274 | ? | 736 | 6 |
| 300346 | ? | 736 | 3 |
| 300760 | ? | 736 | 9 |

## Robustness Tier 分布

- **A (≥3/4 same dir |t|≥2)**: 1
- **B (2/4 same dir |t|≥2)**: 7
- **C (single-stock only)**: 13
- **F (mixed)**: 5
- **F (none |t|≥2)**: 34

## Tier A + B（横向稳健）

```
                hypothesis_id    indicator     regime  h  t_300866  t_002594  t_600765  t_600893  t_300274  t_300346  t_300760  max_pass              robustness verdict_dir
           vol_21d::低_vol::h1      vol_21d      低 vol  1      1.61      0.24      2.44     -0.60      2.71     -1.47      2.15         3 A (≥3/4 same dir |t|≥2)        long
volume_z_21d::D_偏放0_5_1σ::h21 volume_z_21d D.偏放0.5~1σ 21      3.27     -0.04     -0.33      1.76      2.22      1.13     -0.81         2  B (2/4 same dir |t|≥2)        long
          rsi_14::A_超卖_30::h5       rsi_14    A.超卖<30  5      0.69      0.21      0.59      2.23      1.32     -1.91      3.23         2  B (2/4 same dir |t|≥2)        long
         trend::4_弱势_下_跌_::h5        trend  4.弱势(下+跌)  5      1.10      0.80     -2.35     -3.03     -0.46     -0.83     -0.61         2  B (2/4 same dir |t|≥2)       short
        trend::3_反弹_下_涨_::h21        trend  3.反弹(下+涨) 21     -2.92      1.15     -2.04      0.49     -1.43      0.53     -0.84         2  B (2/4 same dir |t|≥2)       short
         rsi_14::D_超买_70::h21       rsi_14    D.超买>70 21      2.47     -0.90      0.40      0.49      2.37     -1.19       NaN         2  B (2/4 same dir |t|≥2)        long
         rsi_14::A_超卖_30::h21       rsi_14    A.超卖<30 21      2.30      1.42      0.08      2.31      1.98      0.37      0.31         2  B (2/4 same dir |t|≥2)        long
   volume_z_21d::E_放量_1σ::h21 volume_z_21d    E.放量>1σ 21      1.90      1.32     -1.49      1.20      2.00      2.13     -0.44         2  B (2/4 same dir |t|≥2)        long
```

## 完整矩阵

参见 `notebooks/cross_stock_signal_matrix.csv`，共 60 个 hypothesis。

## 下一步

- Tier A/B 信号 (8 条) 进入下一阶段：构造组合策略，扣手续费滑点后做 walk-forward
- Tier C 信号需小心——可能是单股偶然或确实存在 stock-specific alpha，待多窗口验证
