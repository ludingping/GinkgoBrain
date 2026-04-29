# A 股多股票 meta-label 横向稳健性 v1

**生成时间**: 2026-04-29 23:05
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
| 300866 | 安克创新 | 736 | 19 |
| 002594 | 比亚迪 | 736 | 9 |
| 600765 | 中航重机 | 736 | 9 |
| 600893 | 航发动力 | 736 | 12 |
| 300274 | ? | 736 | 11 |
| 300346 | ? | 736 | 4 |
| 300760 | ? | 736 | 14 |

## Robustness Tier 分布

- **A (≥3/4 same dir |t|≥2)**: 0
- **B (2/4 same dir |t|≥2)**: 11
- **C (single-stock only)**: 35
- **F (mixed)**: 9
- **F (none |t|≥2)**: 68

## Tier A + B（横向稳健）

```
                        hypothesis_id    indicator    regime  h  t_300866  t_002594  t_600765  t_600893  t_300274  t_300346  t_300760  max_pass             robustness verdict_dir
        rsi_14::B_弱30_50->A_超卖_30::h5       rsi_14   A.超卖<30  5      0.69      0.33      0.59      2.23      1.32     -1.91      3.07         2 B (2/4 same dir |t|≥2)        long
           vol_21d::高_vol->中_vol::h21      vol_21d     中 vol 21      2.34     -0.21     -1.05     -1.85      2.99     -0.19     -0.98         2 B (2/4 same dir |t|≥2)        long
       rsi_14::D_超买_70->C_强50_70::h21       rsi_14  C.强50-70 21      2.74     -1.39      1.06      0.42      2.47     -0.39       NaN         2 B (2/4 same dir |t|≥2)        long
            vol_21d::中_vol->低_vol::h1      vol_21d     低 vol  1      1.10      0.24      2.05     -0.50      2.66     -1.47      1.92         2 B (2/4 same dir |t|≥2)        long
      trend::3_反弹_下_涨_->4_弱势_下_跌_::h5        trend 4.弱势(下+跌)  5      0.74     -2.29     -2.61       NaN      0.30      0.83       NaN         2 B (2/4 same dir |t|≥2)       short
     trend::1_强势_上_涨_->3_反弹_下_涨_::h21        trend 3.反弹(下+涨) 21     -2.52      1.42     -2.05      0.80     -1.43      0.49     -1.39         2 B (2/4 same dir |t|≥2)       short
     trend::2_顶背离_上_跌_->4_弱势_下_跌_::h5        trend 4.弱势(下+跌)  5      0.76      1.18     -1.81     -2.19     -0.68     -2.49      0.02         2 B (2/4 same dir |t|≥2)       short
       rsi_14::C_强50_70->D_超买_70::h21       rsi_14   D.超买>70 21      2.49     -0.90      0.42      0.60      2.37     -0.84       NaN         2 B (2/4 same dir |t|≥2)        long
volume_z_21d::D_偏放0_5_1σ->E_放量_1σ::h1 volume_z_21d   E.放量>1σ  1      1.39      2.30     -0.90      1.26      2.36      1.21     -1.40         2 B (2/4 same dir |t|≥2)        long
       rsi_14::B_弱30_50->A_超卖_30::h21       rsi_14   A.超卖<30 21      2.30      1.13      0.08      2.31      1.98      0.37      0.38         2 B (2/4 same dir |t|≥2)        long
        rsi_14::D_超买_70->C_强50_70::h5       rsi_14  C.强50-70  5      2.05     -1.52      2.19     -0.41      0.58     -1.28       NaN         2 B (2/4 same dir |t|≥2)        long
```

## 完整矩阵

参见 `notebooks/cross_stock_signal_matrix.csv`，共 123 个 hypothesis。

## 下一步

- Tier A/B 信号 (11 条) 进入下一阶段：构造组合策略，扣手续费滑点后做 walk-forward
- Tier C 信号需小心——可能是单股偶然或确实存在 stock-specific alpha，待多窗口验证
