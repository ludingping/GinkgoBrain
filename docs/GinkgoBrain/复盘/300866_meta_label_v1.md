# 300866 meta-label v1 OOS verdict

**生成时间**: 2026-04-29 14:15
**判定**: ⚠️ MIXED — 部分信号可信，需筛选

## 数据
- 周期: 2023-04-04 → 2026-04-22（共 742 个交易日）
- TRAIN: 2023-03-27 → 2025-05-20（519 天，70%）
- TEST : 2025-05-21 → 2026-04-21（223 天）

## 信号库
- 来源: `notebooks/cnstock_300866_signals.csv`
- 共 8 条；tier 分布: {'low': np.int64(4), 'medium': np.int64(2), 'high': np.int64(1), 'trial': np.int64(1)}

## ⚠️ 已知偏差
信号库由**全样本** pattern discovery 产出，TEST 段的数据已被 §3.4 看到过。
本测试只能回答"信号在 TEST 段是否仍能机械地复现"，不是干净 OOS。
真正干净 OOS 需要 v2：先按 TRAIN 重新跑 §3.4，再在 TEST 上验证。

## Signal-level 复现

```
                    signal_id   tier direction  h  size_pct  n_train  mean_train  t_train  n_test  mean_test  t_test  sum_test verdict
 volume_z_21d::C_中性_0_5σ::h21   high      long 21        90      105       5.098     4.05      49      2.355    1.49    115.42    ⚠️ 弱
volume_z_21d::D_偏放0_5_1σ::h21 medium      long 21        70       56       5.910     3.23      19      2.124    0.84     40.36    ⚠️ 弱
        rsi_14::C_强50_70::h21 medium      long 21        70       37       9.168     4.09      13     -6.014   -2.50    -78.19  ❌ 方向反转
        trend::3_反弹_下_涨_::h21    low     short 21        50        8      10.242     2.28       2     11.272    3.71     22.54  💤 太少样本
         rsi_14::A_超卖_30::h21    low      long 21        50       10      14.801     2.49       1     11.905     NaN     11.91  💤 太少样本
    drawdown_60d::A_高点_5_::h1    low     short  1        50       22       0.559     1.37       6      2.013    2.50     12.08    ✅ 可信
        rsi_14::B_弱30_50::h21    low      long 21        50       29       9.203     3.05      12     -3.413   -1.08    -40.96  ❌ 方向反转
 drawdown_60d::D_深回撤_25_::h21  trial      long 21        25        4      20.272     2.41       1      0.091     NaN      0.09  💤 太少样本
```

- ✅ 可信 (|t_test|≥1.5 同向): 1
- ⚠️ 弱 (同向但 t 不显著): 2
- ❌ 方向反转: 2
- 💤 样本不足 (n_test<3): 3

## 组合 P&L (log return)

| 段 | 策略 | Buy & Hold | Δ | Sharpe | avg gross |
|---|---|---|---|---|---|
| TRAIN | +1010.6% | +77.2% | +933.4% | +1.38 | 7.17x |
| TEST  | +114.2% | +13.8% | +100.4% | +0.36 | 7.96x |

## 下一步

- 信号库重做或换股票；或先做单信号深挖（如 trend.反弹 假反弹陷阱）
