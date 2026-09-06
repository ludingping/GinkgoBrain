# 趋势底座 + 持仓信号增量 · H1 资金费率（2026-09-06）

设计全文：`GinkgoRoad/docs/GinkgoBrain/BTC4h-趋势底座-持仓信号增量-设计.md`（v0.1，待确认）

决策：放弃价格波动 RL；目标 = 拿到 BTC 大部分上行、压缩回撤；`gate_only`（UTC 日线 SMA200）为底座，持仓数据做增量。
数据实测：funding 2021-01 起完整；OI 仅 2026-04 起（需 Spider `backfill_oi --since 2021-01-01`）；爆仓仅 2026-04 起、无回填路径。
⇒ 本期只开 H1（funding），H2（OI）等回填。

## 待办

- [x] 1. `utils/signals.py`：合约信号窗口 timeframe 感知（`add_contract_signals(df, timeframe=)`，5min 默认不变）；先写测试 RED
- [x] 2. `utils/data_loader.py::check_contract_coverage`：覆盖 <95% 报错并打印区间，替代三处静默 `fillna(0)`；测试
- [x] 3. `scripts/backtest_rules.py`：无模型规则回测器（`gate_only` / `gate×rule`），复现 gate_only val +17.6% / test +17.2% 作为回归基线
- [x] 4. `scripts/signal_ic_probe.py`：每折 rank IC / 分位价差 / AUC，最近一折单列；合成数据测试
- [x] 5. 跑 H1：funding_z200 / funding_1d_z / cum_3d / cum_7d / rank_90d × k∈{6,18} → `reports/h1_funding_ic_4h.md`
- [-] 6. **跳过（H1 未通过，不进 L3）** H1 通过（|IC|≥0.03、三折同号、最近折不反号）→ 规则 `gate_on & funding_z200>2 → 0.5` 回测 vs gate_only → `reports/h1_funding_gate_rule_4h.md`
- [x] 7. 审查小结 + lessons（见下）
- [x] 9. **H1b → FAIL（见审查）** funding × 趋势强度：`dist_sma200` 特征、探针 `--where`、规则 AND 条件；x∈{5,10,15}% 在 f1/f2 选，f3 只跑一次
- [x] 11. H2 探针（全部 bar + gate-on）→ `reports/signal_ic_probe_h2_oi_4h*.md`；H2a gate-on k=18 PASS
- [x] 12. H2a 规则回测 `gate_on & oi_level_90d_z>{0.667,0.5} → 50%/0%` vs gate_only → `reports/backtest_rules_h2a_oi_level.md`：**验收 FAIL**（见审查）
- [-] 10. **跳过（H1b 未通过）** H1b 通过 → `gate_on & dist<x & funding_cum_3d_z>0.5 → 50%` 回测 vs gate_only
- [x] 13. **H3' 退出速度 → 预注册门槛 FAIL，但 E1 首次动了回撤（见审查）**：dist_sma50 / dd20_atr / dist_low20 日线特征；探针（描述）+ 规则回测 3 条 → val 通过者看 test
- [x] 8. **已由 Claude 执行**（Spider 回填器修为按日归档 266e8da；目标库 192.168.1.68，551,231 行，覆盖 100%）（原：用户侧，Spider 仓库）`uv run python -m ginkgo_spider.scripts.backfill_oi --symbols BTC/USDT --since 2021-01-01` → 覆盖 ≥95% 后开 H2


## 审查（H1 资金费率，2026-09-06）

数据：BTC/USDT 4h，2021-01→2026-09，predicted funding（premium_kline 重建，覆盖 100%）；目标 = 未来 6 根（1 天）/ 18 根（3 天）对数收益；
3 个扩展时间折的**验证窗**（f1 2022-06→2023-11，f2 2023-11→2025-04，f3 2025-04→2026-09），不拟合，直接算 rank IC。
报告：`reports/signal_ic_probe_h1_funding_4h.md`（全部 bar）、`..._gate_on.md`（仅 gate 开启的 5,927 根，**事后分析**）。

### rank IC（f1 / f2 / f3）

| 特征 | k=6 全部 | k=18 全部 | k=18 仅 gate-on |
|---|---|---|---|
| `sig_funding_current`（z200） | −0.024 / +0.003 / −0.014 | −0.007 / +0.032 / −0.026 | +0.036 / +0.052 / −0.007 |
| `funding_cum_3d` | −0.063 / +0.024 / −0.070 | **−0.095 / +0.038 / −0.164** | −0.110 / **+0.115** / −0.213 |
| `funding_cum_7d` | −0.049 / +0.027 / −0.072 | −0.065 / +0.049 / −0.147 | −0.121 / +0.129 / −0.210 |
| `funding_rank_90d` | −0.022 / +0.026 / −0.022 | −0.017 / +0.057 / −0.048 | +0.014 / +0.098 / −0.047 |

f3 的前/后 10% 分位价差（k=18）：cum_3d −171 bps，cum_7d −154 bps。

### 结论
- **H1 未通过预注册门槛**：5 个特征 × 2 个视野共 10 组，全部在 f2 反号；步 6 规则回测按规则跳过。
- **效应存在但方向随 regime 翻转**：累计 funding 在 f1/f3（震荡与熊市）是明确的反向指标（f3 k=18 IC −0.16），但在 f2（2023-11→2025-04 ETF 牛市）是延续指标（+0.04，gate-on 下 +0.12）。"拥挤多头 → 回落"只在无趋势时成立；强趋势里高 funding 是趋势延续的一部分。
- **gate-on 条件没有救回来，反而更糟**：gate 开着的时段正是趋势最强、拥挤最能持续的时段，f2 反号从 +0.04 放大到 +0.12。这说明"gate 开 & funding 高 → 减仓"这条规则在牛市里会系统性减仓吃掉收益，正是设计 §7 预警的情形。
- 若只看 IC 均值（−0.07）或最近一折（−0.16、−171 bps），H1 会被误判为通过。分折 + 最近折不反号这个门槛起了作用。

### H1b（funding × 趋势强度交互，预注册后执行）：FAIL

子集 = gate 开 且 `dist_sma200 < x`（价格距 UTC 日线 SMA200），x ∈ {5%, 10%, 15%}；对照子集 dist ≥ x。
报告：`reports/signal_ic_probe_h1b_{weak_lt,strong_ge}{x}.md`。

| x | 行数 f1/f2/f3 | `funding_cum_3d` k=18 IC | 判定 |
|---|---|---|---|
| 5% | 516 行合计 | −0.22 / +0.08 / −0.43 | n 太小，不是证据 |
| 10% | 228 / 432 / 414 | −0.38 / **+0.12** / −0.10 | f1 n<300；f2 仍反号 |
| 15% | 438 / 540 / 905 | −0.15 / **+0.09** / −0.09 | f2 反号 |
| ≥15%（强趋势对照） | 983 / 2017 / 312 | −0.07 / +0.04 / −0.05 | f2 同样反号 |

- 弱趋势子集里 f2 的反号一个不少：交互假设没有把 f2 隔离出来。f2（2023-11→2025-04）不是"趋势太强"，而是 funding 的**含义变了**——
  现货 ETF 之后正 funding 更多来自基差套利（现货/ETF 多头 + 永续空头）的机构资金流，而不是散户杠杆拥挤，所以高 funding 不再预示回落。
  这是结构性变化，不是阈值能修的。
- 同一数据上连续两个假设（H1、H1b）都失败，按登记规则 **funding 线到此关闭**，不再提 H1c。
- 附带方法修正：探针判定加了每折 n ≥ 300 的下限（x=5% 子集的"PASS"全是 <200 行的噪声）。

### H2（OI，预注册后执行；数据 2021-01→2026-09，覆盖 100%）

| 假设 | 特征 | k=18 IC f1/f2/f3（全部 bar） | k=18 IC（仅 gate-on，n 1421/2557/1217） | 判定 |
|---|---|---|---|---|
| H2a 杠杆水平 | `oi_level_90d_z` | −0.099 / −0.018 / −0.081 | **−0.098 / −0.034 / −0.144** | 全部 bar 差 f2 一点；**gate-on PASS** |
| H2b 去杠杆反转 | `oi_capitulation_1d` | −0.026 / −0.003 / +0.007 | −0.002 / −0.022 / −0.006 | FAIL |
| H2c 新多入场 | `oi_new_longs_1d` | +0.008 / −0.032 / −0.004 | +0.056 / −0.021 / +0.014 | FAIL |
| （既有）`sig_oi_change_zscore` | | +0.017 / −0.035 / +0.004 | 反号 | FAIL |

**H2a 的 L3 规则回测**（预设 3 个变体，val 选、test 只看一次）：

| 区间 | 策略 | 总收益 | 最大回撤 | Calmar | 交易/年 | 验收 |
|---|---|---|---|---|---|---|
| val | gate_only | +17.6% | −33.9% | 0.33 | 12 | — |
| val | z>0.667 → 50% | +20.2% | −33.9% | 0.38 | 31 | FAIL（回撤、换手） |
| val | z>0.5 → 50% | +26.1% | −34.0% | 0.48 | 43 | FAIL（回撤、换手） |
| val | z>0.667 → 0% | +22.8% | −33.8% | 0.43 | 31 | FAIL（回撤、换手） |
| test | 三个变体 | 与 gate_only 相同或更差（z>0.667 在 test 从未触发） | | | | FAIL |

- **信号是真的，但打不中目标。** OI 高位减仓在 val 上多赚 3–8 个点，最大回撤一个基点都没降。
- **诊断（描述性）**：gate_only 在 val 的 −33% 回撤是 2025-01-20 → 2025-04-24；这段时间 `oi_level_90d_z` 月均 −0.53 到 −0.55，564 根里只有 2 根 >0.667。
  那是一段**去杠杆下跌**，不是拥挤多头被清算——"高 OI → 减仓"按定义碰不到它。回撤的真正来源是 **SMA200 门控退出太慢**（跌了三个月 gate 才关）。
- 换手 31–43 笔/年是 z 在阈值附近来回穿越造成的；若日后要用这个信号做收益增强，需要滞回（hysteresis）。本期不做。

### H3'（退出速度，预注册后执行）：门槛 FAIL；E1 是第一个动了回撤的规则

规则 = gate_on & 条件 → 50%。报告：`reports/backtest_rules_h3_exit_speed.md`；描述性 IC：`reports/signal_ic_probe_h3_exit_gate_on.md`（gate-on IC ≈ 0，退出规则本来就不预测均值）。

| val | 总收益 | 最大回撤 | Calmar | Sharpe | 交易/年 | 门槛 |
|---|---|---|---|---|---|---|
| gate_only | +17.6% | −34.0% | 0.33 | 0.30 | 12 | — |
| **E1 跌破日线 SMA50** | **+29.2%** | **−26.0%** | **0.70** | 0.57 | 27 | 只差回撤比：0.77 > 0.67 |
| E2 回撤 > 2×ATR(14) | −3.0% | −40.9% | −0.05 | −0.06 | 41 | 全败：来回止损 |
| E3 创 20 日新低 | +18.1% | −31.3% | 0.37 | 0.31 | 20 | 回撤比 0.92 |

test（2026-04→09，gate_only 回撤只有 −4.6%、0 笔交易）：三条规则都从未触发，与 gate_only 完全相同。

- **E1 通过了四项里的三项**（收益 166%、Calmar 翻倍、换手 27 ≤ 30），回撤从 −34% 压到 −26%，只差"≤ 2/3"这一格。它是 H1/H1b/H2/H3' 里唯一让回撤动了的候选。
- **E2 是反面教材**：ATR 单位的回撤在 4h/日线上每次 5–8% 的震荡都触发，减仓后反弹再加回，成本和错过的反弹把 val 打成负收益。
- **§1.1 的 test 段判定对退出规则不成立**：底座在 test 只有 −4.6% 回撤、0 笔交易，任何风险规则都没有东西可改善，"回撤 ≤ 2/3 × 底座"在这里等价于"必须比 −4.6% 更浅"。
  建议修订：test 段对退出规则改为**"无害"判定**（收益 ≥ 80%、回撤不更深、换手 ≤ 30）。这是方法修订，在看到数据后提出，**需用户拍板**，本次未套用。
- 按登记规则，E1 未通过 val，**没有跑 level 0 变体，也没有试别的 SMA 长度**——那是新假设，要重新登记。

### 下一步（需用户决定）
1. 是否接受 §1.1 test 段"无害"修订。接受后 E1 的状态 = val 差回撤比一格、test 无害。
2. 是否登记 H3'b：E1 的两个先验变体——`level 0`（清仓而非减半）与 `SMA20/SMA100` 两个长度——各只跑一次；仍以 val 回撤比 ≤ 0.67 为准。
3. 若 H3'b 仍不过，回撤目标本身（2/3）可能对单一日线规则过高；那时该讨论的是目标，不是再加规则。

### 旧候选（已被 H3' 覆盖）
- **H3 · 退出速度**：回撤目标要靠更快的退出，不是持仓信号——候选：gate 之外加日线 SMA50 / 2×ATR 追踪止损 / 20 日新低 作为"减仓到 50%"的第二道门。这是趋势跟随的退出设计，评价标准仍是 §1.1 四项。
- **H2a 作为收益增强**：只在 H3 把回撤压下去之后再叠加，且要加滞回控制换手。
- funding 线关闭（H1、H1b）。
- funding 线若要继续，只剩一个合理的新假设 H1b：**funding × 趋势强度交互**（如价格距 SMA200 < x% 时才把高 funding 当反向信号）。这是第二次看数据，必须作为新假设登记，且只允许在 f1/f2 上探索、f3 留出。

### code-reviewer（1 HIGH + 2 MEDIUM + 1 LOW，已修）
- HIGH：`signal_elimination.py` 未传 `required_contracts`，正是当年 6 信号全零事故的入口 → 现按候选池（扣除 `--exclude`）推导必需数据源；funding-only 跑法用 `--exclude sig_oi_*,sig_liq_*`。
- MEDIUM：`backtest_gbdt.py` 的 `required_contracts` 是死代码（从未 `with_contracts=True`）→ 由信号池推导。
- MEDIUM：覆盖检查只看总比例，窗口内部空洞也会被填零 → 新增 `interior_gaps`，必需数据源首次观测之后出现 NaN 直接报错。
- LOW：`meta_labeling_smoke.py` 未用导入。
- 全量测试 318 通过（+4 既有 `test_contract_db` 顺序依赖失败）。

### 已知遗留
- `tests/test_contract_db.py` 4 个用例在整文件运行时失败、单独运行通过（SQLite 引擎 fixture 跨用例泄漏），`utils/db.py` 与该测试均未改动，属既有问题。
- `scripts/signal_linear_baseline_pooled.py` 仍不支持合约数据（本期未用）。

---

# 多币种合并线性基线（2026-09-06）

假设：4h 信号池 AUC≈0.53 的瓶颈是样本量（BTC 单币 ~12k 根 bar），而非特征。若 BTC/ETH/BNB/SOL 合并训练后
BTC 留出段 AUC 明显上升，则多币种 PPO 值得投入；若不动，转向合约数据特征。

- [x] 1. `utils/splits.py::time_folds`：按时间戳的扩展窗口折 + embargo（先写测试 RED）
- [x] 2. `scripts/signal_linear_baseline.py::build_dataset_from_df` 加 `extra_cols`（保留 timestamp）
- [x] 3. `scripts/signal_linear_baseline_pooled.py`（+`--model lgbm`）：三组对照 single / pooled / transfer(leave-target-out)，LogReg，AUC
- [x] 4. 跑 4h 四组合（logreg/lgbm × v2 10 信号 / probe 19 信号）→ `reports/signal_linear_baseline_pooled_4h_*.md`
- [x] 5. 审查小结（见下）

## 审查（多币种合并基线，2026-09-06）

数据：BTC/ETH/BNB/SOL 4h，2021-01→2026-09，每币 12,430 根、合并 49,164 行；目标 = 未来 6 根（1 天）方向；
3 个按时间戳切的扩展窗口折（embargo 6 根）。三种训练方式：single（本币历史）/ pooled（四币）/ transfer（只用其他三币）。

### BTC 留出段 AUC（3 折均值）

| 模型 | 信号池 | single | pooled | transfer | Δ pooled−single |
|---|---|---|---|---|---|
| LogReg | v2 10 信号 | 0.5256 | 0.5287 | 0.5273 | +0.003 |
| LogReg | 19 信号（v2+淘汰 9 个） | 0.5235 | 0.5271 | 0.5248 | +0.004 |
| LightGBM | v2 10 信号 | 0.5138 | 0.5244 | 0.5286 | +0.011 |
| LightGBM | 19 信号 | 0.5148 | 0.5260 | 0.5275 | +0.011 |

其他币同样：ETH 最高（pooled lgbm/19 = 0.541），SOL 最低（≈0.51-0.52），无一越过 0.55。

### 结论
- **样本量不是瓶颈，特征是。** 四倍样本只把 BTC AUC 抬 0.003–0.011，全在折间波动（±0.02）以内；LightGBM 单币 0.514 < LogReg 0.526，说明非线性模型在 1.2 万根上只是过拟合，合并后也只是回到线性水平，并没有挖出新结构。
- **transfer ≈ pooled ≈ single**：只用 ETH/BNB/SOL 训练、在 BTC 上测，和用 BTC 自己训练一样好。信号池里的规律确实是跨币种共享的，但共享的那部分本身只有 0.53 的信息量。
- **最近一折（2025-04→2026-09）所有组合都掉到 0.50–0.52**，正是 PPO v2 的 val/test 区间。当前 regime 下这套技术信号几乎没有方向信息，与 v2 训练"train↑ eval↓"和线上 gate_only 占优一致。
- 决策：**不做多币种 PPO 训练**。下一步转向新数据源：合约信号（funding / OI / 爆仓）在 4h 的边际贡献；`sig_funding_*`/`sig_oi_*`/`sig_liq_*` 在 v2 筛选时因训练期全零被剔除，需先确认 ginkgo_bole 合约表覆盖 2021 起的历史，再用 `--with-contracts` 重跑 `signal_linear_baseline` 和本脚本。

### 复现
```
uv run python -m scripts.signal_linear_baseline_pooled --cache-dir <dir> --model {logreg,lgbm} \
    --signals-config {config/signals_v2_4h.yaml,config/signals_4h_probe19.yaml}
```

---

# BTC/USDT 4h PPO 重训 v2 —— 对齐 Spider 虚拟盘（2026-09-05，PR ludingping/GinkgoSpider#20）

计划全文：`~/.claude/plans/peppy-hugging-magpie.md`

- [x] 1. `utils/signals.py` OBV → `diff(5)`；起点不变性测试（旧定义 RED：Δ=0.139，仅 OBV 一列）
- [x] 2. `envs/signal_layered_env.py` `stop_atr_mult<=0` 禁用止损 + 测试
- [x] 3. `signal_elimination.py` `--exclude` + 零方差列修复（6 列全零合约信号此前会进候选池）→ `config/signals_v2_4h.yaml`：10 信号 = v1 的 11 − drawdown
- [x] 4. backtest `gate_only` / `ppo_gate`（UTC 日线 SMA200，在全量 df 上算）；v1 在新 val（2024-10→2026-04）：PPO +2.0%（553 笔）、ppo_gate −1.35%、gate_only +14.2%、b&h +11.6%
- [x] 5. `config/stage2_4h_signal_v2.yaml`
- [x] 6. 200k smoke（n_stops=0、trades ~400/3354）→ 1M 全量启动
- [x] 7. backtest val + 留出 test（见审查）
- [x] 8. `scripts/export_paper_artifact.py`（zip + signals.yaml[artifact/serving/fingerprint] + golden json；`--verify` 通过；5 单测）→ `artifacts/btc_4h_ppo_gate_v2.*`
- [x] 9. 全量测试 287 通过；code-reviewer 1 HIGH（gate 的 bar 长度改为 diff 众数）+ 2 MEDIUM（零方差判定补 all-NaN；zip 模型类型改读 `policy_class` 字段）已修；Spider 同步清单见下

## 审查（4h v2，2026-09-05）

### 训练过程
- 第一次 1M 步（net [64,64], n_epochs 10, ent 0.003）：**单调过拟合**——训练奖励 −0.44→+0.07/episode，val cum_log −0.02→−0.31，best_model = 10k 步。8.2k 根训练 bar 跑 122 遍。
- 正则化版（net [32,32], n_epochs 4, ent 0.01, lr 1e-4, 300k）：val 在 100k 步达峰（cum_log +0.075，换手 107）后同样下滑；best_model = 100k。**导出的就是这个**。其动作概率仍近均匀（argmax 0.24），正收益来自低敞口而非判断力。

### 结果（Sharpe 按 2190 年化；`gate` = Spider 的 UTC 日线 close>SMA200）

| 区间 | 策略 | 收益 | 最大回撤 | Sharpe | 交易 |
|---|---|---|---|---|---|
| **val** 2024-10-01→2026-04-12 | v2 正则（导出） | +7.8% | −4.1% | 0.79 | 105 |
| | v2 正则 + gate | +4.2% | −3.7% | 0.55 | 59 |
| | v2 首次 1M 最终 ckpt | −17.5% | −41% | −0.41 | 337 |
| | v1（线上，修复后 OBV 输入） | +2.0% | −20% | 0.08 | 553 |
| | v1 + gate | −1.4% | −19% | −0.07 | 354 |
| | **gate_only** | **+17.6%** | −34% | 0.30 | 19 |
| | buy&hold | +11.6% | −50% | 0.16 | 1 |
| **test** 2026-04-13→2026-09-03（虚拟盘同期，从未见过） | v2 正则（导出） | +1.1% | −1.1% | 0.79 | 29 |
| | v2 正则 + gate | +1.7% | −0.8% | 2.18 | 3 |
| | v2 首次 1M 最终 ckpt | +12.2% | −18% | 1.24 | 80 |
| | v2 首次 + gate | +12.4% | −3.1% | 2.93 | 6 |
| | v1 + gate（修复后 OBV） | +6.8% | −3.1% | 2.27 | 8 |
| | **gate_only** | **+17.2%** | −4.6% | 2.93 | 0 |
| | buy&hold | +14.3% | −29% | 0.90 | 1 |

（Spider 实录同期：模型+gate +3.48%，gate_only +14.86%，b&h +2.72%，04-22→09-05。）

### 结论
- **没有任何 PPO 候选在两个区间上同时优于 gate_only。** 首次 1M 模型在 test 上 +12% 只是牛市 beta（val −17.5%）；正则模型两段都正但基本空仓。这与 Spider #23 的实测结论一致：**在当前 4h 信号池上，PPO 层对 gate 没有增量价值**。
- 管线层面的问题已全部修复并有测试覆盖（OBV 起点依赖、obs 滞后、止损/手续费对齐、信号池训练期重筛、年化、留出集、指纹）。导出的 `btc_4h_ppo_gate_v2` 是**与线上契约一致、不过拟合、最低换手**的候选，但部署价值有限——**建议 Spider 先 A/B：同一账户结构下跑 gate_only 与 v2+gate**；若几周后 gate_only 仍占优，应下线 PPO 层，把研究重点转到特征（合约数据 funding/OI/爆仓、更长视野）而不是 RL 超参。
- 4h BTC 训练集本身只有 ~8k 根 bar，任何 MLP 都会背下来；不解决样本量（多币种、更长历史、数据增广）就不该再投入训练算力。

### Spider 侧同步清单（合并 PR #20 与新 artifact 须同时上线）
1. **依赖**：`paper` extra 加 `sb3-contrib>=2.3.0`（与 SB3 同版本 2.8.0）。
2. **predictor.py**：`MaskablePPO.load(path, device="cpu")`；`policy.get_distribution(obs, action_masks=mask[None])`；argmax。mask 见 3。
3. **min_hold 锁 + mask**：持久化 `last_executed_level`（0–4，**gate 之后**实际执行的档位）与 `bars_since_change`；`bars_since_change < 3` 时 mask = one-hot(last_executed_level)，否则全 True；**gate 在锁之后应用**（gate 永远可平仓）；deadband 阈值 0.05→0.02（= 训练 `MIN_REBALANCE_PCT`）。
4. **signals.py**：整体同步 Brain `utils/signals.py`（含 OBV `diff(5)` 与 `sig_mtf_*` 的 `shift(1)`）。启动时用 `signals.yaml.fingerprint` 做 #21 校验：读 `*.golden.json` → `add_indicators+add_signals` → 与 `*.golden_signals.json` 比对 `atol=1e-6`（见 `serving.fingerprint`）。把 `verify_signal_list(train, train)` 换成对 `signals.yaml.signals` 的真实比对。
5. **state 特征对齐**（`serving.state_features`）：`position_ratio` 在**当前 bar 收盘价**重估并 clip[0,1]（现为上一 bar 快照、且 #24 下可 >1）；`unrealized` 用当前 close；`cum_log_return` 存**未 clip** 值并含最新一根，写入 obs 时 clip；`_simulate_fill` 用 `prior_qty*close/portfolio_before` 算 delta，买入手续费从预算内出（#24）。
6. **止损**：无（训练已关闭）；`steps_since_stop` 保持 1.0 即正确。
7. 复制 `GinkgoBrain/artifacts/btc_4h_ppo_gate_v2.{zip,signals.yaml,golden.json,golden_signals.json}` → `ginkgo_spider/paper/artifacts/`，配置 `model.path`/`signals.config_path` 指向 v2。

---

# BTC/USDT 1h PPO 重训 v3（2026-09-05）

计划全文：`~/.claude/plans/peppy-hugging-magpie.md`

## 待办

- [x] 1. `envs/signal_layered_env.py`：obs 滞后修复、`min_hold_steps` + `action_masks()`、`max_episode_steps`；单测（+8）
- [x] 2. `scripts/backtest_signal_layered.py`：periods_per_year（`utils/metrics.py` 新增 canonical `BARS_PER_YEAR`）、白名单、206 前缀、MaskablePPO 加载、baseline 表
- [x] 3. v2 重跑（修正年化）：Sharpe −3.0、3456 笔；验证期为熊市（b&h −14%）；常数仓位奖励表 p=100% → −0.00076/步 ⇒ `risk_aversion_coef` 0.5 → **0.05**
- [x] 7. 200k smoke：熵/换手/TB 指标正常；发现 2×ATR 止损在 1h 过紧（142 次/9167 bars）→ `stop_atr_mult` 4.0
- [x] 4. `agents/trainer.py`：maskable_ppo、Monitor、n_eval_episodes、eval 收益日志（合成数据 smoke 通过）
- [x] 5. `train.py`：`split_by_dates`（新 `utils/splits.py` + 7 测试）、allowed keys 模块级、action_masking 路由
- [x] 6. `config/stage2_1h_signal_v3.yaml`（守卫测试：不得继承 default.yaml 的 env 键）
- [x] 8. 1M 全量训练（21 min，run=BTCUSDT_signal_layered_1h_v3；训练奖励 −0.96→−0.37/episode 仍在上升，best_model = 最终 checkpoint）
- [x] 9. backtest val + 留出 test（见审查）
- [x] 10. `scripts/ppo_signal_1h.py`：改为"固定锚点确定性回放"（无状态文件）；`--verify` 发现 `sig_volume_obv_slope` 非起点不变 → 默认从 `start_date` 全量加载；已用 v3 模型产出 `reports/daily_signals/btc_1h_latest.md`
- [x] 11. 审查小结 / lessons；code-reviewer 1 HIGH + 2 MEDIUM 已修（止损不再重置/叠加 min_hold 锁；`_get_obs` 不可达分支改 raise；`slice_by_dates` 前缀不足报错）

## 审查（2026-09-05）

### 结果（Sharpe 按 8760 年化）

| 区间 | 策略 | 总收益 | 最大回撤 | Sharpe | 交易数 |
|---|---|---|---|---|---|
| **val** 2025-03-27→2026-04-12（熊市） | PPO v3 | −6.4% | −25% | −0.29 | 1118 |
| | buy&hold | −18.1% | −50% | −0.45 | 1 |
| | 4h-trend 规则 | −10.4% | −33% | −0.38 | 108 |
| | v2（旧模型，修正年化） | −53% | −60% | −3.0 | 3456 |
| **test** 2026-04-13→2026-09-03（牛市，从未见过） | **PPO v3** | **−19.8%** | −28% | **−2.81** | 390 |
| | buy&hold | +14.6% | −29% | +0.89 | 1 |
| | 4h-trend 规则 | +6.9% | −19% | +0.63 | 45 |
| | const 25% | +3.5% | −9% | +0.89 | 8 |

### 结论
- 管线的结构性缺陷全部修复并有测试覆盖：obs 滞后、ent_coef、敞口税（risk_aversion 0.5→0.05）、换手限频（MaskablePPO + min_hold）、default.yaml 深合并陷阱、年化常数、无留出集。
- **但 PPO v3 在留出集上没有 alpha，且为负 alpha**：仓位与下一根收益相关 −0.034，3455 根累积成 −0.25 log return；手续费另贡献 −8%。惩罚项合计 −0.7 仍换手 390 次，策略对惩罚梯度无有效响应，在跟随噪声。
- 与 `reports/signal_linear_baseline_1h.md`（AUC 0.536）一致：**1h 信号池信息量太薄，问题不在 RL 算法而在特征**。val 上"优于 b&h"只是熊市里少持仓的副产品。
- 不建议在同一信号池上继续调参训练（会变成在 val 上 p-hacking）。若继续 1h 方向，先做特征侧工作：合约数据（funding/OI/liquidation）信号在 1h 的边际贡献、或更长视野目标（k=24 已是弱边际）。

### 已知遗留
- `sig_volume_obv_slope = znorm(obv.pct_change(5))` 依赖 OBV 累计起点，非起点不变；elimination log 中其 Sharpe −0.000。下次修订信号池时改为 `obv.diff(5)/rolling_volume` 之类的尺度不变形式。
- `tests/test_signals.py::test_tc_a2_all_15_signals_present` 在改动前就失败（信号池已扩到 22 个），与本次无关。
- `scripts/backtest_gbdt.py`、`scripts/signal_elimination.py` 各有一份 `BARS_PER_YEAR`，可改为引用 `utils.metrics.BARS_PER_YEAR`。
