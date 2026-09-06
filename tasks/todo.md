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
