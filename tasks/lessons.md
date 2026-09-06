# Lessons

## 2026-09-05 · BTC 4h PPO v2（对齐 Spider 虚拟盘）

- **看 train/eval 曲线的相对走向，不看绝对值。** 训练奖励单调升、eval 单调降且 best_model 停在第一个 checkpoint = 纯过拟合；8k 根 bar 上 1M 步（122 遍）必然如此。小样本先算"每根 bar 被看多少遍"，超过 ~20 遍就该缩网络/减 epoch/早停。
- **一段区间的好成绩不算证据。** 首次 1M 模型 test +12% 但 val −17.5%；牛市里偏多就是 beta。至少两个 regime 不同的留出段都要看。
- **服务侧能算出来的信号才有资格进池。** `sig_mtf_1d_regime` 需要 ~960 根 4h 历史，Spider 只拉 500 根 → 线上恒为 0；日线还有 UTC/Shanghai 边界差异。信号筛选时用 `--exclude` 把服务侧约束前置。
- **零方差列会让相关性筛选静默出错**（corr=NaN → 排序未定义）。任何进候选池的列先检查 std。
- **导出物要带契约，不只带模型。** `signals.yaml` 里的 `serving:`（obs/state 定义、锁、成本）+ `fingerprint:`（golden 输入与期望输出）让另一个仓库能自检 train/serve 一致性；否则像 v1 那样 obs 错位一根 bar 也没人发现。
- **旧配置的 `end_date` 会让新留出集切片为空**——`backtest_signal_layered.py` 现在会明确报错，但用旧配置比对时记得延长 `end_date`。

## 2026-09-05 · BTC 1h PPO v3

- **配置深合并会静默继承默认值。** `train.py` 用 `config/default.yaml` 做底再 deep_merge stage 配置；任何进入 `_ALLOWED_ENV_KEYS` 的键若 stage 里没写就继承默认（`trade_penalty_coef: 1.0` 即一例）。规则：stage 配置显式写全 env/algo_kwargs；已加守卫测试 `test_v3_config_overrides_every_forwarded_default_env_key`。
- **先跑常数策略奖励表，再训练。** `const_p` 的 mean reward/step 直接暴露奖励整形是否被"敞口税"主导（v2 满仓 −0.00076/步 vs alpha 量级 1e-4）。任何新奖励函数都应先过这一步。
- **年化常数必须由 timeframe 推导。** 硬编码 sqrt(2190) 让 1h 的 Sharpe 全部减半；用 `utils.metrics.bars_per_year(tf)`。
- **验证集 ≠ 留出集。** EvalCallback 选 best_model 用的片段不能再当 OOS 汇报；用日期钉死 `val_start`/`test_start`。v3 在 val 上"跑赢 b&h"，在 test 上却是所有策略中最差——没有留出集这个结论会完全相反。
- **累计型指标（OBV）的派生信号不是起点不变的。** 实时脚本用 400 天窗口重算，`sig_volume_obv_slope` 与训练管线差 2.0。任何在线推理脚本都要做与全历史管线的 `--verify` diff。
- **1h 上 PPO 的失败是特征问题，不是算法问题。** 线性基线 AUC 0.536 已经预示了；在同一信号池上继续调 RL 超参只会在 val 上 p-hacking。
- **shell：`pkill -f <pattern>` 会匹配到自己所在的 bash 命令行并把自己杀掉**（exit 144）。用 `pgrep` 先看，或用 `pkill -f "^python train.py"` 之类的锚定模式。

## 2026-09-06 · 多币种合并线性基线

- **"样本量不够"是假设，不是结论——先用监督基线证伪再动 PPO。** 四币合并 4 倍样本对 BTC AUC 只 +0.003~0.011，一小时就排除了多币种 PPO 这条路，省下的是几天训练。任何"加数据能救 RL"的想法先过 single / pooled / transfer 三臂对照。
- **合并多资产的 CV 必须按时间戳切折并加 embargo，不能按行切。** 同一时刻的 BTC/ETH 高度相关，索引切分会通过兄弟资产把验证期泄漏进训练。`utils.splits.time_folds` 已封装。
- **线性探针验不了样本量假设。** 10 参数的 LogReg 在 1.2 万样本上早已饱和，pooled 不涨是意料之中；要验"更多数据能否让非线性模型学到结构"必须同时跑一个小 GBDT。结果 GBDT 也不涨，结论才站得住。
- **分折看 regime。** 三折均值 0.53 掩盖了最近一折只有 0.51；而线上跑的正是最近这个 regime。汇报 AUC 时把最近一折单独列出。
