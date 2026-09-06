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

## 2026-09-06 · H1 资金费率 IC 探针

- **预注册门槛不是形式。** funding 累计的 IC 均值 −0.07、最近一折 −0.16、分位价差 −171 bps，任何一个单看都会判"通过"；但中间一折（ETF 牛市）反号 +0.04，gate-on 下 +0.12。规则上线后遇到下一次强趋势就会系统性减仓。"三折同号 + 最近折不反号"这个门槛必须在看数据之前写死。
- **拥挤指标的方向取决于 regime。** funding 在震荡/熊市是反向指标，在强趋势里是延续指标。任何"极端 → 反向"假设都要分 regime 验证，并特别看"规则实际会触发的那个 regime"（gate-on 时段）——那里往往正是假设最不成立的地方。
- **事后加条件是 p-hacking 的入口。** gate-on 条件化是合理的分析，但它是第二次看数据；必须标记 post-hoc，结果无论好坏都不能当作"通过"。后续假设只能在前两折探索、最后一折留出。
- **"兼容性填充"要换成显式覆盖检查。** 四处静默 `fillna(0)` 让 6 个合约信号在 4h v2 里"训练期全零"而无人察觉；现在 `check_contract_coverage` 对信号池需要的数据源要求 ≥95% 覆盖，其余只打印。
- **时间计价的窗口必须由 timeframe 推导。** 合约信号的 `rolling(288)` 写死了 5min 假设，在 4h 上变成 48 天均值。凡是"24h 均值 / 1h 平滑"这种以时间定义的窗口，用 `pd.Timedelta(span) / pd.Timedelta(tf)` 换算，并保证不带 tf 的旧调用逐值不变。
- **底座先要能独立复现。** `gate_only` 此前只能作为 PPO 回测的副产品跑；`backtest_rules.py` 让它成为一等公民并精确复现 val +17.60% / test +17.21%，之后所有规则都以它为对照。
- **同一数据上第二个假设失败就停。** H1b 是对 H1 失败的"合理解释"，预注册后照样在同一折反号。这时最诱人的是提 H1c；规则是关闭这条线，把"为什么 f2 不一样"记为结构性观察（ETF 后 funding 由基差套利主导）而不是继续切子集。
- **过滤子集必须报 n。** gate-on 且 dist<5% 只剩 516 行，IC 动辄 ±0.5 还显示 PASS；探针现在每折 n<300 直接判不通过。任何条件化分析先看行数再看 IC。
- **信号通过 IC 门槛 ≠ 规则通过验收；先看回撤发生在哪。** H2a 的 OI 水平 z 三折同号、gate-on 下 IC −0.1，规则却一个基点回撤都没降——那段 −33% 是去杠杆下跌，OI z 全程为负。做"压回撤"的假设前，先把底座的最大回撤episode 标出来，看候选信号在那段是什么值；不在那段起作用的信号再好也是收益增强，不是风控。
- **回填要写到消费方读的那个库。** Spider 本机 .env 指向 localhost（192.168.1.111），Brain 读 192.168.1.68；第一次回填成功写了 20 万行到错的库。跨仓库的数据任务先对 DSN，再看行数。
- **Binance Vision 的 futures metrics 只有 daily 归档。** monthly URL 全部 404 但脚本"正常结束"，只补了 REST 30 天；回填器要把"0 行"当异常而不是"未上市"。
- **`pgrep -f` 会匹配到监视器自己的命令行。** 等待进程退出的 Monitor 因此永远不退出（和 lessons 里 pkill 那条同源）；匹配进程用模块名且排除自身，或改用 pid 文件。
