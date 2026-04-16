# PPO 训练优化：观察归一化 & 奖励整形（GinkgoBrain）

## 1. 背景与目标

当前以 SB3/PPO 在 BTC 1 分钟数据上训练，模型行为在**疯狂频繁交易**与**完全不交易**两个极端之间震荡，无法收敛到有意义的策略。

根因分析确认为五项结构性缺陷（非超参问题），本文档描述对应的修复方案。

---

## 2. 缺陷清单

| 编号 | 位置 | 问题 |
|------|------|------|
| P0 | `envs/stock_env.py` `_get_obs` | 技术指标列未归一化，OBV 绝对值可达亿级，梯度被劫持 |
| P0b | `train.py` + `agents/trainer.py` | 若使用全局统计量（sklearn scaler fit on 全集），归一化系数本身含有未来信息，造成隐式前视偏差；`VecNormalize` 是结构性解法 |
| P1 | `envs/base_env.py` `step` | 手续费只从 portfolio 扣除，reward 无即时交易惩罚信号 |
| P2 | 训练流程 | 直接在 1min 数据上从零训练，信号太稀薄无法建立有效策略 |
| P3 | `config/default.yaml` | commission 使用真实费率，无法约束训练期间的交易频率 |
| P4 | `config/default.yaml` | `window_size=20`，在 1min 数据上仅覆盖 20 分钟，上下文严重不足 |
| P5 | `envs/base_env.py` `step` + `_get_obs` | 无动作惯性约束，模型在两个动作间"反复横跳"，切换成本无法被学习到 |
| P6 | `utils/indicators.py` + `envs/stock_env.py` | 仅使用 1m 单一粒度特征，信噪比极低；缺乏 5m/15m 趋势上下文 |
| P7 | `config/stage*.yaml` | `ent_coef` 全程固定，早期探索不足陷入"只 Hold"局部最优；晚期过高导致已收敛的策略再度崩溃 |
| P8 | `envs/base_env.py` `step` | 简单百分比收益率在插针行情下奖励突变；无风险厌恶，模型对亏损和盈利的敏感度对称，不符合实际交易心理 |
| P9 | `agents/trainer.py` + `envs/` | `window_size=120` + `MlpPolicy` 将 3000+ 维扁平向量输入 MLP，丢失时序结构；内存与训练时间随 window 线性膨胀 |
| P10 | `agents/trainer.py` + `envs/base_env.py` | 单进程训练未利用多核；所有 env 从同一起点 reset，经验相关性高，梯度估计方差大 |

---

## 3. P0：修复观察归一化

### 3.1 当前问题

`_get_obs` 只归一化了 OHLCV：

```python
# 现有代码（不完整）
for col in ["open", "high", "low", "close"]:
    frame[col] = frame[col] / last_close
frame["volume"] = frame["volume"] / vol_max
# ← 其余列（RSI、OBV、MACD、BB…）直接原始值进入观察空间
```

各列数值范围差异：

| 特征 | 未归一化数量级 | 归一化后目标 |
|------|--------------|------------|
| open/high/low/close | 已 ÷ last_close ≈ 1.0 | ✓ 正常 |
| volume | 已 ÷ max ∈ [0,1] | ✓ 正常 |
| RSI / stoch_k / stoch_d | 0–100 | 需 ÷ 100 → [0,1] |
| OBV | ±亿级绝对值 | 需改用 pct_change → [-1,1] 截断 |
| EMA_9 / EMA_21 | BTC 价格绝对值（≈30000–70000） | 需 ÷ last_close |
| BB_upper / BB_lower | 同上 | 需 ÷ last_close |
| MACD / MACD_signal | 价格量纲（绝对差值） | 需 ÷ last_close |
| ATR | 价格量纲（绝对波幅） | 需 ÷ last_close |

### 3.2 修改方案

文件：`envs/stock_env.py`，方法 `_get_obs`。

归一化分三组处理：

**组 A — 价格量纲类**（÷ last_close）

```
open, high, low, close, ema_9, ema_21, bb_upper, bb_lower, macd, macd_signal, atr
```

**组 B — 成交量/动量百分比类**（÷ 100 归入 [0,1]）

```
rsi, stoch_k, stoch_d
```

**组 C — OBV**（改为单步变化率，clip 到 [-1, 1]）

```python
frame["obv"] = frame["obv"].pct_change().fillna(0).clip(-1, 1)
```

**volume** 保持现有方案（÷ 窗口最大值）。

处理顺序：先做 OBV pct_change，再做价格类除法，避免 last_close 变化影响 OBV。

### 3.3 完整实现

```python
def _get_obs(self) -> np.ndarray:
    frame = self.df.iloc[
        self.current_step - self.window_size: self.current_step
    ].copy()

    last_close = frame["close"].iloc[-1] + 1e-8

    # 组 C：OBV 先转变化率
    if "obv" in frame.columns:
        frame["obv"] = frame["obv"].pct_change().fillna(0).clip(-1, 1)

    # 组 A：价格量纲 ÷ last_close
    PRICE_COLS = [
        "open", "high", "low", "close",
        "ema_9", "ema_21", "bb_upper", "bb_lower",
        "macd", "macd_signal", "atr",
    ]
    for col in PRICE_COLS:
        if col in frame.columns:
            frame[col] = frame[col] / last_close

    # volume ÷ 窗口最大值
    vol_max = frame["volume"].max() + 1e-8
    frame["volume"] = frame["volume"] / vol_max

    # 组 B：0–100 指标 ÷ 100
    for col in ["rsi", "stoch_k", "stoch_d"]:
        if col in frame.columns:
            frame[col] = frame[col] / 100.0

    obs = frame.values.astype(np.float32)

    # 拼接持仓比例
    price = float(self.df.loc[self.current_step, "close"])
    portfolio = self._portfolio_value(price) + 1e-8
    pos_ratio = np.full(
        (self.window_size, 1),
        self.position * price / portfolio,
        dtype=np.float32,
    )
    return np.concatenate([obs, pos_ratio], axis=1)
```

### 3.4 P0b：VecNormalize 防止全局统计量泄漏

#### 3.4.1 风险说明

P0 的归一化方案使用**点时间参考量**（`last_close`、窗口内 max），本身不含未来信息。但存在两种常见的工程失误会引入隐式前视偏差：

**失误一：用 sklearn scaler fit 全集**

```python
# ❌ 错误写法——scaler 的 mean/std 包含 test 集的统计量
scaler = StandardScaler().fit(df_full)
df_train_scaled = scaler.transform(df_train)
env = CryptoTradingEnv(df_train_scaled)
```

在时刻 T = 1000（训练集早期），归一化用的均值包含了 T = 50000（未来）的价格信息。模型"看到"的 close ≈ 0.0，实际是相对未来均值的偏差——这是前视偏差。

**失误二：MTF 特征的 rolling 窗口越界**

P6 中 `add_multi_timeframe_indicators` 在 resample 后调用 `add_indicators`，而 `add_indicators` 内部的 `ta` 库指标（如 EMA、MACD）是**纯因果**的（每个值只用过去数据），不存在泄漏。但若后续手动加入任何 `rolling().mean()` 且忘记 `min_periods`，就会在序列开头产生使用未来数据的值。

#### 3.4.2 结构性解法：VecNormalize

SB3 的 `VecNormalize` 是一个 **online running mean/variance** 包装器，在训练期间随数据流动增量更新统计量，在每个时刻 T 只使用 T 之前见过的观察值，从根本上消除全局统计量泄漏问题。

```
每步 obs 流经 VecNormalize 的过程：

  Raw obs (from env) → [running_mean, running_var 更新] → (obs - μ) / σ → policy
```

由于统计量是增量更新的，时刻 T 的归一化系数仅由 [0, T) 的历史决定，**不包含未来数据**。

#### 3.4.3 与 P0 的关系：两层归一化

P0 已经做了**语义层归一化**（价格比率、百分比归一化），使各特征的量级基本可比。在此基础上套 VecNormalize，等于追加了一层**统计层归一化**，处理以下 P0 无法覆盖的问题：

| 场景 | P0 是否处理 | VecNormalize 补充 |
|------|-----------|-----------------|
| OBV 的长期量级漂移 | 部分（pct_change） | ✓ 消除跨市场周期的均值漂移 |
| ATR 在高波动期（2021牛市）vs 低波动期的量级差 | ✗ | ✓ 自适应调整 |
| MTF 特征（m5_macd）的绝对值范围随行情变化 | 部分（÷ last_close） | ✓ 进一步平稳化 |
| 全局 scaler 引入的前视偏差 | ✗（风险存在） | ✓ 结构性消除 |

#### 3.4.4 实现方案

**`agents/trainer.py`**

```python
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


class Trainer:
    def __init__(
        self,
        env,
        eval_env=None,
        ...,
        normalize_obs: bool = True,      # ← 新增
        normalize_reward: bool = False,  # ← 新增（见下方说明）
        clip_obs: float = 10.0,          # ← 新增
    ):
        ...
        # 包装为向量环境
        venv = DummyVecEnv([lambda: env])
        if normalize_obs:
            venv = VecNormalize(
                venv,
                norm_obs=True,
                norm_reward=normalize_reward,
                clip_obs=clip_obs,
                gamma=0.99,
            )
        self.venv = venv

        if eval_env is not None:
            eval_venv = DummyVecEnv([lambda: eval_env])
            if normalize_obs:
                # eval 环境共享训练环境的统计量，不自行更新
                eval_venv = VecNormalize(
                    eval_venv,
                    norm_obs=True,
                    norm_reward=False,
                    training=False,   # ← 冻结统计量
                )
            self.eval_venv = eval_venv
```

> **为什么 `normalize_reward=False`？**
>
> P8 已经将 reward 设计为对数收益率（量级 ±0.001–0.01），并精确标定了 P1/P5 的惩罚系数。若 VecNormalize 再对 reward 做 z-score，会改变惩罚项的相对权重，破坏 P1/P5/P8 的协调关系。观察空间归一化和 reward 归一化应该分开控制，此处只开 obs 归一化。

#### 3.4.5 保存与加载（重要）

VecNormalize 的 running stats 必须和模型权重一起保存，否则推理时用错误的统计量会导致模型行为完全不同：

```python
# 训练结束后
trainer.model.save("models/saved/btc_ppo_stage4.zip")
trainer.venv.save("models/saved/btc_ppo_stage4_vecnorm.pkl")

# 推理 / 评估时
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

eval_env = DummyVecEnv([lambda: CryptoTradingEnv(df_test, ...)])
eval_env = VecNormalize.load("models/saved/btc_ppo_stage4_vecnorm.pkl", eval_env)
eval_env.training = False    # 冻结统计量，不继续更新
eval_env.norm_reward = False

model = PPO.load("models/saved/btc_ppo_stage4.zip", env=eval_env)
```

#### 3.4.6 热启动时的 stats 传递

阶段切换热启动时，上一阶段的 VecNormalize stats **不能直接复用**（因为时间粒度不同，分布不同）。每个阶段重新从零积累 running stats，通常在前 1000–2000 steps 之内 stats 就会趋于稳定（因为 P0 的语义归一化已经将量级控制在合理范围内）。

---

## 4. P1：奖励函数加即时交易惩罚

### 4.1 当前问题

`base_env.py` 的 `step` 方法：

```python
# 现有代码
reward = (new_portfolio - prev_portfolio) / (prev_portfolio + 1e-8)
```

手续费通过 `_execute_action` 减少了 portfolio，但 reward **同时已经包含了 portfolio 的下降**，模型看到的是一个混合信号，而非"这次交易本身让我损失了手续费"的即时归因。

在 1 分钟数据上，单步价格变动（≈0.02%）与手续费（0.05%）同量级，reward 被噪声淹没，无法区分"方向错"和"手续费拖累"。

### 4.2 修改方案

引入 `trade_penalty_coef` 参数（默认 1.0），在发生交易时给 reward 叠加一个即时惩罚项：

```
reward = ΔPortfolio/Portfolio − commission × trade_penalty_coef  （有交易时）
reward = ΔPortfolio/Portfolio                                      （Hold 时）
```

**参数说明**

| 参数 | 推荐初始值 | 作用 |
|------|-----------|------|
| `trade_penalty_coef` | 1.0 | 1.0 = 手续费原值；2.0 = 双倍惩罚，更激进地抑制过度交易 |

### 4.3 改动位置

**`envs/base_env.py`**

1. `__init__` 增加参数：

```python
def __init__(
    self,
    df,
    window_size: int = 20,
    initial_balance: float = 10_000.0,
    commission: float = 0.001,
    trade_penalty_coef: float = 1.0,   # ← 新增
    render_mode: str | None = None,
):
    ...
    self.trade_penalty_coef = trade_penalty_coef
```

2. `_reset_state` 增加交易标志：

```python
def _reset_state(self):
    ...
    self._last_trade_cost = 0.0   # ← 新增，记录当步实际手续费
```

3. `_execute_action` 记录手续费：

```python
def _execute_action(self, action: int, price: float):
    self._last_trade_cost = 0.0
    if action == 1 and self.balance > 0:
        units = (self.balance * (1 - self.commission)) / price
        self.position += units
        self.balance = 0.0
        self._last_trade_cost = self.commission   # ← 新增
        self.trades.append(...)
    elif action == 2 and self.position > 0:
        self.balance += self.position * price * (1 - self.commission)
        self.position = 0.0
        self._last_trade_cost = self.commission   # ← 新增
        self.trades.append(...)
```

4. `step` 叠加惩罚：

```python
reward = (new_portfolio - prev_portfolio) / (prev_portfolio + 1e-8)
reward -= self._last_trade_cost * self.trade_penalty_coef   # ← 新增
```

---

## 5. P2：训练课程（由粗到细）

### 5.1 当前问题

直接在 1 分钟数据上从零训练：
- 单步信号极弱（噪声 >> 趋势）
- 没有基准模型，无法判断是框架问题还是数据问题
- 超参调试成本极高

### 5.2 训练课程

按以下顺序逐级验证：

```
阶段 1：日线（1d）
  目标：reward 曲线上升，eval 夏普 > 0.5
  数据：BTC/USDT 2020-01-01 至今，~1500 根
  window_size: 20
  验收：模型不总是 Hold，有明显的买卖切换

阶段 2：4 小时线（4h）
  目标：策略从日线迁移，eval 夏普 > 0.3
  数据：BTC/USDT 2020-01-01 至今，~6000 根
  window_size: 48（覆盖 8 天）
  验收：胜率 > 45%，最大回撤 < 30%

阶段 3：1 小时线（1h）
  目标：用 4h 权重热启动，进一步 fine-tune
  window_size: 120（覆盖 5 天）
  热启动方式：SB3 model.set_parameters() 或 model.load() 继续训练

阶段 4：1 分钟线（1min）
  目标：用 1h 权重热启动，final fine-tune
  window_size: 120（覆盖 2 小时）
  验收指标同上，最终目标夏普 > 0.5
```

### 5.3 各阶段配置文件

每个阶段独立 yaml，放 `config/` 目录：

```
config/
  default.yaml          # 基础配置
  stage1_daily.yaml     # 阶段 1
  stage2_4h.yaml        # 阶段 2
  stage3_1h.yaml        # 阶段 3
  stage4_1min.yaml      # 阶段 4（最终目标）
```

### 5.4 total_timesteps 设置原则

`total_timesteps` 不应凭感觉设置，应基于**每根 bar 平均被用于梯度更新的次数**推导：

$$\text{有效更新次数/bar} = \frac{\text{total\_timesteps} \times \text{n\_epochs}}{\text{dataset\_bars} \times \text{batch\_size}}$$

目标：每根 bar **至少参与 50–100 次梯度更新**。

**各阶段推导：**

| 阶段 | 训练集 bars | n_epochs | batch_size | 达到 50 次更新/bar | 建议值 | 说明 |
|------|-----------|---------|-----------|-------------------|--------|------|
| 1（1d） | ~1,200 | 10 | 64 | 38,400 | **200K** | 300K 约走 250 遍，边际收益低，过拟合风险 |
| 2（4h） | ~4,800 | 10 | 64 | 153,600 | **600K** | ~125 遍，合理 |
| 3（1h） | ~28,000 | 10 | 64 | 896,000 | **1.5M** | ~54 遍，接近下限 |
| 4（1min） | ~400,000 | 10 | 128 | 6,400,000 | **5M–8M** | CNN 收敛慢，2M 仅走 5 遍严重不足 |

> stage4 数据量基于 `since: "2023-01-01"` + `limit: 500,000` × train_ratio 0.8 ≈ 400K bars。

**total_timesteps 作为上限预算，实际停止由 EvalCallback 决定：**

```python
EvalCallback(
    eval_env,
    eval_freq=20_000,        # 每 20K steps 评估一次
    n_eval_episodes=5,
    best_model_save_path=...,
    verbose=1,
)
```

**收敛的 TensorBoard 判断标准：**

| 指标 | 收敛信号 | 异常信号 |
|------|---------|---------|
| `rollout/ep_rew_mean` | 持续上升后趋于平稳 | 连续 500K steps 不变 → 降 lr 或停止 |
| `train/explained_variance` | 趋近 1.0 | 长期 < 0.5 → value network 未收敛 |
| `train/approx_kl` | 稳定在 0.01–0.03 | 持续 > 0.05 → 降 `clip_range` 或 lr |
| `train/entropy_loss` | 随阶段推进缓慢下降 | 骤降至 0 → `ent_coef` 可能太低 |

---

## 6. P3：训练期间膨胀手续费

### 6.1 原理

真实手续费 0.05% 对 1 分钟价格变动（≈0.02–0.05%）几乎不构成约束。模型发现"随机交易的期望 reward ≈ Hold"，无法区分好坏动作。

训练阶段临时将手续费放大 **10–20 倍**，强迫模型只在信号足够强时才触发交易。收敛后再 fine-tune 到真实费率。

### 6.2 分阶段手续费

| 训练阶段 | commission | 对应倍数 | 说明 |
|---------|-----------|---------|------|
| 阶段 1（日线） | 0.005 | 10× | 宽松约束，建立基础策略 |
| 阶段 2（4h） | 0.003 | 6× | 逐步收紧 |
| 阶段 3（1h） | 0.001 | 2× | 接近真实 |
| 阶段 4（1min） | 0.0005 | 1× | 真实费率 |

### 6.3 配置方式

在各阶段 yaml 的 `env` 块中覆盖：

```yaml
# config/stage1_daily.yaml
env:
  commission: 0.005   # 10× 膨胀
  trade_penalty_coef: 2.0
```

---

## 7. P4：调整 window_size

### 7.1 当前问题

`window_size=20` 在 1 分钟数据上仅覆盖 **20 分钟**，模型看不到任何中期趋势结构（均线交叉、MACD 背离等）。

### 7.2 各时间粒度建议

| 时间粒度 | 当前值 | 建议值 | 覆盖时间 | 理由 |
|---------|--------|--------|---------|------|
| 1d | 20 | 20 | 20 个交易日 | 合理 |
| 4h | 20 | 48 | 8 天 | 覆盖周线周期 |
| 1h | 20 | 120 | 5 天 | 覆盖工作周 |
| 1min | 20 | **120** | 2 小时 | 最低可用上下文 |

### 7.3 对 observation_space 的影响与热启动前提

`window_size` 变化会改变 `observation_space.shape`，直接用原生 `MlpPolicy` 时，policy 会先 `flatten` 观察矩阵再送入全连接层，第一层 `Linear` 的输入维度 = `window_size × n_features`。

- stage1：`20 × 18 ≈ 360`
- stage4（启用 P6 MTF）：`120 × 26 = 3120`

跨阶段输入维度不同 → 第一层权重形状不匹配，**`MlpPolicy` 下无法通过 `model.set_parameters()` 直接热启动**。

**因此课程学习链条（stage1 → stage4 权重迁移）依赖 P9 方案 A（`TradingCNN` 特征提取器）**：CNN 将 `(window_size, n_features)` 压缩为固定维度 `features_dim=256`，之后的 policy / value 头输入维度与 `window_size` 解耦，热启动才成立。

若坚持使用原生 `MlpPolicy`，stage 间只能重新从零训练，或必须固定 `window_size` 与特征集不变；本文档后续默认采用 P9 方案 A，不再单独讨论 MLP 下的热启动。

---

## 8. P5：动作惯性约束（Action Inertia）

### 8.1 问题

P1 的 `trade_penalty_coef` 惩罚"发生了交易"，但无法约束**方向频繁反转**：模型可能学到"尽量少交易，但一交易就反复横跳"。模型在 Hold(0)→Buy(1)→Sell(2)→Buy(1) 的连续切换中，每步的 `trade_penalty_coef` 独立计算，彼此不感知上下文。

### 8.2 原始公式的问题

用户建议的公式：

$$R_{adj} = R - \lambda \cdot |Action_t - Action_{t-1}|$$

在 `Discrete(3)` 下存在语义问题：

| 转变 | `\|At - At-1\|` | 实际语义 | 是否合理 |
|------|----------------|---------|---------|
| Hold→Buy | \|0-1\| = 1 | 首次进场 | 中等惩罚 |
| Hold→Sell | \|0-2\| = 2 | 首次做空（当前不支持） | 惩罚偏大 |
| Buy→Sell | \|1-2\| = 1 | 方向反转 | **惩罚应更大** |
| Buy→Hold | \|1-0\| = 1 | 继续持仓（实际是 Hold） | 不应惩罚 |
| Hold→Hold | 0 | 保持不变 | 正确 |

核心矛盾：`|Buy→Sell|=1` 和 `|Hold→Buy|=1` 相同，但方向反转的破坏性远大于首次进场。

### 8.3 改进方案：基于持仓状态变化的惩罚

将动作映射到**持仓状态**（`is_holding`），用状态变化而非动作编号差值来计算惩罚：

```
is_holding_{t}   = 1 if position > 0 after step t, else 0
state_changed    = is_holding_t != is_holding_{t-1}
penalty          = λ * state_changed
```

| 场景 | state_changed | 惩罚 |
|------|--------------|------|
| Hold → 继续 Hold（action=0） | False | 0 |
| 持仓 → 继续持仓（action=0） | False | 0 |
| 空仓 → 买入（action=1） | True | λ |
| 持仓 → 卖出（action=2） | True | λ |
| 反复横跳：买→卖→买（连续）| True × 每次 | λ × 次数 |

这比原始公式更精确：**只要持仓状态翻转就惩罚，无论从哪个方向翻转，惩罚幅度一致**。

### 8.4 在 Observation 中加入持仓历史

当前 `_get_obs` 末尾追加的是连续 `pos_ratio`（portfolio 中持仓占比）。在全仓/全空的 Discrete(3) 场景下，`pos_ratio` ≈ `is_holding`，但神经网络更容易利用**离散的 0/1 门控信号**来触发状态切换逻辑。

建议同时保留 `pos_ratio`（给模型感知资金利用率）并新增 `is_holding` 列（给模型感知当前状态）。

#### 8.4.1 错误做法：广播当前值到整个窗口

```python
# ❌ 错：window_size 行全部等于当前步的同一个值，成为常数列
state_col = np.full((window_size, 2), [pos_ratio, is_holding], dtype=np.float32)
```

CNN 看到这两列是常数（每个样本内部沿时间轴不变），对时间卷积而言梯度恒为零，**完全无法利用"过去持仓状态的变化序列"这一核心信号**——而这恰恰是惯性约束希望模型习得的东西（例如"刚翻转完的下一步应倾向保持"）。

#### 8.4.2 正确做法：env 内维护持仓历史 buffer

在 env 内维护一个长度为 `window_size` 的滚动 buffer，每步 step 后将当前 `(pos_ratio, is_holding)` 追加到末尾、最早一行丢弃；`_get_obs` 直接取整个 buffer 作为历史列，与 `obs` 的时间轴天然对齐。

**`envs/base_env.py`**

```python
def _reset_state(self):
    ...
    self._last_trade_cost = 0.0
    self._prev_is_holding = 0
    # 持仓历史 buffer：(window_size, 2)，列为 [pos_ratio, is_holding]
    # reset 时初始化为全 0，语义为"历史窗口内均为空仓"
    self._pos_history = np.zeros((self.window_size, 2), dtype=np.float32)

def _update_pos_history(self, price: float):
    portfolio = self._portfolio_value(price) + 1e-8
    pos_ratio = self.position * price / portfolio
    is_holding = float(self.position > 0)
    self._pos_history = np.roll(self._pos_history, -1, axis=0)
    self._pos_history[-1] = [pos_ratio, is_holding]
```

在 `step` 中，**先** `_execute_action` → **再** `current_step += 1` → **再** `_update_pos_history(new_price)` → **最后** `_get_obs()`，确保观察读到的是"截至当前步末"的持仓序列。

**`envs/stock_env.py` 的 `_get_obs`**

```python
# 末尾拼接持仓历史（而非广播当前值）
return np.concatenate([obs, self._pos_history], axis=1)
```

> 注：`n_features` 因此 +2（pos_ratio + is_holding），`_count_features` 的返回值从原 `+1` 改为 `+2`。

### 8.5 改动位置

**`envs/base_env.py`**

1. `__init__` 新增参数：

```python
def __init__(
    self,
    ...
    trade_penalty_coef: float = 1.0,
    action_inertia_coef: float = 0.0,   # ← 新增，λ；各阶段按 12.3 节配置覆盖
    ...
):
    ...
    self.action_inertia_coef = action_inertia_coef
```

2. `_reset_state` 记录上步持仓状态：

```python
def _reset_state(self):
    ...
    self._last_trade_cost = 0.0
    self._prev_is_holding = 0   # ← 新增
```

3. `step` 叠加惯性惩罚：

```python
# P1 已有
reward -= self._last_trade_cost * self.trade_penalty_coef

# P5 新增：持仓状态翻转惩罚
curr_is_holding = int(self.position > 0)
if curr_is_holding != self._prev_is_holding:
    reward -= self.action_inertia_coef
self._prev_is_holding = curr_is_holding
```

**`envs/stock_env.py`**

`_count_features` 的 `+1` 改为 `+2`：

```python
def _count_features(self) -> int:
    extra = [c for c in self.df.columns if c not in self.REQUIRED_COLS]
    return len(self.REQUIRED_COLS) + len(extra) + 2   # +2：pos_ratio + is_holding
```

### 8.6 参数调优建议

> **重要：λ 必须与 reward 量级协调**
>
> P8（第 13 节）将 reward 改为对数收益率，单步典型值 ±0.001–0.01。若 λ 沿用早期草案的 0.5，单次状态翻转惩罚将**比正常 reward 大 50–500 倍**，完全主导梯度信号，模型会直接退化为"全程 Hold"。
>
> 正确的标定原则：**λ 与 `commission × trade_penalty_coef` 同量级**，让"状态翻转的额外惩罚"约等于"一次手续费惩罚"，从而总进场成本 ≈ `2 × commission × trade_penalty_coef`。

| 参数 | 推荐初始值 | 说明 |
|------|-----------|------|
| `action_inertia_coef` (λ) | 见 12.3 节各阶段取值（≈ `commission × β`） | 状态翻转的额外惩罚；大幅偏离 log_return 量级会压垮学习信号 |
| `trade_penalty_coef` (β) | 1.0 | 与 P1 协同 |

**叠加效果**：进场/离场时 reward 同时被扣 `commission × β`（P1，每次交易）+ `λ`（P5，仅状态翻转时），两者同量级时总进场成本约等于 2 倍单次手续费，必须后续价格运动足够大才能回本，天然过滤噪声信号。

---

## 9. 配置文件变更汇总

### `config/default.yaml`（基础默认，不直接用于训练）

```yaml
training:
  algo: ppo
  total_timesteps: 500_000
  policy: MlpPolicy
  algo_kwargs:
    learning_rate: 3.0e-4
    n_steps: 2048
    batch_size: 64
    n_epochs: 10
    gamma: 0.99
    gae_lambda: 0.95
    clip_range: 0.2
    ent_coef: 0.01           # 由各 stage yaml 覆盖（P7）；此处仅作 fallback

env:
  window_size: 20
  initial_balance: 10000.0
  commission: 0.001
  trade_penalty_coef: 1.0       # ← 新增（P1）
  action_inertia_coef: 0.0      # ← 新增（P5），各 stage yaml 覆盖（与 commission×β 同量级）

stock:
  ticker: "AAPL"
  start: "2020-01-01"
  end: "2024-01-01"
  interval: "1d"
  train_ratio: 0.8

crypto:
  symbol: "BTC/USDT"
  exchange: "binance"
  timeframe: "1d"
  since: "2020-01-01"
  limit: 2000
  train_ratio: 0.8
```

### `config/stage4_1min.yaml`（最终目标配置）

```yaml
training:
  algo: ppo
  total_timesteps: 8_000_000   # 上限预算，EvalCallback 早停（P2 §5.4）
  policy: MlpPolicy
  algo_kwargs:
    learning_rate: 1.0e-4
    n_steps: 4096
    batch_size: 128
    n_epochs: 10
    gamma: 0.99
    gae_lambda: 0.95
    clip_range: 0.2
    ent_coef: 0.001

env:
  window_size: 120
  initial_balance: 10000.0
  commission: 0.0005
  trade_penalty_coef: 1.0
  action_inertia_coef: 0.001

crypto:
  symbol: "BTC/USDT"
  exchange: "binance"
  timeframe: "1m"
  since: "2023-01-01"
  limit: 500000
  train_ratio: 0.8
```

---

## 10. 验收标准

| 指标 | 最低要求 | 目标 |
|------|---------|------|
| 夏普比率（测试集） | > 0.3 | > 1.0 |
| 最大回撤 | < 40% | < 20% |
| 胜率 | > 40% | > 55% |
| 平均持仓步数（1min）| > 30 步 | > 120 步 |
| 总交易次数 / 总步数 | < 30% | < 10% |

> 平均持仓步数和交易频率是判断"收笼"效果的直接指标，比 reward 更直观。

---

## 11. P6：多时间粒度特征（Multi-Timeframe Features）

### 11.1 问题

P4 将 `window_size` 提升到 120，但这只是增加了 1m 数据的**深度**（回看更多根 1m 蜡烛）。1m 蜡烛的信噪比本质上很低：EMA 和 RSI 在 1m 粒度上极度震荡，很难反映真实趋势。

模型仅凭 1m 特征，只能看到高频随机波动的局部片段，无法判断当前是在一段 15m 级别上升趋势的中途，还是顶部回调。

### 11.2 核心设计决策：前视偏差问题

将高粒度数据合并到 1m 训练流程时，必须保证"在 1m 时刻 T，模型只能看到 T 之前**已收盘**的 5m/15m 蜡烛"。

**正确做法：预计算后前向填充（forward-fill）**

```
1m 时刻：  ...  08:03  08:04  08:05  08:06  08:07  08:08 ...
5m 已收盘：        ←── 08:00–08:04 ──→           ←── 08:05–08:09 ──→
forward-fill 后：  v(08:00蜡烛)  v(08:00蜡烛)  v(08:05蜡烛)  v(08:05蜡烛) ...
```

- 在 08:03，`m5_rsi` = 上一根完整 5m 蜡烛（08:00–08:04 尚未收盘，用 07:55–07:59 的值）
- 在 08:05，`m5_rsi` = 刚收盘的 08:00–08:04 蜡烛的 RSI
- 在 08:06，`m5_rsi` 不变（08:05–08:09 蜡烛未收盘，维持前值）

**前向填充天然避免前视偏差**，无需在 env 内部做任何实时聚合。

### 11.3 新增特征选择

选择信息密度最高、对趋势判断最有效的指标，不追求全量复制 1m 的指标集：

| 时间粒度 | 特征 | 列名 | 作用 |
|---------|------|------|------|
| 5m | EMA(9) | `m5_ema_9` | 短期趋势方向 |
| 5m | EMA(21) | `m5_ema_21` | 中期趋势方向 |
| 5m | RSI(14) | `m5_rsi` | 5m 动量强度 |
| 5m | MACD | `m5_macd` | 5m 趋势动量 |
| 5m | MACD Signal | `m5_macd_signal` | MACD 交叉信号 |
| 15m | EMA(9) | `m15_ema_9` | 中期趋势（15m） |
| 15m | RSI(14) | `m15_rsi` | 15m 超买超卖 |
| 15m | MACD | `m15_macd` | 15m 趋势动量 |

**不引入**：Stoch、ATR、BB、OBV（高粒度下 OBV 前向填充语义不明确；其余指标信息与 EMA/RSI/MACD 高度重叠）。

### 11.4 实现方案

**新函数：`utils/indicators.py` → `add_multi_timeframe_indicators`**

```python
def add_multi_timeframe_indicators(
    df_1m: pd.DataFrame,
    timeframes: list[str] = ["5min", "15min"],
) -> pd.DataFrame:
    """
    Compute indicators on higher timeframes and merge back to 1m index.

    Uses forward-fill to align higher-TF values to each 1m bar without
    look-ahead bias: at 1m bar T, the value reflects the last *closed*
    higher-TF candle strictly before T.

    Args:
        df_1m:       1-minute OHLCV DataFrame, must have a DatetimeIndex
                     or a 'timestamp' column that becomes the index.
        timeframes:  List of pandas offset strings, e.g. ["5min", "15min"]

    Returns:
        Original df_1m with added columns prefixed by timeframe,
        e.g. m5_ema_9, m15_rsi. NaN rows from indicator warm-up are
        dropped from the *start* only.
    """
    df = df_1m.copy()

    # Ensure DatetimeIndex for resampling
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")

    result = df.copy()

    for tf in timeframes:
        # Prefix: "5min" → "m5", "15min" → "m15"
        minutes = int("".join(filter(str.isdigit, tf)))
        prefix = f"m{minutes}"

        # Resample to higher TF (closed on left, label on left = bar open time)
        ohlcv_tf = df.resample(tf, closed="left", label="left").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()

        # Compute indicators on higher-TF candles
        ind = add_indicators(ohlcv_tf)   # reuse existing function

        # Select only the columns we want
        cols_to_keep = {
            "ema_9":        f"{prefix}_ema_9",
            "ema_21":       f"{prefix}_ema_21",
            "rsi":          f"{prefix}_rsi",
            "macd":         f"{prefix}_macd",
            "macd_signal":  f"{prefix}_macd_signal",
        }
        ind = ind[[c for c in cols_to_keep if c in ind.columns]].rename(
            columns=cols_to_keep
        )

        # Shift by 1 bar: value at bar T is from the bar that *closed before* T
        # This eliminates any intra-bar look-ahead.
        ind = ind.shift(1)

        # Reindex to 1m index and forward-fill
        ind = ind.reindex(result.index, method="ffill")
        result = pd.concat([result, ind], axis=1)

    result = result.reset_index()   # restore timestamp column
    return result.dropna().reset_index(drop=True)
```

> `shift(1)` 是关键：5m 蜡烛在 08:05 收盘，其指标值在 reindex 后从 08:05 开始前向填充，`shift(1)` 将这个值延迟到 08:06 才生效，保证 08:05 这根 1m 蜡烛本身不会看到"自己那根 5m 蜡烛"的完整指标。

### 11.5 归一化规则扩展

P0 引入的分组归一化方案自动覆盖新列，只需在 `stock_env.py` 的 `_get_obs` 扩展各组的列名列表：

```python
# 价格量纲类（÷ last_close）—— 扩展加入多时间粒度 EMA 和 MACD
PRICE_COLS = [
    "open", "high", "low", "close",
    "ema_9", "ema_21", "bb_upper", "bb_lower", "macd", "macd_signal", "atr",
    "m5_ema_9", "m5_ema_21", "m5_macd", "m5_macd_signal",   # ← 新增
    "m15_ema_9",              "m15_macd",                     # ← 新增
]

# 0–100 类（÷ 100）—— 扩展加入多时间粒度 RSI
for col in ["rsi", "stoch_k", "stoch_d", "m5_rsi", "m15_rsi"]:   # ← m5/m15 新增
    if col in frame.columns:
        frame[col] = frame[col] / 100.0
```

### 11.6 调用方式变更（`train.py` / `evaluate.py`）

数据准备阶段，在 `add_indicators` 之后追加调用：

```python
# 原有
df = add_indicators(df_raw)

# 新增（仅 1m 训练时）
if cfg["crypto"]["timeframe"] == "1m":
    df = add_multi_timeframe_indicators(df, timeframes=["5min", "15min"])
```

`env` 构造不需要任何改动，因为新列已经在 `df` 里，`_count_features` 会自动计入。

### 11.7 观察空间维度变化

| 阶段 | 原始特征数 | 新增 MTF 列 | 总特征数（不含 pos 列） |
|------|-----------|------------|----------------------|
| 日/4h/1h（不启用 MTF） | 16（OHLCV+10指标） | 0 | 16 |
| 1min（启用 MTF） | 16 | +8 | 24 |

总 `observation_space.shape` = `(window_size, n_features + 2)` = `(120, 26)`（含 P5 的 pos_ratio + is_holding）。

### 11.8 阶段适用性

| 训练阶段 | 是否启用 MTF | 理由 |
|---------|------------|------|
| 阶段 1（1d） | 否 | 日线已是最低粒度，无更高TF可用 |
| 阶段 2（4h） | 可选 | 可加日线 EMA，效果有限 |
| 阶段 3（1h） | 建议 | 加 4h 和日线 EMA/RSI |
| 阶段 4（1min） | **必须** | 加 5m 和 15m，核心降噪手段 |

---

## 12. P7：分阶段熵系数调度（ent_coef Scheduling）

### 12.1 问题

当前 `default.yaml` 里 `ent_coef: 0.005`，`stage4_1min.yaml` 里 `ent_coef: 0.001`，两个值都针对"已接近收敛"的假设——但第一阶段（1d）模型从零开始，极可能陷入"全程 Hold"局部最优：Hold 的期望 reward 始终非负，而任何交易尝试都触发 P1/P5 的惩罚，低熵系数让模型再也不敢探索出去。

反之，如果在 1min 最终阶段仍保持高熵，好不容易收敛的稳定策略会被持续的随机扰动破坏。

### 12.2 与 P5（action_inertia_coef）的对抗关系

这两个参数作用方向相反：

| 参数 | 效果 | 理想状态 |
|------|------|---------|
| `ent_coef` ↑ | 鼓励动作多样性，抵制过早收敛 | 早期探索阶段高 |
| `action_inertia_coef` ↑ | 惩罚状态切换，鼓励持仓稳定 | 晚期收敛阶段高 |

若两者同时过高：模型在"想随机"和"不敢动"之间撕裂，梯度方向混乱。  
若两者同时过低：无约束探索，等价于当前的问题状态。

正确做法是**跷跷板式调度**：随训练阶段推进，`ent_coef` 递减、`action_inertia_coef` 递增。

### 12.3 各阶段推荐值

λ 的标定原则：**与 `commission × trade_penalty_coef` 同量级**，与 P8 对数收益率的单步典型值（±0.001–0.01）兼容。大幅超出会压垮 reward 学习信号，退化为"全程 Hold"。

| 训练阶段 | `ent_coef` | `action_inertia_coef`（λ） | 对应 `commission × β` | 设计意图 |
|---------|-----------|----------------------------|----------------------|---------|
| 阶段 1（1d） | **0.03** | **0.0** | 0.005 | 强制探索各类动作，不施加惯性约束 |
| 阶段 2（4h） | **0.01** | **0.003** | 0.003 | 与单次手续费等量，轻度抑制方向反转 |
| 阶段 3（1h） | **0.005** | **0.001** | 0.001 | 与单次手续费等量，反转总成本 ≈ 2× 手续费 |
| 阶段 4（1min） | **0.001** | **0.001** | 0.0005 | 略高于手续费，1min 高频噪声下强化惯性约束 |

> **为什么阶段 1 设为 0.0（而非与 commission 同量级）**：此阶段模型需要大量尝试买/卖动作，`ent_coef=0.03` 本身就是"鼓励动作多样性"的强信号；若同时施加惯性惩罚（哪怕量级合理），会和高熵目标互相抵消，退化为噪声。
>
> **为什么阶段 4 不严格等于 commission**：1min 真实费率 0.0005 非常低（P3 设计意图即如此），若 λ 同等低，反转总成本仅 0.001，模型依然可能横跳；略微提高到 0.001（2× commission）给惯性更明确的语义，又不至于压垮 log_return 信号。

### 12.4 各阶段完整超参配置

以下为四个 stage yaml 的核心 `algo_kwargs` + `env` 部分（其余字段继承 `default.yaml`）：

**`config/stage1_daily.yaml`**
```yaml
training:
  total_timesteps: 200_000   # ~166 遍日线数据（P2 §5.4）
  n_envs: 4                  # P10：4 进程并行
  algo_kwargs:
    learning_rate: 3.0e-4
    n_steps: 512             # 4×512=2048 步/update（P10）
    batch_size: 256
    ent_coef: 0.03
    clip_range: 0.2

env:
  window_size: 20
  commission: 0.005
  trade_penalty_coef: 1.0
  action_inertia_coef: 0.0
  risk_aversion_coef: 1.0
  random_start: true         # P10：随机起点降低样本相关性

crypto:
  timeframe: "1d"
  since: "2020-01-01"
  limit: 2000
```

**`config/stage2_4h.yaml`**
```yaml
training:
  total_timesteps: 600_000   # ~125 遍 4h 数据（P2 §5.4）
  n_envs: 6
  algo_kwargs:
    learning_rate: 2.0e-4
    n_steps: 512             # 6×512=3072 步/update
    batch_size: 256
    ent_coef: 0.01
    clip_range: 0.2

env:
  window_size: 48
  commission: 0.003
  trade_penalty_coef: 1.0
  action_inertia_coef: 0.003
  risk_aversion_coef: 1.5
  random_start: true

crypto:
  timeframe: "4h"
  since: "2020-01-01"
  limit: 6000
```

**`config/stage3_1h.yaml`**
```yaml
training:
  total_timesteps: 1_500_000   # ~54 遍 1h 数据（P2 §5.4）
  n_envs: 8
  algo_kwargs:
    learning_rate: 1.5e-4
    n_steps: 512             # 8×512=4096 步/update
    batch_size: 512
    ent_coef: 0.005
    clip_range: 0.2

env:
  window_size: 120
  commission: 0.001
  trade_penalty_coef: 1.0
  action_inertia_coef: 0.001
  risk_aversion_coef: 2.0
  random_start: true

crypto:
  timeframe: "1h"
  since: "2020-01-01"
  limit: 30000
```

**`config/stage4_1min.yaml`**
```yaml
training:
  total_timesteps: 8_000_000   # 上限预算；~100 遍 1min 数据；EvalCallback 早停（P2 §5.4）
  n_envs: 8
  algo_kwargs:
    learning_rate: 1.0e-4
    n_steps: 1024            # 8×1024=8192 步/update
    batch_size: 512
    ent_coef: 0.001
    clip_range: 0.15

env:
  window_size: 120
  commission: 0.0005
  trade_penalty_coef: 1.0
  action_inertia_coef: 0.001
  risk_aversion_coef: 2.0
  random_start: true

crypto:
  timeframe: "1m"
  since: "2023-01-01"
  limit: 500000
```

### 12.5 热启动时的注意事项

SB3 加载上一阶段模型后继续训练时，policy 的内部状态（optimizer moments、log_std 等）会被继承。若新阶段 `ent_coef` 大幅下降，optimizer 动量中积累的"探索倾向"会在头几万步里造成剧烈更新。

建议在阶段切换时将 `learning_rate` **略微降低**（已在上表体现：3e-4 → 2e-4 → 1.5e-4 → 1e-4），让 optimizer 以更保守的步长消化旧动量。

---

## 13. P8：奖励函数重构——对数收益率 + 非对称风险惩罚

### 13.1 当前问题

**问题一：简单百分比收益率在极端行情下奖励突变**

当前：
```python
reward = (new_portfolio - prev_portfolio) / (prev_portfolio + 1e-8)
```

加密货币的"插针"（wick spike）行情中，单根 1m 蜡烛可出现 3%–10% 的瞬间回撤。简单百分比收益在这些蜡烛上的 reward 与正常蜡烛相比量级完全不同，造成梯度爆炸式波动。

简单收益率还有数学上的不对称性：下跌 50% 需要上涨 100% 才能回本，而 `reward = -0.5` 和 `reward = +1.0` 的量级不同，模型对等量的正负行情估值不一致。

**问题二：收益与亏损的敏感度对称**

模型对一步 +0.5% 和 -0.5% 的奖励绝对值相同，而实际交易中：
- 亏损比同等盈利更难弥补（因手续费、滑点、心理成本）
- 大幅回撤比同等涨幅更致命（爆仓风险、资金锁定）

这导致训练出的模型在回撤期"无动于衷"，缺乏对亏损的自我保护行为。

### 13.2 改进一：对数收益率

$$r_t = \ln\left(\frac{V_t}{V_{t-1}}\right)$$

其中 $V_t$ 为当前时刻组合价值。

**优势：**

| 属性 | 简单收益率 | 对数收益率 |
|------|----------|----------|
| 对称性 | 不对称（-50% 需 +100% 回本） | 对称（-ln2 = +ln2 绝对值相等） |
| 极端值处理 | 单步可达 ±100% | 自然压缩（ln(0.5) = -0.693，ln(2) = 0.693） |
| 可加性 | 不可直接累加 | 可累加（多步总收益 = 各步之和） |
| 梯度稳定性 | 插针导致突变 | 对数压缩后梯度更平稳 |

```python
# 替换原有 reward 计算
log_return = np.log(new_portfolio / max(prev_portfolio, 1e-8))
```

### 13.3 改进二：非对称风险惩罚

引入 `risk_aversion_coef`（α），对负收益施加额外惩罚：

$$R_{final} = r_t \cdot (1 + \alpha \cdot \mathbf{1}[r_t < 0])$$

等价逻辑：
- 正收益：`reward = log_return`（不变）
- 负收益：`reward = log_return × (1 + α)`（放大惩罚）

**非对称比例示意（α = 2.0）：**

| 场景 | log_return | 最终 reward | 比较 |
|------|-----------|------------|------|
| 上涨 0.3% | +0.003 | **+0.003** | — |
| 下跌 0.3% | -0.003 | **-0.009** | 亏损惩罚 = 盈利奖励的 3× |
| 插针 -2% | -0.020 | **-0.060** | 极端回撤被重点惩罚 |

> 选择线性非对称方案而非二次方（Sortino 的 $\sigma_{下行}^2$），原因：二次项在 1m 级别的小幅波动（±0.1%）中 squared 后量级趋近于 0，实际无效果；线性放大在任何量级下都有稳定的惩罚信号。

### 13.4 完整 reward 计算公式

整合 P1（手续费）、P5（惯性）、P8（对数 + 非对称）后：

$$R_{step} = \underbrace{r_t \cdot (1 + \alpha \cdot \mathbf{1}[r_t < 0])}_{\text{P8：非对称对数收益}} - \underbrace{c \cdot \beta}_{\text{P1：交易惩罚}} - \underbrace{\lambda \cdot \mathbf{1}[\Delta\text{holding}]}_{\text{P5：惯性惩罚}}$$

其中：
- $r_t = \ln(V_t / V_{t-1})$
- $\alpha$ = `risk_aversion_coef`
- $c$ = `commission`（有交易时），$\beta$ = `trade_penalty_coef`
- $\lambda$ = `action_inertia_coef`（持仓状态翻转时）

### 13.5 改动位置

**`envs/base_env.py`**

1. `__init__` 新增参数：

```python
def __init__(
    self,
    ...
    trade_penalty_coef: float = 1.0,
    action_inertia_coef: float = 0.5,
    risk_aversion_coef: float = 2.0,    # ← 新增（P8），推荐 1.0–3.0
    ...
):
    ...
    self.risk_aversion_coef = risk_aversion_coef
```

2. `step` 方法中 reward 计算全量替换：

```python
def step(self, action: int):
    price = float(self.df.loc[self.current_step, "close"])
    prev_portfolio = self._portfolio_value(price)

    self._execute_action(action, price)

    self.current_step += 1
    done = self.current_step >= len(self.df) - 1

    new_price = float(self.df.loc[self.current_step, "close"])
    new_portfolio = self._portfolio_value(new_price)

    # P8：对数收益率 + 非对称风险惩罚
    log_return = np.log(new_portfolio / max(prev_portfolio, 1e-8))
    if log_return < 0:
        reward = log_return * (1.0 + self.risk_aversion_coef)
    else:
        reward = log_return

    # P1：即时交易惩罚
    reward -= self._last_trade_cost * self.trade_penalty_coef

    # P5：动作惯性惩罚
    curr_is_holding = int(self.position > 0)
    if curr_is_holding != self._prev_is_holding:
        reward -= self.action_inertia_coef
    self._prev_is_holding = curr_is_holding

    self.total_reward += reward
    obs = self._get_obs()
    info = self._get_info()
    return obs, reward, done, False, info
```

### 13.6 参数调优建议

| 参数 | 推荐范围 | 说明 |
|------|---------|------|
| `risk_aversion_coef` | 1.0–3.0 | 1.0 = 2:1 损益比；2.0 = 3:1；3.0 = 4:1。高于 5.0 后模型可能过于保守，完全不做多 |

各阶段建议值与 P7 协调（早期低风险厌恶，允许模型大胆试错；晚期提高以约束回撤行为）：

| 训练阶段 | `risk_aversion_coef` | `ent_coef` | `action_inertia_coef` |
|---------|---------------------|-----------|----------------------|
| 阶段 1（1d） | 1.0 | 0.03 | 0.0 |
| 阶段 2（4h） | 1.5 | 0.01 | 0.003 |
| 阶段 3（1h） | 2.0 | 0.005 | 0.001 |
| 阶段 4（1min） | **2.0** | 0.001 | 0.001 |

> 阶段 4 的 `risk_aversion_coef` 不建议继续提高：1min 数据上的微幅噪声跌幅会被过度惩罚，导致模型宁可 Hold 也不愿意在轻微回调后重新买入。2.0（3:1 损益比）是 1min 场景的合理上限。

---

## 14. P9：策略网络架构选择——MLP vs CNN vs LSTM

### 14.1 当前问题

P4 将 `window_size` 提升到 120，`MlpPolicy` 的处理方式是先将 `(120, 26)` 的观察矩阵**直接 flatten 成 3120 维向量**，再输入全连接层。

这带来两个问题：

1. **时序结构丢失**：MLP 将 120 步前的数据和最新一步平等对待，无法学到"5 分钟前的趋势信号和现在的动作之间的因果关系"
2. **参数量和内存随 window 线性增长**：第一层输入维度 3120 → 隐层 256，参数量约 80 万，batch 训练时显存压力大

### 14.2 三种架构对比

| 架构 | `window_size` | 时序建模能力 | 内存 | 训练速度 | 实现复杂度 | 热启动兼容性 |
|------|-------------|------------|------|---------|-----------|------------|
| `MlpPolicy`（现状） | 120 | 弱（flatten） | 高 | 慢 | 低 | — |
| `MlpPolicy` + 1D-CNN 特征提取器 | 120 | 中（局部模式） | 中 | 中 | 中 | ✓ 观察空间不变 |
| `RecurrentPPO` + LSTM | **1–20** | 强（全局记忆） | 低 | 中 | 中 | ✗ 观察空间改变 |

### 14.3 方案 A：1D-CNN 特征提取器（推荐路线）

**适用场景**：想在保留现有课程学习结构的同时提升时序建模能力，且 stage1→stage4 权重热启动有效。

原理：在 MLP 之前插入一层 1D 卷积，将 `(120, 26)` 的时间序列压缩为固定维度的特征向量，再送入 MLP 决策头。

```
Observation (120, 26)
      ↓
Conv1D(filters=64, kernel=3, stride=1) → ReLU   # 捕捉 3-bar 局部模式
Conv1D(filters=64, kernel=5, stride=2) → ReLU   # 捕捉 5-bar 模式，步长2降采样
Conv1D(filters=32, kernel=3, stride=2) → ReLU
GlobalAveragePooling / Flatten
      ↓
Linear(256) → ReLU
      ↓
PPO Policy head (actor + critic)
```

SB3 支持通过 `policy_kwargs` 注入自定义特征提取器：

```python
# agents/trainer.py 中新增
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn


class TradingCNN(BaseFeaturesExtractor):
    """
    1D-CNN feature extractor for (window_size, n_features) observations.
    Processes the time dimension with causal-style convolutions.
    """

    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_steps, n_features = observation_space.shape

        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        # Compute output dim dynamically
        with torch.no_grad():
            sample = torch.zeros(1, n_features, n_steps)
            cnn_out = self.cnn(sample)
            cnn_flat = cnn_out.flatten(1).shape[1]

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cnn_flat, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (batch, window_size, n_features) → transpose to (batch, n_features, window_size)
        x = obs.permute(0, 2, 1)
        return self.linear(self.cnn(x))
```

在 `Trainer` 中注入：

```python
policy_kwargs = {
    "features_extractor_class": TradingCNN,
    "features_extractor_kwargs": {"features_dim": 256},
    "net_arch": [dict(pi=[128, 64], vf=[128, 64])],
}
model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, ...)
```

> `net_arch` 中 actor (`pi`) 和 critic (`vf`) 各用 `[128, 64]` 两层，比默认的 `[64, 64]` 稍大，因为 CNN 已做了特征压缩，MLP 头可以小一点。

**内存对比**（`window_size=120, n_features=26`）：

| | MlpPolicy（flatten） | CNN 方案 |
|--|---|---|
| 特征提取参数 | 3120×256 ≈ 80 万 | ~15 万（CNN + 线性层） |
| 单 batch (2048 steps) 显存 | ~50 MB | ~18 MB |

### 14.4 方案 B：RecurrentPPO + LSTM（激进路线）

**适用场景**：从头开始训练 stage4（1min），不依赖前序阶段权重热启动，追求最强时序建模。

核心思路：**彻底放弃大 window**，把观察空间改回 `(1, n_features)`（只看当前一根蜡烛），由 LSTM 的 hidden state 积累历史记忆。

```
每步输入: (1, 26)  ← 只有当前 bar
    ↓
LSTM(hidden=256, layers=1)
    ↓
Policy head
```

**依赖**：`pip install sb3-contrib`

```python
from sb3_contrib import RecurrentPPO

model = RecurrentPPO(
    "MlpLstmPolicy",
    env,
    n_steps=2048,
    batch_size=128,
    learning_rate=1e-4,
    ent_coef=0.001,
    policy_kwargs={"lstm_hidden_size": 256, "n_lstm_layers": 1},
    verbose=1,
)
```

env 的 `observation_space` 需要改为 `(n_features,)`（去掉 window 维度），`_get_obs` 只返回当前步特征。

**代价**：LSTM 的 BPTT（截断反向传播）需要正确处理 episode 边界（`done` 信号），`RecurrentPPO` 内部已处理，但 **stage3（MlpPolicy）的权重无法迁移到 stage4（LSTM）**，热启动链条在阶段切换时断裂。

### 14.5 推荐路线

| 目标 | 推荐方案 |
|------|---------|
| 保留课程学习热启动 | **方案 A（CNN）**，stage1→4 观察空间一致 |
| 最终 1min 最优效果 | 阶段 1–3 用方案 A 热启动到 stage3，**阶段 4 切换到方案 B（LSTM）从 stage3 做特征初始化**（只迁移特征提取权重，policy head 重新训练）|
| 快速验证框架可行性 | 保持 `MlpPolicy + window=20`，先验证 reward 曲线能上升，再考虑升级架构 |

### 14.6 对文档其他部分的影响

| 条目 | 方案 A 影响 | 方案 B 影响 |
|------|-----------|-----------|
| P4 `window_size` | 保持 120，CNN 处理效率提升 | 改为 1–20，LSTM 替代窗口 |
| P6 MTF 特征 | 不变，仍通过 forward-fill 加入 df | 不变（MTF 特征加入单步观察） |
| P7 热启动 | 完整保留 stage1→4 权重迁移 | stage3→4 仅可迁移特征层 |
| `train.py` | 加 `policy_kwargs` 传参 | 换用 `RecurrentPPO`，env obs 改形 |

### 14.7 本文档的默认假设

后续实现以**方案 A（CNN 特征提取器）**为默认路线，原因：
- 观察空间形状不变，P0–P8 的所有 env 改动无需调整
- 课程学习链条完整
- 1D-CNN 对于"识别技术形态（头肩顶、双底、均线交叉）"天然有效，与交易策略语义契合

如需切换到方案 B，`envs/base_env.py` 中的 `observation_space` 和 `_get_obs` 需要独立改造，建议另开分支实验。

---

## 15. P10：多进程并行训练（12核 CPU）

### 15.1 问题

当前 `Trainer` 使用单个 env（`DummyVecEnv([lambda: env])`），12 核 CPU 只有 1 核参与 rollout 采集，剩余 11 核空转。

更深层的问题：单 env 的所有 episode 从同一起点（`current_step = window_size`）开始，每次 reset 都重复相同的价格序列段，导致：
- PPO rollout buffer 内的样本高度相关（时序连续）
- 梯度估计方差大，更新方向不稳定
- 多样性不足，模型对某段历史过拟合

### 15.2 核心设计：SubprocVecEnv + 随机起点 reset

**层叠结构：**

```
SubprocVecEnv(n_envs=8)          ← 8 个子进程并行采集 rollout
    └── VecNormalize(P0b)        ← online 归一化，stats 跨 env 聚合
        └── Policy(TradingCNN)   ← 接收归一化后的 obs
```

**随机起点 reset（`random_start` 参数）：**

每个子进程的 env 在 `reset()` 时从训练集内随机选取起点，使 8 个 env 的经验时间段彼此错开，大幅降低 rollout buffer 内的时序相关性。

### 15.3 随机起点 reset 改动

**`envs/base_env.py`**

```python
def __init__(
    self,
    df,
    window_size: int = 20,
    initial_balance: float = 10_000.0,
    commission: float = 0.001,
    trade_penalty_coef: float = 1.0,
    action_inertia_coef: float = 0.5,
    risk_aversion_coef: float = 2.0,
    random_start: bool = False,      # ← 新增
    render_mode: str | None = None,
):
    ...
    self.random_start = random_start

def reset(self, *, seed=None, options=None):
    super().reset(seed=seed)
    self._reset_state()
    if self.random_start:
        # 随机起点：在 [window_size, len-101] 范围内均匀采样
        max_start = len(self.df) - 101
        self.current_step = int(
            self.np_random.integers(self.window_size, max_start)
        )
    obs = self._get_obs()
    return obs, self._get_info()
```

> `random_start=True` 仅用于训练 env；eval env 始终从 `window_size` 开始，保证评估的可重复性。

### 15.4 Trainer 并行化改动

**`agents/trainer.py`**

```python
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize


class Trainer:
    def __init__(
        self,
        env_fn,           # ← 改为工厂函数 callable，而非 env 实例
        eval_env_fn=None,
        n_envs: int = 1,  # ← 新增
        normalize_obs: bool = True,
        normalize_reward: bool = False,
        clip_obs: float = 10.0,
        ...
    ):
        # 训练环境：n_envs > 1 用多进程，=1 用单进程（调试方便）
        if n_envs > 1:
            venv = SubprocVecEnv([env_fn] * n_envs, start_method="fork")
        else:
            venv = DummyVecEnv([env_fn])

        if normalize_obs:
            venv = VecNormalize(
                venv,
                norm_obs=True,
                norm_reward=normalize_reward,
                clip_obs=clip_obs,
            )
        self.venv = venv

        # eval 环境：始终单进程 + 冻结 stats
        if eval_env_fn is not None:
            eval_venv = DummyVecEnv([eval_env_fn])
            if normalize_obs:
                eval_venv = VecNormalize(
                    eval_venv,
                    norm_obs=True,
                    norm_reward=False,
                    training=False,
                )
            self.eval_venv = eval_venv
```

> **为什么 eval env 用 `DummyVecEnv` 而非 `SubprocVecEnv`？**
> eval 是串行评估，多进程会增加 IPC 开销而无收益；且 eval env 的 stats 是冻结的，单进程更安全。

### 15.5 超参随 n_envs 的调整

PPO 的有效 rollout buffer 大小 = `n_envs × n_steps`。并行后每次 update 看到的总步数不变（可以适当降低 `n_steps`），但**批大小应随之扩大**以充分利用更多样的经验：

| 参数 | n_envs=1（当前） | n_envs=8（目标） | 调整原则 |
|------|----------------|----------------|---------|
| `n_steps` | 2048 / 4096 | **512 / 1024** | n_envs×n_steps ≈ 原值，减少每次 update 等待时间 |
| `batch_size` | 64 / 128 | **256 / 512** | 正比扩大，充分利用经验多样性；需整除 n_envs×n_steps |
| `learning_rate` | 3e-4 → 1e-4 | 不变 | PPO 对 lr 不敏感，无需 linear scaling |
| `total_timesteps` | 不变 | 不变 | 语义不变（env steps），wall-clock 时间缩短 ~8× |

**12核机器推荐 `n_envs` 配置：**

| 训练阶段 | 推荐 n_envs | 说明 |
|---------|-----------|------|
| 阶段 1（1d） | **4** | 数据量小（1200 bars），env 运行极快，4 个够用 |
| 阶段 2（4h） | **6** | 数据量适中 |
| 阶段 3（1h） | **8** | 充分并行 |
| 阶段 4（1min） | **8** | 数据量大，8 进程采集效率最高；留 4 核给主进程+OS |

### 15.6 更新后的各阶段核心 rollout 参数

| 阶段 | n_envs | n_steps | batch_size | n_envs×n_steps（每次update的步数） |
|------|--------|---------|-----------|----------------------------------|
| 1（1d） | 4 | 512 | 256 | 2048 |
| 2（4h） | 6 | 512 | 256 | 3072 |
| 3（1h） | 8 | 512 | 512 | 4096 |
| 4（1min） | 8 | 1024 | 512 | 8192 |

### 15.7 内存估算（1min 数据，n_envs=8）

| 项目 | 大小 | 说明 |
|------|------|------|
| df（400K bars × 26 features × float32） | ~41 MB | 每个子进程 fork 后共享 COW |
| 每个子进程 rollout buffer（1024 × 26） | ~0.1 MB | 可忽略 |
| VecNormalize running stats | < 1 MB | 主进程维护 |
| 总估算（8 进程） | **~50–80 MB** | Linux fork + COW 下 df 内存共享 |

> Linux 的 `fork()` 使用写时复制（COW），df 是只读的，8 个子进程**不会**各自复制一份 41MB，实际内存占用远低于 8×41MB。`start_method="fork"` 在 Linux 上正确，macOS 也支持；Windows 需改用 `"spawn"` 且 env_fn 必须可 pickle。

### 15.8 调用方式变更（`train.py`）

```python
# 原来
env = CryptoTradingEnv(df_train, ...)
trainer = Trainer(env, eval_env=eval_env, ...)

# 现在：传工厂函数，Trainer 内部负责构建多进程 env
def make_train_env():
    return CryptoTradingEnv(df_train, random_start=True, ...)

def make_eval_env():
    return CryptoTradingEnv(df_eval, random_start=False, ...)

trainer = Trainer(
    env_fn=make_train_env,
    eval_env_fn=make_eval_env,
    n_envs=8,
    normalize_obs=True,
    ...
)
```

---

## 16. 改动文件清单

| 文件 | 改动类型 | 涉及条目 | 说明 |
|------|---------|---------|------|
| `envs/stock_env.py` | 修改 | P0、P5、P6 | `_get_obs` 完整归一化 + MTF 列归一化扩展；`_count_features` 从 +1 改 +2；追加 `is_holding` 列 |
| `envs/base_env.py` | 修改 | P1、P5、P8、P10 | 加三个惩罚/风险系数 + `random_start` 参数；`_reset_state` 加 `_prev_is_holding`；`_execute_action` 记录 `_last_trade_cost`；`step` 全量重写；`reset` 支持随机起点 |
| `utils/indicators.py` | 修改 | P6 | 新增 `add_multi_timeframe_indicators` 函数 |
| `agents/trainer.py` | 修改 | P0b、P9、P10 | `Trainer.__init__` 接收 `env_fn`（工厂函数）而非 env 实例；加 `n_envs`、`SubprocVecEnv`、`VecNormalize`；新增 `TradingCNN`；`save()` 同步输出 `_vecnorm.pkl` |
| `train.py` / `evaluate.py` | 修改 | P0b、P6、P9、P10 | 改为传 `make_train_env` / `make_eval_env` 工厂函数；加载时恢复 `VecNormalize.load()`；1m 模式调用 MTF；stage4 传 CNN policy_kwargs |
| `config/default.yaml` | 修改 | P0b、P1、P5、P8、P10 | 加 `normalize_obs`、三个惩罚系数、`n_envs`、`random_start` 字段声明 |
| `config/stage1_daily.yaml` | 新增 | P2、P3、P7、P8、P10 | n_envs=4，n_steps=512，batch=256，ent=0.03，commission×10，inertia=0.0，risk_aversion=1.0 |
| `config/stage2_4h.yaml` | 新增 | P2、P3、P7、P8、P10 | n_envs=6，n_steps=512，batch=256，ent=0.01，commission×6，inertia=0.003，risk_aversion=1.5 |
| `config/stage3_1h.yaml` | 新增 | P2、P3、P7、P8、P10 | n_envs=8，n_steps=512，batch=512，ent=0.005，commission×2，inertia=0.001，risk_aversion=2.0 |
| `config/stage4_1min.yaml` | 新增 | P2、P3、P4、P6、P7、P8、P9、P10 | n_envs=8，n_steps=1024，batch=512，ent=0.001，真实费率，inertia=0.001，risk_aversion=2.0，MTF + CNN 启用 |
