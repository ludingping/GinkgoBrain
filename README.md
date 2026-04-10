# GinkgoBrain

基于 [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3) 的强化学习交易策略训练框架，支持**股票**和**加密货币**两类市场。

---

## 特性

- 标准 Gymnasium 环境，兼容所有 SB3 算法（PPO / A2C / SAC / TD3）
- 内置技术指标：EMA、MACD、RSI、布林带、ATR、OBV 等
- 股票数据：[yfinance](https://github.com/ranaroussi/yfinance)；加密货币数据：[ccxt](https://github.com/ccxt/ccxt)（支持 Binance 等主流交易所）
- 评估指标：夏普比率、最大回撤、胜率
- 全配置驱动，无需修改代码即可切换标的/算法/超参数
- TensorBoard 训练曲线 & 自动 checkpoint

---

## 项目结构

```
GinkgoBrain/
├── config/
│   └── default.yaml        # 统一配置文件
├── envs/
│   ├── base_env.py         # BaseTradingEnv（Gymnasium 接口）
│   ├── stock_env.py        # StockTradingEnv
│   └── crypto_env.py       # CryptoTradingEnv
├── agents/
│   └── trainer.py          # Trainer（封装 SB3 训练流程）
├── utils/
│   ├── data_loader.py      # 数据下载（yfinance / ccxt）
│   ├── indicators.py       # 技术指标计算
│   └── metrics.py          # 策略评估指标
├── strategies/             # 自定义策略/奖励扩展（预留）
├── notebooks/              # 分析 Notebook（预留）
├── tests/
│   └── test_envs.py        # Gymnasium 环境合规测试
├── train.py                # 训练入口
├── evaluate.py             # 评估入口
└── requirements.txt
```

---

## 快速开始

### 1. 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 训练

**股票（默认 AAPL，PPO）**

```bash
python train.py --mode stock
```

**加密货币（默认 BTC/USDT，PPO）**

```bash
python train.py --mode crypto
```

**自定义运行名称**

```bash
python train.py --mode crypto --run-name btc_sac_v1
```

**使用自定义配置文件**

```bash
python train.py --config config/my_config.yaml --mode stock
```

训练结束后模型保存至 `models/saved/<run_name>.zip`，TensorBoard 日志在 `models/logs/`。

### 3. 评估

```bash
python evaluate.py --model models/saved/BTCUSDT_ppo --mode crypto --episodes 10
```

输出示例：

```
=== Evaluation Results ===
  mean_reward                   :  0.1823
  std_reward                    :  0.0412
  mean_portfolio_return         :  0.2156
  sharpe_ratio                  :  1.8734
  max_drawdown                  : -0.0923
  win_rate                      :  0.7000
```

### 4. TensorBoard

```bash
tensorboard --logdir models/logs
```

---

## 配置说明

编辑 [config/default.yaml](config/default.yaml)：

```yaml
training:
  algo: ppo              # ppo | a2c | sac | td3
  total_timesteps: 500000
  algo_kwargs:
    learning_rate: 3.0e-4
    n_steps: 2048
    batch_size: 64

env:
  window_size: 20        # 观察窗口（时间步数）
  initial_balance: 10000.0
  commission: 0.001      # 手续费率

stock:
  ticker: "AAPL"
  start: "2020-01-01"
  end: "2024-01-01"
  interval: "1d"         # 1d | 1wk | 1mo
  train_ratio: 0.8

crypto:
  symbol: "BTC/USDT"
  exchange: "binance"    # 任意 ccxt 支持的交易所
  timeframe: "1d"        # 1d | 4h | 1h
  since: "2020-01-01"
  limit: 2000
  train_ratio: 0.8
```

---

## 环境说明

| 项目 | 描述 |
|------|------|
| 观察空间 | `(window_size, n_features)` 的 float32 矩阵，包含 OHLCV + 技术指标 + 持仓比例 |
| 动作空间 | `Discrete(3)`：0=持仓不变，1=全仓买入，2=全部卖出 |
| 奖励函数 | 单步组合价值变化率 `ΔPortfolio / Portfolio` |
| 手续费 | 股票默认 0.1%，加密货币默认 0.05% |

---

## 技术指标

| 类别 | 指标 |
|------|------|
| 趋势 | EMA(9)、EMA(21)、MACD、MACD Signal |
| 动量 | RSI(14)、Stochastic %K/%D |
| 波动率 | Bollinger Bands 上下轨、ATR |
| 成交量 | OBV |

---

## 测试

```bash
pytest tests/ -v
```

---

## 依赖

- Python >= 3.10
- stable-baselines3 >= 2.3.0
- gymnasium >= 0.29.0
- yfinance >= 0.2.40
- ccxt >= 4.3.0
- ta >= 0.11.0
