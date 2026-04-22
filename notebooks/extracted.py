# ── 数据参数 ────────────────────────────────────────────────────
SYMBOL      = "BTC/USDT"
START_DATE  = "2018-01-01 00:00:00"   # doc §12.4 stage1 起点
END_DATE    = "2026-04-13 00:00:00"
TIMEZONE    = "Asia/Shanghai"
DB_TABLE    = "public.crypto_kline_binance"
ONLY_CLOSED = True

TF_RESAMPLE = "1D"

# ── 环境参数（stage1 / 1d）─────────────────
WINDOW_SIZE         = 20
INITIAL_BALANCE     = 10_000.0
COMMISSION          = 0.005     # P3: stage1 模拟点差/滑点
TRADE_PENALTY_COEF  = 1.0        # P1: β
ACTION_INERTIA_COEF = 0.0      # P5: λ
RISK_AVERSION_COEF  = 1.0        # P8: α

TRAIN_RATIO = 0.8

# ── 并发参数（P10）──────────────────────────────────────────────
N_ENVS = 4

# ── PPO 超参数（doc §12.4 stage1）─────────────────────
TOTAL_TIMESTEPS = 200_000      # 上限预算；EvalCallback 早停
PPO_KWARGS = dict(
    learning_rate = 3.0e-04,
    n_steps       = 512,
    batch_size    = 256,
    n_epochs      = 10,
    gamma         = 0.99,
    gae_lambda    = 0.95,
    clip_range    = 0.2,
    ent_coef      = 0.03,
    vf_coef       = 0.5,
    max_grad_norm = 0.5,
)

# ── CNN + Policy head（P9）─────────────────────────────────────
FEATURES_DIM = 256
NET_ARCH     = dict(pi=[128, 64], vf=[128, 64])

# ── 课程热启动 ─────────────────────────────────────────────────
WARM_START_FROM = None             # stage1：无热启动

# ── 输出路径 ────────────────────────────────────────────────────
RUN_NAME  = "BTCUSDT__ppo_stage1_1d"
MODEL_DIR = "../models/saved"
LOG_DIR   = "../models/logs"

#---
import sys
sys.path.insert(0, "..")

import numpy as np
import pandas as pd
import torch
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

from utils.db import read_ohlcv
from agents.ppo_shared import (
    add_indicators,
    add_multi_timeframe_indicators,
    resample_ohlcv,
    CryptoPPOEnv,
    TradingCNN,
    warm_start_policy,
    FEATURE_COLS_BASE,
    FEATURE_COLS_MTF,
    STAGE3_TO_STAGE4_CHANNEL_MAP,
)

# ── OHLCV 加载（1m 原始） ──────────────────────────────────────
tz = TIMEZONE
start_utc = pd.Timestamp(START_DATE, tz=tz).tz_convert("UTC").isoformat()
end_utc   = pd.Timestamp(END_DATE,   tz=tz).tz_convert("UTC").isoformat()

df_raw = read_ohlcv(SYMBOL, start=start_utc, end=end_utc,
                    table=DB_TABLE, only_closed=ONLY_CLOSED)
ts = df_raw["timestamp"]
if ts.dt.tz is None:
    ts = ts.dt.tz_localize("UTC")
df_raw["timestamp"] = ts.dt.tz_convert(tz)

print(f"原始 1m 行数 : {len(df_raw):,}")
print(f"时间范围     : {df_raw['timestamp'].min()} → {df_raw['timestamp'].max()}")
df_raw.head(3)

#---
# ── 1m → 1d resample，再计算基础指标 ───────────
df_tf = resample_ohlcv(df_raw, TF_RESAMPLE)
df_full = add_indicators(df_tf.set_index("timestamp")).dropna().reset_index()

FEATURE_COLS = FEATURE_COLS_BASE
print(f"resample 后行数 : {len(df_full):,}  ({TF_RESAMPLE})")
print(f"特征列数         : {len(FEATURE_COLS)}  (基础)")
print(FEATURE_COLS)
df_full.head(3)

#---
feat_matrix = df_full[FEATURE_COLS].values.astype(np.float32)
timestamps  = df_full["timestamp"].reset_index(drop=True)

n_total = len(feat_matrix)
n_train = int(n_total * TRAIN_RATIO)

train_feat, eval_feat = feat_matrix[:n_train], feat_matrix[n_train:]
train_ts,   eval_ts   = timestamps.iloc[:n_train], timestamps.iloc[n_train:]

print(f"总步数  : {n_total:,}")
print(f"训练集  : {n_train:,} 步  ({train_ts.iloc[0]}  →  {train_ts.iloc[-1]})")
print(f"验证集  : {n_total - n_train:,} 步  ({eval_ts.iloc[0]}  →  {eval_ts.iloc[-1]})")

assert n_train > WINDOW_SIZE + 100 and (n_total - n_train) > WINDOW_SIZE + 100, \
    "数据集太短，请缩小 WINDOW_SIZE 或扩大日期范围"
print("✓ 数据集检查通过")

#---
Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)


def make_train_env():
    env = CryptoPPOEnv(
        train_feat, FEATURE_COLS,
        window_size         = WINDOW_SIZE,
        initial_balance     = INITIAL_BALANCE,
        commission          = COMMISSION,
        trade_penalty_coef  = TRADE_PENALTY_COEF,
        action_inertia_coef = ACTION_INERTIA_COEF,
        risk_aversion_coef  = RISK_AVERSION_COEF,
        random_start        = True,        # P10
    )
    return Monitor(env)


def make_eval_env():
    env = CryptoPPOEnv(
        eval_feat, FEATURE_COLS,
        window_size         = WINDOW_SIZE,
        initial_balance     = INITIAL_BALANCE,
        commission          = COMMISSION,
        trade_penalty_coef  = TRADE_PENALTY_COEF,
        action_inertia_coef = ACTION_INERTIA_COEF,
        risk_aversion_coef  = RISK_AVERSION_COEF,
        random_start        = False,       # eval 固定起点
    )
    return Monitor(env)


# ── 训练 venv：SubprocVecEnv + VecNormalize ────────────────────
train_vec = SubprocVecEnv([make_train_env] * N_ENVS, start_method="fork")
train_vec = VecNormalize(
    train_vec,
    norm_obs    = True,
    norm_reward = False,                    # 保持 P1/P5/P8 精确标定
    clip_obs    = 10.0,
    gamma       = PPO_KWARGS["gamma"],
)

# ── eval venv：冻结 stats ──────────────────────────────────────
eval_vec = DummyVecEnv([make_eval_env])
eval_vec = VecNormalize(
    eval_vec,
    norm_obs    = True,
    norm_reward = False,
    clip_obs    = 10.0,
    training    = False,
    gamma       = PPO_KWARGS["gamma"],
)

callbacks = [
    EvalCallback(
        eval_vec,
        best_model_save_path = f"{MODEL_DIR}/{RUN_NAME}",
        log_path             = f"{LOG_DIR}/{RUN_NAME}",
        eval_freq            = 5_000,
        n_eval_episodes      = 1,
        deterministic        = True,
        render               = False,
        verbose              = 1,
    ),
    CheckpointCallback(
        save_freq   = 50_000,
        save_path   = f"{MODEL_DIR}/{RUN_NAME}/checkpoints",
        name_prefix = RUN_NAME,
    ),
]

policy_kwargs = dict(
    features_extractor_class  = TradingCNN,
    features_extractor_kwargs = dict(features_dim=FEATURES_DIM),
    net_arch                  = NET_ARCH,
)

model = PPO(
    "MlpPolicy",
    train_vec,
    device          = "auto",
    tensorboard_log = LOG_DIR,
    policy_kwargs   = policy_kwargs,
    verbose         = 1,
    **PPO_KWARGS,
)

print(f"观测形状   : {model.observation_space.shape}")
print(f"并行环境   : {N_ENVS}  (每次 update = {N_ENVS * PPO_KWARGS['n_steps']:,} 步)")
print(f"特征提取器 : TradingCNN  (features_dim={FEATURES_DIM}, AdaptiveAvgPool 输出 {TradingCNN.POOL_SIZE})")
print(f"Policy 头  : pi={NET_ARCH['pi']}, vf={NET_ARCH['vf']}")
print(f"训练步数   : {TOTAL_TIMESTEPS:,}  (上限；EvalCallback 早停)")
print(f"TensorBoard: tensorboard --logdir {LOG_DIR}")

#---
model.learn(
    total_timesteps = TOTAL_TIMESTEPS,
    callback        = callbacks,
    tb_log_name     = RUN_NAME,
    progress_bar    = True,
)

# P0b §3.4.5：VecNormalize running stats 必须随模型权重一起保存
model.save(f"{MODEL_DIR}/{RUN_NAME}/final")
train_vec.save(f"{MODEL_DIR}/{RUN_NAME}/final_vecnorm.pkl")

print(f"模型   → {MODEL_DIR}/{RUN_NAME}/final.zip")
print(f"归一化 → {MODEL_DIR}/{RUN_NAME}/final_vecnorm.pkl")
print(f"最佳   → {MODEL_DIR}/{RUN_NAME}/best_model.zip   (供下一阶段热启动)")

#---
# ── 加载模型 + 训练时的 VecNormalize 统计量 ──────────────────
best_model_path = f"{MODEL_DIR}/{RUN_NAME}/best_model"
vecnorm_path    = f"{MODEL_DIR}/{RUN_NAME}/final_vecnorm.pkl"

# 提取 obs_rms（避免 DummyVecEnv 的 auto-reset 抹掉 portfolio_history）
_loader     = VecNormalize.load(vecnorm_path, DummyVecEnv([make_eval_env]))
_obs_mean   = _loader.obs_rms.mean.astype(np.float32)
_obs_var    = _loader.obs_rms.var.astype(np.float32)
_obs_clip   = float(_loader.clip_obs)
_loader.close()

def _apply_vecnorm(obs: np.ndarray) -> np.ndarray:
    return np.clip(
        (obs - _obs_mean) / np.sqrt(_obs_var + 1e-8),
        -_obs_clip, _obs_clip,
    ).astype(np.float32)

eval_model = PPO.load(best_model_path, device="auto")
print(f"加载模型   : {best_model_path}.zip")
print(f"加载归一化 : {vecnorm_path}")

# ── 单 env 确定性 rollout ──────────────────────────────────────
bt_env = CryptoPPOEnv(
    eval_feat, FEATURE_COLS,
    window_size         = WINDOW_SIZE,
    initial_balance     = INITIAL_BALANCE,
    commission          = COMMISSION,
    trade_penalty_coef  = TRADE_PENALTY_COEF,
    action_inertia_coef = ACTION_INERTIA_COEF,
    risk_aversion_coef  = RISK_AVERSION_COEF,
    random_start        = False,
)
obs, _ = bt_env.reset()
done = False
while not done:
    obs_n = _apply_vecnorm(obs)
    action, _ = eval_model.predict(obs_n, deterministic=True)
    obs, reward, done, truncated, info = bt_env.step(int(np.asarray(action).item()))

pv_curve    = np.array(bt_env.portfolio_history)
trades      = bt_env.trades
eval_close  = eval_feat[:, FEATURE_COLS.index("close")]
prices_eval = eval_close[WINDOW_SIZE + 1:]   # 对齐 pv_curve

# ── 指标计算 ───────────────────────────────────────────────────
total_return = pv_curve[-1] / INITIAL_BALANCE - 1
peak         = np.maximum.accumulate(pv_curve)
drawdowns    = (peak - pv_curve) / (peak + 1e-8)
max_dd       = drawdowns.max()

step_rets = np.diff(pv_curve) / (pv_curve[:-1] + 1e-8)
sharpe    = (step_rets.mean() / (step_rets.std() + 1e-8)) * np.sqrt(365)
win_rate  = float((step_rets > 0).mean())

n_trades   = len(trades)
trade_freq = n_trades / len(pv_curve)
holding_steps = 0
last_buy = None
for t in trades:
    if t["side"] == "buy":
        last_buy = t["step"]
    elif t["side"] == "sell" and last_buy is not None:
        holding_steps += t["step"] - last_buy
        last_buy = None
avg_hold = holding_steps / max(sum(1 for t in trades if t["side"] == "sell"), 1)

bh_return = prices_eval[-1] / prices_eval[0] - 1

print(f"{'═'*50}")
print(f"  验证集回测结果  ({RUN_NAME})")
print(f"{'═'*50}")
print(f"  总收益率     : {total_return:+.2%}")
print(f"  买入持有     : {bh_return:+.2%}  (baseline)")
print(f"  最大回撤     : {max_dd:.2%}")
print(f"  夏普比率     : {sharpe:.2f}")
print(f"  胜率         : {win_rate:.2%}")
print(f"  交易次数     : {n_trades}")
print(f"  交易频率     : {trade_freq:.2%}")
print(f"  平均持仓步数 : {avg_hold:.1f}  (参考: > 5)")
print(f"{'═'*50}")

#---
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)
fig.suptitle(f"{SYMBOL}  PPO {RUN_NAME} 验证集回测", fontsize=14)

ts_eval  = eval_ts.iloc[WINDOW_SIZE + 1:].reset_index(drop=True)
bh_curve = INITIAL_BALANCE * (prices_eval / prices_eval[0])

ax0 = axes[0]
ax0.plot(ts_eval, pv_curve, label="PPO 策略", color="#2196f3", linewidth=1.2)
ax0.plot(ts_eval, bh_curve, label="买入持有", color="#ff9800", linewidth=1, linestyle="--", alpha=0.8)
ax0.fill_between(ts_eval, pv_curve, bh_curve, where=pv_curve >= bh_curve, alpha=0.15, color="#4caf50")
ax0.fill_between(ts_eval, pv_curve, bh_curve, where=pv_curve <  bh_curve, alpha=0.15, color="#ef5350")
ax0.set_ylabel("资产价值 (USD)")
ax0.grid(True, alpha=0.3)

buy_idx  = [t["step"] - WINDOW_SIZE for t in trades
            if t["side"] == "buy"  and 0 <= t["step"] - WINDOW_SIZE < len(ts_eval)]
sell_idx = [t["step"] - WINDOW_SIZE for t in trades
            if t["side"] == "sell" and 0 <= t["step"] - WINDOW_SIZE < len(ts_eval)]
if buy_idx:
    ax0.scatter(ts_eval.iloc[buy_idx],  pv_curve[buy_idx],
                color="#4caf50", marker="^", s=40, zorder=5, label="买入")
if sell_idx:
    ax0.scatter(ts_eval.iloc[sell_idx], pv_curve[sell_idx],
                color="#ef5350", marker="v", s=40, zorder=5, label="卖出")
ax0.legend(loc="upper left", fontsize=9)

ax1 = axes[1]
ax1.fill_between(ts_eval, drawdowns * 100, alpha=0.6, color="#ef5350")
ax1.axhline(max_dd * 100, color="red", linestyle="--", linewidth=1,
            label=f"最大回撤 {max_dd:.2%}")
ax1.set_ylabel("回撤 (%)")
ax1.invert_yaxis()
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

roll_w = max(20, min(60, len(step_rets) // 50))
roll_ret    = pd.Series(step_rets)
roll_sharpe = (roll_ret.rolling(roll_w).mean()
               / (roll_ret.rolling(roll_w).std() + 1e-8))
ax2 = axes[2]
ax2.plot(ts_eval.iloc[1:], roll_sharpe, color="#ba68c8", linewidth=1)
ax2.axhline( 0, color="gray",    linestyle="--", linewidth=0.8, alpha=0.5)
ax2.set_ylabel(f"滚动 Sharpe ({roll_w}步, 未年化)")
ax2.grid(True, alpha=0.3)

for ax in axes:
    ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.savefig(f"{MODEL_DIR}/{RUN_NAME}_backtest.png", dpi=120, bbox_inches="tight")
plt.show()
print(f"图表已保存: {MODEL_DIR}/{RUN_NAME}_backtest.png")

#---
