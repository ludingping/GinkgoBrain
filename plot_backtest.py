import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from utils import load_stock_data, load_crypto_data, add_indicators, add_multi_timeframe_indicators
from utils.db import read_ohlcv
from agents.ppo_shared import resample_ohlcv
from envs import CryptoTradingEnv, StockTradingEnv
from agents import Trainer

def split_df(df: pd.DataFrame, ratio: float):
    n = int(len(df) * ratio)
    return df.iloc[:n].reset_index(drop=True), df.iloc[n:].reset_index(drop=True)

def deep_merge(base, update):
    for k, v in update.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            deep_merge(base[k], v)
        else:
            base[k] = v
    return base

def main():
    parser = argparse.ArgumentParser(description="GinkgoBrain visual evaluator")
    parser.add_argument("--model", required=True, help="Path to saved model (without .zip)")
    parser.add_argument("--config", default="config/stage1_daily.yaml")
    parser.add_argument("--mode", choices=["stock", "crypto"], default="crypto")
    args = parser.parse_args()

    # Load Config
    with open("config/default.yaml") as f:
        cfg = yaml.safe_load(f)
    if args.config != "config/default.yaml":
         with open(args.config) as f:
             stage_cfg = yaml.safe_load(f)
         cfg = deep_merge(cfg, stage_cfg)

    env_cfg = cfg["env"]
    train_cfg = cfg["training"]

    print("Loading data...")
    if args.mode == "stock":
        s = cfg["stock"]
        df = add_indicators(load_stock_data(s["ticker"], s["start"], s["end"], s["interval"]))
        train_df, eval_df = split_df(df, s["train_ratio"])
        EnvCls = StockTradingEnv
        price_col = "Close"
    else:
        c = cfg["crypto"]
        start_date = c.get("start_date", "2018-01-01 00:00:00")
        end_date = c.get("end_date", "2026-04-13 00:00:00")
        tz = c.get("timezone", "Asia/Shanghai")
        
        start_utc = pd.Timestamp(start_date, tz=tz).tz_convert("UTC").isoformat()
        end_utc   = pd.Timestamp(end_date,   tz=tz).tz_convert("UTC").isoformat()
        
        print(f"Loading crypto data from DB ({start_date} to {end_date})...")
        df_raw = read_ohlcv(
            c["symbol"], 
            start=start_utc, 
            end=end_utc,
            table=c.get("db_table", "public.crypto_kline_binance"), 
            only_closed=True
        )
        
        ts = df_raw["timestamp"]
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize("UTC")
        df_raw["timestamp"] = ts.dt.tz_convert(tz)
        
        tf_resample = c.get("timeframe", "1d").upper()
        if tf_resample == "1M":
            df_tf = df_raw
        else:
            df_tf = resample_ohlcv(df_raw, tf_resample)
            
        df = add_indicators(df_tf)
        if c["timeframe"].lower() == "1m":
            df = add_multi_timeframe_indicators(df, timeframes=["5min", "15min"])
            
        train_df, eval_df = split_df(df, c["train_ratio"])
        EnvCls = CryptoTradingEnv
        price_col = "close"

    eval_env_cfg = env_cfg.copy()
    eval_env_cfg['random_start'] = False
    
    def make_eval_env():
        return EnvCls(eval_df, **eval_env_cfg)
    
    print(f"Loading model: {args.model}")
    trainer = Trainer.load(args.model, make_eval_env, algo=train_cfg.get("algo", "ppo"), normalize_obs=True)
    
    bt_env = make_eval_env()
    
    print("Running backtest on validation set...")
    obs, info = bt_env.reset()
    portfolio_history = [info.get("portfolio_value", env_cfg.get("initial_balance", 10000))]
    done = False
    while not done:
        obs_n = trainer.venv.normalize_obs(obs)
        action, _ = trainer.model.predict(obs_n, deterministic=True)
        action = int(np.asarray(action).item())
        obs, reward, terminated, truncated, info = bt_env.step(action)
        portfolio_history.append(info.get("portfolio_value", portfolio_history[-1]))
        done = terminated or truncated

    window_size = env_cfg.get("window_size", 20)
    prices_eval = eval_df[price_col].values[window_size:]
    ts_eval = eval_df["timestamp"] if "timestamp" in eval_df.columns else eval_df.index
    ts_eval = ts_eval[window_size:]
    pv_curve = np.array(portfolio_history[1:])  # exclude reset step to align with steps
    trades = bt_env.trades

    initial_balance = env_cfg.get("initial_balance", 10000)

    bh_curve = initial_balance * (prices_eval / prices_eval[0])
    min_len = min(len(ts_eval), len(pv_curve), len(bh_curve))
    ts_eval = ts_eval[:min_len]
    pv_curve = pv_curve[:min_len]
    bh_curve = bh_curve[:min_len]

    peak = np.maximum.accumulate(pv_curve)
    drawdowns = (peak - pv_curve) / (peak + 1e-8)
    max_dd = drawdowns.max()

    step_rets = np.diff(pv_curve) / (pv_curve[:-1] + 1e-8)
    
    # Only calculate sharpe based on active trading steps, or annualized over the whole period
    # Annualizing a 4h timeframe: 6 * 365 = 2190 steps per year
    sharpe = (step_rets.mean() / (step_rets.std() + 1e-8)) * np.sqrt(6 * 365) if args.config.endswith("4h.yaml") else (step_rets.mean() / (step_rets.std() + 1e-8)) * np.sqrt(365)

    # Calculate accurate trade win rate (fraction of round-trip trades that were profitable)
    winning_trades = 0
    total_closed_trades = 0
    entry_price = 0
    
    n_trades   = len(trades)
    trade_freq = n_trades / len(pv_curve) if len(pv_curve) > 0 else 0
    holding_steps = 0
    last_buy = None
    
    for t in trades:
        if t["side"] == "buy":
            entry_price = t["price"]
            last_buy = t["step"]
        elif t["side"] == "sell" and entry_price > 0:
            if t["price"] > entry_price * (1 + 0.001 * 2): # consider commission
                winning_trades += 1
            total_closed_trades += 1
            if last_buy is not None:
                holding_steps += t["step"] - last_buy
            entry_price = 0
            last_buy = None
            
    win_rate = winning_trades / total_closed_trades if total_closed_trades > 0 else 0.0
    avg_hold = holding_steps / max(sum(1 for t in trades if t["side"] == "sell"), 1)

    total_return = pv_curve[-1] / initial_balance - 1
    bh_return = prices_eval[-1] / prices_eval[0] - 1

    print(f"{'═'*50}")
    print(f"  验证集回测结果  ({args.model})")
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

    print("Generating plot...")
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False, gridspec_kw={'height_ratios': [3, 1, 1]})
    fig.suptitle(f"Backtest Output: {os.path.basename(args.model)}", fontsize=14)

    ax0 = axes[0]
    ax0.plot(ts_eval, pv_curve, label="PPO Strategy", color="#2196f3", linewidth=1.2)
    ax0.plot(ts_eval, bh_curve, label="Buy & Hold", color="#ff9800", linewidth=1, linestyle="--", alpha=0.8)
    
    if len(ts_eval) > 0:
        ax0.fill_between(ts_eval, pv_curve, bh_curve, where=(pv_curve >= bh_curve), alpha=0.15, color="#4caf50")
        ax0.fill_between(ts_eval, pv_curve, bh_curve, where=(pv_curve < bh_curve), alpha=0.15, color="#ef5350")
    
    ax0.set_ylabel("Portfolio Value (USD)")
    ax0.grid(True, alpha=0.3)

    buy_x, buy_y, sell_x, sell_y = [], [], [], []
    for t in trades:
        step = t["step"] - window_size - 1
        if 0 <= step < min_len:
            if t["side"] == "buy":
                buy_x.append(ts_eval.iloc[step] if isinstance(ts_eval, pd.Series) else ts_eval[step])
                buy_y.append(pv_curve[step])
            elif t["side"] == "sell":
                sell_x.append(ts_eval.iloc[step] if isinstance(ts_eval, pd.Series) else ts_eval[step])
                sell_y.append(pv_curve[step])

    if buy_x:
        ax0.scatter(buy_x, buy_y, color="#4caf50", marker="^", s=40, zorder=5, label="Buy")
    if sell_x:
        ax0.scatter(sell_x, sell_y, color="#ef5350", marker="v", s=40, zorder=5, label="Sell")
    ax0.legend(loc="upper left", fontsize=9)

    ax1 = axes[1]
    ax1.fill_between(ts_eval, drawdowns * 100, alpha=0.6, color="#ef5350")
    ax1.axhline(max_dd * 100, color="red", linestyle="--", linewidth=1, label=f"Max Drawdown {max_dd:.2%}")
    ax1.set_ylabel("Drawdown (%)")
    ax1.invert_yaxis()
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    roll_w = max(20, min(60, len(step_rets) // 50))
    roll_ret = pd.Series(step_rets)
    roll_sharpe = (roll_ret.rolling(roll_w).mean() / (roll_ret.rolling(roll_w).std() + 1e-8))
    
    ax2 = axes[2]
    # step_rets is length N-1, ts_eval is N. Align them:
    ax2.plot(ts_eval.iloc[1:] if isinstance(ts_eval, pd.Series) else ts_eval[1:], roll_sharpe, color="#ba68c8", linewidth=1)
    ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax2.set_ylabel(f"Rolling Sharpe ({roll_w})")
    ax2.grid(True, alpha=0.3)

    for ax in axes:
        ax.tick_params(axis="x", rotation=30)

    fig.autofmt_xdate()
    plt.tight_layout()
    plot_file = f"{args.model}_backtest.png"
    plt.savefig(plot_file, dpi=120, bbox_inches="tight")
    print(f"Done. Chart saved to {plot_file}")

if __name__ == "__main__":
    main()
