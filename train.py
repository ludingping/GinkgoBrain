"""Entry point: train a trading agent from config."""
import argparse
import yaml
import pandas as pd

from utils import load_stock_data, load_crypto_data, add_indicators, add_multi_timeframe_indicators
from utils.db import read_ohlcv
from agents.ppo_shared import resample_ohlcv
from envs import StockTradingEnv, CryptoTradingEnv
from agents import Trainer, RLlibTrainer
from agents.trainer import TradingCNN


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
    parser = argparse.ArgumentParser(description="GinkgoBrain trainer")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--mode", choices=["stock", "crypto"], default="crypto")
    parser.add_argument(
        "--backend",
        choices=["sb3", "rllib"],
        default=None,
        help="RL backend (overrides config backend field)",
    )
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    with open("config/default.yaml") as f:
        cfg = yaml.safe_load(f)
    
    if args.config != "config/default.yaml":
        with open(args.config) as f:
            stage_cfg = yaml.safe_load(f)
        cfg = deep_merge(cfg, stage_cfg)

    env_cfg = cfg["env"]
    train_cfg = cfg["training"]
    backend = args.backend or train_cfg.get("backend", "sb3")

    if args.mode == "stock":
        s = cfg["stock"]
        df = add_indicators(load_stock_data(s["ticker"], s["start"], s["end"], s["interval"]))
        train_df, eval_df = split_df(df, s["train_ratio"])
        EnvCls = StockTradingEnv
        run_name = args.run_name or f"{s['ticker']}__{train_cfg['algo']}"
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
        run_name = args.run_name or f"{c['symbol'].replace('/', '')}_{train_cfg['algo']}"

    if backend == "rllib":
        env_config = {"df": train_df, **env_cfg}
        trainer = RLlibTrainer(
            env_cls=EnvCls,
            env_config=env_config,
            algo=train_cfg["algo"],
            run_name=run_name,
            algo_kwargs=train_cfg.get("algo_kwargs"),
            num_workers=train_cfg.get("num_workers", 0),
        )
        print(f"[RLlib] Training {run_name} for {train_cfg['n_iterations']} iterations...")
        trainer.train(
            n_iterations=train_cfg["n_iterations"],
            log_interval=train_cfg.get("log_interval", 10),
        )
    else:
        def make_train_env():
            return EnvCls(train_df, **env_cfg)

        def make_eval_env():
            eval_env_cfg = env_cfg.copy()
            eval_env_cfg['random_start'] = False
            return EnvCls(eval_df, **eval_env_cfg)

        n_envs = train_cfg.get("n_envs", 1)
        algo_kwargs = train_cfg.get("algo_kwargs", {})

        # Auto inject TradingCNN if using MlpPolicy and window_size > 1
        if train_cfg["policy"] == "MlpPolicy" and env_cfg.get("window_size", 1) > 1:
            if "policy_kwargs" not in algo_kwargs:
                algo_kwargs["policy_kwargs"] = {
                    "features_extractor_class": TradingCNN,
                    "features_extractor_kwargs": {"features_dim": 256},
                    "net_arch": [dict(pi=[128, 64], vf=[128, 64])],
                }

        trainer = Trainer(
            env_fn=make_train_env,
            eval_env_fn=make_eval_env,
            algo=train_cfg["algo"],
            run_name=run_name,
            policy=train_cfg["policy"],
            algo_kwargs=algo_kwargs,
            n_envs=n_envs,
            normalize_obs=True,
        )
        print(f"[SB3] Training {run_name} for {train_cfg['total_timesteps']:,} timesteps...")
        trainer.train(total_timesteps=train_cfg["total_timesteps"])

    trainer.save()


if __name__ == "__main__":
    main()
