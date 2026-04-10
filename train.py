"""Entry point: train a trading agent from config."""
import argparse
import yaml
import pandas as pd

from utils import load_stock_data, load_crypto_data, add_indicators
from envs import StockTradingEnv, CryptoTradingEnv
from agents import Trainer


def split_df(df: pd.DataFrame, ratio: float):
    n = int(len(df) * ratio)
    return df.iloc[:n].reset_index(drop=True), df.iloc[n:].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="GinkgoBrain trainer")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--mode", choices=["stock", "crypto"], default="stock")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    env_cfg = cfg["env"]
    train_cfg = cfg["training"]

    if args.mode == "stock":
        s = cfg["stock"]
        df = add_indicators(load_stock_data(s["ticker"], s["start"], s["end"], s["interval"]))
        train_df, eval_df = split_df(df, s["train_ratio"])
        EnvCls = StockTradingEnv
        run_name = args.run_name or f"{s['ticker']}__{train_cfg['algo']}"
    else:
        c = cfg["crypto"]
        df = add_indicators(load_crypto_data(c["symbol"], c["exchange"], c["timeframe"], c["since"], c["limit"]))
        train_df, eval_df = split_df(df, c["train_ratio"])
        EnvCls = CryptoTradingEnv
        run_name = args.run_name or f"{c['symbol'].replace('/', '')}_{train_cfg['algo']}"

    train_env = EnvCls(train_df, **env_cfg)
    eval_env = EnvCls(eval_df, **env_cfg)

    trainer = Trainer(
        train_env,
        eval_env=eval_env,
        algo=train_cfg["algo"],
        run_name=run_name,
        policy=train_cfg["policy"],
        algo_kwargs=train_cfg.get("algo_kwargs"),
    )

    print(f"Training {run_name} for {train_cfg['total_timesteps']:,} timesteps...")
    trainer.train(total_timesteps=train_cfg["total_timesteps"])
    trainer.save()


if __name__ == "__main__":
    main()
