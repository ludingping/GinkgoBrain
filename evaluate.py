"""Evaluate a saved model and print performance metrics."""
import argparse
import yaml

from utils import load_stock_data, load_crypto_data, add_indicators, evaluate_policy
from envs import StockTradingEnv, CryptoTradingEnv
from agents import Trainer


def main():
    parser = argparse.ArgumentParser(description="GinkgoBrain evaluator")
    parser.add_argument("--model", required=True, help="Path to saved .zip model")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--mode", choices=["stock", "crypto"], default="stock")
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    env_cfg = cfg["env"]

    if args.mode == "stock":
        s = cfg["stock"]
        df = add_indicators(load_stock_data(s["ticker"], s["start"], s["end"], s["interval"]))
        n = int(len(df) * s["train_ratio"])
        eval_df = df.iloc[n:].reset_index(drop=True)
        env = StockTradingEnv(eval_df, **env_cfg)
    else:
        c = cfg["crypto"]
        df = add_indicators(load_crypto_data(c["symbol"], c["exchange"], c["timeframe"], c["since"], c["limit"]))
        n = int(len(df) * c["train_ratio"])
        eval_df = df.iloc[n:].reset_index(drop=True)
        env = CryptoTradingEnv(eval_df, **env_cfg)

    algo = cfg["training"]["algo"]
    trainer = Trainer.load(args.model, env, algo=algo)

    metrics = evaluate_policy(trainer.model, env, n_episodes=args.episodes)
    print("\n=== Evaluation Results ===")
    for k, v in metrics.items():
        print(f"  {k:30s}: {v:.4f}")


if __name__ == "__main__":
    main()
