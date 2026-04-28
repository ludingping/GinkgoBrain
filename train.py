"""Entry point: train a trading agent from config."""
import argparse
import yaml
import pandas as pd

from utils import load_stock_data, load_crypto_data, add_indicators, add_multi_timeframe_indicators
from utils.data_loader import merge_contract_data
from utils.db import (
    read_funding,
    read_liquidation_agg,
    read_ohlcv,
    read_open_interest,
)
from utils.signals import add_contract_signals, add_signals
from agents.ppo_shared import resample_ohlcv
from envs import StockTradingEnv, CryptoTradingEnv, SignalLayeredEnv, load_signal_list
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
    parser.add_argument(
        "--paradigm",
        choices=["end_to_end", "signal_layered"],
        default=None,
        help="Training paradigm (overrides config `paradigm` field). "
             "`signal_layered` routes through envs.SignalLayeredEnv + "
             "utils.signals.add_signals; `end_to_end` keeps the legacy path.",
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--combiner",
        choices=["ppo", "gbdt"],
        default="ppo",
        help="Combiner type for signal_layered paradigm: "
             "'ppo' (default) trains PPO on SignalLayeredEnv; "
             "'gbdt' trains LightGBM Sharpe-regression combiner.",
    )
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
    paradigm = args.paradigm or cfg.get("paradigm", "end_to_end")

    if paradigm == "signal_layered":
        if args.combiner == "gbdt":
            return _train_gbdt(cfg, args)
        return _train_signal_layered(cfg, train_cfg, env_cfg, backend, args)

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
        
        tf_resample = c.get("timeframe", "1d").lower()
        if tf_resample == "1m":
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


def _load_crypto_df(c: dict, *, with_contracts: bool = False) -> pd.DataFrame:
    """Shared crypto DB → DataFrame loader used by both paradigms.

    Args:
        with_contracts: True 时跨库 join funding/OI/liquidation 到 OHLCV 并立即
            对合约列 fillna(0) —— 避免下游 add_indicators.dropna() 误删合约起点
            前的 OHLCV 行（与 signal_linear_baseline.load_df_from_config 对称）。
    """
    start_date = c.get("start_date", "2018-01-01 00:00:00")
    end_date = c.get("end_date", "2026-04-13 00:00:00")
    tz = c.get("timezone", "Asia/Shanghai")

    start_utc = pd.Timestamp(start_date, tz=tz).tz_convert("UTC").isoformat()
    end_utc = pd.Timestamp(end_date, tz=tz).tz_convert("UTC").isoformat()

    print(f"Loading crypto data from DB ({start_date} to {end_date})...")
    df_raw = read_ohlcv(
        c["symbol"],
        start=start_utc,
        end=end_utc,
        table=c.get("db_table", "public.crypto_kline_binance"),
        only_closed=True,
    )
    ts = df_raw["timestamp"]
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    df_raw["timestamp"] = ts.dt.tz_convert(tz)

    tf_resample = c.get("timeframe", "1d").lower()
    df_tf = df_raw if tf_resample == "1m" else resample_ohlcv(df_raw, tf_resample)

    if with_contracts:
        # ccxt 现货格式（合约表存储约定）；BTCUSDT → BTC/USDT
        ccxt_symbol = c["symbol"]
        if "/" not in ccxt_symbol and ccxt_symbol.endswith("USDT"):
            ccxt_symbol = f"{ccxt_symbol[:-4]}/USDT"

        print(f"Loading contract metadata for {ccxt_symbol}...")
        df_funding = read_funding(ccxt_symbol, start=start_utc, end=end_utc)
        df_oi      = read_open_interest(ccxt_symbol, start=start_utc, end=end_utc)
        df_liq     = read_liquidation_agg(ccxt_symbol, start=start_utc, end=end_utc)
        for sub in (df_funding, df_oi, df_liq):
            if sub.empty:
                continue
            sub_ts = sub["timestamp"]
            if sub_ts.dt.tz is None:
                sub_ts = sub_ts.dt.tz_localize("UTC")
            sub["timestamp"] = sub_ts.dt.tz_convert(tz)

        df_tf = merge_contract_data(
            df_tf, df_funding=df_funding, df_oi=df_oi, df_liq=df_liq,
        )
        for col in (
            "funding_rate", "sum_open_interest", "sum_open_interest_value",
            "liq_long_usd", "liq_short_usd", "liq_total_usd",
        ):
            if col in df_tf.columns:
                df_tf[col] = df_tf[col].fillna(0.0)
        if "funding_origin" in df_tf.columns:
            df_tf["funding_origin"] = df_tf["funding_origin"].fillna("")

    return df_tf


def _train_signal_layered(cfg, train_cfg, env_cfg, backend, args) -> None:
    """Phase 2 path: SignalLayeredEnv + sig_* features from utils.signals."""
    if args.mode != "crypto":
        raise NotImplementedError(
            "signal_layered paradigm currently only supports --mode crypto"
        )
    if backend != "sb3":
        raise NotImplementedError(
            "signal_layered paradigm currently only supports --backend sb3"
        )

    signals_cfg = cfg.get("signals") or {}
    signals_path = signals_cfg.get("config_path", "config/signals_v1.yaml")
    signal_cols = load_signal_list(signals_path)
    print(f"Loaded {len(signal_cols)} signals from {signals_path}")

    # 自动检测候选池是否含合约信号 → 决定是否跨库读 funding/OI/liquidation。
    # 任何 sig_funding_* / sig_oi_* / sig_liq_* 出现即触发；与 signal_linear_baseline
    # 的 --with-contracts 等价但无需 CLI 标志，由 yaml 信号列表驱动。
    needs_contracts = any(
        s.startswith(("sig_funding_", "sig_oi_", "sig_liq_")) for s in signal_cols
    )
    if needs_contracts:
        print("[train] candidate pool contains contract signals → enabling cross-db merge")

    c = cfg["crypto"]
    df_tf = _load_crypto_df(c, with_contracts=needs_contracts)
    df = add_indicators(df_tf)
    df = add_signals(df)
    df = add_contract_signals(df)

    missing = [s for s in signal_cols if s not in df.columns]
    if missing:
        raise ValueError(f"add_signals did not produce required columns: {missing}")

    train_df, eval_df = split_df(df, c.get("train_ratio", 0.75))

    # Drop legacy keys that default.yaml injects for the end-to-end env but
    # SignalLayeredEnv does not accept.
    _ALLOWED_ENV_KEYS = {
        "window_size", "initial_balance", "commission",
        "risk_aversion_coef", "excess_return_coef",
        "action_inertia_coef", "trade_penalty_coef",
        "stop_atr_mult", "stop_cooldown_steps",
        "random_start", "render_mode",
    }
    clean_env_cfg = {k: v for k, v in env_cfg.items() if k in _ALLOWED_ENV_KEYS}
    dropped = set(env_cfg) - _ALLOWED_ENV_KEYS
    if dropped:
        print(f"[signal_layered] ignoring legacy env keys: {sorted(dropped)}")

    def make_train_env():
        return SignalLayeredEnv(train_df, signal_cols=signal_cols, **clean_env_cfg)

    def make_eval_env():
        eval_kwargs = dict(clean_env_cfg)
        eval_kwargs["random_start"] = False
        return SignalLayeredEnv(eval_df, signal_cols=signal_cols, **eval_kwargs)

    algo_kwargs = dict(train_cfg.get("algo_kwargs", {}))
    # signal-layered 观察空间是浓缩特征，走默认 MLP；不注入 TradingCNN。

    run_name = args.run_name or (
        f"{c['symbol'].replace('/', '')}_signal_layered_{c.get('timeframe', '4h')}"
    )

    trainer = Trainer(
        env_fn=make_train_env,
        eval_env_fn=make_eval_env,
        algo=train_cfg["algo"],
        run_name=run_name,
        policy=train_cfg.get("policy", "MlpPolicy"),
        algo_kwargs=algo_kwargs,
        n_envs=train_cfg.get("n_envs", 1),
        normalize_obs=False,   # sig_* 已 ∈ [-1, 1]，state 已 clip
    )
    print(
        f"[SB3][signal_layered] Training {run_name} for "
        f"{train_cfg['total_timesteps']:,} timesteps..."
    )
    trainer.train(total_timesteps=train_cfg["total_timesteps"])
    trainer.save()


def _train_gbdt(cfg: dict, args) -> None:
    """Phase 4 path: LightGBM Sharpe-regression combiner (§6 of design doc)."""
    from agents.gbdt_combiner import GBDTCombiner
    from utils.indicators import add_indicators

    if args.mode != "crypto":
        raise NotImplementedError("gbdt combiner currently only supports --mode crypto")

    signals_cfg = cfg.get("signals") or {}
    signals_path = signals_cfg.get("config_path", "config/signals_v1.yaml")
    signal_cols = load_signal_list(signals_path)
    print(f"Loaded {len(signal_cols)} signals from {signals_path}")

    # 自动检测候选池是否含合约信号 → 决定是否跨库读 funding/OI/liquidation。
    # 任何 sig_funding_* / sig_oi_* / sig_liq_* 出现即触发；与 signal_linear_baseline
    # 的 --with-contracts 等价但无需 CLI 标志，由 yaml 信号列表驱动。
    needs_contracts = any(
        s.startswith(("sig_funding_", "sig_oi_", "sig_liq_")) for s in signal_cols
    )
    if needs_contracts:
        print("[train] candidate pool contains contract signals → enabling cross-db merge")

    c = cfg["crypto"]
    df_tf = _load_crypto_df(c, with_contracts=needs_contracts)
    df = add_indicators(df_tf)
    df = add_signals(df)
    df = add_contract_signals(df)

    missing = [s for s in signal_cols if s not in df.columns]
    if missing:
        raise ValueError(f"add_signals did not produce required columns: {missing}")

    train_ratio = c.get("train_ratio", 0.75)
    train_df, _ = split_df(df, train_ratio)
    print(f"Train period: {train_df['timestamp'].iloc[0]} → {train_df['timestamp'].iloc[-1]} "
          f"({len(train_df)} bars)")

    gbdt_cfg = cfg.get("gbdt", {})
    combiner = GBDTCombiner(gbdt_cfg)
    print(f"Training GBDT combiner (horizon={combiner.horizon}, "
          f"train_ratio={combiner.train_ratio})...")
    combiner.fit(train_df, signal_cols)

    best = combiner.best_iteration
    r2 = combiner.val_r2
    print(f"  Early stopping at tree {best}  |  val R² = {r2:.4f}")

    run_name = args.run_name or (
        f"{c['symbol'].replace('/', '')}_gbdt_{c.get('timeframe', '1d')}"
    )
    save_path = f"models/saved/{run_name}"
    combiner.save(save_path)
    print(f"Saved combiner to {save_path}.lgb + {save_path}.meta.json")


if __name__ == "__main__":
    main()
