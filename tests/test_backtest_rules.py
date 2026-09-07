"""scripts/backtest_rules.py — model-free gate × signal rule backtester (2026-09-06)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.backtest_rules import (
    RuleSpec, enrich, evaluate_acceptance, make_act_fn, parse_rule,
    rule_contract_sources, run_rules_on_slice, parse_period,
)


class _Env:
    def __init__(self, i: int):
        self.current_step = i


# ── parsing ──────────────────────────────────────────────────────────────────

def test_parse_rule_gate_only_and_const() -> None:
    assert parse_rule("gate_only") == RuleSpec("gate_only", {})
    assert parse_rule("const:level=2").params == {"level": 2}


def test_parse_rule_gate_reduce_defaults_side_above() -> None:
    sp = parse_rule("gate_reduce:signal=sig_funding_current,thr=0.667,level=2")
    assert sp.params == {"signal": "sig_funding_current", "thr": 0.667, "level": 2, "side": "above"}
    assert sp.label == "gate_reduce:signal=sig_funding_current,thr=0.667,level=2,side=above"


@pytest.mark.parametrize("bad", [
    "banana", "gate_reduce:signal=x", "gate_reduce:signal=x,thr=1,level=7",
    "gate_reduce:signal=x,thr=1,level=2,side=sideways", "const", "const:level",
])
def test_parse_rule_rejects_bad_specs(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_rule(bad)


def test_parse_period() -> None:
    assert parse_period("val=2024-10-01:2026-04-13") == ("val", "2024-10-01", "2026-04-13")
    assert parse_period("test=2026-04-13:") == ("test", "2026-04-13", None)
    with pytest.raises(ValueError):
        parse_period("2024-10-01")


# ── act_fn semantics ─────────────────────────────────────────────────────────

def test_gate_reduce_act_fn_levels() -> None:
    df = pd.DataFrame({"sig_x": [0.0, 0.9, 0.9, -0.9]})
    gate = np.array([True, True, False, True])
    act = make_act_fn(parse_rule("gate_reduce:signal=sig_x,thr=0.5,level=2"), df, gate)
    assert [act(None, _Env(i))[0] for i in range(4)] == [4, 2, 0, 4]
    act_b = make_act_fn(parse_rule("gate_reduce:signal=sig_x,thr=-0.5,level=1,side=below"), df, gate)
    assert [act_b(None, _Env(i))[0] for i in range(4)] == [4, 4, 0, 1]


def test_gate_reduce_unknown_signal_raises() -> None:
    with pytest.raises(ValueError, match="not in data"):
        make_act_fn(parse_rule("gate_reduce:signal=nope,thr=0,level=2"),
                    pd.DataFrame({"a": [1.0]}), np.array([True]))


def test_rule_contract_sources_from_signal_or_raw_column() -> None:
    specs = [parse_rule("gate_only"),
             parse_rule("gate_reduce:signal=sig_funding_current,thr=0.5,level=2"),
             parse_rule("gate_reduce:signal=sum_open_interest,thr=1,level=2")]
    assert rule_contract_sources(specs) == {"funding", "oi"}


# ── metrics / acceptance ─────────────────────────────────────────────────────

def test_enrich_annualises_and_calmar() -> None:
    m = enrich({"total_return": 0.21, "max_drawdown": -0.10, "sharpe": 1.0,
                "trades": 15, "steps": 2190 * 2, "mean_reward": 0.0}, 2190)
    assert m["years"] == pytest.approx(2.0)
    assert m["ann_return"] == pytest.approx(1.21 ** 0.5 - 1)
    assert m["calmar"] == pytest.approx(m["ann_return"] / 0.10)
    assert m["trades_per_year"] == pytest.approx(7.5)


def test_evaluate_acceptance_thresholds() -> None:
    gate = enrich({"total_return": 0.20, "max_drawdown": -0.30, "sharpe": 0.3,
                   "trades": 10, "steps": 2190, "mean_reward": 0.0}, 2190)
    good = enrich({"total_return": 0.17, "max_drawdown": -0.15, "sharpe": 0.5,
                   "trades": 20, "steps": 2190, "mean_reward": 0.0}, 2190)
    bad = enrich({"total_return": 0.10, "max_drawdown": -0.25, "sharpe": 0.5,
                  "trades": 40, "steps": 2190, "mean_reward": 0.0}, 2190)
    assert evaluate_acceptance(good, gate)["pass"]
    a = evaluate_acceptance(bad, gate)
    assert not a["pass"] and not a["mdd"] and not a["return"] and not a["turnover"]


# ── end-to-end through SignalLayeredEnv on synthetic 4h data ────────────────

def _synthetic_4h(n_days: int = 320, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_days * 6
    ts = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    # uptrend with noise so the SMA200 gate turns on after warmup
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, n)))
    df = pd.DataFrame({"timestamp": ts, "open": close, "high": close * 1.01,
                       "low": close * 0.99, "close": close, "volume": 1.0})
    df["atr"] = close * 0.02
    df["sig_a"] = np.sin(np.arange(n) / 10.0)          # any bounded signal
    df["sig_funding_current"] = rng.uniform(-1, 1, n)
    return df


def test_run_rules_on_slice_gate_only_matches_const_when_gate_always_on() -> None:
    df = _synthetic_4h()
    env_cfg = {"window_size": 24, "commission": 0.0005, "min_hold_steps": 0,
               "stop_atr_mult": 0.0, "random_start": False, "max_episode_steps": None}
    # slice: last 60 days (gate warmed: >200 daily bars before it)
    bt = df.iloc[-60 * 6 - 24:].reset_index(drop=True)
    specs = [parse_rule("gate_only"), parse_rule("const:level=4"),
             parse_rule("gate_reduce:signal=sig_funding_current,thr=0.9,level=2")]
    res = run_rules_on_slice(specs, df, bt, ["sig_a"], env_cfg, 2190)
    assert {"buy_and_hold", "gate_only", "const:level=4"} <= res.keys()
    g, c = res["gate_only"], res["const:level=4"]
    for k in ("total_return", "max_drawdown", "sharpe", "trades"):
        assert np.isfinite(g[k])
    # gate is on for the whole slice in a strong uptrend → identical to const 100 %
    assert g["total_return"] == pytest.approx(c["total_return"])
    reduce_label = [k for k in res if k.startswith("gate_reduce")][0]
    assert res[reduce_label]["trades"] > g["trades"]


def test_gate_reduce_and_condition() -> None:
    df = pd.DataFrame({"sig_x": [0.9, 0.9, 0.9, 0.1], "dist": [0.02, 0.2, np.nan, 0.02]})
    gate = np.array([True, True, True, True])
    sp = parse_rule("gate_reduce:signal=sig_x,thr=0.5,level=2,signal2=dist,thr2=0.1,side2=below")
    act = make_act_fn(sp, df, gate)
    # hit only when sig high AND dist below 0.1 (NaN never hits)
    assert [act(None, _Env(i))[0] for i in range(4)] == [2, 4, 4, 4]
    with pytest.raises(ValueError, match="signal2"):
        parse_rule("gate_reduce:signal=sig_x,thr=0.5,level=2,thr2=0.1")


def test_rule_contract_sources_sees_second_signal_and_probe_features() -> None:
    specs = [parse_rule("gate_reduce:signal=funding_cum_3d_z,thr=0.5,level=2,signal2=dist_sma200,thr2=0.1,side2=below")]
    assert rule_contract_sources(specs) == {"funding"}


def test_add_rule_features_materialises_dist_and_probe_columns() -> None:
    from scripts.backtest_rules import add_rule_features
    full = _synthetic_4h()
    full["funding_rate"] = 1e-4
    bt = full.iloc[-400:].reset_index(drop=True)
    specs = [parse_rule("gate_reduce:signal=funding_cum_3d_z,thr=0.5,level=2,signal2=dist_sma200,thr2=0.1,side2=below")]
    out = add_rule_features(full, bt, specs, "4h")
    assert "dist_sma200" in out.columns and "funding_cum_3d_z" in out.columns
    assert out["dist_sma200"].notna().all()          # 320 days of history → SMA warmed


def test_daily_exit_features_semantics() -> None:
    from scripts.backtest_signal_layered import DAILY_FEATURES, _daily_ohlc
    full = _synthetic_4h(n_days=120, seed=2)
    bt = full.iloc[-30 * 6:].reset_index(drop=True)
    dd = DAILY_FEATURES["dd20_atr"](full, bt)
    lo = DAILY_FEATURES["dist_low20"](full, bt)
    s50 = DAILY_FEATURES["dist_sma50"](full, bt)
    assert np.isfinite(dd).all() and np.isfinite(lo).all() and np.isfinite(s50).all()
    assert (dd <= 1e-12).all()                                  # drawdown from a high is never positive
    # independent rebuild of dist_sma50 for the last bar (last closed UTC day)
    d = _daily_ohlc(full)["close"]
    last_day = (bt["timestamp"].iloc[-1] + pd.Timedelta("4h")).floor("D") - pd.Timedelta(days=1)
    expect = d.loc[last_day] / d.rolling(50).mean().loc[last_day] - 1
    assert s50[-1] == pytest.approx(expect)
    # a planted crash: 15 % drop on the last full day → new 20-day low and deep ATR drawdown
    crash = full.copy()
    last_day_mask = (pd.DatetimeIndex(crash["timestamp"]).tz_convert("UTC").floor("D") == last_day)
    crash.loc[last_day_mask, ["open", "high", "low", "close"]] *= 0.85
    bt_c = crash.iloc[-30 * 6:].reset_index(drop=True)
    assert DAILY_FEATURES["dist_low20"](crash, bt_c)[-1] <= 0.0
    assert DAILY_FEATURES["dd20_atr"](crash, bt_c)[-1] < -2.0
    assert DAILY_FEATURES["dist_sma200"](full, bt)[-1] == pytest.approx(
        d.loc[last_day] / d.rolling(200).mean().loc[last_day] - 1, nan_ok=True)


def test_evaluate_acceptance_noharm_mode_and_return_ratio() -> None:
    gate = enrich({"total_return": 0.17, "max_drawdown": -0.045, "sharpe": 2.9,
                   "trades": 0, "steps": 864, "mean_reward": 0.0}, 2190)
    same = dict(gate)
    # identical to base: fails "improve" (no DD/Calmar improvement) but passes "noharm"
    assert not evaluate_acceptance(same, gate, mode="improve")["pass"]
    assert evaluate_acceptance(same, gate, mode="noharm")["pass"]
    deeper = enrich({"total_return": 0.17, "max_drawdown": -0.05, "sharpe": 2.0,
                     "trades": 0, "steps": 864, "mean_reward": 0.0}, 2190)
    assert not evaluate_acceptance(deeper, gate, mode="noharm")["mdd"]
    low_ret = enrich({"total_return": 0.11, "max_drawdown": -0.02, "sharpe": 2.0,
                      "trades": 4, "steps": 864, "mean_reward": 0.0}, 2190)
    assert not evaluate_acceptance(low_ret, gate, mode="improve")["return"]          # 0.11 < 0.8×0.17
    assert evaluate_acceptance(low_ret, gate, mode="improve", return_ratio=0.6)["pass"]
    with pytest.raises(ValueError):
        evaluate_acceptance(same, gate, mode="whatever")


def test_daily_features_include_sma20_and_sma100() -> None:
    from scripts.backtest_signal_layered import DAILY_FEATURES
    full = _synthetic_4h(n_days=150)
    bt = full.iloc[-60:].reset_index(drop=True)
    assert np.isfinite(DAILY_FEATURES["dist_sma20"](full, bt)).all()
    assert np.isfinite(DAILY_FEATURES["dist_sma100"](full, bt)).all()
