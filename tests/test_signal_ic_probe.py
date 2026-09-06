"""scripts/signal_ic_probe.py — model-free rank-IC falsification test (2026-09-06)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.signal_ic_probe import (
    add_probe_features, auc, decile_spread_bps, evaluate_signal, format_horizon_table,
    forward_log_return, rank_ic, required_sources, verdict,
)
from utils.splits import time_folds


def _df(n: int = 3000, seed: int = 0, planted: float = 0.0) -> pd.DataFrame:
    """4h frame; `planted` = correlation strength of sig_plant with the 6-bar fwd return."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, 0.01, n)
    close = 100 * np.exp(np.cumsum(rets))
    df = pd.DataFrame({"timestamp": pd.date_range("2022-01-01", periods=n, freq="4h", tz="UTC"),
                       "close": close, "funding_rate": rng.normal(1e-4, 5e-5, n)})
    fwd = forward_log_return(df["close"], 6).fillna(0.0)
    df["sig_plant"] = planted * fwd / fwd.std() + rng.normal(0, 1, n)
    df["sig_noise"] = rng.normal(0, 1, n)
    return df


def test_rank_ic_sign_and_magnitude() -> None:
    df = _df(planted=1.0)
    fwd = forward_log_return(df["close"], 6)
    assert rank_ic(df["sig_plant"], fwd) > 0.5
    assert rank_ic(-df["sig_plant"], fwd) < -0.5
    assert abs(rank_ic(df["sig_noise"], fwd)) < 0.05


def test_decile_spread_and_auc_follow_sign() -> None:
    df = _df(planted=1.0)
    fwd = forward_log_return(df["close"], 6)
    assert decile_spread_bps(df["sig_plant"], fwd) > 0
    assert decile_spread_bps(-df["sig_plant"], fwd) < 0
    assert auc(df["sig_plant"], fwd) > 0.7
    assert auc(-df["sig_plant"], fwd) < 0.3


def test_rank_ic_nan_on_degenerate_input() -> None:
    s = pd.Series([1.0] * 50)
    assert np.isnan(rank_ic(s, pd.Series(np.arange(50.0))))


def test_verdict_rules() -> None:
    assert verdict([0.05, 0.04, 0.06]) == (True, "ok")
    assert verdict([-0.05, -0.04, -0.06])[0]
    assert verdict([0.05, -0.04, 0.06]) == (False, "sign flips")
    ok, why = verdict([0.05, 0.02, 0.06])
    assert not ok and "fold [2]" in why
    assert not verdict([0.05, float("nan"), 0.06])[0]
    ok, why = verdict([0.05, 0.04, 0.06], ns=[1000, 120, 900])
    assert not ok and "n<300 on fold [2]" in why
    assert verdict([0.05, 0.04, 0.06], ns=[1000, 800, 900])[0]


def test_probe_features_windows_on_4h() -> None:
    df = _df(n=800)
    out = add_probe_features(df, ["funding_cum_3d", "funding_cum_7d", "funding_rank_90d"], "4h")
    # 3D / 4h = 18 bars
    expect = df["funding_rate"].rolling(18, min_periods=1).sum()
    pd.testing.assert_series_equal(out["funding_cum_3d"], expect, check_names=False)
    assert out["funding_rank_90d"].dropna().between(0, 1).all()
    assert out["funding_cum_7d"].iloc[-1] == pytest.approx(df["funding_rate"].iloc[-42:].sum())


def test_probe_feature_needs_funding_rate() -> None:
    with pytest.raises(ValueError, match="funding_rate"):
        add_probe_features(pd.DataFrame({"close": [1.0]}), ["funding_cum_3d"], "4h")


def test_required_sources_includes_probe_features() -> None:
    assert required_sources(["funding_cum_3d", "sig_rsi"]) == {"funding"}
    assert required_sources(["sig_oi_change_zscore"]) == {"oi"}
    assert required_sources(["sig_rsi"]) == set()


def test_evaluate_signal_planted_passes_and_noise_fails() -> None:
    df = _df(n=4000, planted=0.3)
    folds = time_folds(df["timestamp"], 3, embargo=6 * pd.Timedelta("4h"))
    plant = evaluate_signal(df, "sig_plant", 6, folds)
    noise = evaluate_signal(df, "sig_noise", 6, folds)
    assert len(plant) == 3 and all(f.n > 100 for f in plant)
    assert verdict([f.ic for f in plant])[0]
    assert not verdict([f.ic for f in noise])[0]
    table = format_horizon_table({"sig_plant": plant, "sig_noise": noise})
    assert "**PASS**" in table[2] and "FAIL" in table[3]


def test_evaluate_signal_mask_restricts_rows() -> None:
    df = _df(n=3000, planted=0.3)
    folds = time_folds(df["timestamp"], 3, embargo=6 * pd.Timedelta("4h"))
    mask = np.zeros(len(df), dtype=bool)
    mask[::2] = True
    full = evaluate_signal(df, "sig_plant", 6, folds)
    half = evaluate_signal(df, "sig_plant", 6, folds, mask)
    for a, b in zip(full, half):
        assert abs(b.n - a.n / 2) <= 1
        assert np.sign(b.ic) == np.sign(a.ic)


def test_parse_where_and_mask() -> None:
    from scripts.signal_ic_probe import parse_where, where_mask
    assert parse_where("dist_sma200<0.10") == ("dist_sma200", "<", 0.10)
    assert parse_where("x>=-0.05") == ("x", ">=", -0.05)
    with pytest.raises(ValueError):
        parse_where("dist_sma200")
    df = pd.DataFrame({"a": [0.05, 0.2, np.nan, -0.1], "b": [1.0, 1.0, 1.0, 0.0]})
    m = where_mask(df, [("a", "<", 0.1), ("b", ">", 0.5)])
    assert m.tolist() == [True, False, False, False]
    with pytest.raises(ValueError, match="not in data"):
        where_mask(df, [("zzz", "<", 1.0)])


def test_dist_sma200_feature_and_z_features() -> None:
    df = _df(n=1500)
    out = add_probe_features(df, ["dist_sma200", "funding_cum_3d_z"], "4h")
    d = out["dist_sma200"]
    assert d.isna().sum() > 0 and d.notna().sum() > 0          # SMA warmup then values
    # rebuild independently: last closed UTC day close / SMA200 - 1
    daily = df.set_index("timestamp")["close"].resample("1D", closed="left", label="left").last()
    dist = daily / daily.rolling(200).mean() - 1
    last_day = (df["timestamp"] + pd.Timedelta("4h")).dt.floor("D") - pd.Timedelta(days=1)
    expect = dist.reindex(pd.DatetimeIndex(last_day)).to_numpy()
    np.testing.assert_allclose(d.to_numpy(), expect, equal_nan=True)
    assert out["funding_cum_3d_z"].dropna().between(-1, 1).all()


def test_oi_probe_features_semantics() -> None:
    n = 600
    rng = np.random.default_rng(5)
    df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC"),
                       "close": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))),
                       "sum_open_interest": 1e5 + rng.normal(0, 1e3, n).cumsum()})
    # plant a capitulation at bar 500: OI −10 % and price −5 % over the prior day (6 bars)
    df.loc[495:, "sum_open_interest"] *= 0.90
    df.loc[495:, "close"] *= 0.95
    out = add_probe_features(df, ["oi_level_90d_z", "oi_chg_1d", "oi_capitulation_1d", "oi_new_longs_1d"], "4h")
    assert out["oi_level_90d_z"].dropna().between(-1, 1).all()
    assert out["oi_chg_1d"].iloc[500] == pytest.approx(df["sum_open_interest"].iloc[500] / df["sum_open_interest"].iloc[494] - 1)
    assert out["oi_capitulation_1d"].iloc[497] > 0.05          # both down → positive magnitude
    assert out["oi_new_longs_1d"].iloc[497] == 0.0
    assert (out["oi_capitulation_1d"] >= 0).all() and (out["oi_new_longs_1d"] >= 0).all()
    with pytest.raises(ValueError, match="sum_open_interest"):
        add_probe_features(pd.DataFrame({"close": [1.0], "funding_rate": [0.0]}), ["oi_chg_1d"], "4h")
    assert required_sources(["oi_chg_1d", "funding_cum_3d"]) == {"oi", "funding"}


def test_metrics_ignore_infinite_values() -> None:
    df = _df(planted=1.0)
    fwd = forward_log_return(df["close"], 6)
    sig = df["sig_plant"].copy()
    sig.iloc[10] = np.inf
    sig.iloc[11] = -np.inf
    assert np.isfinite(rank_ic(sig, fwd)) and np.isfinite(auc(sig, fwd))
    folds = time_folds(df["timestamp"], 3, embargo=6 * pd.Timedelta("4h"))
    df2 = df.assign(sig_plant=sig)
    assert evaluate_signal(df2, "sig_plant", 6, folds)[0].n > 0


def test_oi_change_from_zero_is_nan_not_inf() -> None:
    df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=20, freq="4h", tz="UTC"),
                       "close": np.linspace(1, 2, 20), "sum_open_interest": [0.0] * 6 + [1.0] * 14})
    out = add_probe_features(df, ["oi_chg_1d", "oi_capitulation_1d"], "4h")
    assert not np.isinf(out["oi_chg_1d"].to_numpy()).any()
    assert not np.isinf(out["oi_capitulation_1d"].to_numpy()).any()
