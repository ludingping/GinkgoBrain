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
