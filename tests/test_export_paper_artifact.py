"""Tests for scripts/export_paper_artifact.py — golden fingerprint (Spider issue #21)."""
from __future__ import annotations

import numpy as np

from scripts.export_paper_artifact import (
    FINGERPRINT_ATOL, UNSERVABLE_PREFIXES, fingerprint_sha256, golden_signals,
    make_golden_ohlcv,
)

SIGS = ["sig_trend_macd_hist", "sig_trend_slope_21", "sig_mom_rsi_trend",
        "sig_vol_bb_width", "sig_volume_obv_slope", "sig_regime_return_60"]


def test_golden_ohlcv_is_deterministic_and_valid():
    a, b = make_golden_ohlcv(), make_golden_ohlcv()
    assert a.equals(b)
    assert (a["high"] >= a[["open", "close"]].max(axis=1)).all()
    assert (a["low"] <= a[["open", "close"]].min(axis=1)).all()
    assert a["timestamp"].dt.tz is not None


def test_fingerprint_is_stable_across_recomputation():
    g = make_golden_ohlcv()
    e1 = golden_signals(g, SIGS)
    e2 = golden_signals(make_golden_ohlcv(), SIGS)
    assert fingerprint_sha256(e1, SIGS) == fingerprint_sha256(e2, SIGS)
    assert np.nanmax(np.abs(e1[SIGS].to_numpy() - e2[SIGS].to_numpy())) <= FINGERPRINT_ATOL


def test_fingerprint_detects_signal_definition_drift():
    """A definition change on a single column must change the hash and exceed atol."""
    e = golden_signals(make_golden_ohlcv(), SIGS)
    drifted = e.copy()
    drifted["sig_volume_obv_slope"] = (drifted["sig_volume_obv_slope"] * 0.9).round(9)
    assert fingerprint_sha256(e, SIGS) != fingerprint_sha256(drifted, SIGS)
    assert np.nanmax(np.abs(e[SIGS].to_numpy() - drifted[SIGS].to_numpy())) > FINGERPRINT_ATOL


def test_fingerprint_ignores_history_start():
    """Start invariance carried into the artifact check: a golden frame with 400
    extra leading bars must produce the same last-500-bar signals."""
    g_long = make_golden_ohlcv(n=1600)
    g_short = g_long.iloc[400:].reset_index(drop=True)
    e_long = golden_signals(g_long, SIGS)
    e_short = golden_signals(g_short, SIGS)
    assert np.nanmax(np.abs(e_long[SIGS].to_numpy() - e_short[SIGS].to_numpy())) <= FINGERPRINT_ATOL


def test_unservable_prefixes_cover_mtf_and_contracts():
    assert any("sig_mtf_x".startswith(p) for p in UNSERVABLE_PREFIXES)
    assert any("sig_funding_current".startswith(p) for p in UNSERVABLE_PREFIXES)
    assert not any("sig_volume_obv_slope".startswith(p) for p in UNSERVABLE_PREFIXES)
