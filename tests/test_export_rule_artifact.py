"""scripts/export_rule_artifact.py — rule contract + golden fingerprint (2026-09-07)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.export_rule_artifact import (
    CHECK_ROWS, build_payload, export, expected_rows, fingerprint_sha256,
    make_golden_daily_closes, rule_decisions, verify, verify_payload,
)


def test_rule_decisions_conventions() -> None:
    idx = pd.date_range("2024-01-01", periods=6, freq="1D", tz="UTC")
    c = pd.Series([10, 10, 10, 10, 12, 8.0], index=idx)
    d = rule_decisions(c, gate_days=2, exit_days=2)
    # SMA2 at day 5 (12): sma=11 → gate on, exit_ok (12 >= 11) → target 1
    assert d.loc[idx[4], "target"] == 1.0
    # day 6 (8): sma=10 → 8 < 10 → gate off and exit → 0
    assert d.loc[idx[5], "target"] == 0.0 and not d.loc[idx[5], "regime_ok"]
    # equality: close == SMA → gate strict False, exit_ok True
    flat = rule_decisions(pd.Series([10.0] * 5, index=idx[:5]), 2, 2)
    assert not flat["regime_ok"].iloc[-1] and flat["exit_ok"].iloc[-1] and flat["target"].iloc[-1] == 0.0
    # warmup: NaN SMA → False / target 0
    assert d["target"].iloc[0] == 0.0
    # no exit → exit_ok all True
    assert rule_decisions(c, 2, None)["exit_ok"].all()


def test_golden_is_deterministic_and_switches_inside_check_window() -> None:
    a, b = make_golden_daily_closes(), make_golden_daily_closes()
    pd.testing.assert_series_equal(a, b)
    d = rule_decisions(a, 200, 50).iloc[-CHECK_ROWS:]
    long_ = (d["regime_ok"] & d["exit_ok"]).sum()
    exited = (d["regime_ok"] & ~d["exit_ok"]).sum()
    off = (~d["regime_ok"]).sum()
    assert long_ > 0 and exited > 0 and off > 0          # all three states inside the check window


def test_fingerprint_stable_and_detects_convention_drift() -> None:
    g = make_golden_daily_closes()
    rows = expected_rows(rule_decisions(g, 200, 50))
    assert fingerprint_sha256(rows) == fingerprint_sha256(expected_rows(rule_decisions(g, 200, 50)))
    # a flipped exit comparison (> instead of >=) or a different SMA length must change the fingerprint
    assert fingerprint_sha256(rows) != fingerprint_sha256(expected_rows(rule_decisions(g, 200, 20)))
    payload = build_payload("t", 200, 50)
    assert verify_payload(payload) == []
    tampered = dict(payload, rule={**payload["rule"], "exit": {"ma_days": 20, "condition": "x"}})
    assert verify_payload(tampered)


def test_export_and_verify_roundtrip(tmp_path) -> None:
    p = export("btc_4h_gate_sma50_test", 200, 50, tmp_path)
    assert verify(p) == 0
    y = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert y["rule"]["kind"] == "gate_exit_sma" and y["rule"]["no_model"] is True
    assert y["rule"]["exit"]["ma_days"] == 50 and y["rule"]["gate"]["ma_days"] == 200
    assert len(y["fingerprint"]["golden_closes"]) == 320 and len(y["fingerprint"]["expected"]) == CHECK_ROWS
    # gate-only contract
    p2 = export("btc_4h_gate_only_test", 200, None, tmp_path)
    y2 = yaml.safe_load(p2.read_text(encoding="utf-8"))
    assert y2["rule"]["exit"]["ma_days"] is None and verify(p2) == 0
    # corrupt the expected rows → verify fails
    y["fingerprint"]["expected"][-1]["target"] = 1.0 - y["fingerprint"]["expected"][-1]["target"]
    p.write_text(yaml.safe_dump(y, sort_keys=False), encoding="utf-8")
    assert verify(p) == 1


# ─── gate_vol_target (H6 A/B arm, 2026-09-14) ────────────────────────────────

def test_vol_rule_decisions_conventions() -> None:
    from scripts.export_rule_artifact import vol_rule_decisions
    idx = pd.date_range("2024-01-01", periods=40, freq="1D", tz="UTC")
    rng = np.random.default_rng(1)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.01, 0.02, 40))), index=idx)
    dec = vol_rule_decisions(close, gate_days=5, vol_window=10, vol_target=0.3, floor_level=1)
    assert set(dec.columns) == {"regime_ok", "exit_ok", "target"}
    assert dec["exit_ok"].all()
    assert set(dec["target"].unique()) <= {0.0, 0.25, 0.5, 0.75, 1.0}
    # warmup (rvol NaN) with gate on → full; gate off → 0 regardless of vol
    on_warm = dec.index[(dec.regime_ok) & (dec.index < idx[10])]
    assert (dec.loc[on_warm, "target"] == 1.0).all()
    assert (dec.loc[~dec.regime_ok, "target"] == 0.0).all()
    # floor: gate on never below 0.25 with floor_level=1
    assert (dec.loc[dec.regime_ok, "target"] >= 0.25).all()


def test_vol_rule_matches_backtester_quantisation() -> None:
    """Contract decisions == backtest_rules.vol_target_level on daily_realized_vol (no hysteresis)."""
    from scripts.backtest_rules import vol_target_level
    from scripts.backtest_signal_layered import daily_realized_vol
    from scripts.export_rule_artifact import make_golden_daily_closes_vol, vol_rule_decisions
    golden = make_golden_daily_closes_vol()
    frame = pd.DataFrame({"timestamp": golden.index, "open": golden.values, "high": golden.values,
                          "low": golden.values, "close": golden.values, "volume": 1.0})
    rv = daily_realized_vol(frame, frame, 30)            # 1-day bars → value of the same closed day
    dec = vol_rule_decisions(golden, 200, 30, 0.30, 1)
    for i in range(200, len(golden)):
        raw = 4 * min(1.0, 0.30 / rv[i]) if np.isfinite(rv[i]) and rv[i] > 0 else np.nan
        lvl = vol_target_level(raw, 4, hyst=0.0, floor=1)
        expect = lvl / 4 if dec["regime_ok"].iloc[i] else 0.0
        assert dec["target"].iloc[i] == pytest.approx(expect), i


def test_vol_golden_exercises_intermediate_levels_and_roundtrip(tmp_path) -> None:
    from scripts.export_rule_artifact import export, make_golden_daily_closes_vol, vol_rule_decisions
    golden = make_golden_daily_closes_vol()
    assert make_golden_daily_closes_vol().equals(golden)
    tail = vol_rule_decisions(golden, 200, 30, 0.30, 1).iloc[-CHECK_ROWS:]
    targets = set(tail["target"].round(2))
    assert 0.0 in targets and len(targets - {0.0, 1.0}) >= 2, targets
    dst = export("t_vol", 200, None, tmp_path, vol_target=0.30, vol_window=30, floor_level=1)
    payload = yaml.safe_load(dst.read_text(encoding="utf-8"))
    assert payload["rule"]["kind"] == "gate_vol_target"
    assert payload["rule"]["vol"] == {**payload["rule"]["vol"], "window_days": 30, "target_ann": 0.3, "floor_level": 1}
    assert verify(dst) == 0
    assert verify_payload(payload) == []
    payload["rule"]["vol"]["target_ann"] = 0.4                    # drift → fingerprint must fail
    assert verify_payload(payload)
    with pytest.raises(ValueError, match="no exit rule"):
        build_payload("bad", 200, 50, vol_target=0.3, vol_window=30)
