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
