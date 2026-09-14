"""H7 portfolio layer — combine() and metrics() on synthetic legs."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.portfolio_gate_only import VARIANTS, combine, metrics


def _legs():
    ts = pd.date_range("2024-01-01", periods=6, freq="4h", tz="UTC")
    return pd.DataFrame({"BTC": np.log1p([0.10, -0.10, 0.0, 0.05, 0.0, -0.05]),
                         "ETH": np.log1p([-0.10, 0.10, 0.0, 0.05, 0.0, 0.05])}, index=ts)


def test_combine_equal_weight_is_mean_of_simple_returns() -> None:
    legs = _legs()
    port = combine(legs, {"BTC": 0.5, "ETH": 0.5})
    # bar 1: 0.5·0.10 + 0.5·(−0.10) = 0 ; bar 4: 0.05
    assert port.iloc[0] == pytest.approx(0.0)
    assert port.iloc[3] == pytest.approx(np.log1p(0.05))
    assert len(port) == len(legs)


def test_combine_renormalises_over_present_columns_and_drops_gaps() -> None:
    legs = _legs()
    legs.loc[legs.index[2], "ETH"] = np.nan
    port = combine(legs, {"BTC": 0.25, "ETH": 0.25, "SOL": 0.5})   # SOL absent → BTC/ETH 50/50
    assert len(port) == len(legs) - 1
    assert port.iloc[0] == pytest.approx(0.0)
    with pytest.raises(ValueError):
        combine(legs, {"XRP": 1.0})


def test_metrics_matches_summarise_definitions() -> None:
    lr = pd.Series(np.log1p([0.10, -0.10, 0.05]))
    m = metrics(lr, periods_per_year=2190)
    assert m["total_return"] == pytest.approx(1.1 * 0.9 * 1.05 - 1)
    assert m["max_drawdown"] == pytest.approx(-0.10)
    assert m["steps"] == 3 and "calmar" in m and "ann_return" in m


def test_variants_are_the_preregistered_three() -> None:
    assert set(VARIANTS) == {"P-EW4", "P-EW2", "P-BTC50"}
    for w in VARIANTS.values():
        assert sum(w.values()) == pytest.approx(1.0)
