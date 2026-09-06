"""Contract-data coverage gate (utils.data_loader) — 2026-09-06.

背景：merge_asof 在合约数据起点之前留 NaN，四个脚本各自静默 fillna(0)，导致 4h v2
筛选时 6 个合约信号"训练期全零"被剔除而无人察觉。这里把"检查覆盖 → 报错或放行 →
中性填充"收敛为共享函数，并对下游信号需要的数据源强制 ≥95% 覆盖。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from utils.data_loader import (
    ContractCoverageError,
    check_contract_coverage,
    contract_coverage,
    neutral_fill_contract_columns,
    required_contract_sources,
)


def _frame(n: int = 100, funding_from: int = 40, oi: bool = False, liq_from: int = 0) -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    df = pd.DataFrame({"timestamp": ts, "close": np.linspace(1, 2, n)})
    df["funding_rate"] = np.nan
    df.loc[funding_from:, "funding_rate"] = 1e-4
    df["funding_origin"] = None
    df.loc[funding_from:, "funding_origin"] = "premium_kline"
    if oi:
        df["sum_open_interest"] = 1.0
        df["sum_open_interest_value"] = 2.0
    df["liq_long_usd"] = np.nan
    df["liq_short_usd"] = np.nan
    df["liq_total_usd"] = np.nan
    df.loc[liq_from:, ["liq_long_usd", "liq_short_usd", "liq_total_usd"]] = 1.0
    return df


def test_required_contract_sources_maps_signal_prefixes() -> None:
    sigs = ["sig_rsi", "sig_funding_current", "sig_liq_imbalance", "sig_funding_trend"]
    assert required_contract_sources(sigs) == {"funding", "liq"}
    assert required_contract_sources(["sig_rsi"]) == set()
    assert required_contract_sources(["sig_oi_change_zscore"]) == {"oi"}


def test_contract_coverage_reports_fraction_and_range() -> None:
    cov = contract_coverage(_frame(n=100, funding_from=40))
    assert set(cov.index) == {"funding", "oi", "liq"}
    f = cov.loc["funding"]
    assert f["present"] and f["coverage"] == pytest.approx(0.60)
    assert f["first_ts"] == pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(hours=4 * 40)
    assert not cov.loc["oi"]["present"] and cov.loc["oi"]["coverage"] == 0.0
    assert cov.loc["liq"]["coverage"] == pytest.approx(1.0)


def test_check_raises_when_required_source_below_threshold() -> None:
    df = _frame(n=100, funding_from=40)
    with pytest.raises(ContractCoverageError, match="funding.*60"):
        check_contract_coverage(df, required={"funding"}, min_coverage=0.95)


def test_check_raises_when_required_source_absent() -> None:
    with pytest.raises(ContractCoverageError, match="oi"):
        check_contract_coverage(_frame(), required={"oi"})


def test_check_passes_for_covered_required_and_ignores_others() -> None:
    df = _frame(n=100, funding_from=40)          # funding 60%, liq 100%, oi absent
    cov = check_contract_coverage(df, required={"liq"}, min_coverage=0.95)
    assert cov.loc["liq"]["coverage"] == pytest.approx(1.0)


def test_check_with_no_required_only_reports() -> None:
    logged: list[str] = []
    check_contract_coverage(_frame(), required=set(), log=logged.append)
    assert any("funding" in line for line in logged)


def test_neutral_fill_returns_copy_and_fills_all_contract_columns() -> None:
    df = _frame(n=10, funding_from=5, oi=True, liq_from=3)
    out = neutral_fill_contract_columns(df)
    assert df["funding_rate"].isna().any()                    # original untouched
    for col in ("funding_rate", "sum_open_interest", "sum_open_interest_value",
                "liq_long_usd", "liq_short_usd", "liq_total_usd"):
        assert not out[col].isna().any(), col
    assert (out["funding_origin"].iloc[:5] == "").all()
    assert out["close"].equals(df["close"])


def test_check_raises_on_interior_gap_even_when_coverage_is_high() -> None:
    df = _frame(n=100, funding_from=0)
    df.loc[50, "funding_rate"] = np.nan                   # one hole inside the window
    cov = contract_coverage(df)
    assert cov.loc["funding"]["coverage"] == pytest.approx(0.99)
    assert cov.loc["funding"]["interior_gaps"] == 1
    with pytest.raises(ContractCoverageError, match="after first observation"):
        check_contract_coverage(df, required={"funding"}, min_coverage=0.95)
    check_contract_coverage(df, required=set())         # optional source: report only


def test_leading_prefix_is_not_an_interior_gap() -> None:
    cov = contract_coverage(_frame(n=100, funding_from=3))
    assert cov.loc["funding"]["interior_gaps"] == 0
