"""Data loading utilities for stocks (yfinance), crypto (ccxt), and market sentiment."""
from __future__ import annotations

import pandas as pd
import yfinance as yf


def load_stock_data(
    ticker: str,
    start: str,
    end: str,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Download OHLCV data for a stock symbol via yfinance.

    Returns a DataFrame with lowercase column names: open high low close volume.
    """
    df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    df.index = pd.to_datetime(df.index)
    df.index.name = "timestamp"
    return df.reset_index()


def load_crypto_data(
    symbol: str,
    exchange_id: str = "binance",
    timeframe: str = "1d",
    since: str | None = None,
    limit: int = 1000,
) -> pd.DataFrame:
    """
    Download OHLCV data for a crypto pair via ccxt.

    Args:
        symbol:      e.g. "BTC/USDT"
        exchange_id: ccxt exchange id, default "binance"
        timeframe:   e.g. "1d", "4h", "1h"
        since:       ISO date string start, e.g. "2023-01-01"
        limit:       max candles to fetch
    """
    import ccxt

    exchange_cls = getattr(ccxt, exchange_id)
    exchange = exchange_cls({"enableRateLimit": True})

    since_ms = None
    if since:
        since_ms = int(pd.Timestamp(since).timestamp() * 1000)

    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df.dropna().reset_index(drop=True)


_FNG_ORIGIN = pd.Timestamp("2018-02-01")
_FNG_URL    = "https://api.alternative.me/fng/"


def load_fng(limit: int | str = 365) -> pd.DataFrame:
    """
    Fetch the Crypto Fear & Greed Index from alternative.me.

    Args:
        limit: Number of daily records to retrieve.
               Pass ``"ALL"`` to fetch every available day from
               2018-02-01 through today.
               Defaults to 365 (≈ 1 year).

    Returns:
        DataFrame with columns:
            date        – date (UTC, tz-naive)
            value       – index score 0–100
            label       – text classification (e.g. "Fear", "Greed")
        Sorted ascending by date.
    """
    import requests

    if isinstance(limit, str) and limit.upper() == "ALL":
        n = (pd.Timestamp.now().normalize() - _FNG_ORIGIN).days + 1
    else:
        n = int(limit)

    resp = requests.get(_FNG_URL, params={"limit": n, "format": "json"}, timeout=15)
    resp.raise_for_status()
    payload = resp.json()

    records = payload.get("data", [])
    df = pd.DataFrame(records)
    df["date"]  = pd.to_datetime(df["timestamp"].astype(int), unit="s").dt.normalize()
    df["value"] = df["value"].astype(int)
    df = (df.rename(columns={"value_classification": "label"})
            [["date", "value", "label"]]
            .sort_values("date")
            .reset_index(drop=True))
    return df


# =============================================================================
# 合约数据 merge —— 把 Spider 侧 funding / OI / liquidation 对齐到 OHLCV 主表
#
# 对齐规则（设计文档 §7.2）：
#   - K 线为主表，timestamp 为 join key
#   - funding 5min → merge_asof backward（最新已知的 funding 给本 bar）
#   - OI 5min → merge_asof backward（精确 join，5min 对齐时相当于直接匹配）
#   - liquidation 15min 桶 → merge_asof backward（广播到 3 个 5min bar）
# =============================================================================

def _asof_ordered(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.sort_values("timestamp").reset_index(drop=True)
    return out


def merge_contract_data(
    df_ohlcv: pd.DataFrame,
    *,
    df_funding: pd.DataFrame | None = None,
    df_oi: pd.DataFrame | None = None,
    df_liq: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Broadcast funding / OI / liquidation data onto the OHLCV base table.

    Uses ``merge_asof(..., direction='backward')`` so each OHLCV bar picks up
    the most recent contract observation at-or-before its timestamp. Contract
    columns are NaN until the first observation arrives — callers are expected
    to let signal-level fillna(0) handle that (see ``add_contract_signals``).

    Args:
        df_ohlcv: DataFrame with a ``timestamp`` column (ascending).
        df_funding: from :func:`utils.db.read_funding` or None
        df_oi: from :func:`utils.db.read_open_interest` or None
        df_liq: from :func:`utils.db.read_liquidation_agg` or None

    Returns:
        A new DataFrame with extra columns (all optional; missing inputs skipped):
          - ``funding_rate`` (+ ``funding_origin``)
          - ``sum_open_interest``, ``sum_open_interest_value``
          - ``liq_long_usd``, ``liq_short_usd``, ``liq_total_usd``
    """
    if "timestamp" not in df_ohlcv.columns:
        raise ValueError("df_ohlcv must have a 'timestamp' column")

    base = df_ohlcv.sort_values("timestamp").reset_index(drop=True).copy()

    if df_funding is not None and not df_funding.empty:
        f = _asof_ordered(df_funding)[["timestamp", "funding_rate", "origin"]].copy()
        f["funding_rate"] = pd.to_numeric(f["funding_rate"], errors="coerce")
        f = f.rename(columns={"origin": "funding_origin"})
        base = pd.merge_asof(base, f, on="timestamp", direction="backward")

    if df_oi is not None and not df_oi.empty:
        oi = _asof_ordered(df_oi)[
            ["timestamp", "sum_open_interest", "sum_open_interest_value"]
        ].copy()
        oi["sum_open_interest"] = pd.to_numeric(oi["sum_open_interest"], errors="coerce")
        oi["sum_open_interest_value"] = pd.to_numeric(
            oi["sum_open_interest_value"], errors="coerce",
        )
        base = pd.merge_asof(base, oi, on="timestamp", direction="backward")

    if df_liq is not None and not df_liq.empty:
        liq = _asof_ordered(df_liq)[
            ["timestamp", "liq_long_usd", "liq_short_usd", "liq_total_usd"]
        ].copy()
        for col in ("liq_long_usd", "liq_short_usd", "liq_total_usd"):
            liq[col] = pd.to_numeric(liq[col], errors="coerce")
        base = pd.merge_asof(base, liq, on="timestamp", direction="backward")

    return base
