"""Data loading utilities for stocks (yfinance), crypto (ccxt), and market sentiment."""
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
