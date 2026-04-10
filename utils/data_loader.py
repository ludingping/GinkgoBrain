"""Data loading utilities for stocks (yfinance) and crypto (ccxt)."""
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
    return df.reset_index(drop=True)


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
    df = df.set_index("timestamp").reset_index(drop=True)
    return df.dropna()
