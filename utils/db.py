"""PostgreSQL connection via SQLAlchemy, configured from .env"""
import os
import pandas as pd
from functools import lru_cache
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

load_dotenv()


def _build_url() -> str:
    host = os.environ["DB_HOST"]
    port = os.environ.get("DB_PORT", "5432")
    name = os.environ["DB_NAME"]
    user = os.environ["DB_USER"]
    password = os.environ["DB_PASSWORD"]
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}"


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Return a cached SQLAlchemy engine built from .env variables."""
    return create_engine(
        _build_url(),
        pool_size=int(os.environ.get("DB_POOL_SIZE", 5)),
        max_overflow=int(os.environ.get("DB_MAX_OVERFLOW", 10)),
        pool_timeout=int(os.environ.get("DB_POOL_TIMEOUT", 30)),
        pool_pre_ping=True,
    )


def read_ohlcv(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    table: str = "public.crypto_kline_binance",
    only_closed: bool = True,
) -> pd.DataFrame:
    """
    Load OHLCV data from PostgreSQL.

    Compatible with the crypto_kline_binance schema:
      open_time, symbol, open, high, low, close, volume, is_closed

    Returns a DataFrame with columns: timestamp, open, high, low, close, volume
    sorted ascending by open_time.

    Args:
        symbol:      e.g. "BTCUSDT"
        start:       inclusive lower bound, e.g. "2024-01-01"
        end:         exclusive upper bound, e.g. "2024-12-31"
        table:       fully-qualified table name
        only_closed: filter to is_closed = true (skip incomplete candles)
    """
    where = ["symbol = :symbol"]
    params: dict = {"symbol": symbol}

    if start:
        where.append("open_time >= :start")
        params["start"] = start
    if end:
        where.append("open_time < :end")
        params["end"] = end
    if only_closed:
        where.append("is_closed = true")

    query = (
        f"SELECT open_time AS timestamp, open, high, low, close, volume "
        f"FROM {table} "
        f"WHERE {' AND '.join(where)} "
        f"ORDER BY open_time ASC"
    )

    with get_engine().connect() as conn:
        df = pd.read_sql(text(query), conn, params=params, parse_dates=["timestamp"])

    return df.reset_index(drop=True)


def execute(sql: str, params: dict | None = None) -> pd.DataFrame:
    """Run an arbitrary SELECT query and return results as a DataFrame."""
    with get_engine().connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})
