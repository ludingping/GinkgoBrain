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


# =============================================================================
# 合约数据读取（funding rate / open interest / liquidation）
#
# 对应 Spider 侧 schema：
#   crypto_funding_binance       (funding_time, symbol, source, origin, ...)
#   crypto_open_interest_binance (bucket_time, symbol, sum_open_interest, ...)
#   crypto_liquidation_binance   (liquidation_time, symbol, side, price, amount, ...)
#
# 设计要点（见 docs/GinkgoSpider/合约数据-采集与信号-设计.md §7.2）：
#   - funding 按 origin 优先级取值（默认 ws_live > premium_kline）
#   - OI 精确 join 到 5min
#   - liquidation 由逐笔 raw 数据在 pandas 层按 15min 桶聚合成 long/short USD
# =============================================================================

def read_funding(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    source: str = "predicted",
    prefer_origin: tuple[str, ...] = ("ws_live", "premium_kline"),
    table: str = "public.crypto_funding_binance",
) -> pd.DataFrame:
    """
    Load funding rate from ``crypto_funding_binance``.

    Args:
        symbol:        e.g. "BTC/USDT" (ccxt 现货格式，与 Spider 写入一致)
        start:         inclusive lower bound (ISO)
        end:           exclusive upper bound (ISO)
        source:        ``"predicted"`` (训练/推理默认) 或 ``"settled"``
        prefer_origin: 同 (funding_time, symbol, source) 下多 origin 共存时
                       的优先级；默认 live > premium_kline，取第一个非空值
        table:         fully-qualified table name

    Returns:
        DataFrame with columns: ``timestamp, funding_rate, mark_price,
        index_price, origin``，按 timestamp 升序。
    """
    where = ["symbol = :symbol", "source = :source"]
    params: dict = {"symbol": symbol, "source": source}
    if start:
        where.append("funding_time >= :start")
        params["start"] = start
    if end:
        where.append("funding_time < :end")
        params["end"] = end

    query = (
        f"SELECT funding_time AS timestamp, funding_rate, "
        f"       mark_price, index_price, origin "
        f"FROM {table} "
        f"WHERE {' AND '.join(where)} "
        f"ORDER BY funding_time ASC"
    )

    with get_engine().connect() as conn:
        df = pd.read_sql(text(query), conn, params=params, parse_dates=["timestamp"])

    if df.empty:
        return df.reset_index(drop=True)

    # 按 timestamp 去重：每个 timestamp 只保留 prefer_origin 中最靠前的那条
    if len(prefer_origin) > 1 and not df.empty:
        origin_rank = {o: i for i, o in enumerate(prefer_origin)}
        df["_rank"] = df["origin"].map(origin_rank).fillna(len(prefer_origin))
        df = (df.sort_values(["timestamp", "_rank"])
                .drop_duplicates(subset="timestamp", keep="first")
                .drop(columns="_rank"))

    return df.sort_values("timestamp").reset_index(drop=True)


def read_open_interest(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    table: str = "public.crypto_open_interest_binance",
) -> pd.DataFrame:
    """
    Load 5min Open Interest from ``crypto_open_interest_binance``.

    Returns:
        DataFrame columns: ``timestamp, sum_open_interest,
        sum_open_interest_value``，按 timestamp 升序。
    """
    where = ["symbol = :symbol"]
    params: dict = {"symbol": symbol}
    if start:
        where.append("bucket_time >= :start")
        params["start"] = start
    if end:
        where.append("bucket_time < :end")
        params["end"] = end

    query = (
        f"SELECT bucket_time AS timestamp, "
        f"       sum_open_interest, sum_open_interest_value "
        f"FROM {table} "
        f"WHERE {' AND '.join(where)} "
        f"ORDER BY bucket_time ASC"
    )

    with get_engine().connect() as conn:
        df = pd.read_sql(text(query), conn, params=params, parse_dates=["timestamp"])
    return df.reset_index(drop=True)


def read_liquidation_agg(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    bucket: str = "15min",
    table: str = "public.crypto_liquidation_binance",
) -> pd.DataFrame:
    """
    Load liquidation rows from ``crypto_liquidation_binance`` and aggregate
    to fixed-size buckets in pandas (robust to SQLite in tests).

    Side convention (Binance 强平委托方向):
      - ``side='sell'`` 表示强平卖单 → **多头**被爆仓
      - ``side='buy'``  表示强平买单 → **空头**被爆仓

    USD notional 优先用 ``cost``（avg_price × filled），缺失时退回
    ``price × amount``。

    Args:
        symbol: e.g. "BTC/USDT"（Spider 侧 liquidation 表也存现货符号）
        bucket: pandas 时间频率，例如 ``"15min"`` / ``"5min"`` / ``"1h"``

    Returns:
        DataFrame columns:
          ``timestamp`` (bucket 起始), ``liq_long_usd``, ``liq_short_usd``,
          ``liq_total_usd``，按 timestamp 升序。
        空区间时返回空 DataFrame（列齐全）。
    """
    where = ["symbol = :symbol"]
    params: dict = {"symbol": symbol}
    if start:
        where.append("liquidation_time >= :start")
        params["start"] = start
    if end:
        where.append("liquidation_time < :end")
        params["end"] = end

    query = (
        f"SELECT liquidation_time, side, price, amount, avg_price, filled, cost "
        f"FROM {table} "
        f"WHERE {' AND '.join(where)} "
        f"ORDER BY liquidation_time ASC"
    )

    with get_engine().connect() as conn:
        raw = pd.read_sql(
            text(query), conn, params=params,
            parse_dates=["liquidation_time"],
        )

    cols = ["timestamp", "liq_long_usd", "liq_short_usd", "liq_total_usd"]
    if raw.empty:
        return pd.DataFrame(columns=cols)

    # usd = cost 优先；否则 price × amount（Decimal → float）
    def _as_float(s: pd.Series) -> pd.Series:
        return pd.to_numeric(s, errors="coerce")

    price = _as_float(raw["price"])
    amount = _as_float(raw["amount"])
    cost = _as_float(raw["cost"])
    usd = cost.where(cost.notna(), price * amount).fillna(0.0)

    # side='sell' → 多头被爆（long_usd）；side='buy' → 空头被爆（short_usd）
    raw["_usd"] = usd
    raw["_long"] = raw["_usd"].where(raw["side"] == "sell", 0.0)
    raw["_short"] = raw["_usd"].where(raw["side"] == "buy", 0.0)

    raw = raw.set_index("liquidation_time")
    agg = raw.resample(bucket, closed="left", label="left").agg(
        liq_long_usd=("_long", "sum"),
        liq_short_usd=("_short", "sum"),
    )
    agg["liq_total_usd"] = agg["liq_long_usd"] + agg["liq_short_usd"]
    agg = agg.reset_index().rename(columns={"liquidation_time": "timestamp"})
    return agg[cols]
