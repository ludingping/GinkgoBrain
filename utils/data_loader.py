"""Data loading utilities for stocks (yfinance), crypto (ccxt), and market sentiment."""
from __future__ import annotations

import pandas as pd
import yfinance as yf
from sqlalchemy import bindparam, text

from utils import db as _db


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
# 合约多空比 / taker 比读取（Spider 2026-09-08 新表 crypto_position_ratio_binance）
#
# 长格式 (bucket_time, symbol, ratio_type, ratio) → 宽表，列名 = ratio_type。
# 主键 (bucket_time, symbol, ratio_type)，写入 ON CONFLICT DO NOTHING：每桶每类型只有一行，
# origin（vision | rest）只是来源标记，读取时不需要去重。
# 5min → 4h / 1d 的重采样由调用方决定（建议桶末值，与 OI 一致）。
# 缺口保持 NaN：比率没有"中性零"，覆盖检查沿用 check_contract_coverage，不填零。
# 已知洞（Vision 归档）：top_account / top_position 2022-02→04，top_position 2022-07→11，
# taker_vol 2022-02→04；四类都完整的窗口从 2023-01 起。
# =============================================================================

RATIO_TYPES: tuple[str, ...] = ("top_account", "top_position", "global_account", "taker_vol")


def read_position_ratio(
    symbol: str,
    since: str | None = None,
    until: str | None = None,
    ratio_types: tuple[str, ...] = RATIO_TYPES,
    *,
    table: str = "public.crypto_position_ratio_binance",
) -> pd.DataFrame:
    """
    Load Binance long/short & taker ratios from ``crypto_position_ratio_binance``
    and pivot to one column per ``ratio_type``.

    Args:
        symbol:       e.g. "BTC/USDT" (ccxt 现货格式，与 Spider 写入一致)
        since:        inclusive lower bound — pass a tz-aware ISO string
                      (``pd.Timestamp(..., tz=...).tz_convert("UTC").isoformat()``,
                      as ``load_df_from_config`` does); a bare date is interpreted
                      in the DB session timezone, not UTC
        until:        exclusive upper bound, same convention
        ratio_types:  subset of :data:`RATIO_TYPES`; order defines column order
        table:        fully-qualified table name

    Returns:
        DataFrame ``timestamp`` + one float column per requested ratio type,
        ascending by timestamp. Buckets missing a type keep NaN. Empty range →
        empty DataFrame with the same columns.
    """
    ratio_types = tuple(ratio_types)
    unknown = [r for r in ratio_types if r not in RATIO_TYPES]
    if unknown or not ratio_types:
        raise ValueError(f"ratio_types must be a non-empty subset of {RATIO_TYPES}, got {ratio_types}")

    where = ["symbol = :symbol", "ratio_type IN :types"]
    params: dict = {"symbol": symbol, "types": ratio_types}
    if since:
        where.append("bucket_time >= :since")
        params["since"] = since
    if until:
        where.append("bucket_time < :until")
        params["until"] = until

    query = text(
        f"SELECT bucket_time AS timestamp, ratio_type, ratio "
        f"FROM {table} WHERE {' AND '.join(where)} "
        f"ORDER BY bucket_time ASC"
    ).bindparams(bindparam("types", expanding=True))

    with _db.get_contract_engine().connect() as conn:   # 运行时取，便于测试 monkeypatch
        long = pd.read_sql(query, conn, params=params, parse_dates=["timestamp"])

    cols = ["timestamp", *ratio_types]
    if long.empty:
        return pd.DataFrame({c: pd.Series(dtype="datetime64[ns, UTC]" if c == "timestamp" else "float64")
                             for c in cols})

    long["ratio"] = pd.to_numeric(long["ratio"], errors="coerce")
    wide = long.pivot(index="timestamp", columns="ratio_type", values="ratio")
    wide = wide.reindex(columns=list(ratio_types)).astype("float64")
    wide.columns.name = None
    return wide.sort_index().reset_index()[cols]


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
    df_ratio: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Broadcast funding / OI / liquidation / position-ratio data onto the OHLCV base table.

    Uses ``merge_asof(..., direction='backward')`` so each OHLCV bar picks up
    the most recent contract observation at-or-before its timestamp. Contract
    columns are NaN until the first observation arrives — callers are expected
    to let signal-level fillna(0) handle that (see ``add_contract_signals``).

    Args:
        df_ohlcv: DataFrame with a ``timestamp`` column (ascending).
        df_funding: from :func:`utils.db.read_funding` or None
        df_oi: from :func:`utils.db.read_open_interest` or None
        df_liq: from :func:`utils.db.read_liquidation_agg` or None
        df_ratio: from :func:`read_position_ratio` or None

    Returns:
        A new DataFrame with extra columns (all optional; missing inputs skipped):
          - ``funding_rate`` (+ ``funding_origin``)
          - ``sum_open_interest``, ``sum_open_interest_value``
          - ``liq_long_usd``, ``liq_short_usd``, ``liq_total_usd``
          - ``top_account``, ``top_position``, ``global_account``, ``taker_vol``
            (whichever ratio columns ``df_ratio`` carries)
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

    if df_ratio is not None and not df_ratio.empty:
        ratio_cols = [c for c in RATIO_TYPES if c in df_ratio.columns]
        ratio = _asof_ordered(df_ratio)[["timestamp", *ratio_cols]].copy()
        for col in ratio_cols:
            ratio[col] = pd.to_numeric(ratio[col], errors="coerce")
        base = pd.merge_asof(base, ratio, on="timestamp", direction="backward")

    return base


# =============================================================================
# 合约数据覆盖检查（2026-09-06）
#
# merge_contract_data 在合约数据起点之前留 NaN；此前四个脚本各自静默 fillna(0)，
# 导致 4h v2 筛选时 6 个合约信号"训练期全零"被剔除而无人察觉。规则：
#   1. 先 check_contract_coverage —— 打印每个数据源的覆盖区间/比例；
#      下游信号需要的数据源（required）覆盖 < min_coverage 直接报错。
#   2. 通过后再 neutral_fill_contract_columns —— 只为防 add_indicators 的全局
#      dropna 误删起点前的 OHLCV 行；残余 ≤5% 的填零落在 warmup 段。
# =============================================================================

CONTRACT_SOURCE_COLS: dict[str, tuple[str, ...]] = {
    "funding": ("funding_rate",),
    "oi": ("sum_open_interest", "sum_open_interest_value"),
    "liq": ("liq_long_usd", "liq_short_usd", "liq_total_usd"),
    "ratio": RATIO_TYPES,
}
_CONTRACT_SIGNAL_PREFIX = {"sig_funding_": "funding", "sig_oi_": "oi", "sig_liq_": "liq",
                           "sig_ratio_": "ratio"}
# 比率没有中性零（1 = 多空平衡），neutral_fill 不碰；缺口由 check_contract_coverage 暴露
_NO_NEUTRAL_FILL_SOURCES = frozenset({"ratio"})
DEFAULT_MIN_CONTRACT_COVERAGE = 0.95


class ContractCoverageError(ValueError):
    """A contract data source required by the signal pool is missing or too sparse."""


def required_contract_sources(signal_cols) -> set[str]:
    """Map a signal list to the contract sources it needs (``funding`` / ``oi`` / ``liq``)."""
    out: set[str] = set()
    for s in signal_cols:
        for prefix, src in _CONTRACT_SIGNAL_PREFIX.items():
            if str(s).startswith(prefix):
                out.add(src)
    return out


def contract_coverage(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    """Per-source coverage of merged contract columns.

    Returns a DataFrame indexed by source with columns ``present`` (bool),
    ``first_ts`` / ``last_ts`` (first/last row where *every* column of the
    source is non-NaN, NaT if absent) and ``coverage`` (fraction of such rows,
    0.0 if absent).

    A row counts only if all of the source's merged columns are non-NaN: for
    funding / oi / liq the columns come from one row so this equals the old
    primary-column check, but position ratios are pivoted from independent
    long rows and each ratio type has its own holes (top_position 2022-07→11).
    """
    n = len(df)
    rows = []
    for src, cols in CONTRACT_SOURCE_COLS.items():
        present_cols = [c for c in cols if c in df.columns]
        if not present_cols or n == 0:
            rows.append((src, False, pd.NaT, pd.NaT, 0.0, 0))
            continue
        ok = df[present_cols].notna().all(axis=1)
        cnt = int(ok.sum())
        if cnt == 0:
            rows.append((src, False, pd.NaT, pd.NaT, 0.0, 0))
            continue
        ts = df.loc[ok, ts_col] if ts_col in df.columns else pd.Series(df.index[ok])
        # NaN rows *after* the first observation: merge_asof(backward) carries the
        # last value forward, so these are real holes, not the leading warmup.
        interior = int((~ok.to_numpy())[ok.to_numpy().argmax():].sum())
        rows.append((src, True, ts.iloc[0], ts.iloc[-1], cnt / n, interior))
    return pd.DataFrame(
        rows, columns=["source", "present", "first_ts", "last_ts", "coverage", "interior_gaps"],
    ).set_index("source")


def check_contract_coverage(
    df: pd.DataFrame,
    required,
    *,
    min_coverage: float = DEFAULT_MIN_CONTRACT_COVERAGE,
    ts_col: str = "timestamp",
    log=print,
) -> pd.DataFrame:
    """Log coverage of every contract source; raise if a *required* one is below threshold.

    ``required`` is a collection of source names (see :func:`required_contract_sources`).
    Sources not in ``required`` are reported only — e.g. liquidation has no history
    before 2026-04 and must not block a funding-only run.

    A required source fails if its coverage is below ``min_coverage`` **or** it has
    any NaN after its first observation: the subsequent neutral fill is only
    harmless for the leading (warmup) prefix, never for holes inside the window.
    """
    cov = contract_coverage(df, ts_col=ts_col)
    required = set(required)
    for src, r in cov.iterrows():
        tag = "required" if src in required else "optional"
        if r["present"]:
            gaps = f"  interior_gaps={int(r['interior_gaps'])}" if r["interior_gaps"] else ""
            log(f"[contracts] {src:8s} {tag:8s} coverage={r['coverage']:6.1%}  "
                f"{r['first_ts']} → {r['last_ts']}{gaps}")
        else:
            log(f"[contracts] {src:8s} {tag:8s} ABSENT")
    bad = []
    for src in sorted(required):
        r = cov.loc[src]
        if not r["present"]:
            bad.append(f"{src}: absent")
        elif r["coverage"] < min_coverage:
            bad.append(f"{src}: coverage {r['coverage']:.1%} < {min_coverage:.0%}")
        elif r["interior_gaps"]:
            bad.append(f"{src}: {int(r['interior_gaps'])} NaN rows after first observation")
    if bad:
        raise ContractCoverageError(
            "contract data too sparse for the signal pool — " + "; ".join(bad)
            + ". Backfill the source (Spider backfill_* scripts) or drop the signals."
        )
    return cov


def neutral_fill_contract_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with NaN contract columns set to neutral (0.0 / '').

    Position-ratio columns are left untouched: a ratio has no neutral zero.
    """
    out = df.copy()
    for src, cols in CONTRACT_SOURCE_COLS.items():
        if src in _NO_NEUTRAL_FILL_SOURCES:
            continue
        for col in cols:
            if col in out.columns:
                out[col] = out[col].fillna(0.0)
    if "funding_origin" in out.columns:
        out["funding_origin"] = out["funding_origin"].fillna("")
    return out
