"""Direction, oscillation, transition, and contiguous-run states for chart annotation."""
import pandas as pd


def _entry_signals(states: pd.Series) -> pd.Series:
    previous = states.shift().fillna("warmup")
    entering = states.isin(["bull", "bear"]) & states.ne(previous)
    return states.where(entering)


def add_direction_states(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()

    long_states = pd.Series(pd.NA, index=out.index, dtype="string")
    long_valid = out[["close", "sma200", "sma200_slope20"]].notna().all(axis=1)
    long_states.loc[long_valid] = "neutral"
    long_states.loc[(out["close"] >= out["sma200"]) & (out["sma200_slope20"] > 0)] = "bull"
    long_states.loc[(out["close"] < out["sma200"]) & (out["sma200_slope20"] < 0)] = "bear"

    macd_states = pd.Series(pd.NA, index=out.index, dtype="string")
    macd_valid = out[["dif", "dea"]].notna().all(axis=1)
    macd_states.loc[macd_valid] = "neutral"
    macd_states.loc[out["dif"] > out["dea"]] = "bull"
    macd_states.loc[out["dif"] < out["dea"]] = "bear"

    out["long_direction"] = long_states
    out["long_signal"] = _entry_signals(long_states)
    out["macd_direction"] = macd_states
    out["macd_signal"] = _entry_signals(macd_states)
    return out


def add_donchian_states(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    previous_upper = out["donchian_upper20"].shift()
    previous_lower = out["donchian_lower20"].shift()
    events = pd.Series(pd.NA, index=out.index, dtype="string")
    events.loc[out["close"] > previous_upper] = "bull"
    events.loc[out["close"] < previous_lower] = "bear"
    states = events.ffill()
    out["donchian_direction"] = states
    out["donchian_signal"] = _entry_signals(states)
    return out


def add_momentum_range_states(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    required = ["rsi14", "kdj_k", "kdj_d", "cci20"]
    valid = out[required].notna().all(axis=1)
    states = pd.Series(pd.NA, index=out.index, dtype="string")
    states.loc[valid] = "other"
    in_range = (
        out["rsi14"].between(30, 70, inclusive="both")
        & out["kdj_k"].between(20, 80, inclusive="both")
        & out["kdj_d"].between(20, 80, inclusive="both")
        & out["cci20"].between(-100, 100, inclusive="both")
    )
    states.loc[in_range] = "range"
    out["momentum_range"] = states
    return out


def add_volume_direction_states(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    price_change20 = out["close"].pct_change(20)
    obv_change20 = out["obv"].diff(20)
    valid = price_change20.notna() & obv_change20.notna()
    states = pd.Series(pd.NA, index=out.index, dtype="string")
    states.loc[valid] = "neutral"
    states.loc[(price_change20 > 0) & (obv_change20 > 0)] = "bull"
    states.loc[(price_change20 < 0) & (obv_change20 < 0)] = "bear"
    out["volume_direction"] = states
    out["volume_signal"] = _entry_signals(states)
    return out


def direction_runs(states: pd.Series) -> list[tuple[str, int, int]]:
    filled = states.astype("string").fillna("warmup")
    runs = []
    start = 0
    for end in range(1, len(filled) + 1):
        if end == len(filled) or filled.iloc[end] != filled.iloc[start]:
            runs.append((filled.iloc[start], start, end))
            start = end
    return runs
