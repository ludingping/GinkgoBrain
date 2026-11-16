"""Volatility-regime states and causal start/end events for chart annotation."""
import pandas as pd


def add_volatility_regimes(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    required = ["bb_pct_b", "bb_width", "bb_width_rank126"]
    valid = out[required].notna().all(axis=1)
    states = pd.Series(pd.NA, index=out.index, dtype="string")
    states.loc[valid] = "neutral"

    expanding = out["bb_width"] > out["bb_width"].shift()
    trending = ((out["bb_pct_b"] > 1) | (out["bb_pct_b"] < 0)) & expanding
    ranging = out["bb_pct_b"].between(0, 1, inclusive="both") & (out["bb_width_rank126"] <= 0.10)
    states.loc[ranging] = "range"
    states.loc[trending] = "trend"
    out["volatility_regime"] = states

    previous = states.shift().fillna("warmup")
    for regime in ("trend", "range"):
        started = states.eq(regime) & previous.ne(regime)
        ended = states.ne(regime) & previous.eq(regime)
        out[f"volatility_{regime}_start"] = started.fillna(False).to_numpy(dtype=bool)
        out[f"volatility_{regime}_end"] = ended.fillna(False).to_numpy(dtype=bool)
    return out
