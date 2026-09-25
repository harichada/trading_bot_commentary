"""Typed accessors for the ctx_* columns joined onto the indicator frame by indicator_frame(context=...)."""
from __future__ import annotations

import numpy as np
import pandas as pd

CTX_COLUMNS = (
    "rvol_open5", "rvol_open15", "gap_pct", "rs_20", "rs_60", "rs_120",
    "rs_rank_60", "rvol_rank5", "pct_from_52w_high", "adv_dollars_20",
    "adv_bucket", "atr14_pct", "earnings_today", "earnings_prev",
    "eps_surprise_pct", "news_count_24h", "news_sent_24h",
)

_ADV_BUCKET_RANK = {"mega": 3.0, "large": 2.0, "mid": 1.0, "small": 0.0}


def ctx(ind: pd.DataFrame, name: str) -> np.ndarray:
    """ctx_<name> as a float ndarray. Bools become 0/1; adv_bucket strings map to
    mega=3, large=2, mid=1, small=0 (anything else, including missing, is NaN).
    All-NaN of the right length if the column itself is missing."""
    col_name = "ctx_" + name
    if col_name not in ind.columns:
        return np.full(len(ind), np.nan)
    series = ind[col_name]
    if name == "adv_bucket":
        return series.map(_ADV_BUCKET_RANK).astype(float).to_numpy()
    return series.astype(float).to_numpy()


def in_play(ind: pd.DataFrame, top_n: int) -> np.ndarray:
    """True where ctx_rvol_rank5 is within the top_n tightest (lowest) ranks."""
    return ctx(ind, "rvol_rank5") <= top_n
