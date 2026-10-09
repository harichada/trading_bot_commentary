"""prior_high_rsi (2026-10-08): fade a failed retest of the session's prior swing high (short) or
prior swing low (long), filtered by 5-min RSI.

Encodes the user's discretionary MU trades (research/reports/mu_style/2026-10-08-mu-style-analysis.md):
shorts went on within ~0.3-0.7% of the session's prior swing high with RSI(14, 5-min) above ~60,
after the push failed to make a new high; longs mirror at a prior low. The "prior" level is the
extreme of the previous LOOKBACK_BARS bars of the *same session*, excluding the bar being judged,
so yesterday's high and the current bar's own high never count. Stop sits just beyond the level,
target is an R-multiple of that risk, and the hold is capped at TIME_STOP_BARS.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.families.common import Family, candidate_positions, col, make_entry, one_per_session
from strategies.families.models import Entry

LOOKBACK_BARS = 24       # 2 h of 5-min bars define the "prior" swing high/low ...
EXCLUDE_LAST = 3         # ... ending 3 bars (15 min) ago: a high set on the previous bar is a
                         # continuation, not a level being retested
MIN_BARS = 6             # at least 30 min of session history before a level means anything
BREAK_TOL_PCT = 0.05     # a wick this far above the level still counts as "failed to exceed"
TIME_STOP_BARS = 24      # ~2 h hold cap, matching the user's 1-2 h holds
MAX_BOD = 60             # last entry bar 14:30 ET (bar_of_day on a 5-min RTH grid); the first
                         # possible entry follows from MIN_BARS + EXCLUDE_LAST (bar 8, 10:10 ET)


def _session_levels(bars5: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per bar: highest high / lowest low of the LOOKBACK_BARS session bars ending EXCLUDE_LAST
    bars before it (so neither the bar itself nor the last few bars count); NaN until MIN_BARS
    bars fit in that window.

    Vectorised so a ~100k-bar live frame costs milliseconds, not a Python loop: a bar far enough
    into its session has the whole window inside it, so the plain rolling extreme applies; an
    earlier bar takes the session's running extreme instead. The session key is the index date,
    the same notion make_entry uses to keep an intraday entry inside one session, and the bar's
    ordinal within that session is counted here from the rows themselves, so a missing bar can
    never let the rolling window reach into the previous session."""
    day = bars5.index.normalize()
    high, low = bars5["high"], bars5["low"]
    ordinal = high.groupby(day).cumcount().to_numpy()
    roll_hi = high.rolling(LOOKBACK_BARS, min_periods=LOOKBACK_BARS).max().shift(EXCLUDE_LAST).to_numpy(dtype=float)
    roll_lo = low.rolling(LOOKBACK_BARS, min_periods=LOOKBACK_BARS).min().shift(EXCLUDE_LAST).to_numpy(dtype=float)
    cum_hi = high.groupby(day).cummax().groupby(day).shift(EXCLUDE_LAST).to_numpy(dtype=float)
    cum_lo = low.groupby(day).cummin().groupby(day).shift(EXCLUDE_LAST).to_numpy(dtype=float)
    full = ordinal >= LOOKBACK_BARS + EXCLUDE_LAST - 1
    early = (ordinal >= MIN_BARS + EXCLUDE_LAST - 1) & ~full
    hi = np.where(full, roll_hi, np.where(early, cum_hi, np.nan))
    lo = np.where(full, roll_lo, np.where(early, cum_lo, np.nan))
    return hi, lo


def prior_high_rsi_entries(bars5, ind, daily, news, params) -> list[Entry]:
    near = float(params["near_pct"])
    rsi_short = float(params["rsi_short"])
    stop_pct = float(params["stop_pct"])
    target_r = float(params["target_r"])
    h, low, c = (bars5[x].to_numpy(dtype=float) for x in ("high", "low", "close"))
    bod, rsi = col(ind, "bar_of_day"), col(ind, "rsi")
    hi, lo = _session_levels(bars5)
    with np.errstate(invalid="ignore"):
        base = (bod <= MAX_BOD) & ~np.isnan(rsi)
        short_m = (base & ~np.isnan(hi) & (c <= hi) & ((hi - c) / hi * 100 <= near)
                   & (h <= hi * (1 + BREAK_TOL_PCT / 100)) & (rsi >= rsi_short))
        long_m = (base & ~np.isnan(lo) & (c >= lo) & ((c - lo) / lo * 100 <= near)
                  & (low >= lo * (1 - BREAK_TOL_PCT / 100)) & (rsi <= 100 - rsi_short))
    out: list[Entry] = []
    for i in candidate_positions(short_m):
        px = float(bars5["open"].iloc[i + 1])
        stop = hi[i] * (1 + stop_pct / 100)
        e = make_entry(bars5, int(i), "short", stop, px - target_r * (stop - px), TIME_STOP_BARS, "intraday", "prior_high_fade")
        if e:
            out.append(e)
    for i in candidate_positions(long_m):
        px = float(bars5["open"].iloc[i + 1])
        stop = lo[i] * (1 - stop_pct / 100)
        e = make_entry(bars5, int(i), "long", stop, px + target_r * (px - stop), TIME_STOP_BARS, "intraday", "prior_low_fade")
        if e:
            out.append(e)
    return one_per_session(out)


FAMILIES_PRIOR_HIGH_RSI: tuple[Family, ...] = (
    Family("prior_high_rsi", prior_high_rsi_entries,
           {"near_pct": [0.3, 0.6], "rsi_short": [60, 65], "stop_pct": [0.3, 0.6], "target_r": [1.0, 1.5, 2.0]},
           "intraday", ("long", "short")),
)
