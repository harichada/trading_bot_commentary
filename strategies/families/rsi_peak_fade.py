"""rsi_peak_fade (2026-10-15): the user's short rule, "short when RSI is at its max and about to
fall", as a family. Long is the mirror.

Short: the 5-min RSI(14) makes a peak of at least `peak_min` within the last PEAK_MAX_LAG bars,
that peak is the highest RSI of the half hour before it, RSI has since dropped by at least
DROP_MIN points without printing at or above the peak again, it has not collapsed below RSI_MID
(that would be a chase, which is what the retired day_trade_momentum_short lane did), and price
still sits within NEAR_PCT of the bounce high. Entry at the open of the next bar, stop just above
the bounce high but never closer than STOP_MIN_PCT, target `target_r` times the risk, hold capped
at TIME_STOP_BARS. `ext_min` > 0 adds a daily context: the prior close must be that many percent
above the 20-day EMA (the user's MU shorts were in a name extended 3-12% above it); `ext_min`
<= 0 means no context. Replaces prior_high_rsi: the user's fills sit 1-3 bars after the RSI peak,
on the first stall (research/reports/user_trades/, 2026-10-15), not at a retest of a level.

The RSI column from `indicator_frame` is continuous across sessions, so a peak in the first bars of
a session is real and its half-hour window may reach into the prior session; the peak bar itself
must be inside the trigger bar's session so the bounce high is an intraday level. A half hour that
is NaN (RSI warm-up) rejects the peak: only bars that do not exist at all are skipped.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.families.common import FLATTEN_BAR, Family, candidate_positions, col, make_entry, one_per_session
from strategies.families.models import Entry

PEAK_MAX_LAG = 3         # the peak/trough may be 1..3 bars (5-15 min) before the trigger bar
DROP_MIN = 3.0           # RSI must have moved this many points off the peak/trough
PRIOR_WINDOW = 6         # the peak must be the extreme of the 6 bars (30 min) before it
NEAR_PCT = 0.4           # price within this of the bounce high/low at the trigger bar
RSI_MID = 50.0           # a short needs RSI still >= this at the trigger (long: <= this)
STOP_PAD_PCT = 0.05      # stop sits this far beyond the bounce extreme ...
STOP_MIN_PCT = 0.3       # ... and never closer than this to the entry (the user's worst adverse moves)
TIME_STOP_BARS = 24      # ~2 h hold cap, the user's median hold is ~90 min
MAX_BOD = FLATTEN_BAR - 2  # last trigger bar 14:50 ET -> last entry at the 14:55 open, nothing at/after 15:00

Masks = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def _turn_masks(rsi: np.ndarray, bod: np.ndarray, peak_min: float) -> Masks:
    """Per bar i: (short mask, long mask, lag of the qualifying peak, lag of the qualifying trough);
    a lag is 0 when that side does not qualify.

    A peak at lag k qualifies when it is >= peak_min, is the highest RSI of the PRIOR_WINDOW bars
    before it (a NaN in that window rejects; bars before the start of the frame are skipped),
    every print since (including bar i) stayed below it, bar i is at least DROP_MIN below it, and
    the peak bar is inside bar i's session (bod >= k). The trough is the mirror against
    100 - peak_min. At most one lag can qualify per side: a nearer bar cannot be the window
    extreme when the true extreme lies between it and the window."""
    s = pd.Series(rsi)
    pos = np.arange(len(rsi))
    roll_max = s.rolling(PRIOR_WINDOW, min_periods=1).max()
    roll_min = s.rolling(PRIOR_WINDOW, min_periods=1).min()
    roll_nan = s.isna().rolling(PRIOR_WINDOW, min_periods=1).max().astype(bool)
    short = np.zeros(len(rsi), dtype=bool)
    long = np.zeros(len(rsi), dtype=bool)
    lag_s = np.zeros(len(rsi), dtype=int)
    lag_l = np.zeros(len(rsi), dtype=int)
    with np.errstate(invalid="ignore"):
        for k in range(1, PEAK_MAX_LAG + 1):
            p = s.shift(k)
            no_prior = pos - k - 1 < 0                       # the bar before the peak is before the frame
            prior_ok_max = (p >= roll_max.shift(k + 1)) & ~roll_nan.shift(k + 1, fill_value=False)
            prior_ok_min = (p <= roll_min.shift(k + 1)) & ~roll_nan.shift(k + 1, fill_value=False)
            since_below = pd.Series(True, index=s.index)
            since_above = pd.Series(True, index=s.index)
            for j in range(1, k):
                since_below &= s.shift(j) < p
                since_above &= s.shift(j) > p
            in_session = bod >= k
            sk = ((p >= peak_min) & (no_prior | prior_ok_max) & since_below
                  & (s <= p - DROP_MIN) & (s >= RSI_MID) & in_session).to_numpy(dtype=bool)
            lk = ((p <= 100 - peak_min) & (no_prior | prior_ok_min) & since_above
                  & (s >= p + DROP_MIN) & (s <= RSI_MID) & in_session).to_numpy(dtype=bool)
            lag_s[sk] = k
            lag_l[lk] = k
            short |= sk
            long |= lk
    return short, long, lag_s, lag_l


def _bounce_extremes(high: np.ndarray, low: np.ndarray, lag_s: np.ndarray, lag_l: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Highest high from the peak bar through the trigger bar (shorts) and lowest low from the
    trough bar through the trigger bar (longs); NaN where that side does not qualify."""
    lags = range(1, PEAK_MAX_LAG + 1)
    hi_k = {k: pd.Series(high).rolling(k + 1).max().to_numpy() for k in lags}
    lo_k = {k: pd.Series(low).rolling(k + 1).min().to_numpy() for k in lags}
    hi = np.select([lag_s == k for k in lags], [hi_k[k] for k in lags], np.nan)
    lo = np.select([lag_l == k for k in lags], [lo_k[k] for k in lags], np.nan)
    return hi, lo


def rsi_peak_fade_entries(bars5: pd.DataFrame, ind: pd.DataFrame, daily: pd.DataFrame, news: pd.DataFrame,
                          params: dict) -> list[Entry]:
    peak_min = float(params["peak_min"])
    ext_min = float(params["ext_min"])
    target_r = float(params["target_r"])
    high, low, close = (bars5[x].to_numpy(dtype=float) for x in ("high", "low", "close"))
    bod, rsi, d_close, d_ema20 = (col(ind, x) for x in ("bar_of_day", "rsi", "d_close", "d_ema20"))
    short_turn, long_turn, lag_s, lag_l = _turn_masks(rsi, bod, peak_min)
    bounce_hi, bounce_lo = _bounce_extremes(high, low, lag_s, lag_l)
    with np.errstate(invalid="ignore", divide="ignore"):
        ext = (d_close / d_ema20 - 1) * 100
        ctx_short = np.ones(len(close), dtype=bool) if ext_min <= 0 else (ext >= ext_min)
        ctx_long = np.ones(len(close), dtype=bool) if ext_min <= 0 else (ext <= -ext_min)
        base = bod <= MAX_BOD
        short_m = base & ctx_short & short_turn & ((bounce_hi - close) / bounce_hi * 100 <= NEAR_PCT)
        long_m = base & ctx_long & long_turn & ((close - bounce_lo) / bounce_lo * 100 <= NEAR_PCT)
    out: list[Entry] = []
    for i in candidate_positions(short_m):
        px = float(bars5["open"].iloc[i + 1])
        stop = max(bounce_hi[i] * (1 + STOP_PAD_PCT / 100), px * (1 + STOP_MIN_PCT / 100))
        e = make_entry(bars5, int(i), "short", stop, px - target_r * (stop - px), TIME_STOP_BARS, "intraday", "rsi_peak_fade")
        if e:
            out.append(e)
    for i in candidate_positions(long_m):
        px = float(bars5["open"].iloc[i + 1])
        stop = min(bounce_lo[i] * (1 - STOP_PAD_PCT / 100), px * (1 - STOP_MIN_PCT / 100))
        e = make_entry(bars5, int(i), "long", stop, px + target_r * (px - stop), TIME_STOP_BARS, "intraday", "rsi_trough_fade")
        if e:
            out.append(e)
    return one_per_session(out)


FAMILIES_RSI_PEAK_FADE: tuple[Family, ...] = (
    Family("rsi_peak_fade", rsi_peak_fade_entries,
           {"peak_min": [55, 60, 65], "ext_min": [0, 3], "target_r": [1.0, 1.5, 2.0]},
           "intraday", ("long", "short")),
)
