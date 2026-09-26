"""Batch-2 Task C factor families: monthly signals fed by the cross-sectional momentum and
low-volatility context columns (research/strategy_hunt/context.py's mom_12_1 / mom_rank_12_1 /
vol_rank_60).

Both families signal at `bar_of_day == 0` on the first session of each calendar month
(`d_month_start`, strategies/families/indicators.py) and fill at the next bar's open --
`swing2.py`'s slower-family pattern (session-open signal, next-bar fill, `d_atr14`-scaled stop, an
R-multiple target, a time stop). Entry construction is not imported from `swing2.py`: that module
is owned by a task running concurrently in this plan, so a private local `_atr_entries` avoids
coupling this module's behavior to a file it does not control.

* `mom_12_1`      -- 12-1 momentum: long the top-`rank_min` percentile ranked names in an
                     uptrend, short the bottom-`rank_min` percentile names in a downtrend, once a
                     month. `entry_style` (Batch-2 Task B) is a grid axis here, since Task B had
                     landed (`strategies.families.models.ENTRY_STYLES`) by the time this module
                     was written.
* `low_vol_trend` -- long-only: low 60-session realized-volatility names with positive
                     120-session relative strength and price above the 200-day average. No
                     `entry_style` axis (the plan's fallback for a family not listed in Task B).

Short targets in `mom_12_1` are capped at `MAX_SHORT_REWARD_FRAC` of the entry price, mirroring
`swing2.py`'s `overnight_hold`/`mr_short_term` choice: a nominal 20R target below a short entry
goes negative once risk exceeds 5% of price, and `make_entry` (correctly) drops a non-positive
target, which would silently delete the whole short side of the family.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.families.common import Family, candidate_positions, col, make_entry, spaced_by_sessions
from strategies.families.context import ctx
from strategies.families.models import Entry

FACTOR_SIGNAL_BAR = 0             # bar_of_day 0 (09:30-09:35); the d_* values on it are prior-session
MAX_SHORT_REWARD_FRAC = 0.9       # see module docstring

MOM_STOP_ATR = 3.0
MOM_TARGET_R = 20.0
MOM_TIME_STOP = 20

LOW_VOL_STOP_ATR = 2.5
LOW_VOL_TARGET_R = 20.0
LOW_VOL_TIME_STOP = 20


def _entry_style(params: dict) -> str:
    """The Batch-2 execution style from a params dict, defaulting to the market fill so a grid
    without the axis (`low_vol_trend`) behaves exactly like a Batch-1 family."""
    return str(params.get("entry_style", "market"))


def _atr_entries(bars5: pd.DataFrame, mask: np.ndarray, side: str, atr14: np.ndarray, stop_atr: float,
                 time_stop: int, tag: str, target_r: float, entry_style: str = "market") -> list[Entry]:
    """One entry per True in `mask`, filled at the next bar's open, risking `stop_atr x d_atr14`
    and targeting `target_r` risk-multiples away; short rewards are capped (see module docstring).
    A NaN/non-positive ATR or reward yields no entry, so a frame missing `d_atr14` proposes
    nothing rather than raising."""
    sign = 1.0 if side == "long" else -1.0
    out: list[Entry] = []
    for i in candidate_positions(mask):
        px = float(bars5["open"].iloc[i + 1])
        risk = stop_atr * float(atr14[i])
        if not np.isfinite(px) or not risk > 0:
            continue
        reward = target_r * risk
        if side == "short":
            reward = min(reward, px * MAX_SHORT_REWARD_FRAC)
        if not reward > 0:
            continue
        e = make_entry(bars5, int(i), side, px - sign * risk, px + sign * reward, time_stop, "multiday", tag,
                       entry_style=entry_style)
        if e:
            out.append(e)
    return out


def mom_12_1_entries(bars5: pd.DataFrame, ind: pd.DataFrame, daily: pd.DataFrame,
                     news: pd.DataFrame, params: dict) -> list[Entry]:
    """12-1 momentum, monthly: long the top `rank_min` percentile in an uptrend, short the bottom
    `1 - rank_min` percentile in a downtrend."""
    rank_min = float(params["rank_min"])
    style = _entry_style(params)
    bod, atr14 = col(ind, "bar_of_day"), col(ind, "d_atr14")
    close_d, sma200, month_start = col(ind, "d_close"), col(ind, "d_sma200"), col(ind, "d_month_start")
    mom_rank = ctx(ind, "mom_rank_12_1")
    with np.errstate(invalid="ignore"):
        base = (bod == FACTOR_SIGNAL_BAR) & (month_start == 1)
        long_m = base & (mom_rank >= rank_min) & (close_d > sma200)
        short_m = base & (mom_rank <= 1.0 - rank_min) & (close_d < sma200)
    out = _atr_entries(bars5, long_m, "long", atr14, MOM_STOP_ATR, MOM_TIME_STOP, "mom_12_1_long",
                       target_r=MOM_TARGET_R, entry_style=style)
    out += _atr_entries(bars5, short_m, "short", atr14, MOM_STOP_ATR, MOM_TIME_STOP, "mom_12_1_short",
                        target_r=MOM_TARGET_R, entry_style=style)
    return spaced_by_sessions(out, 1)


def low_vol_trend_entries(bars5: pd.DataFrame, ind: pd.DataFrame, daily: pd.DataFrame,
                          news: pd.DataFrame, params: dict) -> list[Entry]:
    """Long low-60-session-realized-volatility names in an uptrend with positive 120-session
    relative strength, monthly."""
    vol_max = float(params["vol_max"])
    bod, atr14 = col(ind, "bar_of_day"), col(ind, "d_atr14")
    close_d, sma200, month_start = col(ind, "d_close"), col(ind, "d_sma200"), col(ind, "d_month_start")
    vol_rank, rs120 = ctx(ind, "vol_rank_60"), ctx(ind, "rs_120")
    with np.errstate(invalid="ignore"):
        mask = ((bod == FACTOR_SIGNAL_BAR) & (month_start == 1) & (vol_rank <= vol_max)
                & (rs120 > 0) & (close_d > sma200))
    out = _atr_entries(bars5, mask, "long", atr14, LOW_VOL_STOP_ATR, LOW_VOL_TIME_STOP,
                       "low_vol_trend", target_r=LOW_VOL_TARGET_R)
    return spaced_by_sessions(out, 1)


FAMILIES_FACTOR: tuple[Family, ...] = (
    Family("mom_12_1", mom_12_1_entries,
           {"rank_min": [0.8, 0.9], "entry_style": ["market", "limit_open"]},
           "multiday", ("long", "short")),
    Family("low_vol_trend", low_vol_trend_entries, {"vol_max": [0.2, 0.3]}, "multiday", ("long",)),
)
