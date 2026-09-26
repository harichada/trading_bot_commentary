# strategies/families/swing2.py
"""Batch-1 slower candidate families: fewer, longer, cheaper trades than the intraday lanes.

Five families, all filling at the open of the bar after the signal bar (`make_entry`), all reading
only lagged daily features (`d_*`, one session behind) and session-level context (`ctx_*`):

* `overnight_hold`   — 15:50 signal, 15:55 fill, exit at the next session's open (hold_kind
                       "overnight"). Buy a down day in a relatively strong name, short an up day
                       in a relatively weak one.
* `mr_short_term`    — 2-period-RSI mean reversion with a 200-day trend filter; bar-0 signal,
                       bar-1 fill, 2-5 session time stop.
* `multi_day_drop`   — three consecutive down closes inside an uptrend; ATR-multiple target.
* `pead_wide`        — post-earnings drift with no ADV-bucket restriction (the wide-universe
                       counterpart of `pead_smallcap`), 10:00 fill.
* `rs_rotation`      — weekly relative-strength rotation: first session of each week only.

Choices this module makes where the Batch-1 plan is not literal, so a reviewer can see them:

1. `overnight_hold` grid axis `move_pct` (the plan writes "down_pct/up_pct ∈ {1.0, 2.0}"). One
   symmetric axis, applied as the required down-move for longs and up-move for shorts. Two
   independent axes would double the grid with combinations that differ only in the threshold of
   the side being scored — inflating the Batch-1 search-accounting count with duplicate rows.
2. `mr_short_term`'s spec exit is "`time_stop_days` sessions, or when the daily close crosses
   `d_sma5`". The simulator's exit vocabulary is stop / target / time stop, and a "close crosses
   a moving average" exit would need a per-bar exit rule it does not have, so the exit here is
   the time stop alone (2/3/5 sessions — shorter than the ~4-session median wait for a 2-period
   RSI to revert, so it is a conservative approximation, not a free option). `d_sma5` is computed
   in `daily_features` for the eventual dynamic-exit variant; this family does not read it.
3. Short targets are capped at `MAX_SHORT_REWARD_FRAC` of the entry price. A nominal 20R target
   below a short entry goes negative once risk exceeds 5% of price, and `make_entry` (correctly)
   drops a non-positive target — which would silently delete the whole short side of a family.
   A 90%-of-price target is still effectively unreachable.
4. `rs_rotation` spacing is `time_stop_days // WEEK_SESSIONS` *signal dates*, not sessions:
   `spaced_by_sessions` counts distinct entry dates, and this family's signals are already one
   week apart, so spacing by 5 or 10 would keep one entry every 5-10 weeks instead of "one
   position at a time". 5-session holds therefore take every week, 10-session holds every other.
5. `mr_short_term` has one `rsi_extreme` axis {5, 10} mirrored as (lo, hi) = (v, 100 - v), where
   the plan writes two independent axes `rsi_lo ∈ {5, 10}` and `rsi_hi ∈ {90, 95}`. The long side
   reads only the low threshold and the short side only the high one, so two axes spend 12 grid
   points on 6 distinct per-side configurations — the same duplication `overnight_hold`'s
   `move_pct` avoids (choice 1), and the same mirroring `mr_20d`'s `rsi3_extreme` already uses.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.families.common import Family, candidate_positions, col, make_entry, spaced_by_sessions
from strategies.families.context import ctx
from strategies.families.models import Entry

SESSION_SIGNAL_BAR = 0          # bar_of_day 0 (09:30-09:35); the d_* values on it are prior-session
OVERNIGHT_SIGNAL_BAR = 76       # 15:50 bar; make_entry fills at the 15:55 bar's open
PEAD_WIDE_SIGNAL_BAR = 5        # the bar closing at 10:00; make_entry fills at the 10:00 bar's open
WEEK_SESSIONS = 5

WIDE_TARGET_R = 20.0            # "effectively no target": exits come from the time stop/next open
MAX_SHORT_REWARD_FRAC = 0.9     # see choice 3 in the module docstring

OVERNIGHT_TIME_STOP = 1         # unused by the overnight exit rule; Entry requires a positive value
MR_STOP_ATR = 2.5
DROP_STOP_ATR = 2.0
DROP_STREAK_MIN = 3.0
PEAD_WIDE_STOP_ATR = 1.5
PEAD_WIDE_TARGET_R = 3.0
RS_ROTATION_STOP_ATR = 2.5


def _atr_entries(bars5: pd.DataFrame, mask: np.ndarray, side: str, atr14: np.ndarray, stop_atr: float,
                 time_stop: int, hold_kind: str, tag: str, *, target_r: float = 0.0,
                 target_atr: float = 0.0) -> list[Entry]:
    """One entry per True in `mask`, filled at the next bar's open, risking `stop_atr × d_atr14`.

    The target is `target_r` risk-multiples away, or `target_atr × d_atr14` when that is given
    instead. A NaN/non-positive ATR or reward yields no entry (NaN comparisons are False), so a
    frame missing `d_atr14` proposes nothing rather than raising."""
    sign = 1.0 if side == "long" else -1.0
    out: list[Entry] = []
    for i in candidate_positions(mask):
        px = float(bars5["open"].iloc[i + 1])
        risk = stop_atr * float(atr14[i])
        if not np.isfinite(px) or not risk > 0:
            continue
        reward = target_atr * float(atr14[i]) if target_atr > 0 else target_r * risk
        if side == "short":
            reward = min(reward, px * MAX_SHORT_REWARD_FRAC)
        if not reward > 0:
            continue
        e = make_entry(bars5, int(i), side, px - sign * risk, px + sign * reward, time_stop, hold_kind, tag)
        if e:
            out.append(e)
    return out


def overnight_hold_entries(bars5: pd.DataFrame, ind: pd.DataFrame, daily: pd.DataFrame,
                           news: pd.DataFrame, params: dict) -> list[Entry]:
    """Long a down day in a relatively strong name (short the mirror), held 15:55 → next open."""
    move, rs_min, stop_atr = float(params["move_pct"]), float(params["rs_min"]), float(params["stop_atr"])
    bod, ret1, atr14 = col(ind, "bar_of_day"), col(ind, "d_ret1"), col(ind, "d_atr14")
    rs20 = ctx(ind, "rs_20")
    with np.errstate(invalid="ignore"):
        base = bod == OVERNIGHT_SIGNAL_BAR
        long_m = base & (ret1 <= -move) & (rs20 >= rs_min)
        short_m = base & (ret1 >= move) & (rs20 <= -rs_min)
    out = _atr_entries(bars5, long_m, "long", atr14, stop_atr, OVERNIGHT_TIME_STOP, "overnight",
                       "overnight_long", target_r=WIDE_TARGET_R)
    out += _atr_entries(bars5, short_m, "short", atr14, stop_atr, OVERNIGHT_TIME_STOP, "overnight",
                        "overnight_short", target_r=WIDE_TARGET_R)
    return spaced_by_sessions(out, 1)


def mr_short_term_entries(bars5: pd.DataFrame, ind: pd.DataFrame, daily: pd.DataFrame,
                          news: pd.DataFrame, params: dict) -> list[Entry]:
    """2-period-RSI extreme with a 200-day trend filter; time-stop exit (see docstring choice 2)."""
    rsi_lo = float(params["rsi_extreme"])           # mirrored to the short side; see choice 5
    rsi_hi = 100.0 - rsi_lo
    hold = int(params["time_stop_days"])
    bod, atr14 = col(ind, "bar_of_day"), col(ind, "d_atr14")
    close_d, sma200, rsi2 = col(ind, "d_close"), col(ind, "d_sma200"), col(ind, "d_rsi2")
    with np.errstate(invalid="ignore"):
        base = bod == SESSION_SIGNAL_BAR
        long_m = base & (rsi2 <= rsi_lo) & (close_d > sma200)
        short_m = base & (rsi2 >= rsi_hi) & (close_d < sma200)
    out = _atr_entries(bars5, long_m, "long", atr14, MR_STOP_ATR, hold, "multiday", "mr_st_long",
                       target_r=WIDE_TARGET_R)
    out += _atr_entries(bars5, short_m, "short", atr14, MR_STOP_ATR, hold, "multiday", "mr_st_short",
                        target_r=WIDE_TARGET_R)
    return spaced_by_sessions(out, hold)


def multi_day_drop_entries(bars5: pd.DataFrame, ind: pd.DataFrame, daily: pd.DataFrame,
                           news: pd.DataFrame, params: dict) -> list[Entry]:
    """Three consecutive down closes inside an uptrend; target is an ATR multiple, not an R multiple."""
    hold, target_atr = int(params["time_stop_days"]), float(params["target_atr"])
    bod, atr14 = col(ind, "bar_of_day"), col(ind, "d_atr14")
    close_d, sma200, streak = col(ind, "d_close"), col(ind, "d_sma200"), col(ind, "d_down_streak")
    with np.errstate(invalid="ignore"):
        mask = (bod == SESSION_SIGNAL_BAR) & (streak >= DROP_STREAK_MIN) & (close_d > sma200)
    out = _atr_entries(bars5, mask, "long", atr14, DROP_STOP_ATR, hold, "multiday", "multi_day_drop",
                       target_atr=target_atr)
    return spaced_by_sessions(out, hold)


def pead_wide_entries(bars5: pd.DataFrame, ind: pd.DataFrame, daily: pd.DataFrame,
                      news: pd.DataFrame, params: dict) -> list[Entry]:
    """Post-earnings drift across the whole universe: no ADV bucket, gap or intraday-direction filter."""
    surprise_min, hold = float(params["surprise_min"]), int(params["time_stop_days"])
    bod, atr14 = col(ind, "bar_of_day"), col(ind, "d_atr14")
    earnings_prev, surprise = ctx(ind, "earnings_prev"), ctx(ind, "eps_surprise_pct")
    with np.errstate(invalid="ignore"):
        base = (bod == PEAD_WIDE_SIGNAL_BAR) & (earnings_prev == 1)
        long_m = base & (surprise >= surprise_min)
        short_m = base & (surprise <= -surprise_min)
    out = _atr_entries(bars5, long_m, "long", atr14, PEAD_WIDE_STOP_ATR, hold, "multiday",
                       "pead_wide_long", target_r=PEAD_WIDE_TARGET_R)
    out += _atr_entries(bars5, short_m, "short", atr14, PEAD_WIDE_STOP_ATR, hold, "multiday",
                        "pead_wide_short", target_r=PEAD_WIDE_TARGET_R)
    return spaced_by_sessions(out, hold)


def rs_rotation_entries(bars5: pd.DataFrame, ind: pd.DataFrame, daily: pd.DataFrame,
                        news: pd.DataFrame, params: dict) -> list[Entry]:
    """Buy the strongest 60-day relative-strength names on the first session of each week."""
    rank_min, hold = float(params["rank_min"]), int(params["time_stop_days"])
    bod, atr14 = col(ind, "bar_of_day"), col(ind, "d_atr14")
    close_d, sma50, week_start = col(ind, "d_close"), col(ind, "d_sma50"), col(ind, "d_week_start")
    rs_rank = ctx(ind, "rs_rank_60")
    with np.errstate(invalid="ignore"):
        mask = ((bod == SESSION_SIGNAL_BAR) & (week_start == 1) & (rs_rank >= rank_min)
                & (close_d > sma50))
    out = _atr_entries(bars5, mask, "long", atr14, RS_ROTATION_STOP_ATR, hold, "multiday",
                       "rs_rotation", target_r=WIDE_TARGET_R)
    return spaced_by_sessions(out, max(1, hold // WEEK_SESSIONS))


FAMILIES_SWING2: tuple[Family, ...] = (
    Family("overnight_hold", overnight_hold_entries,
           {"move_pct": [1.0, 2.0], "rs_min": [0, 5], "stop_atr": [1.5, 3.0]},
           "overnight", ("long", "short")),
    Family("mr_short_term", mr_short_term_entries,
           {"rsi_extreme": [5, 10], "time_stop_days": [2, 3, 5]},
           "multiday", ("long", "short")),
    Family("multi_day_drop", multi_day_drop_entries,
           {"time_stop_days": [3, 5], "target_atr": [1.0, 2.0]},
           "multiday", ("long",)),
    Family("pead_wide", pead_wide_entries,
           {"surprise_min": [2, 5, 10], "time_stop_days": [3, 5, 10]},
           "multiday", ("long", "short")),
    Family("rs_rotation", rs_rotation_entries,
           {"rank_min": [0.9, 0.95], "time_stop_days": [5, 10]},
           "multiday", ("long",)),
)
