"""Entry-time market regime labels — the same thresholds as the live gate, from data known at entry.

v-playbook-desk-b1-6 (final review, Critical 1). The first version of this module labelled a
trade with the regime computed from the *entry day's* SPY and VIX **closes**. That is a
look-ahead: a 10:00 entry was stamped `risk_off` because SPY closed −1% six hours later, and
every regime-conditioned row (and the live gate that reads them) was fitted to information the
desk could not have had when the order went in.

The label is now reconstructed the way `core/market_context.build_market_context` builds it
live, with the same three inputs and the same `classify_regime` thresholds:

* `spy_change_pct` — SPY's day-to-date move: the close of the last 5-minute SPY bar that closed
  **at or before the entry bar's open**, against that session's open. An entry on the session's
  first bar has nothing closed yet, so its day-to-date change is `0.0`.
* `vix_level` — the **prior** session's VIX close.
* `vix_change_pct` — the **prior** session's close-to-close VIX change.

The VIX pair is a documented *lagged proxy*: the research store holds daily VIX only, so an
intraday VIX level is unavailable. Live, `vix_change_pct` is today's move; here it is
yesterday's. Lagging is the conservative direction — it can never smuggle in the future —
but it does mean a regime row is a slightly blunter instrument than the live gate's label on
days when VIX turns intraday.

`unknown` is returned whenever the inputs are not there: no SPY bars for that session, or no
prior VIX session (and no session before that, so no close-to-close change). Trades stamped
`unknown` are dropped from the regime and pooled rows by `score/regime_rows.py`; they still
count in the unconditional row.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

UNKNOWN_REGIME = "unknown"


def classify_regime(spy_change_pct: float, vix_level: float, vix_change_pct: float) -> str:
    """Parity with `core.market_context._classify_regime` — change one and the other must follow.

    `tests/research/strategy_hunt/test_regime.py::test_classify_regime_is_parity_with_core_market_context`
    compares the two functions across the threshold grid so a live-side edit cannot silently
    leave research scoring a different regime than the gate enforces.
    """
    if spy_change_pct > 0.3 and (vix_change_pct < 0 or vix_level < 16):
        return "risk_on"
    if spy_change_pct < -0.5 and vix_change_pct > 5:
        return "risk_off"
    if vix_change_pct > 10:
        return "risk_off"
    return "mixed"


def _prior_vix_by_session(sessions: pd.DatetimeIndex, vix_daily: pd.DataFrame) -> dict:
    """session date -> (prior VIX close, prior close-to-close change %).

    A session with fewer than two VIX closes strictly before it is absent from the mapping,
    which is what makes its label `unknown` rather than a guess."""
    if vix_daily is None or vix_daily.empty or "close" not in vix_daily.columns:
        return {}
    vix = vix_daily["close"].astype(float)
    vix = vix[~vix.index.duplicated(keep="last")].sort_index()
    dates = pd.DatetimeIndex(vix.index).normalize()
    values = vix.to_numpy(dtype=float)
    # `side="left"` so a VIX row stamped with the session's own date is excluded: the entry-time
    # label may only use closes from sessions that had already finished.
    positions = dates.searchsorted(sessions, side="left") - 1
    out: dict = {}
    for day, j in zip(sessions, positions):
        if j < 1:
            continue
        prev, prev2 = values[j], values[j - 1]
        if not np.isfinite(prev) or not np.isfinite(prev2) or prev2 == 0:
            continue
        out[pd.Timestamp(day)] = (float(prev), float((prev / prev2 - 1.0) * 100.0))
    return out


class RegimeLookup:
    """`lookup(entry_ts) -> regime label`, using only bars closed before `entry_ts`.

    Built once per symbol run by `build_regime_lookup` and handed to the simulator, which
    stamps every `SimTrade` with `lookup(entry_ts)` where `entry_ts` is the *fill* bar's
    timestamp. The shuffle control shares this object: SPY is never shuffled, so the control
    tape is labelled by the same real-tape regimes as the real one.
    """

    __slots__ = ("_bar_ns", "_bar_day_ns", "_day_change_pct", "_sessions", "_vix", "_cache")

    def __init__(self, bar_ns: np.ndarray, bar_day_ns: np.ndarray, day_change_pct: np.ndarray,
                 sessions: pd.DatetimeIndex, vix: dict) -> None:
        self._bar_ns = bar_ns
        self._bar_day_ns = bar_day_ns
        self._day_change_pct = day_change_pct
        self._sessions = {pd.Timestamp(d) for d in sessions}
        self._vix = vix
        # Many parameter sets propose entries on the same bars, so the same timestamp is asked
        # for dozens of times per symbol run; the label is a pure function of it.
        self._cache: dict = {}

    def __len__(self) -> int:
        return len(self._sessions)

    def __call__(self, ts) -> str:
        key = pd.Timestamp(ts)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        label = self._label(key)
        self._cache[key] = label
        return label

    def _label(self, ts: pd.Timestamp) -> str:
        day = ts.normalize()
        if day not in self._sessions:
            return UNKNOWN_REGIME
        vix = self._vix.get(day)
        if vix is None:
            return UNKNOWN_REGIME
        i = int(np.searchsorted(self._bar_ns, ts.value, side="left")) - 1
        # `side="left"` excludes the entry bar itself; the bar at i closed at (or before) the
        # entry bar's open. If that bar belongs to an earlier session, nothing has closed today.
        if i >= 0 and self._bar_day_ns[i] == day.value:
            spy_change_pct = float(self._day_change_pct[i])
        else:
            spy_change_pct = 0.0
        if not np.isfinite(spy_change_pct):
            return UNKNOWN_REGIME
        return classify_regime(spy_change_pct, vix[0], vix[1])


def build_regime_lookup(spy5: pd.DataFrame, vix_daily: Optional[pd.DataFrame]) -> RegimeLookup:
    """A `RegimeLookup` from 5-minute SPY bars and daily VIX closes (see the module docstring)."""
    spy5 = spy5.sort_index()
    if spy5.empty:
        return RegimeLookup(np.array([], dtype=np.int64), np.array([], dtype=np.int64),
                            np.array([], dtype=float), pd.DatetimeIndex([]), {})
    days = spy5.index.normalize()
    session_open = spy5["open"].astype(float).groupby(days).transform("first")
    change = (spy5["close"].astype(float) / session_open.replace(0.0, np.nan) - 1.0) * 100.0
    sessions = pd.DatetimeIndex(days.unique())
    return RegimeLookup(spy5.index.asi8.copy(), days.asi8.copy(), change.to_numpy(dtype=float),
                        sessions, _prior_vix_by_session(sessions, vix_daily))
