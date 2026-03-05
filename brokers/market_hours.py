"""Market-hours policies for different asset classes.

Each policy knows when its market is open, closed, or on holiday,
and can answer questions the trading loop needs:

- Is the market open right now?
- When does it next open?
- Is this within the entry window?  The exit window?
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, datetime, timedelta
from typing import Tuple


class MarketHoursPolicy(ABC):
    """Abstract market-hours contract."""

    @abstractmethod
    def is_market_day(self, now: datetime) -> Tuple[bool, datetime]:
        """Check whether *now* falls on a trading day.

        Returns ``(is_open_day, next_open)`` where *next_open* is the
        earliest datetime trading resumes (only meaningful when
        ``is_open_day`` is False).
        """

    @abstractmethod
    def is_in_session(self, now: datetime) -> bool:
        """True if *now* is within the regular trading session."""

    @abstractmethod
    def is_entry_allowed(
        self,
        now: datetime,
        entry_cutoff_hour: int,
        entry_cutoff_min: int,
    ) -> Tuple[bool, str]:
        """Check whether new entries are allowed right now.

        Returns ``(allowed, reason)`` — *reason* explains why if not.
        """

    @abstractmethod
    def is_eod_close(self, now: datetime, eod_hour: int, eod_min: int) -> bool:
        """True if *now* is at or past the end-of-day forced-close time."""

    @abstractmethod
    def is_pre_session(self, now: datetime) -> bool:
        """True if *now* is before the trading day has started.

        Used by the trading loop to sleep until pre-market activity begins.
        """

    @abstractmethod
    def is_post_session(self, now: datetime) -> bool:
        """True if *now* is after the trading day has ended.

        Used by the trading loop to trigger EOD cleanup and sleep.
        """


# ---------------------------------------------------------------------------
# US Equity hours  (NYSE / NASDAQ)
# ---------------------------------------------------------------------------

class USEquityHours(MarketHoursPolicy):
    """NYSE/NASDAQ regular session: 9:30 AM – 4:00 PM ET, weekdays only.

    Holiday calendar is self-contained (no external dependency).  Logic
    extracted from ``GapFadeLiveTrader._is_market_day`` in gap_fade_app.py.
    """

    # -- MarketHoursPolicy interface -----------------------------------------

    def is_market_day(self, now: datetime) -> Tuple[bool, datetime]:
        # Weekend check
        if now.weekday() >= 5:
            days_until_monday = 7 - now.weekday()
            next_open = (now + timedelta(days=days_until_monday)).replace(
                hour=7, minute=0, second=0, microsecond=0)
            return False, next_open

        holidays = self._us_market_holidays(now.year)
        today = now.date()
        if today in holidays:
            next_day = now + timedelta(days=1)
            while next_day.weekday() >= 5 or next_day.date() in holidays:
                next_day += timedelta(days=1)
            next_open = next_day.replace(hour=7, minute=0, second=0, microsecond=0)
            return False, next_open

        return True, now

    def is_in_session(self, now: datetime) -> bool:
        if now.weekday() >= 5:
            return False
        # Regular session 9:30 – 16:00 ET
        t = now.hour * 60 + now.minute
        return 9 * 60 + 30 <= t < 16 * 60

    def is_entry_allowed(
        self,
        now: datetime,
        entry_cutoff_hour: int,
        entry_cutoff_min: int,
    ) -> Tuple[bool, str]:
        # Before market open
        if now.hour < 9 or (now.hour == 9 and now.minute < 30):
            return False, f"market not open yet ({now.strftime('%H:%M')} ET, opens 9:30)"
        # Past entry cutoff
        cutoff = now.replace(
            hour=entry_cutoff_hour, minute=entry_cutoff_min,
            second=0, microsecond=0,
        )
        if now >= cutoff:
            return False, f"past entry cutoff ({entry_cutoff_hour}:{entry_cutoff_min:02d})"
        return True, ''

    def is_eod_close(self, now: datetime, eod_hour: int, eod_min: int) -> bool:
        return (now.hour > eod_hour
                or (now.hour == eod_hour and now.minute >= eod_min))

    def is_pre_session(self, now: datetime) -> bool:
        return now.hour < 7

    def is_post_session(self, now: datetime) -> bool:
        return now.hour >= 16

    # -- Holiday helpers (extracted from GapFadeLiveTrader) ------------------

    @staticmethod
    def _us_market_holidays(year: int) -> set:  # noqa: C901
        """Compute the full set of NYSE holidays for *year*."""
        holidays: set = set()

        # Fixed-date holidays (with Sat→Fri / Sun→Mon observance)
        for m, d in [(1, 1), (6, 19), (7, 4), (12, 25)]:
            dt = date(year, m, d)
            if dt.weekday() == 5:
                holidays.add(date(year, m, d - 1))
            elif dt.weekday() == 6:
                holidays.add(date(year, m, d + 1))
            else:
                holidays.add(dt)

        # Moving holidays
        holidays.add(_nth_weekday(year, 1, 0, 3))   # MLK Day
        holidays.add(_nth_weekday(year, 2, 0, 3))   # Presidents' Day
        holidays.add(_last_weekday(year, 5, 0))      # Memorial Day
        holidays.add(_nth_weekday(year, 9, 0, 1))   # Labor Day
        holidays.add(_nth_weekday(year, 11, 3, 4))  # Thanksgiving
        holidays.add(_good_friday(year))

        return holidays


# ---------------------------------------------------------------------------
# Forex hours  (24/5 continuous session)
# ---------------------------------------------------------------------------

class ForexHours(MarketHoursPolicy):
    """Forex 24/5 session: Sunday 17:00 ET through Friday 16:00 ET.

    The forex market is a single continuous session — there is no
    intraday open/close.  The only closure is the weekend gap from
    Friday 16:00 ET to Sunday 17:00 ET.
    """

    def is_market_day(self, now: datetime) -> Tuple[bool, datetime]:
        if self._in_weekend_gap(now):
            # Next open is Sunday 17:00 of the current or upcoming week
            next_sun = self._next_sunday_5pm(now)
            return False, next_sun
        return True, now

    def is_in_session(self, now: datetime) -> bool:
        return not self._in_weekend_gap(now)

    def is_entry_allowed(
        self,
        now: datetime,
        entry_cutoff_hour: int,
        entry_cutoff_min: int,
    ) -> Tuple[bool, str]:
        if self._in_weekend_gap(now):
            return False, "forex market closed (weekend gap)"
        # Block entries on Fridays past the cutoff (weekend gap risk)
        if now.weekday() == 4:  # Friday
            cutoff = now.replace(
                hour=entry_cutoff_hour, minute=entry_cutoff_min,
                second=0, microsecond=0,
            )
            if now >= cutoff:
                return False, (
                    f"past Friday entry cutoff "
                    f"({entry_cutoff_hour}:{entry_cutoff_min:02d})"
                )
        return True, ''

    def is_eod_close(self, now: datetime, eod_hour: int, eod_min: int) -> bool:
        # Forced close only on Fridays (to avoid weekend gap exposure)
        if now.weekday() != 4:
            return False
        return (now.hour > eod_hour
                or (now.hour == eod_hour and now.minute >= eod_min))

    def is_pre_session(self, now: datetime) -> bool:
        # Forex is 24/5 — no pre-session concept (always False when market open)
        return False

    def is_post_session(self, now: datetime) -> bool:
        # Only during the weekend gap (Fri 16:00 – Sun 17:00 ET)
        return self._in_weekend_gap(now)

    # -- Internal helpers -----------------------------------------------------

    @staticmethod
    def _in_weekend_gap(now: datetime) -> bool:
        """True if *now* falls in the Fri 16:00 – Sun 17:00 ET gap."""
        wd = now.weekday()
        if wd == 5:  # Saturday — always closed
            return True
        if wd == 6:  # Sunday — closed before 17:00
            return now.hour < 17
        if wd == 4:  # Friday — closed at/after 16:00
            return now.hour >= 16
        return False

    @staticmethod
    def _next_sunday_5pm(now: datetime) -> datetime:
        """Return the next Sunday 17:00 ET at or after *now*."""
        days_until_sunday = (6 - now.weekday()) % 7
        if days_until_sunday == 0 and now.hour >= 17:
            days_until_sunday = 7
        candidate = (now + timedelta(days=days_until_sunday)).replace(
            hour=17, minute=0, second=0, microsecond=0,
        )
        return candidate


# ---------------------------------------------------------------------------
# Standalone calendar helpers (usable without a class instance)
# ---------------------------------------------------------------------------

def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """Return the *n*-th occurrence of *weekday* in *month*."""
    first = date(year, month, 1)
    diff = (weekday - first.weekday()) % 7
    return first + timedelta(days=diff + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    """Return the last occurrence of *weekday* in *month*."""
    if month == 12:
        last_day = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        last_day = date(year, month + 1, 1) - timedelta(days=1)
    diff = (last_day.weekday() - weekday) % 7
    return last_day - timedelta(days=diff)


def _good_friday(year: int) -> date:
    """Compute Good Friday via the anonymous Gregorian algorithm."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7  # noqa: E741
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    easter = date(year, month, day + 1)
    return easter - timedelta(days=2)
