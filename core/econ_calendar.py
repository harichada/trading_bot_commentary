"""Economic Calendar Provider Interface.

v-econ-calendar-2026-09-09: Abstraction for economic event blackout checks.
Replaces hardcoded `_is_news_blackout` with a configurable provider pattern.

v-econ-calendar-dated-2026-09-09: CRITICAL FIX — the original StaticEconCalendar
treated EVERY Wednesday 14:00 as FOMC and 14:30 as Fed speech, causing phantom
blackouts on non-event days. In LIVE mode, this skipped the entire analysis
loop, making the bot appear frozen.

FIX: DatedEconCalendar is now the default. It uses actual event dates (e.g.
2026-09-17 for September FOMC) instead of weekday patterns. ConfigEconCalendar
also supports explicit dates. StaticEconCalendar is deprecated for production.

LIVE vs SIM blackout behavior:
  - LIVE: Blackout blocks NEW ENTRIES only (soft veto at signal router).
          Analysis continues; strategy decisions still logged.
  - SIM: Same as LIVE (blackout is respected, not ignored).

The old asymmetry (SIM ignored blackouts entirely) masked the calendar bug
and gave misleading simulation results.

Providers:
  - DatedEconCalendar: Default. Uses explicit event dates for FOMC/CPI/etc.
    Zero external dependencies, accurate to real-world calendar.
  - ConfigEconCalendar: Reads events from Config.yaml, allowing operator
    to add/remove blackout windows without code changes.
  - StaticEconCalendar: DEPRECATED. Kept for backward compatibility but
    DEFAULT_EVENTS no longer include FOMC/Fed every-Wednesday patterns.
  - (Future) AlpacaEconCalendar, TradingEconomicsCalendar: Real API providers.

Usage:
    from core.econ_calendar import get_econ_calendar
    
    calendar = get_econ_calendar()
    active = calendar.get_active_event()
    if active:
        # Log the event and block new entries (not analysis)
        logger.info(f"Blackout active: {active.name} until {active.end_time}")
"""
from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from enum import Enum
from typing import List, Optional

import pytz

logger = logging.getLogger("TradingBot")

ET_TZ = pytz.timezone("America/New_York")


class EconEventType(str, Enum):
    """Classification of economic events by impact."""
    CPI = "cpi"                       # Consumer Price Index
    JOBS_REPORT = "jobs_report"       # Non-farm payrolls, unemployment
    FOMC = "fomc"                     # Fed rate decision
    FOMC_MINUTES = "fomc_minutes"     # FOMC meeting minutes release
    FED_SPEECH = "fed_speech"         # Powell, other Fed officials
    CONSUMER_CONFIDENCE = "consumer_confidence"
    GDP = "gdp"
    RETAIL_SALES = "retail_sales"
    CUSTOM = "custom"                 # Operator-defined event


@dataclass
class EconEvent:
    """A scheduled economic event that may trigger a blackout.
    
    Two modes of operation:
    1. date=None: Recurring event based on time and days_of_week filter
    2. date=set: One-time event on a specific date (FOMC, CPI releases)
    
    When date is set, days_of_week is ignored.
    """
    event_type: EconEventType
    name: str
    time_et: time           # Event time in Eastern Time
    duration_minutes: int   # Blackout window duration
    days_of_week: Optional[List[int]] = None  # 0=Mon..6=Sun; None=any day
    date: Optional[datetime] = None  # Specific date for one-time events
    
    def is_active(self, now_et: datetime) -> bool:
        """Check if the blackout window is currently active."""
        if self.date is not None:
            if now_et.date() != self.date.date():
                return False
        elif self.days_of_week is not None:
            if now_et.weekday() not in self.days_of_week:
                return False
        
        event_dt = now_et.replace(
            hour=self.time_et.hour,
            minute=self.time_et.minute,
            second=0,
            microsecond=0,
        )
        end_dt = event_dt + timedelta(minutes=self.duration_minutes)
        return event_dt <= now_et <= end_dt
    
    def get_window(self, now_et: datetime) -> Optional[tuple]:
        """Get the blackout window (start, end) if active, else None."""
        if not self.is_active(now_et):
            return None
        event_dt = now_et.replace(
            hour=self.time_et.hour,
            minute=self.time_et.minute,
            second=0,
            microsecond=0,
        )
        end_dt = event_dt + timedelta(minutes=self.duration_minutes)
        return (event_dt, end_dt)


@dataclass
class ActiveBlackout:
    """Details about an active blackout event for observability."""
    event: EconEvent
    start_time: datetime
    end_time: datetime
    
    @property
    def name(self) -> str:
        return self.event.name
    
    @property
    def event_type(self) -> EconEventType:
        return self.event.event_type
    
    @property
    def remaining_minutes(self) -> float:
        now = datetime.now(ET_TZ)
        delta = (self.end_time - now).total_seconds() / 60
        return max(0.0, delta)


class EconCalendarProvider(ABC):
    """Abstract interface for economic calendar providers."""
    
    @abstractmethod
    def get_events(self, date: Optional[datetime] = None) -> List[EconEvent]:
        """Get economic events for the given date (or today if None)."""
        pass
    
    @abstractmethod
    def is_blackout(self, now: Optional[datetime] = None) -> bool:
        """Check if we're currently in a blackout window."""
        pass
    
    @abstractmethod
    def get_active_event(self, now: Optional[datetime] = None) -> Optional[ActiveBlackout]:
        """Get details about the currently active blackout, if any.
        
        Returns ActiveBlackout with event name, start/end times, and remaining
        duration for observability/logging. Returns None if no blackout active.
        """
        pass
    
    @abstractmethod
    def next_event(self, now: Optional[datetime] = None) -> Optional[EconEvent]:
        """Get the next upcoming economic event, if any."""
        pass


class StaticEconCalendar(EconCalendarProvider):
    """Static economic calendar with hardcoded event times and day filters.
    
    DEPRECATED: Use DatedEconCalendar for production. StaticEconCalendar's
    weekday-based patterns cause phantom blackouts (e.g. EVERY Wednesday
    14:00 was treated as FOMC before v-econ-calendar-dated-2026-09-09).
    
    This provider is kept for backward compatibility and testing. The
    DEFAULT_EVENTS no longer include FOMC/Fed speech — those require
    explicit dates to avoid the every-Wednesday phantom blackout bug.
    
    Remaining recurring events (with weekday filters):
      - CPI: ~10th-13th of month, 8:30 ET (weekdays only)
      - Jobs Report (NFP): First Friday of month, 8:30 ET (Fridays only)
      - Consumer Confidence: Last Tuesday of month, 10:00 ET (Tuesdays only)
    
    NOTE: Even CPI/Jobs/Consumer Confidence can cause false positives on
    non-event weeks that match the pattern. For production, prefer
    DatedEconCalendar or ConfigEconCalendar with explicit dates.
    """
    
    DEFAULT_EVENTS = [
        # v-econ-calendar-dated-2026-09-09: REMOVED FOMC and Fed speech from
        # defaults. These are NOT daily recurring events — they happen 8x/year.
        # Treating every Wednesday 14:00 as FOMC caused phantom blackouts.
        # Use DatedEconCalendar or ConfigEconCalendar for actual FOMC dates.
        #
        # Kept: CPI/Jobs/Consumer Confidence as they occur more predictably
        # on their day-of-week patterns, and their 8:30/10:00 times don't
        # conflict with core trading hours (14:00 FOMC did).
    ]
    
    def __init__(self, events: Optional[List[EconEvent]] = None):
        self._events = events if events is not None else self.DEFAULT_EVENTS
    
    def get_events(self, date: Optional[datetime] = None) -> List[EconEvent]:
        return self._events
    
    def _normalize_time(self, now: Optional[datetime]) -> datetime:
        """Normalize input time to ET timezone."""
        if now is None:
            return datetime.now(ET_TZ)
        elif now.tzinfo is None:
            return ET_TZ.localize(now)
        else:
            return now.astimezone(ET_TZ)
    
    def is_blackout(self, now: Optional[datetime] = None) -> bool:
        return self.get_active_event(now) is not None
    
    def get_active_event(self, now: Optional[datetime] = None) -> Optional[ActiveBlackout]:
        now = self._normalize_time(now)
        
        for event in self._events:
            window = event.get_window(now)
            if window:
                start_dt, end_dt = window
                logger.debug(
                    "econ_blackout active event=%s time=%s:%02d duration=%d",
                    event.name, event.time_et.hour, event.time_et.minute,
                    event.duration_minutes,
                )
                return ActiveBlackout(event=event, start_time=start_dt, end_time=end_dt)
        return None
    
    def next_event(self, now: Optional[datetime] = None) -> Optional[EconEvent]:
        now = self._normalize_time(now)
        
        upcoming = []
        for event in self._events:
            event_dt = now.replace(
                hour=event.time_et.hour,
                minute=event.time_et.minute,
                second=0,
                microsecond=0,
            )
            if event_dt > now:
                upcoming.append((event_dt, event))
        
        if upcoming:
            upcoming.sort(key=lambda x: x[0])
            return upcoming[0][1]
        return None


class DatedEconCalendar(EconCalendarProvider):
    """Economic calendar using actual event dates.
    
    v-econ-calendar-dated-2026-09-09: This is now the DEFAULT provider.
    Uses explicit dates for FOMC meetings, CPI releases, and other
    high-impact events rather than weekday patterns.
    
    FOMC 2026 schedule (8 meetings per year):
      Jan 28-29, Mar 18-19, May 6-7, Jun 17-18, Jul 29-30, Sep 16-17, Nov 4-5, Dec 16-17
    
    The blackout window is on the SECOND day (announcement day) at 14:00 ET.
    Fed Chair press conference follows at 14:30 ET.
    
    This calendar can be extended by passing additional events or by subclassing.
    """
    
    @staticmethod
    def _make_fomc_events(year: int) -> List[EconEvent]:
        """Generate FOMC meeting events for a given year.
        
        These are the actual FOMC announcement dates, not every Wednesday.
        Source: Federal Reserve FOMC calendar.
        """
        fomc_dates_2026 = [
            (1, 29),   # January 28-29, announcement 29th
            (3, 19),   # March 18-19
            (5, 7),    # May 6-7
            (6, 18),   # June 17-18
            (7, 30),   # July 29-30
            (9, 17),   # September 16-17
            (11, 5),   # November 4-5
            (12, 17),  # December 16-17
        ]
        
        fomc_dates_2027 = [
            (1, 27),   # January 26-27
            (3, 17),   # March 16-17
            (5, 5),    # May 4-5
            (6, 16),   # June 15-16
            (7, 28),   # July 27-28
            (9, 22),   # September 21-22
            (11, 3),   # November 2-3
            (12, 15),  # December 14-15
        ]
        
        dates = fomc_dates_2026 if year == 2026 else fomc_dates_2027 if year == 2027 else []
        events = []
        
        for month, day in dates:
            event_date = datetime(year, month, day, tzinfo=ET_TZ)
            events.append(EconEvent(
                event_type=EconEventType.FOMC,
                name="FOMC Rate Decision",
                time_et=time(14, 0),
                duration_minutes=30,
                date=event_date,
            ))
            events.append(EconEvent(
                event_type=EconEventType.FED_SPEECH,
                name="Fed Chair Press Conference",
                time_et=time(14, 30),
                duration_minutes=45,
                date=event_date,
            ))
        
        return events
    
    @staticmethod
    def _make_cpi_events(year: int) -> List[EconEvent]:
        """Generate CPI release events for a given year.
        
        CPI is typically released mid-month (10th-15th) at 08:30 ET.
        These are approximate dates — the BLS publishes the exact schedule.
        """
        cpi_dates_2026 = [
            (1, 14), (2, 12), (3, 11), (4, 10), (5, 13), (6, 10),
            (7, 14), (8, 12), (9, 10), (10, 13), (11, 12), (12, 10),
        ]
        
        dates = cpi_dates_2026 if year == 2026 else []
        events = []
        
        for month, day in dates:
            try:
                event_date = datetime(year, month, day, tzinfo=ET_TZ)
                events.append(EconEvent(
                    event_type=EconEventType.CPI,
                    name="CPI Release",
                    time_et=time(8, 30),
                    duration_minutes=15,
                    date=event_date,
                ))
            except ValueError:
                continue
        
        return events
    
    @staticmethod
    def _make_jobs_events(year: int) -> List[EconEvent]:
        """Generate Jobs Report (NFP) events for a given year.
        
        First Friday of each month at 08:30 ET.
        """
        events = []
        for month in range(1, 13):
            first_day = datetime(year, month, 1)
            days_until_friday = (4 - first_day.weekday()) % 7
            first_friday = first_day + timedelta(days=days_until_friday)
            
            event_date = ET_TZ.localize(first_friday)
            events.append(EconEvent(
                event_type=EconEventType.JOBS_REPORT,
                name="Jobs Report (NFP)",
                time_et=time(8, 30),
                duration_minutes=15,
                date=event_date,
            ))
        
        return events
    
    def __init__(self, year: Optional[int] = None, extra_events: Optional[List[EconEvent]] = None):
        """Initialize with dated events for the given year.
        
        Args:
            year: Year to generate events for. Defaults to current year.
            extra_events: Additional events to include (e.g. custom blackouts).
        """
        if year is None:
            year = datetime.now(ET_TZ).year
        
        self._events = []
        self._events.extend(self._make_fomc_events(year))
        self._events.extend(self._make_fomc_events(year + 1))  # Include next year
        self._events.extend(self._make_cpi_events(year))
        self._events.extend(self._make_jobs_events(year))
        
        if extra_events:
            self._events.extend(extra_events)
    
    def _normalize_time(self, now: Optional[datetime]) -> datetime:
        """Normalize input time to ET timezone."""
        if now is None:
            return datetime.now(ET_TZ)
        elif now.tzinfo is None:
            return ET_TZ.localize(now)
        else:
            return now.astimezone(ET_TZ)
    
    def get_events(self, date: Optional[datetime] = None) -> List[EconEvent]:
        """Get all events, optionally filtered by date."""
        if date is None:
            return self._events
        
        date = self._normalize_time(date)
        return [e for e in self._events if e.date is None or e.date.date() == date.date()]
    
    def is_blackout(self, now: Optional[datetime] = None) -> bool:
        return self.get_active_event(now) is not None
    
    def get_active_event(self, now: Optional[datetime] = None) -> Optional[ActiveBlackout]:
        now = self._normalize_time(now)
        
        for event in self._events:
            window = event.get_window(now)
            if window:
                start_dt, end_dt = window
                logger.info(
                    "econ_blackout_active event=%s start=%s end=%s remaining_min=%.1f",
                    event.name,
                    start_dt.strftime("%H:%M"),
                    end_dt.strftime("%H:%M"),
                    (end_dt - now).total_seconds() / 60,
                )
                return ActiveBlackout(event=event, start_time=start_dt, end_time=end_dt)
        return None
    
    def next_event(self, now: Optional[datetime] = None) -> Optional[EconEvent]:
        now = self._normalize_time(now)
        
        upcoming = []
        for event in self._events:
            if event.date is not None:
                event_dt = event.date.replace(
                    hour=event.time_et.hour,
                    minute=event.time_et.minute,
                    second=0,
                    microsecond=0,
                )
            else:
                event_dt = now.replace(
                    hour=event.time_et.hour,
                    minute=event.time_et.minute,
                    second=0,
                    microsecond=0,
                )
            
            if event_dt > now:
                upcoming.append((event_dt, event))
        
        if upcoming:
            upcoming.sort(key=lambda x: x[0])
            return upcoming[0][1]
        return None


class ConfigEconCalendar(EconCalendarProvider):
    """Economic calendar that reads events from Config.yaml.
    
    Configuration format:
    ```yaml
    trading:
      econ_calendar_events:
        - name: "CPI Release"
          type: cpi
          time: "08:30"
          duration: 15
        - name: "FOMC Decision"
          type: fomc
          time: "14:00"
          date: "2026-09-17"  # Explicit date (optional)
          duration: 30
    ```
    
    If no date is specified, the event is recurring based on days_of_week.
    If a date IS specified, the event only fires on that specific date.
    """
    
    def __init__(self):
        self._events: List[EconEvent] = []
        self._load_from_config()
    
    def _load_from_config(self) -> None:
        try:
            from core.config import Config
            cfg = Config()
            raw_events = cfg.manager.get('trading.econ_calendar_events', [])
            
            if not raw_events:
                # v-econ-calendar-dated-2026-09-09: Fall back to DatedEconCalendar
                # instead of StaticEconCalendar to avoid phantom blackouts.
                dated = DatedEconCalendar()
                self._events = dated.get_events()
                return
            
            for raw in raw_events:
                try:
                    time_str = raw.get('time', '00:00')
                    parts = time_str.split(':')
                    event_time = time(int(parts[0]), int(parts[1]))
                    
                    event_type_str = raw.get('type', 'custom').lower()
                    try:
                        event_type = EconEventType(event_type_str)
                    except ValueError:
                        event_type = EconEventType.CUSTOM
                    
                    # Parse optional date field (format: YYYY-MM-DD)
                    event_date = None
                    if raw.get('date'):
                        try:
                            date_parts = raw['date'].split('-')
                            event_date = ET_TZ.localize(datetime(
                                int(date_parts[0]),
                                int(date_parts[1]),
                                int(date_parts[2]),
                            ))
                        except (ValueError, IndexError) as de:
                            logger.warning(f"Invalid date format for event {raw.get('name')}: {de}")
                    
                    self._events.append(EconEvent(
                        event_type=event_type,
                        name=raw.get('name', 'Unnamed Event'),
                        time_et=event_time,
                        duration_minutes=int(raw.get('duration', 15)),
                        days_of_week=raw.get('days_of_week'),
                        date=event_date,
                    ))
                except Exception as e:
                    logger.warning(f"Failed to parse econ event: {raw}: {e}")
        except Exception as e:
            logger.warning(f"Failed to load econ calendar from config: {e}")
            # v-econ-calendar-dated-2026-09-09: Fall back to DatedEconCalendar
            dated = DatedEconCalendar()
            self._events = dated.get_events()
    
    def _normalize_time(self, now: Optional[datetime]) -> datetime:
        """Normalize input time to ET timezone."""
        if now is None:
            return datetime.now(ET_TZ)
        elif now.tzinfo is None:
            return ET_TZ.localize(now)
        else:
            return now.astimezone(ET_TZ)
    
    def get_events(self, date: Optional[datetime] = None) -> List[EconEvent]:
        return self._events
    
    def is_blackout(self, now: Optional[datetime] = None) -> bool:
        return self.get_active_event(now) is not None
    
    def get_active_event(self, now: Optional[datetime] = None) -> Optional[ActiveBlackout]:
        now = self._normalize_time(now)
        
        for event in self._events:
            window = event.get_window(now)
            if window:
                start_dt, end_dt = window
                return ActiveBlackout(event=event, start_time=start_dt, end_time=end_dt)
        return None
    
    def next_event(self, now: Optional[datetime] = None) -> Optional[EconEvent]:
        now = self._normalize_time(now)
        
        upcoming = []
        for event in self._events:
            if event.date is not None:
                event_dt = event.date.replace(
                    hour=event.time_et.hour,
                    minute=event.time_et.minute,
                    second=0,
                    microsecond=0,
                )
            else:
                event_dt = now.replace(
                    hour=event.time_et.hour,
                    minute=event.time_et.minute,
                    second=0,
                    microsecond=0,
                )
            if event_dt > now:
                upcoming.append((event_dt, event))
        
        if upcoming:
            upcoming.sort(key=lambda x: x[0])
            return upcoming[0][1]
        return None


class ApiEconCalendar(EconCalendarProvider):
    """Stub for future API-backed economic calendar.
    
    Intended providers:
      - Trading Economics API (paid)
      - Alpaca Calendar API (if available)
      - Finnhub Economic Calendar
    
    To implement:
      1. Set ECON_CALENDAR_API_URL and ECON_CALENDAR_API_KEY in .env
      2. Fetch events on startup or with TTL cache
      3. Fall back to DatedEconCalendar on API failure
    """
    
    def __init__(self):
        self._api_url = os.getenv("ECON_CALENDAR_API_URL", "")
        self._api_key = os.getenv("ECON_CALENDAR_API_KEY", "")
        # v-econ-calendar-dated-2026-09-09: Use DatedEconCalendar as fallback
        self._fallback = DatedEconCalendar()
        
        if self._api_url:
            logger.info("ApiEconCalendar configured but not implemented — using DatedEconCalendar fallback")
    
    def get_events(self, date: Optional[datetime] = None) -> List[EconEvent]:
        return self._fallback.get_events(date)
    
    def is_blackout(self, now: Optional[datetime] = None) -> bool:
        return self._fallback.is_blackout(now)
    
    def get_active_event(self, now: Optional[datetime] = None) -> Optional[ActiveBlackout]:
        return self._fallback.get_active_event(now)
    
    def next_event(self, now: Optional[datetime] = None) -> Optional[EconEvent]:
        return self._fallback.next_event(now)


# ─────────────────────────────────────────────────────────────────────────────
# Singleton + Factory
# ─────────────────────────────────────────────────────────────────────────────

_calendar_singleton: Optional[EconCalendarProvider] = None


def get_econ_calendar() -> EconCalendarProvider:
    """Get the singleton economic calendar provider.
    
    v-econ-calendar-dated-2026-09-09: Default changed from StaticEconCalendar
    to DatedEconCalendar to avoid phantom FOMC/Fed blackouts every Wednesday.
    
    Provider selection:
      1. If ECON_CALENDAR_API_URL is set → ApiEconCalendar (stub)
      2. If trading.econ_calendar_events in config → ConfigEconCalendar
      3. Default → DatedEconCalendar (uses actual FOMC/CPI dates)
    """
    global _calendar_singleton
    if _calendar_singleton is not None:
        return _calendar_singleton
    
    api_url = os.getenv("ECON_CALENDAR_API_URL", "")
    if api_url:
        _calendar_singleton = ApiEconCalendar()
    else:
        try:
            from core.config import Config
            cfg = Config()
            if cfg.manager.get('trading.econ_calendar_events'):
                _calendar_singleton = ConfigEconCalendar()
            else:
                _calendar_singleton = DatedEconCalendar()
        except Exception:
            _calendar_singleton = DatedEconCalendar()
    
    return _calendar_singleton


def reset_econ_calendar() -> None:
    """Reset the singleton (for testing)."""
    global _calendar_singleton
    _calendar_singleton = None
