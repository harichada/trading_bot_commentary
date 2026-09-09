"""Economic Calendar Provider Interface.

v-econ-calendar-2026-09-09: Abstraction for economic event blackout checks.
Replaces hardcoded `_is_news_blackout` with a configurable provider pattern.

Providers:
  - StaticEconCalendar: Default. Uses hardcoded daily event times (CPI/Jobs,
    FOMC, etc.) as a baseline. Always available, zero external dependencies.
  - ConfigEconCalendar: Reads events from Config.yaml, allowing operator
    to add/remove blackout windows without code changes.
  - (Future) AlpacaEconCalendar, TradingEconomicsCalendar: Real API providers.

Usage:
    from core.econ_calendar import get_econ_calendar
    
    calendar = get_econ_calendar()
    if calendar.is_blackout():
        # Skip trading during high-impact event
        return
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
    """A scheduled economic event that may trigger a blackout."""
    event_type: EconEventType
    name: str
    time_et: time           # Event time in Eastern Time
    duration_minutes: int   # Blackout window duration
    days_of_week: Optional[List[int]] = None  # 0=Mon..6=Sun; None=any day
    
    def is_active(self, now_et: datetime) -> bool:
        """Check if the blackout window is currently active."""
        if self.days_of_week is not None:
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
    def next_event(self, now: Optional[datetime] = None) -> Optional[EconEvent]:
        """Get the next upcoming economic event, if any."""
        pass


class StaticEconCalendar(EconCalendarProvider):
    """Static economic calendar with hardcoded event times and day filters.
    
    This is the default provider. Events are approximate recurring times
    for major economic releases with realistic weekday filters:
    
      - CPI: ~10th-13th of month, 8:30 ET (weekdays only)
      - Jobs Report (NFP): First Friday of month, 8:30 ET (Fridays only)
      - Consumer Confidence: Last Tuesday of month, 10:00 ET (Tuesdays only)
      - FOMC: 8 times per year, Wednesdays, 14:00 ET (Wednesdays only)
      - Fed Speech: Variable, after FOMC (Wednesdays only as proxy)
    
    For precise event times, use ConfigEconCalendar with operator-
    maintained schedules, or integrate with a real calendar API.
    
    NOTE: This is conservative — may trigger blackout on non-event days
    that match the weekday pattern. For production, prefer ConfigEconCalendar
    with actual event dates or an API provider.
    """
    
    DEFAULT_EVENTS = [
        EconEvent(
            event_type=EconEventType.CPI,
            name="CPI Release",
            time_et=time(8, 30),
            duration_minutes=15,
            days_of_week=[0, 1, 2, 3, 4],  # Mon-Fri (CPI dates vary)
        ),
        EconEvent(
            event_type=EconEventType.JOBS_REPORT,
            name="Jobs Report (NFP)",
            time_et=time(8, 30),
            duration_minutes=15,
            days_of_week=[4],  # Friday only (first Friday of month)
        ),
        EconEvent(
            event_type=EconEventType.CONSUMER_CONFIDENCE,
            name="Consumer Confidence",
            time_et=time(10, 0),
            duration_minutes=15,
            days_of_week=[1],  # Tuesday only (last Tuesday of month)
        ),
        EconEvent(
            event_type=EconEventType.FOMC,
            name="FOMC Rate Decision",
            time_et=time(14, 0),
            duration_minutes=30,
            days_of_week=[2],  # Wednesday only (FOMC days)
        ),
        EconEvent(
            event_type=EconEventType.FED_SPEECH,
            name="Fed Chair Press Conference",
            time_et=time(14, 30),
            duration_minutes=15,
            days_of_week=[2],  # Wednesday only (after FOMC)
        ),
    ]
    
    def __init__(self, events: Optional[List[EconEvent]] = None):
        self._events = events if events is not None else self.DEFAULT_EVENTS
    
    def get_events(self, date: Optional[datetime] = None) -> List[EconEvent]:
        return self._events
    
    def is_blackout(self, now: Optional[datetime] = None) -> bool:
        if now is None:
            now = datetime.now(ET_TZ)
        elif now.tzinfo is None:
            now = ET_TZ.localize(now)
        else:
            now = now.astimezone(ET_TZ)
        
        for event in self._events:
            if event.is_active(now):
                logger.debug(
                    "econ_blackout active event=%s time=%s:%02d duration=%d",
                    event.name, event.time_et.hour, event.time_et.minute,
                    event.duration_minutes,
                )
                return True
        return False
    
    def next_event(self, now: Optional[datetime] = None) -> Optional[EconEvent]:
        if now is None:
            now = datetime.now(ET_TZ)
        elif now.tzinfo is None:
            now = ET_TZ.localize(now)
        else:
            now = now.astimezone(ET_TZ)
        
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
          duration: 30
    ```
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
                self._events = StaticEconCalendar.DEFAULT_EVENTS
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
                    
                    self._events.append(EconEvent(
                        event_type=event_type,
                        name=raw.get('name', 'Unnamed Event'),
                        time_et=event_time,
                        duration_minutes=int(raw.get('duration', 15)),
                        days_of_week=raw.get('days_of_week'),
                    ))
                except Exception as e:
                    logger.warning(f"Failed to parse econ event: {raw}: {e}")
        except Exception as e:
            logger.warning(f"Failed to load econ calendar from config: {e}")
            self._events = StaticEconCalendar.DEFAULT_EVENTS
    
    def get_events(self, date: Optional[datetime] = None) -> List[EconEvent]:
        return self._events
    
    def is_blackout(self, now: Optional[datetime] = None) -> bool:
        if now is None:
            now = datetime.now(ET_TZ)
        elif now.tzinfo is None:
            now = ET_TZ.localize(now)
        else:
            now = now.astimezone(ET_TZ)
        
        for event in self._events:
            if event.is_active(now):
                return True
        return False
    
    def next_event(self, now: Optional[datetime] = None) -> Optional[EconEvent]:
        if now is None:
            now = datetime.now(ET_TZ)
        elif now.tzinfo is None:
            now = ET_TZ.localize(now)
        else:
            now = now.astimezone(ET_TZ)
        
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


class ApiEconCalendar(EconCalendarProvider):
    """Stub for future API-backed economic calendar.
    
    Intended providers:
      - Trading Economics API (paid)
      - Alpaca Calendar API (if available)
      - Finnhub Economic Calendar
    
    To implement:
      1. Set ECON_CALENDAR_API_URL and ECON_CALENDAR_API_KEY in .env
      2. Fetch events on startup or with TTL cache
      3. Fall back to StaticEconCalendar on API failure
    """
    
    def __init__(self):
        self._api_url = os.getenv("ECON_CALENDAR_API_URL", "")
        self._api_key = os.getenv("ECON_CALENDAR_API_KEY", "")
        self._fallback = StaticEconCalendar()
        
        if self._api_url:
            logger.info("ApiEconCalendar configured but not implemented — using fallback")
    
    def get_events(self, date: Optional[datetime] = None) -> List[EconEvent]:
        return self._fallback.get_events(date)
    
    def is_blackout(self, now: Optional[datetime] = None) -> bool:
        return self._fallback.is_blackout(now)
    
    def next_event(self, now: Optional[datetime] = None) -> Optional[EconEvent]:
        return self._fallback.next_event(now)


# ─────────────────────────────────────────────────────────────────────────────
# Singleton + Factory
# ─────────────────────────────────────────────────────────────────────────────

_calendar_singleton: Optional[EconCalendarProvider] = None


def get_econ_calendar() -> EconCalendarProvider:
    """Get the singleton economic calendar provider.
    
    Provider selection:
      1. If ECON_CALENDAR_API_URL is set → ApiEconCalendar (stub)
      2. If trading.econ_calendar_events in config → ConfigEconCalendar
      3. Default → StaticEconCalendar
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
                _calendar_singleton = StaticEconCalendar()
        except Exception:
            _calendar_singleton = StaticEconCalendar()
    
    return _calendar_singleton


def reset_econ_calendar() -> None:
    """Reset the singleton (for testing)."""
    global _calendar_singleton
    _calendar_singleton = None
