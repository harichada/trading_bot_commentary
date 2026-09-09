"""Tests for Economic Calendar Provider.

v-econ-calendar-2026-09-09: Tests covering:
  - StaticEconCalendar default events
  - EconEvent.is_active() time window detection
  - ConfigEconCalendar loading from yaml
  - Provider singleton selection
"""
import pytest
from datetime import datetime, time, timedelta
from unittest.mock import patch, MagicMock

import pytz

from core.econ_calendar import (
    EconEvent,
    EconEventType,
    StaticEconCalendar,
    ConfigEconCalendar,
    ApiEconCalendar,
    get_econ_calendar,
    reset_econ_calendar,
    ET_TZ,
)


@pytest.fixture(autouse=True)
def reset_calendar():
    """Reset singleton before each test."""
    reset_econ_calendar()
    yield
    reset_econ_calendar()


class TestEconEvent:
    def test_is_active_during_event(self):
        """Test that is_active returns True during event window."""
        event = EconEvent(
            event_type=EconEventType.CPI,
            name="CPI Release",
            time_et=time(8, 30),
            duration_minutes=15,
        )
        
        # Simulate being at 8:35 ET
        during_event = datetime(2026, 9, 9, 8, 35, 0, tzinfo=ET_TZ)
        assert event.is_active(during_event)
    
    def test_is_active_before_event(self):
        """Test that is_active returns False before event window."""
        event = EconEvent(
            event_type=EconEventType.CPI,
            name="CPI Release",
            time_et=time(8, 30),
            duration_minutes=15,
        )
        
        before_event = datetime(2026, 9, 9, 8, 25, 0, tzinfo=ET_TZ)
        assert not event.is_active(before_event)
    
    def test_is_active_after_event(self):
        """Test that is_active returns False after event window."""
        event = EconEvent(
            event_type=EconEventType.CPI,
            name="CPI Release",
            time_et=time(8, 30),
            duration_minutes=15,
        )
        
        after_event = datetime(2026, 9, 9, 8, 50, 0, tzinfo=ET_TZ)
        assert not event.is_active(after_event)
    
    def test_is_active_at_boundary(self):
        """Test edge cases at event boundaries."""
        event = EconEvent(
            event_type=EconEventType.FOMC,
            name="FOMC Decision",
            time_et=time(14, 0),
            duration_minutes=30,
        )
        
        # At start
        at_start = datetime(2026, 9, 9, 14, 0, 0, tzinfo=ET_TZ)
        assert event.is_active(at_start)
        
        # At end
        at_end = datetime(2026, 9, 9, 14, 30, 0, tzinfo=ET_TZ)
        assert event.is_active(at_end)
        
        # Just after end
        after_end = datetime(2026, 9, 9, 14, 30, 1, tzinfo=ET_TZ)
        assert not event.is_active(after_end)
    
    def test_is_active_with_days_of_week(self):
        """Test days_of_week filtering."""
        event = EconEvent(
            event_type=EconEventType.JOBS_REPORT,
            name="Jobs Report",
            time_et=time(8, 30),
            duration_minutes=15,
            days_of_week=[4],  # Friday only
        )
        
        # Friday at 8:35
        friday = datetime(2026, 9, 11, 8, 35, 0, tzinfo=ET_TZ)  # 2026-09-11 is Friday
        assert event.is_active(friday)
        
        # Wednesday at 8:35 (same time, wrong day)
        wednesday = datetime(2026, 9, 9, 8, 35, 0, tzinfo=ET_TZ)  # Wednesday
        assert not event.is_active(wednesday)


class TestStaticEconCalendar:
    def test_default_events_exist(self):
        """Test that static calendar has default events."""
        calendar = StaticEconCalendar()
        events = calendar.get_events()
        
        assert len(events) >= 4
        event_names = [e.name for e in events]
        assert any("CPI" in n or "Jobs" in n for n in event_names)
        assert any("FOMC" in n for n in event_names)
    
    def test_is_blackout_during_cpi(self):
        """Test blackout during CPI release time."""
        calendar = StaticEconCalendar()
        
        # 8:35 ET is during CPI/Jobs blackout
        during_cpi = datetime(2026, 9, 9, 8, 35, 0, tzinfo=ET_TZ)
        assert calendar.is_blackout(during_cpi)
    
    def test_is_blackout_during_fomc(self):
        """Test blackout during FOMC window."""
        calendar = StaticEconCalendar()
        
        # 14:15 ET is during FOMC blackout
        during_fomc = datetime(2026, 9, 9, 14, 15, 0, tzinfo=ET_TZ)
        assert calendar.is_blackout(during_fomc)
    
    def test_no_blackout_normal_hours(self):
        """Test no blackout during normal trading hours."""
        calendar = StaticEconCalendar()
        
        # 11:00 ET is normal trading time
        normal = datetime(2026, 9, 9, 11, 0, 0, tzinfo=ET_TZ)
        assert not calendar.is_blackout(normal)
    
    def test_next_event(self):
        """Test next_event returns upcoming event."""
        calendar = StaticEconCalendar()
        
        # At 7:00 ET, next event should be CPI at 8:30
        morning = datetime(2026, 9, 9, 7, 0, 0, tzinfo=ET_TZ)
        next_ev = calendar.next_event(morning)
        
        assert next_ev is not None
        assert next_ev.time_et == time(8, 30)
    
    def test_no_next_event_after_all(self):
        """Test next_event returns None after all events."""
        calendar = StaticEconCalendar()
        
        # At 17:00 ET, after all events
        evening = datetime(2026, 9, 9, 17, 0, 0, tzinfo=ET_TZ)
        next_ev = calendar.next_event(evening)
        
        assert next_ev is None
    
    def test_custom_events(self):
        """Test creating calendar with custom events."""
        custom_events = [
            EconEvent(
                event_type=EconEventType.CUSTOM,
                name="Custom Event",
                time_et=time(12, 0),
                duration_minutes=10,
            ),
        ]
        calendar = StaticEconCalendar(events=custom_events)
        
        events = calendar.get_events()
        assert len(events) == 1
        assert events[0].name == "Custom Event"


class TestConfigEconCalendar:
    def test_falls_back_to_static_when_no_config(self):
        """Test that ConfigEconCalendar uses defaults when config empty."""
        # Mock config to return empty events
        with patch('core.config.ConfigManager.get') as mock_get:
            mock_get.return_value = []
            
            calendar = ConfigEconCalendar()
            events = calendar.get_events()
            
            # Should fall back to static defaults
            assert len(events) >= 4
    
    def test_loads_events_from_config(self):
        """Test loading events from config yaml format."""
        config_events = [
            {
                'name': 'Test CPI',
                'type': 'cpi',
                'time': '08:30',
                'duration': 20,
            },
            {
                'name': 'Custom Event',
                'type': 'custom',
                'time': '15:00',
                'duration': 15,
            },
        ]
        
        with patch('core.config.ConfigManager.get') as mock_get:
            mock_get.return_value = config_events
            
            calendar = ConfigEconCalendar()
            events = calendar.get_events()
            
            assert len(events) == 2
            assert events[0].name == 'Test CPI'
            assert events[0].duration_minutes == 20
            assert events[1].event_type == EconEventType.CUSTOM


class TestApiEconCalendar:
    def test_uses_fallback_when_no_api_url(self):
        """Test that ApiEconCalendar falls back to static."""
        with patch.dict('os.environ', {}, clear=True):
            calendar = ApiEconCalendar()
            events = calendar.get_events()
            
            # Should return static events
            assert len(events) >= 4


class TestGetEconCalendar:
    def test_singleton_pattern(self):
        """Test that get_econ_calendar returns singleton."""
        cal1 = get_econ_calendar()
        cal2 = get_econ_calendar()
        
        assert cal1 is cal2
    
    def test_reset_creates_new_instance(self):
        """Test that reset creates new instance."""
        cal1 = get_econ_calendar()
        reset_econ_calendar()
        cal2 = get_econ_calendar()
        
        assert cal1 is not cal2
    
    def test_uses_api_provider_when_url_set(self):
        """Test that API provider is selected when env var set."""
        with patch.dict('os.environ', {'ECON_CALENDAR_API_URL': 'http://test.api'}):
            reset_econ_calendar()
            calendar = get_econ_calendar()
            
            assert isinstance(calendar, ApiEconCalendar)
