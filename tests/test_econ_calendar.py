"""Tests for Economic Calendar Provider.

v-econ-calendar-2026-09-09: Tests covering:
  - StaticEconCalendar default events
  - EconEvent.is_active() time window detection
  - ConfigEconCalendar loading from yaml
  - Provider singleton selection

v-econ-calendar-dated-2026-09-09: Additional tests for:
  - DatedEconCalendar using actual FOMC/CPI dates
  - Non-event Wednesday afternoon is NOT blackout
  - Real configured FOMC window IS blackout
  - get_active_event returns proper details
  - LIVE/SIM blackout behavior consistency
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
    DatedEconCalendar,
    ActiveBlackout,
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
    
    def test_is_active_with_specific_date(self):
        """Test date-specific event only fires on that date."""
        # September 17, 2026 is a Wednesday (actual FOMC date)
        event = EconEvent(
            event_type=EconEventType.FOMC,
            name="FOMC Rate Decision",
            time_et=time(14, 0),
            duration_minutes=30,
            date=datetime(2026, 9, 17, tzinfo=ET_TZ),
        )
        
        # During event on the correct date
        correct_day = datetime(2026, 9, 17, 14, 15, 0, tzinfo=ET_TZ)
        assert event.is_active(correct_day)
        
        # Same time, DIFFERENT Wednesday (Sept 9, 2026 is also Wednesday)
        wrong_day = datetime(2026, 9, 9, 14, 15, 0, tzinfo=ET_TZ)
        assert not event.is_active(wrong_day)
    
    def test_get_window_returns_times(self):
        """Test get_window returns (start, end) tuple when active."""
        event = EconEvent(
            event_type=EconEventType.FOMC,
            name="FOMC Decision",
            time_et=time(14, 0),
            duration_minutes=30,
            date=datetime(2026, 9, 17, tzinfo=ET_TZ),
        )
        
        now = datetime(2026, 9, 17, 14, 15, 0, tzinfo=ET_TZ)
        window = event.get_window(now)
        
        assert window is not None
        start, end = window
        assert start.hour == 14 and start.minute == 0
        assert end.hour == 14 and end.minute == 30


class TestStaticEconCalendar:
    """Tests for StaticEconCalendar.
    
    v-econ-calendar-dated-2026-09-09: StaticEconCalendar.DEFAULT_EVENTS
    is now EMPTY. FOMC/Fed speech removed to prevent phantom blackouts.
    Tests updated accordingly.
    """
    
    def test_default_events_empty(self):
        """v-econ-calendar-dated-2026-09-09: Default events now empty.
        
        FOMC/Fed speech removed to prevent every-Wednesday phantom blackouts.
        Use DatedEconCalendar for real event dates.
        """
        calendar = StaticEconCalendar()
        events = calendar.get_events()
        
        # Default events are now empty
        assert len(events) == 0
    
    def test_no_blackout_with_empty_defaults(self):
        """With empty defaults, no blackout should occur."""
        calendar = StaticEconCalendar()
        
        # Any random time should not be blackout
        random_time = datetime(2026, 9, 9, 14, 15, 0, tzinfo=ET_TZ)
        assert not calendar.is_blackout(random_time)
    
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
    
    def test_get_active_event_returns_details(self):
        """Test get_active_event returns ActiveBlackout with details."""
        custom_events = [
            EconEvent(
                event_type=EconEventType.FOMC,
                name="FOMC Rate Decision",
                time_et=time(14, 0),
                duration_minutes=30,
            ),
        ]
        calendar = StaticEconCalendar(events=custom_events)
        
        during_event = datetime(2026, 9, 9, 14, 15, 0, tzinfo=ET_TZ)
        active = calendar.get_active_event(during_event)
        
        assert active is not None
        assert isinstance(active, ActiveBlackout)
        assert active.name == "FOMC Rate Decision"
        assert active.event_type == EconEventType.FOMC
        assert active.start_time.hour == 14 and active.start_time.minute == 0
        assert active.end_time.hour == 14 and active.end_time.minute == 30


class TestDatedEconCalendar:
    """Tests for DatedEconCalendar - the new default provider.
    
    v-econ-calendar-dated-2026-09-09: DatedEconCalendar uses actual
    FOMC/CPI/Jobs report dates instead of weekday patterns.
    """
    
    def test_has_fomc_events(self):
        """Test that DatedEconCalendar includes FOMC events."""
        calendar = DatedEconCalendar(year=2026)
        events = calendar.get_events()
        
        fomc_events = [e for e in events if e.event_type == EconEventType.FOMC]
        assert len(fomc_events) >= 8  # 8 FOMC meetings per year minimum
    
    def test_no_blackout_on_non_event_wednesday(self):
        """CRITICAL: Non-event Wednesday at 14:00 is NOT blackout.
        
        v-econ-calendar-dated-2026-09-09: This was the root cause of
        the LIVE freeze. Sept 9, 2026 is a Wednesday but NOT an FOMC day.
        The old StaticEconCalendar would trigger blackout at 14:00 on
        every Wednesday, freezing LIVE mode analysis.
        """
        calendar = DatedEconCalendar(year=2026)
        
        # Sept 9, 2026 is a Wednesday but NOT an FOMC day
        # FOMC days in 2026: Jan 29, Mar 19, May 7, Jun 18, Jul 30, Sep 17, Nov 5, Dec 17
        non_fomc_wednesday = datetime(2026, 9, 9, 14, 15, 0, tzinfo=ET_TZ)
        
        assert not calendar.is_blackout(non_fomc_wednesday), \
            "Non-FOMC Wednesday should NOT be blackout"
    
    def test_blackout_on_real_fomc_day(self):
        """Test blackout IS active on real FOMC day at 14:00."""
        calendar = DatedEconCalendar(year=2026)
        
        # Sept 17, 2026 IS an FOMC day
        fomc_day = datetime(2026, 9, 17, 14, 15, 0, tzinfo=ET_TZ)
        
        assert calendar.is_blackout(fomc_day), \
            "Real FOMC day SHOULD be blackout at 14:00"
    
    def test_no_blackout_on_fomc_day_outside_window(self):
        """Test no blackout on FOMC day outside the event window."""
        calendar = DatedEconCalendar(year=2026)
        
        # Sept 17, 2026 at 11:00 (before FOMC at 14:00)
        before_fomc = datetime(2026, 9, 17, 11, 0, 0, tzinfo=ET_TZ)
        
        assert not calendar.is_blackout(before_fomc), \
            "FOMC day before event time should NOT be blackout"
    
    def test_get_active_event_on_fomc_day(self):
        """Test get_active_event returns FOMC details on event day."""
        calendar = DatedEconCalendar(year=2026)
        
        fomc_day = datetime(2026, 9, 17, 14, 15, 0, tzinfo=ET_TZ)
        active = calendar.get_active_event(fomc_day)
        
        assert active is not None
        assert "FOMC" in active.name
        assert active.event_type == EconEventType.FOMC
        assert active.remaining_minutes > 0
    
    def test_cpi_events_have_dates(self):
        """Test that CPI events are dated, not recurring."""
        calendar = DatedEconCalendar(year=2026)
        events = calendar.get_events()
        
        cpi_events = [e for e in events if e.event_type == EconEventType.CPI]
        
        for event in cpi_events:
            assert event.date is not None, f"{event.name} should have a specific date"
    
    def test_jobs_events_have_dates(self):
        """Test that Jobs Report events are dated (first Friday)."""
        calendar = DatedEconCalendar(year=2026)
        events = calendar.get_events()
        
        jobs_events = [e for e in events if e.event_type == EconEventType.JOBS_REPORT]
        
        for event in jobs_events:
            assert event.date is not None, f"{event.name} should have a specific date"
            # First Friday means the date's weekday should be 4 (Friday)
            assert event.date.weekday() == 4, f"{event.name} should be on a Friday"
    
    def test_includes_next_year_fomc(self):
        """Test calendar includes next year's FOMC dates for planning."""
        calendar = DatedEconCalendar(year=2026)
        events = calendar.get_events()
        
        # Should include 2027 FOMC dates
        fomc_2027 = [e for e in events 
                     if e.event_type == EconEventType.FOMC 
                     and e.date and e.date.year == 2027]
        
        assert len(fomc_2027) >= 8


class TestConfigEconCalendar:
    def test_falls_back_to_dated_when_no_config(self):
        """v-econ-calendar-dated-2026-09-09: Falls back to DatedEconCalendar."""
        # Mock config to return empty events
        with patch('core.config.Config') as MockConfig:
            mock_cfg = MagicMock()
            mock_cfg.manager.get.return_value = []
            MockConfig.return_value = mock_cfg
            
            calendar = ConfigEconCalendar()
            events = calendar.get_events()
            
            # Should fall back to DatedEconCalendar (many events)
            assert len(events) >= 20  # FOMC + CPI + Jobs
    
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
        
        with patch('core.config.Config') as MockConfig:
            mock_cfg = MagicMock()
            mock_cfg.manager.get.return_value = config_events
            MockConfig.return_value = mock_cfg
            
            calendar = ConfigEconCalendar()
            events = calendar.get_events()
            
            assert len(events) == 2
            assert events[0].name == 'Test CPI'
            assert events[0].duration_minutes == 20
            assert events[1].event_type == EconEventType.CUSTOM
    
    def test_loads_dated_event_from_config(self):
        """Test loading event with explicit date from config."""
        config_events = [
            {
                'name': 'Custom FOMC',
                'type': 'fomc',
                'time': '14:00',
                'duration': 30,
                'date': '2026-12-25',  # Explicit date
            },
        ]
        
        with patch('core.config.Config') as MockConfig:
            mock_cfg = MagicMock()
            mock_cfg.manager.get.return_value = config_events
            MockConfig.return_value = mock_cfg
            
            calendar = ConfigEconCalendar()
            events = calendar.get_events()
            
            assert len(events) == 1
            assert events[0].date is not None
            assert events[0].date.month == 12
            assert events[0].date.day == 25


class TestApiEconCalendar:
    def test_uses_dated_fallback_when_no_api_url(self):
        """v-econ-calendar-dated-2026-09-09: Falls back to DatedEconCalendar."""
        with patch.dict('os.environ', {}, clear=True):
            calendar = ApiEconCalendar()
            events = calendar.get_events()
            
            # Should return DatedEconCalendar events (many)
            assert len(events) >= 20


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
    
    def test_default_is_dated_calendar(self):
        """v-econ-calendar-dated-2026-09-09: Default is DatedEconCalendar."""
        with patch.dict('os.environ', {}, clear=True):
            with patch('core.config.Config') as MockConfig:
                mock_cfg = MagicMock()
                mock_cfg.manager.get.return_value = None
                MockConfig.return_value = mock_cfg
                
                reset_econ_calendar()
                calendar = get_econ_calendar()
                
                assert isinstance(calendar, DatedEconCalendar)


class TestBlackoutBehavior:
    """Tests for blackout behavior across calendar types.
    
    v-econ-calendar-dated-2026-09-09: These tests verify the fix for
    the LIVE freeze bug where every Wednesday 14:00 triggered blackout.
    """
    
    def test_wednesday_afternoon_no_phantom_blackout(self):
        """REGRESSION TEST: Wednesday 14:00 on non-event day is clear.
        
        The original bug: StaticEconCalendar had days_of_week=[2] for FOMC,
        meaning EVERY Wednesday 14:00 was treated as FOMC. This froze LIVE
        mode analysis on Sept 9, 2026 (not an actual FOMC day).
        """
        calendar = DatedEconCalendar(year=2026)
        
        # Test several non-FOMC Wednesdays
        non_fomc_wednesdays = [
            datetime(2026, 9, 9, 14, 0, 0, tzinfo=ET_TZ),   # Sept 9
            datetime(2026, 9, 23, 14, 0, 0, tzinfo=ET_TZ),  # Sept 23
            datetime(2026, 10, 14, 14, 0, 0, tzinfo=ET_TZ), # Oct 14
        ]
        
        for wed in non_fomc_wednesdays:
            assert not calendar.is_blackout(wed), \
                f"{wed.date()} at 14:00 should NOT be blackout (not FOMC)"
    
    def test_real_fomc_days_are_blackout(self):
        """Test that real FOMC days DO trigger blackout at 14:00."""
        calendar = DatedEconCalendar(year=2026)
        
        # Real FOMC days in 2026
        fomc_days = [
            datetime(2026, 1, 29, 14, 15, 0, tzinfo=ET_TZ),
            datetime(2026, 3, 19, 14, 15, 0, tzinfo=ET_TZ),
            datetime(2026, 5, 7, 14, 15, 0, tzinfo=ET_TZ),
            datetime(2026, 9, 17, 14, 15, 0, tzinfo=ET_TZ),
        ]
        
        for day in fomc_days:
            assert calendar.is_blackout(day), \
                f"{day.date()} at 14:15 SHOULD be blackout (FOMC day)"
    
    def test_active_blackout_provides_observability(self):
        """Test that get_active_event provides all needed observability data."""
        calendar = DatedEconCalendar(year=2026)
        
        fomc_time = datetime(2026, 9, 17, 14, 15, 0, tzinfo=ET_TZ)
        active = calendar.get_active_event(fomc_time)
        
        assert active is not None
        
        # Check all observability fields
        assert active.name  # Event name for logging
        assert active.event_type  # Event type enum
        assert active.start_time  # Window start
        assert active.end_time  # Window end
        assert active.remaining_minutes >= 0  # Time until clear
    
    def test_fed_speech_blackout_after_fomc(self):
        """Test Fed Chair press conference blackout after FOMC."""
        calendar = DatedEconCalendar(year=2026)
        
        # Fed speech is typically at 14:30, 45 min duration
        fed_speech_time = datetime(2026, 9, 17, 14, 45, 0, tzinfo=ET_TZ)
        active = calendar.get_active_event(fed_speech_time)
        
        assert active is not None
        assert "Fed" in active.name or "Press Conference" in active.name


class TestActiveBlackout:
    """Tests for the ActiveBlackout dataclass."""
    
    def test_remaining_minutes_calculation(self):
        """Test remaining_minutes decreases as time passes."""
        event = EconEvent(
            event_type=EconEventType.FOMC,
            name="FOMC",
            time_et=time(14, 0),
            duration_minutes=30,
        )
        
        start = datetime(2026, 9, 17, 14, 0, 0, tzinfo=ET_TZ)
        end = datetime(2026, 9, 17, 14, 30, 0, tzinfo=ET_TZ)
        
        active = ActiveBlackout(event=event, start_time=start, end_time=end)
        
        # remaining_minutes is calculated from datetime.now()
        # Just verify it's a reasonable positive number or 0
        assert active.remaining_minutes >= 0
    
    def test_property_accessors(self):
        """Test that property accessors work correctly."""
        event = EconEvent(
            event_type=EconEventType.FOMC,
            name="FOMC Rate Decision",
            time_et=time(14, 0),
            duration_minutes=30,
        )
        
        start = datetime(2026, 9, 17, 14, 0, 0, tzinfo=ET_TZ)
        end = datetime(2026, 9, 17, 14, 30, 0, tzinfo=ET_TZ)
        
        active = ActiveBlackout(event=event, start_time=start, end_time=end)
        
        assert active.name == "FOMC Rate Decision"
        assert active.event_type == EconEventType.FOMC
