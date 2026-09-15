"""Tests for core data models."""
import pytest
from datetime import datetime, timedelta, timezone

from core.models import (
    TradingMode, CommentaryType, SignalType, NewsImpact,
    TradingSignal, Position, MarketData, NewsItem
)


class TestEnums:
    def test_trading_modes(self):
        assert TradingMode.PAPER.value == "paper"
        assert TradingMode.LIVE.value == "live"

    def test_signal_types(self):
        assert SignalType.BUY.value == 1
        assert SignalType.SELL.value == -1
        assert SignalType.HOLD.value == 0

    def test_commentary_types_all_unique(self):
        values = [ct.value for ct in CommentaryType]
        assert len(values) == len(set(values))


class TestTradingSignal:
    def test_create_buy_signal(self):
        signal = TradingSignal(
            symbol='AAPL',
            signal_type=SignalType.BUY,
            strength=0.8,
            entry_price=150.0,
            stop_loss=145.0,
            take_profit=160.0,
            position_size=10,
            reasoning={'strategy': 'momentum'},
            confidence=0.75
        )
        assert signal.symbol == 'AAPL'
        assert signal.signal_type == SignalType.BUY
        assert signal.position_size == 10
        assert isinstance(signal.timestamp, datetime)

    def test_signal_has_timestamp_default(self):
        signal = TradingSignal(
            symbol='MSFT', signal_type=SignalType.SELL, strength=0.5,
            entry_price=300.0, stop_loss=310.0, take_profit=280.0,
            position_size=5, reasoning={}, confidence=0.6
        )
        assert signal.timestamp is not None


class TestPosition:
    def test_create_long_position(self):
        pos = Position(
            symbol='GOOGL', entry_price=140.0, current_price=145.0,
            quantity=20, side='long', stop_loss=135.0, take_profit=155.0,
            entry_time=datetime.now()
        )
        assert pos.unrealized_pnl == 0  # Default
        assert pos.is_long_term is False

    def test_position_with_pnl(self):
        pos = Position(
            symbol='TSLA', entry_price=200.0, current_price=210.0,
            quantity=10, side='long', stop_loss=190.0, take_profit=220.0,
            entry_time=datetime.now(), unrealized_pnl=100.0
        )
        assert pos.unrealized_pnl == 100.0


class TestMarketData:
    def test_create_market_data(self):
        md = MarketData(
            symbol='SPY', timestamp=datetime.now(),
            open=450.0, high=455.0, low=448.0, close=452.0,
            volume=1000000, timeframe='1min'
        )
        assert md.indicators == {}  # Default empty dict

    def test_market_data_with_indicators(self):
        md = MarketData(
            symbol='SPY', timestamp=datetime.now(),
            open=450.0, high=455.0, low=448.0, close=452.0,
            volume=1000000, timeframe='5min',
            indicators={'rsi': 65.0, 'macd': 0.5}
        )
        assert md.indicators['rsi'] == 65.0


class TestNewsItem:
    """Tests for NewsItem including timezone-aware age_hours().
    
    v-widget-tz-fix-2026-09-15: Ensures age_hours() handles mixed
    naive/aware datetimes without TypeError.
    """
    
    def test_age_hours_with_utc_aware_timestamp(self):
        """age_hours should work with UTC-aware published_time."""
        two_hours_ago = datetime.now(timezone.utc) - timedelta(hours=2)
        item = NewsItem(
            id="test-1",
            symbol="AAPL",
            headline="Test headline",
            summary="Test summary",
            source="Test",
            url="https://example.com",
            published_time=two_hours_ago,
        )
        age = item.age_hours()
        assert 1.9 < age < 2.1
    
    def test_age_hours_with_naive_timestamp(self):
        """age_hours should work with naive published_time (assumed UTC)."""
        two_hours_ago = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=2)
        item = NewsItem(
            id="test-2",
            symbol="TSLA",
            headline="Test headline",
            summary="Test summary",
            source="Test",
            url="https://example.com",
            published_time=two_hours_ago,
        )
        age = item.age_hours()
        assert 1.9 < age < 2.1
    
    def test_age_hours_no_typeerror_mixed_tz(self):
        """age_hours must not raise TypeError on offset-aware vs naive comparison.
        
        This was the root cause of the widget path bug:
        'can't subtract offset-naive and offset-aware datetimes'
        """
        aware_ts = datetime.now(timezone.utc) - timedelta(hours=1)
        item = NewsItem(
            id="test-3",
            symbol="NVDA",
            headline="Aware timestamp test",
            summary="",
            source="Test",
            url="",
            published_time=aware_ts,
        )
        try:
            age = item.age_hours()
            assert age > 0
        except TypeError as e:
            pytest.fail(f"age_hours raised TypeError: {e}")
