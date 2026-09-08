"""Tests for NewsBus and news_loop functionality.

v-newsbus-2026-09-08: Tests covering:
  - ScoredNewsItem creation and age computation
  - NewsBus publish/get operations
  - High-impact detection and wake events
  - TTL eviction
  - Thread-safety via async operations
"""
import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from core.news_bus import (
    NewsBus,
    ScoredNewsItem,
    NewsImpactLevel,
    NewsBusStats,
    get_news_bus,
    reset_news_bus,
)


@pytest.fixture
def news_bus():
    """Create a fresh NewsBus for each test."""
    reset_news_bus()
    return NewsBus(ttl_sec=3600, max_items_per_symbol=10)


@pytest.fixture
def sample_item():
    """Create a sample ScoredNewsItem."""
    return ScoredNewsItem(
        id="abc123",
        symbol="AAPL",
        headline="Apple Reports Record Earnings",
        summary="Apple Inc. reported record quarterly earnings today.",
        source="Yahoo Finance",
        source_tier=1,
        url="https://finance.yahoo.com/news/apple-earnings",
        published_time=datetime.now() - timedelta(minutes=30),
        fetched_at=datetime.now(),
        sentiment_score=0.75,
        sentiment_confidence=0.75,
        impact=NewsImpactLevel.HIGH,
    )


class TestScoredNewsItem:
    def test_age_sec(self, sample_item):
        """Test age_sec computation."""
        age = sample_item.age_sec()
        assert 1750 < age < 1850  # ~30 minutes in seconds

    def test_fetch_age_sec(self, sample_item):
        """Test fetch_age_sec computation."""
        age = sample_item.fetch_age_sec()
        assert age < 5  # Just created

    def test_to_dict(self, sample_item):
        """Test serialization to dict."""
        d = sample_item.to_dict()
        assert d["id"] == "abc123"
        assert d["symbol"] == "AAPL"
        assert d["sentiment_score"] == 0.75
        assert d["impact"] == "high"
        assert "age_sec" in d


class TestNewsBus:
    @pytest.mark.asyncio
    async def test_publish_and_get(self, news_bus, sample_item):
        """Test basic publish and retrieve."""
        count = await news_bus.publish([sample_item])
        assert count == 1

        items = await news_bus.get_items("AAPL")
        assert len(items) == 1
        assert items[0].headline == sample_item.headline

    @pytest.mark.asyncio
    async def test_duplicate_rejection(self, news_bus, sample_item):
        """Test that duplicate items (same ID) are rejected."""
        await news_bus.publish([sample_item])
        count = await news_bus.publish([sample_item])
        assert count == 0

        items = await news_bus.get_items("AAPL")
        assert len(items) == 1

    @pytest.mark.asyncio
    async def test_case_insensitive_symbol(self, news_bus, sample_item):
        """Test that symbols are case-insensitive."""
        await news_bus.publish([sample_item])
        
        items_upper = await news_bus.get_items("AAPL")
        items_lower = await news_bus.get_items("aapl")
        items_mixed = await news_bus.get_items("AaPl")
        
        assert len(items_upper) == 1
        assert len(items_lower) == 1
        assert len(items_mixed) == 1

    @pytest.mark.asyncio
    async def test_get_freshest(self, news_bus):
        """Test get_freshest returns most recent item."""
        old_item = ScoredNewsItem(
            id="old1",
            symbol="AAPL",
            headline="Old News",
            summary="",
            source="Test",
            source_tier=2,
            url="http://test.com/old",
            published_time=datetime.now() - timedelta(hours=2),
            fetched_at=datetime.now() - timedelta(hours=2),
            sentiment_score=0.1,
            sentiment_confidence=0.1,
            impact=NewsImpactLevel.LOW,
        )
        new_item = ScoredNewsItem(
            id="new1",
            symbol="AAPL",
            headline="New News",
            summary="",
            source="Test",
            source_tier=1,
            url="http://test.com/new",
            published_time=datetime.now() - timedelta(minutes=5),
            fetched_at=datetime.now(),
            sentiment_score=0.9,
            sentiment_confidence=0.9,
            impact=NewsImpactLevel.HIGH,
        )
        
        await news_bus.publish([old_item, new_item])
        freshest = await news_bus.get_freshest("AAPL")
        
        assert freshest is not None
        assert freshest.id == "new1"

    @pytest.mark.asyncio
    async def test_max_age_filter(self):
        """Test that max_age_sec filters old items."""
        # Use a bus with large TTL to not evict the old item
        bus = NewsBus(ttl_sec=86400, max_items_per_symbol=10)
        
        old_item = ScoredNewsItem(
            id="old1",
            symbol="AAPL",
            headline="Old News",
            summary="",
            source="Test",
            source_tier=2,
            url="http://test.com/old",
            published_time=datetime.now() - timedelta(hours=5),
            fetched_at=datetime.now(),
            sentiment_score=0.1,
            sentiment_confidence=0.1,
            impact=NewsImpactLevel.LOW,
        )
        
        await bus.publish([old_item])
        
        # Should find with large max_age
        items_large = await bus.get_items("AAPL", max_age_sec=86400)
        assert len(items_large) == 1
        
        # Should not find with small max_age
        items_small = await bus.get_items("AAPL", max_age_sec=3600)
        assert len(items_small) == 0


class TestHighImpact:
    @pytest.mark.asyncio
    async def test_high_impact_wake_event(self, news_bus, sample_item):
        """Test that high-impact items set the wake event."""
        assert not news_bus.get_wake_event().is_set()
        
        await news_bus.publish([sample_item])  # HIGH impact
        
        assert news_bus.get_wake_event().is_set()

    @pytest.mark.asyncio
    async def test_has_high_impact(self, news_bus, sample_item):
        """Test has_high_impact detection."""
        assert not news_bus.has_high_impact("AAPL")
        
        await news_bus.publish([sample_item])
        
        assert news_bus.has_high_impact("AAPL", since_sec=60)

    @pytest.mark.asyncio
    async def test_get_symbols_with_high_impact(self, news_bus, sample_item):
        """Test get_symbols_with_high_impact returns correct set."""
        assert len(news_bus.get_symbols_with_high_impact()) == 0
        
        await news_bus.publish([sample_item])
        
        symbols = news_bus.get_symbols_with_high_impact()
        assert "AAPL" in symbols

    @pytest.mark.asyncio
    async def test_on_high_impact_callback(self, news_bus, sample_item):
        """Test that on_high_impact callback is called."""
        callback_called = []
        
        def callback(symbol, item):
            callback_called.append((symbol, item))
        
        bus = NewsBus(on_high_impact=callback)
        await bus.publish([sample_item])
        
        assert len(callback_called) == 1
        assert callback_called[0][0] == "AAPL"


class TestImpactDetection:
    @pytest.mark.parametrize("headline,expected", [
        ("Apple beats earnings expectations", NewsImpactLevel.HIGH),
        ("Apple misses revenue target", NewsImpactLevel.HIGH),
        ("SEC investigates company", NewsImpactLevel.HIGH),
        ("Company announces merger", NewsImpactLevel.HIGH),
        ("CEO resigns from position", NewsImpactLevel.HIGH),
        ("Regular market update", NewsImpactLevel.MEDIUM),
    ])
    def test_detect_impact(self, headline, expected):
        """Test impact detection from headlines."""
        impact = NewsBus.detect_impact(headline)
        assert impact == expected


class TestSourceTier:
    @pytest.mark.parametrize("source,expected_tier", [
        ("Yahoo Finance", 1),
        ("Yahoo Finance RSS", 1),
        ("Google News", 2),
        ("MarketWatch", 3),
        ("Unknown Source", 3),
    ])
    def test_get_source_tier(self, source, expected_tier):
        """Test source tier classification."""
        tier = NewsBus.get_source_tier(source)
        assert tier == expected_tier


class TestStats:
    @pytest.mark.asyncio
    async def test_stats_update(self, news_bus, sample_item):
        """Test that stats are updated correctly."""
        stats_before = news_bus.get_stats()
        assert stats_before.total_items == 0
        
        await news_bus.publish([sample_item])
        
        stats_after = news_bus.get_stats()
        assert stats_after.total_items == 1
        assert stats_after.symbols_tracked == 1
        assert stats_after.high_impact_items == 1
        assert stats_after.refresh_count == 1

    @pytest.mark.asyncio
    async def test_eviction_stats(self, news_bus):
        """Test that eviction stats are tracked."""
        # Create items that will be evicted due to max_items_per_symbol
        items = []
        for i in range(15):
            items.append(ScoredNewsItem(
                id=f"item{i}",
                symbol="AAPL",
                headline=f"News {i}",
                summary="",
                source="Test",
                source_tier=2,
                url=f"http://test.com/{i}",
                published_time=datetime.now() - timedelta(minutes=i),
                fetched_at=datetime.now(),
                sentiment_score=0.5,
                sentiment_confidence=0.5,
                impact=NewsImpactLevel.MEDIUM,
            ))
        
        await news_bus.publish(items)
        
        stats = news_bus.get_stats()
        assert stats.items_evicted == 5  # 15 - 10 (max_items_per_symbol)


class TestSingleton:
    def test_get_news_bus_singleton(self):
        """Test that get_news_bus returns the same instance."""
        reset_news_bus()
        bus1 = get_news_bus()
        bus2 = get_news_bus()
        assert bus1 is bus2

    def test_reset_news_bus(self):
        """Test that reset_news_bus creates a new instance."""
        bus1 = get_news_bus()
        reset_news_bus()
        bus2 = get_news_bus()
        assert bus1 is not bus2
