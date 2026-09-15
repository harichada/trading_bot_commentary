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
    ensure_utc_aware,
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


class TestNewsGate:
    """v-newsbus-gates-2026-09-09: tests for the news thesis gate system."""
    
    @pytest.fixture
    def gate_bus(self):
        """Create a fresh NewsBus for gate tests with long TTL."""
        reset_news_bus()
        return NewsBus(ttl_sec=86400, max_items_per_symbol=50)
    
    @pytest.mark.asyncio
    async def test_gate_veto_no_corroboration(self, gate_bus):
        """Test that empty bus returns VETO_NO_CORROBORATION."""
        from core.news_bus import NewsGateAction
        
        result = await gate_bus.evaluate_gate("AAPL", max_age_sec=1800)
        
        assert result.action == NewsGateAction.VETO_NO_CORROBORATION
        assert result.size_multiplier == 0.0
        assert result.corroboration_n == 0
        assert result.is_veto()
        assert gate_bus.get_stats().gate_veto_no_corroboration == 1
    
    @pytest.mark.asyncio
    async def test_gate_veto_stale(self, gate_bus):
        """Test that stale news returns VETO_STALE (not NO_CORROBORATION)."""
        from core.news_bus import NewsGateAction
        
        # Create an old item (2 hours old, within TTL but outside freshness window)
        old_item = ScoredNewsItem(
            id="old1",
            symbol="AAPL",
            headline="Old News",
            summary="",
            source="Yahoo Finance",
            source_tier=1,
            url="http://test.com/old",
            published_time=datetime.now() - timedelta(hours=2),
            fetched_at=datetime.now() - timedelta(hours=2),
            sentiment_score=0.5,
            sentiment_confidence=0.5,
            impact=NewsImpactLevel.MEDIUM,
        )
        await gate_bus.publish([old_item])
        
        # Gate with 30-min max_age should return VETO_STALE (not NO_CORROBORATION)
        # because we HAVE news, it's just too old
        result = await gate_bus.evaluate_gate("AAPL", max_age_sec=1800)
        
        assert result.action == NewsGateAction.VETO_STALE
        assert result.is_veto()
        assert result.news_age_sec is not None
        assert result.news_age_sec > 1800  # Older than threshold
        assert gate_bus.get_stats().gate_veto_stale == 1
    
    @pytest.mark.asyncio
    async def test_gate_veto_low_tier(self, gate_bus):
        """Test that scrape-only sources are vetoed when tier floor is 2."""
        from core.news_bus import NewsGateAction
        
        # Create a scrape-tier item (tier 3)
        scrape_item = ScoredNewsItem(
            id="scrape1",
            symbol="AAPL",
            headline="Scrape News",
            summary="",
            source="MarketWatch",
            source_tier=3,
            url="http://test.com/scrape",
            published_time=datetime.now() - timedelta(minutes=5),
            fetched_at=datetime.now(),
            sentiment_score=0.5,
            sentiment_confidence=0.5,
            impact=NewsImpactLevel.MEDIUM,
        )
        await gate_bus.publish([scrape_item])
        
        # Gate with tier floor 2 should veto tier-3 only sources
        result = await gate_bus.evaluate_gate(
            "AAPL", max_age_sec=1800, source_tier_floor=2
        )
        
        assert result.action == NewsGateAction.VETO_LOW_TIER
        assert result.size_multiplier == 0.0
        assert result.source_tier_min == 3
        assert result.is_veto()
        assert gate_bus.get_stats().gate_veto_low_tier == 1
    
    @pytest.mark.asyncio
    async def test_gate_reduced_size_single_source(self, gate_bus):
        """Test that single-source fresh news gives REDUCED_SIZE (0.5×)."""
        from core.news_bus import NewsGateAction
        
        # Create a single fresh tier-1 item
        fresh_item = ScoredNewsItem(
            id="fresh1",
            symbol="AAPL",
            headline="Fresh News",
            summary="",
            source="Yahoo Finance",
            source_tier=1,
            url="http://test.com/fresh",
            published_time=datetime.now() - timedelta(minutes=5),
            fetched_at=datetime.now(),
            sentiment_score=0.5,
            sentiment_confidence=0.5,
            impact=NewsImpactLevel.MEDIUM,
        )
        await gate_bus.publish([fresh_item])
        
        result = await gate_bus.evaluate_gate(
            "AAPL", max_age_sec=1800, source_tier_floor=2, min_corroboration=2
        )
        
        assert result.action == NewsGateAction.REDUCED_SIZE
        assert result.size_multiplier == 0.5
        assert result.corroboration_n == 1
        assert not result.is_veto()
        assert gate_bus.get_stats().gate_size_reduced == 1
    
    @pytest.mark.asyncio
    async def test_gate_full_size_multi_source(self, gate_bus):
        """Test that multi-source fresh news gives FULL_SIZE (1.0×)."""
        from core.news_bus import NewsGateAction
        
        # Create items from two different sources
        item1 = ScoredNewsItem(
            id="multi1",
            symbol="AAPL",
            headline="News from Yahoo",
            summary="",
            source="Yahoo Finance",
            source_tier=1,
            url="http://test.com/yahoo",
            published_time=datetime.now() - timedelta(minutes=5),
            fetched_at=datetime.now(),
            sentiment_score=0.5,
            sentiment_confidence=0.5,
            impact=NewsImpactLevel.MEDIUM,
        )
        item2 = ScoredNewsItem(
            id="multi2",
            symbol="AAPL",
            headline="News from Google",
            summary="",
            source="Google News",
            source_tier=2,
            url="http://test.com/google",
            published_time=datetime.now() - timedelta(minutes=3),
            fetched_at=datetime.now(),
            sentiment_score=0.6,
            sentiment_confidence=0.6,
            impact=NewsImpactLevel.MEDIUM,
        )
        await gate_bus.publish([item1, item2])
        
        result = await gate_bus.evaluate_gate(
            "AAPL", max_age_sec=1800, source_tier_floor=2, min_corroboration=2
        )
        
        assert result.action == NewsGateAction.FULL_SIZE
        assert result.size_multiplier == 1.0
        assert result.corroboration_n == 2
        assert result.source_tier_min == 1
        assert not result.is_veto()
        assert gate_bus.get_stats().gate_pass == 1
    
    @pytest.mark.asyncio
    async def test_gate_full_size_high_impact_single(self, gate_bus):
        """Test that high-impact single source gives FULL_SIZE even below min_corroboration."""
        from core.news_bus import NewsGateAction
        
        # Create a single HIGH impact item
        high_impact_item = ScoredNewsItem(
            id="high1",
            symbol="AAPL",
            headline="Apple Beats Earnings Expectations",
            summary="Record quarterly results.",
            source="Yahoo Finance",
            source_tier=1,
            url="http://test.com/earnings",
            published_time=datetime.now() - timedelta(minutes=5),
            fetched_at=datetime.now(),
            sentiment_score=0.8,
            sentiment_confidence=0.8,
            impact=NewsImpactLevel.HIGH,
        )
        await gate_bus.publish([high_impact_item])
        
        # Use min_corroboration=2 to prove high-impact bypasses corroboration requirement
        result = await gate_bus.evaluate_gate(
            "AAPL", max_age_sec=1800, source_tier_floor=2, min_corroboration=2
        )
        
        assert result.action == NewsGateAction.FULL_SIZE
        assert result.size_multiplier == 1.0
        assert result.corroboration_n == 1  # Only one source
        assert "high_impact" in result.reason
    
    @pytest.mark.asyncio
    async def test_gate_custom_single_source_multiplier(self, gate_bus):
        """Test that single_source_multiplier is passed through from config."""
        from core.news_bus import NewsGateAction
        
        # Create a single fresh non-high-impact item
        fresh_item = ScoredNewsItem(
            id="custom1",
            symbol="AAPL",
            headline="Regular News",
            summary="",
            source="Yahoo Finance",
            source_tier=1,
            url="http://test.com/custom",
            published_time=datetime.now() - timedelta(minutes=5),
            fetched_at=datetime.now(),
            sentiment_score=0.5,
            sentiment_confidence=0.5,
            impact=NewsImpactLevel.MEDIUM,
        )
        await gate_bus.publish([fresh_item])
        
        # Use custom multiplier of 0.3 instead of default 0.5
        result = await gate_bus.evaluate_gate(
            "AAPL", max_age_sec=1800, source_tier_floor=2, 
            min_corroboration=2, single_source_multiplier=0.3
        )
        
        assert result.action == NewsGateAction.REDUCED_SIZE
        assert result.size_multiplier == 0.3  # Custom value, not 0.5
    
    @pytest.mark.asyncio
    async def test_gate_stats_accumulate(self, gate_bus):
        """Test that gate stats accumulate correctly."""
        # Trigger different gate outcomes
        await gate_bus.evaluate_gate("EMPTY", max_age_sec=1800)  # veto_no_corroboration
        
        item = ScoredNewsItem(
            id="t1", symbol="TIER3",
            headline="Scrape", summary="", source="MarketWatch", source_tier=3,
            url="http://test.com/1",
            published_time=datetime.now() - timedelta(minutes=5),
            fetched_at=datetime.now(),
            sentiment_score=0.5, sentiment_confidence=0.5,
            impact=NewsImpactLevel.MEDIUM,
        )
        await gate_bus.publish([item])
        await gate_bus.evaluate_gate("TIER3", max_age_sec=1800, source_tier_floor=2)  # veto_low_tier
        
        stats = gate_bus.get_stats()
        assert stats.gate_veto_no_corroboration >= 1
        assert stats.gate_veto_low_tier >= 1


class TestTimezoneAwareness:
    """v-newsbus-tz-fix-2026-09-15: Tests for timezone-aware datetime handling.
    
    RCA: news gate exception 'can't subtract offset-naive and offset-aware datetimes'
    caused fail-closed spam when published_time was UTC-aware but comparison used
    naive datetime.now(). This test class ensures the fix handles all edge cases.
    """
    
    @pytest.fixture
    def tz_bus(self):
        """Create a fresh NewsBus for TZ tests."""
        reset_news_bus()
        return NewsBus(ttl_sec=86400, max_items_per_symbol=50)
    
    @pytest.mark.asyncio
    async def test_get_items_with_naive_published_time(self, tz_bus):
        """Test get_items works when published_time is naive."""
        naive_item = ScoredNewsItem(
            id="naive1",
            symbol="TEST",
            headline="Naive datetime test",
            summary="",
            source="Test",
            source_tier=1,
            url="http://test.com/naive",
            published_time=datetime.now() - timedelta(minutes=10),  # naive
            fetched_at=datetime.now(),  # naive
            sentiment_score=0.5,
            sentiment_confidence=0.5,
            impact=NewsImpactLevel.MEDIUM,
        )
        
        await tz_bus.publish([naive_item])
        items = await tz_bus.get_items("TEST", max_age_sec=3600)
        
        assert len(items) == 1
        assert items[0].id == "naive1"
    
    @pytest.mark.asyncio
    async def test_get_items_with_aware_published_time(self, tz_bus):
        """Test get_items works when published_time is UTC-aware."""
        from datetime import timezone
        
        aware_item = ScoredNewsItem(
            id="aware1",
            symbol="TEST",
            headline="Aware datetime test",
            summary="",
            source="Test",
            source_tier=1,
            url="http://test.com/aware",
            published_time=datetime.now(timezone.utc) - timedelta(minutes=10),  # aware
            fetched_at=datetime.now(timezone.utc),  # aware
            sentiment_score=0.5,
            sentiment_confidence=0.5,
            impact=NewsImpactLevel.MEDIUM,
        )
        
        await tz_bus.publish([aware_item])
        items = await tz_bus.get_items("TEST", max_age_sec=3600)
        
        assert len(items) == 1
        assert items[0].id == "aware1"
    
    @pytest.mark.asyncio
    async def test_get_items_mixed_naive_and_aware(self, tz_bus):
        """Test get_items works with a mix of naive and aware published_time.
        
        This is the exact scenario that caused the original TypeError.
        """
        from datetime import timezone
        
        naive_item = ScoredNewsItem(
            id="mixed_naive",
            symbol="MIX",
            headline="Naive in mix",
            summary="",
            source="Test A",
            source_tier=1,
            url="http://test.com/mixnaive",
            published_time=datetime.now() - timedelta(minutes=5),  # naive
            fetched_at=datetime.now(),
            sentiment_score=0.5,
            sentiment_confidence=0.5,
            impact=NewsImpactLevel.MEDIUM,
        )
        aware_item = ScoredNewsItem(
            id="mixed_aware",
            symbol="MIX",
            headline="Aware in mix",
            summary="",
            source="Test B",
            source_tier=1,
            url="http://test.com/mixaware",
            published_time=datetime.now(timezone.utc) - timedelta(minutes=3),  # aware
            fetched_at=datetime.now(timezone.utc),
            sentiment_score=0.6,
            sentiment_confidence=0.6,
            impact=NewsImpactLevel.MEDIUM,
        )
        
        # This should NOT raise TypeError
        await tz_bus.publish([naive_item, aware_item])
        items = await tz_bus.get_items("MIX", max_age_sec=3600)
        
        assert len(items) == 2
    
    @pytest.mark.asyncio
    async def test_evaluate_gate_with_aware_published_time(self, tz_bus):
        """Test evaluate_gate works with UTC-aware published_time.
        
        This covers the news gate exception path that was failing.
        """
        from datetime import timezone
        from core.news_bus import NewsGateAction
        
        aware_item = ScoredNewsItem(
            id="gate_aware",
            symbol="GATE",
            headline="Gate test with aware time",
            summary="",
            source="Yahoo Finance",
            source_tier=1,
            url="http://test.com/gateaware",
            published_time=datetime.now(timezone.utc) - timedelta(minutes=5),
            fetched_at=datetime.now(timezone.utc),
            sentiment_score=0.5,
            sentiment_confidence=0.5,
            impact=NewsImpactLevel.MEDIUM,
        )
        
        await tz_bus.publish([aware_item])
        
        # This should NOT raise TypeError
        result = await tz_bus.evaluate_gate("GATE", max_age_sec=1800)
        
        assert result.action in (NewsGateAction.FULL_SIZE, NewsGateAction.REDUCED_SIZE)
        assert result.news_age_sec is not None
        assert result.news_age_sec >= 0  # age_sec should never be negative
    
    @pytest.mark.asyncio
    async def test_age_sec_never_negative_with_aware_time(self, tz_bus):
        """Test that age_sec() is never negative, even with aware times."""
        from datetime import timezone
        
        item = ScoredNewsItem(
            id="age_test",
            symbol="AGE",
            headline="Age test",
            summary="",
            source="Test",
            source_tier=1,
            url="http://test.com/age",
            published_time=datetime.now(timezone.utc) - timedelta(minutes=30),
            fetched_at=datetime.now(timezone.utc),
            sentiment_score=0.5,
            sentiment_confidence=0.5,
            impact=NewsImpactLevel.MEDIUM,
        )
        
        age = item.age_sec()
        assert age >= 0
        assert 1750 < age < 1850  # ~30 minutes in seconds
    
    @pytest.mark.asyncio
    async def test_has_high_impact_with_aware_fetched_at(self, tz_bus):
        """Test has_high_impact works with aware fetched_at."""
        from datetime import timezone
        
        aware_high_impact = ScoredNewsItem(
            id="hi_aware",
            symbol="HI",
            headline="High impact aware",
            summary="",
            source="Yahoo Finance",
            source_tier=1,
            url="http://test.com/hiaware",
            published_time=datetime.now(timezone.utc) - timedelta(minutes=1),
            fetched_at=datetime.now(timezone.utc),  # aware
            sentiment_score=0.8,
            sentiment_confidence=0.8,
            impact=NewsImpactLevel.HIGH,
        )
        
        await tz_bus.publish([aware_high_impact])
        
        # This should NOT raise TypeError
        assert tz_bus.has_high_impact("HI", since_sec=300)


class TestEnsureUtcAware:
    """Tests for the ensure_utc_aware helper function."""
    
    def test_naive_becomes_utc_aware(self):
        """Test that naive datetime becomes UTC-aware."""
        from core.news_bus import ensure_utc_aware
        from datetime import timezone
        
        naive_dt = datetime(2026, 9, 15, 12, 0, 0)  # No tzinfo
        result = ensure_utc_aware(naive_dt)
        
        assert result.tzinfo is not None
        assert result.tzinfo == timezone.utc
        assert result.year == 2026
        assert result.hour == 12
    
    def test_utc_aware_stays_utc(self):
        """Test that UTC-aware datetime stays UTC."""
        from core.news_bus import ensure_utc_aware
        from datetime import timezone
        
        utc_dt = datetime(2026, 9, 15, 12, 0, 0, tzinfo=timezone.utc)
        result = ensure_utc_aware(utc_dt)
        
        assert result.tzinfo == timezone.utc
        assert result == utc_dt
    
    def test_other_tz_converts_to_utc(self):
        """Test that other timezone-aware datetime converts to UTC."""
        from core.news_bus import ensure_utc_aware
        from datetime import timezone
        
        # Create a datetime in UTC+5 offset
        plus5 = timezone(timedelta(hours=5))
        other_tz_dt = datetime(2026, 9, 15, 17, 0, 0, tzinfo=plus5)  # 17:00 UTC+5 = 12:00 UTC
        
        result = ensure_utc_aware(other_tz_dt)
        
        assert result.tzinfo == timezone.utc
        assert result.hour == 12  # Converted to UTC
    
    def test_subtraction_works_after_ensure(self):
        """Test that two ensure_utc_aware'd datetimes can be subtracted."""
        from core.news_bus import ensure_utc_aware
        from datetime import timezone
        
        naive1 = datetime(2026, 9, 15, 12, 0, 0)
        aware2 = datetime(2026, 9, 15, 12, 30, 0, tzinfo=timezone.utc)
        
        safe1 = ensure_utc_aware(naive1)
        safe2 = ensure_utc_aware(aware2)
        
        # This should NOT raise TypeError
        diff = (safe2 - safe1).total_seconds()
        
        assert diff == 1800  # 30 minutes in seconds
