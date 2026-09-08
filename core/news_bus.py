"""NewsBus: centralized news sentiment store for the trading engine.

v-newsbus-2026-09-08: single source of truth for news sentiment.
Strategies read from this bus; none fetch independently. The news_loop
is the sole writer, refreshing from FreeNewsAggregator on a ~20s cadence.

Thread-safe and async-safe via asyncio.Lock. Items are evicted after TTL
(default 4 hours). High-impact alerts are tracked separately for fast
wake-on-news triggering in the analysis loop.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable, Dict, List, Optional, Set

logger = logging.getLogger("TradingBot")


class NewsImpactLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class ScoredNewsItem:
    """A news item with sentiment scoring and metadata."""
    id: str
    symbol: str
    headline: str
    summary: str
    source: str
    source_tier: int  # 1=primary (Yahoo), 2=secondary (Google), 3=scrape
    url: str
    published_time: datetime
    fetched_at: datetime
    sentiment_score: float  # VADER compound, -1 to +1
    sentiment_confidence: float
    impact: NewsImpactLevel = NewsImpactLevel.MEDIUM

    def age_sec(self) -> float:
        """Seconds since publication."""
        return (datetime.now() - self.published_time).total_seconds()

    def fetch_age_sec(self) -> float:
        """Seconds since we fetched this item."""
        return (datetime.now() - self.fetched_at).total_seconds()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "headline": self.headline,
            "summary": self.summary,
            "source": self.source,
            "source_tier": self.source_tier,
            "url": self.url,
            "published_time": self.published_time.isoformat(),
            "fetched_at": self.fetched_at.isoformat(),
            "sentiment_score": self.sentiment_score,
            "sentiment_confidence": self.sentiment_confidence,
            "impact": self.impact.value,
            "age_sec": self.age_sec(),
        }


@dataclass
class NewsBusStats:
    """Observability metrics for the NewsBus."""
    total_items: int = 0
    symbols_tracked: int = 0
    high_impact_items: int = 0
    last_refresh: Optional[datetime] = None
    refresh_count: int = 0
    items_evicted: int = 0
    fetch_errors: int = 0
    wake_events_fired: int = 0


class NewsBus:
    """Thread-safe, async-safe news sentiment store.
    
    The news_loop writes to this bus; strategies read from it.
    High-impact items trigger wake events for the analysis loop.
    """

    HIGH_IMPACT_KEYWORDS = frozenset([
        "earnings", "beat", "miss", "sec", "investigation",
        "guidance", "upgrade", "downgrade", "fda", "approval",
        "merger", "acquisition", "buyout", "lawsuit", "bankruptcy",
        "ceo", "cfo", "resign", "fired", "outlook", "forecast",
    ])

    SOURCE_TIERS = {
        "Yahoo Finance": 1,
        "Yahoo Finance RSS": 1,
        "Google News": 2,
        "MarketWatch": 3,
    }

    def __init__(
        self,
        ttl_sec: float = 14400.0,  # 4 hours
        max_items_per_symbol: int = 50,
        on_high_impact: Optional[Callable[[str, ScoredNewsItem], None]] = None,
    ):
        self._ttl_sec = ttl_sec
        self._max_items_per_symbol = max_items_per_symbol
        self._on_high_impact = on_high_impact
        
        self._lock = asyncio.Lock()
        self._items: Dict[str, List[ScoredNewsItem]] = defaultdict(list)
        self._high_impact_events: Dict[str, List[ScoredNewsItem]] = defaultdict(list)
        self._stats = NewsBusStats()
        self._wake_event = asyncio.Event()

    async def publish(self, items: List[ScoredNewsItem]) -> int:
        """Publish scored news items to the bus. Returns count of new items added."""
        if not items:
            return 0

        new_count = 0
        high_impact_new = []

        async with self._lock:
            for item in items:
                symbol = item.symbol.upper()
                existing_ids = {i.id for i in self._items[symbol]}
                
                if item.id in existing_ids:
                    continue

                self._items[symbol].append(item)
                new_count += 1

                if item.impact == NewsImpactLevel.HIGH:
                    self._high_impact_events[symbol].append(item)
                    high_impact_new.append(item)
                    self._stats.high_impact_items += 1

            self._stats.total_items += new_count
            self._stats.symbols_tracked = len(self._items)
            self._stats.last_refresh = datetime.now()
            self._stats.refresh_count += 1

            self._enforce_limits()
            self._evict_stale()

        if high_impact_new:
            self._wake_event.set()
            for item in high_impact_new:
                if self._on_high_impact:
                    try:
                        self._on_high_impact(item.symbol, item)
                    except Exception as e:
                        logger.debug(f"on_high_impact callback error: {e}")
                self._stats.wake_events_fired += 1
                logger.info(
                    "news_bus high_impact symbol=%s headline=%s sentiment=%.3f source=%s age_sec=%.0f",
                    item.symbol, item.headline[:60], item.sentiment_score,
                    item.source, item.age_sec(),
                )

        return new_count

    async def get_items(
        self,
        symbol: str,
        max_age_sec: Optional[float] = None,
        min_sentiment: Optional[float] = None,
    ) -> List[ScoredNewsItem]:
        """Get news items for a symbol, optionally filtered by age and sentiment."""
        symbol = symbol.upper()
        max_age = max_age_sec if max_age_sec is not None else self._ttl_sec

        async with self._lock:
            items = self._items.get(symbol, [])
            now = datetime.now()
            
            result = []
            for item in items:
                age = (now - item.published_time).total_seconds()
                if age > max_age:
                    continue
                if min_sentiment is not None and abs(item.sentiment_score) < min_sentiment:
                    continue
                result.append(item)

            result.sort(key=lambda x: x.published_time, reverse=True)
            return result

    async def get_freshest(self, symbol: str) -> Optional[ScoredNewsItem]:
        """Get the most recently published item for a symbol."""
        items = await self.get_items(symbol)
        return items[0] if items else None

    def has_high_impact(self, symbol: str, since_sec: float = 300) -> bool:
        """Check if there's a high-impact item for symbol in the last N seconds.
        
        Synchronous for fast checks in the analysis loop wake logic.
        """
        symbol = symbol.upper()
        cutoff = datetime.now() - timedelta(seconds=since_sec)
        
        events = self._high_impact_events.get(symbol, [])
        return any(e.fetched_at >= cutoff for e in events)

    def get_symbols_with_high_impact(self, since_sec: float = 300) -> Set[str]:
        """Get all symbols with high-impact news in the last N seconds.
        
        Synchronous for fast wake checks.
        """
        cutoff = datetime.now() - timedelta(seconds=since_sec)
        result = set()
        
        for symbol, events in self._high_impact_events.items():
            if any(e.fetched_at >= cutoff for e in events):
                result.add(symbol)
        
        return result

    async def get_aggregate_sentiment(self, symbol: str, max_age_sec: float = 3600) -> dict:
        """Get aggregated sentiment stats for a symbol."""
        items = await self.get_items(symbol, max_age_sec=max_age_sec)
        
        if not items:
            return {
                "symbol": symbol,
                "avg_sentiment": 0.0,
                "article_count": 0,
                "high_impact_count": 0,
                "freshest_age_sec": None,
                "sources": [],
            }

        scores = [i.sentiment_score for i in items]
        high_impact = [i for i in items if i.impact == NewsImpactLevel.HIGH]
        sources = list(set(i.source for i in items))
        freshest = min(items, key=lambda x: x.published_time)

        return {
            "symbol": symbol,
            "avg_sentiment": sum(scores) / len(scores),
            "article_count": len(items),
            "high_impact_count": len(high_impact),
            "freshest_age_sec": freshest.age_sec(),
            "sources": sources,
        }

    def get_wake_event(self) -> asyncio.Event:
        """Get the wake event for analysis loop early-wake triggering."""
        return self._wake_event

    def clear_wake_event(self) -> None:
        """Clear the wake event after analysis loop has processed it."""
        self._wake_event.clear()

    def get_stats(self) -> NewsBusStats:
        """Get observability stats."""
        return self._stats

    def record_fetch_error(self) -> None:
        """Record a fetch error for observability."""
        self._stats.fetch_errors += 1

    def _enforce_limits(self) -> None:
        """Enforce max items per symbol (called under lock)."""
        for symbol in self._items:
            items = self._items[symbol]
            if len(items) > self._max_items_per_symbol:
                items.sort(key=lambda x: x.published_time, reverse=True)
                evicted = len(items) - self._max_items_per_symbol
                self._items[symbol] = items[:self._max_items_per_symbol]
                self._stats.items_evicted += evicted

    def _evict_stale(self) -> None:
        """Evict items older than TTL (called under lock)."""
        cutoff = datetime.now() - timedelta(seconds=self._ttl_sec)
        
        for symbol in list(self._items.keys()):
            before = len(self._items[symbol])
            self._items[symbol] = [
                i for i in self._items[symbol]
                if i.published_time >= cutoff
            ]
            evicted = before - len(self._items[symbol])
            self._stats.items_evicted += evicted

            if not self._items[symbol]:
                del self._items[symbol]

        for symbol in list(self._high_impact_events.keys()):
            self._high_impact_events[symbol] = [
                e for e in self._high_impact_events[symbol]
                if e.fetched_at >= cutoff
            ]
            if not self._high_impact_events[symbol]:
                del self._high_impact_events[symbol]

        self._stats.symbols_tracked = len(self._items)

    @classmethod
    def detect_impact(cls, headline: str, summary: str = "") -> NewsImpactLevel:
        """Detect impact level from headline/summary keywords."""
        text = f"{headline} {summary}".lower()
        for keyword in cls.HIGH_IMPACT_KEYWORDS:
            if keyword in text:
                return NewsImpactLevel.HIGH
        return NewsImpactLevel.MEDIUM

    @classmethod
    def get_source_tier(cls, source: str) -> int:
        """Get the tier for a source (1=primary, 2=secondary, 3=scrape)."""
        for key, tier in cls.SOURCE_TIERS.items():
            if key.lower() in source.lower():
                return tier
        return 3

    @classmethod
    def make_item_id(cls, url: str) -> str:
        """Generate a stable ID from a URL."""
        return hashlib.md5(url.encode()).hexdigest()[:12]


_bus_singleton: Optional[NewsBus] = None


def get_news_bus(
    ttl_sec: float = 14400.0,
    max_items_per_symbol: int = 50,
    on_high_impact: Optional[Callable[[str, ScoredNewsItem], None]] = None,
) -> NewsBus:
    """Get or create the singleton NewsBus instance."""
    global _bus_singleton
    if _bus_singleton is None:
        _bus_singleton = NewsBus(
            ttl_sec=ttl_sec,
            max_items_per_symbol=max_items_per_symbol,
            on_high_impact=on_high_impact,
        )
    return _bus_singleton


def reset_news_bus() -> None:
    """Reset the singleton (for testing)."""
    global _bus_singleton
    _bus_singleton = None
