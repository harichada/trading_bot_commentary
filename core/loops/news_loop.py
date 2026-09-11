"""v-newsbus-2026-09-08: news_loop refreshes the NewsBus from FreeNewsAggregator.

This loop is the sole writer to NewsBus. It fetches news for all symbols
in the dynamic watchlist on a ~20s cadence, scores them with VADER, and
publishes to the bus. High-impact items trigger wake events that the
analysis loop can use for early entry evaluation.

The news_loop is registered with TaskSupervisor at NORMAL priority.
Failures are logged and swallowed per-iteration; only structural errors
surface to the supervisor.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional

from nltk.sentiment import SentimentIntensityAnalyzer

from core.config import Config
from core.news_bus import (
    NewsBus,
    ScoredNewsItem,
    NewsImpactLevel,
    get_news_bus,
)
from strategies.news_strategy import FreeNewsAggregator

if TYPE_CHECKING:
    from core.engine import TradingEngineWithCommentary

logger = logging.getLogger("TradingBot")


class NewsLoop:
    """Refreshes NewsBus from FreeNewsAggregator on a configured cadence.
    
    The loop iterates through all symbols in the engine's dynamic_watchlist,
    fetches news from Yahoo/Google/MarketWatch, scores with VADER, and
    publishes to the NewsBus. High-impact items are detected and trigger
    wake events for the analysis loop.
    """

    def __init__(
        self,
        engine: "TradingEngineWithCommentary",
        cadence_sec: Optional[float] = None,
        bus: Optional[NewsBus] = None,
    ) -> None:
        self.engine = engine
        self.cadence_sec = cadence_sec if cadence_sec is not None else self._get_cadence()
        self.bus = bus or get_news_bus()
        self.aggregator = FreeNewsAggregator()
        self.analyzer = SentimentIntensityAnalyzer()
        self._last_run: Optional[datetime] = None
        self._symbols_refreshed: int = 0
        self._items_published: int = 0

    def _get_cadence(self) -> float:
        """Get cadence from config or default to 20s."""
        try:
            return float(Config().manager.get("trading.news_loop_sec", 20.0))
        except Exception:
            return 20.0

    async def run(self) -> None:
        """Main loop: refresh news for watchlist symbols on cadence."""
        while getattr(self.engine, "is_running", False):
            try:
                await self._tick()
            except Exception as exc:
                logger.warning("news_loop: tick failed: %s", exc, exc_info=True)
                self.bus.record_fetch_error()
            await asyncio.sleep(self.cadence_sec)

    async def _tick(self) -> None:
        """Single refresh cycle."""
        watchlist = getattr(self.engine, "dynamic_watchlist", None) or []
        if not watchlist:
            logger.debug("news_loop: no watchlist, skipping")
            return

        start = datetime.now()
        total_new = 0
        symbols_processed = 0

        batch_size = 5
        for i in range(0, len(watchlist), batch_size):
            batch = watchlist[i:i + batch_size]
            tasks = [self._fetch_and_score(sym) for sym in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for sym, result in zip(batch, results):
                if isinstance(result, Exception):
                    logger.debug("news_loop: fetch error for %s: %s", sym, result)
                    self.bus.record_fetch_error()
                    continue
                
                if result:
                    new_count = await self.bus.publish(result)
                    total_new += new_count
                    symbols_processed += 1

        self._last_run = datetime.now()
        self._symbols_refreshed = symbols_processed
        self._items_published = total_new

        elapsed_ms = (datetime.now() - start).total_seconds() * 1000
        stats = self.bus.get_stats()

        logger.info(
            "news_loop tick symbols=%d new_items=%d bus_total=%d high_impact=%d elapsed_ms=%.0f",
            symbols_processed, total_new, stats.total_items, stats.high_impact_items, elapsed_ms,
        )

    async def _fetch_and_score(self, symbol: str) -> List[ScoredNewsItem]:
        """Fetch news for a symbol and convert to ScoredNewsItem list."""
        try:
            raw_items = await self.aggregator.fetch_news(symbol, hours=4)
        except Exception as e:
            logger.debug("news_loop: aggregator.fetch_news failed for %s: %s", symbol, e)
            return []

        if not raw_items:
            return []

        scored = []
        now = datetime.now()

        for item in raw_items:
            text = f"{item.headline} {getattr(item, 'summary', '')}"
            vader_scores = self.analyzer.polarity_scores(text)
            sentiment = vader_scores["compound"]
            confidence = abs(sentiment)

            impact = NewsBus.detect_impact(item.headline, getattr(item, "summary", ""))
            source_tier = NewsBus.get_source_tier(getattr(item, "source", "Unknown"))

            scored_item = ScoredNewsItem(
                id=NewsBus.make_item_id(getattr(item, "url", "") or item.headline),
                symbol=symbol.upper(),
                headline=item.headline,
                summary=getattr(item, "summary", "")[:500],
                source=getattr(item, "source", "Unknown"),
                source_tier=source_tier,
                url=getattr(item, "url", ""),
                published_time=getattr(item, "published_time", now),
                fetched_at=now,
                sentiment_score=sentiment,
                sentiment_confidence=confidence,
                impact=impact,
            )
            scored.append(scored_item)

        return scored

    def get_status(self) -> dict:
        """Get loop status for observability."""
        stats = self.bus.get_stats()
        return {
            "last_run": self._last_run.isoformat() if self._last_run else None,
            "cadence_sec": self.cadence_sec,
            "symbols_refreshed": self._symbols_refreshed,
            "items_published": self._items_published,
            "bus_stats": {
                "total_items": stats.total_items,
                "symbols_tracked": stats.symbols_tracked,
                "high_impact_items": stats.high_impact_items,
                "refresh_count": stats.refresh_count,
                "items_evicted": stats.items_evicted,
                "fetch_errors": stats.fetch_errors,
                "wake_events_fired": stats.wake_events_fired,
            },
        }
