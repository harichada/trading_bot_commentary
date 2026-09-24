"""v-newsbus-2026-09-08: news_loop refreshes the NewsBus from FreeNewsAggregator.

This loop is the sole writer to NewsBus. It fetches news for all symbols
in the dynamic watchlist on a ~20s cadence, scores them with VADER, and
publishes to the bus. High-impact items trigger wake events that the
analysis loop can use for early entry evaluation.

v-theme-shock-logger-2026-09-14: added AlpacaNewsBusPublisher to fetch
Alpaca news and publish to NewsBus as a tier-1 source. Runs on a separate
cadence (default 60s) from the main news loop. Reuses ALPACA_API_KEY/SECRET
from existing news_verifier.py integration.

The news_loop is registered with TaskSupervisor at NORMAL priority.
Failures are logged and swallowed per-iteration; only structural errors
surface to the supervisor.
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, List, Optional

import aiohttp
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

ALPACA_NEWS_URL = "https://data.alpaca.markets/v1beta1/news"


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
        now = datetime.now(timezone.utc)

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


class AlpacaNewsBusPublisher:
    """v-theme-shock-logger-2026-09-14: Alpaca news publisher for NewsBus.
    
    Fetches news from Alpaca News API and publishes to NewsBus as a
    tier-1 source. Runs on a separate cadence (default 60s) from the
    main news loop. Reuses ALPACA_API_KEY/SECRET from .env.
    
    This is the Alpaca reuse path required by the spec — NOT a new
    ingest system. Leverages existing keys and news_verifier patterns.
    """

    SOURCE_NAME = "Alpaca News"
    SOURCE_TIER = 1

    def __init__(
        self,
        engine: "TradingEngineWithCommentary",
        bus: Optional[NewsBus] = None,
        cadence_sec: Optional[float] = None,
    ) -> None:
        self.engine = engine
        self.bus = bus or get_news_bus()
        self._alpaca_key = os.getenv("ALPACA_API_KEY", "")
        self._alpaca_secret = os.getenv("ALPACA_SECRET_KEY", "")
        self._cadence_sec = cadence_sec or Config().ALPACA_NEWS_BUS_INTERVAL_SEC
        self._analyzer = SentimentIntensityAnalyzer()
        self._last_run: Optional[datetime] = None
        self._items_published: int = 0

    def is_enabled(self) -> bool:
        """Check if Alpaca news bus is enabled and configured."""
        return (
            Config().ENABLE_ALPACA_NEWS_BUS
            and bool(self._alpaca_key)
            and bool(self._alpaca_secret)
        )

    async def run(self) -> None:
        """Main loop: fetch Alpaca news for watchlist on cadence."""
        if not self.is_enabled():
            logger.info("alpaca_news_bus: disabled or no API keys, exiting")
            return

        logger.info(
            "alpaca_news_bus: started (cadence=%.1fs)",
            self._cadence_sec,
        )

        while getattr(self.engine, "is_running", False):
            try:
                await self._tick()
            except Exception as exc:
                logger.warning("alpaca_news_bus: tick failed: %s", exc, exc_info=True)
                self.bus.record_fetch_error()
            await asyncio.sleep(self._cadence_sec)

    async def _tick(self) -> None:
        """Single fetch cycle for Alpaca news."""
        watchlist = getattr(self.engine, "dynamic_watchlist", None) or []
        if not watchlist:
            return

        start = datetime.now()
        total_new = 0

        batch_size = 10
        for i in range(0, len(watchlist), batch_size):
            batch = watchlist[i:i + batch_size]
            items = await self._fetch_alpaca_news(batch)
            if items:
                new_count = await self.bus.publish(items)
                total_new += new_count

        self._last_run = datetime.now()
        self._items_published += total_new

        elapsed_ms = (datetime.now() - start).total_seconds() * 1000
        if total_new > 0:
            logger.info(
                "alpaca_news_bus tick new_items=%d elapsed_ms=%.0f",
                total_new, elapsed_ms,
            )

    async def _fetch_alpaca_news(
        self,
        symbols: List[str],
        hours: int = 4,
    ) -> List[ScoredNewsItem]:
        """Fetch news from Alpaca for a batch of symbols."""
        if not symbols:
            return []

        now_utc = datetime.now(timezone.utc)
        start = (now_utc - timedelta(hours=hours)).isoformat(
            timespec="seconds"
        ).replace("+00:00", "Z")

        params = {
            "symbols": ",".join(symbols),
            "start": start,
            "limit": 50,
            "sort": "desc",
            "include_content": "false",
        }
        headers = {
            "APCA-API-KEY-ID": self._alpaca_key,
            "APCA-API-SECRET-KEY": self._alpaca_secret,
            "accept": "application/json",
        }

        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            ) as sess:
                async with sess.get(
                    ALPACA_NEWS_URL, params=params, headers=headers
                ) as r:
                    if r.status != 200:
                        logger.debug(
                            "alpaca_news_bus: http %d for %s",
                            r.status, ",".join(symbols)
                        )
                        return []
                    data = await r.json()
        except Exception as e:
            logger.debug("alpaca_news_bus: fetch error: %s", e)
            return []

        articles = data.get("news") or []
        return self._convert_to_scored_items(articles)

    def _convert_to_scored_items(
        self,
        articles: List[dict],
    ) -> List[ScoredNewsItem]:
        """Convert Alpaca news articles to ScoredNewsItem list."""
        now = datetime.now(timezone.utc)
        scored = []

        for article in articles:
            headline = article.get("headline", "")
            summary = article.get("summary", "") or ""
            symbols = article.get("symbols", [])

            if not headline or not symbols:
                continue

            ts_str = article.get("created_at") or article.get("updated_at")
            try:
                # v-theme-shock-hotfix-2026-09-14: preserve UTC timezone info
                # instead of stripping it. This fixes age_sec() negative values.
                pub_time = datetime.fromisoformat(
                    ts_str.replace("Z", "+00:00")
                )
            except (TypeError, ValueError):
                pub_time = datetime.now(timezone.utc)

            text = f"{headline} {summary}"
            vader_scores = self._analyzer.polarity_scores(text)
            sentiment = vader_scores["compound"]
            confidence = abs(sentiment)

            impact = NewsBus.detect_impact(headline, summary)
            url = article.get("url", "")
            article_id = article.get("id") or NewsBus.make_item_id(url or headline)

            for symbol in symbols:
                scored.append(ScoredNewsItem(
                    id=f"alpaca_{article_id}_{symbol}",
                    symbol=symbol.upper(),
                    headline=headline,
                    summary=summary[:500],
                    source=self.SOURCE_NAME,
                    source_tier=self.SOURCE_TIER,
                    url=url,
                    published_time=pub_time,
                    fetched_at=now,
                    sentiment_score=sentiment,
                    sentiment_confidence=confidence,
                    impact=impact,
                ))

        return scored

    def get_status(self) -> dict:
        """Get Alpaca publisher status for observability."""
        return {
            "enabled": self.is_enabled(),
            "last_run": self._last_run.isoformat() if self._last_run else None,
            "cadence_sec": self._cadence_sec,
            "items_published": self._items_published,
        }
