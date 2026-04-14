"""Minimal news sentiment widget.

Backs the /api/sentiment/* endpoints by reusing FreeNewsAggregator (Yahoo +
RSS + MarketWatch) from strategies.news_strategy plus VADER scoring. No
external paid APIs, no LLM. Stubs out social and earnings sub-trackers
(return empty until you wire real sources).
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Any, Deque, Dict, List, Optional

from nltk.sentiment import SentimentIntensityAnalyzer

from strategies.news_strategy import FreeNewsAggregator

logger = logging.getLogger("TradingBot")


@dataclass
class SentimentAlert:
    id: str
    symbol: str
    severity: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "severity": self.severity,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "acknowledged": self.acknowledged,
        }


class _EmptySocialTracker:
    """Stub: returns neutral social sentiment until a real source is wired."""

    async def get_social_sentiment(self, symbol: str) -> Dict[str, Any]:
        return {
            "symbol": symbol,
            "score": 0.0,
            "mentions": 0,
            "sources": [],
            "note": "social tracking not configured",
        }


class _EmptyEarningsCalendar:
    """Stub: returns empty earnings list until a real source is wired."""

    async def get_upcoming_earnings(self, symbols: List[str]) -> List[Dict[str, Any]]:
        return []


class NewsSentimentEngine:
    """Per-symbol news fetch + VADER scoring + simple alerting."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self.watchlist: List[str] = list(config.get("watchlist") or [])
        self.lookback_hours: int = int(config.get("lookback_hours", 24))
        self.alert_threshold: float = float(config.get("alert_threshold", 0.5))

        self.aggregator = FreeNewsAggregator()
        self.analyzer = SentimentIntensityAnalyzer()
        self.social_tracker = _EmptySocialTracker()
        self.earnings_calendar = _EmptyEarningsCalendar()

        # cache: symbol -> (last_update, sentiment_dict, [news_items])
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl = timedelta(minutes=5)
        self._alerts: Deque[SentimentAlert] = deque(maxlen=200)

    def _score(self, text: str) -> float:
        return float(self.analyzer.polarity_scores(text or "")["compound"])

    async def _refresh_symbol(self, symbol: str) -> Dict[str, Any]:
        """Fetch news for one symbol, score it, cache, raise alerts."""
        cached = self._cache.get(symbol)
        if cached and datetime.now() - cached["timestamp"] < self._cache_ttl:
            return cached

        try:
            items = await self.aggregator.fetch_news(symbol, self.lookback_hours)
        except Exception as e:
            logger.warning("News fetch failed for %s: %s", symbol, e)
            items = []

        scores: List[float] = []
        headlines: List[Dict[str, Any]] = []
        for item in items:
            text = f"{item.headline} {getattr(item, 'summary', '')}"
            score = self._score(text)
            item.sentiment_score = score
            item.sentiment_confidence = abs(score)
            scores.append(score)
            headlines.append({
                "symbol": symbol,
                "headline": item.headline,
                "summary": getattr(item, "summary", ""),
                "source": getattr(item, "source", ""),
                "url": getattr(item, "url", ""),
                "sentiment": score,
                "timestamp": getattr(item, "timestamp", datetime.now()).isoformat()
                    if hasattr(getattr(item, "timestamp", None), "isoformat")
                    else str(getattr(item, "timestamp", "")),
            })

        avg = sum(scores) / len(scores) if scores else 0.0
        result = {
            "symbol": symbol,
            "score": avg,
            "article_count": len(items),
            "headlines": headlines,
            "timestamp": datetime.now(),
        }
        self._cache[symbol] = result

        if abs(avg) >= self.alert_threshold and len(items) >= 3:
            severity = "high" if abs(avg) >= 0.7 else "medium"
            direction = "positive" if avg > 0 else "negative"
            self._alerts.append(SentimentAlert(
                id=str(uuid.uuid4()),
                symbol=symbol,
                severity=severity,
                message=f"{symbol}: strong {direction} sentiment ({avg:+.2f}) across {len(items)} articles",
            ))

        return result

    async def update(self) -> Dict[str, Any]:
        """Refresh all watchlist symbols and return aggregate sentiment."""
        if not self.watchlist:
            return {"market_sentiment": 0.0, "symbols": {}, "timestamp": datetime.now().isoformat()}

        results = await asyncio.gather(
            *(self._refresh_symbol(s) for s in self.watchlist),
            return_exceptions=True,
        )
        symbols: Dict[str, Dict[str, Any]] = {}
        scores: List[float] = []
        for sym, res in zip(self.watchlist, results):
            if isinstance(res, Exception):
                symbols[sym] = {"score": 0.0, "article_count": 0, "error": str(res)}
                continue
            symbols[sym] = {"score": res["score"], "article_count": res["article_count"]}
            if res["article_count"] > 0:
                scores.append(res["score"])

        return {
            "market_sentiment": sum(scores) / len(scores) if scores else 0.0,
            "symbols": symbols,
            "timestamp": datetime.now().isoformat(),
        }

    def get_symbol_sentiment(self, symbol: str) -> Dict[str, Any]:
        cached = self._cache.get(symbol)
        if not cached:
            return {"symbol": symbol, "score": 0.0, "article_count": 0, "stale": True}
        return {
            "symbol": symbol,
            "score": cached["score"],
            "article_count": cached["article_count"],
            "timestamp": cached["timestamp"].isoformat(),
        }

    def get_headlines(self, symbol: Optional[str], limit: int = 50) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        symbols = [symbol] if symbol else list(self._cache.keys())
        for sym in symbols:
            cached = self._cache.get(sym)
            if cached:
                out.extend(cached["headlines"])
        out.sort(key=lambda h: h.get("timestamp", ""), reverse=True)
        return out[:limit]

    def acknowledge_alert(self, alert_id: str) -> bool:
        for a in self._alerts:
            if a.id == alert_id:
                a.acknowledged = True
                return True
        return False

    def get_sentiment_price_correlation(self, symbol: str) -> Dict[str, Any]:
        # Real implementation requires historical price + sentiment time series.
        # Return neutral until that is wired.
        return {"symbol": symbol, "correlation": 0.0, "samples": 0,
                "note": "correlation backfill not implemented"}


_engine_singleton: Optional[NewsSentimentEngine] = None


def get_sentiment_engine(config: Optional[Dict[str, Any]] = None) -> NewsSentimentEngine:
    global _engine_singleton
    if _engine_singleton is None:
        _engine_singleton = NewsSentimentEngine(config or {})
    return _engine_singleton
