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

    @staticmethod
    def _label(score_neg1_to_1: float) -> str:
        if score_neg1_to_1 >= 0.15:
            return "bullish"
        if score_neg1_to_1 <= -0.15:
            return "bearish"
        return "neutral"

    async def update(self) -> Dict[str, Any]:
        """Refresh watchlist and return the dashboard-shaped payload.

        Output contract (consumed by templates/dashboard.html updateSentimentTab/
        updateSentimentWidget):
          market_sentiment: {score(-100..100), label, confidence(0..1),
                             trend, source_count, last_updated}
          symbol_sentiments: {SYM: {score(-100..100), label, trend, article_count}}
          velocities:        {SYM: {recent_count, is_spike}}
          headlines:         [{symbol, headline, summary, source, url,
                               sentiment_score(-100..100), timestamp}]
          earnings:          []  (stub)
        """
        if not self.watchlist:
            return self._empty_payload()

        results = await asyncio.gather(
            *(self._refresh_symbol(s) for s in self.watchlist),
            return_exceptions=True,
        )

        symbol_sentiments: Dict[str, Dict[str, Any]] = {}
        velocities: Dict[str, Dict[str, Any]] = {}
        all_headlines: List[Dict[str, Any]] = []
        sources: set = set()
        scores: List[float] = []

        for sym, res in zip(self.watchlist, results):
            if isinstance(res, Exception):
                symbol_sentiments[sym] = {
                    "score": 0, "label": "neutral",
                    "trend": "--", "article_count": 0,
                    "error": str(res),
                }
                velocities[sym] = {"recent_count": 0, "is_spike": False}
                continue

            raw_score = float(res["score"])  # -1..+1
            article_count = int(res["article_count"])
            symbol_sentiments[sym] = {
                "score": round(raw_score * 100, 1),  # -100..+100 for the UI
                "label": self._label(raw_score),
                "trend": "--",
                "article_count": article_count,
            }
            velocities[sym] = {
                "recent_count": article_count,
                "is_spike": article_count >= 8,  # arbitrary spike threshold
            }
            for h in res["headlines"]:
                # Re-emit with score on -100..+100 so frontend Math.round() works
                all_headlines.append({
                    **h,
                    "sentiment_score": round(float(h.get("sentiment", 0.0)) * 100, 1),
                })
                if h.get("source"):
                    sources.add(h["source"])

            if article_count > 0:
                scores.append(raw_score)

        market_avg = sum(scores) / len(scores) if scores else 0.0
        # Confidence: how strong the signal is, scaled by coverage
        coverage = len(scores) / len(self.watchlist) if self.watchlist else 0.0
        confidence = min(1.0, abs(market_avg) * coverage * 1.5)

        # Most recent first
        all_headlines.sort(key=lambda h: h.get("timestamp", ""), reverse=True)

        return {
            "market_sentiment": {
                "score": round(market_avg * 100, 1),
                "label": self._label(market_avg),
                "confidence": round(confidence, 3),
                "trend": "--",
                "source_count": len(sources),
                "last_updated": datetime.now().isoformat(),
            },
            "symbol_sentiments": symbol_sentiments,
            "velocities": velocities,
            "headlines": all_headlines[:50],
            "earnings": [],
            "timestamp": datetime.now().isoformat(),
            # Backward-compat: keep numeric market_sentiment under a separate key
            "market_sentiment_raw": round(market_avg, 4),
        }

    def _empty_payload(self) -> Dict[str, Any]:
        return {
            "market_sentiment": {
                "score": 0, "label": "neutral", "confidence": 0,
                "trend": "--", "source_count": 0,
                "last_updated": datetime.now().isoformat(),
            },
            "symbol_sentiments": {},
            "velocities": {},
            "headlines": [],
            "earnings": [],
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
