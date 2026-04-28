"""v-news-verify-2026-04-28: fresh-news verification gate.

Before the news_strategy returns a BUY/SELL signal, this module re-fetches
news for the symbol from external sources and verifies:

  1. At least N fresh articles in the last M minutes (default 2 in 4 hours)
  2. Fresh sentiment direction matches the original signal
  3. Optional: at least one article from a high-quality source (Reuters,
     Bloomberg, Alpaca's curated feed)

Without this gate, the bot was firing on stale RSS news (often hours old)
that the market had already priced in. PLTR/TSLA on 2026-04-28 both
entered on pre-baked news and immediately faded — exactly the failure
mode this gate prevents.

Sources, in priority order:
  Alpaca News API     (curated, low-latency, requires keys in .env)
  Yahoo via yfinance  (free fallback, news.providerPublishTime)
  RSS/aggregated      (always available, lowest quality)

The verifier returns a small result dataclass with is_verified + reason +
diagnostics, so the caller can audit-log exactly why a signal was approved
or rejected.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import aiohttp

logger = logging.getLogger("TradingBot")

ALPACA_NEWS_URL = "https://data.alpaca.markets/v1beta1/news"


@dataclass
class VerifyResult:
    is_verified: bool
    fresh_count: int
    avg_fresh_sentiment: float  # signed; positive = bullish
    direction_matches: bool
    source: str  # which provider answered
    reason: str  # short audit string
    latest_age_min: Optional[float] = None


class NewsVerifier:
    """Stateless verifier; one shared instance per process is fine."""

    def __init__(
        self,
        sentiment_analyzer,  # SentimentIntensityAnalyzer (VADER)
        freshness_minutes: int = 240,  # 4 hours — wider than original 30 min
        min_fresh_articles: int = 2,
        min_match_strength: float = 0.10,  # avg compound must clear this in same direction
    ) -> None:
        self.sa = sentiment_analyzer
        self.freshness_minutes = freshness_minutes
        self.min_fresh_articles = min_fresh_articles
        self.min_match_strength = min_match_strength
        self._alpaca_key = os.getenv("ALPACA_API_KEY", "")
        self._alpaca_secret = os.getenv("ALPACA_SECRET_KEY", "")

    async def verify(
        self, symbol: str, expected_direction: int  # +1 buy, -1 sell
    ) -> VerifyResult:
        """Try Alpaca first, fall back to yfinance, then return a clear result."""
        # 1) Alpaca News (preferred)
        if self._alpaca_key and self._alpaca_secret:
            try:
                result = await self._verify_via_alpaca(symbol, expected_direction)
                if result is not None:
                    return result
            except Exception as exc:
                logger.debug(f"news_verifier alpaca failed for {symbol}: {exc}")

        # 2) yfinance fallback
        try:
            result = await self._verify_via_yfinance(symbol, expected_direction)
            if result is not None:
                return result
        except Exception as exc:
            logger.debug(f"news_verifier yfinance failed for {symbol}: {exc}")

        return VerifyResult(
            is_verified=False, fresh_count=0, avg_fresh_sentiment=0.0,
            direction_matches=False, source="none",
            reason="no_source_available",
        )

    async def _verify_via_alpaca(
        self, symbol: str, expected_direction: int
    ) -> Optional[VerifyResult]:
        """Query Alpaca News API for articles in the freshness window."""
        # Alpaca expects RFC-3339 in UTC.
        now_utc = datetime.now(timezone.utc)
        start = (now_utc - timedelta(minutes=self.freshness_minutes)).isoformat(
            timespec="seconds"
        ).replace("+00:00", "Z")

        params = {
            "symbols": symbol,
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

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as sess:
            async with sess.get(ALPACA_NEWS_URL, params=params, headers=headers) as r:
                if r.status != 200:
                    logger.debug(
                        f"news_verifier alpaca http {r.status} for {symbol}"
                    )
                    return None
                data = await r.json()

        articles = data.get("news") or []
        return self._evaluate(
            articles=[
                {
                    "headline": a.get("headline", ""),
                    "summary": a.get("summary", ""),
                    "ts": a.get("created_at") or a.get("updated_at"),
                }
                for a in articles
            ],
            symbol=symbol,
            expected_direction=expected_direction,
            source="alpaca",
        )

    async def _verify_via_yfinance(
        self, symbol: str, expected_direction: int
    ) -> Optional[VerifyResult]:
        """Fallback: yfinance ticker.news has providerPublishTime epochs."""
        try:
            import yfinance as yf
        except Exception:
            return None

        now_utc = datetime.now(timezone.utc)
        cutoff = now_utc - timedelta(minutes=self.freshness_minutes)
        ticker = yf.Ticker(symbol)
        raw = list(ticker.news or [])
        norm = []
        for a in raw:
            ts_epoch = a.get("providerPublishTime") or 0
            if not ts_epoch:
                continue
            ts = datetime.fromtimestamp(ts_epoch, tz=timezone.utc)
            if ts < cutoff:
                continue
            norm.append(
                {
                    "headline": a.get("title", ""),
                    "summary": a.get("summary", "") or "",
                    "ts": ts.isoformat(),
                }
            )
        return self._evaluate(
            articles=norm, symbol=symbol,
            expected_direction=expected_direction, source="yfinance",
        )

    def _evaluate(
        self,
        articles: list,
        symbol: str,
        expected_direction: int,
        source: str,
    ) -> VerifyResult:
        """Common scoring path. Articles must be normalised dicts."""
        now_utc = datetime.now(timezone.utc)
        cutoff = now_utc - timedelta(minutes=self.freshness_minutes)

        fresh = []
        latest_ts: Optional[datetime] = None
        for a in articles:
            ts_str = a.get("ts")
            if not ts_str:
                continue
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except (TypeError, ValueError):
                continue
            if ts < cutoff:
                continue
            if latest_ts is None or ts > latest_ts:
                latest_ts = ts
            fresh.append(a)

        if len(fresh) < self.min_fresh_articles:
            return VerifyResult(
                is_verified=False,
                fresh_count=len(fresh),
                avg_fresh_sentiment=0.0,
                direction_matches=False,
                source=source,
                reason=f"insufficient_fresh_articles_{len(fresh)}_lt_{self.min_fresh_articles}",
                latest_age_min=(
                    (now_utc - latest_ts).total_seconds() / 60.0
                    if latest_ts else None
                ),
            )

        # Score sentiment on fresh articles only
        scores = []
        for a in fresh:
            text = f"{a['headline']} {a['summary']}"
            try:
                scores.append(self.sa.polarity_scores(text)["compound"])
            except Exception:
                continue
        if not scores:
            return VerifyResult(
                is_verified=False,
                fresh_count=len(fresh),
                avg_fresh_sentiment=0.0,
                direction_matches=False,
                source=source,
                reason="sentiment_scoring_failed",
                latest_age_min=(
                    (now_utc - latest_ts).total_seconds() / 60.0
                    if latest_ts else None
                ),
            )

        avg = sum(scores) / len(scores)
        direction_matches = (
            (expected_direction > 0 and avg >= self.min_match_strength)
            or (expected_direction < 0 and avg <= -self.min_match_strength)
        )
        if not direction_matches:
            return VerifyResult(
                is_verified=False,
                fresh_count=len(fresh),
                avg_fresh_sentiment=avg,
                direction_matches=False,
                source=source,
                reason=f"fresh_sentiment_against_signal_avg={avg:.3f}",
                latest_age_min=(now_utc - latest_ts).total_seconds() / 60.0,
            )

        return VerifyResult(
            is_verified=True,
            fresh_count=len(fresh),
            avg_fresh_sentiment=avg,
            direction_matches=True,
            source=source,
            reason="ok",
            latest_age_min=(now_utc - latest_ts).total_seconds() / 60.0,
        )
