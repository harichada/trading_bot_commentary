"""Off-event-loop sentiment scoring helpers for the live news loops.

v-playbook-desk-b1-fix-b-2026-09-25 (whole-branch review, Important
finding I1): `core/loops/news_loop.py` used to call
`core.sentiment_backend.score()` synchronously inside `async def`
bodies, once per headline. With `SENTIMENT_BACKEND=finbert` the first
call ran transformers pipeline construction + CUDA init on the
event-loop thread (a multi-second freeze of order monitoring and
streaming), and every later headline was a separate 1-item GPU forward
pass on the loop thread. Even VADER's lexicon load had drifted onto the
loop from `NewsLoop.__init__`.

This module pushes every `score()` call onto a worker thread via
`asyncio.to_thread`, and batches a tick's headlines into a single call
instead of one call per headline:

- `score_headlines_off_loop` -- score a batch of headlines off the loop.
- `warmup_sentiment_backend` -- force the configured backend to load
  once, off the loop, at loop start, so the model-load cost never lands
  on the first real headline (or on the loop thread at all).
- `score_news_bus_items` / `score_alpaca_articles` -- the two live
  call-site builders (`NewsLoop` and `AlpacaNewsBusPublisher`), each
  scoring its whole tick's batch of headlines in one off-loop call
  before building `ScoredNewsItem`s. Also keeps `news_loop.py` under
  the file's 400-line budget.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Tuple

from core.news_bus import NewsBus, ScoredNewsItem
from core.sentiment_backend import score as score_sentiment

logger = logging.getLogger("TradingBot")


async def score_headlines_off_loop(texts: List[str]) -> List[float]:
    """Score `texts` once, off the event loop, in a single backend call.

    Returns `[]` without spawning a worker thread when there is nothing
    to score (a tick with no fetched headlines).
    """
    if not texts:
        return []
    return await asyncio.to_thread(score_sentiment, texts)


async def warmup_sentiment_backend(loop_name: str = "news_loop") -> None:
    """Force the configured sentiment backend to load, off the loop.

    Intended to run once at loop start. Failures are logged and
    swallowed -- a warm-up failure must not prevent the loop from
    running; later real scoring calls simply keep falling back to
    VADER, same as any other backend load failure.
    """
    try:
        await asyncio.to_thread(score_sentiment, ["warmup"])
    except Exception as exc:
        logger.warning("%s: sentiment backend warm-up failed: %s", loop_name, exc)


async def score_news_bus_items(
    fetch_results: List[Tuple[str, object]],
    bus: NewsBus,
) -> Dict[str, List[ScoredNewsItem]]:
    """Score every headline `NewsLoop` fetched this tick, in one call.

    `fetch_results` is a list of `(symbol, raw_items_or_Exception)`
    pairs, one per symbol attempted this tick. Returns
    `{symbol: [ScoredNewsItem, ...]}` for symbols that had raw items;
    symbols with no items or a fetch exception are omitted (a fetch
    exception also records a bus fetch error, matching prior
    per-symbol behavior).
    """
    now = datetime.now(timezone.utc)
    texts: List[str] = []
    positions: List[tuple] = []

    for symbol, result in fetch_results:
        if isinstance(result, Exception):
            logger.debug("news_loop: fetch error for %s: %s", symbol, result)
            bus.record_fetch_error()
            continue
        for item in result:
            texts.append(f"{item.headline} {getattr(item, 'summary', '')}")
            positions.append((symbol, item))

    scores = await score_headlines_off_loop(texts)

    scored_by_symbol: Dict[str, List[ScoredNewsItem]] = {}
    for (symbol, item), sentiment in zip(positions, scores):
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
        scored_by_symbol.setdefault(symbol, []).append(scored_item)

    return scored_by_symbol


async def score_alpaca_articles(
    articles: List[dict],
    source_name: str,
    source_tier: int,
) -> List[ScoredNewsItem]:
    """Score every Alpaca article in this batch in one off-loop call.

    Mirrors `score_news_bus_items` for `AlpacaNewsBusPublisher`, which
    fans one article out to a `ScoredNewsItem` per symbol it tags.
    """
    now = datetime.now(timezone.utc)
    parsed: List[tuple] = []
    texts: List[str] = []

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
            pub_time = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except (TypeError, ValueError):
            pub_time = datetime.now(timezone.utc)

        url = article.get("url", "")
        article_id = article.get("id") or NewsBus.make_item_id(url or headline)
        parsed.append((headline, summary, symbols, pub_time, article_id, url))
        texts.append(f"{headline} {summary}")

    scores = await score_headlines_off_loop(texts)

    scored: List[ScoredNewsItem] = []
    for (headline, summary, symbols, pub_time, article_id, url), sentiment in zip(parsed, scores):
        confidence = abs(sentiment)
        impact = NewsBus.detect_impact(headline, summary)

        for symbol in symbols:
            scored.append(ScoredNewsItem(
                id=f"alpaca_{article_id}_{symbol}",
                symbol=symbol.upper(),
                headline=headline,
                summary=summary[:500],
                source=source_name,
                source_tier=source_tier,
                url=url,
                published_time=pub_time,
                fetched_at=now,
                sentiment_score=sentiment,
                sentiment_confidence=confidence,
                impact=impact,
            ))

    return scored
