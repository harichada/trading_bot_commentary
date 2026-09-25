"""Tests for core.loops.news_loop -- sentiment scoring off the event loop.

v-playbook-desk-b1-fix-b-2026-09-25 (whole-branch review, Important
finding I1): `_fetch_and_score` (NewsLoop) and `_convert_to_scored_items`
(AlpacaNewsBusPublisher) used to call `score_sentiment([text])[0]`
synchronously, once per headline, inside `async def` bodies -- putting
FinBERT pipeline construction / CUDA init (and even VADER's lexicon
load) on the event-loop thread. These tests cover the fix:

  - a tick collects every headline it needs and scores them in a single
    `asyncio.to_thread(score_sentiment, headlines)` call (not one call
    per headline), verified with a recording wrapper around
    `asyncio.to_thread`;
  - a one-time warm-up at loop start forces the backend to load off the
    loop before the first real headline, and swallows a failing backend;
  - VADER scores are unchanged for a known headline (batching must not
    change what gets scored or how).

No real model is ever loaded: `core.sentiment_backend._finbert_factory`
is reset to a raising stub (same pattern as
`tests/core/test_sentiment_backend.py`), and the default backend
(config-driven) is VADER -- pure CPU lexicon scoring, safe to run for
real in these tests.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List
from unittest.mock import AsyncMock, MagicMock

import pytest


def _raising_finbert_factory():
    raise RuntimeError("FinBERT disabled in tests")


@pytest.fixture(autouse=True)
def _reset_sentiment_backend_state():
    """Never let a test reach the real FinBERT factory (no model load).

    Mirrors the autouse fixture in tests/core/test_sentiment_backend.py.
    """
    import core.sentiment_backend as sb

    sb._finbert_factory = _raising_finbert_factory
    sb._FINBERT_PIPELINE = None
    sb._FINBERT_LOAD_FAILED = False
    sb._FINBERT_WARNED = False
    yield
    sb._finbert_factory = _raising_finbert_factory
    sb._FINBERT_PIPELINE = None
    sb._FINBERT_LOAD_FAILED = False
    sb._FINBERT_WARNED = False


@pytest.fixture(autouse=True)
def _clear_sentiment_backend_env(monkeypatch):
    monkeypatch.delenv("SENTIMENT_BACKEND", raising=False)


class _RecordingToThread:
    """Stand-in for asyncio.to_thread that records every call and runs
    the function inline (synchronously) so tests stay deterministic."""

    def __init__(self):
        self.calls: list = []

    async def __call__(self, fn, *args, **kwargs):
        self.calls.append((fn, args, kwargs))
        return fn(*args, **kwargs)


def _make_news_item(headline: str, summary: str = "", url: str = ""):
    """A minimal stand-in for strategies.news_strategy.NewsItem -- the
    news_loop call sites only read these attributes off raw items."""

    class _RawNewsItem:
        pass

    item = _RawNewsItem()
    item.headline = headline
    item.summary = summary
    item.source = "Yahoo Finance"
    item.url = url or f"https://example.com/{headline[:10]}"
    item.published_time = datetime.now(timezone.utc)
    return item


def _make_engine(watchlist: List[str]):
    engine = MagicMock()
    engine.is_running = False
    engine.dynamic_watchlist = watchlist
    return engine


class TestNewsLoopScoresOncePerTick:
    @pytest.mark.asyncio
    async def test_tick_scores_all_headlines_in_a_single_off_loop_call(self, monkeypatch):
        from core.loops.news_loop import NewsLoop
        from core.news_bus import NewsBus
        import core.loops.news_scoring as news_scoring

        engine = _make_engine(["AAA", "BBB"])
        bus = NewsBus()
        loop = NewsLoop(engine, cadence_sec=999, bus=bus)

        raw_by_symbol = {
            "AAA": [
                _make_news_item("AAA beats on earnings", "strong guidance"),
                _make_news_item("AAA announces buyback"),
            ],
            "BBB": [
                _make_news_item("BBB misses estimates", "outlook cut"),
            ],
        }

        async def fake_fetch_news(symbol, hours=4):
            return raw_by_symbol.get(symbol, [])

        loop.aggregator.fetch_news = fake_fetch_news

        recorder = _RecordingToThread()
        monkeypatch.setattr(news_scoring.asyncio, "to_thread", recorder)

        calls = {"n": 0}

        def fake_score_sentiment(headlines, backend=None):
            calls["n"] += 1
            # Deterministic, distinguishable scores by position.
            return [0.1 * (i + 1) for i in range(len(headlines))]

        monkeypatch.setattr(news_scoring, "score_sentiment", fake_score_sentiment)

        await loop._tick()

        # Scored exactly once per tick, in a single to_thread call, with
        # every headline collected this tick -- not one call per headline.
        assert len(recorder.calls) == 1
        fn, args, kwargs = recorder.calls[0]
        assert fn is fake_score_sentiment
        texts = args[0]
        assert len(texts) == 3
        assert texts[0] == "AAA beats on earnings strong guidance"
        assert texts[1] == "AAA announces buyback "
        assert texts[2] == "BBB misses estimates outlook cut"
        assert calls["n"] == 1

        # Scores mapped back to the right items, in the same order.
        aaa_items = bus._items["AAA"]
        bbb_items = bus._items["BBB"]
        assert len(aaa_items) == 2
        assert len(bbb_items) == 1
        scores_by_headline = {
            i.headline: i.sentiment_score for i in aaa_items + bbb_items
        }
        assert scores_by_headline["AAA beats on earnings"] == pytest.approx(0.1)
        assert scores_by_headline["AAA announces buyback"] == pytest.approx(0.2)
        assert scores_by_headline["BBB misses estimates"] == pytest.approx(0.3)

        status = loop.get_status()
        assert status["symbols_refreshed"] == 2
        assert status["items_published"] == 3

    @pytest.mark.asyncio
    async def test_tick_with_no_headlines_does_not_spawn_a_thread(self, monkeypatch):
        """No fetched news this tick -> nothing to score -> to_thread is
        never called (score_headlines_off_loop short-circuits on [])."""
        from core.loops.news_loop import NewsLoop
        from core.news_bus import NewsBus
        import core.loops.news_scoring as news_scoring

        engine = _make_engine(["AAA"])
        bus = NewsBus()
        loop = NewsLoop(engine, cadence_sec=999, bus=bus)

        async def fake_fetch_news(symbol, hours=4):
            return []

        loop.aggregator.fetch_news = fake_fetch_news

        recorder = _RecordingToThread()
        monkeypatch.setattr(news_scoring.asyncio, "to_thread", recorder)

        await loop._tick()

        assert recorder.calls == []
        assert loop.get_status()["symbols_refreshed"] == 0


class TestNewsLoopWarmup:
    @pytest.mark.asyncio
    async def test_run_warms_up_backend_once_before_looping(self, monkeypatch):
        from core.loops.news_loop import NewsLoop
        from core.news_bus import NewsBus
        import core.loops.news_scoring as news_scoring

        engine = _make_engine([])
        engine.is_running = False  # loop body never executes; only warm-up runs
        bus = NewsBus()
        loop = NewsLoop(engine, cadence_sec=999, bus=bus)

        calls = {"n": 0, "texts": None}

        def fake_score_sentiment(headlines, backend=None):
            calls["n"] += 1
            calls["texts"] = list(headlines)
            return [0.0 for _ in headlines]

        monkeypatch.setattr(news_scoring, "score_sentiment", fake_score_sentiment)

        assert loop._warmed_up is False
        await loop.run()
        assert loop._warmed_up is True
        assert calls["n"] == 1
        assert calls["texts"] == ["warmup"]

        # Calling run() again must not warm up a second time.
        await loop.run()
        assert calls["n"] == 1

    @pytest.mark.asyncio
    async def test_warmup_swallows_a_failing_backend(self, monkeypatch, caplog):
        from core.loops.news_loop import NewsLoop
        from core.news_bus import NewsBus
        import core.loops.news_scoring as news_scoring

        engine = _make_engine([])
        engine.is_running = False
        bus = NewsBus()
        loop = NewsLoop(engine, cadence_sec=999, bus=bus)

        def broken_score_sentiment(headlines, backend=None):
            raise RuntimeError("no GPU / model not found")

        monkeypatch.setattr(news_scoring, "score_sentiment", broken_score_sentiment)

        with caplog.at_level("WARNING"):
            await loop.run()  # must not raise

        assert loop._warmed_up is True
        warnings = [r for r in caplog.records if "warm-up failed" in r.message]
        assert len(warnings) == 1

    @pytest.mark.asyncio
    async def test_warmup_helper_runs_off_the_loop(self, monkeypatch):
        """warmup_sentiment_backend itself must go through asyncio.to_thread,
        not call score_sentiment directly on the loop thread."""
        from core.loops.news_scoring import warmup_sentiment_backend
        import core.loops.news_scoring as news_scoring

        recorder = _RecordingToThread()
        monkeypatch.setattr(news_scoring.asyncio, "to_thread", recorder)

        calls = {"n": 0}

        def fake_score_sentiment(headlines, backend=None):
            calls["n"] += 1
            return [0.0 for _ in headlines]

        monkeypatch.setattr(news_scoring, "score_sentiment", fake_score_sentiment)

        await warmup_sentiment_backend("news_loop")

        assert len(recorder.calls) == 1
        assert recorder.calls[0][0] is fake_score_sentiment
        assert recorder.calls[0][1][0] == ["warmup"]
        assert calls["n"] == 1


class TestVaderScoresUnchanged:
    @pytest.mark.asyncio
    async def test_vader_score_matches_direct_call_for_known_headline(self):
        """Batching headlines into one call must not change VADER's
        per-headline output vs. the old one-call-per-headline path."""
        from core.loops.news_loop import NewsLoop
        from core.news_bus import NewsBus
        from core.sentiment_backend import score as score_sentiment

        headline = "Excellent earnings beat, strong guidance boost"
        summary = "shares jump on the news"
        expected = score_sentiment([f"{headline} {summary}"])[0]

        engine = _make_engine(["AAA"])
        bus = NewsBus()
        loop = NewsLoop(engine, cadence_sec=999, bus=bus)

        async def fake_fetch_news(symbol, hours=4):
            return [_make_news_item(headline, summary)]

        loop.aggregator.fetch_news = fake_fetch_news

        await loop._tick()

        items = bus._items["AAA"]
        assert len(items) == 1
        assert items[0].sentiment_score == pytest.approx(expected)
        assert expected > 0.2  # sanity: genuinely bullish under VADER


class TestAlpacaPublisherScoresOncePerTick:
    @pytest.mark.asyncio
    async def test_convert_to_scored_items_scores_once_off_loop(self, monkeypatch):
        from core.loops.news_loop import AlpacaNewsBusPublisher
        from core.news_bus import NewsBus
        import core.loops.news_scoring as news_scoring

        engine = _make_engine([])
        bus = NewsBus()
        publisher = AlpacaNewsBusPublisher(engine, bus=bus)

        articles = [
            {
                "id": "1",
                "headline": "Company beats estimates",
                "summary": "raises guidance",
                "symbols": ["NVDA", "AMD"],
                "created_at": "2026-09-14T12:00:00Z",
                "url": "https://example.com/a",
            },
            {
                "id": "2",
                "headline": "Company misses estimates",
                "summary": "cuts outlook",
                "symbols": ["INTC"],
                "created_at": "2026-09-14T12:05:00Z",
                "url": "https://example.com/b",
            },
        ]

        recorder = _RecordingToThread()
        monkeypatch.setattr(news_scoring.asyncio, "to_thread", recorder)

        def fake_score_sentiment(headlines, backend=None):
            return [0.5, -0.5]

        monkeypatch.setattr(news_scoring, "score_sentiment", fake_score_sentiment)

        items = await publisher._convert_to_scored_items(articles)

        assert len(recorder.calls) == 1
        texts = recorder.calls[0][1][0]
        assert texts == [
            "Company beats estimates raises guidance",
            "Company misses estimates cuts outlook",
        ]

        # One article fans out to a ScoredNewsItem per symbol; sentiment
        # for a given article is shared across its symbols.
        assert len(items) == 3
        by_symbol = {i.symbol: i.sentiment_score for i in items}
        assert by_symbol["NVDA"] == pytest.approx(0.5)
        assert by_symbol["AMD"] == pytest.approx(0.5)
        assert by_symbol["INTC"] == pytest.approx(-0.5)

    @pytest.mark.asyncio
    async def test_alpaca_run_warms_up_backend_once(self, monkeypatch):
        from core.loops.news_loop import AlpacaNewsBusPublisher
        from core.news_bus import NewsBus
        import core.loops.news_scoring as news_scoring

        monkeypatch.setenv("ALPACA_API_KEY", "test_key")
        monkeypatch.setenv("ENABLE_ALPACA_NEWS_BUS", "1")

        engine = _make_engine([])
        engine.is_running = False
        bus = NewsBus()
        publisher = AlpacaNewsBusPublisher(
            engine, bus=bus, cadence_sec=999,
        )
        publisher._alpaca_key = "test_key"
        publisher._alpaca_secret = "test_secret"

        calls = {"n": 0}

        def fake_score_sentiment(headlines, backend=None):
            calls["n"] += 1
            return [0.0 for _ in headlines]

        monkeypatch.setattr(news_scoring, "score_sentiment", fake_score_sentiment)

        assert publisher._warmed_up is False
        await publisher.run()
        assert publisher._warmed_up is True
        assert calls["n"] == 1
