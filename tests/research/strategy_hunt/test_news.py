import numpy as np
import pandas as pd
import pytest


class _Resp:
    def __init__(self, payload, status=200):
        self._p = payload
        self.status_code = status
    def json(self):
        return self._p
    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _Session:
    def __init__(self, pages):
        self.pages = list(pages)
        self.calls = []
    def get(self, url, headers=None, params=None, timeout=None):
        self.calls.append(dict(params))
        return _Resp(self.pages.pop(0))


def test_score_headlines_sign():
    from research.strategy_hunt.data.news import score_headlines
    s = score_headlines(["Excellent earnings beat, strong guidance boost", "Terrible earnings miss, cuts outlook, shares plunge"])
    assert s[0] > 0.2 and s[1] < -0.2


def test_fetch_news_paginates_and_converts_to_et():
    from research.strategy_hunt.data.news import fetch_news
    sess = _Session([
        {"news": [{"created_at": "2025-01-02T14:35:00Z", "headline": "Excellent beats", "source": "benzinga", "symbols": ["NVDA"]}],
         "next_page_token": "abc"},
        {"news": [{"created_at": "2025-01-02T15:00:00Z", "headline": "Terrible miss plunge", "source": "benzinga", "symbols": ["NVDA"]}],
         "next_page_token": None},
    ])
    df = fetch_news("NVDA", pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-03"), sess, "k", "s")
    assert len(df) == 2
    assert df.iloc[0]["created_at"] == pd.Timestamp("2025-01-02 09:35")   # UTC-5 in January
    assert sess.calls[1]["page_token"] == "abc"
    assert df.iloc[0]["sentiment"] > 0 and df.iloc[1]["sentiment"] < 0


def test_update_news_is_incremental(tmp_path):
    from research.strategy_hunt.data import news as N
    sess = _Session([
        {"news": [{"created_at": "2025-01-02T14:35:00Z", "headline": "A", "source": "x", "symbols": ["NVDA"]}], "next_page_token": None},
        {"news": [{"created_at": "2025-01-03T14:35:00Z", "headline": "B", "source": "x", "symbols": ["NVDA"]}], "next_page_token": None},
    ])
    n1 = N.update_news(tmp_path, "NVDA", sess, "k", "s", now=pd.Timestamp("2025-01-04"))
    n2 = N.update_news(tmp_path, "NVDA", sess, "k", "s", now=pd.Timestamp("2025-01-04"))
    assert (n1, n2) == (1, 1)
    assert sess.calls[1]["start"].startswith("2025-01-02T14:35")
    df = N.load_news(tmp_path, "NVDA")
    assert list(df["headline"]) == ["A", "B"]


def test_load_news_missing(tmp_path):
    from research.strategy_hunt.data.news import load_news
    df = load_news(tmp_path, "ZZZZ")
    assert df.empty and list(df.columns) == ["created_at", "headline", "source", "sentiment"]


# v-playbook-desk-b1-2026-09-25: Task 7 pluggable sentiment backend coverage.
# `available_backends`/`score` are monkeypatched on the `news` module (not on
# core.sentiment_backend) so these tests stay focused on news.py's column
# logic and never risk a real FinBERT model load/download.

def test_fetch_news_stores_vader_column_always():
    from research.strategy_hunt.data import news as N
    sess = _Session([
        {"news": [{"created_at": "2025-01-02T14:35:00Z", "headline": "Excellent beats", "source": "benzinga", "symbols": ["NVDA"]}],
         "next_page_token": None},
    ])
    df = N.fetch_news("NVDA", pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-03"), sess, "k", "s")
    assert "sentiment_vader" in df.columns
    assert df.iloc[0]["sentiment_vader"] > 0
    # default configured backend is vader -> sentiment mirrors sentiment_vader
    assert df.iloc[0]["sentiment"] == df.iloc[0]["sentiment_vader"]


def test_fetch_news_finbert_column_nan_when_unavailable(monkeypatch):
    from research.strategy_hunt.data import news as N
    monkeypatch.setattr(N, "available_backends", lambda: ("vader",))
    sess = _Session([
        {"news": [{"created_at": "2025-01-02T14:35:00Z", "headline": "Excellent beats", "source": "benzinga", "symbols": ["NVDA"]}],
         "next_page_token": None},
    ])
    df = N.fetch_news("NVDA", pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-03"), sess, "k", "s")
    assert "sentiment_finbert" in df.columns
    assert pd.isna(df.iloc[0]["sentiment_finbert"])


def test_fetch_news_finbert_column_populated_and_selected_when_configured(monkeypatch):
    from research.strategy_hunt.data import news as N

    monkeypatch.setattr(N, "available_backends", lambda: ("vader", "finbert"))

    def fake_score(headlines, backend=None):
        return [0.42 for _ in headlines] if backend == "finbert" else [0.1 for _ in headlines]

    monkeypatch.setattr(N, "score", fake_score)
    monkeypatch.setenv("SENTIMENT_BACKEND", "finbert")

    sess = _Session([
        {"news": [{"created_at": "2025-01-02T14:35:00Z", "headline": "Excellent beats", "source": "benzinga", "symbols": ["NVDA"]}],
         "next_page_token": None},
    ])
    df = N.fetch_news("NVDA", pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-03"), sess, "k", "s")
    assert df.iloc[0]["sentiment_finbert"] == pytest.approx(0.42)
    assert df.iloc[0]["sentiment_vader"] == pytest.approx(0.1)
    assert df.iloc[0]["sentiment"] == pytest.approx(0.42)


def test_fetch_news_vader_backend_never_touches_finbert_factory(monkeypatch):
    """Critical finding: with SENTIMENT_BACKEND=vader (the default),
    `_score_columns` must not attempt a FinBERT load at all — a counting
    stub that would raise if called must see zero calls.
    """
    import core.sentiment_backend as sb

    calls = {"n": 0}

    def counting_raising_factory():
        calls["n"] += 1
        raise RuntimeError("factory should never be reached for vader backend")

    monkeypatch.setattr(sb, "_finbert_factory", counting_raising_factory)
    monkeypatch.delenv("SENTIMENT_BACKEND", raising=False)

    from research.strategy_hunt.data import news as N
    sess = _Session([
        {"news": [{"created_at": "2025-01-02T14:35:00Z", "headline": "Excellent beats", "source": "benzinga", "symbols": ["NVDA"]}],
         "next_page_token": None},
    ])
    df = N.fetch_news("NVDA", pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-03"), sess, "k", "s")

    assert calls["n"] == 0
    assert pd.isna(df.iloc[0]["sentiment_finbert"])
    assert df.iloc[0]["sentiment"] == pytest.approx(df.iloc[0]["sentiment_vader"])


def test_fetch_news_finbert_backend_populates_column_via_fake_pipeline(monkeypatch):
    """Critical finding: with SENTIMENT_BACKEND=finbert and a fake FinBERT
    pipeline injected at the `core.sentiment_backend` factory level (no
    monkeypatching of news.py's own functions), `sentiment_finbert` must be
    populated with real (fake) scores and selected into `sentiment`.
    """
    import core.sentiment_backend as sb

    class _FakePipeline:
        def __call__(self, texts, **kwargs):
            return [[{"label": "positive", "score": 0.8}, {"label": "negative", "score": 0.1},
                     {"label": "neutral", "score": 0.1}] for _ in texts]

    monkeypatch.setattr(sb, "_finbert_factory", lambda: _FakePipeline())
    monkeypatch.setenv("SENTIMENT_BACKEND", "finbert")

    from research.strategy_hunt.data import news as N
    sess = _Session([
        {"news": [{"created_at": "2025-01-02T14:35:00Z", "headline": "Excellent beats", "source": "benzinga", "symbols": ["NVDA"]}],
         "next_page_token": None},
    ])
    df = N.fetch_news("NVDA", pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-03"), sess, "k", "s")

    assert df.iloc[0]["sentiment_finbert"] == pytest.approx(0.8 - 0.1)
    assert df.iloc[0]["sentiment"] == pytest.approx(0.8 - 0.1)


def test_fetch_news_sentiment_falls_back_to_vader_when_finbert_nan(monkeypatch):
    """Minor finding: when SENTIMENT_BACKEND=finbert but FinBERT is
    unavailable (factory raises, via the conftest default), `sentiment`
    must fall back to `sentiment_vader` rather than being poisoned with NaN.
    """
    monkeypatch.setenv("SENTIMENT_BACKEND", "finbert")

    from research.strategy_hunt.data import news as N
    sess = _Session([
        {"news": [{"created_at": "2025-01-02T14:35:00Z", "headline": "Excellent beats", "source": "benzinga", "symbols": ["NVDA"]}],
         "next_page_token": None},
    ])
    df = N.fetch_news("NVDA", pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-03"), sess, "k", "s")

    assert pd.isna(df.iloc[0]["sentiment_finbert"])
    assert not pd.isna(df.iloc[0]["sentiment"])
    assert df.iloc[0]["sentiment"] == pytest.approx(df.iloc[0]["sentiment_vader"])


def test_load_news_returns_extra_columns_when_present(tmp_path):
    from research.strategy_hunt.data import news as N
    df = pd.DataFrame({
        "created_at": [pd.Timestamp("2025-01-02 09:35")],
        "headline": ["A"],
        "source": ["x"],
        "sentiment": [0.1],
        "sentiment_vader": [0.1],
        "sentiment_finbert": [0.2],
    })
    p = N.news_path(tmp_path, "NVDA")
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    loaded = N.load_news(tmp_path, "NVDA")
    assert list(loaded.columns) == ["created_at", "headline", "source", "sentiment", "sentiment_vader", "sentiment_finbert"]


def test_load_news_legacy_file_without_extra_columns(tmp_path):
    from research.strategy_hunt.data import news as N
    df = pd.DataFrame({
        "created_at": [pd.Timestamp("2025-01-02 09:35")],
        "headline": ["A"],
        "source": ["x"],
        "sentiment": [0.1],
    })
    p = N.news_path(tmp_path, "NVDA")
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    loaded = N.load_news(tmp_path, "NVDA")
    assert list(loaded.columns) == ["created_at", "headline", "source", "sentiment"]


# v-playbook-desk-b1-2026-09-25: Task 7b `rescore_news` coverage -- re-scoring the pre-Task-7
# legacy cache (4 columns, no sentiment_vader/sentiment_finbert) with the configured backend.


def _legacy_cache(tmp_path, symbol="NVDA", n=3):
    from research.strategy_hunt.data import news as N
    df = pd.DataFrame({
        "created_at": [pd.Timestamp("2025-01-02 09:35") + pd.Timedelta(minutes=i) for i in range(n)],
        "headline": [f"Excellent beats {i}" for i in range(n)],
        "source": ["benzinga"] * n,
        "sentiment": [0.1] * n,
    })
    p = N.news_path(tmp_path, symbol)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    return p


def test_rescore_news_missing_cache_returns_0(tmp_path):
    from research.strategy_hunt.data.news import rescore_news
    assert rescore_news(tmp_path, "ZZZZ") == 0
    assert not (tmp_path / "news" / "ZZZZ.parquet").exists()


def test_rescore_news_legacy_cache_gains_both_columns_and_sentiment_follows_finbert(monkeypatch, tmp_path):
    import core.sentiment_backend as sb
    from research.strategy_hunt.data import news as N

    class _FakePipeline:
        def __init__(self):
            self.calls = 0

        def __call__(self, texts, **kwargs):
            self.calls += len(texts)
            return [[{"label": "positive", "score": 0.9}, {"label": "negative", "score": 0.05},
                     {"label": "neutral", "score": 0.05}] for _ in texts]

    fake = _FakePipeline()
    monkeypatch.setattr(sb, "_finbert_factory", lambda: fake)
    monkeypatch.setenv("SENTIMENT_BACKEND", "finbert")

    _legacy_cache(tmp_path, n=3)
    n = N.rescore_news(tmp_path, "NVDA")
    assert n == 3
    assert fake.calls == 3

    df = N.load_news(tmp_path, "NVDA")
    assert list(df.columns) == ["created_at", "headline", "source", "sentiment", "sentiment_vader", "sentiment_finbert"]
    assert np.allclose(df["sentiment_finbert"], 0.85)
    assert not df["sentiment_vader"].isna().any()
    assert (df["sentiment"] == df["sentiment_finbert"]).all()


def test_rescore_news_vader_backend_leaves_finbert_nan_and_sentiment_equals_vader(tmp_path):
    from research.strategy_hunt.data import news as N

    _legacy_cache(tmp_path, n=2)
    n = N.rescore_news(tmp_path, "NVDA", backend="vader")
    assert n == 2

    df = N.load_news(tmp_path, "NVDA")
    assert df["sentiment_finbert"].isna().all()
    assert (df["sentiment"] == df["sentiment_vader"]).all()


def test_rescore_news_does_not_rescore_already_scored_rows(monkeypatch, tmp_path):
    """Fully-scored cache (sentiment_vader + sentiment_finbert both non-NaN) -> the fake FinBERT
    pipeline must never be called, and no rows are reported as re-scored."""
    import core.sentiment_backend as sb
    from research.strategy_hunt.data import news as N

    class _CountingPipeline:
        def __init__(self):
            self.calls = 0

        def __call__(self, texts, **kwargs):
            self.calls += len(texts)
            return [[{"label": "positive", "score": 0.5}, {"label": "negative", "score": 0.1},
                     {"label": "neutral", "score": 0.4}] for _ in texts]

    fake = _CountingPipeline()
    monkeypatch.setattr(sb, "_finbert_factory", lambda: fake)
    monkeypatch.setenv("SENTIMENT_BACKEND", "finbert")

    df = pd.DataFrame({
        "created_at": [pd.Timestamp("2025-01-02 09:35")],
        "headline": ["Excellent beats"],
        "source": ["benzinga"],
        "sentiment": [0.4],
        "sentiment_vader": [0.1],
        "sentiment_finbert": [0.4],
    })
    p = N.news_path(tmp_path, "NVDA")
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)

    n = N.rescore_news(tmp_path, "NVDA")
    assert n == 0
    assert fake.calls == 0


def test_rescore_news_chunks_scoring_calls(monkeypatch, tmp_path):
    """RESCORE_CHUNK_SIZE headlines per score() call, not one giant call for the whole file."""
    from research.strategy_hunt.data import news as N

    monkeypatch.setattr(N, "RESCORE_CHUNK_SIZE", 2)
    calls = []

    def fake_score(headlines, backend=None):
        calls.append(len(headlines))
        return [0.1 for _ in headlines]

    monkeypatch.setattr(N, "score", fake_score)
    _legacy_cache(tmp_path, n=5)

    n = N.rescore_news(tmp_path, "NVDA", backend="vader")
    assert n == 5
    assert calls == [2, 2, 1]
