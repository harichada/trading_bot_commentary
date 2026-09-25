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
