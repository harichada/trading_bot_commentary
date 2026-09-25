"""Tests for core.sentiment_backend — pluggable VADER/FinBERT scoring.

v-playbook-desk-b1-2026-09-25: covers the shared `score()` entrypoint,
`available_backends()`, config-driven default backend selection, and the
FinBERT path via a fake pipeline injected through the module-level
`_finbert_factory` so no model download ever occurs in tests.
"""
import os

import pytest


@pytest.fixture(autouse=True)
def _reset_sentiment_backend_state():
    """Reset module-level caches so tests don't leak state into each other."""
    import core.sentiment_backend as sb

    sb._finbert_factory = sb._default_finbert_factory
    sb._FINBERT_PIPELINE = None
    sb._FINBERT_LOAD_FAILED = False
    sb._FINBERT_WARNED = False
    yield
    sb._finbert_factory = sb._default_finbert_factory
    sb._FINBERT_PIPELINE = None
    sb._FINBERT_LOAD_FAILED = False
    sb._FINBERT_WARNED = False


@pytest.fixture(autouse=True)
def _clear_sentiment_backend_env(monkeypatch):
    monkeypatch.delenv("SENTIMENT_BACKEND", raising=False)


class _FakeFinbertPipeline:
    """Deterministic fake standing in for the transformers pipeline.

    Called with a batch of texts; returns one list-of-label-dicts per text,
    mimicking `pipeline("text-classification", top_k=None)` batch output.
    """

    def __init__(self, label_for):
        self.label_for = label_for
        self.calls = []

    def __call__(self, texts, **kwargs):
        self.calls.append(list(texts))
        out = []
        for t in texts:
            p_pos, p_neg, p_neu = self.label_for(t)
            out.append([
                {"label": "positive", "score": p_pos},
                {"label": "negative", "score": p_neg},
                {"label": "neutral", "score": p_neu},
            ])
        return out


def _bullish_bearish_scores(text):
    if "beat" in text.lower():
        return (0.9, 0.05, 0.05)
    if "miss" in text.lower():
        return (0.05, 0.9, 0.05)
    return (0.2, 0.2, 0.6)


class TestScoreVader:
    def test_score_default_backend_is_vader(self):
        from core.sentiment_backend import score

        result = score(["Excellent earnings beat, strong guidance boost",
                         "Terrible earnings miss, cuts outlook, shares plunge"])
        assert result[0] > 0.2
        assert result[1] < -0.2

    def test_score_values_in_range(self):
        from core.sentiment_backend import score

        result = score(["ok", "great news beat", "bad news miss plunge", ""])
        for v in result:
            assert -1.0 <= v <= 1.0

    def test_score_empty_list_returns_empty(self):
        from core.sentiment_backend import score

        assert score([]) == []

    def test_score_explicit_vader_backend(self):
        from core.sentiment_backend import score

        result = score(["great beat"], backend="vader")
        assert result[0] > 0

    def test_score_unknown_backend_falls_back_to_vader(self):
        from core.sentiment_backend import score

        result = score(["great beat"], backend="not-a-real-backend")
        assert result[0] > 0


class TestScoreBackendSelection:
    def test_backend_none_uses_config_default(self, monkeypatch):
        monkeypatch.delenv("SENTIMENT_BACKEND", raising=False)
        from core.config import Config
        from core.sentiment_backend import score

        assert Config().SENTIMENT_BACKEND == "vader"
        result = score(["great beat"])
        assert result[0] > 0

    def test_backend_none_respects_env_override(self, monkeypatch):
        import core.sentiment_backend as sb

        fake = _FakeFinbertPipeline(_bullish_bearish_scores)
        sb._finbert_factory = lambda: fake
        monkeypatch.setenv("SENTIMENT_BACKEND", "finbert")

        result = sb.score(["Company beat estimates"])
        assert result[0] == pytest.approx(0.9 - 0.05)
        assert fake.calls  # pipeline was actually invoked


class TestFinbertBackend:
    def test_score_finbert_uses_fake_pipeline(self):
        import core.sentiment_backend as sb

        fake = _FakeFinbertPipeline(_bullish_bearish_scores)
        sb._finbert_factory = lambda: fake

        result = sb.score(["Company beat estimates", "Company miss estimates"], backend="finbert")
        assert result[0] == pytest.approx(0.9 - 0.05)
        assert result[1] == pytest.approx(0.05 - 0.9)

    def test_finbert_pipeline_loaded_lazily_and_cached(self):
        import core.sentiment_backend as sb

        load_count = {"n": 0}
        fake = _FakeFinbertPipeline(_bullish_bearish_scores)

        def factory():
            load_count["n"] += 1
            return fake

        sb._finbert_factory = factory
        assert load_count["n"] == 0

        sb.score(["beat"], backend="finbert")
        sb.score(["beat"], backend="finbert")
        assert load_count["n"] == 1  # loaded once, reused (singleton)

    def test_finbert_batches_large_input(self):
        import core.sentiment_backend as sb

        fake = _FakeFinbertPipeline(_bullish_bearish_scores)
        sb._finbert_factory = lambda: fake

        headlines = [f"beat #{i}" for i in range(70)]
        result = sb.score(headlines, backend="finbert")
        assert len(result) == 70
        assert all(v == pytest.approx(0.9 - 0.05) for v in result)
        # batch size 32 -> 70 headlines span 3 calls
        assert len(fake.calls) == 3
        assert [len(c) for c in fake.calls] == [32, 32, 6]

    def test_finbert_unavailable_falls_back_to_vader_with_one_time_warning(self, caplog):
        import core.sentiment_backend as sb

        def broken_factory():
            raise RuntimeError("no GPU / model not found")

        sb._finbert_factory = broken_factory

        with caplog.at_level("WARNING"):
            result = sb.score(["great beat"], backend="finbert")
        assert result[0] > 0  # vader fallback still scores correctly
        warnings = [r for r in caplog.records if "falling back to VADER" in r.message or "FinBERT" in r.message]
        assert len(warnings) >= 1

        caplog.clear()
        with caplog.at_level("WARNING"):
            sb.score(["another headline"], backend="finbert")
        # second failure should not re-warn (one-time warning)
        second_warnings = [r for r in caplog.records if "FinBERT" in r.message]
        assert len(second_warnings) == 0


class TestAvailableBackends:
    def test_available_backends_always_includes_vader(self):
        import core.sentiment_backend as sb

        sb._finbert_factory = lambda: (_ for _ in ()).throw(RuntimeError("unavailable"))
        backends = sb.available_backends()
        assert "vader" in backends
        assert "finbert" not in backends

    def test_available_backends_includes_finbert_when_loadable(self):
        import core.sentiment_backend as sb

        fake = _FakeFinbertPipeline(_bullish_bearish_scores)
        sb._finbert_factory = lambda: fake
        backends = sb.available_backends()
        assert backends == ("vader", "finbert")

    def test_available_backends_returns_tuple(self):
        import core.sentiment_backend as sb

        assert isinstance(sb.available_backends(), tuple)
