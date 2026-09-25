"""Tests for core.sentiment_backend — pluggable VADER/FinBERT scoring.

v-playbook-desk-b1-2026-09-25: covers the shared `score()` entrypoint,
`available_backends()`, config-driven default backend selection, and the
FinBERT path via a fake pipeline injected through the module-level
`_finbert_factory` so no model download ever occurs in tests.
"""
import os
import threading
import time

import pytest


def _raising_finbert_factory():
    raise RuntimeError("FinBERT disabled in tests")


@pytest.fixture(autouse=True)
def _reset_sentiment_backend_state():
    """Reset module-level caches so tests don't leak state into each other.

    v-playbook-desk-b1-2026-09-25 (Task 7 fix round 1, Critical finding):
    default `_finbert_factory` to a stub that raises, not
    `_default_finbert_factory` — a test that reaches the real factory
    without injecting its own fake must fail loudly instead of silently
    downloading/loading the real model onto the GPU. Tests that exercise
    the FinBERT path inject their own fake factory before calling in.
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


class TestFinbertConcurrency:
    def test_lazy_finbert_load_is_thread_safe(self):
        """Important finding: lazy singleton init must be locked.

        8 threads race to trigger the first FinBERT call. A slow fake
        factory widens the race window; without a lock, multiple threads
        observe `_FINBERT_PIPELINE is None` before any of them finishes
        building it, so the factory runs more than once.
        """
        import core.sentiment_backend as sb

        load_count = {"n": 0}
        lock = threading.Lock()

        def slow_factory():
            with lock:
                load_count["n"] += 1
            time.sleep(0.05)
            return _FakeFinbertPipeline(_bullish_bearish_scores)

        sb._finbert_factory = slow_factory

        barrier = threading.Barrier(8)
        errors = []

        def worker():
            try:
                barrier.wait(timeout=5)
                sb.score(["beat"], backend="finbert")
            except Exception as exc:  # pragma: no cover - surfaced via errors list
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not errors
        assert load_count["n"] == 1


class TestFinbertPerBatchFallback:
    def test_batch_failure_falls_back_to_vader_for_that_batch_only(self):
        """Minor finding: a failing batch must not discard earlier batches.

        33 headlines -> batches of [32, 1] (FINBERT_BATCH_SIZE=32). The fake
        pipeline raises on the second call (the 1-headline batch), so the
        first 32 scores must still be FinBERT values and only the 33rd
        falls back to VADER.
        """
        import core.sentiment_backend as sb

        class _FailSecondBatchPipeline:
            def __init__(self):
                self.calls = 0

            def __call__(self, texts, **kwargs):
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError("simulated batch failure")
                out = []
                for _ in texts:
                    out.append([
                        {"label": "positive", "score": 0.9},
                        {"label": "negative", "score": 0.05},
                        {"label": "neutral", "score": 0.05},
                    ])
                return out

        fake = _FailSecondBatchPipeline()
        sb._finbert_factory = lambda: fake

        headlines = [f"beat #{i}" for i in range(33)]
        result = sb.score(headlines, backend="finbert")

        assert len(result) == 33
        assert fake.calls == 2
        finbert_value = pytest.approx(0.9 - 0.05)
        for v in result[:32]:
            assert v == finbert_value
        vader_expected = sb._score_vader([headlines[32]])[0]
        assert result[32] == pytest.approx(vader_expected)
        assert result[32] != finbert_value


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


# ---------------------------------------------------------------------------
# Model source resolution (local safetensors dir vs hub id)
# ---------------------------------------------------------------------------
# transformers >= 5 refuses pickle (.bin) checkpoints on torch < 2.6, and the
# hub copy of ProsusAI/finbert ships only .bin. The factory therefore prefers
# a locally prepared safetensors directory (scripts/prepare_finbert.py) and
# always requests safetensors loading.


def test_finbert_model_source_prefers_prepared_local_dir(tmp_path, monkeypatch):
    import core.sentiment_backend as sb

    local = tmp_path / "finbert"
    local.mkdir()
    (local / "config.json").write_text("{}")
    (local / "model.safetensors").write_bytes(b"")
    monkeypatch.setenv("SENTIMENT_FINBERT_PATH", str(local))

    assert sb._finbert_model_source() == str(local)


def test_finbert_model_source_falls_back_to_hub_id_when_dir_incomplete(tmp_path, monkeypatch):
    import core.sentiment_backend as sb

    local = tmp_path / "finbert"
    local.mkdir()
    (local / "config.json").write_text("{}")  # weights missing
    monkeypatch.setenv("SENTIMENT_FINBERT_PATH", str(local))

    assert sb._finbert_model_source() == sb.FINBERT_MODEL


def test_finbert_model_source_falls_back_to_hub_id_when_dir_missing(tmp_path, monkeypatch):
    import core.sentiment_backend as sb

    monkeypatch.setenv("SENTIMENT_FINBERT_PATH", str(tmp_path / "nope"))

    assert sb._finbert_model_source() == sb.FINBERT_MODEL


def test_config_sentiment_finbert_path_default_and_env(monkeypatch):
    from core.config import Config

    monkeypatch.delenv("SENTIMENT_FINBERT_PATH", raising=False)
    assert Config().SENTIMENT_FINBERT_PATH == "models/finbert"
    monkeypatch.setenv("SENTIMENT_FINBERT_PATH", "/x/y")
    assert Config().SENTIMENT_FINBERT_PATH == "/x/y"
