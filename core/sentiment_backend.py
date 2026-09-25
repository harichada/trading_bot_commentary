"""Pluggable sentiment scoring backend shared by research and live.

v-playbook-desk-b1-2026-09-25 (Task 7, docs/superpowers/specs/2026-09-25-
playbook-desk-design.md §7.5): `score()` computes a headline sentiment in
[-1, 1] using either VADER (default; CPU; always available) or FinBERT
(`ProsusAI/finbert` on GPU when available). Both the live NewsBus
(`core/loops/news_loop.py`) and the research news cache
(`research/strategy_hunt/data/news.py` via `research/strategy_hunt/data/
sentiment.py`) call into this module so they always agree on scoring for
a given `SENTIMENT_BACKEND`.

FinBERT is loaded lazily as a module-level singleton behind
`_finbert_factory`, which tests monkeypatch with a fake pipeline so no
model is ever downloaded during the test suite. If transformers/torch are
missing, no GPU/model is available, or loading fails for any other
reason, FinBERT scoring falls back to VADER and logs a one-time warning.
"""
from __future__ import annotations

import logging
from typing import Callable, List, Optional, Sequence

logger = logging.getLogger("TradingBot")

FINBERT_MODEL = "ProsusAI/finbert"
FINBERT_BATCH_SIZE = 32
FINBERT_MAX_LENGTH = 64

_VADER_ANALYZER = None
_FINBERT_PIPELINE = None
_FINBERT_LOAD_FAILED = False
_FINBERT_WARNED = False


def _vader_analyzer():
    """Lazily-constructed VADER analyzer singleton."""
    global _VADER_ANALYZER
    if _VADER_ANALYZER is None:
        from nltk.sentiment import SentimentIntensityAnalyzer

        _VADER_ANALYZER = SentimentIntensityAnalyzer()
    return _VADER_ANALYZER


def _default_finbert_factory():
    """Build the real FinBERT text-classification pipeline.

    Not called in tests: tests replace the module-level `_finbert_factory`
    reference with a fake before any code path can reach here, so this
    function never runs (and no model download happens) during the suite.
    """
    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "text-classification",
        model=FINBERT_MODEL,
        top_k=None,
        device=device,
        truncation=True,
        max_length=FINBERT_MAX_LENGTH,
    )


# Module-level factory hook. Tests monkeypatch this attribute directly
# (`core.sentiment_backend._finbert_factory = lambda: fake_pipeline`) to
# inject a fake pipeline and avoid any real model download.
_finbert_factory: Callable[[], object] = _default_finbert_factory


def _get_finbert_pipeline():
    """Return the cached FinBERT pipeline, loading it on first use.

    Returns None (without raising) if the pipeline cannot be built, and
    logs a one-time warning in that case.
    """
    global _FINBERT_PIPELINE, _FINBERT_LOAD_FAILED, _FINBERT_WARNED
    if _FINBERT_PIPELINE is not None:
        return _FINBERT_PIPELINE
    if _FINBERT_LOAD_FAILED:
        return None
    try:
        _FINBERT_PIPELINE = _finbert_factory()
        return _FINBERT_PIPELINE
    except Exception as exc:
        _FINBERT_LOAD_FAILED = True
        if not _FINBERT_WARNED:
            logger.warning(
                "sentiment_backend: FinBERT unavailable (%s); falling back to VADER",
                exc,
            )
            _FINBERT_WARNED = True
        return None


def _score_vader(headlines: Sequence[str]) -> List[float]:
    analyzer = _vader_analyzer()
    return [float(analyzer.polarity_scores(h or "")["compound"]) for h in headlines]


def _finbert_probs(label_scores: Sequence[dict]) -> tuple:
    probs = {str(d.get("label", "")).lower(): float(d.get("score", 0.0)) for d in label_scores}
    return probs.get("positive", 0.0), probs.get("negative", 0.0)


def _score_finbert(headlines: Sequence[str]) -> List[float]:
    pipe = _get_finbert_pipeline()
    if pipe is None:
        return _score_vader(headlines)

    texts = [h or "" for h in headlines]
    scores: List[float] = []
    try:
        for i in range(0, len(texts), FINBERT_BATCH_SIZE):
            batch = texts[i:i + FINBERT_BATCH_SIZE]
            results = pipe(
                batch,
                batch_size=FINBERT_BATCH_SIZE,
                truncation=True,
                max_length=FINBERT_MAX_LENGTH,
            )
            for label_scores in results:
                p_pos, p_neg = _finbert_probs(label_scores)
                scores.append(float(p_pos - p_neg))
    except Exception as exc:
        global _FINBERT_WARNED
        if not _FINBERT_WARNED:
            logger.warning(
                "sentiment_backend: FinBERT scoring failed (%s); falling back to VADER",
                exc,
            )
            _FINBERT_WARNED = True
        return _score_vader(headlines)
    return scores


_BACKENDS: dict = {
    "vader": _score_vader,
    "finbert": _score_finbert,
}


def available_backends() -> tuple:
    """Backends usable in this process right now.

    Always includes "vader". Includes "finbert" only if the FinBERT
    pipeline can actually be loaded (attempting the lazy load, which is
    cached for subsequent calls).
    """
    backends = ["vader"]
    if _get_finbert_pipeline() is not None:
        backends.append("finbert")
    return tuple(backends)


def score(headlines: List[str], backend: Optional[str] = None) -> List[float]:
    """Score `headlines` in [-1, 1] using the given (or configured) backend.

    `backend=None` resolves to `Config().SENTIMENT_BACKEND`. An unknown
    backend name falls back to VADER.
    """
    if not headlines:
        return []
    if backend is None:
        from core.config import Config

        backend = Config().SENTIMENT_BACKEND
    fn = _BACKENDS.get(backend, _score_vader)
    return fn(list(headlines))
