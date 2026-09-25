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
import os
import threading
from typing import Callable, List, Optional, Sequence

logger = logging.getLogger("TradingBot")

FINBERT_MODEL = "ProsusAI/finbert"
FINBERT_BATCH_SIZE = 32
FINBERT_MAX_LENGTH = 64

_VADER_ANALYZER = None
_FINBERT_PIPELINE = None
_FINBERT_LOAD_FAILED = False
_FINBERT_WARNED = False
# v-playbook-desk-b1-2026-09-25 (Task 7 fix round 1, Important finding):
# guards the lazy singleton init below against concurrent first-callers.
_FINBERT_LOCK = threading.Lock()


def _vader_analyzer():
    """Lazily-constructed VADER analyzer singleton."""
    global _VADER_ANALYZER
    if _VADER_ANALYZER is None:
        from nltk.sentiment import SentimentIntensityAnalyzer

        _VADER_ANALYZER = SentimentIntensityAnalyzer()
    return _VADER_ANALYZER


FINBERT_LOCAL_REQUIRED_FILES = ("config.json", "model.safetensors")


def _finbert_model_source() -> str:
    """Resolve where to load FinBERT from: a prepared local safetensors
    directory (`Config().SENTIMENT_FINBERT_PATH`, built by
    `scripts/prepare_finbert.py`) when it is complete, else the hub id.

    The hub copy of ProsusAI/finbert ships only a pickle checkpoint, which
    transformers >= 5 refuses on torch < 2.6 (CVE-2025-32434), so a local
    safetensors copy is the reliable path on this machine.
    """
    from core.config import Config

    local = Config().SENTIMENT_FINBERT_PATH
    try:
        if local and all(
            os.path.isfile(os.path.join(local, name))
            for name in FINBERT_LOCAL_REQUIRED_FILES
        ):
            return local
    except OSError as exc:  # unreadable path: fall through to the hub id
        logger.debug("sentiment_backend: cannot inspect %s (%s)", local, exc)
    return FINBERT_MODEL


def _default_finbert_factory():
    """Build the real FinBERT text-classification pipeline.

    Not called in tests: tests replace the module-level `_finbert_factory`
    reference with a fake before any code path can reach here, so this
    function never runs (and no model download happens) during the suite.
    """
    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    source = _finbert_model_source()
    logger.info("sentiment_backend: loading FinBERT from %s (device=%s)", source, device)
    return pipeline(
        "text-classification",
        model=source,
        top_k=None,
        device=device,
        truncation=True,
        max_length=FINBERT_MAX_LENGTH,
        model_kwargs={"use_safetensors": True},
    )


# Module-level factory hook. Tests monkeypatch this attribute directly
# (`core.sentiment_backend._finbert_factory = lambda: fake_pipeline`) to
# inject a fake pipeline and avoid any real model download.
_finbert_factory: Callable[[], object] = _default_finbert_factory


def _get_finbert_pipeline():
    """Return the cached FinBERT pipeline, loading it on first use.

    Returns None (without raising) if the pipeline cannot be built, and
    logs a one-time warning in that case.

    Double-checked locking: the common-case fast path (pipeline already
    built, or a previous load already failed) never touches the lock.
    Only a first caller that finds neither state set takes `_FINBERT_LOCK`,
    and re-checks both conditions inside it, so concurrent first-callers
    still build the pipeline exactly once instead of racing the factory.
    """
    global _FINBERT_PIPELINE, _FINBERT_LOAD_FAILED, _FINBERT_WARNED
    if _FINBERT_PIPELINE is not None:
        return _FINBERT_PIPELINE
    if _FINBERT_LOAD_FAILED:
        return None
    with _FINBERT_LOCK:
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
    """Score `headlines` with FinBERT, batch by batch.

    v-playbook-desk-b1-2026-09-25 (Task 7 fix round 1, Minor finding): each
    batch is scored independently. A batch that raises falls back to VADER
    for just that batch (with a one-time warning) instead of discarding
    every earlier batch's real FinBERT scores.
    """
    pipe = _get_finbert_pipeline()
    if pipe is None:
        return _score_vader(headlines)

    global _FINBERT_WARNED
    texts = [h or "" for h in headlines]
    scores: List[float] = []
    for i in range(0, len(texts), FINBERT_BATCH_SIZE):
        batch = texts[i:i + FINBERT_BATCH_SIZE]
        try:
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
            if not _FINBERT_WARNED:
                logger.warning(
                    "sentiment_backend: FinBERT scoring failed (%s); falling back to VADER",
                    exc,
                )
                _FINBERT_WARNED = True
            scores.extend(_score_vader(batch))
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
