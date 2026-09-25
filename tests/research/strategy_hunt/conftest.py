"""Shared fixtures for tests/research/strategy_hunt.

v-playbook-desk-b1-2026-09-25 (Task 7 fix round 1, Critical finding): every
test under this package must run with `core.sentiment_backend._finbert_factory`
replaced by a stub that raises, so a test that (directly or via
`research/strategy_hunt/data/news.py`) reaches the real FinBERT lazy-load path
gets a clear `RuntimeError` instead of silently downloading/loading a 440 MB
model onto the GPU. Tests that need FinBERT behavior inject their own fake
factory, which overrides this default for the duration of that test.
"""
import pytest


def _raising_finbert_factory():
    raise RuntimeError("FinBERT disabled in tests")


@pytest.fixture(autouse=True)
def _no_real_finbert_in_strategy_hunt_tests():
    """Default `_finbert_factory` to a raising stub; reset module caches."""
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
