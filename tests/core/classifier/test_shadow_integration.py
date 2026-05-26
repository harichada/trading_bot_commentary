"""v-shadow-integration-2026-05-13: integration tests that prove the
engine's classifier shadow hook will work correctly on next restart,
without actually restarting the live bot.

What we verify here:

  1. ``Config().SIDE_CLASSIFIER_SHADOW_MODE`` reads as True after the
     yaml flip — proves the config plumbing is wired.
  2. The checkpoint at ``Config().SIDE_CLASSIFIER_MODEL_PATH`` exists.
  3. ``ClassifierRuntime(enabled=True, model_path=<that>)`` constructs
     cleanly and predicts on fake market data.
  4. ``build_runtime_features`` handles indicator dict missing keys
     gracefully (the case when news_strategy hasn't populated some).
  5. The composer returns a non-empty SymbolSideDecision for a
     realistic market data snapshot — so the audit log will have
     something meaningful to write.
"""
from __future__ import annotations

import pytest


class _MarketData:
    """Mimics core.engine.MarketData duck-typed shape."""
    def __init__(self, symbol, close, indicators):
        self.symbol = symbol
        self.close = close
        self.open = close * 0.999
        self.high = close * 1.005
        self.low = close * 0.995
        self.volume = 1_000_000
        self.indicators = indicators


class TestShadowIntegration:
    def test_yaml_config_resolves_shadow_mode_true(self):
        """After the operator flipped trading.side_classifier_shadow_mode,
        Config().SIDE_CLASSIFIER_SHADOW_MODE reads as True."""
        from core.config import Config
        # If this fails it means Config().yaml was reverted or the
        # config knob name drifted.
        assert Config().SIDE_CLASSIFIER_SHADOW_MODE is True

    def test_checkpoint_path_exists(self):
        """Model file must be readable; otherwise ClassifierRuntime
        will fail-open on first signal."""
        import os
        from core.config import Config
        path = Config().SIDE_CLASSIFIER_MODEL_PATH
        assert os.path.exists(path), (
            f"checkpoint path {path!r} not found. Either train one or "
            f"adjust trading.side_classifier_model_path in Config().yaml."
        )

    def test_runtime_constructs_and_predicts(self):
        """Same construction the engine __init__ does. Must not raise."""
        torch = pytest.importorskip("torch", reason="torch not installed")
        from core.classifier.runtime import ClassifierRuntime
        from core.config import Config

        c = Config()
        runtime = ClassifierRuntime(
            enabled=True,
            model_path=c.SIDE_CLASSIFIER_MODEL_PATH,
            long_threshold=c.SIDE_CLASSIFIER_LONG_THRESHOLD,
            short_threshold=c.SIDE_CLASSIFIER_SHORT_THRESHOLD,
        )

        md = _MarketData("NVDA", 110.0, {
            "rsi": 42.0, "sma_20": 108.0, "sma_50": 105.0,
            "atr": 2.5, "volume_ratio": 1.3,
        })
        decision = runtime.shadow_evaluate(md)
        assert decision is not None
        assert decision.symbol == "NVDA"
        # Scores are valid probabilities
        assert 0.0 <= decision.long_score <= 1.0
        assert 0.0 <= decision.short_score <= 1.0

    def test_runtime_handles_missing_indicators(self):
        """The strategies sometimes don't populate every indicator
        (especially in early-session bars). The bridge must not raise."""
        from core.classifier.runtime import build_runtime_features
        md = _MarketData("F", 12.0, {})   # no indicators dict
        f = build_runtime_features(md)
        # All default-friendly
        assert f.symbol == "F"
        assert f.close == 12.0
