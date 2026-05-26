"""v-classifier-composer-2026-05-13: tests for the rule ∩ trained
composition contract.

The trained model is mocked here — we're testing composition logic,
not the model itself. The model gets its own tests when it lands.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def _features(**overrides):
    from core.classifier.types import SymbolFeatures

    base = dict(symbol="TEST", close=100.0)
    base.update(overrides)
    return SymbolFeatures(**base)


class TestComposerWithoutTrainedModel:
    def test_rule_only_when_predictor_none(self):
        from core.classifier import classify, Side

        f = _features(
            sma_20_daily=110.0, sma_50_daily=105.0,
            higher_highs=2, up_vs_down_volume_5d=1.8, rs_vs_spy_5d=0.03,
            intraday_trend_slope=0.4, news_sentiment_7d=0.2,
        )
        d = classify(f, trained_predictor=None)
        assert Side.LONG in d.allowed_sides
        assert Side.SHORT not in d.allowed_sides

    def test_hard_gate_short_circuits_trained_call(self):
        """If rule says ∅, the trained predictor must NOT be called —
        saves GPU work and keeps hard gates absolute."""
        from core.classifier import classify

        called = {"n": 0}

        class _Tracker:
            def predict(self, features):
                called["n"] += 1
                raise AssertionError("should not be called")

        f = _features(earnings_within_2d=True)
        d = classify(f, trained_predictor=_Tracker())
        assert d.is_forbidden()
        assert called["n"] == 0


class TestComposerIntersection:
    def test_trained_narrows_rule_to_subset(self):
        """Rule allows {LONG, SHORT}, trained allows {LONG} only ⇒
        final is {LONG}. Trained narrowed.

        Thresholds dropped to 0.4 to make a "both sides allowed" rule
        outcome reachable from neutral features. The composition logic
        under test is independent of threshold choice."""
        from core.classifier import classify, Side
        from core.classifier.types import SymbolSideDecision

        trained_decision = SymbolSideDecision(
            symbol="TEST",
            allowed_sides=frozenset({Side.LONG}),
            long_score=0.7, short_score=0.3,
            primary_reason="model_says_long",
            feature_dump={},
        )
        predictor = MagicMock()
        predictor.predict.return_value = trained_decision

        # Neutral features + low threshold → rule allows both sides
        f = _features(close=100.0, sma_20_daily=100.0, sma_50_daily=100.0)
        d = classify(f, trained_predictor=predictor,
                     long_threshold=0.4, short_threshold=0.4)
        assert Side.LONG in d.allowed_sides
        assert Side.SHORT not in d.allowed_sides

    def test_trained_cannot_add_sides_rule_removed(self):
        """Even if trained says SHORT is allowed, if rule's
        hard-side-gate (lower_lows>=3) removed LONG, the combined
        result still excludes LONG."""
        from core.classifier import classify, Side
        from core.classifier.types import SymbolSideDecision

        trained_decision = SymbolSideDecision(
            symbol="TEST",
            allowed_sides=frozenset({Side.LONG, Side.SHORT}),
            long_score=0.9, short_score=0.9,
            primary_reason="model_says_both",
            feature_dump={},
        )
        predictor = MagicMock()
        predictor.predict.return_value = trained_decision

        f = _features(lower_lows=3)   # rule removes LONG
        d = classify(f, trained_predictor=predictor)
        assert Side.LONG not in d.allowed_sides   # rule's removal stands

    def test_trained_blocks_all_when_rule_allowed_one(self):
        from core.classifier import classify, Side
        from core.classifier.types import SymbolSideDecision

        trained_decision = SymbolSideDecision(
            symbol="TEST",
            allowed_sides=frozenset(),   # model: don't trade
            long_score=0.3, short_score=0.3,
            primary_reason="model_no_edge",
            feature_dump={},
        )
        predictor = MagicMock()
        predictor.predict.return_value = trained_decision

        # Strong uptrend features → rule says LONG. Trained vetoes.
        f = _features(
            sma_20_daily=110.0, sma_50_daily=105.0,
            higher_highs=2, up_vs_down_volume_5d=1.8, rs_vs_spy_5d=0.03,
            intraday_trend_slope=0.4,
        )
        d = classify(f, trained_predictor=predictor)
        assert d.is_forbidden()


class TestComposerFailOpen:
    def test_predictor_exception_falls_back_to_rule(self):
        """A crashed/broken trained predictor must not block trading.
        Rule layer continues serving; latch flips True."""
        from unittest.mock import patch
        from core.classifier import classify
        import core.classifier.composer as composer_mod

        composer_mod._TRAINED_UNAVAILABLE_LOGGED = False

        class _Broken:
            def predict(self, features):
                raise RuntimeError("CUDA OOM or whatever")

        f = _features(sma_20_daily=110.0, sma_50_daily=105.0)
        with patch.object(composer_mod.logger, "warning") as mock_warn:
            d = classify(f, trained_predictor=_Broken())

        # Should be the rule-only decision (frozenset, may be empty
        # or non-empty depending on rule scoring).
        assert isinstance(d.allowed_sides, frozenset)
        # Warning fired exactly once for this session.
        assert mock_warn.call_count == 1
        # And the latch is now True.
        assert composer_mod._TRAINED_UNAVAILABLE_LOGGED is True

    def test_predictor_exception_warning_fires_once_per_session(self):
        """Warning fires once across many broken calls. Uses uptrend
        features so the rule layer doesn't short-circuit on ∅ before
        the broken predictor is consulted."""
        from unittest.mock import patch
        from core.classifier import classify
        import core.classifier.composer as composer_mod

        composer_mod._TRAINED_UNAVAILABLE_LOGGED = False

        class _Broken:
            def predict(self, features):
                raise RuntimeError("broken")

        # Uptrend features so rule allows LONG and the predictor is reached.
        f = _features(
            sma_20_daily=110.0, sma_50_daily=105.0,
            up_vs_down_volume_5d=1.8, rs_vs_spy_5d=0.03,
        )
        with patch.object(composer_mod.logger, "warning") as mock_warn:
            classify(f, trained_predictor=_Broken())
            classify(f, trained_predictor=_Broken())
            classify(f, trained_predictor=_Broken())

        assert mock_warn.call_count == 1, (
            f"expected exactly 1 warning across 3 broken calls, "
            f"got {mock_warn.call_count}"
        )
