"""v-stub-predictor-2026-05-13: tests for the always-return-fixed-decision
predictor. This is what integration tests use when they need a
trained_predictor argument but don't want to load a real checkpoint.
"""
from __future__ import annotations

import pytest


class TestStubPredictor:
    def test_returns_configured_decision(self):
        from core.classifier import Side, SymbolFeatures
        from core.classifier.trained.stub_predictor import StubPredictor

        stub = StubPredictor(
            allowed_sides=frozenset({Side.LONG}),
            long_score=0.8, short_score=0.2,
        )
        d = stub.predict(SymbolFeatures(symbol="ANY", close=100.0))
        assert Side.LONG in d.allowed_sides
        assert Side.SHORT not in d.allowed_sides
        assert d.long_score == 0.8
        assert d.short_score == 0.2
        assert d.symbol == "ANY"

    def test_default_allows_both(self):
        from core.classifier import Side, SymbolFeatures
        from core.classifier.trained.stub_predictor import StubPredictor

        stub = StubPredictor()
        d = stub.predict(SymbolFeatures(symbol="X", close=50.0))
        assert Side.LONG in d.allowed_sides
        assert Side.SHORT in d.allowed_sides

    def test_satisfies_predictor_protocol(self):
        """The protocol defines ``.predict(features) -> SymbolSideDecision``.
        Duck-typed: any class with a matching signature passes."""
        from core.classifier.trained.stub_predictor import StubPredictor
        from core.classifier.trained.predictor_protocol import TrainedPredictor

        stub = StubPredictor()
        # isinstance check works for Protocol with @runtime_checkable
        assert isinstance(stub, TrainedPredictor)

    def test_composes_with_main_classifier(self):
        """StubPredictor plugs into core.classifier.classify() exactly
        like the real one will."""
        from core.classifier import classify, Side, SymbolFeatures
        from core.classifier.trained.stub_predictor import StubPredictor

        stub = StubPredictor(
            allowed_sides=frozenset({Side.LONG}),
            long_score=0.7, short_score=0.3,
        )

        # Uptrend features → rule allows LONG. Stub agrees → final LONG.
        f = SymbolFeatures(
            symbol="NVDA", close=110.0,
            sma_20_daily=108.0, sma_50_daily=105.0,
            higher_highs=2, up_vs_down_volume_5d=1.8,
            rs_vs_spy_5d=0.03,
        )
        d = classify(f, trained_predictor=stub)
        assert Side.LONG in d.allowed_sides
