"""v-side-classifier-types-2026-05-13: contract types for the side
classifier subsystem. Tests pin the immutability + field set so the
rule-based and trained-model implementations agree on shape."""
from __future__ import annotations

import pytest


class TestSide:
    def test_side_has_long_and_short(self):
        from core.classifier.types import Side

        assert Side.LONG.value == "long"
        assert Side.SHORT.value == "short"

    def test_side_is_hashable_for_frozenset(self):
        from core.classifier.types import Side

        s = frozenset({Side.LONG, Side.SHORT})
        assert Side.LONG in s
        assert Side.SHORT in s

    def test_no_third_side(self):
        """Two sides only. HOLD / NEITHER is represented by empty
        allowed_sides, not by a Side enum value."""
        from core.classifier.types import Side

        assert len(list(Side)) == 2


class TestSymbolSideDecision:
    def test_minimal_construction(self):
        from core.classifier.types import Side, SymbolSideDecision

        d = SymbolSideDecision(
            symbol="NVDA",
            allowed_sides=frozenset({Side.LONG}),
            long_score=0.72,
            short_score=0.15,
            primary_reason="uptrend confirmed",
            feature_dump={},
        )
        assert d.symbol == "NVDA"
        assert Side.LONG in d.allowed_sides
        assert Side.SHORT not in d.allowed_sides

    def test_empty_allowed_sides_is_valid(self):
        """∅ allowed_sides means 'do not trade this symbol today'.
        That's a normal, expected outcome — not an error state."""
        from core.classifier.types import SymbolSideDecision

        d = SymbolSideDecision(
            symbol="XYZ",
            allowed_sides=frozenset(),
            long_score=0.3,
            short_score=0.3,
            primary_reason="earnings_within_window",
            feature_dump={},
        )
        assert d.allowed_sides == frozenset()
        assert d.is_forbidden() is True
        assert d.allows_long() is False
        assert d.allows_short() is False

    def test_both_sides_allowed(self):
        """Range-bound symbols can be valid for both sides."""
        from core.classifier.types import Side, SymbolSideDecision

        d = SymbolSideDecision(
            symbol="ORCL",
            allowed_sides=frozenset({Side.LONG, Side.SHORT}),
            long_score=0.60,
            short_score=0.58,
            primary_reason="range_bound",
            feature_dump={},
        )
        assert d.allows_long() is True
        assert d.allows_short() is True
        assert d.is_forbidden() is False

    def test_frozen_dataclass_rejects_mutation(self):
        from core.classifier.types import Side, SymbolSideDecision

        d = SymbolSideDecision(
            symbol="NVDA",
            allowed_sides=frozenset({Side.LONG}),
            long_score=0.5,
            short_score=0.2,
            primary_reason="x",
            feature_dump={},
        )
        with pytest.raises((AttributeError, Exception)):
            d.symbol = "AAPL"   # frozen


class TestSymbolFeatures:
    def test_minimal_construction_with_defaults(self):
        """SymbolFeatures has many fields with sensible neutral defaults
        so tests can construct partial fixtures without listing every
        field."""
        from core.classifier.types import SymbolFeatures

        f = SymbolFeatures(symbol="NVDA", close=100.0)
        # Defaults that should produce a neutral (range-bound) decision
        assert f.symbol == "NVDA"
        assert f.close == 100.0
        # All other fields have defaults; this should not raise
        assert hasattr(f, "rsi_14")
        assert hasattr(f, "sma_20_daily")
        assert hasattr(f, "earnings_within_2d")
        assert f.earnings_within_2d is False   # hard-gate default: no event

    def test_hard_gate_fields_present(self):
        """Every hard-gate input listed in PLAN_symbol_side_classifier.md §5
        must be addressable on the dataclass."""
        from core.classifier.types import SymbolFeatures

        f = SymbolFeatures(symbol="X", close=10.0)
        for field in (
            "earnings_within_2d",
            "halt_today",
            "daily_atr_pct",
            "avg_daily_volume_20d",
            "hard_to_borrow",
            "higher_highs",
            "lower_lows",
            "news_sentiment_7d",
        ):
            assert hasattr(f, field), f"missing hard-gate field: {field}"
