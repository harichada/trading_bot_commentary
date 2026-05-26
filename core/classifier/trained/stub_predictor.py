"""Always-returns-fixed-decision predictor.

v-stub-predictor-2026-05-13. Used by integration tests that need a
trained_predictor argument but don't want to load a real checkpoint.
Also useful as a safe fallback when a real model fails to load —
defaults to "allow both sides" so the composer ends up rule-only.

Not part of the public surface: imports its own protocol from the
sibling module to declare conformance.
"""
from __future__ import annotations

from typing import FrozenSet

from core.classifier.types import Side, SymbolFeatures, SymbolSideDecision


class StubPredictor:
    """Returns the same SymbolSideDecision for every input.

    Parameters
    ----------
    allowed_sides
        Sides the stub will report. Default: both LONG and SHORT
        (effectively a no-op in the composer, since the intersection
        won't narrow what the rule layer allowed).
    long_score, short_score
        Surfaced unchanged on every prediction.
    primary_reason
        Logged when this predictor is consulted; useful for trail
        debugging in audit lines.
    """

    def __init__(
        self,
        allowed_sides: FrozenSet[Side] = frozenset({Side.LONG, Side.SHORT}),
        long_score: float = 0.5,
        short_score: float = 0.5,
        primary_reason: str = "stub_predictor",
    ) -> None:
        self._allowed_sides = allowed_sides
        self._long_score = long_score
        self._short_score = short_score
        self._primary_reason = primary_reason

    def predict(self, features: SymbolFeatures) -> SymbolSideDecision:
        return SymbolSideDecision(
            symbol=features.symbol,
            allowed_sides=self._allowed_sides,
            long_score=self._long_score,
            short_score=self._short_score,
            primary_reason=self._primary_reason,
            feature_dump={"stub": True},
        )
