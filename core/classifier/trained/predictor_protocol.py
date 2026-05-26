"""Duck-typed contract for trained predictors.

v-predictor-protocol-2026-05-13. Any class that exposes
``.predict(features: SymbolFeatures) -> SymbolSideDecision`` satisfies
the protocol and can be passed to ``core.classifier.composer.classify``.

This keeps the trained-model implementation free to swap (FFN → TFT →
ensemble of N models) without touching the composer or the live
trading path.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from core.classifier.types import SymbolFeatures, SymbolSideDecision


@runtime_checkable
class TrainedPredictor(Protocol):
    """Contract for trained side-classifier predictors.

    Implementations should be stateless from the caller's perspective:
    no clock dependence, no side effects on stored state, no I/O
    beyond what's needed to evaluate the model. The composer caches
    decisions per symbol with a TTL; the predictor itself shouldn't
    second-guess that.
    """

    def predict(self, features: SymbolFeatures) -> SymbolSideDecision:
        """Return a side decision for one symbol-day's features."""
        ...
