"""Side-classifier composer.

v-classifier-composer-2026-05-13. The single entry point a caller
uses to ask "which sides are allowed on this symbol today?". It
intersects the rule-based decision with the trained-model decision
(when the trained model is loaded) and returns the combined result.

Composition discipline:
  - The rule-based layer's *hard gates* always win. If rule says
    ``∅``, the composer returns ``∅`` without even consulting the
    trained model — saves GPU work and keeps event-blackouts /
    halts / illiquidity absolute.
  - The trained layer can only *narrow* what the rule layer allowed.
    It cannot add sides the rule layer removed.

The trained model is opt-in. If unavailable (not installed, model
artifact missing, CUDA absent), the composer returns the rule
decision unchanged and logs at most one WARNING per process startup.
This is the fail-open path: a missing experimental model must not
take down trading.

This module is intentionally NOT imported by ``core/engine.py`` in
the live trading path. Integration happens only when
``Config.USE_SIDE_CLASSIFIER`` (and the per-strategy switches
``Config.SIDE_CLASSIFIER_NEWS_STRATEGY``,
``Config.SIDE_CLASSIFIER_MEAN_REVERSION``) are flipped to True. See
`.claude/PLAN_symbol_side_classifier.md` §10 and the rollout flavor
B (per-strategy gradual) chosen on 2026-05-13.
"""
from __future__ import annotations

import logging
from typing import Optional

from core.classifier.rule_based import classify_rule_based
from core.classifier.types import Side, SymbolFeatures, SymbolSideDecision

logger = logging.getLogger("TradingBot")

# Module-level flag so we warn about missing trained model only once
# per process, not on every signal evaluation.
_TRAINED_UNAVAILABLE_LOGGED = False


def classify(
    features: SymbolFeatures,
    trained_predictor: Optional[object] = None,
    *,
    long_threshold: float = 0.55,
    short_threshold: float = 0.55,
) -> SymbolSideDecision:
    """Compose rule + trained decisions for one symbol.

    Parameters
    ----------
    features
        SymbolFeatures for the symbol-day under evaluation.
    trained_predictor
        Optional object with a ``.predict(features) -> SymbolSideDecision``
        method. When ``None``, the composer is rule-only and returns
        the rule decision unchanged. The interface is duck-typed; the
        composer doesn't depend on the TFT module so the rule layer
        works in environments where torch is not installed.
    long_threshold, short_threshold
        Forwarded to the rule layer. Same defaults as the rule module.

    Returns
    -------
    SymbolSideDecision
        ``allowed_sides`` is the intersection of rule and trained.
        ``primary_reason`` describes the binding constraint.
    """
    rule = classify_rule_based(
        features,
        long_threshold=long_threshold,
        short_threshold=short_threshold,
    )

    # Hard-gate short-circuit: rule says ∅, no need to bother the model.
    if rule.is_forbidden():
        return rule

    if trained_predictor is None:
        return rule

    trained: Optional[SymbolSideDecision]
    try:
        trained = trained_predictor.predict(features)
    except Exception as exc:
        global _TRAINED_UNAVAILABLE_LOGGED
        if not _TRAINED_UNAVAILABLE_LOGGED:
            logger.warning(
                "side_classifier.composer: trained_predictor.predict raised "
                "(%s); falling back to rule-only decisions for this session.",
                exc,
            )
            _TRAINED_UNAVAILABLE_LOGGED = True
        trained = None

    if trained is None:
        return rule

    # Intersect: trained can only narrow what rule allowed.
    combined = rule.allowed_sides & trained.allowed_sides
    if combined == rule.allowed_sides:
        primary = f"rule:{rule.primary_reason}|trained:agrees"
    elif combined == frozenset():
        primary = f"rule_allowed={sorted(s.value for s in rule.allowed_sides)}|trained:blocked_all"
    else:
        narrowed = rule.allowed_sides - combined
        primary = (
            f"trained:narrowed_out={sorted(s.value for s in narrowed)}|"
            f"rule:{rule.primary_reason}"
        )

    feature_dump = {
        "rule_dump": rule.feature_dump,
        "rule_allowed": sorted(s.value for s in rule.allowed_sides),
        "trained_allowed": sorted(s.value for s in trained.allowed_sides),
        "trained_long_score": trained.long_score,
        "trained_short_score": trained.short_score,
    }

    return SymbolSideDecision(
        symbol=features.symbol,
        allowed_sides=combined,
        # Surface the more conservative score per side — the score
        # actually backing the gating decision.
        long_score=min(rule.long_score, trained.long_score),
        short_score=min(rule.short_score, trained.short_score),
        primary_reason=primary,
        feature_dump=feature_dump,
    )
