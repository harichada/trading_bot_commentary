"""Side-classifier subsystem.

v-side-classifier-package-2026-05-13. Decides per-symbol which sides
(subset of {LONG, SHORT}) the bot is allowed to enter today.

Public surface (stable):
  - ``Side``, ``SymbolFeatures``, ``SymbolSideDecision`` from .types
  - ``classify_rule_based`` from .rule_based — rule layer only
  - ``classify`` from .composer — full rule ∩ trained composition

This package is **intentionally not imported by ``core/engine.py``**
on the live trading path until ``Config.USE_SIDE_CLASSIFIER`` is
flipped. See ``.claude/PLAN_symbol_side_classifier.md`` for design
and the rollout flavor B (per-strategy gradual) for activation order.
The invariant is enforced by ``v-classifier-stays-isolated-2026-05-13``
in ``tests/test_recent_fixes.py::TestClassifierStaysIsolated``.
"""
from core.classifier.composer import classify
from core.classifier.rule_based import classify_rule_based
from core.classifier.types import Side, SymbolFeatures, SymbolSideDecision

__all__ = [
    "Side",
    "SymbolFeatures",
    "SymbolSideDecision",
    "classify",
    "classify_rule_based",
]
