"""v-position-fsm-2026-04-30 (Phase 1): position lifecycle state machine.

Single source of truth for:
  (a) the legal states a position can be in
  (b) the legal transitions between them
  (c) the SET of exit rules each state is allowed to evaluate

Anti-pattern this replaces: the engine had 6 exit rules (stop_loss,
take_profit, breakeven_lift, scale_out_1R, trailing_stop_atr,
proactive_exit) racing in the same `for` loop with no priority and no
mutual exclusion. RIOT 2026-04-30 was a textbook failure: stop_loss was
already breached, but proactive_exit fired LATER on a different rule
and exited the trade at -2.74R instead of the planned -1R. Six rules,
one position, no orchestration → cost $260 of unintended slippage on
that single trade.

This module models the position as an FSM. At any instant the position
is in exactly one state, and only that state's `allowed_exits` are
permitted to evaluate. Transitions are explicit and one-way (no
LIVE → OPENING regression). EXITING is a sink — once we've decided to
close, no other rule can second-guess.

Read this file as the spec. Behaviour lives in Position.transition_to()
and the engine's state-aware dispatcher.
"""
from __future__ import annotations

from enum import Enum
from typing import FrozenSet


class PositionState(str, Enum):
    """Lifecycle of a single open position.

    Stored as the str value (not the enum object) on disk via _save_state
    so the JSON is human-readable and forward-compatible if the enum
    grows. Loader resolves str back to enum.
    """
    OPENING       = "opening"        # entry order placed, fill not confirmed
    LIVE          = "live"           # filled, no profit-protective ratchet yet
    AT_BREAKEVEN  = "at_breakeven"   # stop has been ratcheted to entry (+0.5R reached)
    AT_1R         = "at_1r"          # +1R reached, scale-out partial done
    TRAILING      = "trailing"       # remainder running on ATR trail
    EXITING       = "exiting"        # close order in flight, terminal pre-CLOSED
    CLOSED        = "closed"         # fully realized, removed from tracking
    ZOMBIE        = "zombie"         # broker close failed; manual intervention needed


# Exit rules the engine knows how to evaluate. Centralised here so the
# state-table below can reference them by stable string keys without
# importing the engine's procedural code.
class ExitRule(str, Enum):
    HARD_STOP            = "hard_stop"            # the original entry stop
    TAKE_PROFIT          = "take_profit"          # the original entry target
    BREAKEVEN_LIFT       = "breakeven_lift"       # ratchet stop to entry at +0.5R
    SCALE_OUT_1R         = "scale_out_1r"         # partial close at +1R
    TRAILING_STOP_ATR    = "trailing_stop_atr"    # ATR trail (after 1R)
    PROACTIVE_EXIT       = "proactive_exit"       # indicator-flip early bail
    THESIS_REVALIDATE    = "thesis_revalidate"    # news verifier re-check
    BROKER_FILL_TIMEOUT  = "broker_fill_timeout"  # OPENING took too long, abort
    # v-time-stop-2026-05-19: TIME_STOP exit rule for OversoldBounceV2
    TIME_STOP            = "time_stop"            # bail if thesis hasn't played out in time


# ──────────────────────────────────────────────────────────────────────
# THE STATE-EXIT TABLE — the architectural contract.
#
# Each state maps to the EXACT set of exit rules permitted to evaluate
# while the position is in that state. Anything not listed is physically
# disconnected — the dispatcher will not even call it. This is the
# user's "explicit guarding" requirement: not `if state == X: rule()`
# scattered through the loop, but a single immutable map consulted by
# the dispatcher.
# ──────────────────────────────────────────────────────────────────────
ALLOWED_EXITS: dict[PositionState, FrozenSet[ExitRule]] = {
    PositionState.OPENING: frozenset({
        # Pre-fill: only watch for fill timeout. No P&L-based logic;
        # there's no fill price yet to compute against.
        ExitRule.BROKER_FILL_TIMEOUT,
    }),
    PositionState.LIVE: frozenset({
        # Fresh trade, hasn't moved in our favour yet.
        # Hard stop and take-profit are always-on safety belts.
        # Breakeven_lift is the next state-transition trigger.
        # Proactive_exit is allowed because thesis can break early.
        # Thesis_revalidate is allowed (separate from proactive — it's
        # the news-verifier path, slow cadence).
        ExitRule.HARD_STOP,
        ExitRule.TAKE_PROFIT,
        ExitRule.BREAKEVEN_LIFT,
        ExitRule.PROACTIVE_EXIT,
        ExitRule.THESIS_REVALIDATE,
        # v-time-stop-2026-05-19: TIME_STOP exit rule for OversoldBounceV2
        ExitRule.TIME_STOP,
    }),
    PositionState.AT_BREAKEVEN: frozenset({
        # Stop is now at entry — protected from a full -1R loss.
        # Original take-profit is still valid.
        # Scale_out_1R is the next ratchet up.
        # Proactive_exit STAYS on because a thesis break here still
        # warrants taking the small win/breakeven instead of waiting.
        # v-trail-from-breakeven-2026-04-30: TRAILING_STOP_ATR added.
        # Without this, a +0.80R favorable move (e.g. MRVL today) could
        # round-trip to entry and exit flat — even though the trail
        # would have locked +0.40R+ of it. The trail's own gate
        # (TRAIL_ACTIVATION_ATR_MULT) decides WHEN the trail updates,
        # so this just removes the FSM-level block, doesn't make the
        # trail more aggressive.
        ExitRule.HARD_STOP,           # stop is at entry now (set by breakeven)
        ExitRule.TAKE_PROFIT,
        ExitRule.SCALE_OUT_1R,
        ExitRule.TRAILING_STOP_ATR,   # ← Bug 2 fix
        ExitRule.PROACTIVE_EXIT,
        ExitRule.THESIS_REVALIDATE,
        # v-time-stop-2026-05-19: TIME_STOP exit rule for OversoldBounceV2
        ExitRule.TIME_STOP,
    }),
    PositionState.AT_1R: frozenset({
        # Half is out, half remains. The remainder is now riding on
        # the trailing stop. The "proactive_exit" rule's premise
        # (catch losers early) no longer applies — we're in profit.
        # Same with thesis_revalidate: this is a winner running.
        ExitRule.HARD_STOP,            # remains valid (stop was raised)
        ExitRule.TAKE_PROFIT,
        ExitRule.TRAILING_STOP_ATR,
    }),
    PositionState.TRAILING: frozenset({
        # Pure trailing-stop management. No more discretionary exits;
        # the trail IS the exit policy. PROACTIVE_EXIT IS NOT IN THIS
        # SET — that was the architectural fix the user asked for.
        ExitRule.TRAILING_STOP_ATR,
        ExitRule.TAKE_PROFIT,          # if the original target is still meaningful
    }),
    PositionState.EXITING: frozenset(),  # terminal pre-CLOSED — no rules fire
    PositionState.CLOSED: frozenset(),   # gone from tracking; here for completeness
    PositionState.ZOMBIE: frozenset(),   # operator must intervene; engine stays hands-off
}


# Legal transitions. Used by Position.transition_to() to reject illegal
# moves at runtime (e.g. LIVE → OPENING is impossible; AT_1R → LIVE
# would un-do a scale-out). Keeps the FSM honest under refactor.
LEGAL_TRANSITIONS: dict[PositionState, FrozenSet[PositionState]] = {
    PositionState.OPENING:      frozenset({PositionState.LIVE, PositionState.EXITING, PositionState.ZOMBIE}),
    PositionState.LIVE:         frozenset({PositionState.AT_BREAKEVEN, PositionState.EXITING, PositionState.ZOMBIE}),
    PositionState.AT_BREAKEVEN: frozenset({PositionState.AT_1R, PositionState.EXITING, PositionState.ZOMBIE}),
    PositionState.AT_1R:        frozenset({PositionState.TRAILING, PositionState.EXITING, PositionState.ZOMBIE}),
    PositionState.TRAILING:     frozenset({PositionState.EXITING, PositionState.ZOMBIE}),
    PositionState.EXITING:      frozenset({PositionState.CLOSED, PositionState.ZOMBIE}),
    PositionState.CLOSED:       frozenset(),
    PositionState.ZOMBIE:       frozenset({PositionState.CLOSED}),  # operator-resolved → CLOSED
}


def is_exit_rule_allowed(state: PositionState, rule: ExitRule) -> bool:
    """Single dispatcher predicate. The engine calls this before every
    exit-rule evaluation. The rule is physically not invoked if the
    answer is False — no `if` clause inside the rule, no shared mutable
    state to guard. Clean dispatcher pattern."""
    return rule in ALLOWED_EXITS.get(state, frozenset())


def can_transition(from_state: PositionState, to_state: PositionState) -> bool:
    """Legality check. Used by Position.transition_to() to fail loudly
    on illegal transitions instead of silently corrupting state."""
    return to_state in LEGAL_TRANSITIONS.get(from_state, frozenset())


# ──────────────────────────────────────────────────────────────────────
# Atomic transition primitive.
#
# Architectural requirement (user-stated): "the transition from LIVE to
# EXITING must be atomic — no two async tasks can trigger an exit
# simultaneously."
#
# We achieve this with an asyncio.Lock held PER POSITION (not a global
# lock — that would serialise all close attempts across all symbols
# unnecessarily). The lock guards the CHECK-AND-SWAP of the state
# field. Without the lock, the proactive-exit task and the trailing-
# stop task could both read state==LIVE, both decide to close, and both
# call broker.close() — double-fill, double-pnl, double-everything.
#
# The lock is acquired inside try_transition; if the position is
# already in the target state (or further along), the call returns
# False and the caller's exit logic short-circuits. This is the
# "compare-and-swap" pattern in async form.
# ──────────────────────────────────────────────────────────────────────

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.models import Position

_logger = logging.getLogger("TradingBot")


async def try_transition(
    position: "Position",
    target: PositionState,
    reason: str,
    *,
    audit_fn=None,
) -> bool:
    """Atomically attempt a state transition on `position`.

    Returns True iff the transition happened (caller may now perform
    the side-effect — e.g. submit the close order). Returns False if:
      - target is the current state already (idempotent no-op)
      - transition is illegal per LEGAL_TRANSITIONS
      - another task already advanced past `target`
    On False the caller MUST short-circuit; their work has already been
    done (or was never legal).

    `audit_fn(component, symbol, action, reason, **kv)` is the engine's
    structured-audit hook. We log every transition attempt — accepted,
    rejected, illegal — so the post-mortem trail is complete.
    """
    # Acquire per-position lock. The lock is created lazily by the engine
    # at the point a Position is added to tracking (see engine.py:
    # _ensure_position_lock). Defensive fallback creates one here if
    # somehow a Position arrives without one — better than crashing on
    # a None.acquire().
    if position._state_lock is None:
        try:
            position._state_lock = asyncio.Lock()
        except RuntimeError:
            # No running loop yet — caller is mis-using us. Refuse.
            _logger.error(
                "try_transition called outside running loop for %s",
                position.symbol,
            )
            return False

    async with position._state_lock:
        try:
            current = PositionState(position.state)
        except ValueError:
            # Unknown state on the position — treat as ZOMBIE (operator
            # must inspect). This protects against partial-restore from
            # a corrupted state file.
            _logger.error(
                "position_fsm: unknown state %r on %s — promoting to ZOMBIE",
                position.state, position.symbol,
            )
            position.state = PositionState.ZOMBIE.value
            position.zombie_reason = f"unknown_state_{position.state!r}"
            position.state_changed_at = datetime.now(timezone.utc).isoformat()
            return False

        # Idempotency: already in target state → no-op success-False.
        if current == target:
            return False

        if not can_transition(current, target):
            if audit_fn is not None:
                audit_fn(
                    "position_fsm", position.symbol, "transition_rejected",
                    "illegal_transition",
                    from_state=current.value, to_state=target.value,
                    reason_attempted=reason,
                )
            _logger.warning(
                "position_fsm: illegal transition %s → %s on %s (reason=%s)",
                current.value, target.value, position.symbol, reason,
            )
            return False

        # Commit the transition.
        position.state = target.value
        position.state_reason = reason
        position.state_changed_at = datetime.now(timezone.utc).isoformat()
        if target == PositionState.EXITING:
            position.exiting_started_at = position.state_changed_at

        if audit_fn is not None:
            audit_fn(
                "position_fsm", position.symbol, "transition",
                reason,
                from_state=current.value, to_state=target.value,
            )
        _logger.info(
            "position_fsm: %s %s → %s (%s)",
            position.symbol, current.value, target.value, reason,
        )
        return True


def infer_state_from_legacy_flags(
    persisted_state: Optional[str],
    breakeven_lifted: bool,
    scaled_out: bool,
    trailing_stop: Optional[float],
) -> PositionState:
    """v-fsm-state-migration-2026-04-30: legacy positions persisted
    before the FSM existed have no `state` field. Worse, positions that
    DID have `breakeven_lift` or `scale_out` fire on a pre-FSM run come
    back with the side-effect flags set (breakeven_lifted=True,
    scaled_out=True, trailing_stop=$X) but state defaults to "live" —
    the FSM and the actual position are now out of sync.

    On load we INFER the correct FSM state from the side-effect flags,
    in priority order: trailing_stop set > scaled_out > breakeven_lifted.
    Persisted `state` field, if present and consistent, takes precedence.
    """
    # If persisted state is consistent with flags, trust it.
    if persisted_state:
        try:
            ps = PositionState(persisted_state)
        except ValueError:
            ps = None
        if ps is not None and ps not in (PositionState.LIVE, PositionState.OPENING):
            return ps  # already a meaningful FSM state, keep it
    # Otherwise infer from flags. Highest progress flag wins.
    if scaled_out and trailing_stop is not None:
        return PositionState.TRAILING
    if scaled_out:
        return PositionState.AT_1R
    if breakeven_lifted:
        return PositionState.AT_BREAKEVEN
    return PositionState.LIVE


def reset_to_live_after_zombie(position: "Position", reason: str) -> None:
    """Operator escape hatch — only used by the dashboard's manual
    'clear zombie' action. Forces ZOMBIE → CLOSED. Not exposed to the
    automatic engine path."""
    position.state = PositionState.CLOSED.value
    position.state_reason = f"operator_clear: {reason}"
    position.state_changed_at = datetime.now(timezone.utc).isoformat()
    position.zombie_reason = None
