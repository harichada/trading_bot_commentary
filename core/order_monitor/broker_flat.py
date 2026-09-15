"""v-broker-flat-detection-2026-09-10: Broker-flat detection and handling.

This module handles the detection of positions that have been closed externally
at the broker (e.g., user manual close, broker stop filled without our knowledge).

Key scenarios handled:
  - Oversold/overbought bracket rejections indicating position already closed
  - Quantity sync when local qty diverges from broker qty
  - Proper FSM transition to CLOSED state for external closes

v-broker-leg-authority-2026-09-14: Extended with broker-leg authority check.
When Config.ENABLE_BROKER_LEG_AUTHORITY is True (default), the software stop
path checks if the broker's OCO/stop leg is WORKING or FILLED before entering
the supervised close confirmation flow. If broker is already protecting (or
has already closed) the position, software skips confirmation to prevent
FTFT-style races.

HANDS_OFF guards are NOT in this module — they're applied at the filtering
level in monitor.py before any of these functions are called.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from core.engine import TradingEngineWithCommentary
    from core.models import Position

logger = logging.getLogger("TradingBot")


def is_broker_flat_rejection(rejection_reason: str) -> bool:
    """v-broker-flat-detection-2026-09-10: detect oversold/overbought rejection.

    Returns True if the rejection reason indicates the position is already
    flat at the broker (the shares are gone, so any sell/cover order is
    invalid).

    Known Schwab rejection patterns:
      - "oversold position" → selling more than owned (long already closed)
      - "overbought position" → covering more than shorted (short already closed)
      - variations with "oversold", "overbought", "insufficient shares"
    """
    if not rejection_reason:
        return False
    reason_lower = rejection_reason.lower()
    flat_indicators = [
        'oversold',
        'overbought',
        'insufficient shares',
        'insufficient position',
        'no position',
        'position not found',
    ]
    return any(indicator in reason_lower for indicator in flat_indicators)


async def check_broker_position_qty(
    engine: "TradingEngineWithCommentary",
    symbol: str,
) -> int:
    """v-broker-flat-detection-2026-09-10: query Schwab for actual position qty.

    Returns the absolute quantity at the broker for `symbol`, or 0 if
    the position doesn't exist (broker-flat). This is the authoritative
    source when handling bracket rejections — if Schwab says we're flat,
    we must stop trying to manage/re-bracket that position.

    Returns 0 on any error (fail-safe: if we can't verify, assume flat
    to prevent infinite re_bracket loops).
    """
    if not engine.schwab_client or not engine.account_id:
        return 0
    try:
        schwab_positions = await engine.get_schwab_positions()
        for pos_data in schwab_positions:
            if pos_data.get('symbol') == symbol:
                return abs(pos_data.get('quantity', 0))
        return 0  # symbol not in broker positions → flat
    except Exception as exc:
        logger.warning(
            "broker_position_check_error symbol=%s err=%s",
            symbol, exc,
        )
        return 0  # fail-safe: assume flat on error


async def handle_broker_flat_detected(
    engine: "TradingEngineWithCommentary",
    position: "Position",
    reason: str,
) -> None:
    """v-broker-flat-detection-2026-09-10: handle external close detection.

    Called when we discover the broker shows qty=0 for a position we
    thought was still open. This is the canonical "user closed at Schwab"
    or "broker stop filled externally" scenario.

    v-broker-leg-authority-2026-09-14: added ENABLE_GHOST_FLATTEN_AFTER_BROKER_FLAT
    flag control. When False (default), ghost positions are logged/alerted but
    NOT auto-removed — operator must manually reconcile. When True, position
    is auto-removed from tracking.

    Actions (always):
      1. Clear managed_by_bot → stop all exit management
      2. Cancel any working exit orders for this symbol
      3. Clear bracket IDs
      4. Audit for post-mortem

    Actions (when ENABLE_GHOST_FLATTEN_AFTER_BROKER_FLAT=True OR explicit close):
      5. Mark position as CLOSED (external close)
      6. Remove from active tracking

    This breaks the infinite re_bracket loop that occurs when:
      - Bot thinks it has a position (ghost qty in self.positions)
      - Schwab rejects bracket for oversold/overbought (shares gone)
      - _handle_bracket_rejected calls _re_bracket_position
      - New bracket rejected → repeat forever
    """
    from core.commentary import TradingCommentary
    from core.config import Config
    from core.models import CommentaryType, clear_bracket_ids
    from core.position_state import PositionState, try_transition

    symbol = position.symbol
    cfg = Config()
    ghost_flatten_enabled = cfg.ENABLE_GHOST_FLATTEN_AFTER_BROKER_FLAT

    engine._audit(
        "order_monitor", symbol, "broker_flat_detected",
        reason,
        local_qty=position.quantity,
        managed_by_bot=getattr(position, 'managed_by_bot', False),
        ghost_flatten_enabled=ghost_flatten_enabled,
    )

    # Clear managed_by_bot to stop any other exit paths
    position.managed_by_bot = False

    # Cancel any working exit orders for this symbol
    try:
        await engine._cancel_existing_orders(symbol)
    except Exception as exc:
        logger.warning(
            "broker_flat_cancel_orders_error symbol=%s err=%s",
            symbol, exc,
        )

    # Clear bracket IDs (already rejected/cancelled anyway)
    clear_bracket_ids(position)

    # v-broker-leg-authority-2026-09-14: respect ENABLE_GHOST_FLATTEN_AFTER_BROKER_FLAT
    # When disabled, we alert but don't auto-remove the ghost position.
    # Reasons to keep the ghost: operator may want to investigate desync,
    # or the broker API may have returned stale data.
    if ghost_flatten_enabled:
        # Transition to CLOSED state
        await try_transition(
            position,
            PositionState.CLOSED,
            f"broker_flat_{reason}",
            audit_fn=engine._audit,
        )

        # Remove from active tracking (will be picked up by next sync)
        engine.positions.pop(symbol, None)

        engine.commentary.add_commentary(TradingCommentary(
            timestamp=datetime.now(),
            type=CommentaryType.WARNING,
            symbol=symbol,
            title=f"🔄 External Close Detected",
            message=(
                f"Position {symbol} was closed externally (broker shows flat). "
                f"Reason: {reason}. Bot will stop managing this position."
            ),
            data={'reason': reason, 'auto_flattened': True},
            importance=9
        ))

        logger.info(
            "broker_flat_handled symbol=%s reason=%s — position removed from tracking",
            symbol, reason,
        )
    else:
        # Shadow-log mode: alert but don't auto-remove
        engine.commentary.add_commentary(TradingCommentary(
            timestamp=datetime.now(),
            type=CommentaryType.WARNING,
            symbol=symbol,
            title=f"⚠️ Ghost Position Detected",
            message=(
                f"Position {symbol} appears closed at broker (shows flat) but "
                f"local state still shows LIVE. Reason: {reason}. "
                f"ENABLE_GHOST_FLATTEN_AFTER_BROKER_FLAT=False, so position "
                f"NOT auto-removed. Manual reconciliation required."
            ),
            data={'reason': reason, 'auto_flattened': False},
            importance=10
        ))

        logger.warning(
            "broker_flat_ghost_detected symbol=%s reason=%s — "
            "ghost_flatten_disabled, position NOT removed (manual action required)",
            symbol, reason,
        )
        engine._audit(
            "order_monitor", symbol, "ghost_position_shadow_logged",
            reason,
            local_qty=position.quantity,
            desk_action="alert_only",
        )


async def is_broker_leg_authoritative(
    engine: "TradingEngineWithCommentary",
    position: "Position",
) -> tuple[bool, str]:
    """v-broker-leg-authority-2026-09-14: check if broker's stop/OCO leg is authoritative.

    Returns (is_authoritative, reason) tuple:
      - (True, "stop_working") if stop order is WORKING at broker
      - (True, "stop_filled") if stop order is already FILLED at broker
      - (True, "oco_working") if bracket/OCO order is WORKING at broker
      - (True, "oco_filled") if bracket/OCO order is already FILLED at broker
      - (False, "no_bracket") if position has no bracket_order_id
      - (False, "no_client") if no Schwab client available
      - (False, "fetch_error") if order status fetch failed
      - (False, "not_protective") if order exists but is not in protective state

    When this returns (True, _), the software stop/exit path should skip
    supervised close confirmation — the broker is already protecting the
    position or has already closed it. This prevents FTFT-style races.

    Only checks bracket_order_id if set on position. Does not query if
    position has no tracked bracket.
    """
    from core.config import Config

    if not Config().ENABLE_BROKER_LEG_AUTHORITY:
        return (False, "feature_disabled")

    bracket_id = getattr(position, 'bracket_order_id', None)
    if not bracket_id:
        return (False, "no_bracket")

    if not engine.schwab_client or not engine.account_hash:
        return (False, "no_client")

    try:
        response = engine.schwab_client.get_order(bracket_id, engine.account_hash)
        if response.status_code != 200:
            logger.debug(
                "broker_leg_authority_check: could not fetch bracket %s: %d",
                bracket_id, response.status_code,
            )
            return (False, "fetch_error")

        order_info = response.json()
        status = order_info.get('status', 'UNKNOWN')
        order_type = order_info.get('orderStrategyType', '')

        if status == 'WORKING':
            reason = "oco_working" if order_type == 'OCO' else "stop_working"
            engine._audit(
                "broker_leg_authority", position.symbol, "authoritative",
                reason,
                bracket_id=bracket_id,
                status=status,
            )
            return (True, reason)

        if status == 'FILLED':
            reason = "oco_filled" if order_type == 'OCO' else "stop_filled"
            engine._audit(
                "broker_leg_authority", position.symbol, "authoritative",
                reason,
                bracket_id=bracket_id,
                status=status,
            )
            return (True, reason)

        return (False, "not_protective")

    except Exception as exc:
        logger.warning(
            "broker_leg_authority_check_error symbol=%s bracket_id=%s err=%s",
            position.symbol, bracket_id, exc,
        )
        return (False, "fetch_error")
