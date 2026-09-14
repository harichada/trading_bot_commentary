"""v-broker-flat-detection-2026-09-10: Broker-flat detection and handling.

This module handles the detection of positions that have been closed externally
at the broker (e.g., user manual close, broker stop filled without our knowledge).

Key scenarios handled:
  - Oversold/overbought bracket rejections indicating position already closed
  - Quantity sync when local qty diverges from broker qty
  - Proper FSM transition to CLOSED state for external closes

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

    Actions:
      1. Clear managed_by_bot → stop all exit management
      2. Cancel any working exit orders for this symbol
      3. Mark position as CLOSED (external close)
      4. Remove from active tracking
      5. Audit for post-mortem

    This breaks the infinite re_bracket loop that occurs when:
      - Bot thinks it has a position (ghost qty in self.positions)
      - Schwab rejects bracket for oversold/overbought (shares gone)
      - _handle_bracket_rejected calls _re_bracket_position
      - New bracket rejected → repeat forever
    """
    from core.commentary import TradingCommentary
    from core.models import CommentaryType, clear_bracket_ids
    from core.position_state import PositionState, try_transition

    symbol = position.symbol

    engine._audit(
        "order_monitor", symbol, "broker_flat_detected",
        reason,
        local_qty=position.quantity,
        managed_by_bot=getattr(position, 'managed_by_bot', False),
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
        data={'reason': reason},
        importance=9
    ))

    logger.info(
        "broker_flat_handled symbol=%s reason=%s — position removed from tracking",
        symbol, reason,
    )
