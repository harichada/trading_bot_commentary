"""v-order-monitor-2026-09-10: Bracket order handling (fills, replace, re-bracket).

This module handles:
  - Bracket fill detection (stop or TP hit)
  - Full and partial fill handling
  - Trail stop replacement
  - Re-bracketing positions after partial fills or recovery

All functions receive the engine reference as first parameter for access to:
  - schwab_client for broker API calls
  - commentary for user-facing messages
  - _audit for structured logging
  - positions for state management
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from core.engine import TradingEngineWithCommentary
    from core.models import Position

from core.order_monitor.broker_flat import (
    check_broker_position_qty,
    handle_broker_flat_detected,
    is_broker_flat_rejection,
)

logger = logging.getLogger("TradingBot")

# Maximum re-bracket attempts before giving up
MAX_RE_BRACKET_ATTEMPTS = 3


async def refresh_bracket_child_ids(
    engine: "TradingEngineWithCommentary",
    position: "Position",
    orders_by_id: Dict[str, dict],
    all_working_orders: list,
) -> None:
    """Extract stop_order_id and tp_order_id from the OCO structure.

    Schwab OCO orders have childOrderStrategies containing the two legs.
    """
    bracket_id = position.bracket_order_id
    parent_order = orders_by_id.get(bracket_id)

    if not parent_order:
        # Try to fetch the order directly
        try:
            response = engine.schwab_client.get_order(
                bracket_id, engine.account_hash
            )
            if response.status_code == 200:
                parent_order = response.json()
        except Exception:
            pass

    if not parent_order:
        return

    # Extract child order IDs from OCO structure
    children = parent_order.get('childOrderStrategies', [])
    for child in children:
        child_id = str(child.get('orderId', ''))
        if not child_id:
            continue

        # Determine if this is stop or TP based on order type
        order_type = child.get('orderType', '')
        stop_price = child.get('stopPrice')

        if order_type == 'STOP_LIMIT' or stop_price:
            position.stop_order_id = child_id
            if stop_price:
                position.broker_stop_price = float(stop_price)
        elif order_type == 'LIMIT':
            position.tp_order_id = child_id

    if position.stop_order_id or position.tp_order_id:
        engine._audit(
            "order_monitor", position.symbol, "child_ids_extracted",
            "bracket_structure",
            bracket_id=bracket_id,
            stop_order_id=position.stop_order_id,
            tp_order_id=position.tp_order_id,
        )


async def check_trail_replace(
    engine: "TradingEngineWithCommentary",
    position: "Position",
    parent_order: dict,
) -> None:
    """Check if software trailing stop has moved and replace broker stop.

    When position.trailing_stop moves beyond the broker's stop price,
    cancel the old OCO and place a new one with updated stop level.
    """
    # Only check if trailing is active
    trail = getattr(position, 'trailing_stop', None)
    if trail is None:
        return

    broker_stop = getattr(position, 'broker_stop_price', None)
    if broker_stop is None:
        return

    # For long positions, trailing_stop should be rising (tighter)
    # Only replace if the new stop is significantly better
    if position.side == 'long':
        should_replace = trail > broker_stop * 1.005  # 0.5% threshold
    else:
        should_replace = trail < broker_stop * 0.995

    if not should_replace:
        return

    engine._audit(
        "order_monitor", position.symbol, "trail_replace_needed",
        "software_trail_moved",
        broker_stop=round(broker_stop, 4),
        software_trail=round(trail, 4),
        side=position.side,
    )

    # Cancel existing bracket and place new one
    await replace_bracket_with_new_stop(engine, position, trail)


async def replace_bracket_with_new_stop(
    engine: "TradingEngineWithCommentary",
    position: "Position",
    new_stop: float,
) -> None:
    """Cancel existing bracket and place new OCO with updated stop.

    This is the trail-replace mechanism: when software trailing stop
    tightens, we want the broker protection to match so we don't leave
    stale far stops that give back gains.
    """
    from core.commentary import TradingCommentary
    from core.models import CommentaryType, clear_bracket_ids

    symbol = position.symbol
    bracket_id = position.bracket_order_id

    if not bracket_id:
        return

    engine.commentary.add_commentary(TradingCommentary(
        timestamp=datetime.now(),
        type=CommentaryType.RISK_ASSESSMENT,
        symbol=symbol,
        title=f"🔄 Replacing Bracket Stop",
        message=(
            f"Software trail at ${new_stop:.2f} is tighter than "
            f"broker stop at ${position.broker_stop_price:.2f}. "
            f"Updating broker protection to match."
        ),
        data={
            'old_broker_stop': position.broker_stop_price,
            'new_stop': new_stop,
        },
        importance=7
    ))

    # Cancel existing bracket
    try:
        cancel_response = engine.schwab_client.cancel_order(
            bracket_id, engine.account_hash
        )
        if cancel_response.status_code not in [200, 201, 202]:
            logger.warning(
                "order_monitor: failed to cancel bracket %s for %s: %d",
                bracket_id, symbol, cancel_response.status_code,
            )
            return
    except Exception as exc:
        logger.error(
            "order_monitor: error canceling bracket %s for %s: %s",
            bracket_id, symbol, exc,
        )
        return

    # Clear old bracket IDs
    clear_bracket_ids(position)

    # Wait briefly for cancellation to process
    await asyncio.sleep(0.5)

    # Place new bracket with updated stop
    try:
        from schwab.orders.common import one_cancels_other, Duration, Session, OrderType
        from schwab.orders.equities import equity_sell_limit

        # Use current take_profit from position
        tp_price = position.take_profit

        # Create new TP order
        take_profit_order = equity_sell_limit(
            symbol,
            position.quantity,
            tp_price
        ).set_duration(Duration.GOOD_TILL_CANCEL).set_session(Session.NORMAL)

        # Create new stop order with updated price
        stop_loss_order = (equity_sell_limit(
            symbol,
            position.quantity,
            new_stop * 0.995  # Limit slightly below stop
        ).set_order_type(OrderType.STOP_LIMIT)
        .set_stop_price(new_stop)
        .set_duration(Duration.GOOD_TILL_CANCEL)
        .set_session(Session.NORMAL))

        oco_order = one_cancels_other(take_profit_order, stop_loss_order)

        response = engine.schwab_client.place_order(
            engine.account_hash, oco_order.build()
        )

        if response.status_code in [200, 201]:
            new_order_id = response.headers.get('Location', '').split('/')[-1]
            position.bracket_order_id = new_order_id
            position.broker_stop_price = new_stop

            engine._audit(
                "order_monitor", symbol, "bracket_replaced",
                "trail_stop_updated",
                old_bracket_id=bracket_id,
                new_bracket_id=new_order_id,
                new_stop=round(new_stop, 4),
            )

            engine.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.RISK_ASSESSMENT,
                symbol=symbol,
                title=f"✅ Bracket Stop Updated",
                message=f"New stop at ${new_stop:.2f} now active at broker.",
                data={'new_bracket_id': new_order_id, 'new_stop': new_stop},
                importance=7
            ))
        else:
            logger.error(
                "order_monitor: failed to place replacement bracket for %s: %d",
                symbol, response.status_code,
            )
            # Alert loudly — position now has NO broker protection
            engine.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.WARNING,
                symbol=symbol,
                title=f"🚨 BRACKET REPLACEMENT FAILED",
                message=(
                    f"Canceled old bracket but could not place new one. "
                    f"Position {symbol} has NO broker-side stop/TP! "
                    f"Software exits still active. Manual review needed."
                ),
                importance=10
            ))
    except Exception as exc:
        logger.error(
            "order_monitor: error placing replacement bracket for %s: %s",
            symbol, exc, exc_info=True,
        )
        engine.commentary.add_commentary(TradingCommentary(
            timestamp=datetime.now(),
            type=CommentaryType.WARNING,
            symbol=symbol,
            title=f"🚨 BRACKET REPLACEMENT ERROR",
            message=f"Error: {exc}. Position may lack broker protection.",
            importance=10
        ))


async def handle_bracket_not_working(
    engine: "TradingEngineWithCommentary",
    position: "Position",
) -> None:
    """Handle case where bracket order is no longer WORKING.

    Query the order status directly to determine what happened:
      - FILLED: stop or TP hit → sync position state
      - CANCELED: need to re-protect or alert
      - REJECTED: place new protection or soft-halt
    """
    symbol = position.symbol
    bracket_id = position.bracket_order_id

    if not bracket_id:
        return

    # Fetch the bracket order status
    try:
        response = engine.schwab_client.get_order(bracket_id, engine.account_hash)
        if response.status_code != 200:
            logger.warning(
                "order_monitor: could not fetch bracket %s for %s: %d",
                bracket_id, symbol, response.status_code,
            )
            return
        order_info = response.json()
    except Exception as exc:
        logger.error(
            "order_monitor: error fetching bracket %s for %s: %s",
            bracket_id, symbol, exc,
        )
        return

    status = order_info.get('status', 'UNKNOWN')

    if status == 'FILLED':
        await handle_bracket_fill(engine, position, order_info)
    elif status in ['CANCELED', 'EXPIRED']:
        await handle_bracket_canceled(engine, position, order_info, status)
    elif status == 'REJECTED':
        await handle_bracket_rejected(engine, position, order_info)
    elif status == 'WORKING':
        # Still working — probably just not in our filtered list
        pass
    else:
        logger.info(
            "order_monitor: bracket %s for %s has status %s",
            bracket_id, symbol, status,
        )


async def handle_bracket_fill(
    engine: "TradingEngineWithCommentary",
    position: "Position",
    order_info: dict,
) -> None:
    """Handle a bracket order fill (stop or TP hit).

    Determine which leg filled, sync Position state (reduce or close),
    cancel the sibling leg if still open, and emit commentary.
    """
    symbol = position.symbol

    # Determine fill details from order activity
    fill_price = None
    fill_qty = 0
    filled_leg = "unknown"

    # Check child order statuses
    children = order_info.get('childOrderStrategies', [])
    for child in children:
        child_status = child.get('status', '')
        child_type = child.get('orderType', '')

        if child_status == 'FILLED':
            # This is the leg that filled
            activities = child.get('orderActivityCollection', [])
            for activity in activities:
                if activity.get('executionType') == 'FILL':
                    legs = activity.get('executionLegs', [])
                    if legs:
                        fill_price = legs[0].get('price')
                        fill_qty = legs[0].get('quantity', 0)

            if child_type == 'STOP_LIMIT' or child.get('stopPrice'):
                filled_leg = "stop"
            elif child_type == 'LIMIT':
                filled_leg = "take_profit"

    engine._audit(
        "order_monitor", symbol, "bracket_fill",
        filled_leg,
        bracket_id=position.bracket_order_id,
        fill_price=fill_price,
        fill_qty=fill_qty,
        position_qty=position.quantity,
    )

    # Handle partial vs full fill
    remaining_qty = position.quantity - fill_qty if fill_qty else 0

    if remaining_qty > 0:
        # Partial fill — update position qty and re-bracket
        await handle_partial_bracket_fill(
            engine, position, fill_price, fill_qty, remaining_qty, filled_leg
        )
    else:
        # Full fill — close position
        await handle_full_bracket_fill(
            engine, position, fill_price, filled_leg
        )


async def handle_full_bracket_fill(
    engine: "TradingEngineWithCommentary",
    position: "Position",
    fill_price: Optional[float],
    filled_leg: str,
) -> None:
    """Handle a full bracket fill (position fully closed by broker)."""
    from core.commentary import TradingCommentary
    from core.models import CommentaryType, clear_bracket_ids

    symbol = position.symbol

    # Clear bracket tracking
    clear_bracket_ids(position)

    # Determine exit reason for trade record
    if filled_leg == "stop":
        exit_reason = "stop_loss_broker"
        title = f"🛑 Stop Loss Filled by Broker"
        message = f"Bracket stop hit at ${fill_price:.2f}. Position closed."
    elif filled_leg == "take_profit":
        exit_reason = "take_profit_broker"
        title = f"🎯 Take Profit Filled by Broker"
        message = f"Bracket target hit at ${fill_price:.2f}. Position closed."
    else:
        exit_reason = "bracket_fill_broker"
        title = f"📊 Bracket Order Filled"
        message = f"Bracket order filled at ${fill_price:.2f}."

    engine.commentary.add_commentary(TradingCommentary(
        timestamp=datetime.now(),
        type=CommentaryType.DECISION,
        symbol=symbol,
        title=title,
        message=message,
        data={
            'fill_price': fill_price,
            'filled_leg': filled_leg,
            'entry_price': position.entry_price,
        },
        importance=9
    ))

    # Calculate P&L
    if fill_price:
        if position.side == 'long':
            pnl = (fill_price - position.entry_price) * position.quantity
        else:
            pnl = (position.entry_price - fill_price) * position.quantity
    else:
        pnl = position.unrealized_pnl

    # Record trade
    trade_record = {
        'symbol': symbol,
        'entry_time': position.entry_time.isoformat() if hasattr(position.entry_time, 'isoformat') else str(position.entry_time),
        'exit_time': datetime.now().isoformat(),
        'entry_price': position.entry_price,
        'exit_price': fill_price or position.current_price,
        'quantity': position.quantity,
        'side': position.side,
        'pnl': pnl,
        'exit_reason': exit_reason,
        'mode': 'live',
    }
    engine.trade_history.append(trade_record)

    # Update risk manager
    if pnl < 0:
        engine.risk_manager.consecutive_losses += 1
    else:
        engine.risk_manager.consecutive_losses = 0

    # Remove position
    if symbol in engine.positions:
        del engine.positions[symbol]

    # Clean up exit manager tracking
    if hasattr(engine, 'exit_manager'):
        engine.exit_manager.close_position_tracking(symbol)

    await engine._save_state()

    engine._audit(
        "order_monitor", symbol, "position_closed",
        exit_reason,
        fill_price=fill_price,
        pnl=round(pnl, 2) if pnl else None,
    )


async def handle_partial_bracket_fill(
    engine: "TradingEngineWithCommentary",
    position: "Position",
    fill_price: Optional[float],
    fill_qty: int,
    remaining_qty: int,
    filled_leg: str,
) -> None:
    """Handle a partial bracket fill — update qty and re-bracket remaining.

    v-qty-sync-2026-09-11: Added broker position verification when fill_qty
    looks suspicious (fill_qty < 10% of local qty or remaining_qty seems
    inconsistent). The re_bracket call will sync to broker truth, but we
    also proactively check here to provide better audit trail.
    """
    from core.commentary import TradingCommentary
    from core.models import CommentaryType, TradingMode, clear_bracket_ids

    symbol = position.symbol
    local_qty = position.quantity

    # v-qty-sync-2026-09-11: detect suspicious qty mismatches early
    # If fill_qty is much smaller than local_qty, there may be untracked fills
    fill_ratio = fill_qty / local_qty if local_qty > 0 else 0
    is_suspicious = fill_ratio < 0.1 and fill_qty < 100  # <10% and small absolute

    if is_suspicious and engine.mode == TradingMode.LIVE and engine.schwab_client:
        broker_qty = await check_broker_position_qty(engine, symbol)
        if broker_qty == 0:
            # Broker is flat — this was actually a full close, not partial
            logger.warning(
                "partial_fill_was_actually_full symbol=%s fill_qty=%d local_qty=%d broker_flat=True",
                symbol, fill_qty, local_qty,
            )
            engine._audit(
                "order_monitor", symbol, "partial_fill_override",
                "broker_confirmed_flat",
                fill_qty=fill_qty,
                local_qty=local_qty,
            )
            # Handle as full fill since broker confirms flat
            await handle_full_bracket_fill(engine, position, fill_price, filled_leg)
            return
        elif broker_qty != remaining_qty:
            # Broker qty differs from calculated remaining — will be synced in re_bracket
            engine._audit(
                "order_monitor", symbol, "partial_fill_qty_mismatch",
                "broker_will_sync",
                fill_qty=fill_qty,
                local_qty=local_qty,
                calculated_remaining=remaining_qty,
                broker_qty=broker_qty,
            )

    engine.commentary.add_commentary(TradingCommentary(
        timestamp=datetime.now(),
        type=CommentaryType.DECISION,
        symbol=symbol,
        title=f"📊 Partial Bracket Fill",
        message=(
            f"Partial {filled_leg} fill: {fill_qty} shares at "
            f"${fill_price:.2f}. {remaining_qty} shares remaining."
        ),
        data={
            'fill_price': fill_price,
            'fill_qty': fill_qty,
            'remaining_qty': remaining_qty,
            'filled_leg': filled_leg,
        },
        importance=8
    ))

    # Update position quantity
    old_qty = position.quantity
    position.quantity = remaining_qty

    # Clear old bracket IDs
    old_bracket_id = position.bracket_order_id
    clear_bracket_ids(position)

    engine._audit(
        "order_monitor", symbol, "partial_fill",
        "qty_reduced",
        fill_qty=fill_qty,
        remaining_qty=remaining_qty,
        old_bracket_id=old_bracket_id,
    )

    # Re-bracket remaining shares (will sync qty to broker truth if needed)
    await re_bracket_position(engine, position)


async def handle_bracket_canceled(
    engine: "TradingEngineWithCommentary",
    position: "Position",
    order_info: dict,
    status: str,
) -> None:
    """Handle bracket order cancellation or expiration.

    Policy: attempt to re-place the bracket. If that fails, alert loudly
    but do NOT soft-halt new entries (software exits still protect).
    """
    from core.commentary import TradingCommentary
    from core.models import CommentaryType, clear_bracket_ids

    symbol = position.symbol

    engine._audit(
        "order_monitor", symbol, "bracket_canceled",
        status.lower(),
        bracket_id=position.bracket_order_id,
    )

    engine.commentary.add_commentary(TradingCommentary(
        timestamp=datetime.now(),
        type=CommentaryType.WARNING,
        symbol=symbol,
        title=f"⚠️ Bracket Order {status}",
        message=(
            f"Bracket was {status.lower()}. "
            f"Attempting to re-place broker protection."
        ),
        importance=8
    ))

    # Clear old bracket and attempt to re-place
    clear_bracket_ids(position)

    await re_bracket_position(engine, position)


async def handle_bracket_rejected(
    engine: "TradingEngineWithCommentary",
    position: "Position",
    order_info: dict,
) -> None:
    """Handle bracket order rejection.

    v-broker-flat-detection-2026-09-10: enhanced policy:
      1. Check if rejection indicates broker-flat (oversold/overbought)
      2. If so, verify with broker and handle as external close
      3. Otherwise, attempt to re-place with adjusted prices
      4. Track re_bracket attempts to prevent infinite loops
    """
    from core.commentary import TradingCommentary
    from core.models import CommentaryType, clear_bracket_ids

    symbol = position.symbol
    rejection_reason = order_info.get('statusDescription', 'Unknown reason')

    engine._audit(
        "order_monitor", symbol, "bracket_rejected",
        "broker_rejection",
        bracket_id=position.bracket_order_id,
        rejection_detail=rejection_reason,
    )

    engine.commentary.add_commentary(TradingCommentary(
        timestamp=datetime.now(),
        type=CommentaryType.WARNING,
        symbol=symbol,
        title=f"🚨 Bracket Order REJECTED",
        message=(
            f"Broker rejected bracket: {rejection_reason}. "
            f"Position has NO broker-side protection! "
            f"Software exits still active."
        ),
        data={'reason': rejection_reason},
        importance=10
    ))

    # Clear the rejected bracket
    clear_bracket_ids(position)

    # v-broker-flat-detection-2026-09-10: check if this is an oversold/overbought
    # rejection, which indicates the position is already closed at the broker
    if is_broker_flat_rejection(rejection_reason):
        logger.info(
            "bracket_rejected_flat_indicator symbol=%s reason=%s — checking broker position",
            symbol, rejection_reason,
        )
        broker_qty = await check_broker_position_qty(engine, symbol)
        if broker_qty == 0:
            # Confirmed broker-flat — handle as external close, DO NOT re_bracket
            await handle_broker_flat_detected(engine, position, "reject_oversold_overbought")
            return
        else:
            # Broker still shows position — log the mismatch but proceed
            logger.warning(
                "bracket_rejected_flat_mismatch symbol=%s reason=%s broker_qty=%d — proceeding with re_bracket",
                symbol, rejection_reason, broker_qty,
            )

    # v-broker-flat-detection-2026-09-10: track re_bracket attempts to prevent
    # infinite loops even if broker-flat detection fails
    re_bracket_attempts = getattr(position, '_re_bracket_attempts', 0) + 1
    position._re_bracket_attempts = re_bracket_attempts

    if re_bracket_attempts > MAX_RE_BRACKET_ATTEMPTS:
        engine._audit(
            "order_monitor", symbol, "re_bracket_exhausted",
            "max_attempts_exceeded",
            attempts=re_bracket_attempts,
            rejection_detail=rejection_reason,
        )
        engine.commentary.add_commentary(TradingCommentary(
            timestamp=datetime.now(),
            type=CommentaryType.WARNING,
            symbol=symbol,
            title=f"🛑 Re-Bracket Attempts Exhausted",
            message=(
                f"Failed to re-bracket after {MAX_RE_BRACKET_ATTEMPTS} attempts. "
                f"Last rejection: {rejection_reason}. "
                f"Position has software exits only (no broker stop)."
            ),
            importance=10
        ))
        return

    # Attempt to re-place with current prices
    # (The rejection may have been due to price validation)
    await re_bracket_position(engine, position)


async def re_bracket_position(
    engine: "TradingEngineWithCommentary",
    position: "Position",
) -> None:
    """Place new bracket orders for a position (after partial or recovery).

    v-broker-flat-detection-2026-09-10: added broker position verification
    before placing bracket to prevent placing orders for positions that
    no longer exist at the broker.

    v-qty-sync-2026-09-11: sync local qty to broker truth before placing
    bracket. If broker shows fewer shares than local, we had untracked
    fills (partial fills, external closes). Sync to broker qty and
    re-bracket with correct amount. If broker is flat, clean up.
    """
    from core.commentary import TradingCommentary
    from core.models import CommentaryType, TradingMode

    symbol = position.symbol

    # v-broker-flat-detection-2026-09-10 + v-qty-sync-2026-09-11:
    # Verify position qty at broker before placing bracket.
    # If broker_qty != local_qty, sync local to broker truth.
    if engine.mode == TradingMode.LIVE and engine.schwab_client:
        broker_qty = await check_broker_position_qty(engine, symbol)
        local_qty = position.quantity

        if broker_qty == 0:
            logger.warning(
                "re_bracket_aborted_broker_flat symbol=%s local_qty=%d — broker shows flat",
                symbol, local_qty,
            )
            engine._audit(
                "order_monitor", symbol, "re_bracket_aborted",
                "broker_flat_pre_check",
                local_qty=local_qty,
            )
            # Handle as external close instead of placing invalid bracket
            await handle_broker_flat_detected(engine, position, "re_bracket_pre_check_flat")
            return

        # v-qty-sync-2026-09-11: sync local qty to broker truth
        if broker_qty != local_qty:
            logger.warning(
                "qty_desync_detected symbol=%s local_qty=%d broker_qty=%d — syncing to broker",
                symbol, local_qty, broker_qty,
            )
            engine._audit(
                "order_monitor", symbol, "qty_desync",
                "sync_to_broker",
                local_qty=local_qty,
                broker_qty=broker_qty,
                delta=local_qty - broker_qty,
            )
            engine.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.WARNING,
                symbol=symbol,
                title=f"⚠️ Quantity Desync Detected",
                message=(
                    f"Local qty {local_qty} ≠ broker qty {broker_qty}. "
                    f"Syncing to broker truth. Possible untracked fills."
                ),
                data={
                    'local_qty': local_qty,
                    'broker_qty': broker_qty,
                    'delta': local_qty - broker_qty,
                },
                importance=9
            ))
            position.quantity = broker_qty

    try:
        from schwab.orders.common import one_cancels_other, Duration, Session, OrderType
        from schwab.orders.equities import equity_sell_limit

        # Use current stop/TP from position (may have been trailed)
        stop_price = position.trailing_stop or position.stop_loss
        tp_price = position.take_profit
        qty = position.quantity

        if not stop_price or not tp_price or qty <= 0:
            logger.warning(
                "order_monitor: cannot re-bracket %s: stop=%s tp=%s qty=%d",
                symbol, stop_price, tp_price, qty,
            )
            return

        take_profit_order = equity_sell_limit(
            symbol, qty, tp_price
        ).set_duration(Duration.GOOD_TILL_CANCEL).set_session(Session.NORMAL)

        stop_loss_order = (equity_sell_limit(
            symbol, qty, stop_price * 0.995
        ).set_order_type(OrderType.STOP_LIMIT)
        .set_stop_price(stop_price)
        .set_duration(Duration.GOOD_TILL_CANCEL)
        .set_session(Session.NORMAL))

        oco_order = one_cancels_other(take_profit_order, stop_loss_order)

        response = engine.schwab_client.place_order(
            engine.account_hash, oco_order.build()
        )

        if response.status_code in [200, 201]:
            new_order_id = response.headers.get('Location', '').split('/')[-1]
            position.bracket_order_id = new_order_id
            position.broker_stop_price = stop_price

            engine._audit(
                "order_monitor", symbol, "re_bracket",
                "new_bracket_placed",
                new_bracket_id=new_order_id,
                qty=qty,
                stop=round(stop_price, 4),
                tp=round(tp_price, 4),
            )

            engine.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.RISK_ASSESSMENT,
                symbol=symbol,
                title=f"✅ Re-Bracketed Position",
                message=(
                    f"New bracket placed for {qty} shares: "
                    f"stop ${stop_price:.2f}, target ${tp_price:.2f}"
                ),
                importance=7
            ))
        else:
            logger.error(
                "order_monitor: re-bracket failed for %s: %d",
                symbol, response.status_code,
            )
            engine.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.WARNING,
                symbol=symbol,
                title=f"⚠️ Re-Bracket Failed",
                message="Could not place new bracket. Software exits still active.",
                importance=8
            ))
    except Exception as exc:
        logger.error(
            "order_monitor: re-bracket error for %s: %s",
            symbol, exc, exc_info=True,
        )
