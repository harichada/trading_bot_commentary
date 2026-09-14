"""v-order-monitor-2026-09-10: Realtime WORKING bracket/OCO order monitoring.

The OrderMonitor class is the main orchestrator for bracket order monitoring.
It runs as an async loop registered with the TaskSupervisor and handles:
  - Position filtering (bot-managed with brackets, excluding HANDS_OFF)
  - Orphan bracket bootstrap at startup
  - Dispatching to bracket handlers for fills, cancels, rejects

HANDS_OFF guards are applied at the filtering level before any bracket
operations:
  - HANDS_OFF_DENYLIST symbols (MU, SNAP, SPCX, HQGE) — never monitored
  - is_external, is_manually_managed, is_long_term flags — skip monitoring

The denylist check is independent of the flags, so a denylist symbol is
protected even if none of those flags happen to be set.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from core.engine import TradingEngineWithCommentary
    from core.models import Position

from core.config import Config
from core.order_monitor.brackets import (
    check_trail_replace,
    handle_bracket_not_working,
    refresh_bracket_child_ids,
)

logger = logging.getLogger("TradingBot")


class OrderMonitor:
    """Monitors WORKING bracket/OCO orders for bot-managed positions.

    Responsibilities:
      1. Detect stop/TP fills → sync Position state, cancel sibling leg
      2. Detect cancel/reject → re-place missing stop or soft-halt
      3. Detect partials → update qty, re-bracket remaining
      4. Trail replace → when software trail moves, replace broker stop

    Only monitors positions where:
      - managed_by_bot=True
      - NOT in HANDS_OFF_DENYLIST (MU, SNAP, SPCX, HQGE)
      - NOT is_external / is_manually_managed / is_long_term
      - Has bracket_order_id set (bot placed a bracket)

    Cadence: every 5 seconds (tunable via Config.ORDER_MONITOR_INTERVAL_SEC)
    """

    def __init__(self, engine: "TradingEngineWithCommentary") -> None:
        self.engine = engine
        self._oco_bootstrap_done = False

    async def run(self) -> None:
        """Main order monitor loop.

        Poll WORKING bracket/OCO orders for bot-managed symbols.
        """
        from core.config import Config

        cadence = getattr(Config(), 'ORDER_MONITOR_INTERVAL_SEC', 5.0)

        while self.engine.is_running:
            try:
                # Skip if no Schwab client
                if not self.engine.schwab_client or not self.engine.account_hash:
                    await asyncio.sleep(cadence)
                    continue

                # v-bootstrap-oco-2026-09-10: run bootstrap once if not done at startup
                # This catches the case where startup didn't have Schwab client ready
                if not self._oco_bootstrap_done:
                    try:
                        attached = await self.bootstrap_orphan_brackets()
                        self._oco_bootstrap_done = True
                        if attached > 0:
                            logger.info(
                                "order_monitor_bootstrap_oco: attached %d orphan brackets",
                                attached,
                            )
                    except Exception as exc:
                        logger.warning(f"order_monitor bootstrap_orphan_brackets failed: {exc}")
                        self._oco_bootstrap_done = True  # Don't retry on failure

                # Only monitor LIVE positions that we manage with brackets
                bot_positions = self.get_bracket_monitored_positions()
                if not bot_positions:
                    await asyncio.sleep(cadence)
                    continue

                # Fetch WORKING orders from Schwab
                working_orders = await self.fetch_working_orders()
                if working_orders is None:
                    await asyncio.sleep(cadence)
                    continue

                # Build lookup: order_id -> order_info
                orders_by_id = {
                    str(o.get('orderId')): o
                    for o in working_orders
                    if o.get('orderId')
                }

                # Process each bot-managed position with a bracket
                for symbol, position in bot_positions:
                    try:
                        await self._monitor_position_bracket(
                            position, orders_by_id, working_orders
                        )
                    except Exception as exc:
                        logger.error(
                            "order_monitor: error processing %s: %s",
                            symbol, exc, exc_info=True,
                        )

            except Exception as exc:
                logger.error(
                    "order_monitor_loop: unhandled error: %s",
                    exc, exc_info=True,
                )
                # Don't re-raise — let the loop continue after sleep

            await asyncio.sleep(cadence)

    def get_bracket_monitored_positions(self) -> List[Tuple[str, "Position"]]:
        """Return (symbol, position) pairs for positions we should monitor.

        Criteria:
          - In self.positions (LIVE mode, not simulated)
          - managed_by_bot=True
          - NOT in HANDS_OFF_DENYLIST (MU, SNAP, SPCX, HQGE)
          - NOT is_external / is_manually_managed / is_long_term
          - Has bracket_order_id (bot placed a bracket for this position)
        """
        denylist = Config().HANDS_OFF_DENYLIST
        result = []
        for symbol, pos in list(self.engine.positions.items()):
            if pos is None:
                continue
            # Skip non-bot-managed
            if not getattr(pos, 'managed_by_bot', False):
                continue
            # v-hands-off-denylist-2026-09-14: skip denylist symbols unconditionally
            if symbol.upper() in denylist:
                continue
            # Skip external/manual/long-term
            if getattr(pos, 'is_external', False):
                continue
            if getattr(pos, 'is_manually_managed', False):
                continue
            if getattr(pos, 'is_long_term', False):
                continue
            # Skip if no bracket (software exits only)
            if not getattr(pos, 'bracket_order_id', None):
                continue
            result.append((symbol, pos))
        return result

    def get_positions_missing_brackets(self) -> List[Tuple[str, "Position"]]:
        """Return (symbol, position) pairs for positions that SHOULD have brackets but don't.

        v-bootstrap-oco-2026-09-10: Used by bootstrap logic to find LIVE positions
        that pass all filters for bracket monitoring EXCEPT they're missing
        bracket_order_id. These are candidates for attaching orphan WORKING OCOs.

        Criteria (same as get_bracket_monitored_positions minus bracket_order_id check):
          - In self.positions (LIVE mode, not simulated)
          - managed_by_bot=True
          - NOT in HANDS_OFF_DENYLIST (MU, SNAP, SPCX, HQGE)
          - NOT is_external / is_manually_managed / is_long_term
          - MISSING bracket_order_id (inverse of normal monitor filter)
        """
        denylist = Config().HANDS_OFF_DENYLIST
        result = []
        for symbol, pos in list(self.engine.positions.items()):
            if pos is None:
                continue
            # Skip non-bot-managed
            if not getattr(pos, 'managed_by_bot', False):
                continue
            # v-hands-off-denylist-2026-09-14: skip denylist symbols unconditionally
            if symbol.upper() in denylist:
                continue
            # Skip external/manual/long-term (Hari's 4 LT holds safe)
            if getattr(pos, 'is_external', False):
                continue
            if getattr(pos, 'is_manually_managed', False):
                continue
            if getattr(pos, 'is_long_term', False):
                continue
            # Include only if MISSING bracket (inverse of normal filter)
            if getattr(pos, 'bracket_order_id', None):
                continue  # Already has bracket, skip
            result.append((symbol, pos))
        return result

    async def bootstrap_orphan_brackets(self) -> int:
        """Bootstrap/attach existing WORKING Schwab OCO brackets to positions missing bracket_order_id.

        v-bootstrap-oco-2026-09-10: Called at engine startup (and optionally once at
        first order_monitor tick) to reconcile bot-managed LIVE positions that are
        missing bracket_order_id with any orphan WORKING OCO orders on the account.

        This handles the case where the bot deployed with open positions that don't
        have bracket tracking — either from a prior version, a restart, or state loss.
        Without this, such positions are invisible to order_monitor_loop until the
        next _place_bracket_orders call (which may never happen for existing positions).

        Matching logic:
          1. Fetch WORKING orders from Schwab (OCO/bracket type)
          2. Group OCOs by symbol
          3. For each position missing bracket_order_id:
             a. Find candidate OCOs for that symbol
             b. Prefer exact qty match; if no exact, use any match
             c. If multiple candidates remain with same priority, alert HIGH and skip
             d. If single candidate, attach it and extract child IDs
          4. Persist via _save_state()

        Returns: count of positions successfully attached
        """
        from core.commentary import TradingCommentary
        from core.models import CommentaryType

        if not self.engine.schwab_client or not self.engine.account_hash:
            return 0

        positions_missing = self.get_positions_missing_brackets()
        if not positions_missing:
            logger.debug("bootstrap_oco: no positions missing brackets")
            return 0

        # Fetch WORKING orders from Schwab
        working_orders = await self.fetch_working_orders()
        if working_orders is None:
            logger.warning("bootstrap_oco: failed to fetch working orders")
            return 0

        # Filter to OCO orders only and group by symbol
        ocos_by_symbol: Dict[str, list] = {}
        for order in working_orders:
            order_type = order.get('orderStrategyType', '')
            if order_type != 'OCO':
                continue

            # Extract symbol from childOrderStrategies
            children = order.get('childOrderStrategies', [])
            for child in children:
                legs = child.get('orderLegCollection', [])
                for leg in legs:
                    sym = leg.get('instrument', {}).get('symbol')
                    if sym:
                        if sym not in ocos_by_symbol:
                            ocos_by_symbol[sym] = []
                        if order not in ocos_by_symbol[sym]:
                            ocos_by_symbol[sym].append(order)
                        break

        if not ocos_by_symbol:
            logger.debug("bootstrap_oco: no WORKING OCO orders found")
            return 0

        attached_count = 0

        for symbol, position in positions_missing:
            candidates = ocos_by_symbol.get(symbol, [])
            if not candidates:
                logger.debug("bootstrap_oco: no OCO candidates for %s", symbol)
                continue

            # Extract qty and filter by exact match first
            def get_oco_qty(oco: dict) -> Optional[int]:
                """Extract quantity from OCO order children."""
                for child in oco.get('childOrderStrategies', []):
                    for leg in child.get('orderLegCollection', []):
                        qty = leg.get('quantity')
                        if qty is not None:
                            return int(qty)
                return None

            position_qty = int(position.quantity)
            exact_qty_matches = [o for o in candidates if get_oco_qty(o) == position_qty]

            if len(exact_qty_matches) == 1:
                # Single exact match — attach it
                chosen = exact_qty_matches[0]
            elif len(exact_qty_matches) > 1:
                # Multiple exact qty matches — ambiguous, alert HIGH and skip
                self.engine._audit(
                    "bootstrap_oco", symbol, "ambiguous_skip",
                    "multiple_ocos_exact_qty_match",
                    candidate_count=len(exact_qty_matches),
                    position_qty=position_qty,
                )
                self.engine.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.WARNING,
                    symbol=symbol,
                    title=f"⚠️ Multiple OCO Orders Found for {symbol}",
                    message=f"Found {len(exact_qty_matches)} WORKING OCO orders with qty={position_qty}. "
                            f"Cannot determine which to attach — manual intervention required.",
                    importance=9,  # HIGH importance
                ))
                continue
            elif len(candidates) == 1:
                # Single candidate (even if qty doesn't match) — attach with warning
                chosen = candidates[0]
                oco_qty = get_oco_qty(chosen)
                if oco_qty != position_qty:
                    self.engine._audit(
                        "bootstrap_oco", symbol, "attach_qty_mismatch",
                        "single_oco_qty_differs",
                        position_qty=position_qty,
                        oco_qty=oco_qty,
                    )
            else:
                # Multiple candidates, none with exact qty — ambiguous
                self.engine._audit(
                    "bootstrap_oco", symbol, "ambiguous_skip",
                    "multiple_ocos_no_exact_qty",
                    candidate_count=len(candidates),
                    position_qty=position_qty,
                )
                self.engine.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.WARNING,
                    symbol=symbol,
                    title=f"⚠️ Multiple OCO Orders Found for {symbol}",
                    message=f"Found {len(candidates)} WORKING OCO orders for {symbol} but none "
                            f"match position qty={position_qty}. Manual intervention required.",
                    importance=9,
                ))
                continue

            # Attach the chosen OCO
            order_id = str(chosen.get('orderId', ''))
            if not order_id:
                logger.warning("bootstrap_oco: chosen OCO has no orderId for %s", symbol)
                continue

            position.bracket_order_id = order_id

            # Extract child order IDs and broker stop price
            children = chosen.get('childOrderStrategies', [])
            for child in children:
                child_id = str(child.get('orderId', ''))
                if not child_id:
                    continue

                order_type = child.get('orderType', '')
                stop_price = child.get('stopPrice')

                if order_type == 'STOP_LIMIT' or stop_price:
                    position.stop_order_id = child_id
                    if stop_price:
                        position.broker_stop_price = float(stop_price)
                elif order_type == 'LIMIT':
                    position.tp_order_id = child_id

            attached_count += 1

            self.engine._audit(
                "bootstrap_oco", symbol, "attached",
                "orphan_oco_attached",
                bracket_order_id=order_id,
                stop_order_id=position.stop_order_id,
                tp_order_id=position.tp_order_id,
                broker_stop_price=position.broker_stop_price,
                position_qty=position_qty,
            )

            _stop = position.broker_stop_price or 0.0
            self.engine.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.INFO,
                symbol=symbol,
                title=f"🔗 OCO Bracket Attached for {symbol}",
                message=f"Found and attached existing WORKING OCO order (stop @ ${_stop:.2f})",
                importance=6,  # Low-noise
            ))

        if attached_count > 0:
            self.engine._save_state()
            logger.info(
                "bootstrap_oco: attached %d orphan brackets to positions",
                attached_count,
            )

        return attached_count

    async def fetch_working_orders(self) -> Optional[list]:
        """Fetch all WORKING orders from Schwab."""
        try:
            response = self.engine.schwab_client.get_orders_for_account(
                self.engine.account_hash,
                from_entered_datetime=datetime.now() - timedelta(days=7),
                status=self.engine.schwab_client.Order.Status.WORKING
            )
            if response.status_code == 200:
                return response.json()
            logger.warning(
                "order_monitor: fetch_working_orders got status %d",
                response.status_code,
            )
            return None
        except Exception as exc:
            logger.error("order_monitor: fetch_working_orders error: %s", exc)
            return None

    async def _monitor_position_bracket(
        self,
        position: "Position",
        orders_by_id: Dict[str, dict],
        all_working_orders: list,
    ) -> None:
        """Monitor a single position's bracket orders.

        Checks:
          1. Is the bracket still WORKING? If not, why?
          2. Did stop fill? → close position, cancel TP
          3. Did TP fill? → close position, cancel stop
          4. Was either leg canceled/rejected? → re-protect or alert
          5. Has software trail moved? → replace broker stop
        """
        symbol = position.symbol
        bracket_id = position.bracket_order_id

        # Try to refresh child order IDs if we don't have them yet
        if not position.stop_order_id or not position.tp_order_id:
            await refresh_bracket_child_ids(
                self.engine, position, orders_by_id, all_working_orders
            )

        # Check if parent OCO is still WORKING
        parent_order = orders_by_id.get(bracket_id)

        if parent_order:
            # Parent still WORKING — check child statuses and trail replace
            await check_trail_replace(self.engine, position, parent_order)
        else:
            # Parent not in WORKING list — need to query it directly
            await handle_bracket_not_working(self.engine, position)
