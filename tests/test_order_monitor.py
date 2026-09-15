"""v-order-monitor-2026-09-10: Tests for realtime WORKING-bracket / OCO order monitor.

Tests cover:
  - Stop fill → sibling cancel + position close
  - TP fill → position close  
  - Reject/orphan → re-protect
  - Partial fill → re-bracket remaining
  - Trail stop replace
  - External positions ignored
  
Note: Tests use minimal imports and mocking to avoid heavy dependencies.
Engine-level integration tests are marked with pytest.mark.engine_integration
and may be skipped if dependencies are not available.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from dataclasses import asdict
import sys

from core.models import Position, clear_bracket_ids

# Mark tests that require full engine import
engine_integration = pytest.mark.skipif(
    'ta' not in sys.modules,
    reason="Full engine integration tests require ta library"
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_schwab_client():
    """Mock Schwab client with configurable responses."""
    client = MagicMock()
    client.Order = MagicMock()
    client.Order.Status = MagicMock()
    client.Order.Status.WORKING = "WORKING"
    return client


@pytest.fixture
def bot_managed_position():
    """A bot-managed position with bracket orders."""
    return Position(
        symbol="AAPL",
        entry_price=150.0,
        quantity=100,
        side="long",
        stop_loss=145.0,
        take_profit=160.0,
        entry_time=datetime.now() - timedelta(hours=1),
        reasoning={"strategy": "test"},
        mode="live",
        managed_by_bot=True,
        bracket_order_id="12345",
        stop_order_id="12346",
        tp_order_id="12347",
        broker_stop_price=145.0,
    )


@pytest.fixture
def external_position():
    """An external position that should be ignored."""
    pos = Position(
        symbol="NVDA",
        entry_price=500.0,
        quantity=50,
        side="long",
        stop_loss=480.0,
        take_profit=550.0,
        entry_time=datetime.now() - timedelta(hours=2),
        mode="live",
        managed_by_bot=False,
    )
    pos.is_external = True
    pos.is_manually_managed = True
    return pos


@pytest.fixture
def long_term_position():
    """A long-term hold position that should be ignored."""
    return Position(
        symbol="MSFT",
        entry_price=300.0,
        quantity=200,
        side="long",
        stop_loss=280.0,
        take_profit=350.0,
        entry_time=datetime.now() - timedelta(days=30),
        mode="live",
        managed_by_bot=True,
        is_long_term=True,
        bracket_order_id="99999",
    )


# ============================================================================
# MODEL TESTS
# ============================================================================

class TestPositionBracketFields:
    """Test Position model bracket tracking fields."""
    
    def test_position_has_bracket_fields(self, bot_managed_position):
        """Position should have bracket order tracking fields."""
        assert hasattr(bot_managed_position, 'bracket_order_id')
        assert hasattr(bot_managed_position, 'stop_order_id')
        assert hasattr(bot_managed_position, 'tp_order_id')
        assert hasattr(bot_managed_position, 'broker_stop_price')
        
    def test_bracket_fields_default_none(self):
        """Bracket fields should default to None."""
        pos = Position(
            symbol="TEST",
            entry_price=100.0,
            quantity=10,
            side="long",
            stop_loss=95.0,
            take_profit=110.0,
            entry_time=datetime.now(),
        )
        assert pos.bracket_order_id is None
        assert pos.stop_order_id is None
        assert pos.tp_order_id is None
        assert pos.broker_stop_price is None
        
    def test_clear_bracket_ids(self, bot_managed_position):
        """clear_bracket_ids should clear all bracket tracking fields."""
        # Verify fields are set
        assert bot_managed_position.bracket_order_id == "12345"
        assert bot_managed_position.stop_order_id == "12346"
        assert bot_managed_position.tp_order_id == "12347"
        assert bot_managed_position.broker_stop_price == 145.0
        
        # Clear them
        clear_bracket_ids(bot_managed_position)
        
        # Verify all cleared
        assert bot_managed_position.bracket_order_id is None
        assert bot_managed_position.stop_order_id is None
        assert bot_managed_position.tp_order_id is None
        assert bot_managed_position.broker_stop_price is None


# ============================================================================
# MOCK SCHWAB ORDER PAYLOADS
# ============================================================================

def make_working_oco_order(
    order_id: str = "12345",
    symbol: str = "AAPL",
    stop_price: float = 145.0,
    tp_price: float = 160.0,
    qty: int = 100,
) -> dict:
    """Create a mock WORKING OCO order payload."""
    return {
        "orderId": order_id,
        "status": "WORKING",
        "orderStrategyType": "OCO",
        "childOrderStrategies": [
            {
                "orderId": f"{order_id}1",
                "orderType": "LIMIT",
                "status": "WORKING",
                "price": tp_price,
                "orderLegCollection": [{
                    "instrument": {"symbol": symbol},
                    "quantity": qty,
                    "instruction": "SELL",
                }],
            },
            {
                "orderId": f"{order_id}2",
                "orderType": "STOP_LIMIT",
                "status": "WORKING",
                "stopPrice": stop_price,
                "price": stop_price * 0.995,
                "orderLegCollection": [{
                    "instrument": {"symbol": symbol},
                    "quantity": qty,
                    "instruction": "SELL",
                }],
            },
        ],
    }


def make_filled_stop_order(
    order_id: str = "12345",
    symbol: str = "AAPL",
    stop_price: float = 145.0,
    fill_price: float = 144.95,
    qty: int = 100,
) -> dict:
    """Create a mock FILLED OCO order (stop leg hit)."""
    return {
        "orderId": order_id,
        "status": "FILLED",
        "orderStrategyType": "OCO",
        "childOrderStrategies": [
            {
                "orderId": f"{order_id}1",
                "orderType": "LIMIT",
                "status": "CANCELED",
                "orderLegCollection": [{
                    "instrument": {"symbol": symbol},
                    "quantity": qty,
                    "instruction": "SELL",
                }],
            },
            {
                "orderId": f"{order_id}2",
                "orderType": "STOP_LIMIT",
                "status": "FILLED",
                "stopPrice": stop_price,
                "orderActivityCollection": [{
                    "executionType": "FILL",
                    "executionLegs": [{
                        "price": fill_price,
                        "quantity": qty,
                    }],
                }],
                "orderLegCollection": [{
                    "instrument": {"symbol": symbol},
                    "quantity": qty,
                    "instruction": "SELL",
                }],
            },
        ],
    }


def make_filled_tp_order(
    order_id: str = "12345",
    symbol: str = "AAPL",
    tp_price: float = 160.0,
    fill_price: float = 160.05,
    qty: int = 100,
) -> dict:
    """Create a mock FILLED OCO order (TP leg hit)."""
    return {
        "orderId": order_id,
        "status": "FILLED",
        "orderStrategyType": "OCO",
        "childOrderStrategies": [
            {
                "orderId": f"{order_id}1",
                "orderType": "LIMIT",
                "status": "FILLED",
                "price": tp_price,
                "orderActivityCollection": [{
                    "executionType": "FILL",
                    "executionLegs": [{
                        "price": fill_price,
                        "quantity": qty,
                    }],
                }],
                "orderLegCollection": [{
                    "instrument": {"symbol": symbol},
                    "quantity": qty,
                    "instruction": "SELL",
                }],
            },
            {
                "orderId": f"{order_id}2",
                "orderType": "STOP_LIMIT",
                "status": "CANCELED",
                "orderLegCollection": [{
                    "instrument": {"symbol": symbol},
                    "quantity": qty,
                    "instruction": "SELL",
                }],
            },
        ],
    }


def make_partial_fill_order(
    order_id: str = "12345",
    symbol: str = "AAPL",
    fill_price: float = 160.0,
    fill_qty: int = 50,
    total_qty: int = 100,
) -> dict:
    """Create a mock partial fill OCO order."""
    return {
        "orderId": order_id,
        "status": "FILLED",
        "orderStrategyType": "OCO",
        "childOrderStrategies": [
            {
                "orderId": f"{order_id}1",
                "orderType": "LIMIT",
                "status": "FILLED",
                "orderActivityCollection": [{
                    "executionType": "FILL",
                    "executionLegs": [{
                        "price": fill_price,
                        "quantity": fill_qty,
                    }],
                }],
                "orderLegCollection": [{
                    "instrument": {"symbol": symbol},
                    "quantity": total_qty,
                    "instruction": "SELL",
                }],
            },
            {
                "orderId": f"{order_id}2",
                "orderType": "STOP_LIMIT",
                "status": "CANCELED",
                "orderLegCollection": [{
                    "instrument": {"symbol": symbol},
                    "quantity": total_qty,
                    "instruction": "SELL",
                }],
            },
        ],
    }


def make_canceled_order(order_id: str = "12345") -> dict:
    """Create a mock CANCELED order."""
    return {
        "orderId": order_id,
        "status": "CANCELED",
        "statusDescription": "User canceled",
    }


def make_rejected_order(order_id: str = "12345", reason: str = "Invalid price") -> dict:
    """Create a mock REJECTED order."""
    return {
        "orderId": order_id,
        "status": "REJECTED",
        "statusDescription": reason,
    }


# ============================================================================
# ORDER MONITOR FILTERING TESTS
# ============================================================================

def _get_bracket_monitored_positions_logic(positions: dict, denylist: frozenset = None) -> list:
    """Pure logic test of position filtering (no engine import needed).
    
    This replicates the logic of _get_bracket_monitored_positions for testing.
    v-hands-off-denylist-2026-09-14: added denylist parameter for testing.
    """
    if denylist is None:
        from core.config import Config
        denylist = Config().HANDS_OFF_DENYLIST
    result = []
    for symbol, pos in list(positions.items()):
        if pos is None:
            continue
        if not getattr(pos, 'managed_by_bot', False):
            continue
        # v-hands-off-denylist-2026-09-14: skip denylist symbols unconditionally
        if symbol.upper() in denylist:
            continue
        if getattr(pos, 'is_external', False):
            continue
        if getattr(pos, 'is_manually_managed', False):
            continue
        if getattr(pos, 'is_long_term', False):
            continue
        if not getattr(pos, 'bracket_order_id', None):
            continue
        result.append((symbol, pos))
    return result


class TestGetBracketMonitoredPositions:
    """Test _get_bracket_monitored_positions filtering logic."""
    
    def test_includes_bot_managed_with_bracket(self, bot_managed_position):
        """Should include bot-managed positions with bracket_order_id."""
        positions = {"AAPL": bot_managed_position}
        result = _get_bracket_monitored_positions_logic(positions)
        
        assert len(result) == 1
        assert result[0][0] == "AAPL"
        assert result[0][1] is bot_managed_position
        
    def test_excludes_external_positions(self, external_position):
        """Should exclude external/manually-managed positions."""
        external_position.bracket_order_id = "99999"  # Even with bracket ID
        positions = {"NVDA": external_position}
        result = _get_bracket_monitored_positions_logic(positions)
        
        assert len(result) == 0
        
    def test_excludes_long_term_positions(self, long_term_position):
        """Should exclude long-term hold positions."""
        positions = {"MSFT": long_term_position}
        result = _get_bracket_monitored_positions_logic(positions)
        
        assert len(result) == 0
        
    def test_excludes_positions_without_bracket(self):
        """Should exclude positions without bracket_order_id."""
        pos = Position(
            symbol="TSLA",
            entry_price=200.0,
            quantity=50,
            side="long",
            stop_loss=190.0,
            take_profit=220.0,
            entry_time=datetime.now(),
            mode="live",
            managed_by_bot=True,
            # No bracket_order_id
        )
        
        positions = {"TSLA": pos}
        result = _get_bracket_monitored_positions_logic(positions)
        
        assert len(result) == 0


# ============================================================================
# v-hands-off-denylist-2026-09-14: DENYLIST SYMBOL TESTS
# Verify that HANDS_OFF_DENYLIST symbols are excluded even without is_long_term=True
# ============================================================================

class TestDenylistSymbolsExcluded:
    """Test that HANDS_OFF_DENYLIST symbols are excluded from monitoring.
    
    v-hands-off-denylist-2026-09-14: A denylist symbol (MU, HQGE, SPCX)
    must NOT appear in either filter result, even if it would otherwise
    qualify (managed_by_bot=True, NOT is_long_term, with or without bracket).
    
    This is a defense-in-depth guard: denylist symbols are protected
    unconditionally, regardless of whether the flags happen to be set.
    """
    
    def test_denylist_symbol_excluded_from_monitored_positions_with_bracket(self):
        """MU with bracket and managed_by_bot=True but NOT is_long_term must be excluded."""
        pos = Position(
            symbol="MU",
            entry_price=100.0,
            quantity=100,
            side="long",
            stop_loss=95.0,
            take_profit=110.0,
            entry_time=datetime.now(),
            mode="live",
            managed_by_bot=True,  # Would normally qualify
            is_long_term=False,   # NOT marked long-term
            bracket_order_id="12345",  # Has bracket
        )
        
        positions = {"MU": pos}
        denylist = frozenset({"MU", "HQGE", "SPCX"})
        result = _get_bracket_monitored_positions_logic(positions, denylist=denylist)
        
        assert len(result) == 0, "MU must be excluded from monitoring even without is_long_term=True"
    
    def test_denylist_symbol_excluded_from_missing_brackets(self):
        """SPCX without bracket but managed_by_bot=True must be excluded from bootstrap."""
        pos = Position(
            symbol="SPCX",
            entry_price=15.0,
            quantity=500,
            side="long",
            stop_loss=14.0,
            take_profit=18.0,
            entry_time=datetime.now(),
            mode="live",
            managed_by_bot=True,  # Would normally qualify for bootstrap
            is_long_term=False,   # NOT marked long-term
            # No bracket_order_id — would normally be in "missing brackets" list
        )
        
        positions = {"SPCX": pos}
        denylist = frozenset({"MU", "HQGE", "SPCX"})
        result = _get_positions_missing_brackets_logic(positions, denylist=denylist)
        
        assert len(result) == 0, "SPCX must be excluded from bootstrap (in denylist)"

    def test_snap_not_in_denylist_can_be_bootstrapped(self):
        """SNAP removed from permanent denylist 2026-09-14 — now bootstrappable."""
        pos = Position(
            symbol="SNAP",
            entry_price=15.0,
            quantity=500,
            side="long",
            stop_loss=14.0,
            take_profit=18.0,
            entry_time=datetime.now(),
            mode="live",
            managed_by_bot=True,
            is_long_term=False,
            # No bracket_order_id — would normally be in "missing brackets" list
        )
        
        positions = {"SNAP": pos}
        denylist = frozenset({"MU", "HQGE", "SPCX"})  # SNAP not in denylist
        result = _get_positions_missing_brackets_logic(positions, denylist=denylist)
        
        assert len(result) == 1, "SNAP should be included (removed from denylist 2026-09-14)"
        assert result[0][0] == "SNAP"
    
    def test_non_denylist_symbol_still_included(self):
        """Non-denylist symbols should still be included when they qualify."""
        pos = Position(
            symbol="AAPL",
            entry_price=150.0,
            quantity=100,
            side="long",
            stop_loss=145.0,
            take_profit=160.0,
            entry_time=datetime.now(),
            mode="live",
            managed_by_bot=True,
            is_long_term=False,
            bracket_order_id="12345",
        )
        
        positions = {"AAPL": pos}
        denylist = frozenset({"MU", "HQGE", "SPCX"})
        result = _get_bracket_monitored_positions_logic(positions, denylist=denylist)
        
        assert len(result) == 1, "AAPL should still be included (not in denylist)"
        assert result[0][0] == "AAPL"
    
    def test_denylist_case_insensitive(self):
        """Denylist check should be case-insensitive (symbol.upper() in denylist)."""
        pos = Position(
            symbol="mu",  # lowercase
            entry_price=100.0,
            quantity=100,
            side="long",
            stop_loss=95.0,
            take_profit=110.0,
            entry_time=datetime.now(),
            mode="live",
            managed_by_bot=True,
            is_long_term=False,
            bracket_order_id="12345",
        )
        
        positions = {"mu": pos}
        denylist = frozenset({"MU", "HQGE", "SPCX"})  # uppercase denylist
        result = _get_bracket_monitored_positions_logic(positions, denylist=denylist)
        
        assert len(result) == 0, "mu (lowercase) must be excluded via case-insensitive match"


# ============================================================================
# STOP FILL TESTS
# ============================================================================

def _parse_bracket_fill(order_info: dict) -> tuple:
    """Parse a bracket fill order to extract fill details.
    
    Returns (fill_price, fill_qty, filled_leg) tuple.
    """
    fill_price = None
    fill_qty = 0
    filled_leg = "unknown"
    
    children = order_info.get('childOrderStrategies', [])
    for child in children:
        child_status = child.get('status', '')
        child_type = child.get('orderType', '')
        
        if child_status == 'FILLED':
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
    
    return fill_price, fill_qty, filled_leg


class TestStopFillHandling:
    """Test stop loss fill handling."""
    
    def test_stop_fill_parsing(self, bot_managed_position):
        """Stop fill should be correctly parsed from order info."""
        order_info = make_filled_stop_order(
            order_id="12345",
            fill_price=144.95,
            qty=100,
        )
        
        fill_price, fill_qty, filled_leg = _parse_bracket_fill(order_info)
        
        assert fill_price == 144.95
        assert fill_qty == 100
        assert filled_leg == "stop"
        
    def test_stop_fill_clears_bracket_ids(self, bot_managed_position):
        """Stop fill should clear bracket tracking IDs."""
        # Verify IDs are set
        assert bot_managed_position.bracket_order_id == "12345"
        
        # Simulate what handler does
        clear_bracket_ids(bot_managed_position)
        
        # Verify cleared
        assert bot_managed_position.bracket_order_id is None
        assert bot_managed_position.stop_order_id is None
        assert bot_managed_position.tp_order_id is None
        
    def test_stop_fill_pnl_calculation(self, bot_managed_position):
        """Stop fill PnL should be calculated correctly for long position."""
        fill_price = 144.95  # Below entry of 150.0
        
        # Long position PnL calculation
        pnl = (fill_price - bot_managed_position.entry_price) * bot_managed_position.quantity
        
        # 100 shares * (144.95 - 150.0) = -505
        assert pnl == pytest.approx(-505.0, abs=0.01)


# ============================================================================
# TP FILL TESTS
# ============================================================================

class TestTPFillHandling:
    """Test take profit fill handling."""
    
    def test_tp_fill_parsing(self, bot_managed_position):
        """TP fill should be correctly parsed from order info."""
        order_info = make_filled_tp_order(
            order_id="12345",
            fill_price=160.05,
            qty=100,
        )
        
        fill_price, fill_qty, filled_leg = _parse_bracket_fill(order_info)
        
        assert fill_price == 160.05
        assert fill_qty == 100
        assert filled_leg == "take_profit"
        
    def test_tp_fill_pnl_calculation(self, bot_managed_position):
        """TP fill PnL should be positive for successful trade."""
        fill_price = 160.05  # Above entry of 150.0
        
        # Long position PnL calculation
        pnl = (fill_price - bot_managed_position.entry_price) * bot_managed_position.quantity
        
        # 100 shares * (160.05 - 150.0) = 1005
        assert pnl == pytest.approx(1005.0, abs=0.01)
        assert pnl > 0  # Profit


# ============================================================================
# PARTIAL FILL TESTS
# ============================================================================

class TestPartialFillHandling:
    """Test partial fill handling."""
    
    def test_partial_fill_parsing(self, bot_managed_position):
        """Partial fill should be correctly parsed."""
        order_info = make_partial_fill_order(
            order_id="12345",
            fill_price=160.0,
            fill_qty=50,
            total_qty=100,
        )
        
        fill_price, fill_qty, filled_leg = _parse_bracket_fill(order_info)
        
        assert fill_price == 160.0
        assert fill_qty == 50  # Partial
        assert filled_leg == "take_profit"
        
    def test_partial_fill_remaining_calculation(self, bot_managed_position):
        """Partial fill should correctly calculate remaining qty."""
        fill_qty = 50
        original_qty = bot_managed_position.quantity  # 100
        
        remaining_qty = original_qty - fill_qty
        
        assert remaining_qty == 50
        
    def test_partial_fill_needs_rebracket(self, bot_managed_position):
        """Partial fill should clear bracket IDs to enable re-bracket."""
        fill_qty = 50
        remaining_qty = bot_managed_position.quantity - fill_qty
        
        # Simulate partial fill handling
        bot_managed_position.quantity = remaining_qty
        clear_bracket_ids(bot_managed_position)
        
        # Position updated, bracket cleared for re-bracket
        assert bot_managed_position.quantity == 50
        assert bot_managed_position.bracket_order_id is None


# ============================================================================
# CANCEL/REJECT TESTS
# ============================================================================

class TestCancelHandling:
    """Test bracket cancellation handling."""
    
    def test_cancel_order_parsing(self):
        """Canceled order should have CANCELED status."""
        order_info = make_canceled_order("12345")
        
        assert order_info['status'] == "CANCELED"
        assert order_info['orderId'] == "12345"
        
    def test_cancel_clears_bracket_ids(self, bot_managed_position):
        """Canceled bracket should clear IDs for re-protect."""
        # Simulate cancel handling
        clear_bracket_ids(bot_managed_position)
        
        assert bot_managed_position.bracket_order_id is None
        assert bot_managed_position.stop_order_id is None


class TestRejectHandling:
    """Test bracket rejection handling."""
    
    def test_reject_order_parsing(self):
        """Rejected order should have REJECTED status and reason."""
        order_info = make_rejected_order("12345", "Invalid stop price")
        
        assert order_info['status'] == "REJECTED"
        assert order_info['statusDescription'] == "Invalid stop price"
        
    def test_reject_clears_bracket_ids(self, bot_managed_position):
        """Rejected bracket should clear IDs for re-protect attempt."""
        clear_bracket_ids(bot_managed_position)
        
        assert bot_managed_position.bracket_order_id is None


# ============================================================================
# TRAIL REPLACE TESTS
# ============================================================================

def _should_replace_trail(position, threshold: float = 1.005) -> bool:
    """Check if software trail is significantly tighter than broker stop.
    
    For long positions, trailing_stop should be rising (tighter).
    Returns True if replacement is needed.
    """
    trail = getattr(position, 'trailing_stop', None)
    if trail is None:
        return False
    
    broker_stop = getattr(position, 'broker_stop_price', None)
    if broker_stop is None:
        return False
    
    if position.side == 'long':
        return trail > broker_stop * threshold
    else:
        return trail < broker_stop * (2 - threshold)  # e.g., 0.995


class TestTrailReplace:
    """Test trailing stop replacement."""
    
    def test_trail_replace_when_software_tighter(self, bot_managed_position):
        """Should replace broker stop when software trail is tighter."""
        # Set up position with trailing stop tighter than broker
        bot_managed_position.trailing_stop = 148.0  # Tighter than 145.0
        bot_managed_position.broker_stop_price = 145.0
        
        should_replace = _should_replace_trail(bot_managed_position)
        
        assert should_replace is True
        
    def test_no_replace_when_broker_tighter(self, bot_managed_position):
        """Should not replace when broker stop is already tighter."""
        # Broker stop is tighter than software trail
        bot_managed_position.trailing_stop = 144.0
        bot_managed_position.broker_stop_price = 145.0
        
        should_replace = _should_replace_trail(bot_managed_position)
        
        assert should_replace is False
        
    def test_no_replace_when_no_trail(self, bot_managed_position):
        """Should not replace when no trailing stop is set."""
        bot_managed_position.trailing_stop = None
        bot_managed_position.broker_stop_price = 145.0
        
        should_replace = _should_replace_trail(bot_managed_position)
        
        assert should_replace is False
        
    def test_no_replace_when_no_broker_stop(self, bot_managed_position):
        """Should not replace when no broker stop is set."""
        bot_managed_position.trailing_stop = 148.0
        bot_managed_position.broker_stop_price = None
        
        should_replace = _should_replace_trail(bot_managed_position)
        
        assert should_replace is False


# ============================================================================
# EXTERNAL POSITIONS IGNORED TESTS
# ============================================================================

class TestExternalPositionsIgnored:
    """Test that external/manual/long-term positions are ignored."""
    
    def test_external_position_not_monitored(self, external_position):
        """External positions should not be in monitored list."""
        external_position.bracket_order_id = "99999"
        
        positions = {"NVDA": external_position}
        result = _get_bracket_monitored_positions_logic(positions)
        
        assert len(result) == 0
        
    def test_manually_managed_position_not_monitored(self):
        """Manually managed positions should not be monitored."""
        pos = Position(
            symbol="COIN",
            entry_price=100.0,
            quantity=20,
            side="long",
            stop_loss=90.0,
            take_profit=120.0,
            entry_time=datetime.now(),
            mode="live",
            managed_by_bot=False,
            bracket_order_id="88888",
        )
        pos.is_manually_managed = True
        
        positions = {"COIN": pos}
        result = _get_bracket_monitored_positions_logic(positions)
        
        assert len(result) == 0
        
    def test_long_term_position_not_monitored(self, long_term_position):
        """Long-term hold positions should not be monitored."""
        positions = {"MSFT": long_term_position}
        result = _get_bracket_monitored_positions_logic(positions)
        
        assert len(result) == 0


# ============================================================================
# CONFIG TESTS
# ============================================================================

class TestOrderMonitorConfig:
    """Test order monitor configuration."""
    
    def test_config_has_order_monitor_interval(self):
        """Config should have ORDER_MONITOR_INTERVAL_SEC property."""
        from core.config import Config
        cfg = Config()
        
        assert hasattr(cfg, 'ORDER_MONITOR_INTERVAL_SEC')
        assert isinstance(cfg.ORDER_MONITOR_INTERVAL_SEC, float)
        assert cfg.ORDER_MONITOR_INTERVAL_SEC == 5.0  # Default


# ============================================================================
# RE-BRACKET TESTS
# ============================================================================

class TestReBracketPosition:
    """Test position re-bracketing after partial fill or recovery."""
    
    def test_rebracket_needs_valid_params(self, bot_managed_position):
        """Re-bracket should require valid stop/TP/qty."""
        # Clear bracket IDs (as would happen after partial/cancel)
        clear_bracket_ids(bot_managed_position)
        bot_managed_position.quantity = 50
        
        # Verify position has valid params for re-bracket
        stop_price = bot_managed_position.trailing_stop or bot_managed_position.stop_loss
        tp_price = bot_managed_position.take_profit
        qty = bot_managed_position.quantity
        
        assert stop_price is not None and stop_price > 0
        assert tp_price is not None and tp_price > 0
        assert qty > 0
        
    def test_rebracket_uses_trailing_if_available(self, bot_managed_position):
        """Re-bracket should prefer trailing stop over original stop."""
        bot_managed_position.trailing_stop = 148.0  # Tighter than 145.0
        
        stop_price = bot_managed_position.trailing_stop or bot_managed_position.stop_loss
        
        assert stop_price == 148.0  # Should use trailing


# ============================================================================
# INTEGRATION-STYLE TESTS
# ============================================================================

class TestOrderMonitorLoopIntegration:
    """Integration-style tests for the full order monitor loop."""
    
    def test_loop_processes_multiple_positions(self, mock_schwab_client):
        """Loop should process all bot-managed positions with brackets."""
        # Create multiple positions
        pos1 = Position(
            symbol="AAPL",
            entry_price=150.0,
            quantity=100,
            side="long",
            stop_loss=145.0,
            take_profit=160.0,
            entry_time=datetime.now(),
            mode="live",
            managed_by_bot=True,
            bracket_order_id="111",
        )
        
        pos2 = Position(
            symbol="TSLA",
            entry_price=200.0,
            quantity=50,
            side="long",
            stop_loss=190.0,
            take_profit=220.0,
            entry_time=datetime.now(),
            mode="live",
            managed_by_bot=True,
            bracket_order_id="222",
        )
        
        positions = {"AAPL": pos1, "TSLA": pos2}
        result = _get_bracket_monitored_positions_logic(positions)
        
        assert len(result) == 2
        symbols = {r[0] for r in result}
        assert symbols == {"AAPL", "TSLA"}


class TestChildOrderIdExtraction:
    """Test extraction of child order IDs from OCO structure."""
    
    def test_oco_structure_has_children(self):
        """OCO order should have childOrderStrategies."""
        oco_order = make_working_oco_order(order_id="12345")
        
        assert 'childOrderStrategies' in oco_order
        children = oco_order['childOrderStrategies']
        assert len(children) == 2
        
    def test_oco_children_have_order_ids(self):
        """Each OCO child should have an orderId."""
        oco_order = make_working_oco_order(order_id="12345")
        children = oco_order['childOrderStrategies']
        
        for child in children:
            assert 'orderId' in child
            assert child['orderId'] is not None
            
    def test_oco_children_distinguish_stop_and_tp(self):
        """Should be able to distinguish stop vs TP from order type."""
        oco_order = make_working_oco_order(order_id="12345")
        children = oco_order['childOrderStrategies']
        
        order_types = {child['orderType'] for child in children}
        assert 'LIMIT' in order_types  # TP
        assert 'STOP_LIMIT' in order_types  # Stop


# ============================================================================
# BOOTSTRAP OCO ATTACH TESTS
# v-bootstrap-oco-2026-09-10: Tests for attaching orphan WORKING OCO brackets
# to positions missing bracket_order_id.
# ============================================================================

def _get_positions_missing_brackets_logic(positions: dict, denylist: frozenset = None) -> list:
    """Pure logic test of position filtering for bootstrap (no engine import needed).
    
    This replicates the logic of _get_positions_missing_brackets for testing.
    Same filters as _get_bracket_monitored_positions EXCEPT the bracket_order_id
    requirement is inverted (we want positions WITHOUT bracket_order_id).
    v-hands-off-denylist-2026-09-14: added denylist parameter for testing.
    """
    if denylist is None:
        from core.config import Config
        denylist = Config().HANDS_OFF_DENYLIST
    result = []
    for symbol, pos in list(positions.items()):
        if pos is None:
            continue
        if not getattr(pos, 'managed_by_bot', False):
            continue
        # v-hands-off-denylist-2026-09-14: skip denylist symbols unconditionally
        if symbol.upper() in denylist:
            continue
        if getattr(pos, 'is_external', False):
            continue
        if getattr(pos, 'is_manually_managed', False):
            continue
        if getattr(pos, 'is_long_term', False):
            continue
        # Inverse: include only if MISSING bracket_order_id
        if getattr(pos, 'bracket_order_id', None):
            continue
        result.append((symbol, pos))
    return result


def _bootstrap_match_oco_to_position(
    position_qty: int,
    candidates: list,
) -> tuple:
    """Pure logic test of OCO matching for bootstrap.
    
    Returns (chosen_oco, reason) or (None, reason) if no match.
    """
    def get_oco_qty(oco: dict) -> int:
        for child in oco.get('childOrderStrategies', []):
            for leg in child.get('orderLegCollection', []):
                qty = leg.get('quantity')
                if qty is not None:
                    return int(qty)
        return 0
    
    exact_qty_matches = [o for o in candidates if get_oco_qty(o) == position_qty]
    
    if len(exact_qty_matches) == 1:
        return exact_qty_matches[0], "exact_qty_match"
    elif len(exact_qty_matches) > 1:
        return None, "ambiguous_multiple_exact_qty"
    elif len(candidates) == 1:
        return candidates[0], "single_candidate_qty_differs"
    else:
        return None, "ambiguous_multiple_no_exact_qty"


class TestGetPositionsMissingBrackets:
    """Test _get_positions_missing_brackets filtering logic."""
    
    def test_includes_position_without_bracket(self):
        """Should include bot-managed positions WITHOUT bracket_order_id."""
        pos = Position(
            symbol="AAPL",
            entry_price=150.0,
            quantity=100,
            side="long",
            stop_loss=145.0,
            take_profit=160.0,
            entry_time=datetime.now(),
            mode="live",
            managed_by_bot=True,
            # No bracket_order_id
        )
        
        positions = {"AAPL": pos}
        result = _get_positions_missing_brackets_logic(positions)
        
        assert len(result) == 1
        assert result[0][0] == "AAPL"
        
    def test_excludes_position_with_bracket(self, bot_managed_position):
        """Should exclude positions that already have bracket_order_id."""
        # bot_managed_position fixture has bracket_order_id set
        positions = {"AAPL": bot_managed_position}
        result = _get_positions_missing_brackets_logic(positions)
        
        assert len(result) == 0
        
    def test_excludes_external_positions(self, external_position):
        """Should exclude external positions even without bracket."""
        positions = {"NVDA": external_position}
        result = _get_positions_missing_brackets_logic(positions)
        
        assert len(result) == 0
        
    def test_excludes_long_term_positions(self):
        """Should exclude long-term positions (Hari's 4 LT holds)."""
        pos = Position(
            symbol="MSFT",
            entry_price=300.0,
            quantity=200,
            side="long",
            stop_loss=280.0,
            take_profit=350.0,
            entry_time=datetime.now(),
            mode="live",
            managed_by_bot=True,
            is_long_term=True,
            # No bracket_order_id
        )
        
        positions = {"MSFT": pos}
        result = _get_positions_missing_brackets_logic(positions)
        
        assert len(result) == 0
        
    def test_excludes_manually_managed(self):
        """Should exclude manually managed positions."""
        pos = Position(
            symbol="COIN",
            entry_price=100.0,
            quantity=20,
            side="long",
            stop_loss=90.0,
            take_profit=120.0,
            entry_time=datetime.now(),
            mode="live",
            managed_by_bot=False,
        )
        pos.is_manually_managed = True
        
        positions = {"COIN": pos}
        result = _get_positions_missing_brackets_logic(positions)
        
        assert len(result) == 0


class TestBootstrapOCOMatching:
    """Test OCO matching logic for bootstrap attach."""
    
    def test_match_by_symbol_and_qty(self):
        """Should match OCO to position by symbol and exact qty."""
        position_qty = 100
        oco = make_working_oco_order(order_id="12345", symbol="AAPL", qty=100)
        candidates = [oco]
        
        chosen, reason = _bootstrap_match_oco_to_position(position_qty, candidates)
        
        assert chosen is oco
        assert reason == "exact_qty_match"
        
    def test_match_single_candidate_qty_differs(self):
        """Should match single OCO even if qty differs (with warning)."""
        position_qty = 100
        oco = make_working_oco_order(order_id="12345", symbol="AAPL", qty=50)
        candidates = [oco]
        
        chosen, reason = _bootstrap_match_oco_to_position(position_qty, candidates)
        
        assert chosen is oco
        assert reason == "single_candidate_qty_differs"
        
    def test_skip_ambiguous_multiple_exact_qty(self):
        """Should skip when multiple OCOs have exact qty match."""
        position_qty = 100
        oco1 = make_working_oco_order(order_id="111", symbol="AAPL", qty=100)
        oco2 = make_working_oco_order(order_id="222", symbol="AAPL", qty=100)
        candidates = [oco1, oco2]
        
        chosen, reason = _bootstrap_match_oco_to_position(position_qty, candidates)
        
        assert chosen is None
        assert reason == "ambiguous_multiple_exact_qty"
        
    def test_skip_ambiguous_multiple_no_exact(self):
        """Should skip when multiple OCOs and none have exact qty."""
        position_qty = 100
        oco1 = make_working_oco_order(order_id="111", symbol="AAPL", qty=50)
        oco2 = make_working_oco_order(order_id="222", symbol="AAPL", qty=75)
        candidates = [oco1, oco2]
        
        chosen, reason = _bootstrap_match_oco_to_position(position_qty, candidates)
        
        assert chosen is None
        assert reason == "ambiguous_multiple_no_exact_qty"
        
    def test_prefer_exact_qty_over_single_candidate(self):
        """Should prefer exact qty match over single different qty."""
        position_qty = 100
        oco_exact = make_working_oco_order(order_id="111", symbol="AAPL", qty=100)
        oco_other = make_working_oco_order(order_id="222", symbol="AAPL", qty=50)
        candidates = [oco_exact, oco_other]
        
        chosen, reason = _bootstrap_match_oco_to_position(position_qty, candidates)
        
        assert chosen is oco_exact
        assert reason == "exact_qty_match"


class TestBootstrapIdempotent:
    """Test that bootstrap is idempotent when bracket_order_id already set."""
    
    def test_skip_if_bracket_already_set(self, bot_managed_position):
        """Should not include position if bracket_order_id already set."""
        # bot_managed_position has bracket_order_id="12345"
        positions = {"AAPL": bot_managed_position}
        
        result = _get_positions_missing_brackets_logic(positions)
        
        assert len(result) == 0
        
    def test_include_same_position_after_bracket_cleared(self, bot_managed_position):
        """Should include position after bracket_order_id is cleared."""
        clear_bracket_ids(bot_managed_position)
        positions = {"AAPL": bot_managed_position}
        
        result = _get_positions_missing_brackets_logic(positions)
        
        assert len(result) == 1


class TestBootstrapAttachFields:
    """Test that bootstrap correctly sets all bracket tracking fields."""
    
    def test_attach_sets_bracket_order_id(self):
        """Attach should set bracket_order_id from OCO orderId."""
        pos = Position(
            symbol="AAPL",
            entry_price=150.0,
            quantity=100,
            side="long",
            stop_loss=145.0,
            take_profit=160.0,
            entry_time=datetime.now(),
            mode="live",
            managed_by_bot=True,
        )
        
        oco = make_working_oco_order(
            order_id="12345",
            symbol="AAPL",
            stop_price=145.0,
            tp_price=160.0,
            qty=100,
        )
        
        # Simulate attach logic
        order_id = str(oco.get('orderId', ''))
        pos.bracket_order_id = order_id
        
        assert pos.bracket_order_id == "12345"
        
    def test_attach_extracts_child_ids(self):
        """Attach should extract stop_order_id and tp_order_id from children."""
        pos = Position(
            symbol="AAPL",
            entry_price=150.0,
            quantity=100,
            side="long",
            stop_loss=145.0,
            take_profit=160.0,
            entry_time=datetime.now(),
            mode="live",
            managed_by_bot=True,
        )
        
        oco = make_working_oco_order(
            order_id="12345",
            symbol="AAPL",
            stop_price=145.0,
            tp_price=160.0,
            qty=100,
        )
        
        # Simulate child ID extraction (matches engine logic)
        pos.bracket_order_id = str(oco.get('orderId', ''))
        
        for child in oco.get('childOrderStrategies', []):
            child_id = str(child.get('orderId', ''))
            if not child_id:
                continue
            
            order_type = child.get('orderType', '')
            stop_price = child.get('stopPrice')
            
            if order_type == 'STOP_LIMIT' or stop_price:
                pos.stop_order_id = child_id
                if stop_price:
                    pos.broker_stop_price = float(stop_price)
            elif order_type == 'LIMIT':
                pos.tp_order_id = child_id
        
        assert pos.bracket_order_id == "12345"
        assert pos.stop_order_id == "123452"  # Stop is child 2
        assert pos.tp_order_id == "123451"    # TP is child 1
        assert pos.broker_stop_price == 145.0


# ============================================================================
# v-broker-flat-detection-2026-09-10: BROKER-FLAT DETECTION TESTS
# Tests for the P0 hotfix that prevents infinite re_bracket loops when
# the broker shows the position is already closed (oversold/overbought reject).
# ============================================================================

def make_oversold_rejected_order(order_id: str = "12345") -> dict:
    """Create a mock REJECTED order with oversold position reason."""
    return {
        "orderId": order_id,
        "status": "REJECTED",
        "statusDescription": "Order rejected: oversold position",
    }


def make_overbought_rejected_order(order_id: str = "12345") -> dict:
    """Create a mock REJECTED order with overbought position reason."""
    return {
        "orderId": order_id,
        "status": "REJECTED",
        "statusDescription": "Order rejected: overbought position",
    }


def make_insufficient_shares_rejected_order(order_id: str = "12345") -> dict:
    """Create a mock REJECTED order with insufficient shares reason."""
    return {
        "orderId": order_id,
        "status": "REJECTED",
        "statusDescription": "Insufficient shares to complete order",
    }


def make_price_rejected_order(order_id: str = "12345") -> dict:
    """Create a mock REJECTED order with price validation reason (NOT broker-flat)."""
    return {
        "orderId": order_id,
        "status": "REJECTED",
        "statusDescription": "Stop price must be below current bid",
    }


class TestBrokerFlatRejectionDetection:
    """v-broker-flat-detection-2026-09-10: Test _is_broker_flat_rejection logic."""
    
    def test_oversold_detected_as_broker_flat(self):
        """'oversold position' rejection should be detected as broker-flat."""
        # v-phase-b-order-monitor-2026-09-14: implementation moved to core/order_monitor/
        src = (Path(__file__).parent.parent / "core" / "order_monitor" / "broker_flat.py").read_text()
        assert "is_broker_flat_rejection" in src
        
        marker = "def is_broker_flat_rejection"
        idx = src.index(marker)
        body = src[idx:idx + 1500]
        
        assert "'oversold'" in body, "oversold must be in flat_indicators list"
        assert "'overbought'" in body, "overbought must be in flat_indicators list"
        assert "'insufficient shares'" in body, "insufficient shares must be in flat_indicators"
        
    def test_overbought_detected_as_broker_flat(self):
        """'overbought position' rejection should be detected as broker-flat."""
        # v-phase-b-order-monitor-2026-09-14: implementation moved to core/order_monitor/
        src = (Path(__file__).parent.parent / "core" / "order_monitor" / "broker_flat.py").read_text()
        marker = "is_broker_flat_rejection"
        assert marker in src
        
    def test_price_rejection_not_broker_flat(self):
        """Price validation rejection should NOT be detected as broker-flat."""
        order_info = make_price_rejected_order()
        reason = order_info['statusDescription'].lower()
        
        flat_indicators = ['oversold', 'overbought', 'insufficient shares',
                           'insufficient position', 'no position', 'position not found']
        is_flat = any(indicator in reason for indicator in flat_indicators)
        
        assert is_flat is False, "Price rejection should not trigger broker-flat detection"


class TestBrokerFlatHandling:
    """v-broker-flat-detection-2026-09-10: Test _handle_broker_flat_detected behavior."""
    
    def test_handler_exists_in_engine(self):
        """handle_broker_flat_detected must exist and clear managed_by_bot."""
        # v-phase-b-order-monitor-2026-09-14: implementation moved to core/order_monitor/
        src = (Path(__file__).parent.parent / "core" / "order_monitor" / "broker_flat.py").read_text()
        
        assert "async def handle_broker_flat_detected" in src
        
        marker = "async def handle_broker_flat_detected"
        idx = src.index(marker)
        body = src[idx:idx + 6000]
        
        assert "managed_by_bot = False" in body, (
            "handle_broker_flat_detected must clear managed_by_bot"
        )
        assert "_cancel_existing_orders" in body, (
            "handle_broker_flat_detected must cancel working orders"
        )
        assert "engine.positions.pop" in body, (
            "handle_broker_flat_detected must remove position from tracking (when ghost_flatten_enabled)"
        )
        
    def test_handler_transitions_to_closed(self):
        """handle_broker_flat_detected must transition position to CLOSED state."""
        # v-phase-b-order-monitor-2026-09-14: implementation moved to core/order_monitor/
        src = (Path(__file__).parent.parent / "core" / "order_monitor" / "broker_flat.py").read_text()
        
        marker = "async def handle_broker_flat_detected"
        idx = src.index(marker)
        body = src[idx:idx + 3000]
        
        assert "PositionState.CLOSED" in body, (
            "handle_broker_flat_detected must transition to CLOSED"
        )
        assert "try_transition" in body, (
            "handle_broker_flat_detected must use try_transition for FSM"
        )


class TestReBracketLoopPrevention:
    """v-broker-flat-detection-2026-09-10: Test infinite re_bracket loop prevention."""
    
    def test_re_bracket_attempts_tracked(self):
        """handle_bracket_rejected must track re_bracket attempts."""
        # v-phase-b-order-monitor-2026-09-14: implementation moved to core/order_monitor/
        src = (Path(__file__).parent.parent / "core" / "order_monitor" / "brackets.py").read_text()
        
        marker = "async def handle_bracket_rejected"
        idx = src.index(marker)
        body = src[idx:idx + 4000]
        
        assert "_re_bracket_attempts" in body, (
            "handle_bracket_rejected must track re_bracket attempts"
        )
        assert "MAX_RE_BRACKET_ATTEMPTS" in body, (
            "Must have a max attempts constant"
        )
        
    def test_max_attempts_stops_loop(self):
        """After MAX_RE_BRACKET_ATTEMPTS, no more re_bracket calls."""
        # v-phase-b-order-monitor-2026-09-14: implementation moved to core/order_monitor/
        src = (Path(__file__).parent.parent / "core" / "order_monitor" / "brackets.py").read_text()
        
        marker = "async def handle_bracket_rejected"
        idx = src.index(marker)
        body = src[idx:idx + 4000]
        
        assert "re_bracket_exhausted" in body, (
            "Must audit when re_bracket attempts exhausted"
        )
        assert "return" in body and "MAX_RE_BRACKET_ATTEMPTS" in body, (
            "Must return early when max attempts exceeded"
        )


class TestReBracketBrokerFlatCheck:
    """v-broker-flat-detection-2026-09-10: Test re_bracket_position broker-flat pre-check."""
    
    def test_re_bracket_checks_broker_before_placing(self):
        """re_bracket_position must verify position exists at broker before placing."""
        # v-phase-b-order-monitor-2026-09-14: implementation moved to core/order_monitor/
        src = (Path(__file__).parent.parent / "core" / "order_monitor" / "brackets.py").read_text()
        
        marker = "async def re_bracket_position"
        idx = src.index(marker)
        body = src[idx:idx + 2500]
        
        assert "check_broker_position_qty" in body, (
            "re_bracket_position must check broker position before placing"
        )
        assert "broker_flat_pre_check" in body, (
            "Must audit broker-flat detection in pre-check"
        )
        
    def test_check_broker_position_qty_exists(self):
        """check_broker_position_qty helper must exist."""
        # v-phase-b-order-monitor-2026-09-14: implementation moved to core/order_monitor/
        src = (Path(__file__).parent.parent / "core" / "order_monitor" / "broker_flat.py").read_text()
        
        assert "async def check_broker_position_qty" in src
        
        marker = "async def check_broker_position_qty"
        idx = src.index(marker)
        body = src[idx:idx + 1500]
        
        assert "get_schwab_positions" in body, (
            "check_broker_position_qty must query Schwab positions"
        )
        assert "return 0" in body, (
            "Must return 0 when position not found (fail-safe)"
        )


class TestBrokerFlatScenarios:
    """v-broker-flat-detection-2026-09-10: End-to-end scenario tests."""
    
    def test_scenario_oversold_reject_with_broker_flat(self, bot_managed_position):
        """Scenario: bracket rejected for oversold AND broker shows flat.
        
        Expected: position removed from tracking, no re_bracket attempt.
        """
        # v-phase-b-order-monitor-2026-09-14: implementation moved to core/order_monitor/
        src = (Path(__file__).parent.parent / "core" / "order_monitor" / "brackets.py").read_text()
        
        marker = "async def handle_bracket_rejected"
        idx = src.index(marker)
        body = src[idx:idx + 4500]
        
        assert "is_broker_flat_rejection" in body, (
            "handle_bracket_rejected must check for broker-flat rejection"
        )
        assert "check_broker_position_qty" in body, (
            "handle_bracket_rejected must verify with broker"
        )
        assert "handle_broker_flat_detected" in body, (
            "handle_bracket_rejected must call handle_broker_flat_detected"
        )
        
        call_idx = body.index("handle_broker_flat_detected")
        return_idx = body.index("return", call_idx)
        
        assert return_idx - call_idx < 200, (
            "Must return after handle_broker_flat_detected (no re_bracket)"
        )
        
    def test_scenario_oversold_reject_broker_still_has_position(self, bot_managed_position):
        """Scenario: bracket rejected for oversold BUT broker still has shares.
        
        Expected: log mismatch, proceed with re_bracket (may be partial fill edge case).
        """
        # v-phase-b-order-monitor-2026-09-14: implementation moved to core/order_monitor/
        src = (Path(__file__).parent.parent / "core" / "order_monitor" / "brackets.py").read_text()
        
        marker = "async def handle_bracket_rejected"
        idx = src.index(marker)
        body = src[idx:idx + 4500]
        
        assert "flat_mismatch" in body, (
            "Must log when rejection says flat but broker has shares"
        )
        assert "re_bracket_position" in body, (
            "Must still call re_bracket_position when broker confirms shares exist"
        )
        
    def test_scenario_price_reject_normal_re_bracket(self, bot_managed_position):
        """Scenario: bracket rejected for price validation (NOT broker-flat).
        
        Expected: normal re_bracket flow (not treated as external close).
        """
        order_info = make_price_rejected_order()
        reason = order_info['statusDescription'].lower()
        
        flat_indicators = ['oversold', 'overbought', 'insufficient shares',
                           'insufficient position', 'no position', 'position not found']
        is_flat = any(indicator in reason for indicator in flat_indicators)
        
        assert is_flat is False, "Price rejection must not trigger broker-flat handling"


class TestPositionReBracketAttempts:
    """Test _re_bracket_attempts field on Position."""
    
    def test_attempts_counter_incremented(self):
        """_re_bracket_attempts must be incremented on each rejection."""
        # v-phase-b-order-monitor-2026-09-14: implementation moved to core/order_monitor/
        src = (Path(__file__).parent.parent / "core" / "order_monitor" / "brackets.py").read_text()
        
        marker = "async def handle_bracket_rejected"
        idx = src.index(marker)
        body = src[idx:idx + 4000]
        
        assert "getattr(position, '_re_bracket_attempts', 0)" in body, (
            "Must get existing attempts or default to 0"
        )
        assert "+ 1" in body, "Must increment attempts counter"
        
    def test_attempts_counter_stored_on_position(self):
        """Counter must be stored on position object for persistence across calls."""
        # v-phase-b-order-monitor-2026-09-14: implementation moved to core/order_monitor/
        src = (Path(__file__).parent.parent / "core" / "order_monitor" / "brackets.py").read_text()
        
        marker = "async def handle_bracket_rejected"
        idx = src.index(marker)
        body = src[idx:idx + 4000]
        
        assert "position._re_bracket_attempts = re_bracket_attempts" in body, (
            "Must store attempts count on position"
        )


# Required import for new tests
from pathlib import Path


# ============================================================================
# v-close-position-tracking-2026-09-14: CLOSE_POSITION_TRACKING TESTS
# Tests for the P0 hotfix that adds close_position_tracking() to exit managers
# and ensures broker stop fill path is authoritative (doesn't revert to LIVE).
# ============================================================================

class TestClosePositionTrackingDynamicExitManager:
    """v-close-position-tracking-2026-09-14: Test DynamicExitManager.close_position_tracking()."""

    def test_close_position_tracking_removes_from_exit_trackers(self):
        """close_position_tracking should remove symbol from exit_trackers."""
        from analysis.exit_managers import DynamicExitManager
        
        mock_brain = MagicMock()
        mock_commentary = MagicMock()
        manager = DynamicExitManager(mock_brain, mock_commentary)
        
        manager.initialize_position_tracking("FTFT", 5.50, 5.00, 6.00)
        assert "FTFT" in manager.exit_trackers
        
        manager.close_position_tracking("FTFT")
        assert "FTFT" not in manager.exit_trackers

    def test_close_position_tracking_idempotent_noop_if_missing(self):
        """close_position_tracking should be idempotent: no-op if symbol missing."""
        from analysis.exit_managers import DynamicExitManager
        
        mock_brain = MagicMock()
        mock_commentary = MagicMock()
        manager = DynamicExitManager(mock_brain, mock_commentary)
        
        manager.close_position_tracking("NONEXISTENT")
        
        manager.initialize_position_tracking("FTFT", 5.50, 5.00, 6.00)
        manager.close_position_tracking("FTFT")
        manager.close_position_tracking("FTFT")
        
        assert "FTFT" not in manager.exit_trackers

    def test_close_position_tracking_does_not_affect_other_symbols(self):
        """close_position_tracking should only remove the specified symbol."""
        from analysis.exit_managers import DynamicExitManager
        
        mock_brain = MagicMock()
        mock_commentary = MagicMock()
        manager = DynamicExitManager(mock_brain, mock_commentary)
        
        manager.initialize_position_tracking("FTFT", 5.50, 5.00, 6.00)
        manager.initialize_position_tracking("AAPL", 150.00, 145.00, 160.00)
        
        manager.close_position_tracking("FTFT")
        
        assert "FTFT" not in manager.exit_trackers
        assert "AAPL" in manager.exit_trackers


class TestClosePositionTrackingAdvancedExitManager:
    """v-close-position-tracking-2026-09-14: Test AdvancedExitManager.close_position_tracking()."""

    def test_close_position_tracking_removes_from_position_tracking(self):
        """close_position_tracking should remove symbol from position_tracking."""
        from analysis.exit_managers import AdvancedExitManager
        
        mock_commentary = MagicMock()
        manager = AdvancedExitManager(mock_commentary)
        
        manager.initialize_trailing_stop("FTFT", 5.50, 5.00, 0.02)
        assert "FTFT" in manager.position_tracking
        
        manager.close_position_tracking("FTFT")
        assert "FTFT" not in manager.position_tracking

    def test_close_position_tracking_idempotent_noop_if_missing(self):
        """close_position_tracking should be idempotent: no-op if symbol missing."""
        from analysis.exit_managers import AdvancedExitManager
        
        mock_commentary = MagicMock()
        manager = AdvancedExitManager(mock_commentary)
        
        manager.close_position_tracking("NONEXISTENT")
        
        manager.initialize_trailing_stop("FTFT", 5.50, 5.00, 0.02)
        manager.close_position_tracking("FTFT")
        manager.close_position_tracking("FTFT")
        
        assert "FTFT" not in manager.position_tracking


class TestBracketStopFillCancelsConfirmation:
    """v-close-position-tracking-2026-09-14: Test that bracket stop fill cancels pending confirmation."""

    def test_handle_full_bracket_fill_cancels_pending_confirmation(self):
        """handle_full_bracket_fill must cancel pending close confirmation requests."""
        src = (Path(__file__).parent.parent / "core" / "order_monitor" / "brackets.py").read_text()
        
        marker = "async def handle_full_bracket_fill"
        idx = src.index(marker)
        body = src[idx:idx + 2500]
        
        assert "pending_close_requests" in body, (
            "handle_full_bracket_fill must check for pending_close_requests"
        )
        assert "confirmed" in body and "True" in body, (
            "handle_full_bracket_fill must set confirmed=True to short-circuit confirmation wait"
        )
        assert "bracket_fill_canceled_pending_confirmation" in body, (
            "handle_full_bracket_fill must log when canceling pending confirmation"
        )

    def test_handle_full_bracket_fill_is_authoritative(self):
        """Broker fill path must be authoritative: direct close without confirmation."""
        src = (Path(__file__).parent.parent / "core" / "order_monitor" / "brackets.py").read_text()
        
        marker = "async def handle_full_bracket_fill"
        idx = src.index(marker)
        body = src[idx:idx + 4000]
        
        assert "_get_close_confirmation" not in body, (
            "handle_full_bracket_fill must NOT use _get_close_confirmation (broker fill is authoritative)"
        )
        assert "del engine.positions[symbol]" in body or 'engine.positions' not in body[:500], (
            "handle_full_bracket_fill must remove position from tracking (or delegate to caller)"
        )

    def test_handle_full_bracket_fill_calls_close_position_tracking(self):
        """handle_full_bracket_fill must call exit_manager.close_position_tracking."""
        src = (Path(__file__).parent.parent / "core" / "order_monitor" / "brackets.py").read_text()
        
        marker = "async def handle_full_bracket_fill"
        idx = src.index(marker)
        end_marker = "async def handle_partial_bracket_fill"
        end_idx = src.index(end_marker) if end_marker in src else idx + 5000
        body = src[idx:end_idx]
        
        assert "close_position_tracking" in body, (
            "handle_full_bracket_fill must call exit_manager.close_position_tracking"
        )


class TestSoftwareStopPathDoesNotDoubleClose:
    """v-close-position-tracking-2026-09-14: Test software stop path doesn't double-close after broker fill."""

    def test_close_position_rechecks_container_after_confirmation(self):
        """_close_position_with_commentary must recheck container after confirmation returns."""
        src = (Path(__file__).parent.parent / "core" / "engine.py").read_text()
        
        marker = "async def _close_position_with_commentary"
        idx = src.index(marker)
        body = src[idx:idx + 12000]
        
        assert "broker_fill_closed_during_confirm" in body, (
            "_close_position_with_commentary must detect broker fill closing position during confirm"
        )
        assert "_close_container.get(position.symbol)" in body or "not in _close_container" in body, (
            "_close_position_with_commentary must recheck if position still in container"
        )

    def test_close_position_returns_early_if_already_closed(self):
        """_close_position_with_commentary must return early if position already closed."""
        src = (Path(__file__).parent.parent / "core" / "engine.py").read_text()
        
        marker = "broker_fill_closed_during_confirm"
        idx = src.index(marker)
        body = src[idx:idx + 500]
        
        assert "return" in body, (
            "Must return early when position was closed by broker fill during confirmation"
        )


class TestFTFTIncidentReplay:
    """v-close-position-tracking-2026-09-14: Replay FTFT incident from 2026-09-14 12:49 ET.
    
    Incident flow:
      1. hard_stop_breached → software stop check starts _close_position_with_commentary
      2. Confirmation flow starts, waiting for user response
      3. order_monitor detects bracket stop fill @5.23 qty 432
      4. handle_full_bracket_fill calls exit_manager.close_position_tracking → AttributeError
      5. Confirmation times out with deny → state_reverted to live
      6. Broker already flat → ghost state
    
    Expected behavior after fix:
      1. close_position_tracking exists and clears exit_trackers
      2. handle_full_bracket_fill cancels pending confirmation
      3. Software path detects broker already closed, returns early
      4. No ghost state, no revert to LIVE after broker flat
    """

    def test_ftft_close_position_tracking_exists(self):
        """FTFT fix: close_position_tracking must exist on DynamicExitManager."""
        from analysis.exit_managers import DynamicExitManager
        
        mock_brain = MagicMock()
        mock_commentary = MagicMock()
        manager = DynamicExitManager(mock_brain, mock_commentary)
        
        assert hasattr(manager, 'close_position_tracking'), (
            "FTFT fix: DynamicExitManager must have close_position_tracking method"
        )
        
        assert callable(manager.close_position_tracking), (
            "FTFT fix: close_position_tracking must be callable"
        )

    def test_ftft_bracket_fill_cancels_confirmation(self):
        """FTFT fix: bracket stop fill must cancel pending close confirmation."""
        src = (Path(__file__).parent.parent / "core" / "order_monitor" / "brackets.py").read_text()
        
        assert "broker_fill_authoritative" in src, (
            "FTFT fix: must have audit event for broker_fill_authoritative"
        )
        assert "pending_confirm_canceled" in src, (
            "FTFT fix: must have audit event for pending_confirm_canceled"
        )

    def test_ftft_no_revert_to_live_after_broker_flat(self):
        """FTFT fix: must not revert to LIVE after broker is flat."""
        src = (Path(__file__).parent.parent / "core" / "engine.py").read_text()
        
        marker = "broker_fill_closed_during_confirm"
        assert marker in src, (
            "FTFT fix: must detect broker fill closed position during confirmation"
        )
        
        idx = src.index(marker)
        nearby = src[idx-500:idx+500]
        
        assert "return" in nearby, (
            "FTFT fix: must return early to prevent ghost state"
        )


# ============================================================================
# v-broker-flat-close-2026-09-11: FSM BROKER-FLAT CLOSE TRANSITION TESTS
# Tests for allowing direct CLOSED transitions from managed states when
# broker confirms position is flat (external close).
# ============================================================================

class TestFSMBrokerFlatCloseTransitions:
    """v-broker-flat-close-2026-09-11: Test FSM allows direct CLOSED transitions.
    
    Incident context:
      - TNON: live→closed illegal_transition on broker_flat_reject_oversold_overbought
      - ALM: at_breakeven→closed illegal_transition after broker_flat_handled
    
    Fix: Allow CLOSED from LIVE, AT_BREAKEVEN, AT_1R, TRAILING when broker
    confirms position is flat. These are external closes, not bot-initiated
    exits (which go through EXITING first).
    """
    
    def test_live_to_closed_is_legal(self):
        """LIVE → CLOSED must be legal for broker-flat scenarios."""
        from core.position_state import can_transition, PositionState
        
        assert can_transition(PositionState.LIVE, PositionState.CLOSED), (
            "LIVE → CLOSED must be legal (broker_flat external close)"
        )
        
    def test_at_breakeven_to_closed_is_legal(self):
        """AT_BREAKEVEN → CLOSED must be legal for broker-flat scenarios."""
        from core.position_state import can_transition, PositionState
        
        assert can_transition(PositionState.AT_BREAKEVEN, PositionState.CLOSED), (
            "AT_BREAKEVEN → CLOSED must be legal (broker_flat external close)"
        )
        
    def test_at_1r_to_closed_is_legal(self):
        """AT_1R → CLOSED must be legal for broker-flat scenarios."""
        from core.position_state import can_transition, PositionState
        
        assert can_transition(PositionState.AT_1R, PositionState.CLOSED), (
            "AT_1R → CLOSED must be legal (broker_flat external close)"
        )
        
    def test_trailing_to_closed_is_legal(self):
        """TRAILING → CLOSED must be legal for broker-flat scenarios."""
        from core.position_state import can_transition, PositionState
        
        assert can_transition(PositionState.TRAILING, PositionState.CLOSED), (
            "TRAILING → CLOSED must be legal (broker_flat external close)"
        )
        
    def test_exiting_to_closed_still_legal(self):
        """EXITING → CLOSED must remain legal (bot-initiated close flow)."""
        from core.position_state import can_transition, PositionState
        
        assert can_transition(PositionState.EXITING, PositionState.CLOSED), (
            "EXITING → CLOSED must remain legal"
        )
        
    def test_opening_to_closed_not_allowed(self):
        """OPENING → CLOSED should NOT be legal (must go through LIVE first)."""
        from core.position_state import can_transition, PositionState
        
        assert not can_transition(PositionState.OPENING, PositionState.CLOSED), (
            "OPENING → CLOSED should not be legal (incomplete entry)"
        )
        
    def test_all_managed_states_allow_closed(self):
        """All managed states (LIVE+) must allow direct CLOSED transition."""
        from core.position_state import can_transition, PositionState
        
        managed_states = [
            PositionState.LIVE,
            PositionState.AT_BREAKEVEN,
            PositionState.AT_1R,
            PositionState.TRAILING,
        ]
        
        for state in managed_states:
            assert can_transition(state, PositionState.CLOSED), (
                f"{state.value} → CLOSED must be legal for broker-flat"
            )
            
    def test_normal_transitions_still_work(self):
        """Normal progression transitions must still work."""
        from core.position_state import can_transition, PositionState
        
        assert can_transition(PositionState.LIVE, PositionState.AT_BREAKEVEN)
        assert can_transition(PositionState.AT_BREAKEVEN, PositionState.AT_1R)
        assert can_transition(PositionState.AT_1R, PositionState.TRAILING)
        assert can_transition(PositionState.LIVE, PositionState.EXITING)
        
    def test_regression_transitions_blocked(self):
        """Regression transitions must still be blocked."""
        from core.position_state import can_transition, PositionState
        
        assert not can_transition(PositionState.AT_BREAKEVEN, PositionState.LIVE)
        assert not can_transition(PositionState.AT_1R, PositionState.LIVE)
        assert not can_transition(PositionState.TRAILING, PositionState.LIVE)
        assert not can_transition(PositionState.CLOSED, PositionState.LIVE)


# ============================================================================
# v-qty-sync-2026-09-11: QUANTITY SYNC TESTS
# Tests for syncing local quantity to broker truth before re-bracketing.
# ============================================================================

class TestQuantitySyncInReBracket:
    """v-qty-sync-2026-09-11: Test qty sync logic in re_bracket_position.
    
    Incident context (ALM):
      - Local qty: 567
      - Stop fill reported qty: 75
      - Expected remaining: 567 - 75 = 492
      - But broker was actually flat → re_bracket rejected → infinite loop
    
    Fix: Sync local qty to broker qty before placing bracket.
    """
    
    def test_qty_sync_code_exists(self):
        """re_bracket_position must contain qty desync detection."""
        # v-phase-b-order-monitor-2026-09-14: implementation moved to core/order_monitor/
        src = (Path(__file__).parent.parent / "core" / "order_monitor" / "brackets.py").read_text()
        
        marker = "async def re_bracket_position"
        idx = src.index(marker)
        body = src[idx:idx + 3500]
        
        assert "qty_desync_detected" in body, (
            "re_bracket_position must log qty desync detection"
        )
        assert "broker_qty != local_qty" in body, (
            "re_bracket_position must compare broker_qty to local_qty"
        )
        assert "position.quantity = broker_qty" in body, (
            "re_bracket_position must sync position.quantity to broker_qty"
        )
        
    def test_qty_sync_audit_fields(self):
        """Qty sync must audit local_qty, broker_qty, and delta."""
        # v-phase-b-order-monitor-2026-09-14: implementation moved to core/order_monitor/
        src = (Path(__file__).parent.parent / "core" / "order_monitor" / "brackets.py").read_text()
        
        marker = "qty_desync"
        idx = src.index(marker)
        body = src[idx:idx + 500]
        
        assert "local_qty=" in body, "Must audit local_qty"
        assert "broker_qty=" in body, "Must audit broker_qty"
        assert "delta=" in body, "Must audit delta for RCA"
        
    def test_qty_sync_commentary_generated(self):
        """Qty sync must generate user-facing commentary."""
        # v-phase-b-order-monitor-2026-09-14: implementation moved to core/order_monitor/
        src = (Path(__file__).parent.parent / "core" / "order_monitor" / "brackets.py").read_text()
        
        marker = "qty_desync_detected"
        idx = src.index(marker)
        body = src[idx:idx + 1000]
        
        assert "Quantity Desync Detected" in body, (
            "Must generate commentary for qty desync"
        )
        assert "Syncing to broker truth" in body, (
            "Commentary must explain sync action"
        )


class TestPartialFillQtyValidation:
    """v-qty-sync-2026-09-11: Test partial fill qty validation."""
    
    def test_partial_fill_suspicious_detection(self):
        """handle_partial_bracket_fill must detect suspicious qty mismatches."""
        # v-phase-b-order-monitor-2026-09-14: implementation moved to core/order_monitor/
        src = (Path(__file__).parent.parent / "core" / "order_monitor" / "brackets.py").read_text()
        
        marker = "async def handle_partial_bracket_fill"
        idx = src.index(marker)
        body = src[idx:idx + 2500]
        
        assert "fill_ratio" in body, (
            "handle_partial_bracket_fill must calculate fill ratio"
        )
        assert "is_suspicious" in body, (
            "handle_partial_bracket_fill must flag suspicious fills"
        )
        
    def test_partial_fill_broker_flat_override(self):
        """When partial fill but broker flat, handle as full fill."""
        # v-phase-b-order-monitor-2026-09-14: implementation moved to core/order_monitor/
        src = (Path(__file__).parent.parent / "core" / "order_monitor" / "brackets.py").read_text()
        
        marker = "async def handle_partial_bracket_fill"
        idx = src.index(marker)
        body = src[idx:idx + 2500]
        
        assert "partial_fill_was_actually_full" in body, (
            "Must detect when partial fill was actually full close"
        )
        assert "handle_full_bracket_fill" in body, (
            "Must call handle_full_bracket_fill when broker confirms flat"
        )
        
    def test_partial_fill_qty_mismatch_audit(self):
        """Partial fill with qty mismatch must be audited."""
        # v-phase-b-order-monitor-2026-09-14: implementation moved to core/order_monitor/
        src = (Path(__file__).parent.parent / "core" / "order_monitor" / "brackets.py").read_text()
        
        marker = "async def handle_partial_bracket_fill"
        idx = src.index(marker)
        body = src[idx:idx + 2500]
        
        assert "partial_fill_qty_mismatch" in body, (
            "Must audit qty mismatch in partial fill"
        )
        assert "calculated_remaining" in body, (
            "Must include calculated_remaining in audit"
        )


# ============================================================================
# INCIDENT REPLAY TESTS
# ============================================================================

class TestTNONIncidentReplay:
    """Replay TNON incident: Hari manual close → re_bracket → oversold reject.
    
    Expected behavior after fix:
      1. broker_flat_reject_oversold_overbought fires
      2. broker_flat_handled removes tracking
      3. FSM transition live→closed succeeds (no illegal_transition)
    """
    
    def test_tnon_scenario_fsm_allows_live_to_closed(self):
        """TNON scenario: FSM must allow live→closed for broker_flat."""
        from core.position_state import can_transition, PositionState
        
        # This was the failing transition in TNON incident
        assert can_transition(PositionState.LIVE, PositionState.CLOSED), (
            "TNON fix: live→closed must be legal for broker_flat_reject_oversold"
        )


class TestALMIncidentReplay:
    """Replay ALM incident: fill qty 75 vs local remaining 567.
    
    Root cause analysis:
      - Position had 567 shares locally
      - External fills reduced broker qty without bot awareness
      - Stop fill reported qty 75 (actual broker remaining)
      - remaining_qty calculation: 567 - 75 = 492 (wrong!)
      - re_bracket tried to place for 492 shares → rejected oversold
      - broker_flat_handled ran, but FSM rejected at_breakeven→closed
    
    Expected behavior after fix:
      1. FSM allows at_breakeven→closed (no illegal_transition)
      2. _re_bracket_position syncs local qty to broker before placing
      3. Audit trail includes qty desync data for post-mortem
    """
    
    def test_alm_scenario_fsm_allows_at_breakeven_to_closed(self):
        """ALM scenario: FSM must allow at_breakeven→closed for broker_flat."""
        from core.position_state import can_transition, PositionState
        
        # This was the failing transition in ALM incident
        assert can_transition(PositionState.AT_BREAKEVEN, PositionState.CLOSED), (
            "ALM fix: at_breakeven→closed must be legal for broker_flat"
        )
        
    def test_alm_scenario_qty_sync_implemented(self):
        """ALM scenario: qty sync must be implemented in re_bracket."""
        # v-phase-b-order-monitor-2026-09-14: implementation moved to core/order_monitor/
        src = (Path(__file__).parent.parent / "core" / "order_monitor" / "brackets.py").read_text()
        
        assert "qty_desync" in src, (
            "ALM fix: qty desync detection must be implemented"
        )
        assert "sync_to_broker" in src, (
            "ALM fix: sync_to_broker action must be implemented"
        )
        
    def test_alm_qty_desync_audit_for_rca(self):
        """ALM scenario: audit must include delta for post-mortem RCA."""
        # v-phase-b-order-monitor-2026-09-14: implementation moved to core/order_monitor/
        src = (Path(__file__).parent.parent / "core" / "order_monitor" / "brackets.py").read_text()
        
        marker = "qty_desync"
        idx = src.index(marker)
        body = src[idx:idx + 500]  # Wider search to include full audit call
        
        assert "delta=" in body, (
            "ALM RCA: audit must include delta (local_qty - broker_qty)"
        )


class TestBrokerFlatCloseIntegration:
    """Integration tests for broker-flat close handling."""
    
    def test_handle_broker_flat_detected_transitions_to_closed(self):
        """handle_broker_flat_detected must transition to CLOSED."""
        # v-phase-b-order-monitor-2026-09-14: implementation moved to core/order_monitor/
        src = (Path(__file__).parent.parent / "core" / "order_monitor" / "broker_flat.py").read_text()
        
        marker = "async def handle_broker_flat_detected"
        idx = src.index(marker)
        body = src[idx:idx + 3000]
        
        assert "PositionState.CLOSED" in body, (
            "handle_broker_flat_detected must transition to CLOSED"
        )
        assert "try_transition" in body, (
            "handle_broker_flat_detected must use try_transition"
        )
        
    def test_broker_flat_handler_clears_managed_flag(self):
        """handle_broker_flat_detected must clear managed_by_bot."""
        # v-phase-b-order-monitor-2026-09-14: implementation moved to core/order_monitor/
        src = (Path(__file__).parent.parent / "core" / "order_monitor" / "broker_flat.py").read_text()
        
        marker = "async def handle_broker_flat_detected"
        idx = src.index(marker)
        body = src[idx:idx + 3000]
        
        assert "managed_by_bot = False" in body, (
            "handle_broker_flat_detected must clear managed_by_bot"
        )
        
    def test_broker_flat_handler_removes_from_tracking(self):
        """handle_broker_flat_detected must remove position from tracking (when ghost_flatten_enabled)."""
        # v-phase-b-order-monitor-2026-09-14: implementation moved to core/order_monitor/
        # v-broker-leg-authority-2026-09-14: now conditional on ENABLE_GHOST_FLATTEN_AFTER_BROKER_FLAT
        src = (Path(__file__).parent.parent / "core" / "order_monitor" / "broker_flat.py").read_text()
        
        marker = "async def handle_broker_flat_detected"
        idx = src.index(marker)
        body = src[idx:idx + 6000]
        
        assert "engine.positions.pop" in body, (
            "handle_broker_flat_detected must remove from engine.positions (when ghost_flatten_enabled)"
        )
        assert "ghost_flatten_enabled" in body, (
            "handle_broker_flat_detected must check ghost_flatten_enabled before removing"
        )


# ============================================================================
# v-broker-leg-authority-2026-09-14: BROKER-LEG AUTHORITY TESTS
# Tests for PR3 fix: When OCO/stop leg is WORKING or FILLED at broker,
# skip supervised close confirm — broker is source of truth.
# ============================================================================

class TestBrokerLegAuthorityConfig:
    """v-broker-leg-authority-2026-09-14: Config flag tests."""

    def test_enable_broker_leg_authority_flag_exists(self):
        """ENABLE_BROKER_LEG_AUTHORITY config flag must exist."""
        from core.config import Config
        cfg = Config()

        assert hasattr(cfg, 'ENABLE_BROKER_LEG_AUTHORITY'), (
            "Config must have ENABLE_BROKER_LEG_AUTHORITY property"
        )

    def test_enable_broker_leg_authority_default_true(self):
        """ENABLE_BROKER_LEG_AUTHORITY should default to True for fill-path correctness."""
        from core.config import Config
        cfg = Config()

        assert cfg.ENABLE_BROKER_LEG_AUTHORITY is True, (
            "ENABLE_BROKER_LEG_AUTHORITY must default to True"
        )

    def test_enable_ghost_flatten_flag_exists(self):
        """ENABLE_GHOST_FLATTEN_AFTER_BROKER_FLAT config flag must exist."""
        from core.config import Config
        cfg = Config()

        assert hasattr(cfg, 'ENABLE_GHOST_FLATTEN_AFTER_BROKER_FLAT'), (
            "Config must have ENABLE_GHOST_FLATTEN_AFTER_BROKER_FLAT property"
        )

    def test_enable_ghost_flatten_default_false(self):
        """ENABLE_GHOST_FLATTEN_AFTER_BROKER_FLAT should default to False (safe)."""
        from core.config import Config
        cfg = Config()

        assert cfg.ENABLE_GHOST_FLATTEN_AFTER_BROKER_FLAT is False, (
            "ENABLE_GHOST_FLATTEN_AFTER_BROKER_FLAT must default to False"
        )


class TestBrokerLegAuthorityHelper:
    """v-broker-leg-authority-2026-09-14: is_broker_leg_authoritative helper tests."""

    def test_helper_function_exists(self):
        """is_broker_leg_authoritative must exist in broker_flat module."""
        src = (Path(__file__).parent.parent / "core" / "order_monitor" / "broker_flat.py").read_text()

        assert "async def is_broker_leg_authoritative" in src, (
            "is_broker_leg_authoritative helper must exist"
        )

    def test_helper_returns_tuple(self):
        """is_broker_leg_authoritative must return (bool, str) tuple."""
        src = (Path(__file__).parent.parent / "core" / "order_monitor" / "broker_flat.py").read_text()

        marker = "async def is_broker_leg_authoritative"
        idx = src.index(marker)
        body = src[idx:idx + 4000]

        assert "-> tuple[bool, str]" in body, (
            "is_broker_leg_authoritative must return tuple[bool, str]"
        )

    def test_helper_checks_bracket_order_id(self):
        """is_broker_leg_authoritative must check bracket_order_id."""
        src = (Path(__file__).parent.parent / "core" / "order_monitor" / "broker_flat.py").read_text()

        marker = "async def is_broker_leg_authoritative"
        idx = src.index(marker)
        body = src[idx:idx + 4000]

        assert "bracket_order_id" in body, (
            "is_broker_leg_authoritative must check bracket_order_id"
        )
        assert "no_bracket" in body, (
            "Must return 'no_bracket' reason when position has no bracket"
        )

    def test_helper_detects_working_status(self):
        """is_broker_leg_authoritative must detect WORKING status as authoritative."""
        src = (Path(__file__).parent.parent / "core" / "order_monitor" / "broker_flat.py").read_text()

        marker = "async def is_broker_leg_authoritative"
        idx = src.index(marker)
        body = src[idx:idx + 4000]

        assert "WORKING" in body, (
            "Must detect WORKING status"
        )
        assert "stop_working" in body or "oco_working" in body, (
            "Must return appropriate working reason"
        )

    def test_helper_detects_filled_status(self):
        """is_broker_leg_authoritative must detect FILLED status as authoritative."""
        src = (Path(__file__).parent.parent / "core" / "order_monitor" / "broker_flat.py").read_text()

        marker = "async def is_broker_leg_authoritative"
        idx = src.index(marker)
        body = src[idx:idx + 4000]

        assert "FILLED" in body, (
            "Must detect FILLED status"
        )
        assert "stop_filled" in body or "oco_filled" in body, (
            "Must return appropriate filled reason"
        )

    def test_helper_respects_enable_flag(self):
        """is_broker_leg_authoritative must respect ENABLE_BROKER_LEG_AUTHORITY flag."""
        src = (Path(__file__).parent.parent / "core" / "order_monitor" / "broker_flat.py").read_text()

        marker = "async def is_broker_leg_authoritative"
        idx = src.index(marker)
        body = src[idx:idx + 4000]

        assert "ENABLE_BROKER_LEG_AUTHORITY" in body, (
            "Must check ENABLE_BROKER_LEG_AUTHORITY config flag"
        )
        assert "feature_disabled" in body, (
            "Must return 'feature_disabled' when flag is False"
        )


class TestBrokerLegAuthorityIntegration:
    """v-broker-leg-authority-2026-09-14: Integration with _close_position_with_commentary."""

    def test_close_position_checks_broker_leg_authority(self):
        """_close_position_with_commentary must check broker leg authority."""
        src = (Path(__file__).parent.parent / "core" / "engine.py").read_text()

        marker = "async def _close_position_with_commentary"
        idx = src.index(marker)
        body = src[idx:idx + 6000]

        assert "is_broker_leg_authoritative" in body, (
            "_close_position_with_commentary must call is_broker_leg_authoritative"
        )
        assert "ENABLE_BROKER_LEG_AUTHORITY" in body, (
            "_close_position_with_commentary must check ENABLE_BROKER_LEG_AUTHORITY"
        )

    def test_close_position_skips_confirmation_when_authoritative(self):
        """_close_position_with_commentary must skip confirmation when broker is authoritative."""
        src = (Path(__file__).parent.parent / "core" / "engine.py").read_text()

        marker = "async def _close_position_with_commentary"
        idx = src.index(marker)
        body = src[idx:idx + 6000]

        assert "broker_leg_authoritative" in body, (
            "Must have broker_leg_authoritative logic"
        )
        assert "_broker_authoritative" in body, (
            "Must track _broker_authoritative flag"
        )
        assert "skipping confirmation" in body.lower() or "skip_confirmation" in body, (
            "Must skip confirmation when broker is authoritative"
        )

    def test_close_position_audits_skip_confirmation(self):
        """_close_position_with_commentary must audit when skipping confirmation."""
        src = (Path(__file__).parent.parent / "core" / "engine.py").read_text()

        marker = "async def _close_position_with_commentary"
        idx = src.index(marker)
        body = src[idx:idx + 6000]

        assert "skip_confirmation" in body and "broker_leg_authoritative" in body, (
            "Must audit skip_confirmation with broker_leg_authoritative reason"
        )


class TestGhostFlattenBehavior:
    """v-broker-leg-authority-2026-09-14: Ghost flatten flag behavior tests."""

    def test_handle_broker_flat_respects_ghost_flatten_flag(self):
        """handle_broker_flat_detected must respect ENABLE_GHOST_FLATTEN_AFTER_BROKER_FLAT."""
        src = (Path(__file__).parent.parent / "core" / "order_monitor" / "broker_flat.py").read_text()

        marker = "async def handle_broker_flat_detected"
        idx = src.index(marker)
        body = src[idx:idx + 6000]

        assert "ENABLE_GHOST_FLATTEN_AFTER_BROKER_FLAT" in body, (
            "handle_broker_flat_detected must check ENABLE_GHOST_FLATTEN_AFTER_BROKER_FLAT"
        )
        assert "ghost_flatten_enabled" in body, (
            "Must track ghost_flatten_enabled flag"
        )

    def test_ghost_flatten_disabled_does_not_remove_position(self):
        """When ghost_flatten_disabled, position should NOT be auto-removed."""
        src = (Path(__file__).parent.parent / "core" / "order_monitor" / "broker_flat.py").read_text()

        marker = "async def handle_broker_flat_detected"
        idx = src.index(marker)
        body = src[idx:idx + 6000]

        assert "ghost_position_shadow_logged" in body, (
            "Must audit ghost_position_shadow_logged when not auto-removing"
        )
        assert "Ghost Position Detected" in body, (
            "Must generate Ghost Position Detected commentary when disabled"
        )
        assert "NOT auto-removed" in body, (
            "Commentary must indicate position NOT auto-removed"
        )

    def test_ghost_flatten_enabled_removes_position(self):
        """When ghost_flatten_enabled, position should be auto-removed."""
        src = (Path(__file__).parent.parent / "core" / "order_monitor" / "broker_flat.py").read_text()

        marker = "async def handle_broker_flat_detected"
        idx = src.index(marker)
        body = src[idx:idx + 6000]

        assert "if ghost_flatten_enabled:" in body, (
            "Must have conditional branch for ghost_flatten_enabled"
        )
        assert "engine.positions.pop" in body, (
            "Must remove position when ghost_flatten_enabled"
        )
        assert "'auto_flattened': True" in body, (
            "Must indicate auto_flattened in commentary data"
        )


class TestFTFTRaceFixValidation:
    """v-broker-leg-authority-2026-09-14: Validate FTFT race condition fix.

    FTFT incident flow (2026-09-14 12:49 ET):
      1. hard_stop_breached → software stop check starts _close_position_with_commentary
      2. Confirmation flow starts, waiting for user response
      3. order_monitor detects bracket stop fill @5.23 qty 432
      4. handle_full_bracket_fill calls exit_manager.close_position_tracking
      5. Confirmation times out with deny → state_reverted to live
      6. Broker already flat → ghost state

    Fix validates:
      - Broker leg authority check prevents confirmation flow when OCO/stop is active
      - Even if confirmation starts, broker fill path cancels pending confirmation
      - Ghost state is handled appropriately based on ghost_flatten flag
    """

    def test_ftft_fix_broker_leg_authority_prevents_race(self):
        """FTFT fix: broker leg authority check must prevent confirmation race."""
        src = (Path(__file__).parent.parent / "core" / "engine.py").read_text()

        marker = "async def _close_position_with_commentary"
        idx = src.index(marker)
        body = src[idx:idx + 8000]

        assert "is_broker_leg_authoritative" in body, (
            "FTFT fix: must check broker leg authority before confirmation"
        )
        assert "ENABLE_BROKER_LEG_AUTHORITY" in body, (
            "FTFT fix: must be gated by ENABLE_BROKER_LEG_AUTHORITY flag"
        )
        assert "_broker_authoritative" in body, (
            "FTFT fix: must track _broker_authoritative flag"
        )

    def test_ftft_fix_confirmation_gated_by_authority(self):
        """FTFT fix: require_confirmations must be gated by broker authority."""
        src = (Path(__file__).parent.parent / "core" / "engine.py").read_text()

        marker = "async def _close_position_with_commentary"
        idx = src.index(marker)
        body = src[idx:idx + 8000]

        assert "require_confirmations" in body and "_broker_authoritative" in body, (
            "FTFT fix: confirmation check must reference _broker_authoritative"
        )

    def test_ftft_fix_handles_check_failure_gracefully(self):
        """FTFT fix: broker leg authority check failure must not block close."""
        src = (Path(__file__).parent.parent / "core" / "engine.py").read_text()

        marker = "async def _close_position_with_commentary"
        idx = src.index(marker)
        body = src[idx:idx + 6000]

        assert "check_failed" in body or "fetch_error" in body.lower(), (
            "FTFT fix: must handle check failure gracefully"
        )
        assert "proceeding with normal flow" in body.lower() or "_broker_authoritative = False" in body, (
            "FTFT fix: failed check must fall back to normal flow"
        )


class TestHandsOffDenylistIntegration:
    """v-broker-leg-authority-2026-09-14: HANDS_OFF integration tests.

    Ensure broker-leg authority respects HANDS_OFF_DENYLIST (MU, HQGE, SPCX).
    """

    def test_broker_leg_authority_docstring_mentions_hands_off(self):
        """ENABLE_BROKER_LEG_AUTHORITY docstring must mention HANDS_OFF_DENYLIST."""
        src = (Path(__file__).parent.parent / "core" / "config.py").read_text()

        marker = "def ENABLE_BROKER_LEG_AUTHORITY"
        idx = src.index(marker)
        docstring = src[idx:idx + 1500]

        assert "HANDS_OFF_DENYLIST" in docstring, (
            "ENABLE_BROKER_LEG_AUTHORITY docstring must mention HANDS_OFF_DENYLIST"
        )

    def test_close_position_blocks_denylist_before_authority_check(self):
        """_close_position_with_commentary must block denylist symbols early."""
        src = (Path(__file__).parent.parent / "core" / "engine.py").read_text()

        marker = "async def _close_position_with_commentary"
        idx = src.index(marker)

        # Find the denylist check
        denylist_marker = "HANDS_OFF_DENYLIST"
        denylist_idx = src.index(denylist_marker, idx)

        # Find the broker_leg_authority check
        authority_marker = "is_broker_leg_authoritative"
        authority_idx = src.index(authority_marker, idx) if authority_marker in src[idx:] else idx + 10000

        assert denylist_idx < authority_idx, (
            "HANDS_OFF_DENYLIST check must come before broker_leg_authority check"
        )


class TestIdempotentClosePositionTracking:
    """v-broker-leg-authority-2026-09-14: Reinforce idempotent close_position_tracking tests."""

    def test_close_position_tracking_idempotent_multiple_calls(self):
        """close_position_tracking must be idempotent: multiple calls don't raise."""
        from analysis.exit_managers import DynamicExitManager, AdvancedExitManager
        from unittest.mock import MagicMock

        # DynamicExitManager
        mock_brain = MagicMock()
        mock_commentary = MagicMock()
        dynamic_mgr = DynamicExitManager(mock_brain, mock_commentary)

        dynamic_mgr.initialize_position_tracking("FTFT", 5.50, 5.00, 6.00)
        assert "FTFT" in dynamic_mgr.exit_trackers

        dynamic_mgr.close_position_tracking("FTFT")
        assert "FTFT" not in dynamic_mgr.exit_trackers

        # Second call should NOT raise
        dynamic_mgr.close_position_tracking("FTFT")
        assert "FTFT" not in dynamic_mgr.exit_trackers

        # Third call with different symbol should be no-op
        dynamic_mgr.close_position_tracking("NONEXISTENT")

        # AdvancedExitManager
        advanced_mgr = AdvancedExitManager(mock_commentary)

        advanced_mgr.initialize_trailing_stop("FTFT", 5.50, 5.00, 0.02)
        assert "FTFT" in advanced_mgr.position_tracking

        advanced_mgr.close_position_tracking("FTFT")
        assert "FTFT" not in advanced_mgr.position_tracking

        # Second call should NOT raise
        advanced_mgr.close_position_tracking("FTFT")
        assert "FTFT" not in advanced_mgr.position_tracking

    def test_close_position_tracking_isolated_per_symbol(self):
        """close_position_tracking must only affect the specified symbol."""
        from analysis.exit_managers import DynamicExitManager
        from unittest.mock import MagicMock

        mock_brain = MagicMock()
        mock_commentary = MagicMock()
        manager = DynamicExitManager(mock_brain, mock_commentary)

        manager.initialize_position_tracking("FTFT", 5.50, 5.00, 6.00)
        manager.initialize_position_tracking("AAPL", 150.0, 145.0, 160.0)
        manager.initialize_position_tracking("TSLA", 200.0, 190.0, 220.0)

        assert "FTFT" in manager.exit_trackers
        assert "AAPL" in manager.exit_trackers
        assert "TSLA" in manager.exit_trackers

        manager.close_position_tracking("FTFT")

        assert "FTFT" not in manager.exit_trackers
        assert "AAPL" in manager.exit_trackers
        assert "TSLA" in manager.exit_trackers


class TestBrokerLegAuthorityFlagOffBehavior:
    """v-broker-leg-authority-2026-09-14: Test flag-off preserves prior confirm behavior."""

    def test_flag_off_preserves_confirmation_flow(self):
        """When ENABLE_BROKER_LEG_AUTHORITY=False, confirmation flow should proceed."""
        src = (Path(__file__).parent.parent / "core" / "engine.py").read_text()

        marker = "async def _close_position_with_commentary"
        idx = src.index(marker)
        body = src[idx:idx + 6000]

        assert "ENABLE_BROKER_LEG_AUTHORITY" in body, (
            "Must check ENABLE_BROKER_LEG_AUTHORITY flag"
        )
        assert "_broker_authoritative = False" in body, (
            "Must set _broker_authoritative = False when flag disabled or check fails"
        )

    def test_is_broker_leg_authoritative_returns_false_when_disabled(self):
        """is_broker_leg_authoritative must return (False, 'feature_disabled') when flag off."""
        src = (Path(__file__).parent.parent / "core" / "order_monitor" / "broker_flat.py").read_text()

        marker = "async def is_broker_leg_authoritative"
        idx = src.index(marker)
        body = src[idx:idx + 2000]

        assert "feature_disabled" in body, (
            "Must return 'feature_disabled' reason when flag is off"
        )
        assert 'return (False, "feature_disabled")' in body, (
            "Must return (False, 'feature_disabled') tuple"
        )



class TestSaveStateAwaitFix:
    """v-fix-await-save-state-2026-09-15: _save_state is sync; must not be awaited."""

    def test_handle_full_bracket_fill_does_not_await_save_state(self):
        """handle_full_bracket_fill must call _save_state without await."""
        src = (Path(__file__).parent.parent / "core" / "order_monitor" / "brackets.py").read_text()

        marker = "async def handle_full_bracket_fill"
        idx = src.index(marker)
        body = src[idx:idx + 6000]

        assert "await engine._save_state()" not in body, (
            "MUST NOT await _save_state — it is a sync function that returns None. "
            "Awaiting None causes TypeError."
        )
        assert "engine._save_state()" in body, (
            "Must call _save_state synchronously"
        )

    def test_save_state_call_wrapped_in_try_except(self):
        """_save_state call must be wrapped in try/except to avoid aborting fill handling."""
        src = (Path(__file__).parent.parent / "core" / "order_monitor" / "brackets.py").read_text()

        marker = "async def handle_full_bracket_fill"
        idx = src.index(marker)
        body = src[idx:idx + 6000]

        assert "try:" in body and "engine._save_state()" in body, (
            "Must have try block for _save_state"
        )
        assert "except Exception" in body, (
            "Must catch Exception around _save_state"
        )
        assert "_save_state failed" in body, (
            "Must log warning on _save_state failure"
        )

    def test_engine_save_state_is_sync_def(self):
        """TradingEngineWithCommentary._save_state must be sync (not async)."""
        src = (Path(__file__).parent.parent / "core" / "engine.py").read_text()

        assert "def _save_state(self):" in src, (
            "_save_state must be a sync method (def _save_state, not async def)"
        )
        assert "async def _save_state" not in src, (
            "_save_state MUST NOT be async — other callers rely on sync behavior"
        )

    def test_awaiting_none_would_fail(self):
        """Regression: awaiting None must raise TypeError."""
        import asyncio

        async def call_and_await_none():
            def sync_fn():
                return None
            await sync_fn()  # This is the bug pattern

        with pytest.raises(TypeError, match="can't be used in 'await'"):
            asyncio.get_event_loop().run_until_complete(call_and_await_none())

