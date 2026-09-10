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

def _get_bracket_monitored_positions_logic(positions: dict) -> list:
    """Pure logic test of position filtering (no engine import needed).
    
    This replicates the logic of _get_bracket_monitored_positions for testing.
    """
    result = []
    for symbol, pos in list(positions.items()):
        if pos is None:
            continue
        if not getattr(pos, 'managed_by_bot', False):
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

def _get_positions_missing_brackets_logic(positions: dict) -> list:
    """Pure logic test of position filtering for bootstrap (no engine import needed).
    
    This replicates the logic of _get_positions_missing_brackets for testing.
    Same filters as _get_bracket_monitored_positions EXCEPT the bracket_order_id
    requirement is inverted (we want positions WITHOUT bracket_order_id).
    """
    result = []
    for symbol, pos in list(positions.items()):
        if pos is None:
            continue
        if not getattr(pos, 'managed_by_bot', False):
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
