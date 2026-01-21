"""
Tests for Core Modules
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.config import (
    Config, BrokerConfig, RiskConfig, TradingConfig,
    StrategyConfig, TradingMode, PositionSizingMethod, get_config
)
from core.state import (
    Order, Position, PortfolioState, TradingState,
    OrderSide, OrderType, OrderStatus, PositionSide
)
from core.events import (
    EventBus, Event, EventType, get_event_bus, emit, emit_sync
)


class TestConfig:
    """Tests for Configuration System"""

    def test_default_config(self):
        """Test default configuration values"""
        config = Config()

        assert config.trading.mode == TradingMode.SIMULATION
        assert config.risk.max_position_size == 0.10
        assert config.risk.max_positions == 5
        assert len(config.trading.watchlist) > 0

    def test_broker_config(self):
        """Test broker configuration"""
        broker = BrokerConfig(
            api_key='test_key',
            api_secret='test_secret',
            account_id='12345'
        )

        assert broker.api_key == 'test_key'
        assert broker.api_secret == 'test_secret'

    def test_broker_config_from_env(self):
        """Test broker config from environment"""
        with patch.dict('os.environ', {
            'SCHWAB_API_KEY': 'env_key',
            'SCHWAB_API_SECRET': 'env_secret'
        }):
            broker = BrokerConfig.from_env()
            assert broker.api_key == 'env_key'
            assert broker.api_secret == 'env_secret'

    def test_risk_config(self):
        """Test risk configuration"""
        risk = RiskConfig(
            max_position_size=0.05,
            max_daily_loss=0.02,
            max_positions=3
        )

        assert risk.max_position_size == 0.05
        assert risk.max_daily_loss == 0.02
        assert risk.max_positions == 3
        assert risk.sizing_method == PositionSizingMethod.FIXED_PERCENTAGE

    def test_trading_config(self):
        """Test trading configuration"""
        trading = TradingConfig(
            mode=TradingMode.PAPER,
            watchlist=['AAPL', 'GOOGL', 'NVDA'],
            allow_premarket=True
        )

        assert trading.mode == TradingMode.PAPER
        assert 'AAPL' in trading.watchlist
        assert trading.allow_premarket is True

    def test_config_to_dict(self):
        """Test config serialization"""
        config = Config()
        d = config.to_dict()

        assert 'broker' in d
        assert 'risk' in d
        assert 'trading' in d
        assert 'strategy' in d
        assert d['trading']['mode'] == 'simulation'

    def test_config_validation_success(self):
        """Test successful config validation"""
        config = Config()
        errors = config.validate()

        assert len(errors) == 0

    def test_config_validation_failure(self):
        """Test config validation with errors"""
        config = Config()
        config.risk.max_position_size = 1.5  # Invalid: > 1
        config.trading.watchlist = []  # Invalid: empty

        errors = config.validate()

        assert len(errors) > 0
        assert any('max_position_size' in e for e in errors)
        assert any('watchlist' in e for e in errors)

    def test_config_live_mode_validation(self):
        """Test live mode requires credentials"""
        config = Config()
        config.trading.mode = TradingMode.LIVE
        config.broker.api_key = ''
        config.broker.api_secret = ''

        errors = config.validate()

        assert any('api_key' in e for e in errors)
        assert any('api_secret' in e for e in errors)


class TestState:
    """Tests for State Objects"""

    def test_order_creation(self):
        """Test order creation"""
        order = Order(
            id='order123',
            symbol='AAPL',
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100
        )

        assert order.id == 'order123'
        assert order.symbol == 'AAPL'
        assert order.side == OrderSide.BUY
        assert order.quantity == 100
        assert order.status == OrderStatus.PENDING

    def test_order_is_active(self):
        """Test order active status"""
        active_order = Order(
            id='1', symbol='AAPL', side=OrderSide.BUY,
            order_type=OrderType.MARKET, quantity=100,
            status=OrderStatus.SUBMITTED
        )
        assert active_order.is_active is True

        filled_order = Order(
            id='2', symbol='AAPL', side=OrderSide.BUY,
            order_type=OrderType.MARKET, quantity=100,
            status=OrderStatus.FILLED
        )
        assert filled_order.is_active is False

    def test_order_is_complete(self):
        """Test order complete status"""
        pending = Order(
            id='1', symbol='AAPL', side=OrderSide.BUY,
            order_type=OrderType.MARKET, quantity=100,
            status=OrderStatus.PENDING
        )
        assert pending.is_complete is False

        filled = Order(
            id='2', symbol='AAPL', side=OrderSide.BUY,
            order_type=OrderType.MARKET, quantity=100,
            status=OrderStatus.FILLED
        )
        assert filled.is_complete is True

    def test_order_to_dict(self):
        """Test order serialization"""
        order = Order(
            id='order123',
            symbol='AAPL',
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100,
            price=150.0
        )

        d = order.to_dict()
        assert d['id'] == 'order123'
        assert d['side'] == 'buy'
        assert d['order_type'] == 'limit'
        assert d['price'] == 150.0

    def test_position_creation(self):
        """Test position creation"""
        position = Position(
            symbol='AAPL',
            side=PositionSide.LONG,
            quantity=100,
            entry_price=150.0,
            current_price=155.0
        )

        assert position.symbol == 'AAPL'
        assert position.side == PositionSide.LONG
        assert position.quantity == 100

    def test_position_unrealized_pnl_long(self):
        """Test long position P&L calculation"""
        position = Position(
            symbol='AAPL',
            side=PositionSide.LONG,
            quantity=100,
            entry_price=150.0,
            current_price=160.0
        )

        assert position.unrealized_pnl == 1000.0  # (160-150) * 100
        assert position.unrealized_pnl_pct == pytest.approx(6.67, rel=0.01)

    def test_position_unrealized_pnl_short(self):
        """Test short position P&L calculation"""
        position = Position(
            symbol='AAPL',
            side=PositionSide.SHORT,
            quantity=100,
            entry_price=150.0,
            current_price=140.0
        )

        assert position.unrealized_pnl == 1000.0  # (150-140) * 100

    def test_position_market_value(self):
        """Test market value calculation"""
        position = Position(
            symbol='AAPL',
            side=PositionSide.LONG,
            quantity=100,
            entry_price=150.0,
            current_price=155.0
        )

        assert position.market_value == 15500.0
        assert position.cost_basis == 15000.0

    def test_position_with_price(self):
        """Test position price update (immutable)"""
        original = Position(
            symbol='AAPL',
            side=PositionSide.LONG,
            quantity=100,
            entry_price=150.0,
            current_price=155.0
        )

        updated = original.with_price(160.0)

        assert original.current_price == 155.0  # Original unchanged
        assert updated.current_price == 160.0
        assert updated.entry_price == 150.0

    def test_portfolio_state(self):
        """Test portfolio state"""
        position = Position(
            symbol='AAPL',
            side=PositionSide.LONG,
            quantity=100,
            entry_price=150.0,
            current_price=160.0
        )

        portfolio = PortfolioState(
            timestamp=datetime.now(),
            account_balance=100000.0,
            buying_power=84000.0,
            cash=84000.0,
            positions={'AAPL': position},
            pending_orders={}
        )

        assert portfolio.total_market_value == 16000.0
        assert portfolio.total_unrealized_pnl == 1000.0
        assert portfolio.position_count == 1

    def test_trading_state(self):
        """Test trading state"""
        portfolio = PortfolioState(
            timestamp=datetime.now(),
            account_balance=100000.0,
            buying_power=100000.0,
            cash=100000.0,
            positions={},
            pending_orders={}
        )

        state = TradingState(
            mode='simulation',
            is_running=True,
            portfolio=portfolio
        )

        assert state.mode == 'simulation'
        assert state.is_running is True
        assert state.circuit_breaker_active is False


class TestEvents:
    """Tests for Event System"""

    def test_event_creation(self):
        """Test event creation"""
        event = Event(
            type=EventType.ORDER_FILLED,
            data={'order_id': '123', 'price': 150.0},
            source='test'
        )

        assert event.type == EventType.ORDER_FILLED
        assert event.data['order_id'] == '123'
        assert event.source == 'test'

    def test_event_to_dict(self):
        """Test event serialization"""
        event = Event(
            type=EventType.PRICE_UPDATE,
            data={'symbol': 'AAPL', 'price': 150.0},
            source='data_provider'
        )

        d = event.to_dict()
        assert d['type'] == 'price.update'
        assert d['data']['symbol'] == 'AAPL'
        assert d['source'] == 'data_provider'

    def test_event_bus_singleton(self):
        """Test event bus singleton pattern"""
        bus1 = EventBus.get_instance()
        bus2 = EventBus.get_instance()

        assert bus1 is bus2

    def test_subscribe_and_publish_sync(self):
        """Test synchronous subscribe and publish"""
        bus = EventBus()
        received_events = []

        def handler(event):
            received_events.append(event)

        bus.subscribe(EventType.ORDER_FILLED, handler)

        event = Event(
            type=EventType.ORDER_FILLED,
            data={'order_id': '123'}
        )
        bus.publish_sync(event)

        assert len(received_events) == 1
        assert received_events[0].data['order_id'] == '123'

    @pytest.mark.asyncio
    async def test_subscribe_and_publish_async(self):
        """Test asynchronous subscribe and publish"""
        bus = EventBus()
        received_events = []

        async def async_handler(event):
            received_events.append(event)

        bus.subscribe(EventType.POSITION_OPENED, async_handler, is_async=True)

        event = Event(
            type=EventType.POSITION_OPENED,
            data={'symbol': 'AAPL'}
        )
        await bus.publish(event)

        assert len(received_events) == 1
        assert received_events[0].data['symbol'] == 'AAPL'

    def test_unsubscribe(self):
        """Test unsubscribe from events"""
        bus = EventBus()
        received = []

        def handler(event):
            received.append(event)

        bus.subscribe(EventType.ERROR, handler)
        bus.unsubscribe(EventType.ERROR, handler)

        event = Event(type=EventType.ERROR, data={})
        bus.publish_sync(event)

        assert len(received) == 0

    def test_event_history(self):
        """Test event history tracking"""
        bus = EventBus()

        for i in range(5):
            event = Event(
                type=EventType.PRICE_UPDATE,
                data={'index': i}
            )
            bus.publish_sync(event)

        history = bus.get_history(EventType.PRICE_UPDATE)
        assert len(history) == 5

        history = bus.get_history(limit=3)
        assert len(history) == 3

    def test_event_history_limit(self):
        """Test event history max limit"""
        bus = EventBus()
        bus._max_history = 10

        for i in range(15):
            event = Event(type=EventType.INFO, data={'i': i})
            bus.publish_sync(event)

        assert len(bus._history) == 10

    def test_clear_history(self):
        """Test clearing event history"""
        bus = EventBus()

        event = Event(type=EventType.INFO, data={})
        bus.publish_sync(event)

        bus.clear_history()
        assert len(bus._history) == 0

    @pytest.mark.asyncio
    async def test_emit_convenience_function(self):
        """Test emit convenience function"""
        # Reset singleton for clean test
        EventBus._instance = None
        bus = get_event_bus()
        received = []

        async def handler(event):
            received.append(event)

        bus.subscribe(EventType.SIGNAL_GENERATED, handler, is_async=True)

        await emit(EventType.SIGNAL_GENERATED, {'symbol': 'AAPL'}, source='test')

        assert len(received) == 1
        assert received[0].source == 'test'

    def test_emit_sync_convenience_function(self):
        """Test emit_sync convenience function"""
        # Reset singleton for clean test
        EventBus._instance = None
        bus = get_event_bus()
        received = []

        def handler(event):
            received.append(event)

        bus.subscribe(EventType.WARNING, handler)

        emit_sync(EventType.WARNING, {'message': 'test'}, source='system')

        assert len(received) == 1

    def test_handler_error_handling(self):
        """Test error handling in handlers"""
        bus = EventBus()

        def bad_handler(event):
            raise ValueError("Handler error")

        def good_handler(event):
            pass

        bus.subscribe(EventType.INFO, bad_handler)
        bus.subscribe(EventType.INFO, good_handler)

        # Should not raise, just log error
        event = Event(type=EventType.INFO, data={})
        bus.publish_sync(event)

    def test_multiple_handlers_same_event(self):
        """Test multiple handlers for same event"""
        bus = EventBus()
        results = []

        def handler1(event):
            results.append('handler1')

        def handler2(event):
            results.append('handler2')

        bus.subscribe(EventType.ORDER_SUBMITTED, handler1)
        bus.subscribe(EventType.ORDER_SUBMITTED, handler2)

        event = Event(type=EventType.ORDER_SUBMITTED, data={})
        bus.publish_sync(event)

        assert 'handler1' in results
        assert 'handler2' in results
