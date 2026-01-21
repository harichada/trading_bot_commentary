"""
Trading Engine

Core orchestration for the trading system. Manages the trading loop,
coordinates between components, and maintains system state.
"""

import asyncio
import logging
from datetime import datetime, time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field

from .config import Config, TradingMode, get_config
from .state import (
    TradingState, PortfolioState, Position, Order,
    OrderSide, OrderType, OrderStatus, PositionSide
)
from .events import EventBus, Event, EventType, get_event_bus, emit

logger = logging.getLogger(__name__)


@dataclass
class EngineStatus:
    """Current engine status"""
    is_running: bool = False
    mode: TradingMode = TradingMode.SIMULATION
    last_analysis_time: Optional[datetime] = None
    last_price_update: Optional[datetime] = None
    error_count: int = 0
    last_error: Optional[str] = None
    uptime_seconds: float = 0.0
    start_time: Optional[datetime] = None


class TradingEngine:
    """
    Core trading engine that orchestrates all trading operations.

    Responsibilities:
    - Manage the main trading loop
    - Coordinate between data, strategy, risk, and execution components
    - Maintain system state
    - Handle lifecycle (start, stop, pause)
    - Emit events for system-wide communication

    Usage:
        engine = TradingEngine(config)
        await engine.start()
        # ... trading runs ...
        await engine.stop()
    """

    _instance: Optional['TradingEngine'] = None

    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self.event_bus = get_event_bus()

        # State
        self._status = EngineStatus(mode=self.config.trading.mode)
        self._state: Optional[TradingState] = None

        # Component references (injected)
        self._data_provider = None
        self._strategy_manager = None
        self._risk_manager = None
        self._order_executor = None
        self._broker = None

        # Tasks
        self._main_loop_task: Optional[asyncio.Task] = None
        self._price_update_task: Optional[asyncio.Task] = None

        # Callbacks
        self._on_signal_callbacks: List[Callable] = []
        self._on_trade_callbacks: List[Callable] = []

        # Lock for state modifications
        self._state_lock = asyncio.Lock()

    @classmethod
    def get_instance(cls, config: Optional[Config] = None) -> 'TradingEngine':
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset singleton (for testing)"""
        cls._instance = None

    # Component injection
    def set_data_provider(self, provider):
        """Inject data provider component"""
        self._data_provider = provider

    def set_strategy_manager(self, manager):
        """Inject strategy manager component"""
        self._strategy_manager = manager

    def set_risk_manager(self, manager):
        """Inject risk manager component"""
        self._risk_manager = manager

    def set_order_executor(self, executor):
        """Inject order executor component"""
        self._order_executor = executor

    def set_broker(self, broker):
        """Inject broker connection"""
        self._broker = broker

    @property
    def status(self) -> EngineStatus:
        """Get current engine status"""
        if self._status.start_time:
            self._status.uptime_seconds = (
                datetime.now() - self._status.start_time
            ).total_seconds()
        return self._status

    @property
    def state(self) -> Optional[TradingState]:
        """Get current trading state"""
        return self._state

    @property
    def is_running(self) -> bool:
        """Check if engine is running"""
        return self._status.is_running

    @property
    def mode(self) -> TradingMode:
        """Get current trading mode"""
        return self._status.mode

    async def start(self) -> bool:
        """
        Start the trading engine.

        Returns:
            True if started successfully, False otherwise
        """
        if self._status.is_running:
            logger.warning("Engine already running")
            return False

        logger.info(f"Starting trading engine in {self._status.mode.value} mode")

        try:
            # Initialize state
            await self._initialize_state()

            # Validate configuration
            errors = self.config.validate()
            if errors:
                for error in errors:
                    logger.error(f"Config validation error: {error}")
                return False

            # Connect to broker if needed
            if self._status.mode in [TradingMode.PAPER, TradingMode.LIVE]:
                if not await self._connect_broker():
                    logger.error("Failed to connect to broker")
                    return False

            # Start background tasks
            self._status.is_running = True
            self._status.start_time = datetime.now()

            self._main_loop_task = asyncio.create_task(self._main_loop())
            self._price_update_task = asyncio.create_task(self._price_update_loop())

            # Emit started event
            await emit(EventType.ENGINE_STARTED, {
                'mode': self._status.mode.value,
                'timestamp': datetime.now().isoformat()
            }, source='engine')

            logger.info("Trading engine started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start engine: {e}")
            self._status.last_error = str(e)
            self._status.error_count += 1
            return False

    async def stop(self):
        """Stop the trading engine gracefully"""
        if not self._status.is_running:
            return

        logger.info("Stopping trading engine...")
        self._status.is_running = False

        # Cancel background tasks
        if self._main_loop_task:
            self._main_loop_task.cancel()
            try:
                await self._main_loop_task
            except asyncio.CancelledError:
                pass

        if self._price_update_task:
            self._price_update_task.cancel()
            try:
                await self._price_update_task
            except asyncio.CancelledError:
                pass

        # Emit stopped event
        await emit(EventType.ENGINE_STOPPED, {
            'uptime_seconds': self._status.uptime_seconds,
            'timestamp': datetime.now().isoformat()
        }, source='engine')

        # Save state
        if self._state:
            self._state.save()

        logger.info("Trading engine stopped")

    async def set_mode(self, mode: TradingMode):
        """Change trading mode (requires restart)"""
        if self._status.is_running:
            logger.warning("Cannot change mode while running - stop first")
            return

        old_mode = self._status.mode
        self._status.mode = mode
        self.config.trading.mode = mode

        await emit(EventType.MODE_CHANGED, {
            'old_mode': old_mode.value,
            'new_mode': mode.value
        }, source='engine')

        logger.info(f"Mode changed from {old_mode.value} to {mode.value}")

    async def _initialize_state(self):
        """Initialize trading state"""
        # Try to load existing state
        self._state = TradingState.load()

        if self._state is None:
            # Create fresh state
            self._state = TradingState(
                mode=self._status.mode.value,
                is_running=True,
                portfolio=PortfolioState(
                    timestamp=datetime.now(),
                    account_balance=100000.0,  # Default for simulation
                    buying_power=100000.0,
                    cash=100000.0,
                    positions={},
                    pending_orders={}
                )
            )
            logger.info("Created fresh trading state")
        else:
            self._state.is_running = True
            logger.info("Loaded existing trading state")

    async def _connect_broker(self) -> bool:
        """Connect to broker"""
        if self._broker is None:
            logger.warning("No broker configured")
            return self._status.mode == TradingMode.SIMULATION

        try:
            connected = await self._broker.connect()
            if connected:
                logger.info("Connected to broker")
                # Sync account state
                await self._sync_account_state()
            return connected
        except Exception as e:
            logger.error(f"Broker connection failed: {e}")
            return False

    async def _sync_account_state(self):
        """Sync state with broker account"""
        if self._broker is None:
            return

        try:
            account_info = await self._broker.get_account()
            if account_info:
                async with self._state_lock:
                    self._state.portfolio.account_balance = account_info.get('balance', 0)
                    self._state.portfolio.buying_power = account_info.get('buying_power', 0)
                    self._state.portfolio.cash = account_info.get('cash', 0)

            # Sync positions
            positions = await self._broker.get_positions()
            if positions:
                async with self._state_lock:
                    for pos_data in positions:
                        symbol = pos_data['symbol']
                        self._state.portfolio.positions[symbol] = Position(
                            symbol=symbol,
                            side=PositionSide(pos_data.get('side', 'long')),
                            quantity=pos_data.get('quantity', 0),
                            entry_price=pos_data.get('average_price', 0),
                            current_price=pos_data.get('current_price', 0),
                            is_external=True,  # Mark as external
                            is_manually_managed=True
                        )

        except Exception as e:
            logger.error(f"Failed to sync account state: {e}")

    async def _main_loop(self):
        """Main trading loop"""
        interval = self.config.trading.analysis_interval_seconds

        while self._status.is_running:
            try:
                # Check market hours
                if not self._is_market_hours():
                    await asyncio.sleep(60)
                    continue

                # Run analysis cycle
                await self._analysis_cycle()

                self._status.last_analysis_time = datetime.now()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                self._status.error_count += 1
                self._status.last_error = str(e)

                # Emit error event
                await emit(EventType.ERROR, {
                    'error': str(e),
                    'context': 'main_loop'
                }, source='engine')

            await asyncio.sleep(interval)

    async def _price_update_loop(self):
        """Price update loop"""
        interval = self.config.trading.price_update_interval_seconds

        while self._status.is_running:
            try:
                await self._update_prices()
                self._status.last_price_update = datetime.now()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error updating prices: {e}")

            await asyncio.sleep(interval)

    async def _analysis_cycle(self):
        """Single analysis cycle"""
        if self._data_provider is None or self._strategy_manager is None:
            return

        watchlist = self.config.trading.watchlist

        for symbol in watchlist:
            try:
                # Get market data
                data = await self._data_provider.get_data(symbol)
                if data is None:
                    continue

                # Generate signals
                signals = await self._strategy_manager.analyze(symbol, data)

                for signal in signals:
                    await self._process_signal(signal)

            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")

        # Manage existing positions
        await self._manage_positions()

        # Emit analysis complete event
        await emit(EventType.ANALYSIS_COMPLETE, {
            'symbols_analyzed': len(watchlist),
            'timestamp': datetime.now().isoformat()
        }, source='engine')

    async def _process_signal(self, signal: Dict[str, Any]):
        """Process a trading signal"""
        if self._risk_manager is None:
            return

        symbol = signal.get('symbol')
        action = signal.get('action')  # 'buy', 'sell', 'hold'
        confidence = signal.get('confidence', 0)

        # Emit signal event
        await emit(EventType.SIGNAL_GENERATED, signal, source='engine')

        # Call signal callbacks
        for callback in self._on_signal_callbacks:
            try:
                await callback(signal)
            except Exception as e:
                logger.error(f"Error in signal callback: {e}")

        if action == 'hold' or confidence < 0.5:
            return

        # Risk check
        risk_approved, risk_reason = await self._risk_manager.check_signal(signal)

        if not risk_approved:
            await emit(EventType.SIGNAL_REJECTED, {
                'signal': signal,
                'reason': risk_reason
            }, source='engine')
            return

        # Calculate position size
        position_size = await self._risk_manager.calculate_position_size(
            symbol,
            self._state.portfolio.buying_power
        )

        if position_size <= 0:
            return

        # Create order
        if action == 'buy':
            await self._submit_order(
                symbol=symbol,
                side=OrderSide.BUY,
                quantity=position_size,
                signal=signal
            )
        elif action == 'sell':
            if symbol in self._state.portfolio.positions:
                pos = self._state.portfolio.positions[symbol]
                # Skip externally managed positions
                if pos.is_manually_managed:
                    return
                await self._submit_order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=pos.quantity,
                    signal=signal
                )

    async def _submit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        signal: Optional[Dict] = None
    ):
        """Submit an order"""
        import uuid

        order = Order(
            id=str(uuid.uuid4()),
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            metadata={'signal': signal} if signal else {}
        )

        logger.info(f"Submitting order: {side.value} {quantity} {symbol}")

        # Add to pending orders
        async with self._state_lock:
            self._state.portfolio.pending_orders[order.id] = order

        # Emit event
        await emit(EventType.ORDER_SUBMITTED, order.to_dict(), source='engine')

        # Execute order
        if self._order_executor:
            result = await self._order_executor.execute(order)
            await self._handle_order_result(order, result)
        elif self._status.mode == TradingMode.SIMULATION:
            # Simulate fill
            await self._simulate_fill(order)

    async def _simulate_fill(self, order: Order):
        """Simulate order fill in simulation mode"""
        if self._data_provider:
            price = await self._data_provider.get_price(order.symbol)
        else:
            price = 100.0  # Fallback

        await self._handle_order_fill(order, price, order.quantity)

    async def _handle_order_result(self, order: Order, result: Dict):
        """Handle order execution result"""
        if result.get('status') == 'filled':
            await self._handle_order_fill(
                order,
                result.get('fill_price'),
                result.get('fill_quantity')
            )
        elif result.get('status') == 'rejected':
            await emit(EventType.ORDER_REJECTED, {
                'order': order.to_dict(),
                'reason': result.get('reason')
            }, source='engine')

            async with self._state_lock:
                if order.id in self._state.portfolio.pending_orders:
                    del self._state.portfolio.pending_orders[order.id]

    async def _handle_order_fill(self, order: Order, fill_price: float, fill_quantity: int):
        """Handle filled order"""
        async with self._state_lock:
            # Remove from pending
            if order.id in self._state.portfolio.pending_orders:
                del self._state.portfolio.pending_orders[order.id]

            if order.side == OrderSide.BUY:
                # Create or add to position
                if order.symbol in self._state.portfolio.positions:
                    pos = self._state.portfolio.positions[order.symbol]
                    # Average in
                    total_qty = pos.quantity + fill_quantity
                    avg_price = (
                        (pos.entry_price * pos.quantity) +
                        (fill_price * fill_quantity)
                    ) / total_qty

                    self._state.portfolio.positions[order.symbol] = Position(
                        symbol=order.symbol,
                        side=pos.side,
                        quantity=total_qty,
                        entry_price=avg_price,
                        current_price=fill_price,
                        stop_loss=pos.stop_loss,
                        take_profit=pos.take_profit,
                        strategy=pos.strategy
                    )
                else:
                    # New position
                    stop_loss = fill_price * (1 - self.config.risk.default_stop_loss_pct)
                    take_profit = fill_price * (1 + self.config.risk.default_take_profit_pct)

                    self._state.portfolio.positions[order.symbol] = Position(
                        symbol=order.symbol,
                        side=PositionSide.LONG,
                        quantity=fill_quantity,
                        entry_price=fill_price,
                        current_price=fill_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        strategy=order.metadata.get('signal', {}).get('strategy', 'unknown')
                    )

                # Update cash
                self._state.portfolio.cash -= fill_price * fill_quantity

                await emit(EventType.POSITION_OPENED, {
                    'symbol': order.symbol,
                    'quantity': fill_quantity,
                    'price': fill_price
                }, source='engine')

            else:  # SELL
                if order.symbol in self._state.portfolio.positions:
                    pos = self._state.portfolio.positions[order.symbol]

                    # Calculate P&L
                    pnl = (fill_price - pos.entry_price) * fill_quantity
                    self._state.portfolio.daily_pnl += pnl
                    self._state.portfolio.total_pnl += pnl

                    if fill_quantity >= pos.quantity:
                        # Close position
                        del self._state.portfolio.positions[order.symbol]
                        await emit(EventType.POSITION_CLOSED, {
                            'symbol': order.symbol,
                            'pnl': pnl,
                            'price': fill_price
                        }, source='engine')
                    else:
                        # Reduce position
                        self._state.portfolio.positions[order.symbol] = Position(
                            symbol=order.symbol,
                            side=pos.side,
                            quantity=pos.quantity - fill_quantity,
                            entry_price=pos.entry_price,
                            current_price=fill_price,
                            stop_loss=pos.stop_loss,
                            take_profit=pos.take_profit,
                            strategy=pos.strategy
                        )

                        await emit(EventType.POSITION_UPDATED, {
                            'symbol': order.symbol,
                            'pnl': pnl
                        }, source='engine')

                    # Update cash
                    self._state.portfolio.cash += fill_price * fill_quantity

        # Emit fill event
        await emit(EventType.ORDER_FILLED, {
            'order_id': order.id,
            'symbol': order.symbol,
            'side': order.side.value,
            'quantity': fill_quantity,
            'price': fill_price
        }, source='engine')

        # Track consecutive losses
        if order.side == OrderSide.SELL:
            if self._state.portfolio.positions.get(order.symbol):
                pos = self._state.portfolio.positions[order.symbol]
                pnl = (fill_price - pos.entry_price) * fill_quantity
                if pnl < 0:
                    self._state.consecutive_losses += 1
                else:
                    self._state.consecutive_losses = 0

            # Check circuit breaker
            if self._state.consecutive_losses >= self.config.risk.max_consecutive_losses:
                await self._trigger_circuit_breaker()

        # Call trade callbacks
        for callback in self._on_trade_callbacks:
            try:
                await callback({
                    'order': order.to_dict(),
                    'fill_price': fill_price,
                    'fill_quantity': fill_quantity
                })
            except Exception as e:
                logger.error(f"Error in trade callback: {e}")

    async def _manage_positions(self):
        """Manage existing positions (stop loss, take profit, etc.)"""
        if self._state is None:
            return

        for symbol, position in list(self._state.portfolio.positions.items()):
            # Skip manually managed positions
            if position.is_manually_managed or position.is_external:
                continue

            # Check stop loss
            if position.stop_loss and position.current_price <= position.stop_loss:
                logger.info(f"Stop loss triggered for {symbol}")
                await emit(EventType.STOP_LOSS_HIT, {
                    'symbol': symbol,
                    'price': position.current_price,
                    'stop_loss': position.stop_loss
                }, source='engine')

                if not self.config.trading.manual_close_only:
                    await self._submit_order(
                        symbol=symbol,
                        side=OrderSide.SELL,
                        quantity=position.quantity
                    )

            # Check take profit
            elif position.take_profit and position.current_price >= position.take_profit:
                logger.info(f"Take profit triggered for {symbol}")
                await emit(EventType.TAKE_PROFIT_HIT, {
                    'symbol': symbol,
                    'price': position.current_price,
                    'take_profit': position.take_profit
                }, source='engine')

                if not self.config.trading.manual_close_only:
                    await self._submit_order(
                        symbol=symbol,
                        side=OrderSide.SELL,
                        quantity=position.quantity
                    )

    async def _update_prices(self):
        """Update prices for all positions"""
        if self._data_provider is None or self._state is None:
            return

        async with self._state_lock:
            for symbol, position in self._state.portfolio.positions.items():
                try:
                    price = await self._data_provider.get_price(symbol)
                    if price:
                        self._state.portfolio.positions[symbol] = position.with_price(price)

                        await emit(EventType.PRICE_UPDATE, {
                            'symbol': symbol,
                            'price': price
                        }, source='engine')

                except Exception as e:
                    logger.error(f"Failed to update price for {symbol}: {e}")

            # Update portfolio timestamp
            self._state.portfolio.timestamp = datetime.now()

    async def _trigger_circuit_breaker(self):
        """Trigger circuit breaker"""
        if not self.config.risk.circuit_breaker_enabled:
            return

        logger.warning("Circuit breaker triggered!")
        self._state.circuit_breaker_active = True

        await emit(EventType.CIRCUIT_BREAKER_TRIGGERED, {
            'consecutive_losses': self._state.consecutive_losses,
            'timestamp': datetime.now().isoformat()
        }, source='engine')

    def _is_market_hours(self) -> bool:
        """Check if within market hours"""
        now = datetime.now()
        market_open = time(9, 30)
        market_close = time(16, 0)

        # Weekend check
        if now.weekday() >= 5:
            return False

        current_time = now.time()

        # Regular hours
        if market_open <= current_time <= market_close:
            return True

        # Pre-market
        if self.config.trading.allow_premarket:
            premarket_open = time(4, 0)
            if premarket_open <= current_time < market_open:
                return True

        # After-hours
        if self.config.trading.allow_afterhours:
            afterhours_close = time(20, 0)
            if market_close < current_time <= afterhours_close:
                return True

        return False

    # Public methods for external control

    def on_signal(self, callback: Callable):
        """Register callback for signals"""
        self._on_signal_callbacks.append(callback)

    def on_trade(self, callback: Callable):
        """Register callback for trades"""
        self._on_trade_callbacks.append(callback)

    async def manual_buy(self, symbol: str, quantity: int) -> bool:
        """Manually submit a buy order"""
        if not self._status.is_running:
            return False

        await self._submit_order(symbol, OrderSide.BUY, quantity)
        return True

    async def manual_sell(self, symbol: str, quantity: Optional[int] = None) -> bool:
        """Manually submit a sell order"""
        if not self._status.is_running:
            return False

        if symbol not in self._state.portfolio.positions:
            return False

        pos = self._state.portfolio.positions[symbol]
        qty = quantity or pos.quantity

        await self._submit_order(symbol, OrderSide.SELL, qty)
        return True

    async def close_all_positions(self):
        """Close all positions"""
        for symbol, pos in list(self._state.portfolio.positions.items()):
            if not pos.is_manually_managed:
                await self._submit_order(symbol, OrderSide.SELL, pos.quantity)

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary"""
        if self._state is None:
            return {}

        return {
            'account_balance': self._state.portfolio.account_balance,
            'cash': self._state.portfolio.cash,
            'buying_power': self._state.portfolio.buying_power,
            'total_market_value': self._state.portfolio.total_market_value,
            'total_unrealized_pnl': self._state.portfolio.total_unrealized_pnl,
            'daily_pnl': self._state.portfolio.daily_pnl,
            'total_pnl': self._state.portfolio.total_pnl,
            'position_count': self._state.portfolio.position_count,
            'positions': {
                k: v.to_dict()
                for k, v in self._state.portfolio.positions.items()
            }
        }


# Convenience function
def get_engine() -> TradingEngine:
    """Get the global engine instance"""
    return TradingEngine.get_instance()
