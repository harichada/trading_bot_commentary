#!/usr/bin/env python3
"""
Institutional Trading System - Main Integration Layer
=====================================================
This module integrates all institutional-grade components into a unified system.

Features:
- Unified system initialization
- Component orchestration
- Event-driven architecture
- Comprehensive monitoring
- Production-ready deployment
"""

import asyncio
import logging
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

# Core institutional modules
from institutional_core import (
    InstitutionalTradingCore,
    TradingState,
    TradingStateMachine,
    AtomicStateManager,
    IdempotentOrderManager,
    PositionReconciler,
    HealthMonitor,
    EnhancedCircuitBreaker,
    PositionSizeValidator,
    FeatureValidator,
    HealthCheck,
)

from infrastructure import (
    Environment,
    InfrastructureConfig,
    ConfigurationManager,
    FeatureFlagManager,
    GracefulShutdownManager,
    RateLimiter,
    DeploymentManager,
    DependencyContainer,
)

# Set up logging
logger = logging.getLogger('InstitutionalTradingSystem')


# ============================================================================
# EVENT BUS
# ============================================================================

@dataclass
class TradingEvent:
    """Base class for all trading events"""
    event_type: str
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    correlation_id: str = ""


class EventBus:
    """
    Central event bus for system-wide communication.
    Implements publish-subscribe pattern for loose coupling.
    """

    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._event_history: List[TradingEvent] = []
        self._max_history = 1000
        self._lock = asyncio.Lock()

    def subscribe(self, event_type: str, handler: Callable):
        """Subscribe to an event type"""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)
        logger.debug(f"Subscribed to event: {event_type}")

    def unsubscribe(self, event_type: str, handler: Callable):
        """Unsubscribe from an event type"""
        if event_type in self._subscribers:
            self._subscribers[event_type].remove(handler)

    async def publish(self, event: TradingEvent):
        """Publish an event to all subscribers"""
        async with self._lock:
            # Store in history
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history = self._event_history[-self._max_history:]

        # Notify subscribers
        handlers = self._subscribers.get(event.event_type, [])
        handlers.extend(self._subscribers.get('*', []))  # Wildcard subscribers

        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                logger.error(f"Event handler error for {event.event_type}: {e}")

    def get_recent_events(self, event_type: str = None, limit: int = 100) -> List[TradingEvent]:
        """Get recent events, optionally filtered by type"""
        events = self._event_history[-limit:]
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return events


# ============================================================================
# SYSTEM COMPONENTS
# ============================================================================

class SystemComponent:
    """Base class for all system components"""

    def __init__(self, name: str, event_bus: EventBus):
        self.name = name
        self.event_bus = event_bus
        self._running = False
        self._health_status = "unknown"

    async def start(self):
        """Start the component"""
        self._running = True
        self._health_status = "healthy"
        logger.info(f"Component started: {self.name}")

    async def stop(self):
        """Stop the component"""
        self._running = False
        self._health_status = "stopped"
        logger.info(f"Component stopped: {self.name}")

    def is_healthy(self) -> bool:
        """Check component health"""
        return self._health_status == "healthy"

    async def health_check(self) -> HealthCheck:
        """Perform health check"""
        return HealthCheck(
            component=self.name,
            status="OK" if self.is_healthy() else "CRITICAL",
            message=f"Component {self.name} is {self._health_status}",
            checked_at=datetime.now()
        )


# ============================================================================
# TRADING COORDINATOR
# ============================================================================

class TradingCoordinator(SystemComponent):
    """
    Coordinates trading operations across all components.
    Ensures proper sequencing and state management.
    """

    def __init__(self, event_bus: EventBus, institutional_core: InstitutionalTradingCore):
        super().__init__("TradingCoordinator", event_bus)
        self.core = institutional_core
        self._trading_enabled = False
        self._pending_signals: List[Dict] = []

    async def start(self):
        await super().start()

        # Subscribe to events
        self.event_bus.subscribe("signal.generated", self._handle_signal)
        self.event_bus.subscribe("position.closed", self._handle_position_closed)
        self.event_bus.subscribe("risk.limit_breach", self._handle_risk_breach)

    async def enable_trading(self):
        """Enable trading operations"""
        # Check preconditions
        if self.core.state_machine.state != TradingState.CONNECTED:
            logger.warning("Cannot enable trading: not in CONNECTED state")
            return False

        # Run reconciliation
        reconciled = await self._run_reconciliation()
        if not reconciled:
            logger.warning("Cannot enable trading: reconciliation failed")
            return False

        # Transition to trading state
        if self.core.state_machine.transition_to(TradingState.TRADING, "trading_enabled"):
            self._trading_enabled = True
            await self.event_bus.publish(TradingEvent(
                event_type="trading.enabled",
                source=self.name
            ))
            logger.info("Trading enabled")
            return True

        return False

    async def disable_trading(self, reason: str = "manual"):
        """Disable trading operations"""
        self._trading_enabled = False
        await self.event_bus.publish(TradingEvent(
            event_type="trading.disabled",
            source=self.name,
            data={"reason": reason}
        ))
        logger.info(f"Trading disabled: {reason}")

    async def _run_reconciliation(self) -> bool:
        """Run position reconciliation"""
        self.core.state_machine.transition_to(TradingState.RECONCILING, "reconciliation_start")

        try:
            # This would call actual broker API
            # For now, return success
            await asyncio.sleep(0.1)  # Simulate API call

            self.core.state_machine.transition_to(TradingState.CONNECTED, "reconciliation_complete")
            return True

        except Exception as e:
            logger.error(f"Reconciliation failed: {e}")
            self.core.state_machine.transition_to(TradingState.CONNECTED, "reconciliation_failed")
            return False

    async def _handle_signal(self, event: TradingEvent):
        """Handle trading signal"""
        if not self._trading_enabled:
            logger.debug("Signal ignored: trading disabled")
            return

        signal_data = event.data
        logger.info(f"Processing signal: {signal_data.get('symbol')} {signal_data.get('direction')}")

        # Validate signal
        if not self._validate_signal(signal_data):
            return

        # Check if operation allowed
        if not self.core.state_machine.is_operation_allowed('place_order'):
            logger.warning("Order placement not allowed in current state")
            return

        # Queue for execution
        self._pending_signals.append(signal_data)

    def _validate_signal(self, signal_data: Dict) -> bool:
        """Validate a trading signal"""
        required_fields = ['symbol', 'direction', 'quantity', 'price']
        for field in required_fields:
            if field not in signal_data:
                logger.warning(f"Signal missing required field: {field}")
                return False
        return True

    async def _handle_position_closed(self, event: TradingEvent):
        """Handle position closed event"""
        symbol = event.data.get('symbol')
        pnl = event.data.get('pnl', 0)
        logger.info(f"Position closed: {symbol}, P&L: ${pnl:.2f}")

    async def _handle_risk_breach(self, event: TradingEvent):
        """Handle risk limit breach"""
        breach_type = event.data.get('type')
        logger.warning(f"Risk limit breach: {breach_type}")

        # Transition to risk locked state
        self.core.state_machine.transition_to(TradingState.RISK_LOCKED, f"risk_breach_{breach_type}")
        await self.disable_trading("risk_breach")


# ============================================================================
# MARKET DATA MANAGER
# ============================================================================

class MarketDataManager(SystemComponent):
    """
    Manages market data collection and distribution.
    """

    def __init__(self, event_bus: EventBus, rate_limiter: RateLimiter):
        super().__init__("MarketDataManager", event_bus)
        self.rate_limiter = rate_limiter
        self._subscriptions: Set[str] = set()
        self._last_quotes: Dict[str, Dict] = {}
        self._data_stale_threshold = timedelta(seconds=30)

    async def subscribe_symbol(self, symbol: str):
        """Subscribe to market data for a symbol"""
        self._subscriptions.add(symbol)
        logger.info(f"Subscribed to market data: {symbol}")

    async def unsubscribe_symbol(self, symbol: str):
        """Unsubscribe from market data"""
        self._subscriptions.discard(symbol)
        logger.info(f"Unsubscribed from market data: {symbol}")

    async def get_quote(self, symbol: str) -> Optional[Dict]:
        """Get current quote for a symbol"""
        # Check rate limit
        if not await self.rate_limiter.acquire():
            logger.warning("Rate limit exceeded for market data")
            return self._last_quotes.get(symbol)

        # Fetch quote (simulated)
        quote = await self._fetch_quote(symbol)

        if quote:
            self._last_quotes[symbol] = quote
            await self.event_bus.publish(TradingEvent(
                event_type="market.quote",
                source=self.name,
                data=quote
            ))

        return quote

    async def _fetch_quote(self, symbol: str) -> Optional[Dict]:
        """Fetch quote from data source"""
        # This would integrate with actual data provider
        return {
            'symbol': symbol,
            'bid': 100.0,
            'ask': 100.05,
            'last': 100.02,
            'volume': 1000000,
            'timestamp': datetime.now().isoformat()
        }

    def is_data_stale(self, symbol: str) -> bool:
        """Check if data for symbol is stale"""
        quote = self._last_quotes.get(symbol)
        if not quote:
            return True

        quote_time = datetime.fromisoformat(quote['timestamp'])
        return datetime.now() - quote_time > self._data_stale_threshold


# ============================================================================
# STRATEGY EXECUTOR
# ============================================================================

class StrategyExecutor(SystemComponent):
    """
    Executes trading strategies and generates signals.
    """

    def __init__(self, event_bus: EventBus, feature_validator: FeatureValidator):
        super().__init__("StrategyExecutor", event_bus)
        self.feature_validator = feature_validator
        self._strategies: Dict[str, Callable] = {}
        self._strategy_performance: Dict[str, Dict] = {}

    def register_strategy(self, name: str, strategy_fn: Callable):
        """Register a trading strategy"""
        self._strategies[name] = strategy_fn
        self._strategy_performance[name] = {
            'trades': 0,
            'wins': 0,
            'total_pnl': 0.0,
            'enabled': True
        }
        logger.info(f"Registered strategy: {name}")

    async def evaluate_strategies(self, market_data: Dict) -> List[Dict]:
        """Evaluate all strategies and return signals"""
        signals = []

        for name, strategy in self._strategies.items():
            if not self._strategy_performance[name]['enabled']:
                continue

            try:
                signal = await self._evaluate_strategy(name, strategy, market_data)
                if signal:
                    signals.append(signal)

            except Exception as e:
                logger.error(f"Strategy {name} error: {e}")
                self._check_strategy_health(name)

        return signals

    async def _evaluate_strategy(self, name: str, strategy: Callable, market_data: Dict) -> Optional[Dict]:
        """Evaluate a single strategy"""
        # Extract features
        features = self._extract_features(market_data)

        # Validate features
        is_valid, issues, cleaned_features = self.feature_validator.validate(features)
        if not is_valid:
            logger.warning(f"Feature validation issues in {name}: {issues}")
            features = cleaned_features

        # Run strategy
        if asyncio.iscoroutinefunction(strategy):
            signal = await strategy(market_data, features)
        else:
            signal = strategy(market_data, features)

        if signal:
            signal['strategy'] = name
            await self.event_bus.publish(TradingEvent(
                event_type="signal.generated",
                source=self.name,
                data=signal
            ))

        return signal

    def _extract_features(self, market_data: Dict) -> 'np.ndarray':
        """Extract features from market data"""
        import numpy as np
        # Simplified feature extraction
        return np.array([
            market_data.get('price', 0),
            market_data.get('volume', 0),
            market_data.get('volatility', 0),
        ])

    def _check_strategy_health(self, name: str):
        """Check if strategy should be quarantined"""
        perf = self._strategy_performance[name]

        # Quarantine after 5 consecutive errors
        if perf.get('consecutive_errors', 0) >= 5:
            perf['enabled'] = False
            logger.warning(f"Strategy {name} quarantined due to errors")


# ============================================================================
# EXECUTION ENGINE
# ============================================================================

class ExecutionEngine(SystemComponent):
    """
    Handles order execution and fill management.
    """

    def __init__(self, event_bus: EventBus, order_manager: IdempotentOrderManager,
                 position_validator: PositionSizeValidator,
                 circuit_breaker: EnhancedCircuitBreaker):
        super().__init__("ExecutionEngine", event_bus)
        self.order_manager = order_manager
        self.position_validator = position_validator
        self.circuit_breaker = circuit_breaker
        self._pending_orders: Dict[str, Dict] = {}

    async def execute_order(self, order_request: Dict) -> Dict:
        """Execute a trading order"""
        # Check circuit breaker
        if not self.circuit_breaker.can_execute():
            return {'status': 'BLOCKED', 'reason': 'circuit_breaker_open'}

        # Validate position size
        is_valid, message = self.position_validator.validate(
            symbol=order_request['symbol'],
            quantity=order_request['quantity'],
            price=order_request['price'],
            portfolio_value=order_request.get('portfolio_value', 100000)
        )

        if not is_valid:
            return {'status': 'REJECTED', 'reason': message}

        # Submit order with idempotency
        async def execute():
            # Actual broker API call would go here
            return {
                'order_id': f"ORD_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                'status': 'FILLED',
                'fill_price': order_request['price'],
                'fill_quantity': order_request['quantity']
            }

        result = await self.order_manager.check_or_submit(
            symbol=order_request['symbol'],
            side=order_request['side'],
            quantity=order_request['quantity'],
            price=order_request['price'],
            timestamp=datetime.now(),
            execute_fn=execute
        )

        if result['status'] == 'SUCCESS':
            self.circuit_breaker.record_success()
            await self.event_bus.publish(TradingEvent(
                event_type="order.filled",
                source=self.name,
                data=result['result']
            ))
        elif result['status'] == 'FAILED':
            self.circuit_breaker.record_failure()

        return result


# ============================================================================
# INSTITUTIONAL TRADING SYSTEM
# ============================================================================

class InstitutionalTradingSystem:
    """
    Main system orchestrator that integrates all components.
    """

    def __init__(self, config_path: str = None):
        # Configuration
        self.config_manager = ConfigurationManager()
        if config_path:
            self.config = self.config_manager.load_config(config_path)
        else:
            self.config = self.config_manager.load_config()

        # Core infrastructure
        self.event_bus = EventBus()
        self.shutdown_manager = GracefulShutdownManager()
        self.deployment_manager = DeploymentManager(self.config_manager)
        self.feature_flags = FeatureFlagManager(self.config_manager)

        # Institutional core
        self.institutional_core = InstitutionalTradingCore({
            'state_file': 'trading_state_institutional.json',
            'max_position_value': self.config.trading.max_position_value,
            'max_portfolio_percent': self.config.trading.max_portfolio_percent,
        })

        # Rate limiters
        self.api_rate_limiter = RateLimiter(
            rate=self.config.api.rate_limit_per_second,
            burst=self.config.api.rate_limit_per_second * 2
        )

        # System components
        self.components: Dict[str, SystemComponent] = {}
        self._setup_components()

        # Register shutdown handlers
        self._register_shutdown_handlers()

        logger.info("Institutional Trading System initialized")

    def _setup_components(self):
        """Initialize all system components"""
        # Trading Coordinator
        self.trading_coordinator = TradingCoordinator(
            self.event_bus,
            self.institutional_core
        )
        self.components['trading_coordinator'] = self.trading_coordinator

        # Market Data Manager
        self.market_data = MarketDataManager(
            self.event_bus,
            self.api_rate_limiter
        )
        self.components['market_data'] = self.market_data

        # Strategy Executor
        self.strategy_executor = StrategyExecutor(
            self.event_bus,
            self.institutional_core.feature_validator
        )
        self.components['strategy_executor'] = self.strategy_executor

        # Execution Engine
        self.execution_engine = ExecutionEngine(
            self.event_bus,
            self.institutional_core.order_manager,
            self.institutional_core.position_validator,
            self.institutional_core.circuit_breakers['order']
        )
        self.components['execution_engine'] = self.execution_engine

    def _register_shutdown_handlers(self):
        """Register graceful shutdown handlers"""
        # Priority 10: Save state
        self.shutdown_manager.register_handler(
            self._save_state_handler,
            priority=10
        )

        # Priority 20: Stop components
        self.shutdown_manager.register_handler(
            self._stop_components_handler,
            priority=20
        )

        # Priority 30: Close positions if needed
        self.shutdown_manager.register_handler(
            self._close_positions_handler,
            priority=30
        )

    async def _save_state_handler(self):
        """Shutdown handler: save state"""
        logger.info("Saving system state...")
        await self.institutional_core.save_state()

    async def _stop_components_handler(self):
        """Shutdown handler: stop components"""
        logger.info("Stopping components...")
        for name, component in self.components.items():
            try:
                await component.stop()
            except Exception as e:
                logger.error(f"Error stopping {name}: {e}")

    async def _close_positions_handler(self):
        """Shutdown handler: optionally close positions"""
        # Only close positions if configured to do so
        if self.feature_flags.is_enabled('close_positions_on_shutdown'):
            logger.info("Closing positions...")
            # Implementation would go here

    async def start(self):
        """Start the trading system"""
        logger.info("Starting Institutional Trading System...")

        # Initialize deployment info
        self.deployment_manager.initialize(
            version="2.0.0",
            commit_hash="institutional"
        )

        # Initialize institutional core
        await self.institutional_core.initialize()

        # Start all components
        for name, component in self.components.items():
            try:
                await component.start()
            except Exception as e:
                logger.error(f"Failed to start {name}: {e}")
                raise

        # Run health checks
        await self._run_health_checks()

        # Mark as ready
        self.deployment_manager.set_ready(True)

        logger.info("Institutional Trading System started successfully")

    async def stop(self):
        """Stop the trading system"""
        logger.info("Stopping Institutional Trading System...")
        await self.shutdown_manager.shutdown()

    async def _run_health_checks(self):
        """Run health checks on all components"""
        health_results = {}

        for name, component in self.components.items():
            health_results[name] = await component.health_check()

        # Check institutional core health
        core_health = await self.institutional_core.health_monitor.run_all_checks()
        health_results.update(core_health)

        # Log results
        unhealthy = [k for k, v in health_results.items()
                     if hasattr(v, 'status') and v.status != 'OK']

        if unhealthy:
            logger.warning(f"Unhealthy components: {unhealthy}")
        else:
            logger.info("All components healthy")

        return health_results

    async def run_trading_loop(self):
        """Main trading loop"""
        # Enable trading
        await self.trading_coordinator.enable_trading()

        try:
            while not self.shutdown_manager.shutdown_requested:
                # Check health periodically
                await self._run_health_checks()

                # Process market data
                # This would integrate with actual market data

                # Save state periodically
                await self.institutional_core.save_state()

                # Sleep before next iteration
                await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"Trading loop error: {e}")
            raise

        finally:
            await self.trading_coordinator.disable_trading("shutdown")

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'deployment': self.deployment_manager.get_info(),
            'state_machine': self.institutional_core.state_machine.serialize(),
            'health': self.deployment_manager.health_check(),
            'components': {
                name: comp.is_healthy()
                for name, comp in self.components.items()
            },
            'circuit_breakers': {
                name: cb.get_metrics()
                for name, cb in self.institutional_core.circuit_breakers.items()
            },
            'rate_limiters': {
                'api': self.api_rate_limiter.get_stats()
            }
        }


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

async def main():
    """Main entry point for the institutional trading system"""
    import argparse

    parser = argparse.ArgumentParser(description='Institutional Trading System')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--paper', action='store_true', help='Run in paper trading mode')
    args = parser.parse_args()

    # Create and start system
    system = InstitutionalTradingSystem(config_path=args.config)

    try:
        await system.start()
        await system.run_trading_loop()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        await system.stop()


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    asyncio.run(main())
