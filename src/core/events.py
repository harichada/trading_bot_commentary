"""
Event System

Pub/sub event system for decoupled component communication.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Callable, Optional
import asyncio
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Event types in the system"""
    # Lifecycle events
    ENGINE_STARTED = "engine.started"
    ENGINE_STOPPED = "engine.stopped"
    MODE_CHANGED = "mode.changed"

    # Market data events
    PRICE_UPDATE = "price.update"
    QUOTE_RECEIVED = "quote.received"
    MARKET_OPEN = "market.open"
    MARKET_CLOSE = "market.close"

    # Signal events
    SIGNAL_GENERATED = "signal.generated"
    SIGNAL_REJECTED = "signal.rejected"

    # Order events
    ORDER_SUBMITTED = "order.submitted"
    ORDER_FILLED = "order.filled"
    ORDER_CANCELLED = "order.cancelled"
    ORDER_REJECTED = "order.rejected"

    # Position events
    POSITION_OPENED = "position.opened"
    POSITION_UPDATED = "position.updated"
    POSITION_CLOSED = "position.closed"
    STOP_LOSS_HIT = "position.stop_loss"
    TAKE_PROFIT_HIT = "position.take_profit"

    # Risk events
    RISK_LIMIT_APPROACHED = "risk.limit_approached"
    RISK_LIMIT_BREACHED = "risk.limit_breached"
    CIRCUIT_BREAKER_TRIGGERED = "risk.circuit_breaker"
    DAILY_LOSS_LIMIT = "risk.daily_loss"

    # System events
    ERROR = "system.error"
    WARNING = "system.warning"
    INFO = "system.info"

    # Analysis events
    ANALYSIS_COMPLETE = "analysis.complete"
    STRATEGY_SIGNAL = "strategy.signal"
    ML_PREDICTION = "ml.prediction"


@dataclass
class Event:
    """Event object"""
    type: EventType
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)
    source: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'type': self.type.value,
            'timestamp': self.timestamp.isoformat(),
            'data': self.data,
            'source': self.source,
        }


class EventBus:
    """
    Central event bus for the trading system.

    Usage:
        bus = EventBus()

        # Subscribe to events
        bus.subscribe(EventType.ORDER_FILLED, my_handler)

        # Publish events
        await bus.publish(Event(
            type=EventType.ORDER_FILLED,
            data={'order_id': '123', 'price': 150.00}
        ))
    """

    _instance: Optional['EventBus'] = None

    def __init__(self):
        self._handlers: Dict[EventType, List[Callable]] = defaultdict(list)
        self._async_handlers: Dict[EventType, List[Callable]] = defaultdict(list)
        self._history: List[Event] = []
        self._max_history = 1000

    @classmethod
    def get_instance(cls) -> 'EventBus':
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def subscribe(self, event_type: EventType, handler: Callable, is_async: bool = False):
        """
        Subscribe to an event type.

        Args:
            event_type: The type of event to subscribe to
            handler: Function to call when event is published
            is_async: Whether the handler is async
        """
        if is_async:
            self._async_handlers[event_type].append(handler)
        else:
            self._handlers[event_type].append(handler)

        logger.debug(f"Subscribed handler to {event_type.value}")

    def unsubscribe(self, event_type: EventType, handler: Callable):
        """Unsubscribe from an event type"""
        if handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)
        if handler in self._async_handlers[event_type]:
            self._async_handlers[event_type].remove(handler)

    async def publish(self, event: Event):
        """
        Publish an event to all subscribers.

        Args:
            event: The event to publish
        """
        # Store in history
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        logger.debug(f"Publishing event: {event.type.value}")

        # Call sync handlers
        for handler in self._handlers[event.type]:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error in sync handler for {event.type.value}: {e}")

        # Call async handlers
        for handler in self._async_handlers[event.type]:
            try:
                await handler(event)
            except Exception as e:
                logger.error(f"Error in async handler for {event.type.value}: {e}")

    def publish_sync(self, event: Event):
        """Publish event synchronously (only calls sync handlers)"""
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        for handler in self._handlers[event.type]:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error in handler for {event.type.value}: {e}")

    def get_history(self, event_type: Optional[EventType] = None, limit: int = 100) -> List[Event]:
        """Get event history"""
        if event_type:
            events = [e for e in self._history if e.type == event_type]
        else:
            events = self._history

        return events[-limit:]

    def clear_history(self):
        """Clear event history"""
        self._history = []


# Convenience functions
def get_event_bus() -> EventBus:
    """Get the global event bus instance"""
    return EventBus.get_instance()


async def emit(event_type: EventType, data: Dict[str, Any] = None, source: str = "system"):
    """Quick emit an event"""
    event = Event(type=event_type, data=data or {}, source=source)
    await get_event_bus().publish(event)


def emit_sync(event_type: EventType, data: Dict[str, Any] = None, source: str = "system"):
    """Quick emit an event synchronously"""
    event = Event(type=event_type, data=data or {}, source=source)
    get_event_bus().publish_sync(event)
