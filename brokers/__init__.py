"""brokers — Multi-broker abstraction layer for Rudra Trading Engine.

Usage::

    from brokers import AlpacaBrokerAdapter, OandaBrokerAdapter, OrderResult
"""

from brokers.base import AbstractBroker, BrokerConfig, OrderResult, TickStreamer
from brokers.market_hours import ForexHours, MarketHoursPolicy, USEquityHours
from brokers.alpaca_adapter import AlpacaBrokerAdapter, AlpacaTickStreamerAdapter
from brokers.oanda_adapter import OandaBrokerAdapter, OandaTickStreamer
from brokers.coordinator import MultiBrokerCoordinator, CombinedRiskManager, BrokerSlot

try:
    from brokers.ibkr_adapter import IBKRBrokerAdapter
    from brokers.ibkr_streamer import IBKRTickStreamer
    from brokers.ibkr_connection import IBKRConnectionManager
except ImportError:
    pass  # ib_insync not installed

__all__ = [
    'AbstractBroker',
    'BrokerConfig',
    'OrderResult',
    'TickStreamer',
    'MarketHoursPolicy',
    'USEquityHours',
    'ForexHours',
    'AlpacaBrokerAdapter',
    'AlpacaTickStreamerAdapter',
    'OandaBrokerAdapter',
    'OandaTickStreamer',
    'IBKRBrokerAdapter',
    'IBKRTickStreamer',
    'IBKRConnectionManager',
    'MultiBrokerCoordinator',
    'CombinedRiskManager',
    'BrokerSlot',
]
