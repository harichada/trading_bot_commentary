#!/usr/bin/env python3
"""
Enhanced Schwab Streaming Client for Real-Time Trading
Handles multiple symbol subscriptions with automatic reconnection
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
import aiohttp
import websockets
from schwab.streaming import StreamClient
from schwab import client as schwab_client

logger = logging.getLogger(__name__)

@dataclass
class StreamingQuote:
    """Real-time quote data structure"""
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    last: float
    volume: int
    bid_size: int
    ask_size: int
    high: float
    low: float
    open: float
    previous_close: float
    change: float
    change_percent: float
    spread: float = field(init=False)
    
    def __post_init__(self):
        self.spread = self.ask - self.bid if self.ask and self.bid else 0.0

@dataclass
class StreamingTrade:
    """Real-time trade data structure"""
    symbol: str
    timestamp: datetime
    price: float
    size: int
    side: str  # 'buy' or 'sell'
    trade_id: str

class EnhancedSchwabStreamClient:
    """Enhanced Schwab streaming client with real-time data handling"""
    
    def __init__(self, schwab_client: schwab_client.Client, commentary_system=None):
        self.schwab_client = schwab_client
        self.commentary_system = commentary_system
        self.stream_client = StreamClient(schwab_client)
        
        # Subscription management
        self.subscribed_symbols: set = set()
        self.quote_callbacks: Dict[str, List[Callable]] = {}
        self.trade_callbacks: Dict[str, List[Callable]] = {}
        self.level2_callbacks: Dict[str, List[Callable]] = {}
        
        # Connection management
        self.is_connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.reconnect_delay = 1.0
        
        # Data storage
        self.latest_quotes: Dict[str, StreamingQuote] = {}
        self.latest_trades: Dict[str, List[StreamingTrade]] = {}
        self.level2_data: Dict[str, Dict] = {}
        
        # Performance tracking
        self.message_count = 0
        self.last_message_time = time.time()
        self.connection_start_time = None
        
        # Event loop
        self.loop = None
        self.stream_task = None
        
    async def connect(self):
        """Establish streaming connection"""
        try:
            logger.info("Connecting to Schwab streaming service...")
            
            # Initialize stream client
            await self.stream_client.connect()
            
            # Set up message handlers
            self.stream_client.on_message = self._handle_message
            self.stream_client.on_error = self._handle_error
            self.stream_client.on_close = self._handle_close
            
            self.is_connected = True
            self.connection_start_time = time.time()
            self.reconnect_attempts = 0
            
            logger.info("Successfully connected to Schwab streaming service")
            
            # Add commentary
            if self.commentary_system:
                self.commentary_system.add_commentary(
                    TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.TECHNICAL,
                        symbol=None,
                        title="Streaming Connection Established",
                        message="Real-time data streaming is now active. Monitoring market data for subscribed symbols.",
                        importance=3
                    )
                )
            
        except Exception as e:
            logger.error(f"Failed to connect to streaming service: {e}")
            await self._handle_reconnection()
    
    async def subscribe_symbols(self, symbols: List[str]):
        """Subscribe to real-time data for multiple symbols"""
        if not self.is_connected:
            logger.warning("Not connected to streaming service. Attempting to connect...")
            await self.connect()
        
        try:
            for symbol in symbols:
                if symbol not in self.subscribed_symbols:
                    logger.info(f"Subscribing to real-time data for {symbol}")
                    
                    # Subscribe to quote data
                    await self.stream_client.subscribe_quotes([symbol])
                    
                    # Subscribe to trade data
                    await self.stream_client.subscribe_trades([symbol])
                    
                    # Subscribe to level 2 data (if available)
                    try:
                        await self.stream_client.subscribe_level2([symbol])
                    except Exception as e:
                        logger.warning(f"Level 2 subscription not available for {symbol}: {e}")
                    
                    self.subscribed_symbols.add(symbol)
                    
                    # Initialize data structures
                    self.latest_trades[symbol] = []
                    self.level2_data[symbol] = {}
                    
                    logger.info(f"Successfully subscribed to {symbol}")
            
            # Add commentary
            if self.commentary_system:
                self.commentary_system.add_commentary(
                    TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.TECHNICAL,
                        symbol=None,
                        title=f"Subscribed to {len(symbols)} Symbols",
                        message=f"Now monitoring real-time data for: {', '.join(symbols)}",
                        importance=4
                    )
                )
                
        except Exception as e:
            logger.error(f"Failed to subscribe to symbols: {e}")
            raise
    
    async def unsubscribe_symbols(self, symbols: List[str]):
        """Unsubscribe from real-time data for symbols"""
        try:
            for symbol in symbols:
                if symbol in self.subscribed_symbols:
                    logger.info(f"Unsubscribing from {symbol}")
                    
                    await self.stream_client.unsubscribe_quotes([symbol])
                    await self.stream_client.unsubscribe_trades([symbol])
                    
                    try:
                        await self.stream_client.unsubscribe_level2([symbol])
                    except:
                        pass
                    
                    self.subscribed_symbols.discard(symbol)
                    
                    # Clean up data
                    self.latest_quotes.pop(symbol, None)
                    self.latest_trades.pop(symbol, None)
                    self.level2_data.pop(symbol, None)
                    
        except Exception as e:
            logger.error(f"Failed to unsubscribe from symbols: {e}")
    
    def add_quote_callback(self, symbol: str, callback: Callable[[StreamingQuote], None]):
        """Add callback for quote updates"""
        if symbol not in self.quote_callbacks:
            self.quote_callbacks[symbol] = []
        self.quote_callbacks[symbol].append(callback)
    
    def add_trade_callback(self, symbol: str, callback: Callable[[StreamingTrade], None]):
        """Add callback for trade updates"""
        if symbol not in self.trade_callbacks:
            self.trade_callbacks[symbol] = []
        self.trade_callbacks[symbol].append(callback)
    
    def add_level2_callback(self, symbol: str, callback: Callable[[Dict], None]):
        """Add callback for level 2 updates"""
        if symbol not in self.level2_callbacks:
            self.level2_callbacks[symbol] = []
        self.level2_callbacks[symbol].append(callback)
    
    async def _handle_message(self, message: Dict):
        """Handle incoming streaming messages"""
        try:
            self.message_count += 1
            self.last_message_time = time.time()
            
            message_type = message.get('type')
            symbol = message.get('symbol')
            
            if message_type == 'quote':
                await self._handle_quote_message(message)
            elif message_type == 'trade':
                await self._handle_trade_message(message)
            elif message_type == 'level2':
                await self._handle_level2_message(message)
            elif message_type == 'heartbeat':
                await self._handle_heartbeat(message)
            else:
                logger.debug(f"Unknown message type: {message_type}")
                
        except Exception as e:
            logger.error(f"Error handling streaming message: {e}")
    
    async def _handle_quote_message(self, message: Dict):
        """Handle quote update messages"""
        try:
            symbol = message['symbol']
            timestamp = datetime.fromtimestamp(message['timestamp'] / 1000)
            
            quote = StreamingQuote(
                symbol=symbol,
                timestamp=timestamp,
                bid=message.get('bid', 0.0),
                ask=message.get('ask', 0.0),
                last=message.get('last', 0.0),
                volume=message.get('volume', 0),
                bid_size=message.get('bidSize', 0),
                ask_size=message.get('askSize', 0),
                high=message.get('high', 0.0),
                low=message.get('low', 0.0),
                open=message.get('open', 0.0),
                previous_close=message.get('previousClose', 0.0),
                change=message.get('change', 0.0),
                change_percent=message.get('changePercent', 0.0)
            )
            
            # Store latest quote
            self.latest_quotes[symbol] = quote
            
            # Trigger callbacks
            if symbol in self.quote_callbacks:
                for callback in self.quote_callbacks[symbol]:
                    try:
                        await callback(quote)
                    except Exception as e:
                        logger.error(f"Error in quote callback: {e}")
            
        except Exception as e:
            logger.error(f"Error processing quote message: {e}")
    
    async def _handle_trade_message(self, message: Dict):
        """Handle trade update messages"""
        try:
            symbol = message['symbol']
            timestamp = datetime.fromtimestamp(message['timestamp'] / 1000)
            
            trade = StreamingTrade(
                symbol=symbol,
                timestamp=timestamp,
                price=message.get('price', 0.0),
                size=message.get('size', 0),
                side=message.get('side', 'unknown'),
                trade_id=message.get('tradeId', '')
            )
            
            # Store trade (keep last 100 trades)
            if symbol not in self.latest_trades:
                self.latest_trades[symbol] = []
            
            self.latest_trades[symbol].append(trade)
            if len(self.latest_trades[symbol]) > 100:
                self.latest_trades[symbol] = self.latest_trades[symbol][-100:]
            
            # Trigger callbacks
            if symbol in self.trade_callbacks:
                for callback in self.trade_callbacks[symbol]:
                    try:
                        await callback(trade)
                    except Exception as e:
                        logger.error(f"Error in trade callback: {e}")
            
        except Exception as e:
            logger.error(f"Error processing trade message: {e}")
    
    async def _handle_level2_message(self, message: Dict):
        """Handle level 2 market depth messages"""
        try:
            symbol = message['symbol']
            
            # Store level 2 data
            self.level2_data[symbol] = {
                'bids': message.get('bids', []),
                'asks': message.get('asks', []),
                'timestamp': datetime.now()
            }
            
            # Trigger callbacks
            if symbol in self.level2_callbacks:
                for callback in self.level2_callbacks[symbol]:
                    try:
                        await callback(self.level2_data[symbol])
                    except Exception as e:
                        logger.error(f"Error in level2 callback: {e}")
            
        except Exception as e:
            logger.error(f"Error processing level2 message: {e}")
    
    async def _handle_heartbeat(self, message: Dict):
        """Handle heartbeat messages"""
        logger.debug("Received heartbeat from streaming service")
    
    async def _handle_error(self, error: Exception):
        """Handle streaming errors"""
        logger.error(f"Streaming error: {error}")
        
        if self.commentary_system:
            self.commentary_system.add_commentary(
                TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.WARNING,
                    symbol=None,
                    title="Streaming Error",
                    message=f"Real-time data stream error: {str(error)}",
                    importance=7
                )
            )
    
    async def _handle_close(self):
        """Handle connection close"""
        logger.warning("Streaming connection closed")
        self.is_connected = False
        
        if self.commentary_system:
            self.commentary_system.add_commentary(
                TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.WARNING,
                    symbol=None,
                    title="Streaming Connection Lost",
                    message="Real-time data stream connection has been lost. Attempting to reconnect...",
                    importance=6
                )
            )
        
        await self._handle_reconnection()
    
    async def _handle_reconnection(self):
        """Handle automatic reconnection"""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error("Max reconnection attempts reached")
            return
        
        self.reconnect_attempts += 1
        delay = self.reconnect_delay * (2 ** (self.reconnect_attempts - 1))  # Exponential backoff
        
        logger.info(f"Attempting to reconnect in {delay} seconds (attempt {self.reconnect_attempts})")
        
        await asyncio.sleep(delay)
        
        try:
            await self.connect()
            
            # Resubscribe to symbols
            if self.subscribed_symbols:
                symbols_list = list(self.subscribed_symbols)
                self.subscribed_symbols.clear()  # Clear to avoid duplicates
                await self.subscribe_symbols(symbols_list)
                
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
            await self._handle_reconnection()
    
    def get_latest_quote(self, symbol: str) -> Optional[StreamingQuote]:
        """Get latest quote for a symbol"""
        return self.latest_quotes.get(symbol)
    
    def get_latest_trades(self, symbol: str, count: int = 10) -> List[StreamingTrade]:
        """Get latest trades for a symbol"""
        trades = self.latest_trades.get(symbol, [])
        return trades[-count:] if trades else []
    
    def get_level2_data(self, symbol: str) -> Optional[Dict]:
        """Get level 2 data for a symbol"""
        return self.level2_data.get(symbol)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        uptime = time.time() - self.connection_start_time if self.connection_start_time else 0
        
        return {
            'is_connected': self.is_connected,
            'subscribed_symbols': list(self.subscribed_symbols),
            'message_count': self.message_count,
            'uptime_seconds': uptime,
            'last_message_time': self.last_message_time,
            'reconnect_attempts': self.reconnect_attempts
        }
    
    async def disconnect(self):
        """Disconnect from streaming service"""
        try:
            if self.is_connected:
                await self.stream_client.disconnect()
                self.is_connected = False
                logger.info("Disconnected from streaming service")
        except Exception as e:
            logger.error(f"Error disconnecting: {e}")

# Import required classes
from trading_bot_commentary_updated import TradingCommentary, CommentaryType 