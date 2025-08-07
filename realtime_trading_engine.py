#!/usr/bin/env python3
"""
Real-Time Trading Engine with Streaming Data
Processes streaming data and executes trades in real-time
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from schwab_streaming_client import EnhancedSchwabStreamClient, StreamingQuote, StreamingTrade
from trading_bot_commentary_updated import (
    TradingCommentary, CommentaryType, TradingSignal, Position, 
    TradingEngineWithCommentary, TradingMode
)

logger = logging.getLogger(__name__)

@dataclass
class RealTimeSignal:
    """Real-time trading signal with streaming data"""
    symbol: str
    signal_type: str  # 'buy', 'sell', 'hold'
    strength: float
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: int
    reasoning: Dict[str, Any]
    timestamp: datetime
    trigger_quote: StreamingQuote
    market_conditions: Dict[str, Any]

class RealTimeTradingEngine:
    """Real-time trading engine using streaming data"""
    
    def __init__(self, schwab_client, commentary_system, mode: TradingMode = TradingMode.SIMULATION_WITH_COMMENTARY):
        self.schwab_client = schwab_client
        self.commentary_system = commentary_system
        self.mode = mode
        
        # Streaming client
        self.stream_client = EnhancedSchwabStreamClient(schwab_client, commentary_system)
        
        # Trading state
        self.positions: Dict[str, Position] = {}
        self.pending_signals: List[RealTimeSignal] = []
        self.signal_history: List[RealTimeSignal] = []
        
        # Performance tracking
        self.total_pnl = 0.0
        self.daily_pnl = 0.0
        self.trade_count = 0
        self.win_count = 0
        
        # Real-time analysis
        self.price_alerts: Dict[str, Dict] = {}
        self.volatility_tracker: Dict[str, List[float]] = {}
        self.volume_spikes: Dict[str, List[Dict]] = {}
        
        # Signal processing
        self.signal_processors: List[Callable] = []
        self.execution_callbacks: List[Callable] = []
        
        # Market conditions
        self.current_market_regime = "normal"
        self.market_volatility = 0.0
        self.overall_sentiment = 0.0
        
        # Risk management
        self.max_positions = 5
        self.max_daily_loss = -1000.0
        self.position_size_limit = 0.1  # 10% of account per position
        
        # Start time
        self.start_time = datetime.now()
        
    async def start(self):
        """Start the real-time trading engine"""
        try:
            logger.info("Starting Real-Time Trading Engine...")
            
            # Connect to streaming service
            await self.stream_client.connect()
            
            # Set up callbacks
            self._setup_streaming_callbacks()
            
            # Add commentary
            self.commentary_system.add_commentary(
                TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.DECISION,
                    symbol=None,
                    title="Real-Time Trading Engine Started",
                    message="Engine is now processing streaming data and ready for real-time trading decisions.",
                    importance=5
                )
            )
            
            logger.info("Real-Time Trading Engine started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start Real-Time Trading Engine: {e}")
            raise
    
    def _setup_streaming_callbacks(self):
        """Set up callbacks for streaming data"""
        # Quote callback for all symbols
        self.stream_client.add_quote_callback("*", self._handle_quote_update)
        
        # Trade callback for all symbols
        self.stream_client.add_trade_callback("*", self._handle_trade_update)
        
        # Level 2 callback for all symbols
        self.stream_client.add_level2_callback("*", self._handle_level2_update)
    
    async def subscribe_to_symbols(self, symbols: List[str]):
        """Subscribe to real-time data for trading symbols"""
        try:
            await self.stream_client.subscribe_symbols(symbols)
            
            # Initialize tracking for each symbol
            for symbol in symbols:
                self.volatility_tracker[symbol] = []
                self.volume_spikes[symbol] = []
                self.price_alerts[symbol] = {}
            
            logger.info(f"Subscribed to {len(symbols)} symbols for real-time trading")
            
        except Exception as e:
            logger.error(f"Failed to subscribe to symbols: {e}")
            raise
    
    async def _handle_quote_update(self, quote: StreamingQuote):
        """Handle real-time quote updates"""
        try:
            symbol = quote.symbol
            current_price = quote.last
            
            # Update volatility tracking
            self._update_volatility_tracking(symbol, current_price)
            
            # Check for price alerts
            await self._check_price_alerts(symbol, quote)
            
            # Process real-time signals
            await self._process_real_time_signals(symbol, quote)
            
            # Update existing positions
            await self._update_positions(symbol, quote)
            
            # Check for exit conditions
            await self._check_exit_conditions(symbol, quote)
            
        except Exception as e:
            logger.error(f"Error handling quote update for {quote.symbol}: {e}")
    
    async def _handle_trade_update(self, trade: StreamingTrade):
        """Handle real-time trade updates"""
        try:
            symbol = trade.symbol
            
            # Update volume spike tracking
            self._update_volume_tracking(symbol, trade)
            
            # Check for unusual activity
            await self._check_unusual_activity(symbol, trade)
            
            # Process trade-based signals
            await self._process_trade_signals(symbol, trade)
            
        except Exception as e:
            logger.error(f"Error handling trade update for {trade.symbol}: {e}")
    
    async def _handle_level2_update(self, level2_data: Dict):
        """Handle level 2 market depth updates"""
        try:
            # Analyze order flow
            order_flow_analysis = self._analyze_order_flow(level2_data)
            
            # Check for large orders
            large_orders = self._detect_large_orders(level2_data)
            
            # Process level 2 signals
            if order_flow_analysis or large_orders:
                await self._process_level2_signals(level2_data, order_flow_analysis, large_orders)
                
        except Exception as e:
            logger.error(f"Error handling level 2 update: {e}")
    
    def _update_volatility_tracking(self, symbol: str, price: float):
        """Update volatility tracking for a symbol"""
        if symbol not in self.volatility_tracker:
            self.volatility_tracker[symbol] = []
        
        self.volatility_tracker[symbol].append(price)
        
        # Keep last 100 prices for volatility calculation
        if len(self.volatility_tracker[symbol]) > 100:
            self.volatility_tracker[symbol] = self.volatility_tracker[symbol][-100:]
        
        # Calculate current volatility
        if len(self.volatility_tracker[symbol]) > 10:
            prices = np.array(self.volatility_tracker[symbol])
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns) * np.sqrt(252)  # Annualized
            self.volatility_tracker[symbol] = volatility
    
    def _update_volume_tracking(self, symbol: str, trade: StreamingTrade):
        """Update volume tracking for a symbol"""
        if symbol not in self.volume_spikes:
            self.volume_spikes[symbol] = []
        
        # Add trade to volume tracking
        self.volume_spikes[symbol].append({
            'timestamp': trade.timestamp,
            'price': trade.price,
            'size': trade.size,
            'side': trade.side
        })
        
        # Keep last 50 trades
        if len(self.volume_spikes[symbol]) > 50:
            self.volume_spikes[symbol] = self.volume_spikes[symbol][-50:]
    
    async def _check_price_alerts(self, symbol: str, quote: StreamingQuote):
        """Check for price alerts and triggers"""
        if symbol not in self.price_alerts:
            return
        
        current_price = quote.last
        
        for alert_id, alert in self.price_alerts[symbol].items():
            if alert['triggered']:
                continue
            
            trigger_price = alert['price']
            trigger_type = alert['type']  # 'above' or 'below'
            
            if trigger_type == 'above' and current_price >= trigger_price:
                await self._trigger_price_alert(symbol, alert, quote)
            elif trigger_type == 'below' and current_price <= trigger_price:
                await self._trigger_price_alert(symbol, alert, quote)
    
    async def _trigger_price_alert(self, symbol: str, alert: Dict, quote: StreamingQuote):
        """Trigger a price alert"""
        alert['triggered'] = True
        
        # Add commentary
        self.commentary_system.add_commentary(
            TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.WARNING,
                symbol=symbol,
                title=f"Price Alert: {alert['type'].title()} {alert['price']}",
                message=f"{symbol} has reached {alert['price']} (current: {quote.last}). {alert.get('message', '')}",
                importance=6
            )
        )
        
        # Execute alert action if specified
        if 'action' in alert:
            await self._execute_alert_action(symbol, alert['action'], quote)
    
    async def _process_real_time_signals(self, symbol: str, quote: StreamingQuote):
        """Process real-time trading signals"""
        try:
            # Get current market conditions
            market_conditions = self._get_market_conditions(symbol)
            
            # Generate real-time signal
            signal = await self._generate_real_time_signal(symbol, quote, market_conditions)
            
            if signal and signal.confidence > 0.7:  # High confidence threshold
                # Add to pending signals
                self.pending_signals.append(signal)
                
                # Add commentary
                self.commentary_system.add_commentary(
                    TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.SIGNAL_GENERATION,
                        symbol=symbol,
                        title=f"Real-Time {signal.signal_type.upper()} Signal",
                        message=f"Generated {signal.signal_type} signal for {symbol} at {signal.entry_price}. "
                               f"Confidence: {signal.confidence:.2%}. Strength: {signal.strength:.2f}",
                        data=signal.reasoning,
                        confidence=signal.confidence,
                        importance=8
                    )
                )
                
                # Execute signal if auto-execution is enabled
                if self.mode == TradingMode.LIVE:
                    await self._execute_signal(signal)
                
        except Exception as e:
            logger.error(f"Error processing real-time signals for {symbol}: {e}")
    
    async def _generate_real_time_signal(self, symbol: str, quote: StreamingQuote, market_conditions: Dict) -> Optional[RealTimeSignal]:
        """Generate real-time trading signal"""
        try:
            # Get recent price data
            recent_prices = self.volatility_tracker.get(symbol, [])
            if len(recent_prices) < 20:
                return None
            
            # Calculate technical indicators
            indicators = self._calculate_real_time_indicators(recent_prices)
            
            # Analyze volume patterns
            volume_analysis = self._analyze_volume_patterns(symbol)
            
            # Generate signal based on indicators and conditions
            signal_type, strength, confidence = self._determine_signal_type(indicators, volume_analysis, market_conditions)
            
            if signal_type == 'hold':
                return None
            
            # Calculate position parameters
            entry_price = quote.last
            stop_loss, take_profit = self._calculate_risk_levels(signal_type, entry_price, indicators)
            position_size = self._calculate_position_size(symbol, entry_price, confidence)
            
            # Create reasoning
            reasoning = {
                'indicators': indicators,
                'volume_analysis': volume_analysis,
                'market_conditions': market_conditions,
                'signal_strength': strength,
                'confidence_factors': self._get_confidence_factors(indicators, volume_analysis)
            }
            
            return RealTimeSignal(
                symbol=symbol,
                signal_type=signal_type,
                strength=strength,
                confidence=confidence,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                reasoning=reasoning,
                timestamp=datetime.now(),
                trigger_quote=quote,
                market_conditions=market_conditions
            )
            
        except Exception as e:
            logger.error(f"Error generating real-time signal for {symbol}: {e}")
            return None
    
    def _calculate_real_time_indicators(self, prices: List[float]) -> Dict[str, float]:
        """Calculate technical indicators from recent prices"""
        if len(prices) < 20:
            return {}
        
        prices_array = np.array(prices)
        
        # RSI
        rsi = self._calculate_rsi(prices_array, 14)
        
        # Moving averages
        sma_20 = np.mean(prices_array[-20:])
        sma_50 = np.mean(prices_array[-50:]) if len(prices) >= 50 else sma_20
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(prices_array, 20)
        
        # MACD
        macd_line, signal_line, histogram = self._calculate_macd(prices_array)
        
        # Volatility
        volatility = np.std(prices_array[-20:]) / np.mean(prices_array[-20:])
        
        return {
            'rsi': rsi,
            'sma_20': sma_20,
            'sma_50': sma_50,
            'bb_upper': bb_upper,
            'bb_middle': bb_middle,
            'bb_lower': bb_lower,
            'macd_line': macd_line,
            'signal_line': signal_line,
            'macd_histogram': histogram,
            'volatility': volatility,
            'current_price': prices_array[-1]
        }
    
    def _analyze_volume_patterns(self, symbol: str) -> Dict[str, Any]:
        """Analyze volume patterns for a symbol"""
        if symbol not in self.volume_spikes:
            return {}
        
        trades = self.volume_spikes[symbol]
        if len(trades) < 10:
            return {}
        
        # Calculate average volume
        volumes = [trade['size'] for trade in trades]
        avg_volume = np.mean(volumes)
        
        # Check for volume spikes
        recent_volumes = volumes[-5:]
        volume_spike = any(v > avg_volume * 2 for v in recent_volumes)
        
        # Analyze buy/sell pressure
        buy_volume = sum(trade['size'] for trade in trades if trade['side'] == 'buy')
        sell_volume = sum(trade['size'] for trade in trades if trade['side'] == 'sell')
        buy_pressure = buy_volume / (buy_volume + sell_volume) if (buy_volume + sell_volume) > 0 else 0.5
        
        return {
            'avg_volume': avg_volume,
            'volume_spike': volume_spike,
            'buy_pressure': buy_pressure,
            'recent_volumes': recent_volumes
        }
    
    def _determine_signal_type(self, indicators: Dict, volume_analysis: Dict, market_conditions: Dict) -> tuple:
        """Determine signal type, strength, and confidence"""
        signal_type = 'hold'
        strength = 0.0
        confidence = 0.0
        
        # Get indicator values
        rsi = indicators.get('rsi', 50)
        current_price = indicators.get('current_price', 0)
        sma_20 = indicators.get('sma_20', current_price)
        bb_upper = indicators.get('bb_upper', current_price)
        bb_lower = indicators.get('bb_lower', current_price)
        macd_line = indicators.get('macd_line', 0)
        signal_line = indicators.get('signal_line', 0)
        
        # Buy signals
        buy_signals = 0
        total_signals = 0
        
        # RSI oversold
        if rsi < 30:
            buy_signals += 1
        total_signals += 1
        
        # Price above SMA
        if current_price > sma_20:
            buy_signals += 1
        total_signals += 1
        
        # MACD bullish crossover
        if macd_line > signal_line:
            buy_signals += 1
        total_signals += 1
        
        # Volume spike with buy pressure
        if volume_analysis.get('volume_spike', False) and volume_analysis.get('buy_pressure', 0.5) > 0.6:
            buy_signals += 1
        total_signals += 1
        
        # Sell signals
        sell_signals = 0
        
        # RSI overbought
        if rsi > 70:
            sell_signals += 1
        
        # Price below SMA
        if current_price < sma_20:
            sell_signals += 1
        
        # MACD bearish crossover
        if macd_line < signal_line:
            sell_signals += 1
        
        # Volume spike with sell pressure
        if volume_analysis.get('volume_spike', False) and volume_analysis.get('buy_pressure', 0.5) < 0.4:
            sell_signals += 1
        
        # Determine signal
        if buy_signals >= 3 and buy_signals > sell_signals:
            signal_type = 'buy'
            strength = buy_signals / total_signals
            confidence = min(0.9, strength + 0.3)
        elif sell_signals >= 3 and sell_signals > buy_signals:
            signal_type = 'sell'
            strength = sell_signals / total_signals
            confidence = min(0.9, strength + 0.3)
        
        return signal_type, strength, confidence
    
    def _calculate_risk_levels(self, signal_type: str, entry_price: float, indicators: Dict) -> tuple:
        """Calculate stop loss and take profit levels"""
        volatility = indicators.get('volatility', 0.02)
        
        if signal_type == 'buy':
            stop_loss = entry_price * (1 - volatility * 2)  # 2x volatility
            take_profit = entry_price * (1 + volatility * 3)  # 3x volatility
        else:  # sell
            stop_loss = entry_price * (1 + volatility * 2)
            take_profit = entry_price * (1 - volatility * 3)
        
        return stop_loss, take_profit
    
    def _calculate_position_size(self, symbol: str, entry_price: float, confidence: float) -> int:
        """Calculate position size based on confidence and risk"""
        # Base position size
        base_size = 100
        
        # Adjust for confidence
        confidence_multiplier = confidence * 2  # 0-2x multiplier
        
        # Adjust for current positions
        current_positions = len(self.positions)
        position_multiplier = max(0.1, 1 - (current_positions / self.max_positions))
        
        # Calculate final size
        final_size = int(base_size * confidence_multiplier * position_multiplier)
        
        return max(10, min(final_size, 1000))  # Min 10, max 1000 shares
    
    def _get_market_conditions(self, symbol: str) -> Dict[str, Any]:
        """Get current market conditions"""
        return {
            'regime': self.current_market_regime,
            'volatility': self.market_volatility,
            'sentiment': self.overall_sentiment,
            'time_of_day': datetime.now().hour,
            'day_of_week': datetime.now().weekday()
        }
    
    async def _execute_signal(self, signal: RealTimeSignal):
        """Execute a trading signal"""
        try:
            # Check if we can take the position
            if not self._can_take_position(signal.symbol, signal.signal_type):
                return
            
            # Create position
            position = Position(
                symbol=signal.symbol,
                entry_price=signal.entry_price,
                current_price=signal.entry_price,
                quantity=signal.position_size,
                side='long' if signal.signal_type == 'buy' else 'short',
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                entry_time=datetime.now(),
                reasoning=signal.reasoning
            )
            
            # Add position
            self.positions[signal.symbol] = position
            
            # Add to signal history
            self.signal_history.append(signal)
            
            # Add commentary
            self.commentary_system.add_commentary(
                TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.DECISION,
                    symbol=signal.symbol,
                    title=f"Executed {signal.signal_type.upper()} Signal",
                    message=f"Executed {signal.signal_type} order for {signal.symbol}: "
                           f"{signal.position_size} shares at ${signal.entry_price:.2f}. "
                           f"Stop: ${signal.stop_loss:.2f}, Target: ${signal.take_profit:.2f}",
                    data=signal.reasoning,
                    confidence=signal.confidence,
                    importance=9
                )
            )
            
            logger.info(f"Executed {signal.signal_type} signal for {signal.symbol}")
            
        except Exception as e:
            logger.error(f"Error executing signal for {signal.symbol}: {e}")
    
    def _can_take_position(self, symbol: str, signal_type: str) -> bool:
        """Check if we can take a new position"""
        # Check if already have position in this symbol
        if symbol in self.positions:
            return False
        
        # Check position limit
        if len(self.positions) >= self.max_positions:
            return False
        
        # Check daily loss limit
        if self.daily_pnl <= self.max_daily_loss:
            return False
        
        return True
    
    async def _update_positions(self, symbol: str, quote: StreamingQuote):
        """Update existing positions with new quote data"""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        position.current_price = quote.last
        
        # Calculate unrealized P&L
        if position.side == 'long':
            position.unrealized_pnl = (quote.last - position.entry_price) * position.quantity
        else:  # short
            position.unrealized_pnl = (position.entry_price - quote.last) * position.quantity
    
    async def _check_exit_conditions(self, symbol: str, quote: StreamingQuote):
        """Check for exit conditions on positions"""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        current_price = quote.last
        
        # Check stop loss
        if position.side == 'long' and current_price <= position.stop_loss:
            await self._close_position(symbol, "Stop Loss Hit", quote)
        elif position.side == 'short' and current_price >= position.stop_loss:
            await self._close_position(symbol, "Stop Loss Hit", quote)
        
        # Check take profit
        elif position.side == 'long' and current_price >= position.take_profit:
            await self._close_position(symbol, "Take Profit Hit", quote)
        elif position.side == 'short' and current_price <= position.take_profit:
            await self._close_position(symbol, "Take Profit Hit", quote)
    
    async def _close_position(self, symbol: str, reason: str, quote: StreamingQuote):
        """Close a position"""
        try:
            position = self.positions[symbol]
            
            # Calculate P&L
            if position.side == 'long':
                pnl = (quote.last - position.entry_price) * position.quantity
            else:  # short
                pnl = (position.entry_price - quote.last) * position.quantity
            
            # Update tracking
            self.total_pnl += pnl
            self.daily_pnl += pnl
            self.trade_count += 1
            if pnl > 0:
                self.win_count += 1
            
            # Remove position
            del self.positions[symbol]
            
            # Add commentary
            self.commentary_system.add_commentary(
                TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.DECISION,
                    symbol=symbol,
                    title=f"Position Closed: {reason}",
                    message=f"Closed {position.side} position in {symbol}. "
                           f"Entry: ${position.entry_price:.2f}, Exit: ${quote.last:.2f}, "
                           f"P&L: ${pnl:.2f}",
                    data={'pnl': pnl, 'reason': reason},
                    importance=8
                )
            )
            
            logger.info(f"Closed position in {symbol}: {reason}, P&L: ${pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Error closing position in {symbol}: {e}")
    
    # Technical indicator calculations
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_bollinger_bands(self, prices: np.ndarray, period: int = 20) -> tuple:
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            return prices[-1], prices[-1], prices[-1]
        
        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        
        upper = sma + (std * 2)
        lower = sma - (std * 2)
        
        return upper, sma, lower
    
    def _calculate_macd(self, prices: np.ndarray) -> tuple:
        """Calculate MACD"""
        if len(prices) < 26:
            return 0, 0, 0
        
        ema_12 = np.mean(prices[-12:])  # Simplified EMA
        ema_26 = np.mean(prices[-26:])
        
        macd_line = ema_12 - ema_26
        signal_line = macd_line  # Simplified signal line
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        win_rate = self.win_count / self.trade_count if self.trade_count > 0 else 0
        
        return {
            'total_pnl': self.total_pnl,
            'daily_pnl': self.daily_pnl,
            'trade_count': self.trade_count,
            'win_count': self.win_count,
            'win_rate': win_rate,
            'current_positions': len(self.positions),
            'uptime': (datetime.now() - self.start_time).total_seconds(),
            'streaming_stats': self.stream_client.get_connection_stats()
        }
    
    async def stop(self):
        """Stop the real-time trading engine"""
        try:
            # Close all positions
            for symbol in list(self.positions.keys()):
                quote = self.stream_client.get_latest_quote(symbol)
                if quote:
                    await self._close_position(symbol, "Engine Shutdown", quote)
            
            # Disconnect from streaming
            await self.stream_client.disconnect()
            
            logger.info("Real-Time Trading Engine stopped")
            
        except Exception as e:
            logger.error(f"Error stopping Real-Time Trading Engine: {e}") 