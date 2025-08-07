#!/usr/bin/env python3
"""
Real-Time Integration Module
Integrates streaming data with the main trading bot system
"""

import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime

from schwab_streaming_client import EnhancedSchwabStreamClient
from realtime_trading_engine import RealTimeTradingEngine
from trading_bot_commentary_updated import (
    TradingEngineWithCommentary, CommentarySystem, TradingMode,
    TradingCommentary, CommentaryType
)

logger = logging.getLogger(__name__)

class RealTimeTradingBot:
    """Main real-time trading bot that integrates streaming with the existing system"""
    
    def __init__(self, schwab_client, mode: TradingMode = TradingMode.SIMULATION_WITH_COMMENTARY):
        self.schwab_client = schwab_client
        self.mode = mode
        
        # Initialize components
        self.commentary_system = CommentarySystem()
        self.realtime_engine = RealTimeTradingEngine(schwab_client, self.commentary_system, mode)
        
        # Trading symbols to monitor
        self.trading_symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX',
            'AMD', 'INTC', 'CRM', 'ADBE', 'PYPL', 'UBER', 'LYFT', 'ZM'
        ]
        
        # Performance tracking
        self.start_time = datetime.now()
        self.is_running = False
        
        # WebSocket connection for real-time updates
        self.websocket_connections = []
        
    async def start(self):
        """Start the real-time trading bot"""
        try:
            logger.info("Starting Real-Time Trading Bot...")
            
            # Start the real-time engine
            await self.realtime_engine.start()
            
            # Subscribe to trading symbols
            await self.realtime_engine.subscribe_to_symbols(self.trading_symbols)
            
            # Set up real-time commentary broadcasting
            await self._setup_commentary_broadcasting()
            
            self.is_running = True
            
            # Add startup commentary
            self.commentary_system.add_commentary(
                TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.DECISION,
                    symbol=None,
                    title="Real-Time Trading Bot Started",
                    message=f"Bot is now actively trading {len(self.trading_symbols)} symbols using real-time streaming data. "
                           f"Mode: {self.mode.value}",
                    importance=5
                )
            )
            
            logger.info("Real-Time Trading Bot started successfully")
            
            # Start the main trading loop
            await self._trading_loop()
            
        except Exception as e:
            logger.error(f"Failed to start Real-Time Trading Bot: {e}")
            raise
    
    async def _trading_loop(self):
        """Main trading loop that runs continuously"""
        try:
            while self.is_running:
                # Process any pending signals
                await self._process_pending_signals()
                
                # Update market conditions
                await self._update_market_conditions()
                
                # Check for system health
                await self._check_system_health()
                
                # Sleep briefly to prevent excessive CPU usage
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Error in trading loop: {e}")
            await self.stop()
    
    async def _process_pending_signals(self):
        """Process any pending trading signals"""
        try:
            pending_signals = self.realtime_engine.pending_signals.copy()
            self.realtime_engine.pending_signals.clear()
            
            for signal in pending_signals:
                # Add detailed commentary for each signal
                await self._add_signal_commentary(signal)
                
                # Execute signal if in live mode
                if self.mode == TradingMode.LIVE:
                    await self._execute_live_signal(signal)
                
        except Exception as e:
            logger.error(f"Error processing pending signals: {e}")
    
    async def _add_signal_commentary(self, signal):
        """Add detailed commentary for a trading signal"""
        try:
            # Create detailed reasoning
            reasoning = signal.reasoning
            
            # Add market context
            market_context = f"Market Regime: {signal.market_conditions.get('regime', 'unknown')}, "
            market_context += f"Volatility: {signal.market_conditions.get('volatility', 0):.2%}, "
            market_context += f"Sentiment: {signal.market_conditions.get('sentiment', 0):.2f}"
            
            # Add technical analysis
            indicators = reasoning.get('indicators', {})
            technical_analysis = f"RSI: {indicators.get('rsi', 0):.1f}, "
            technical_analysis += f"SMA20: ${indicators.get('sma_20', 0):.2f}, "
            technical_analysis += f"Volatility: {indicators.get('volatility', 0):.2%}"
            
            # Add volume analysis
            volume_analysis = reasoning.get('volume_analysis', {})
            volume_info = f"Avg Volume: {volume_analysis.get('avg_volume', 0):,.0f}, "
            volume_info += f"Buy Pressure: {volume_analysis.get('buy_pressure', 0.5):.1%}"
            
            # Create comprehensive message
            message = f"Real-time {signal.signal_type.upper()} signal for {signal.symbol} at ${signal.entry_price:.2f}. "
            message += f"Confidence: {signal.confidence:.1%}, Strength: {signal.strength:.2f}. "
            message += f"Stop: ${signal.stop_loss:.2f}, Target: ${signal.take_profit:.2f}. "
            message += f"Position Size: {signal.position_size} shares. "
            message += f"{market_context}. {technical_analysis}. {volume_info}."
            
            self.commentary_system.add_commentary(
                TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.SIGNAL_GENERATION,
                    symbol=signal.symbol,
                    title=f"Real-Time {signal.signal_type.upper()} Signal - {signal.symbol}",
                    message=message,
                    data=reasoning,
                    confidence=signal.confidence,
                    importance=8
                )
            )
            
        except Exception as e:
            logger.error(f"Error adding signal commentary: {e}")
    
    async def _execute_live_signal(self, signal):
        """Execute a signal in live trading mode"""
        try:
            # Add execution commentary
            self.commentary_system.add_commentary(
                TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.DECISION,
                    symbol=signal.symbol,
                    title=f"Executing Live {signal.signal_type.upper()} Order",
                    message=f"Executing live {signal.signal_type} order for {signal.symbol}: "
                           f"{signal.position_size} shares at ${signal.entry_price:.2f}",
                    importance=9
                )
            )
            
            # Execute the signal through the real-time engine
            await self.realtime_engine._execute_signal(signal)
            
        except Exception as e:
            logger.error(f"Error executing live signal: {e}")
            
            # Add error commentary
            self.commentary_system.add_commentary(
                TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.WARNING,
                    symbol=signal.symbol,
                    title="Signal Execution Failed",
                    message=f"Failed to execute {signal.signal_type} signal for {signal.symbol}: {str(e)}",
                    importance=7
                )
            )
    
    async def _update_market_conditions(self):
        """Update market conditions based on streaming data"""
        try:
            # Get streaming stats
            stream_stats = self.realtime_engine.stream_client.get_connection_stats()
            
            # Update market regime based on overall performance
            performance_stats = self.realtime_engine.get_performance_stats()
            
            # Determine market regime
            if performance_stats['win_rate'] > 0.7:
                regime = "bullish"
            elif performance_stats['win_rate'] < 0.3:
                regime = "bearish"
            else:
                regime = "neutral"
            
            self.realtime_engine.current_market_regime = regime
            
            # Update market volatility
            total_volatility = 0
            count = 0
            for symbol in self.trading_symbols:
                if symbol in self.realtime_engine.volatility_tracker:
                    vol = self.realtime_engine.volatility_tracker[symbol]
                    if isinstance(vol, float):
                        total_volatility += vol
                        count += 1
            
            if count > 0:
                self.realtime_engine.market_volatility = total_volatility / count
            
        except Exception as e:
            logger.error(f"Error updating market conditions: {e}")
    
    async def _check_system_health(self):
        """Check system health and add status commentary"""
        try:
            # Get performance stats
            stats = self.realtime_engine.get_performance_stats()
            stream_stats = stats['streaming_stats']
            
            # Check if streaming is healthy
            if not stream_stats['is_connected']:
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
            
            # Add periodic status update (every 5 minutes)
            uptime = stats['uptime']
            if int(uptime) % 300 == 0:  # Every 5 minutes
                self.commentary_system.add_commentary(
                    TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.TECHNICAL,
                        symbol=None,
                        title="System Status Update",
                        message=f"Bot running for {int(uptime/60)} minutes. "
                               f"Total P&L: ${stats['total_pnl']:.2f}, "
                               f"Win Rate: {stats['win_rate']:.1%}, "
                               f"Active Positions: {stats['current_positions']}, "
                               f"Streaming Messages: {stream_stats['message_count']}",
                        importance=3
                    )
                )
                
        except Exception as e:
            logger.error(f"Error checking system health: {e}")
    
    async def _setup_commentary_broadcasting(self):
        """Set up real-time commentary broadcasting"""
        try:
            # Subscribe to commentary updates
            self.commentary_system.subscribe(self._broadcast_commentary)
            
            logger.info("Commentary broadcasting set up successfully")
            
        except Exception as e:
            logger.error(f"Error setting up commentary broadcasting: {e}")
    
    async def _broadcast_commentary(self, commentary):
        """Broadcast commentary to connected clients"""
        try:
            # Convert commentary to dict for broadcasting
            commentary_dict = commentary.to_dict()
            
            # Add additional metadata
            commentary_dict['bot_status'] = {
                'is_running': self.is_running,
                'uptime': (datetime.now() - self.start_time).total_seconds(),
                'mode': self.mode.value
            }
            
            # Broadcast to all connected WebSocket clients
            for websocket in self.websocket_connections:
                try:
                    await websocket.send_json(commentary_dict)
                except Exception as e:
                    logger.warning(f"Failed to send commentary to WebSocket: {e}")
                    # Remove disconnected WebSocket
                    self.websocket_connections.remove(websocket)
                    
        except Exception as e:
            logger.error(f"Error broadcasting commentary: {e}")
    
    async def add_websocket_connection(self, websocket):
        """Add a new WebSocket connection for real-time updates"""
        self.websocket_connections.append(websocket)
        
        # Send initial status
        try:
            stats = self.realtime_engine.get_performance_stats()
            await websocket.send_json({
                'type': 'status',
                'data': {
                    'bot_status': 'running',
                    'uptime': stats['uptime'],
                    'total_pnl': stats['total_pnl'],
                    'win_rate': stats['win_rate'],
                    'active_positions': stats['current_positions'],
                    'subscribed_symbols': self.trading_symbols
                }
            })
        except Exception as e:
            logger.error(f"Error sending initial status: {e}")
    
    async def remove_websocket_connection(self, websocket):
        """Remove a WebSocket connection"""
        if websocket in self.websocket_connections:
            self.websocket_connections.remove(websocket)
    
    def get_status(self) -> Dict:
        """Get current bot status"""
        try:
            stats = self.realtime_engine.get_performance_stats()
            stream_stats = stats['streaming_stats']
            
            return {
                'is_running': self.is_running,
                'mode': self.mode.value,
                'uptime': stats['uptime'],
                'total_pnl': stats['total_pnl'],
                'daily_pnl': stats['daily_pnl'],
                'win_rate': stats['win_rate'],
                'trade_count': stats['trade_count'],
                'active_positions': stats['current_positions'],
                'subscribed_symbols': self.trading_symbols,
                'streaming_connected': stream_stats['is_connected'],
                'streaming_messages': stream_stats['message_count'],
                'streaming_uptime': stream_stats['uptime_seconds']
            }
            
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return {'error': str(e)}
    
    async def stop(self):
        """Stop the real-time trading bot"""
        try:
            logger.info("Stopping Real-Time Trading Bot...")
            
            self.is_running = False
            
            # Stop the real-time engine
            await self.realtime_engine.stop()
            
            # Close all WebSocket connections
            for websocket in self.websocket_connections:
                try:
                    await websocket.close()
                except:
                    pass
            self.websocket_connections.clear()
            
            # Add shutdown commentary
            self.commentary_system.add_commentary(
                TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.DECISION,
                    symbol=None,
                    title="Real-Time Trading Bot Stopped",
                    message="Bot has been stopped. All positions closed and connections terminated.",
                    importance=5
                )
            )
            
            logger.info("Real-Time Trading Bot stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping Real-Time Trading Bot: {e}")

# Example usage
async def main():
    """Example usage of the real-time trading bot"""
    try:
        # Initialize Schwab client (you'll need to set up authentication)
        # schwab_client = schwab_client.Client(...)
        
        # Create and start the bot
        # bot = RealTimeTradingBot(schwab_client, TradingMode.SIMULATION_WITH_COMMENTARY)
        # await bot.start()
        
        print("Real-Time Trading Bot integration ready")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 