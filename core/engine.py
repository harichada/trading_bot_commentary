"""Trading Engine with Commentary - Core orchestrator for the trading bot."""

import asyncio
import json
import logging
import os
import time
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from core.models import (TradingMode, CommentaryType, SignalType, NewsImpact,
                         TradingSignal, Position, MarketData, NewsItem)
from core.config import Config, config, config_manager, logger, TradingLossBreaker
from core.commentary import TradingCommentary, CommentarySystem
from core.brain import TradingBrain
from core.websocket_manager import ConnectionManager
from analysis.anomaly import AnomalyDetector, DataValidator
from analysis.behavioral import BehavioralAnalyzer
from analysis.exit_managers import AdvancedExitManager, DynamicExitManager
from analysis.alternative_data import AlternativeDataIntegrator, MarketNeutralStrategies
from analysis.technical import TechnicalAnalyzerWithCommentary
from analysis.screener import StockScreener
from ml.predictor import MLPredictorWithCommentary
from risk.manager import RiskManagerWithCommentary
from risk.backtest import PerformanceAnalyzer
from strategies.builtin import (BreakoutStrategyWithCommentary,
                                MeanReversionStrategyWithCommentary,
                                MomentumStrategyWithCommentary)
from strategies.news_strategy import FreeNewsSignalStrategy
from data_providers.realtime import RealTimeDataProvider, DummyDataProvider
from data_providers.schwab import SchwabDataProvider

from trading_exceptions import *
from circuit_breaker import api_circuit_breaker, order_circuit_breaker
from error_recovery import ErrorRecoveryManager


def log_signal(*args, **kwargs): pass
def log_decision(*args, **kwargs): pass
def log_trade(*args, **kwargs): pass
def log_system_event(*args, **kwargs): pass
def log_pnl(*args, **kwargs): pass
def log_error(*args, **kwargs): pass
def log_performance(*args, **kwargs): pass

# Schwab SDK imports
try:
    from schwab import auth, client as schwab_client_module
    from schwab.orders.equities import equity_buy_market, equity_sell_market, equity_buy_limit, equity_sell_limit
    from schwab.orders.common import Duration, Session
    SCHWAB_AVAILABLE = True
except ImportError:
    SCHWAB_AVAILABLE = False


class TradingEngineWithCommentary:
    """Trading engine that explains its decisions in real-time"""
    
    def __init__(self, mode: TradingMode = TradingMode.SIMULATION_WITH_COMMENTARY, connection_manager=None):
        self.mode = mode
        self.connection_manager = connection_manager  # Store connection manager for broadcasting
        self.schwab_client = None
        self.data_provider = None
        self.positions = {}
        
        # Trading hours configuration from config file
        config = Config()
        self.allow_premarket = config.manager.get('trading.extended_hours.allow_premarket', False)
        self.allow_afterhours = config.manager.get('trading.extended_hours.allow_afterhours', False)
        self.use_limit_orders_extended_hours = config.manager.get('trading.extended_hours.use_limit_orders', True)
        self.simulated_positions = {}
        self.trade_history = []
        self.is_running = False
        self.account_id = None
        self.account_hash = None
        self._last_bp_check = datetime.now() - timedelta(minutes=5)  # Force initial check
        # Commentary system
        self.commentary = CommentarySystem()
        # Initialize brain BEFORE ml_predictor
        self.brain = TradingBrain()
        # Error recovery manager
        self.error_recovery = ErrorRecoveryManager()
        # Initialize components
        self._init_schwab_client()
        # Order tracking
        self.pending_orders = {}
        self.order_id_to_symbol = {}
        self.last_order_check = datetime.now()
        
        # Risk manager
        self.risk_manager = RiskManagerWithCommentary(
            account_balance=100000,
            commentary_system=self.commentary
        )
        
        # Technical analyzer
        self.technical_analyzer = TechnicalAnalyzerWithCommentary(
            commentary_system=self.commentary
        )
        # ML predictor
        from ml.predictor import MLPredictorWithCommentary
        self.ml_predictor = MLPredictorWithCommentary(
            commentary_system=self.commentary
        )
        self.ml_predictor.set_brain(self.brain)
        # Meta-model (shadow mode): evaluates mean-reversion signals and
        # logs a trust score without affecting trade decisions. Absent
        # model file → feature silently disabled.
        self.meta_inference = None
        try:
            from ml.meta_inference import MetaInference
            meta_path = Path("ml_meta_model.pkl")
            if meta_path.exists():
                self.meta_inference = MetaInference.load(meta_path)
                logger.info(
                    "meta_shadow_enabled primary=%s features=%d",
                    self.meta_inference.primary_strategy,
                    len(self.meta_inference.feature_names),
                )
        except (FileNotFoundError, KeyError, ImportError) as exc:
            # Expected failure modes: missing file, missing bundle keys,
            # missing dependency. Log once and continue — bot still trades.
            logger.warning("meta_shadow_disabled reason=%s", exc)
        except Exception:
            # Anything else (corrupt bundle, import failure of a subclass,
            # etc.) logs with traceback so a silent corruption is loud.
            logger.exception("meta_shadow_disabled_unexpected")
        # Scale-out + trailing stop manager
        from analysis.scale_trail_manager import ScaleTrailManager
        self.scale_trail = ScaleTrailManager()
        # Postgres audit logger — every decision + trade persisted to DB
        self.db_logger = None
        try:
            from data_providers.db_logger import DbLogger
            self.db_logger = DbLogger()
            logger.info("db_logger_enabled")
        except Exception as exc:
            logger.warning("db_logger_disabled reason=%s", exc)
        # Stock screener
        self.screener = StockScreener(
            self.schwab_client,
            self.commentary
        ) if self.schwab_client else None
        
        # Dynamic watchlist
        self.dynamic_watchlist = ['NVDA', 'TSLA', 'PLTR']  # Default symbols
        self.last_screener_run = None
        # Trading strategies
        self.strategies = [
            BreakoutStrategyWithCommentary(self.commentary),
            MeanReversionStrategyWithCommentary(self.commentary),
        ]
        # Momentum disabled 2026-04-15: backtest PF 0.78–0.83 across 1/5/15-min
        # timeframes — consistently losing. Re-enable via ENABLE_MOMENTUM=1.
        if os.environ.get("ENABLE_MOMENTUM", "0") == "1":
            self.strategies.append(MomentumStrategyWithCommentary(self.commentary))
        self.strategies.append(FreeNewsSignalStrategy(self.commentary))
        
	    # Initialize brain and exit manager
        self.brain = TradingBrain()
        self.exit_manager = DynamicExitManager(self.brain, self.commentary)

	    # Load previous state
        self._load_state()

        # Connection manager for WebSocket
        self.connection_manager = connection_manager or ConnectionManager()
        # Confirmation settings
        self.require_confirmations = Config().REQUIRE_CLOSE_CONFIRMATION
        self.confirm_only_losses = Config().CONFIRM_ONLY_LOSSES
        self.confirm_threshold_percent = Config().CONFIRM_THRESHOLD_PERCENT
        self.manual_close_only = True  # NEW: Only close positions manually

        # Market state tracking
        self.market_state = {
            'vix': 0,
            'market_trend': 'neutral',
            'breadth': {},
            'economic_events': []
        }
        
        # Enhanced error recovery and circuit breakers
        self.circuit_breaker = TradingLossBreaker()
        self.error_counts = {}
        self.last_error_time = None
        self.consecutive_errors = 0
        self.max_consecutive_errors = 5
        
        # Performance tracking
        self.performance_analyzer = PerformanceAnalyzer(config_manager.config)
        
        # Advanced components integration
        self.behavioral_analyzer = BehavioralAnalyzer(self.commentary)
        self.data_validator = DataValidator(self.commentary)
        self.advanced_exit_manager = AdvancedExitManager(self.commentary)
        self.alternative_data_integrator = AlternativeDataIntegrator(self.commentary)
        self.market_neutral_strategies = MarketNeutralStrategies(self.commentary)

        self.pro_trading_wrapper = None


        # Add initial commentary
        self.commentary.add_commentary(TradingCommentary(
            timestamp=datetime.now(),
            type=CommentaryType.MARKET_ANALYSIS,
            symbol=None,
            title="🚀 Trading Bot Initialized",
            message=f"Starting in {mode.value} mode with enhanced error recovery and circuit breakers. I'll explain my thinking process as I analyze markets.",
            importance=10
        ))
        self._cached_buying_power = 0
        
        # Track loss alerts to avoid spamming
        self.loss_alert_tracker = {}  # {symbol: {'last_alert_time': datetime, 'last_loss_pct': float}}
        self._last_bp_check = datetime.now()
        self.day_trades_count = 0
        self.performance_metrics = {
            'slippage': [],
            'fill_rate': {'attempts': 0, 'fills': 0}
        }

    @property
    def auto_close_disabled(self) -> bool:
        """True when the bot must not auto-close any positions.

        Simulation always auto-closes on stop-loss / take-profit —
        otherwise the simulated P&L stops reflecting what live
        execution would produce, which defeats the purpose of sim.
        The manual-close-only safety applies only to LIVE mode,
        where it prevents the bot from touching real broker positions.
        """
        if self.mode != TradingMode.LIVE:
            return False
        return bool(self.manual_close_only)

    def _audit(self, component: str, symbol, action: str, reason: str, **details) -> None:
        """Structured audit log for engine-level decisions.

        Format (one line per event, shell-greppable):
          engine_decision component=NAME symbol=SYM action=ACTION reason=R k=v k=v

        Every decision gate that accepts, skips, or modifies a trade must
        emit one of these so trading_bot.log is a complete audit trail.
        Also writes to Postgres bot_decisions for SQL queryability.
        """
        kv = " ".join(f"{k}={v}" for k, v in details.items())
        logger.info(
            "engine_decision component=%s symbol=%s action=%s reason=%s mode=%s %s",
            component, symbol or "-", action, reason, self.mode.value, kv,
        )
        # Async DB write — fire-and-forget so it never blocks the loop
        if hasattr(self, 'db_logger') and self.db_logger is not None:
            try:
                asyncio.get_event_loop().create_task(
                    self.db_logger.log_decision(
                        component=component,
                        symbol=symbol or None,
                        action=action,
                        reason=reason,
                        mode=self.mode.value,
                        signal_type=details.get("signal_type"),
                        confidence=details.get("confidence"),
                        strength=details.get("strength"),
                        meta_proba=details.get("proba"),
                        atr=details.get("atr"),
                        stop_distance=details.get("stop_dist") or details.get("stop_distance"),
                        price=details.get("price"),
                        **{k: v for k, v in details.items()
                           if k not in ("signal_type", "confidence", "strength",
                                        "proba", "atr", "stop_dist", "stop_distance", "price")},
                    )
                )
            except Exception:
                pass  # never break trading loop for DB

    # Add to rest brain - caution state:
    def reset_brain_state(self):
        """Reset brain to neutral confident state"""
        self.brain.emotional_state = {
            'confidence': 0.7,
            'risk_appetite': 0.6,
            'patience': 0.8,
            'fear_level': 0.2,
            'greed_level': 0.3,
            'last_update': datetime.now().isoformat()
        }
        # Clear recent bad memories
        self.brain.memories = self.brain.memories[:-10] if len(self.brain.memories) > 10 else []
        self.brain.consecutive_losses = 0
        self.brain.save_memories()
        
        self.commentary.add_commentary(TradingCommentary(
            timestamp=datetime.now(),
            type=CommentaryType.PSYCHOLOGY,
            symbol=None,
            title="🧠 Brain Reset",
            message="Cleared recent negative memories. Fresh start!",
            importance=8
        ))

    def _load_state(self):
        """Load previous trading state including simulated positions."""
        state_file = Path("trading_state.json")
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    self.trade_history = state.get('trade_history', [])
                    self.risk_manager.consecutive_losses = state.get('consecutive_losses', 0)

                    # Load real position long-term flags
                    positions_data = state.get('positions_data', {})
                    for symbol, pos_data in positions_data.items():
                        if symbol in self.positions:
                            self.positions[symbol].is_long_term = pos_data.get('is_long_term', False)

                    # Restore simulated positions so they survive restarts
                    sim_data = state.get('simulated_positions', {})
                    for symbol, pd in sim_data.items():
                        if symbol in self.simulated_positions:
                            continue  # don't overwrite if already loaded
                        try:
                            entry_time = datetime.fromisoformat(pd['entry_time'])
                        except (KeyError, ValueError):
                            entry_time = datetime.now()
                        pos = Position(
                            symbol=pd.get('symbol', symbol),
                            entry_price=pd['entry_price'],
                            current_price=pd.get('current_price', pd['entry_price']),
                            quantity=pd['quantity'],
                            side=pd.get('side', 'long'),
                            stop_loss=pd['stop_loss'],
                            take_profit=pd['take_profit'],
                            entry_time=entry_time,
                            unrealized_pnl=pd.get('unrealized_pnl', 0),
                            reasoning=pd.get('reasoning', {}),
                            is_long_term=pd.get('is_long_term', False),
                            scaled_out=pd.get('scaled_out', False),
                            original_stop=pd.get('original_stop'),
                            trailing_stop=pd.get('trailing_stop'),
                        )
                        self.simulated_positions[symbol] = pos
                    if sim_data:
                        logger.info(
                            "restored_sim_positions count=%d symbols=%s",
                            len(sim_data), list(sim_data.keys()),
                        )

                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.PSYCHOLOGY,
                        symbol=None,
                        title="☕ Good Morning!",
                        message=f"Back at the desk. Yesterday's P&L: ${state.get('daily_pnl', 0):.2f}. "
                               f"Let's see what the market has for us today.",
                        importance=8
                    ))
            except Exception as e:
                logger.error(f"Error loading state: {e}")
    
    @order_circuit_breaker
    async def _execute_real_trade(self, signal) -> bool:
        """Execute real trade through Schwab with proper error handling"""
        
        if not self.schwab_client or not self.account_id:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.WARNING,
                symbol=signal.symbol,
                title=f"❌ Trade Execution Failed",
                message="Schwab client not properly initialized",
                importance=10
            ))
            raise AuthenticationException("Schwab client not initialized - check credentials")
        
        try:
            # Validate OCO prices first
            # Check if this is a short position (SELL signal without existing position)
            is_short = signal.signal_type == SignalType.SELL and signal.symbol not in self.positions
            if not self._validate_oco_prices(signal.symbol, signal.stop_loss, signal.take_profit, is_short):
                return False
            
            # Check buying power
            quote = self.data_provider.get_quote(signal.symbol)
            if not quote:
                raise StaleDataException(60, 60)
                
            price = quote.get('ask', signal.entry_price) if signal.signal_type == SignalType.BUY else quote.get('bid', signal.entry_price)
            required = signal.position_size * price
            
            # Check available funds
            if self.risk_manager.buying_power < required:
                raise InsufficientFundsException(required, self.risk_manager.buying_power)
                                    
            from schwab.orders.equities import (equity_buy_market, equity_sell_market, 
                                                equity_sell_short_market, equity_buy_limit, 
                                                equity_sell_limit, equity_sell_short_limit)
            from schwab.orders.common import Duration, Session
            
            # Check if we should use limit orders
            use_limit, reason = self.should_use_limit_order()
            is_regular, session = self.is_market_hours()
            
            # Block trading if market is closed and extended hours not allowed
            if session == 'closed':
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.WARNING,
                    symbol=signal.symbol,
                    title=f"🚫 Market Closed",
                    message=f"Cannot place order - market is closed. Will retry during market hours.",
                    importance=8
                ))
                return None
            elif session == 'premarket' and not self.allow_premarket:
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.WARNING,
                    symbol=signal.symbol,
                    title=f"🚫 Premarket Trading Disabled",
                    message=f"Order blocked - premarket trading is disabled for safety. Enable in settings if needed.",
                    importance=8
                ))
                return None
            elif session == 'afterhours' and not self.allow_afterhours:
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.WARNING,
                    symbol=signal.symbol,
                    title=f"🚫 After-Hours Trading Disabled",
                    message=f"Order blocked - after-hours trading is disabled for safety. Enable in settings if needed.",
                    importance=8
                ))
                return None
            
            # Get current price for limit orders
            current_price = signal.entry_price
            
            # Build order based on type and market hours
            if use_limit or self.use_limit_orders_extended_hours:
                # Use limit orders during extended hours
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.INFO,
                    symbol=signal.symbol,
                    title=f"⚠️ Using Limit Order",
                    message=f"{reason}. Placing limit order at ${current_price:.2f}",
                    importance=7
                ))
                
                slippage = Config().LIMIT_ORDER_SLIPPAGE
                if signal.signal_type == SignalType.BUY:
                    limit_price = current_price * (1 + slippage)
                    order_builder = equity_buy_limit(signal.symbol, signal.position_size, limit_price)
                else:  # SELL/SHORT
                    limit_price = current_price * (1 - slippage)
                    try:
                        order_builder = equity_sell_short_limit(signal.symbol, signal.position_size, limit_price)
                    except AttributeError:
                        order_builder = equity_sell_limit(signal.symbol, signal.position_size, limit_price)
                        order_builder.set_instruction('SELL_SHORT')
            else:
                # Use market orders during regular hours
                if signal.signal_type == SignalType.BUY:
                    order_builder = equity_buy_market(signal.symbol, signal.position_size)
                else:  # SHORT
                    try:
                        order_builder = equity_sell_short_market(signal.symbol, signal.position_size)
                    except AttributeError:
                        order_builder = equity_sell_market(signal.symbol, signal.position_size)
                        order_builder.set_instruction('SELL_SHORT')
            
            # Set order parameters
            order_builder.set_duration(Duration.DAY)
            
            # Set session based on market hours
            if session == 'premarket' or session == 'afterhours':
                order_builder.set_session(Session.EXTENDED)
            else:
                order_builder.set_session(Session.NORMAL)
            
            # Build the order
            order = order_builder.build()
            
            # Place the order
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.DECISION,
                symbol=signal.symbol,
                title=f"📤 Placing Live Order",
                message=f"Sending {'BUY' if signal.signal_type == SignalType.BUY else 'SELL'} order for {signal.position_size} shares",
                importance=9
            ))
            
            response = self.schwab_client.place_order(self.account_hash, order)

            if response.status_code in [200, 201]:
                # Extract order ID from Location header
                order_id = response.headers.get('Location', '').split('/')[-1]

                logger.info(
                    f"Order placed: {signal.signal_type.value} {signal.position_size} {signal.symbol}",
                    extra={'symbol': signal.symbol, 'action': signal.signal_type.value,
                           'quantity': signal.position_size, 'order_id': order_id,
                           'price': getattr(signal, 'price', None)}
                )

                # Track pending order
                self.pending_orders[order_id] = {
                    'symbol': signal.symbol,
                    'signal': signal,
                    'status': 'PENDING',
                    'placed_time': datetime.now()
                }
                self.order_id_to_symbol[order_id] = signal.symbol
                
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.DECISION,
                    symbol=signal.symbol,
                    title=f"✅ Order Submitted",
                    message=f"Order ID: {order_id} - Waiting for fill confirmation",
                    data={
                        'order_id': order_id,
                        'quantity': signal.position_size,
                        'expected_price': signal.entry_price
                    },
                    importance=9
                ))

                # Wait a moment and check order status immediately
                await asyncio.sleep(2)
                # Wait for fill verification
                if await self._verify_order_fill(order_id, signal):
                    await self._place_bracket_orders(signal)

                    # Log the trade for analytics
                    log_trade(
                        symbol=signal.symbol,
                        action='BUY' if signal.signal_type == SignalType.BUY else 'SELL',
                        quantity=signal.position_size,
                        price=signal.entry_price,
                        order_type='LIMIT' if use_limit else 'MARKET',
                        side='long' if signal.signal_type == SignalType.BUY else 'short',
                        reason=signal.reasoning.get('primary_reason', 'strategy_signal'),
                        strategy=signal.reasoning.get('strategy', 'unknown'),
                        confidence=signal.confidence,
                        stop_loss=signal.stop_loss,
                        take_profit=signal.take_profit,
                        account_balance=self.risk_manager.account_balance,
                        mode=self.mode.value,
                        order_id=order_id
                    )
                    log_system_event('TRADE_EXECUTED', f"Executed {signal.signal_type.value} for {signal.symbol}", 'INFO', {
                        'symbol': signal.symbol,
                        'quantity': signal.position_size,
                        'price': signal.entry_price,
                        'order_id': order_id
                    })

                    return True
                else:
                    return False                
            else:
                # Parse rejection reason
                rejection = self._parse_order_rejection(response)
                
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.WARNING,
                    symbol=signal.symbol,
                    title=f"❌ Order Rejected: {rejection['reason']}",
                    message=rejection['message'],
                    data={'details': rejection['details']},
                    importance=10
                ))
                return False
                
        except InsufficientFundsException as e:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.WARNING,
                symbol=signal.symbol,
                title=f"💸 Insufficient Funds",
                message=e.args[0],
                data=e.details,
                importance=9
            ))
            logger.warning(f"Insufficient funds for {signal.symbol}: {e}")
            self.error_recovery.increment_error_count('InsufficientFunds', signal.symbol)
            action = self.error_recovery.suggest_recovery_action(e)
            if action:
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.INFO,
                    symbol=signal.symbol,
                    title="💡 Suggested Action",
                    message=action,
                    importance=7
                ))
            return False
            
        except RateLimitException as e:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.WARNING,
                symbol=signal.symbol,
                title=f"⏳ Rate Limited",
                message=f"API rate limit hit. Retry after {e.retry_after}s",
                importance=8
            ))
            logger.warning(f"Rate limited: {e}")
            # Schedule retry
            asyncio.create_task(self._retry_order_after_delay(signal, e.retry_after))
            return False
            
        except (NetworkException, requests.exceptions.RequestException) as e:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.WARNING,
                symbol=signal.symbol,
                title=f"🌐 Network Error",
                message="Connection issue - will retry",
                importance=7
            ))
            logger.error(f"Network error: {e}")
            # Retry with exponential backoff
            asyncio.create_task(self._retry_order_with_backoff(signal))
            return False
            
        except StaleDataException as e:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.WARNING,
                symbol=signal.symbol,
                title=f"📊 Stale Data",
                message="Market data too old - refreshing",
                importance=6
            ))
            logger.warning(f"Stale data for {signal.symbol}")
            return False
            
        except CircuitBreakerException as e:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.WARNING,
                symbol=signal.symbol,
                title=f"🔌 Circuit Breaker Activated",
                message=f"Too many failures - pausing for {e.cooldown_minutes} minutes",
                importance=9
            ))
            logger.warning(f"Circuit breaker activated: {e}")
            self.error_recovery.record_circuit_breaker_open('order_placement', str(e))
            return False
            
        except AuthenticationException as e:
            # Fatal - stop trading
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.WARNING,
                symbol=signal.symbol,
                title=f"🔐 Authentication Failed",
                message="Invalid credentials - stopping bot",
                importance=10
            ))
            logger.critical(f"Authentication failed: {e}")
            self.stop_trading = True
            raise
            
        except Exception as e:
            # Unknown error - log details for debugging
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.WARNING,
                symbol=signal.symbol,
                title=f"❌ Unexpected Error",
                message=f"Unknown error: {type(e).__name__}",
                data={'error': str(e), 'traceback': traceback.format_exc()},
                importance=10
            ))
            logger.error(f"Unexpected order error: {e}", exc_info=True)
            return False
    async def _retry_order_after_delay(self, signal, delay: float):
        """Retry order after specified delay"""
        await asyncio.sleep(delay)
        await self._execute_real_trade(signal)
    
    async def _retry_order_with_backoff(self, signal, attempt: int = 0, max_attempts: int = 3):
        """Retry order with exponential backoff"""
        if attempt >= max_attempts:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.WARNING,
                symbol=signal.symbol,
                title=f"❌ Max Retries Exceeded",
                message=f"Failed after {max_attempts} attempts",
                importance=9
            ))
            return
        
        delay = get_retry_delay(NetworkException("Network error"), attempt)
        await asyncio.sleep(delay)
        
        try:
            await self._execute_real_trade(signal)
        except (NetworkException, requests.exceptions.RequestException):
            await self._retry_order_with_backoff(signal, attempt + 1, max_attempts)
    
    # Add these methods after _execute_real_trade in TradingEngineWithCommentary class:

    async def _check_existing_orders(self, symbol: str) -> Dict[str, Any]:
        """Check for existing orders on a symbol"""
        try:
            response = self.schwab_client.get_orders_for_account(
                self.account_hash,
                from_entered_datetime=datetime.now() - timedelta(days=1),
                status=self.schwab_client.Order.Status.WORKING  # Only active orders
            )
            
            if response.status_code == 200:
                orders = response.json()
                symbol_orders = []
                
                for order in orders:
                    # Check main order
                    order_symbol = self._extract_symbol_from_order(order)
                    if order_symbol == symbol:
                        symbol_orders.append(order)
                    
                    # Check child orders (OCO)
                    for child in order.get('childOrderStrategies', []):
                        child_symbol = self._extract_symbol_from_order(child)
                        if child_symbol == symbol:
                            symbol_orders.append(order)
                            break
                
                return {
                    'has_orders': len(symbol_orders) > 0,
                    'orders': symbol_orders,
                    'count': len(symbol_orders)
                }
        except Exception as e:
            logger.error(f"Error checking orders: {e}")
        
        return {'has_orders': False, 'orders': [], 'count': 0}

    def _extract_symbol_from_order(self, order: Dict) -> Optional[str]:
        """Extract symbol from order structure"""
        try:
            legs = order.get('orderLegCollection', [])
            if legs:
                return legs[0].get('instrument', {}).get('symbol')
        except (KeyError, IndexError, TypeError) as e:
            logger.debug(f"Could not extract symbol from order: {e}")
        return None

    async def _cancel_existing_orders(self, symbol: str) -> bool:
        """Cancel existing orders for a symbol"""
        try:
            existing = await self._check_existing_orders(symbol)
            
            if existing['has_orders']:
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.DECISION,
                    symbol=symbol,
                    title=f"🔄 Canceling Existing Orders",
                    message=f"Found {existing['count']} existing orders for {symbol}",
                    importance=7
                ))
                
                for order in existing['orders']:
                    order_id = order.get('orderId')
                    if order_id:
                        try:
                            response = self.schwab_client.cancel_order(self.account_id, order_id)
                            if response.status_code in [200, 201, 202]:
                                logger.info(f"Canceled order {order_id}")
                        except Exception as e:
                            logger.error(f"Failed to cancel order {order_id}: {e}")
                
                # Wait for cancellations to process
                await asyncio.sleep(1)
                return True
        
        except Exception as e:
            logger.error(f"Error canceling orders: {e}")
        
        return False
    '''
    async def _check_buying_power(self, required_amount: float) -> Tuple[bool, float]:
        """Check if we have sufficient buying power"""
        try:
            if hasattr(self, '_cached_buying_power'):
                # Use cached value if recent (within 30 seconds)
                if (datetime.now() - self._last_bp_check).total_seconds() < 30:
                    return self._cached_buying_power >= required_amount, self._cached_buying_power
            
            response = self.schwab_client.get_account(self.account_hash)
            logger.info(f" Checking Buying Power: {response.json()}")
            if response.status_code == 200:
                data = response.json()
                balances = data.get('securitiesAccount', {}).get('currentBalances', {})
                
                # Get appropriate buying power based on account type
                buying_power = balances.get('buyingPower', 0)
                
                # For margin accounts
                if 'dayTradingBuyingPower' in balances:
                    buying_power = min(buying_power, balances['dayTradingBuyingPower'])
                
                # Cache the result
                self._cached_buying_power = buying_power
                self._last_bp_check = datetime.now()
                
                # Update risk manager
                self.risk_manager.buying_power = buying_power
                
                return buying_power >= required_amount, buying_power
                
        except Exception as e:
            logger.error(f"Error checking buying power: {e}")
        
        return False, 0
    '''
    async def _check_buying_power(self, required_amount: float) -> Tuple[bool, float]:
        """Check if we have sufficient buying power"""
        try:
            # Check if we should use cache
            cache_age = (datetime.now() - self._last_bp_check).total_seconds()
        
            # Only use cache if we have a valid cached value AND it's recent
            if self._cached_buying_power > 0 and cache_age < 30:
                logger.debug(f"Using cached buying power: ${self._cached_buying_power:,.2f} (age: {cache_age:.1f}s)")
                return self._cached_buying_power >= required_amount, self._cached_buying_power
            
            # Fetch fresh data
            response = self.schwab_client.get_account(self.account_hash)
            logger.info(f" Buying power check: {response.json()}")
            if response.status_code != 200:
                logger.error(f"Failed to get account data: {response.status_code}")
                return False, 0
            
            data = response.json()
            
            # Log the structure for debugging
            logger.debug(f"Account data keys: {list(data.keys())}")
            
            # Navigate to the correct location
            account_data = data.get('securitiesAccount', {})
            
            # Try multiple possible locations for buying power
            buying_power = 0
            
            # Method 1: Check currentBalances
            if 'currentBalances' in account_data:
                balances = account_data['currentBalances']
                logger.debug(f"Balance fields available: {list(balances.keys())}")
                
                # Try different field names
                buying_power = (
                    balances.get('buyingPower', 0) or
                    balances.get('availableFunds', 0) or
                    balances.get('availableFundsNonMarginableTrade', 0) or
                    balances.get('buyingPowerNonMarginableTrade', 0)
                )
                
                # For margin accounts, check day trading buying power
                if 'dayTradingBuyingPower' in balances:
                    day_trading_bp = balances.get('dayTradingBuyingPower', 0)
                    if day_trading_bp > 0:
                        buying_power = min(buying_power, day_trading_bp)
            
            # Method 2: Check projectedBalances if currentBalances didn't work
            if buying_power == 0 and 'projectedBalances' in account_data:
                projected = account_data['projectedBalances']
                buying_power = (
                    projected.get('buyingPower', 0) or
                    projected.get('availableFunds', 0) or
                    projected.get('availableFundsNonMarginableTrade', 0)
                )
            
            # Method 3: Calculate from cash and margin if still 0
            if buying_power == 0 and 'currentBalances' in account_data:
                balances = account_data['currentBalances']
                cash = balances.get('cashBalance', 0) or balances.get('totalCash', 0)
                margin = balances.get('marginBalance', 0)
                
                # For cash accounts, buying power is usually just cash
                # For margin accounts, it could be cash + margin available
                account_type = account_data.get('type', 'CASH')
                if account_type == 'MARGIN':
                    buying_power = cash + margin
                else:
                    buying_power = cash
            
            # Log what we found
            logger.info(f"Buying power found: ${buying_power:,.2f} (required: ${required_amount:,.2f})")
            
            # Cache the result
            self._cached_buying_power = buying_power
            self._last_bp_check = datetime.now()
            
            # Update risk manager
            self.risk_manager.buying_power = buying_power
            
            return buying_power >= required_amount, buying_power
            
        except Exception as e:
            logger.error(f"Error checking buying power: {e}", exc_info=True)
            # Return cached value if available
            if hasattr(self, '_cached_buying_power') and self._cached_buying_power > 0:
                logger.warning(f"Using last known buying power: ${self._cached_buying_power:,.2f}")
                return self._cached_buying_power >= required_amount, self._cached_buying_power
        
        return False, 0
    def _parse_order_rejection(self, response) -> Dict[str, Any]:
        """Parse Schwab order rejection reasons"""
        try:
            if hasattr(response, 'text'):
                error_data = json.loads(response.text)
                error_message = str(error_data).lower()
                
                # Common rejection reasons
                if 'buying power' in error_message or 'insufficient funds' in error_message:
                    return {
                        'reason': 'INSUFFICIENT_FUNDS',
                        'message': 'Not enough buying power for this trade',
                        'details': error_data
                    }
                elif 'duplicate' in error_message:
                    return {
                        'reason': 'DUPLICATE_ORDER',
                        'message': 'A similar order already exists for this symbol',
                        'details': error_data
                    }
                elif 'market closed' in error_message or 'outside hours' in error_message:
                    return {
                        'reason': 'MARKET_CLOSED',
                        'message': 'Cannot place order - market is closed',
                        'details': error_data
                    }
                elif 'position' in error_message and 'exist' in error_message:
                    return {
                        'reason': 'POSITION_EXISTS',
                        'message': 'Position already exists in this symbol',
                        'details': error_data
                    }
                elif 'halted' in error_message:
                    return {
                        'reason': 'SYMBOL_HALTED',
                        'message': 'Trading is halted for this symbol',
                        'details': error_data
                    }
                elif 'invalid' in error_message and 'price' in error_message:
                    return {
                        'reason': 'INVALID_PRICE',
                        'message': 'Order price is invalid (check stop/limit prices)',
                        'details': error_data
                    }
        except Exception as e:
            logger.debug(f"Could not parse order rejection details: {e}")

        return {
            'reason': 'UNKNOWN',
            'message': f'Order rejected - Status: {response.status_code}',
            'details': response.text if hasattr(response, 'text') else None
        }

    def _validate_oco_prices(self, symbol: str, stop_price: float, take_profit: float, is_short: bool = False) -> bool:
        """Validate OCO order prices before submission for both long and short positions"""
        try:
            # Get current quote
            quote = self.data_provider.get_quote(symbol)
            if not quote:
                return True  # Skip validation if no quote
            
            current_price = quote.get('last', 0)
            bid = quote.get('bid', current_price)
            ask = quote.get('ask', current_price)
            
            if is_short:
                # For short positions (sell to open):
                # - Stop loss must be ABOVE current ask (we're buying back at a loss)
                # - Take profit must be BELOW current bid (we're buying back at a profit)
                
                if stop_price <= ask:
                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.WARNING,
                        symbol=symbol,
                        title=f"⚠️ Invalid Stop Price for Short",
                        message=f"Stop ${stop_price:.2f} must be above ask ${ask:.2f} for short position",
                        importance=8
                    ))
                    return False
                
                if take_profit >= bid:
                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.WARNING,
                        symbol=symbol,
                        title=f"⚠️ Invalid Target Price for Short",
                        message=f"Target ${take_profit:.2f} must be below bid ${bid:.2f} for short position",
                        importance=8
                    ))
                    return False
            else:
                # For long positions (existing logic):
                # - Stop loss must be below current bid
                # - Take profit must be above current ask
                
                if stop_price >= bid:
                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.WARNING,
                        symbol=symbol,
                        title=f"⚠️ Invalid Stop Price",
                        message=f"Stop ${stop_price:.2f} must be below bid ${bid:.2f}",
                        importance=8
                    ))
                    return False
                
                if take_profit <= ask:
                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.WARNING,
                        symbol=symbol,
                        title=f"⚠️ Invalid Target Price",
                        message=f"Target ${take_profit:.2f} should be above ask ${ask:.2f}",
                        importance=8
                    ))
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Price validation error: {e}")
            return True  # Allow order if validation fails
    
    async def _place_bracket_orders(self, signal, parent_order_id: str):
        """Place stop loss and take profit orders as OCO"""
        try:
            from schwab.orders.common import one_cancels_other, Duration, Session, OrderType
            from schwab.orders.equities import equity_sell_limit

            # Create take profit order (limit sell)
            take_profit_order = equity_sell_limit(
                signal.symbol,
                signal.position_size,
                signal.take_profit
            ).set_duration(Duration.GOOD_TILL_CANCEL).set_session(Session.NORMAL)

            # Create stop loss order (stop limit - using stop price slightly below limit)
            # For stop limit, we set both stop price and limit price
            stop_loss_order = (equity_sell_limit(
                signal.symbol,
                signal.position_size,
                signal.stop_loss * 0.995  # Limit price slightly below stop
            ).set_order_type(OrderType.STOP_LIMIT)
            .set_stop_price(signal.stop_loss)
            .set_duration(Duration.GOOD_TILL_CANCEL)
            .set_session(Session.NORMAL))

            # Create OCO order using helper function
            oco_order = one_cancels_other(take_profit_order, stop_loss_order)

            # Build and place the OCO order
            response = self.schwab_client.place_order(self.account_hash, oco_order.build())

            if response.status_code in [200, 201]:
                # Extract order ID from response
                order_id = response.headers.get('Location', '').split('/')[-1]

                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.RISK_ASSESSMENT,
                    symbol=signal.symbol,
                    title=f"✅ OCO Bracket Order Placed",
                    message=f"Stop Loss: ${signal.stop_loss:.2f} | Take Profit: ${signal.take_profit:.2f}",
                    data={
                        'stop_loss': signal.stop_loss,
                        'take_profit': signal.take_profit,
                        'order_type': 'OCO',
                        'order_id': order_id
                    },
                    importance=7
                ))
            else:
                rejection = self._parse_order_rejection(response)
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.WARNING,
                    symbol=signal.symbol,
                    title=f"⚠️ OCO Order Failed: {rejection['reason']}",
                    message=rejection['message'],
                    data={'details': rejection['details']},
                    importance=7
                ))

        except Exception as e:
            logger.error(f"Bracket order error: {e}", exc_info=True)
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.WARNING,
                symbol=signal.symbol,
                title=f"⚠️ Bracket Order Error",
                message=f"Could not place stop/target orders: {str(e)}",
                importance=7
            ))
    def _validate_oco_prices(self, symbol: str, stop_price: float, take_profit: float, is_short: bool = False) -> bool:
        """Validate OCO order prices before submission for both long and short positions"""
        try:
            # Get current quote
            quote = self.data_provider.get_quote(symbol)
            if not quote:
                return True  # Skip validation if no quote
            
            current_price = quote.get('last', 0)
            bid = quote.get('bid', current_price)
            ask = quote.get('ask', current_price)
            
            if is_short:
                # For short positions (sell to open):
                # - Stop loss must be ABOVE current ask (we're buying back at a loss)
                # - Take profit must be BELOW current bid (we're buying back at a profit)
                
                if stop_price <= ask:
                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.WARNING,
                        symbol=symbol,
                        title=f"⚠️ Invalid Stop Price for Short",
                        message=f"Stop ${stop_price:.2f} must be above ask ${ask:.2f} for short position",
                        importance=8
                    ))
                    return False
                
                if take_profit >= bid:
                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.WARNING,
                        symbol=symbol,
                        title=f"⚠️ Invalid Target Price for Short",
                        message=f"Target ${take_profit:.2f} must be below bid ${bid:.2f} for short position",
                        importance=8
                    ))
                    return False
            else:
                # For long positions (existing logic):
                # - Stop loss must be below current bid
                # - Take profit must be above current ask
                
                if stop_price >= bid:
                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.WARNING,
                        symbol=symbol,
                        title=f"⚠️ Invalid Stop Price",
                        message=f"Stop ${stop_price:.2f} must be below bid ${bid:.2f}",
                        importance=8
                    ))
                    return False
                
                if take_profit <= ask:
                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.WARNING,
                        symbol=symbol,
                        title=f"⚠️ Invalid Target Price",
                        message=f"Target ${take_profit:.2f} should be above ask ${ask:.2f}",
                        importance=8
                    ))
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Price validation error: {e}")
            return True  # Allow order if validation fails
    async def _check_specific_order_status(self, order_id: str) -> str:
        """Check status of a specific order immediately"""
        try:
            response = self.schwab_client.get_order(self.account_hash, order_id)
            
            if response.status_code == 200:
                order_info = response.json()
                status = order_info.get('status', 'UNKNOWN')
                
                if order_id in self.pending_orders:
                    self.pending_orders[order_id]['status'] = status
                    
                    if status == 'REJECTED':
                        symbol = self.pending_orders[order_id]['symbol']
                        self.commentary.add_commentary(TradingCommentary(
                            timestamp=datetime.now(),
                            type=CommentaryType.WARNING,
                            symbol=symbol,
                            title=f"❌ Order Rejected by Broker",
                            message=f"Order {order_id} was rejected. Not proceeding with position.",
                            data={'order_info': order_info},
                            importance=10
                        ))
                        # Remove from tracking
                        del self.pending_orders[order_id]
                        if order_id in self.order_id_to_symbol:
                            del self.order_id_to_symbol[order_id]
                    
                    elif status == 'FILLED':
                        self.commentary.add_commentary(TradingCommentary(
                            timestamp=datetime.now(),
                            type=CommentaryType.DECISION,
                            symbol=self.pending_orders[order_id]['symbol'],
                            title=f"✅ Order Filled",
                            message=f"Order {order_id} has been filled",
                            importance=8
                        ))
                
                return status
        except Exception as e:
            logger.error(f"Error checking order {order_id}: {e}")
            return 'UNKNOWN'
        
    async def _verify_order_fill(self, order_id: str, signal, max_retries: int = 3) -> bool:
        """Verify order filled with retries"""
        for attempt in range(max_retries):
            status = await self._check_specific_order_status(order_id)
            if status == 'FILLED':
                return True
            elif status in ['REJECTED', 'CANCELED']:
                return False
            await asyncio.sleep(2)
        
        # Cancel unfilled order
        try:
            self.schwab_client.cancel_order(self.account_hash, order_id)
            del self.pending_orders[order_id]
        except Exception as e:
            logger.warning(f"Failed to cancel unfilled order {order_id}: {e}")
        return False
    
    async def _check_order_status(self):
        """Check status of pending orders"""
        if not self.pending_orders or not self.schwab_client:
            return
        
        # Check orders every 10 seconds
        if (datetime.now() - self.last_order_check).total_seconds() < 10:
            return
        
        self.last_order_check = datetime.now()
        
        for order_id, order_data in list(self.pending_orders.items()):
            try:
                response = self.schwab_client.get_order(
                    self.account_hash,
                    order_id
                    #fields=[schwab_client.Orders.Fields.EXECUTION_LEGS]
                )
                
                if response.status_code == 200:
                    order_info = response.json()
                    logger.info(f"Existing Orders : {order_info}")
                    status = order_info.get('status', 'UNKNOWN')
                    
                    if status == 'FILLED':
                        # Get fill price
                        fill_price = signal.entry_price  # Default
                        if 'orderActivityCollection' in order_info:
                            for activity in order_info['orderActivityCollection']:
                                if activity.get('executionType') == 'FILL':
                                    legs = activity.get('executionLegs', [])
                                    if legs:
                                        fill_price = legs[0].get('price', signal.entry_price)
                                        expected = order_data['signal'].entry_price
                                        slippage = abs(fill_price - expected) / expected
                                        self.performance_metrics['slippage'].append(slippage)
                        # Create position
                        signal = order_data['signal']
                        position = Position(
                            symbol=signal.symbol,
                            entry_price=fill_price,
                            current_price=fill_price,
                            quantity=signal.position_size,
                            side='long' if signal.signal_type == SignalType.BUY else 'short',
                            stop_loss=signal.stop_loss,
                            take_profit=signal.take_profit,
                            entry_time=datetime.now(),
                            reasoning=signal.reasoning
                        )
                        self.positions[signal.symbol] = position
                        
                        # Initialize exit tracking
                        self.exit_manager.initialize_position_tracking(
                            signal.symbol,
                            fill_price,
                            signal.stop_loss,
                            signal.take_profit
                        )
                        
                        self.commentary.add_commentary(TradingCommentary(
                            timestamp=datetime.now(),
                            type=CommentaryType.DECISION,
                            symbol=signal.symbol,
                            title=f"💰 Order Filled!",
                            message=f"Position opened at ${fill_price:.2f}",
                            data={
                                'order_id': order_id,
                                'fill_price': fill_price,
                                'quantity': signal.position_size
                            },
                            importance=9
                        ))
                        
                        del self.pending_orders[order_id]
                        
                    elif status in ['CANCELED', 'REJECTED', 'EXPIRED']:
                        self.commentary.add_commentary(TradingCommentary(
                            timestamp=datetime.now(),
                            type=CommentaryType.WARNING,
                            symbol=order_data['symbol'],
                            title=f"❌ Order {status}",
                            message=f"Order {order_id} was {status.lower()}",
                            importance=8
                        ))
                        del self.pending_orders[order_id]
                        
            except Exception as e:
                logger.error(f"Error checking order {order_id}: {e}")
    
    async def _close_real_position(self, position, exit_portion: float = 1.0):
        """Close real position through Schwab"""
        if not self.schwab_client:
            return False
        
        try:
            from schwab.orders.equities import (equity_sell_market, equity_buy_market, 
                                                equity_buy_to_cover_market, equity_sell_limit,
                                                equity_buy_limit, equity_buy_to_cover_limit)
            from schwab.orders.common import Duration, Session
            
            quantity_to_sell = int(position.quantity * exit_portion)
            
            # Check market hours for order type
            use_limit, reason = self.should_use_limit_order()
            is_regular, session = self.is_market_hours()
            
            # Get current price for limit orders
            current_price = position.current_price if hasattr(position, 'current_price') else position.entry_price
            
            # Create appropriate order based on position side and market hours
            if use_limit or self.use_limit_orders_extended_hours:
                # Use limit orders during extended hours
                logger.info(f"Using limit order to close position: {reason}")
                
                if position.side == 'long':
                    # Sell with limit slightly below current for quick fill
                    limit_price = current_price * 0.999
                    order_builder = equity_sell_limit(position.symbol, quantity_to_sell, limit_price)
                else:  # SHORT position - need to buy to cover
                    # Buy to cover with limit slightly above current for quick fill
                    limit_price = current_price * 1.001
                    try:
                        order_builder = equity_buy_to_cover_limit(position.symbol, quantity_to_sell, limit_price)
                    except AttributeError:
                        order_builder = equity_buy_limit(position.symbol, quantity_to_sell, limit_price)
                        order_builder.set_instruction('BUY_TO_COVER')
            else:
                # Use market orders during regular hours
                if position.side == 'long':
                    order_builder = equity_sell_market(position.symbol, quantity_to_sell)
                else:  # SHORT position - need to buy to cover
                    try:
                        order_builder = equity_buy_to_cover_market(position.symbol, quantity_to_sell)
                    except AttributeError:
                        order_builder = equity_buy_market(position.symbol, quantity_to_sell)
                        order_builder.set_instruction('BUY_TO_COVER')

            order_builder.set_duration(Duration.DAY)
            
            # Set session based on market hours
            if session == 'premarket' or session == 'afterhours':
                order_builder.set_session(Session.EXTENDED)
            else:
                order_builder.set_session(Session.NORMAL)
            
            order = order_builder.build()
            
            response = self.schwab_client.place_order(self.account_hash, order)
            
            if response.status_code in [200, 201]:
                order_id = response.headers.get('Location', '').split('/')[-1]

                logger.info(
                    f"Close order placed: SELL {quantity_to_sell} {position.symbol}",
                    extra={'symbol': position.symbol, 'action': 'SELL',
                           'quantity': quantity_to_sell, 'order_id': order_id,
                           'reason': reason}
                )

                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.DECISION,
                    symbol=position.symbol,
                    title=f"📤 Closing Position",
                    message=f"Selling {quantity_to_sell} shares at market",
                    importance=9
                ))

                # Track this as a closing order
                self.pending_orders[order_id] = {
                    'symbol': position.symbol,
                    'type': 'CLOSE',
                    'quantity': quantity_to_sell,
                    'status': 'PENDING',
                    'placed_time': datetime.now()
                }
                
                return True
            else:
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.WARNING,
                    symbol=position.symbol,
                    title=f"❌ Close Order Failed",
                    message=f"Could not close position: {response.status_code}",
                    importance=10
                ))
                return False
                
        except Exception as e:
            logger.error(f"Position close error: {e}")
            return False
    async def _update_real_positions(self):
        """Update real positions from account - syncs ALL Schwab positions"""
        if not self.schwab_client or self.mode != TradingMode.LIVE:
            return

        try:
            # Get all Schwab positions
            schwab_positions = await self.get_schwab_positions()

            # Create a set of symbols from Schwab
            schwab_symbols = {pos['symbol'] for pos in schwab_positions}

            # Add or update positions from Schwab
            for pos_data in schwab_positions:
                symbol = pos_data['symbol']

                if symbol not in self.positions:
                    # Create new position for externally opened position
                    # IMPORTANT: External positions have NO automatic stop loss/take profit
                    # They are flagged as manually managed and the bot won't take actions on them
                    position = Position(
                        symbol=symbol,
                        quantity=abs(pos_data['quantity']),
                        entry_price=pos_data['average_price'],
                        current_price=pos_data['current_price'],
                        stop_loss=0,  # No automatic stop loss for external positions
                        take_profit=float('inf'),  # No automatic take profit
                        entry_time=datetime.now(),
                        side='long' if pos_data['quantity'] > 0 else 'short',
                        reasoning={'source': 'external', 'strategy': 'manual_entry'}
                    )
                    position.unrealized_pnl = pos_data['total_pnl']
                    position.is_external = True  # Flag as externally created
                    position.is_manually_managed = True  # Bot won't auto-manage this
                    self.positions[symbol] = position

                    # Log that we found an external position
                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.MARKET_ANALYSIS,
                        symbol=symbol,
                        title=f"📥 External Position Detected (Manual Mode)",
                        message=f"Found {abs(pos_data['quantity'])} shares of {symbol} "
                                f"({'long' if pos_data['quantity'] > 0 else 'short'})\n"
                                f"Entry: ${pos_data['average_price']:.2f}, "
                                f"Current: ${pos_data['current_price']:.2f}\n"
                                f"⚠️ This position is MANUALLY MANAGED - bot will NOT auto-close",
                        data=pos_data,
                        importance=7
                    ))
                else:
                    # Update existing position prices only
                    position = self.positions[symbol]
                    position.current_price = pos_data['current_price']
                    position.unrealized_pnl = pos_data['total_pnl']
            
            # Remove positions that no longer exist in Schwab
            positions_to_remove = []
            for symbol in self.positions:
                if symbol not in schwab_symbols:
                    positions_to_remove.append(symbol)
            
            for symbol in positions_to_remove:
                position = self.positions[symbol]
                
                # Try to get current market price for more accurate P&L
                exit_price = position.current_price
                try:
                    if self.data_provider:
                        quote = self.data_provider.get_quote(symbol)
                        if quote and 'last' in quote:
                            exit_price = quote['last']
                except Exception as e:
                    logger.debug(f"Could not get quote for {symbol}: {e}")
                
                # Calculate final P&L
                if position.side == 'short':
                    final_pnl = (position.entry_price - exit_price) * position.quantity
                else:  # long
                    final_pnl = (exit_price - position.entry_price) * position.quantity
                
                # Update risk manager - Note: P&L will be updated from Schwab on next sync
                # We don't track P&L internally anymore
                if final_pnl < 0:
                    self.risk_manager.consecutive_losses += 1
                else:
                    self.risk_manager.consecutive_losses = 0
                
                # Record in trade history
                self.trade_history.append({
                    'symbol': position.symbol,
                    'entry_time': position.entry_time.isoformat(),
                    'exit_time': datetime.now().isoformat(),
                    'entry_price': position.entry_price,
                    'exit_price': exit_price,
                    'quantity': position.quantity,
                    'pnl': final_pnl,
                    'exit_reason': 'external_close',
                    'reasoning': position.reasoning
                })
                
                # Remove from positions
                del self.positions[symbol]
                
                # Add commentary
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.DECISION,
                    symbol=symbol,
                    title=f"📤 Position Closed Externally",
                    message=f"{symbol} closed outside bot - P&L: ${final_pnl:.2f} ({'profit' if final_pnl > 0 else 'loss'})",
                    importance=6
                ))
                        
        except Exception as e:
            logger.error(f"Position update error: {e}")
    def _in_warmup_window(self) -> bool:
        """True if we're within WARMUP_MINUTES_BEFORE_OPEN of regular-hours
        open on a weekday. Lets the bot pre-scan the watchlist so by 09:30
        ET it already has indicator state and a candidate list ready."""
        warmup_minutes = Config().WARMUP_MINUTES_BEFORE_OPEN
        if warmup_minutes <= 0:
            return False
        try:
            import pytz
            from datetime import time, timedelta
            now_et = datetime.now(pytz.timezone("US/Eastern"))
            if now_et.weekday() >= 5:  # weekend
                return False
            open_dt = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
            warmup_start = open_dt - timedelta(minutes=warmup_minutes)
            return warmup_start <= now_et < open_dt
        except Exception:
            return False

    def is_market_hours(self) -> tuple[bool, str]:
        """Check if we're in regular market hours
        
        Returns:
            (is_regular_hours, session_type)
            session_type: 'regular', 'premarket', 'afterhours', or 'closed'
        """
        from datetime import time
        import pytz
        
        # Get current time in Eastern timezone
        eastern = pytz.timezone('US/Eastern')
        now = datetime.now(eastern)
        current_time = now.time()
        weekday = now.weekday()
        
        # Skip weekends
        if weekday >= 5:  # Saturday = 5, Sunday = 6
            return False, 'closed'
        
        # Define market sessions (Eastern Time)
        premarket_start = time(4, 0)    # 4:00 AM ET
        regular_start = time(9, 30)     # 9:30 AM ET
        regular_end = time(16, 0)       # 4:00 PM ET
        afterhours_end = time(20, 0)    # 8:00 PM ET
        
        # Check which session we're in
        if current_time < premarket_start:
            return False, 'closed'
        elif current_time < regular_start:
            return False, 'premarket'
        elif current_time < regular_end:
            return True, 'regular'
        elif current_time < afterhours_end:
            return False, 'afterhours'
        else:
            return False, 'closed'
    
    def should_use_limit_order(self) -> tuple[bool, str]:
        """Determine if we should use limit orders based on market hours
        
        Returns:
            (use_limit, reason)
        """
        is_regular, session = self.is_market_hours()
        
        if is_regular:
            return False, "Regular hours - market orders OK"
        
        if session == 'closed':
            return True, "Market closed - orders will queue"
        elif session == 'premarket':
            if not self.allow_premarket:
                return True, "Premarket trading disabled"
            return True, "Premarket - using limit orders for safety"
        elif session == 'afterhours':
            if not self.allow_afterhours:
                return True, "After-hours trading disabled"
            return True, "After-hours - using limit orders for safety"
        
        return True, "Extended hours - using limit orders"
    
    def get_todays_trades(self):
        """Get actual trades from Schwab (both open positions and closed trades from today)"""
        if self.mode == TradingMode.LIVE and self.schwab_client:
            return self.get_schwab_trades_today()
        else:
            # Fallback to internal trade history for simulation mode
            return self.get_internal_trades_today()
    
    def get_internal_trades_today(self):
        """Get trades from internal history (used for simulation)"""
        today = datetime.now().date()
        todays_trades = []
        
        for trade in self.trade_history:
            # Parse the entry time
            try:
                if isinstance(trade.get('entry_time'), str):
                    entry_time = datetime.fromisoformat(trade['entry_time'].replace('Z', '+00:00'))
                else:
                    entry_time = trade.get('entry_time', datetime.now())
                
                # Check if trade is from today
                if entry_time.date() == today:
                    todays_trades.append(trade)
            except Exception as e:
                logger.debug(f"Error parsing trade time: {e}")
                # If we can't parse the time, include it if it's recent
                if len(self.trade_history) <= 50 and trade in self.trade_history[-50:]:
                    todays_trades.append(trade)
        
        # Sort by entry time (most recent first)
        todays_trades.sort(key=lambda x: x.get('entry_time', ''), reverse=True)
        
        return todays_trades
    
    def get_schwab_trades_today(self):
        """Get actual trades from Schwab API for today"""
        try:
            today = datetime.now().date()
            todays_trades = []
            
            # Get today's filled orders from Schwab
            from_time = datetime.combine(today, datetime.min.time())
            to_time = datetime.now()
            
            # Get orders for today
            response = self.schwab_client.get_orders_for_account(
                self.account_hash,
                from_entered_datetime=from_time,
                to_entered_datetime=to_time
            )
            
            if response.status_code == 200:
                orders = response.json()
                
                for order in orders:
                    # Process each order
                    status = order.get('status', '')
                    
                    # Include both FILLED orders (closed trades) and WORKING orders (open positions)
                    if status in ['FILLED', 'WORKING', 'QUEUED', 'ACCEPTED']:
                        # Extract order details
                        order_activities = order.get('orderActivityCollection', [])
                        order_legs = order.get('orderLegCollection', [])
                        
                        for leg in order_legs:
                            symbol = leg.get('instrument', {}).get('symbol', 'Unknown')
                            quantity = leg.get('quantity', 0)
                            instruction = leg.get('instruction', '')  # BUY, SELL, etc.
                            
                            # Get execution details if filled
                            if order_activities:
                                for activity in order_activities:
                                    execution_legs = activity.get('executionLegs', [])
                                    for exec_leg in execution_legs:
                                        exec_price = exec_leg.get('price', 0)
                                        exec_quantity = exec_leg.get('quantity', 0)
                                        exec_time = exec_leg.get('time', '')
                                        
                                        # Create trade record
                                        trade = {
                                            'symbol': symbol,
                                            'entry_time': order.get('enteredTime', ''),
                                            'exit_time': exec_time if status == 'FILLED' else None,
                                            'entry_price': exec_price if instruction in ['BUY', 'BUY_TO_OPEN'] else 0,
                                            'exit_price': exec_price if instruction in ['SELL', 'SELL_TO_CLOSE'] else 0,
                                            'quantity': exec_quantity,
                                            'status': status,
                                            'instruction': instruction,
                                            'pnl': 0,  # Will calculate below
                                            'reason': 'Schwab Order'
                                        }
                                        todays_trades.append(trade)
                            else:
                                # For working orders without execution yet
                                if status == 'WORKING':
                                    # Get current position info to show unrealized P&L
                                    positions = self.positions.get(symbol)
                                    current_pnl = positions.unrealized_pnl if positions else 0
                                    
                                    trade = {
                                        'symbol': symbol,
                                        'entry_time': order.get('enteredTime', ''),
                                        'exit_time': None,  # Still open
                                        'entry_price': order.get('price', 0),
                                        'exit_price': 0,
                                        'quantity': quantity,
                                        'status': 'OPEN',
                                        'instruction': instruction,
                                        'pnl': current_pnl,
                                        'reason': 'Open Position'
                                    }
                                    todays_trades.append(trade)
            
            # Also add current open positions that might not have orders today
            current_positions = self.positions
            for symbol, position in current_positions.items():
                # Check if we already have this position in trades
                if not any(t['symbol'] == symbol and t['status'] == 'OPEN' for t in todays_trades):
                    trade = {
                        'symbol': symbol,
                        'entry_time': position.entry_time.isoformat() if hasattr(position.entry_time, 'isoformat') else str(position.entry_time),
                        'exit_time': None,
                        'entry_price': position.entry_price,
                        'exit_price': 0,
                        'quantity': position.quantity,
                        'status': 'OPEN',
                        'pnl': position.unrealized_pnl,
                        'reason': 'Open Position'
                    }
                    todays_trades.append(trade)
            
            # Sort by time (most recent first)
            todays_trades.sort(key=lambda x: x.get('entry_time', ''), reverse=True)
            
            return todays_trades
            
        except Exception as e:
            logger.error(f"Error getting Schwab trades: {e}")
            # Fallback to internal history
            return self.get_internal_trades_today()
    
    async def _broadcast_trade_update(self, trade_record):
        """Broadcast a single trade update to all connected WebSocket clients"""
        if hasattr(self, 'connection_manager') and self.connection_manager:
            try:
                await self.connection_manager.broadcast({
                    'type': 'trade_update',
                    'data': {
                        'new_trade': trade_record,
                        'all_trades': self.get_todays_trades()
                    }
                })
            except Exception as e:
                logger.debug(f"Error broadcasting trade update: {e}")
    
    def _save_state(self):
        """Save current trading state including simulated positions."""
        # Save real position metadata (long-term flags etc.)
        positions_data = {}
        for symbol, pos in self.positions.items():
            positions_data[symbol] = {
                'is_long_term': getattr(pos, 'is_long_term', False),
                'entry_price': pos.entry_price,
                'quantity': pos.quantity,
                'side': pos.side,
                'entry_time': pos.entry_time.isoformat()
            }

        # Save simulated positions in full so they survive restarts.
        # Without this, every restart wipes sim positions → duplicate
        # entries, lost P&L tracking, and broken trailing stops.
        sim_data = {}
        for symbol, pos in self.simulated_positions.items():
            if pos is None or pos.quantity <= 0:
                continue
            sim_data[symbol] = {
                'symbol': pos.symbol,
                'entry_price': pos.entry_price,
                'current_price': pos.current_price,
                'quantity': pos.quantity,
                'side': pos.side,
                'stop_loss': pos.stop_loss,
                'take_profit': pos.take_profit,
                'entry_time': pos.entry_time.isoformat(),
                'unrealized_pnl': pos.unrealized_pnl,
                'reasoning': pos.reasoning,
                'is_long_term': getattr(pos, 'is_long_term', False),
                'scaled_out': getattr(pos, 'scaled_out', False),
                'original_stop': getattr(pos, 'original_stop', None),
                'trailing_stop': getattr(pos, 'trailing_stop', None),
            }

        state = {
            'trade_history': self.trade_history[-100:],
            'schwab_pnl': self.risk_manager.schwab_daily_pnl,
            'consecutive_losses': self.risk_manager.consecutive_losses,
            'positions_data': positions_data,
            'simulated_positions': sim_data,
            'last_save': datetime.now().isoformat()
        }

        with open("trading_state.json", 'w') as f:
            json.dump(state, f, indent=2, default=str)

        # Sync open positions to Postgres for SQL queryability
        if self.db_logger is not None:
            all_positions = {**self.positions, **self.simulated_positions}
            try:
                asyncio.get_event_loop().create_task(
                    self.db_logger.sync_positions(all_positions)
                )
            except Exception:
                pass

    def _init_schwab_client(self):
        """Initialize Schwab client with commentary"""
        try:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.TECHNICAL,
                symbol=None,
                title="🔌 Connecting to Schwab API",
                message="Attempting to establish connection with broker...",
                importance=7
            ))
            
            # Check credentials
            if not Config().SCHWAB_API_KEY:
                raise ValueError("SCHWAB_API_KEY not set")
            if not Config().SCHWAB_APP_SECRET:
                raise ValueError("SCHWAB_SECRET not set")
            
            # Check if token exists
            if Config().SCHWAB_TOKEN_PATH.exists():
                try:
                    # Fixed token loading
                    with open(Config().SCHWAB_TOKEN_PATH, 'r') as f:
                        token_file_data = json.load(f)
                        # Handle both formats
                        if 'token' in token_file_data:
                            token_data = token_file_data['token']
                        else:
                            token_data = token_file_data
                    
                    # Create client from token
                    self.schwab_client = auth.client_from_token_file(
                        Config().SCHWAB_TOKEN_PATH,
                        Config().SCHWAB_API_KEY,
                        Config().SCHWAB_APP_SECRET
                    )
                    
                    # Test if token is valid
                    response_hash = self.schwab_client.get_account_numbers()
                    response = self.schwab_client.get_accounts()
                    if response.status_code == 200:
                        self.data_provider = SchwabDataProvider(self.schwab_client, self.commentary)
                        accounts = response.json()
                        accounts_hash = response_hash.json()
                        #logger.info(f"Account Info: {accounts}")
                        if accounts:
                            self.account_id = Config().SCHWAB_ACCOUNT_NUMBER or accounts[0].get('securitiesAccount', {}).get('accountNumber')
                            self.account_hash = accounts_hash[0].get('hashValue')
                                                        
                        self.commentary.add_commentary(TradingCommentary(
                            timestamp=datetime.now(),
                            type=CommentaryType.TECHNICAL,
                            symbol=None,
                            title="✅ Schwab Connected",
                            message="Successfully connected to Schwab API. Ready to trade!",
                            importance=8
                        ))
                        return
                    else:
                        logger.warning("Token expired or invalid")
                except Exception as e:
                    logger.warning(f"Token loading failed: {e}")
            
            # If we get here, need to authenticate
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.WARNING,
                symbol=None,
                title="⚠️ Authentication Required",
                message="Schwab token not found or invalid. Running in simulation mode with REAL market data from yfinance.",
                importance=8
            ))

            # Use RealTimeDataProvider for real prices even in simulation mode
            self.data_provider = RealTimeDataProvider(self.commentary)

        except Exception as e:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.WARNING,
                symbol=None,
                title="⚠️ Broker Connection Issue",
                message=f"Could not connect to Schwab: {str(e)}. Running in simulation mode with REAL market data.",
                importance=8
            ))

            # Use RealTimeDataProvider for real prices even in simulation mode
            self.data_provider = RealTimeDataProvider(self.commentary)
     
    async def sync_positions_with_schwab(self):
        """Sync internal position tracking with actual Schwab positions"""
        if not self.schwab_client or self.mode != TradingMode.LIVE:
            return

        try:
            schwab_positions = await self.get_schwab_positions()

            # Clear out internal tracking and rebuild from Schwab
            self.positions.clear()

            for pos_data in schwab_positions:
                symbol = pos_data['symbol']

                # Create Position object for each Schwab position
                # IMPORTANT: All synced positions are marked as external/manually managed
                # The bot will NOT take automatic actions on these positions
                position = Position(
                    symbol=symbol,
                    entry_price=pos_data['average_price'],
                    current_price=pos_data['current_price'],
                    quantity=abs(pos_data['quantity']),
                    side='long' if pos_data['quantity'] > 0 else 'short',
                    stop_loss=0,  # No automatic stop loss
                    take_profit=float('inf'),  # No automatic take profit
                    entry_time=datetime.now() - timedelta(hours=1)  # Approximate
                )

                position.unrealized_pnl = pos_data['total_pnl']
                position.is_external = True  # All synced positions are external
                position.is_manually_managed = True  # Bot won't auto-manage
                self.positions[symbol] = position

            logger.info(f"Synced {len(self.positions)} positions from Schwab (all marked as manually managed)")

            # Update commentary
            if self.positions:
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.ACCOUNT_UPDATE,
                    symbol=None,
                    title="📊 Position Sync Complete",
                    message=f"Synced {len(self.positions)} positions from Schwab: {', '.join(self.positions.keys())}\n"
                           f"⚠️ All positions marked as MANUALLY MANAGED - bot will not auto-close",
                    importance=5
                ))
                
        except Exception as e:
            logger.error(f"Failed to sync positions with Schwab: {e}")
    
    async def start(self):
        """Start the trading engine with commentary"""
        self.is_running = True

        # Sync positions with Schwab on startup
        if self.mode == TradingMode.LIVE and self.schwab_client:
            await self.sync_positions_with_schwab()

        risk_per_trade = Config().MAX_RISK_PER_TRADE
        self.commentary.add_commentary(TradingCommentary(
            timestamp=datetime.now(),
            type=CommentaryType.MARKET_ANALYSIS,
            symbol=None,
            title="🎯 Trading Session Started",
            message="Beginning market analysis. I'll explain each step of my decision-making process.",
            data={
                'mode': self.mode.value,
                'account_balance': self.risk_manager.account_balance,
                'risk_per_trade': f"{risk_per_trade:.1%}"
            },
            importance=9
        ))
        
        # In start() method, after self.is_running = True
        if not self.ml_predictor.model.is_trained:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.MARKET_ANALYSIS,
                symbol=None,
                title="🎯 Initial Model Training Required",
                message="Collecting historical data for first-time model training...",
                importance=9
            ))
            # Train on initial symbols
            self.ml_predictor.retrain_model(
                self.data_provider,
                ['PLTR', 'NVDA', 'TSLA']
            )
        
        # Sync with existing Schwab positions on startup
        if self.mode == TradingMode.LIVE and self.schwab_client:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.MARKET_ANALYSIS,
                symbol=None,
                title="🔄 Syncing with Schwab Account",
                message="Checking for existing positions in your account...",
                importance=8
            ))
            
            await self._update_and_track_real_positions()
            
            if self.positions:
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.MARKET_ANALYSIS,
                    symbol=None,
                    title=f"📊 Found {len(self.positions)} Existing Positions",
                    message=f"Tracking positions: {', '.join(self.positions.keys())}",
                    importance=8
                ))

        analysis_count = 0
        self._paused_for_session: Optional[str] = None  # track session to avoid log spam

        while self.is_running:
            try:
                # Full pause outside tradable sessions. Sim and live both
                # stop analysing until the next trading session begins —
                # keeps the audit log honest (no afterhours "decisions"
                # during closed markets) and saves API quota.
                is_regular, session = self.is_market_hours()
                in_warmup = self._in_warmup_window()
                tradable = (
                    is_regular
                    or (session == "premarket" and self.allow_premarket)
                    or (session == "afterhours" and self.allow_afterhours)
                    or in_warmup  # pre-open analysis window — loop runs but
                                  # signal-router gate still blocks entries
                )
                if not tradable:
                    if self._paused_for_session != session:
                        next_open = self._get_next_market_open()
                        self.commentary.add_commentary(TradingCommentary(
                            timestamp=datetime.now(),
                            type=CommentaryType.MARKET_ANALYSIS,
                            symbol=None,
                            title=f"🌙 Market {session.title()} — Bot Paused",
                            message=(f"Markets are currently {session}. "
                                     f"Pausing analysis until the next regular "
                                     f"session opens at {next_open}."),
                            data={"session": session, "next_open": next_open,
                                  "allow_premarket": self.allow_premarket,
                                  "allow_afterhours": self.allow_afterhours},
                            importance=4,
                        ))
                        self._audit("market_hours", None, "pause",
                                    f"session_{session}", next_open=next_open)
                        self._paused_for_session = session
                    # Still refresh displayed state so the dashboard shows
                    # live quotes, P&L, and session metadata while paused.
                    # Read-only quote fetches only — no strategy eval, no orders.
                    try:
                        await self._update_position_prices()
                    except Exception as exc:
                        logger.debug("paused_refresh_error err=%s", exc)
                    # Recheck every 60s so we pick up the next session
                    # promptly; also keeps the websocket alive.
                    await asyncio.sleep(60)
                    continue

                if self._paused_for_session is not None:
                    # Waking back up — announce and clear the flag
                    if in_warmup:
                        title = "🔎 Pre-Market Warmup — Scanning Watchlist"
                        msg = (f"{Config().WARMUP_MINUTES_BEFORE_OPEN} minutes "
                               "before regular open. Building indicator state "
                               "and ranking candidates. New entries still blocked "
                               "until 09:30 ET.")
                        resume_reason = "warmup"
                    else:
                        title = "☀️ Market Open — Resuming Analysis"
                        msg = f"Entering {session} session. Bot is active."
                        resume_reason = f"session_{session}"
                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.MARKET_ANALYSIS,
                        symbol=None,
                        title=title,
                        message=msg,
                        importance=6,
                    ))
                    self._audit("market_hours", None, "resume", resume_reason)
                    self._paused_for_session = None

                analysis_count += 1
                
                # Sync with Schwab account data periodically (every 5 analyses)
                if analysis_count % 5 == 0 and self.schwab_client:
                    try:
                        # Sync account info
                        account_info = await self._get_real_account_info()
                        if account_info:
                            self.risk_manager.sync_with_schwab_data(account_info)
                            
                            # Log the sync to commentary
                            self.commentary.add_commentary(TradingCommentary(
                                timestamp=datetime.now(),
                                type=CommentaryType.ACCOUNT_UPDATE,
                                symbol=None,
                                title="📊 Account Sync",
                                message=f"Synced with Schwab: Daily P&L=${account_info.get('day_pnl', 0):.2f}",
                                data={'schwab_pnl': account_info.get('day_pnl', 0),
                                      'buying_power': account_info.get('buying_power', 0)},
                                importance=4
                            ))
                        
                        # Also sync positions every 5 analyses
                        await self.sync_positions_with_schwab()
                        
                    except Exception as e:
                        logger.error(f"Failed to sync with Schwab: {e}")
                
                # Update position prices before analysis
                await self._update_position_prices()
                
                # Risk assessment with detailed explanation
                await self._assess_market_conditions()
                await self._analyze_premarket_gaps() 
                # Check if we should trade (with explanation)
                should_trade, reason = await self._evaluate_trading_conditions()
                
                if should_trade:
                    # Analyze markets with commentary
                    await self._analyze_markets_with_commentary()
                else:
                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.RISK_ASSESSMENT,
                        symbol=None,
                        title="🛑 Trading Paused",
                        message=f"Not trading right now: {reason}",
                        importance=6
                    ))
                
                # Manage existing positions with commentary
                await self._manage_positions_with_commentary()
                
                # Emergency stop check - ONLY use Schwab P&L
                # Get fresh P&L from Schwab for accuracy
                schwab_pnl = 0
                if self.schwab_client and analysis_count % 5 == 0:  # Check every 5 cycles
                    fresh_info = await self._get_real_account_info()
                    if fresh_info:
                        schwab_pnl = fresh_info.get('day_pnl', 0)
                        # Update risk manager with Schwab data
                        self.risk_manager.schwab_daily_pnl = schwab_pnl
                        logger.info(f"Schwab P&L Check: ${schwab_pnl:.2f} (Limit: ${self.risk_manager.account_balance * 0.05:.2f})")
                
                # Only check emergency stop with real Schwab P&L
                if schwab_pnl < 0 and abs(schwab_pnl) > self.risk_manager.account_balance * 0.05:
                    # Double-check with fresh Schwab data before closing positions
                    if self.schwab_client:
                        fresh_account_info = await self._get_real_account_info()
                        if fresh_account_info:
                            fresh_pnl = fresh_account_info.get('day_pnl', 0)
                            logger.warning(f"Emergency stop check - Fresh P&L: ${fresh_pnl:.2f}")
                            
                            # If fresh data shows we're not in loss, abort emergency stop
                            if fresh_pnl >= 0:
                                logger.warning(f"Emergency stop ABORTED - Fresh data shows profit: ${fresh_pnl:.2f}")
                                self.commentary.add_commentary(TradingCommentary(
                                    timestamp=datetime.now(),
                                    type=CommentaryType.INFO,
                                    symbol=None,
                                    title="✅ False Alarm",
                                    message=f"Emergency stop cancelled - Account is actually in profit: ${fresh_pnl:.2f}",
                                    importance=7
                                ))
                                continue  # Skip emergency stop
                    
                    # Check if manual close only is enabled
                    if self.manual_close_only:
                        self.commentary.add_commentary(TradingCommentary(
                            timestamp=datetime.now(),
                            type=CommentaryType.WARNING,
                            symbol=None,
                            title="🚨 EMERGENCY STOP (Manual Mode)",
                            message=f"Daily loss of ${abs(pnl_to_check):.2f} exceeded 5% limit (${self.risk_manager.account_balance * 0.05:.2f}). "
                                   f"Manual close only is ON - YOU must close positions manually!",
                            importance=10
                        ))
                        
                        # Stop trading but DON'T close positions
                        self.is_running = False
                        break
                    else:
                        self.commentary.add_commentary(TradingCommentary(
                            timestamp=datetime.now(),
                            type=CommentaryType.WARNING,
                            symbol=None,
                            title="🚨 EMERGENCY STOP",
                            message=f"Daily loss of ${abs(pnl_to_check):.2f} exceeded 5% limit (${self.risk_manager.account_balance * 0.05:.2f}). Closing all positions.",
                            importance=10
                        ))
                        
                        # Close all positions
                        for symbol in list(self.positions.keys()):
                            position = self.positions[symbol]
                            await self._close_position_with_commentary(position, "emergency_stop")
                    
                    self.is_running = False
                    break
                # Check order status for real trades
                if self.mode == TradingMode.LIVE:
                    await self._check_order_status()
                    await self._update_real_positions()
                    if (datetime.now() - self._last_bp_check).total_seconds() > 300:
                        await self._check_buying_power(0)
                
                # After the position management section - retrain ML model periodically
                if analysis_count % 100 == 0:  # Every 100 cycles
                    # ScalpingMLModel doesn't have async retrain_model, just train()
                    # The training happens automatically via update_with_outcome
                    logger.info(f"ML model analysis cycle {analysis_count} - model is self-retraining via online learning")

		        # Save state periodically
                if analysis_count % 10 == 0:  # Every 10 analysis cycles
                    self._save_state()
                    self.brain.save_memories()

                    # Log performance snapshot for analytics
                    positions_count = len(self.positions) + len(self.simulated_positions)
                    total_unrealized = sum(getattr(p, 'unrealized_pnl', 0) for p in self.positions.values())
                    total_unrealized += sum(getattr(p, 'unrealized_pnl', 0) for p in self.simulated_positions.values())

                    log_performance(
                        account_balance=self.risk_manager.account_balance,
                        daily_pnl=self.risk_manager.schwab_daily_pnl if hasattr(self.risk_manager, 'schwab_daily_pnl') else 0,
                        total_pnl=total_unrealized,
                        win_rate=self.performance_analyzer.get_win_rate() if hasattr(self, 'performance_analyzer') else 0,
                        total_trades=len(self.trade_history),
                        open_positions=positions_count,
                        mode=self.mode.value,
                        unrealized_pnl=total_unrealized,
                        buying_power=self.risk_manager.buying_power
                    )

                    # Also save ML model periodically to preserve buffer
                    if hasattr(self, 'ml_predictor') and self.ml_predictor:
                        if hasattr(self.ml_predictor, 'save_model'):
                            self.ml_predictor.save_model(create_version=False)  # Don't create version every time
                # Wait before next analysis
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.WARNING,
                    symbol=None,
                    title="❌ Error in Trading Loop",
                    message=f"Encountered an error: {str(e)}",
                    importance=9
                ))
                logger.error(f"Trading loop error: {e}", exc_info=True)
                await asyncio.sleep(60)
    
    async def _update_position_prices(self):
        """Update current prices for all positions"""
        try:
            if self.mode == TradingMode.SIMULATION_WITH_COMMENTARY:
                # Update simulated positions
                for symbol, position in self.simulated_positions.items():
                    # Skip positions that are being closed or partially-
                    # constructed (zero quantity or zero entry price).
                    # Overwriting their stored P&L with a nonsense value
                    # can corrupt trading_state.json — especially now that
                    # this refresh also runs during the paused loop.
                    if not position or position.quantity == 0 or position.entry_price == 0:
                        continue
                    try:
                        # Get current quote
                        quote = await self.data_provider.get_current_quote(symbol)
                        if quote and 'price' in quote:
                            old_price = position.current_price
                            position.current_price = quote['price']
                            # Update unrealized PnL
                            position.unrealized_pnl = (position.current_price - position.entry_price) * position.quantity
                            # Log significant price movements (guard /0)
                            if old_price:
                                price_change = abs(position.current_price - old_price) / old_price
                                if price_change > 0.01:
                                    logger.debug(f"Updated {symbol} price: ${old_price:.2f} -> ${position.current_price:.2f}")
                    except Exception as e:
                        logger.debug(f"Error updating price for {symbol}: {e}")
            
            elif self.mode == TradingMode.LIVE and self.schwab_client:
                # Update real positions
                for symbol, position in self.positions.items():
                    if position:
                        try:
                            response = self.schwab_client.get_quote(symbol)
                            if response.status_code == 200:
                                quote_data = response.json()
                                if symbol in quote_data:
                                    old_price = position.current_price
                                    position.current_price = quote_data[symbol]['quote']['lastPrice']
                                    position.unrealized_pnl = (position.current_price - position.entry_price) * position.quantity
                                    
                                    # Log significant price movements
                                    price_change = abs(position.current_price - old_price) / old_price
                                    if price_change > 0.01:  # More than 1% change
                                        logger.debug(f"Updated {symbol} price: ${old_price:.2f} -> ${position.current_price:.2f}")
                        except Exception as e:
                            logger.debug(f"Error updating price for {symbol}: {e}")
                            
        except Exception as e:
            logger.error(f"Error in _update_position_prices: {e}")
    
    async def _assess_market_conditions(self):
        """Assess overall market conditions with detailed commentary"""
        self.commentary.add_commentary(TradingCommentary(
            timestamp=datetime.now(),
            type=CommentaryType.MARKET_ANALYSIS,
            symbol=None,
            title="🔍 Assessing Market Conditions",
            message="Let me check the overall market health before making any trading decisions...",
            importance=7
        ))
        
        # Get market breadth
        if self.data_provider:
            breadth = self.data_provider.calculate_market_breadth()
            self.market_state['breadth'] = breadth
            self.market_state['vix'] = breadth.get('vix', 20)
            
            # Interpret VIX
            vix_interpretation = ""
            vix_value = self.market_state['vix']
            if vix_value < 15:
                vix_interpretation = "Low volatility - Markets are calm, good for trend following"
            elif vix_value < 25:
                vix_interpretation = "Normal volatility - Standard trading conditions"
            elif vix_value < 35:
                vix_interpretation = "Elevated volatility - Higher risk, but also opportunities"
            else:
                vix_interpretation = "High volatility - Extreme caution needed, reduce position sizes"
            
            # Interpret breadth
            advance_decline = breadth.get('advance_decline', 0)
            breadth_interpretation = ""
            if advance_decline > 1.5:
                breadth_interpretation = "Strong bullish breadth - More stocks advancing"
                self.market_state['market_trend'] = 'bullish'
            elif advance_decline < -1.5:
                breadth_interpretation = "Strong bearish breadth - More stocks declining"
                self.market_state['market_trend'] = 'bearish'
            else:
                breadth_interpretation = "Neutral market breadth"
                self.market_state['market_trend'] = 'neutral'
            
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.MARKET_ANALYSIS,
                symbol=None,
                title="📊 Market Breadth Analysis",
                message=f"{vix_interpretation}\n{breadth_interpretation}",
                data={
                    'vix': vix_value,
                    'advance_decline_ratio': advance_decline,
                    'new_highs': breadth.get('new_highs', 0),
                    'new_lows': breadth.get('new_lows', 0),
                    'put_call_ratio': breadth.get('put_call_ratio', 1.0)
                },
                confidence=0.8,
                importance=8
            ))
    
    async def _analyze_premarket_gaps(self):
        """Analyze pre-market gaps for fade opportunities"""
        if 'gaps' not in self.market_state:
            self.market_state['gaps'] = {}
        
        for symbol in self.dynamic_watchlist:
            try:
                quote = self.data_provider.get_quote(symbol)
                if not quote:
                    continue
                    
                # Get today's open and yesterday's close
                open_price = quote.get('open', 0)
                prev_close = quote.get('close', 0)  # This is previous day's close
                
                # Skip if prices are invalid
                if open_price <= 0 or prev_close <= 0:
                    continue
                    
                gap_percent = ((open_price - prev_close) / prev_close) * 100
                
                if abs(gap_percent) > 2:
                    self.market_state['gaps'][symbol] = {
                        'percent': gap_percent,
                        'direction': 'up' if gap_percent > 0 else 'down',
                        'fade_probability': 0.65 if abs(gap_percent) > 5 else 0.45,
                        'open': open_price,
                        'prev_close': prev_close
                    }
                    
                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.MARKET_ANALYSIS,
                        symbol=symbol,
                        title=f"🌅 Gap Detected: {symbol}",
                        message=f"{gap_percent:.1f}% gap {self.market_state['gaps'][symbol]['direction']} (Open: ${open_price:.2f}, Prev Close: ${prev_close:.2f})",
                        importance=7
                    ))
            except Exception as e:
                logger.debug(f"Gap analysis error for {symbol}: {e}")

    async def _evaluate_trading_conditions(self) -> Tuple[bool, str]:
        """Evaluate if we should trade with detailed reasoning"""
        # In commentary mode, we always "trade" but don't execute real orders
        if self.mode == TradingMode.SIMULATION_WITH_COMMENTARY:
            if self.risk_manager.margin_call:
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.WARNING,
                    symbol=None,
                    title="⚠️ Margin Call Active",
                    message="Account has a margin call! In real trading, I would stop here. "
                            "But in commentary mode, I'll continue analyzing to show you my thought process.",
                    data={
                        'margin_call': True,
                        'buying_power': self.risk_manager.buying_power
                    },
                    importance=10
                ))
            return True, "Simulation mode - always analyze"
        
        if self._is_news_blackout():
            return False, "Major economic news event"
        # For other modes, check actual conditions
        allowed, reason = self.risk_manager.check_trading_allowed()
        
        if not allowed:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.RISK_ASSESSMENT,
                symbol=None,
                title="🚫 Trading Conditions Not Met",
                message=f"I'm not trading because: {reason}",
                data={
                    'daily_pnl': self.risk_manager.schwab_daily_pnl,  # Use Schwab P&L
                    'consecutive_losses': self.risk_manager.consecutive_losses,
                    'buying_power': self.risk_manager.buying_power
                },
                importance=8
            ))
        
        return allowed, reason
    
    async def _analyze_markets_with_commentary(self):
        """Analyze markets with detailed commentary"""
        # Run screener if it's time
        if self.screener and (not self.last_screener_run or 
                            (datetime.now() - self.last_screener_run).total_seconds() > 120):
            try:
                await self.screener.screen_stocks()
                
                # Update watchlist with top movers
                screener_symbols = self.screener.get_watchlist_symbols()
                if screener_symbols:
                    # Combine with default symbols, keeping unique
                    self.dynamic_watchlist = list(set(screener_symbols + ['NVDA', 'TSLA', 'PLTR']))[:10]
                    
                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.MARKET_ANALYSIS,
                        symbol=None,
                        title="📊 Watchlist Updated",
                        message=f"Now tracking: {', '.join(self.dynamic_watchlist)}",
                        importance=6
                    ))
                
                self.last_screener_run = datetime.now()
            except Exception as e:
                logger.error(f"Screener error: {e}")
        
        # Use dynamic watchlist
        watchlist = self.dynamic_watchlist

        # Early session guard: skip first 15 min after open (09:30-09:45 ET).
        # Indicators computed on <15 bars of data are unreliable — ATR is
        # microscopic, volume ratios are skewed by the opening auction.
        # The NIO 7,052-share oversizing was caused by ATR=$0.028 on 20 min of data.
        import pytz
        et = datetime.now(pytz.timezone("America/New_York"))
        market_open_time = et.replace(hour=9, minute=45, second=0, microsecond=0)
        if et < market_open_time and et.hour == 9 and et.minute >= 30:
            self._audit("early_session", None, "skip", "first_15min",
                        time=et.strftime("%H:%M"))
            return

        # Get current positions (both tracked and from Schwab)
        positions_held = set(self.positions.keys())
        
        # Add Schwab positions if in live mode
        if self.mode == TradingMode.LIVE and self.schwab_client:
            schwab_positions = await self.get_schwab_positions()
            for pos in schwab_positions:
                positions_held.add(pos['symbol'])
        
        # Filter watchlist to exclude symbols we already have positions in
        symbols_to_analyze = [s for s in watchlist if s not in positions_held]
        
        if not symbols_to_analyze:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.MARKET_ANALYSIS,
                symbol=None,
                title="📊 All Watchlist Symbols Have Positions",
                message=f"Currently holding positions in all watchlist symbols: {', '.join(positions_held)}",
                importance=5
            ))
            return
        
        self.commentary.add_commentary(TradingCommentary(
            timestamp=datetime.now(),
            type=CommentaryType.MARKET_ANALYSIS,
            symbol=None,
            title="📈 Scanning Watchlist",
            message=f"Analyzing {len(symbols_to_analyze)} symbols for trading opportunities. "
                f"Market trend: {self.market_state['market_trend']}\n"
                f"Skipping {', '.join(positions_held)} (already have positions)",
            importance=6
        ))
        
        for symbol in symbols_to_analyze:
            try:
                # Rate limiting check
                await asyncio.sleep(0.5)
                
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.MARKET_ANALYSIS,
                    symbol=symbol,
                    title=f"🔎 Analyzing {symbol}",
                    message="Fetching price data and calculating technical indicators...",
                    importance=4
                ))
                
                # Get market data
                if self.data_provider:
                    data = self.data_provider.get_market_data(symbol)
                    if data.empty:
                        self.commentary.add_commentary(TradingCommentary(
                            timestamp=datetime.now(),
                            type=CommentaryType.WARNING,
                            symbol=symbol,
                            title=f"⚠️ No Data: {symbol}",
                            message="Could not retrieve price data for this symbol",
                            importance=5
                        ))
                        continue
                    
                    # Get quote
                    quote = self.data_provider.get_quote(symbol)
                    if not quote:
                        quote = {'last': data['Close'].iloc[-1]}

                    # Minimum price filter — penny stocks have spreads
                    # wider than their ATR, making stops meaningless.
                    # HQGE ($0.019) cost -$4,681 across 49 repeat trades.
                    current_quote_price = float(quote.get('last', data['Close'].iloc[-1]))
                    if current_quote_price < 5.0:
                        self._audit("price_filter", symbol, "skip",
                                    "below_min_price", price=round(current_quote_price, 4))
                        continue

                    # Technical analysis with commentary
                    indicators = await self.technical_analyzer.analyze_with_commentary(data, symbol)
                    
                    # Create market data object with proper type conversion
                    market_data = MarketData(
                        symbol=symbol,
                        timestamp=datetime.now(),
                        open=float(data['Open'].iloc[-1]),
                        high=float(data['High'].iloc[-1]),
                        low=float(data['Low'].iloc[-1]),
                        close=float(quote.get('last', data['Close'].iloc[-1])),
                        volume=int(quote.get('volume', data['Volume'].iloc[-1])),
                        timeframe='5min',
                        indicators=indicators
                    )
                    
                    # Validate market data
                    if np.isnan(market_data.close) or market_data.close <= 0:
                        self.commentary.add_commentary(TradingCommentary(
                            timestamp=datetime.now(),
                            type=CommentaryType.WARNING,
                            symbol=symbol,
                            title=f"⚠️ Invalid Price Data: {symbol}",
                            message=f"Price data is invalid (close=${market_data.close}), skipping analysis",
                            importance=6
                        ))
                        continue
                    
                    ml_signal, ml_explanation = (0, {})
                    if Config().ML_PREDICTION_ENABLED:
                        ml_signal, ml_explanation = self.ml_predictor.predict_with_commentary(
                            indicators, symbol, data
                        )
                    
                    # Check each strategy
                    for strategy in self.strategies:
                        signal = await strategy.generate_signal_with_commentary(market_data)
                        
                        if signal:
                            # Process the signal with full explanation
                            await self._process_signal_with_commentary(signal, ml_signal, ml_explanation)
                
            except Exception as e:
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.WARNING,
                    symbol=symbol,
                    title=f"⚠️ Error Analyzing {symbol}",
                    message=f"Encountered error: {str(e)}",
                    importance=5
                ))
                logger.error(f"Error analyzing {symbol}: {e}", exc_info=True)
    
    def _shadow_meta_evaluate_sync(self, signal) -> dict | None:
        """Synchronous core — runs in a thread, returns result for async wrapper.

        Returns a dict with audit fields if evaluation succeeds, None otherwise.
        The async wrapper emits the _audit() call on the event loop so the
        DB write (which requires an async context) works correctly.
        """
        if self.meta_inference is None:
            return None
        strategy = signal.reasoning.get("strategy", "")
        if not strategy.startswith(self.meta_inference.primary_strategy):
            return None
        try:
            df = self.data_provider.get_market_data(
                signal.symbol, frequency_type="minute", frequency=5,
            )
            if df is None or df.empty or len(df) < 50:
                return None
            feat_df = self.ml_predictor.model.feature_extractor.batch_extract(df)
            if feat_df.empty:
                return None
            side = 1 if signal.signal_type == SignalType.BUY else -1
            features = feat_df.iloc[-1].to_dict()
            result = self.meta_inference.predict(features, side=side)
            return {
                "symbol": signal.symbol,
                "strategy": strategy,
                "side": side,
                "proba": round(result.win_probability, 4),
                "thr_065": result.would_trade_at_065,
                "thr_070": result.would_trade_at_070,
                "thr_075": result.would_trade_at_075,
            }
        except Exception as exc:
            logger.debug("meta_shadow_error symbol=%s err=%s", signal.symbol, exc)
            return None

    async def _shadow_meta_evaluate(self, signal) -> None:
        """Async wrapper ��� offloads sync body to a thread, emits audit on event loop."""
        if self.meta_inference is None:
            return
        try:
            result = await asyncio.to_thread(self._shadow_meta_evaluate_sync, signal)
            if result is not None:
                side_label = "long" if result["side"] == 1 else "short"
                self._audit(
                    "meta_shadow", result["symbol"], "evaluated",
                    f"{result['strategy']}_{side_label}",
                    proba=result["proba"],
                    thr_065=result["thr_065"],
                    thr_070=result["thr_070"],
                    thr_075=result["thr_075"],
                )
        except Exception as exc:
            logger.debug("meta_shadow_dispatch_error err=%s", exc)

    async def _process_signal_with_commentary(self, signal, ml_signal, ml_explanation):
        """Process trading signal with detailed explanation"""
        self._audit("signal_router", signal.symbol, "received",
                    signal.reasoning.get("strategy", "unknown"),
                    signal_type=signal.signal_type.value,
                    confidence=round(float(getattr(signal, "confidence", 0)), 3),
                    strength=round(float(getattr(signal, "strength", 0)), 3),
                    ml_signal=ml_signal)

        # Shadow meta-model evaluation — logs only, no decision impact.
        # Dispatched via asyncio.to_thread so the sync sklearn/xgb call
        # does not block the event loop.
        await self._shadow_meta_evaluate(signal)

        # Market-hours gate — applies to BOTH sim and live so simulation
        # accurately previews live behaviour (sim used to enter trades
        # 24/7 which gave misleading fills on afterhours noise).
        is_regular, session = self.is_market_hours()
        if not is_regular:
            if session == "premarket" and self.allow_premarket:
                pass
            elif session == "afterhours" and self.allow_afterhours:
                pass
            else:
                self._audit("market_hours", signal.symbol, "skip",
                            f"session_{session}",
                            allow_premarket=self.allow_premarket,
                            allow_afterhours=self.allow_afterhours)
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.WARNING,
                    symbol=signal.symbol,
                    title=f"🌙 Market {session.title()} — Skipping Entry",
                    message=(f"Signal received during {session} session. "
                             "New trades only during regular hours (09:30-16:00 ET). "
                             "Toggle trading.extended_hours.allow_* to override."),
                    importance=5,
                ))
                return

        # CRITICAL: Check real Schwab positions FIRST before internal tracking
        if self.mode == TradingMode.LIVE and self.schwab_client:
            schwab_positions = await self.get_schwab_positions()
            existing_position = next((pos for pos in schwab_positions if pos['symbol'] == signal.symbol), None)
            
            if existing_position:
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.WARNING,
                    symbol=signal.symbol,
                    title=f"🚫 Existing Schwab Position Found",
                    message=f"Found existing position: {existing_position['quantity']} shares at ${existing_position['average_price']:.2f}\n"
                        f"Current P&L: ${existing_position['total_pnl']:.2f} ({existing_position['pnl_percent']:.1f}%)",
                    data=existing_position,
                    importance=9
                ))
                
                # Add to bot tracking if not already tracked
                if signal.symbol not in self.positions:
                    position = Position(
                        symbol=signal.symbol,
                        entry_price=existing_position['average_price'],
                        current_price=existing_position['current_price'],
                        quantity=existing_position['quantity'],
                        side='long' if existing_position['quantity'] > 0 else 'short',
                        stop_loss=existing_position['average_price'] * (1 - Config().DEFAULT_STOP_LOSS_PCT),
                        take_profit=existing_position['average_price'] * (1 + Config().DEFAULT_TAKE_PROFIT_PCT),
                        entry_time=datetime.now(),
                        unrealized_pnl=existing_position['total_pnl'],
                        reasoning={'source': 'existing_schwab_position'}
                    )
                    self.positions[signal.symbol] = position
                    
                    # Initialize exit tracking
                    if hasattr(self, 'exit_manager'):
                        self.exit_manager.initialize_position_tracking(
                            signal.symbol,
                            position.entry_price,
                            position.stop_loss,
                            position.take_profit
                        )

                self._audit("signal_router", signal.symbol, "skip",
                            "existing_schwab_position",
                            qty=existing_position["quantity"],
                            avg=existing_position["average_price"])
                return

        # Check internal tracking (for simulation mode or as fallback)
        if signal.symbol in self.positions or signal.symbol in self.simulated_positions:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.DECISION,
                symbol=signal.symbol,
                title=f"📍 Already Tracking Position",
                message=f"Already tracking position in {signal.symbol}, skipping this signal.",
                importance=3
            ))
            self._audit("signal_router", signal.symbol, "skip", "already_tracking")
            return

        # Sector correlation guard — limit to 1 open position per
        # correlated group to prevent cluster stop-outs (e.g., MARA +
        # RIOT + COIN all dropping together when BTC falls).
        _CORRELATED_GROUPS = {
            "crypto_miners": {"MARA", "RIOT", "COIN", "CLSK", "BTBT", "CIFR", "WULF", "HUT"},
            "ev_makers": {"TSLA", "RIVN", "NIO", "LCID", "XPEV"},
            "china_tech": {"BABA", "JD", "PDD", "BIDU", "NIO", "XPEV"},
            "meme_retail": {"AMC", "GME", "BBBY", "FUBO"},
            "semiconductors": {"NVDA", "AMD", "INTC", "MU", "AVGO", "QCOM"},
        }
        active_positions = set(self.positions.keys()) | set(self.simulated_positions.keys())
        for group_name, members in _CORRELATED_GROUPS.items():
            if signal.symbol in members:
                overlap = active_positions & members
                if overlap:
                    self._audit("correlation_guard", signal.symbol, "skip",
                                f"correlated_with_{group_name}",
                                existing=list(overlap))
                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.RISK_ASSESSMENT,
                        symbol=signal.symbol,
                        title=f"🔗 Correlated Position Blocked",
                        message=(
                            f"Already holding {', '.join(overlap)} in the "
                            f"{group_name.replace('_', ' ')} group. "
                            f"Skipping {signal.symbol} to avoid cluster risk."
                        ),
                        importance=7,
                    ))
                    return

        # EARLY BUYING POWER CHECK for LIVE mode
        if self.mode == TradingMode.LIVE:
            # First check if we have ANY buying power before doing calculations
            min_buying_power = Config().MIN_BUYING_POWER or 100  # Default $100
            has_power, available_bp = await self._check_buying_power(min_buying_power)
            logger.info(f"Has buying power or not: {has_power} {available_bp}")
            if not has_power or (available_bp is not None and available_bp < min_buying_power):
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.WARNING,
                    symbol=signal.symbol,
                    title=f"❌ No Buying Power",
                    message=f"Cannot trade - buying power is ${available_bp or 0:.2f} (need at least ${min_buying_power})",
                    importance=9
                ))
                self._audit("risk", signal.symbol, "skip", "no_buying_power",
                            available_bp=round(float(available_bp or 0), 2),
                            min_required=min_buying_power)
                return
            
            # Update risk manager with current buying power
            self.risk_manager.buying_power = available_bp
            # Check for existing orders
            if self.mode == TradingMode.LIVE:
                existing_orders = await self._check_existing_orders(signal.symbol)
                if existing_orders['has_orders']:
                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.WARNING,
                        symbol=signal.symbol,
                        title=f"⚠️ Existing Orders Found",
                        message=f"Found {existing_orders['count']} existing orders for {signal.symbol}",
                        importance=7
                    ))
                    
                    # Cancel existing orders if configured
                    if Config().AUTO_CANCEL_EXISTING_ORDERS:
                        self._audit("signal_router", signal.symbol, "cancel_and_retry",
                                    "existing_orders",
                                    existing_count=existing_orders["count"])
                        await self._cancel_existing_orders(signal.symbol)
                    else:
                        self._audit("signal_router", signal.symbol, "skip",
                                    "existing_orders",
                                    existing_count=existing_orders["count"])
                        return
        # Check with brain if we should take this trade
        should_trade, brain_reason = self.brain.should_take_trade(
            signal.symbol, 
            signal.reasoning.get('strategy', 'unknown'),
            signal.confidence
        )
        if not should_trade:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.PSYCHOLOGY,
                symbol=signal.symbol,
                title=f"🧠 Experience Says No",
                message=brain_reason,
                data={
                    'pattern_history': self.brain.get_pattern_confidence(signal.reasoning.get('strategy', 'unknown')),
                    'emotional_state': self.brain.emotional_state
                },
                importance=7
            ))
            self._audit("brain", signal.symbol, "skip", "experience_says_no",
                        strategy=signal.reasoning.get("strategy", "unknown"),
                        brain_reason=brain_reason[:60].replace(" ", "_"))
            return
	
	# Add more human-like pre-trade thoughts
        pre_trade_thoughts = [
            f"Alright, {signal.symbol} is setting up nicely. Let me check the level 2...",
            f"Interesting setup on {signal.symbol}. This reminds me of last week's trade.",
            f"{signal.symbol} looking ready. Volume confirms - let's do this.",
            f"Been watching {signal.symbol} all morning. Finally getting the entry I wanted."
        ]
        
        self.commentary.add_commentary(TradingCommentary(
            timestamp=datetime.now(),
            type=CommentaryType.PSYCHOLOGY,
            symbol=signal.symbol,
            title=f"💭 Pre-Trade Thoughts",
            message=np.random.choice(pre_trade_thoughts),
            importance=5
        ))

        # Evaluate the signal
        self.commentary.add_commentary(TradingCommentary(
            timestamp=datetime.now(),
            type=CommentaryType.SIGNAL_GENERATION,
            symbol=signal.symbol,
            title=f"🎯 Trading Signal Generated!",
            message=f"Strategy '{signal.reasoning.get('strategy', 'unknown')}' found a potential {signal.signal_type.name} opportunity",
            data={
                'entry_price': signal.entry_price,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit,
                'risk_reward_ratio': (signal.take_profit - signal.entry_price) / (signal.entry_price - signal.stop_loss),
                'signal_strength': signal.strength,
                'ml_confirmation': ml_signal == 1
            },
            confidence=signal.confidence,
            importance=8
        ))



        # Check ML confirmation
        # Check ML confirmation (1 for BUY, -1 for SELL)
        expected_ml_signal = 1 if signal.signal_type == SignalType.BUY else -1
        if ml_signal != expected_ml_signal and Config().ML_PREDICTION_ENABLED:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.DECISION,
                symbol=signal.symbol,
                title=f"🤖 ML Model Disagrees",
                message="The machine learning model doesn't confirm this signal. "
                       "I typically wait for ML confirmation to increase win probability.",
                data=ml_explanation,
                importance=7
            ))

            # Log the decision to skip
            log_decision(
                symbol=signal.symbol,
                decision='SKIP',
                reason='ML model disagrees with strategy signal',
                factors={'ml_signal': ml_signal, 'expected': expected_ml_signal},
                signals={'strategy': signal.strength, 'ml': ml_signal},
                indicators=ml_explanation,
                confidence=signal.confidence,
                mode=self.mode.value
            )

            # Veto policy: when ML disagrees with high confidence, skip the trade
            # in ALL modes (not just live). Previously sim mode ignored the veto,
            # which let strategies enter against trained-model signals (LCID 2026-04-14).
            ml_confidence = float(ml_explanation.get("confidence", 0.0)) if isinstance(ml_explanation, dict) else 0.0
            ml_veto_threshold = Config().ML_VETO_CONFIDENCE
            if ml_confidence >= ml_veto_threshold:
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.DECISION,
                    symbol=signal.symbol,
                    title=f"🛑 ML Veto",
                    message=f"Skipping trade — ML confidence {ml_confidence:.0%} ≥ veto threshold "
                            f"{ml_veto_threshold:.0%}. The model strongly disagrees with the strategy.",
                    importance=8
                ))
                self._audit("ml", signal.symbol, "skip", "ml_veto_high_confidence",
                            ml_signal=ml_signal, ml_conf=round(ml_confidence, 3),
                            threshold=round(ml_veto_threshold, 3))
                return

            # Treat sim and live identically so simulation previews match
            # what live would do. Previously sim took the trade anyway after
            # a sub-veto-threshold ML disagreement, which made simulation a
            # poor predictor of live behaviour.
            self._audit("ml", signal.symbol, "skip", "ml_disagreement",
                        ml_signal=ml_signal, ml_conf=round(ml_confidence, 3),
                        expected=expected_ml_signal)
            return
        # Multi-timeframe confirmation
        try:
            daily_data = self.data_provider.get_market_data(signal.symbol, frequency_type='daily', frequency=1)
            if not daily_data.empty and len(daily_data) > 20:
                daily_trend = 'up' if daily_data['Close'].iloc[-1] > daily_data['Close'].iloc[-20] else 'down'
                
                if (signal.signal_type == SignalType.BUY and daily_trend == 'down') or \
                (signal.signal_type == SignalType.SELL and daily_trend == 'up'):
                    signal.confidence *= 0.7
                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.RISK_ASSESSMENT,
                        symbol=signal.symbol,
                        title=f"⚠️ Against Daily Trend",
                        message=f"Signal is against daily trend ({daily_trend}), reducing confidence",
                        importance=6
                    ))
        except Exception as e:
            logger.error(f"Multi-timeframe check failed: {e}")

        # ================================================================
        # PROFESSIONAL TRADING FILTER - Institutional Quality Gate
        # This is where we filter out low-quality setups that retail loses on
        # ================================================================
        if self.pro_trading_wrapper is not None:
            try:
                # Get market data for regime/context analysis
                market_df = None
                if self.data_provider:
                    raw_data = self.data_provider.get_market_data(signal.symbol)
                    if not raw_data.empty:
                        # Rename columns to lowercase for pro modules
                        market_df = raw_data.rename(columns={
                            'Open': 'open', 'High': 'high', 'Low': 'low',
                            'Close': 'close', 'Volume': 'volume'
                        })

                # Build additional signals for confluence
                additional_signals = {
                    'trend_aligned': ml_signal == (1 if signal.signal_type == SignalType.BUY else -1),
                    'ml_confirmed': ml_signal == (1 if signal.signal_type == SignalType.BUY else -1),
                    'volume_confirmed': signal.reasoning.get('volume_confirmation', False),
                    'momentum_confirmed': signal.reasoning.get('momentum_confirmed', False),
                }

                # Evaluate through professional filter
                pro_decision = self.pro_trading_wrapper.evaluate_trade_signal(
                    symbol=signal.symbol,
                    direction='long' if signal.signal_type == SignalType.BUY else 'short',
                    strategy=signal.reasoning.get('strategy', 'unknown'),
                    entry_price=signal.entry_price,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    position_size=signal.position_size if signal.position_size else 100,
                    confidence=signal.confidence,
                    market_data=market_df,
                    additional_signals=additional_signals
                )

                if not pro_decision.allowed:
                    # Trade blocked by professional filters
                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.DECISION,
                        symbol=signal.symbol,
                        title=f"🏛️ Professional Filter: BLOCKED",
                        message=pro_decision.final_recommendation,
                        data={
                            'reasons_against': pro_decision.reasons_against,
                            'regime': pro_decision.regime,
                            'session': pro_decision.session,
                            'quality_score': pro_decision.quality_score,
                            'confluence_score': pro_decision.confluence_score
                        },
                        importance=8
                    ))

                    # Log the blocked trade for analysis
                    log_decision(
                        symbol=signal.symbol,
                        decision='BLOCKED_BY_PRO_FILTER',
                        reason='; '.join(pro_decision.reasons_against[:3]),
                        factors={
                            'regime': pro_decision.regime,
                            'session': pro_decision.session,
                            'quality': pro_decision.quality_score,
                            'confluence': pro_decision.confluence_score
                        },
                        signals={'strategy': signal.strength, 'ml': ml_signal},
                        indicators=signal.reasoning.get('indicators', {}),
                        confidence=signal.confidence,
                        mode=self.mode.value
                    )
                    return  # Don't take this trade

                # Trade approved - use adjusted parameters from pro filter
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.DECISION,
                    symbol=signal.symbol,
                    title=f"🏛️ Professional Filter: APPROVED",
                    message=f"Quality: {pro_decision.quality_score:.0f}/100, "
                           f"Confluence: {pro_decision.confluence_score}, "
                           f"R-Target: {pro_decision.r_multiple_target:.1f}",
                    data={
                        'reasons_for': pro_decision.reasons_for,
                        'regime': pro_decision.regime,
                        'session': pro_decision.session,
                        'adjusted_stop': pro_decision.stop_loss,
                        'adjusted_size': pro_decision.position_size
                    },
                    importance=7
                ))

                # Apply adjustments from pro filter
                signal.stop_loss = pro_decision.stop_loss
                signal.position_size = int(pro_decision.position_size) if pro_decision.position_size else signal.position_size

            except Exception as e:
                logger.error(f"Professional filter error (continuing with trade): {e}")
                # On error, continue with original signal - fail open

        # Check portfolio concentration
        if self.positions:
            total_value = sum(p.current_price * p.quantity for p in self.positions.values()) 
            max_position_value = total_value * Config().MAX_POSITION_VALUE_PCT
            
            if signal.position_size * signal.entry_price > max_position_value:
                signal.position_size = int(max_position_value / signal.entry_price)
        # Calculate position size with explanation
        position_size = await self.risk_manager.calculate_position_size_with_commentary(
            signal, signal.entry_price
        )
        
        if position_size == 0:
            return
        
        signal.position_size = position_size

	# After creating position, initialize exit tracking
        if self.mode == TradingMode.SIMULATION_WITH_COMMENTARY:
            self.exit_manager.initialize_position_tracking(
                signal.symbol,
                signal.entry_price,
                signal.stop_loss,
                signal.take_profit
            )
        
        # Final decision
        self.commentary.add_commentary(TradingCommentary(
            timestamp=datetime.now(),
            type=CommentaryType.DECISION,
            symbol=signal.symbol,
            title=f"✅ Trade Decision: {signal.signal_type.name}",
            message=f"After careful analysis, I've decided to {'BUY' if signal.signal_type.value == 1 else 'SELL'} {position_size} shares of {signal.symbol}",
            data={
                'total_cost': position_size * signal.entry_price,
                'max_risk': position_size * (signal.entry_price - signal.stop_loss),
                'max_reward': position_size * (signal.take_profit - signal.entry_price),
                'reasoning': signal.reasoning
            },
            confidence=signal.confidence,
            importance=10
        ))
        self._audit("signal_router", signal.symbol, "accepted",
                    signal.reasoning.get("strategy", "unknown"),
                    side=signal.signal_type.name,
                    qty=position_size,
                    entry=round(signal.entry_price, 2),
                    stop=round(signal.stop_loss, 2),
                    target=round(signal.take_profit, 2),
                    total_cost=round(position_size * signal.entry_price, 2))
        
        # Execute the trade based on mode
        position = None
        if self.mode == TradingMode.SIMULATION_WITH_COMMENTARY:
            # Create simulated position
            position = Position(
                symbol=signal.symbol,
                entry_price=signal.entry_price,
                current_price=signal.entry_price,
                quantity=signal.position_size,
                side='long' if signal.signal_type == SignalType.BUY else 'short',
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                entry_time=datetime.now(),
                reasoning=signal.reasoning,
                original_stop=signal.stop_loss,
            )
            self.simulated_positions[signal.symbol] = position
            
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.DECISION,
                symbol=signal.symbol,
                title=f"📋 Simulated Position Opened",
                message=f"Position tracked in simulation. I'll monitor it and explain my management decisions.",
                importance=7
            ))
            
        elif self.mode == TradingMode.LIVE:
            # Execute real trade
            success = await self._execute_real_trade(signal)
            if success:
                # Don't create position here - wait for order fill confirmation
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.DECISION,
                    symbol=signal.symbol,
                    title=f"📋 Order Placed Successfully",
                    message=f"Waiting for order fill confirmation before tracking position",
                    importance=8
                ))
            else:
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.WARNING,
                    symbol=signal.symbol,
                    title=f"❌ Trade Execution Failed",
                    message=f"Could not execute trade for {signal.symbol}",
                    importance=10
                ))
        # Only initialize exit tracking if position was created
        if position and hasattr(self, 'exit_manager'):
            self.exit_manager.initialize_position_tracking(
                signal.symbol,
                signal.entry_price,
                signal.stop_loss,
                signal.take_profit
            )

        # Register position with professional exit manager for R-multiple tracking
        if position and self.pro_trading_wrapper is not None:
            try:
                # Get ATR for the symbol
                atr = None
                if self.data_provider:
                    raw_data = self.data_provider.get_market_data(signal.symbol)
                    if not raw_data.empty and len(raw_data) >= 14:
                        high = raw_data['High'].values
                        low = raw_data['Low'].values
                        close = raw_data['Close'].values
                        tr = np.maximum(
                            high[1:] - low[1:],
                            np.maximum(
                                np.abs(high[1:] - close[:-1]),
                                np.abs(low[1:] - close[:-1])
                            )
                        )
                        atr = float(np.mean(tr[-14:]))

                # Open position in professional exit manager
                self.pro_trading_wrapper.exit_manager.open_position(
                    symbol=signal.symbol,
                    direction='long' if signal.signal_type == SignalType.BUY else 'short',
                    entry_price=signal.entry_price,
                    position_size=signal.position_size,
                    stop_loss=signal.stop_loss,
                    atr=atr
                )

                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.DECISION,
                    symbol=signal.symbol,
                    title=f"🏛️ Pro Exit Manager: Tracking",
                    message=f"Position registered for R-multiple management. "
                           f"Scale-out targets: 1R (50%), 2R (25%), 3R (25%)",
                    importance=6
                ))
            except Exception as e:
                logger.error(f"Failed to register position with pro exit manager: {e}")

    async def close_position_manually(self, symbol: str, position_type: str):
        """Manually close a position"""
        position = None
        if position_type == 'real':
            position = self.positions.get(symbol)
        elif position_type == 'simulated':
            position = self.simulated_positions.get(symbol)

        if position:
            await self._close_position_with_commentary(position, "manual_override")
   
    async def _manage_positions_with_commentary(self):
        """Manage positions with detailed commentary"""
        positions_to_check = self.simulated_positions if self.mode == TradingMode.SIMULATION_WITH_COMMENTARY else self.positions
        
        if not positions_to_check:
            return
        
        for symbol, position in list(positions_to_check.items()):
            try:
                # Check if position is None
                if position is None:
                    logger.warning(f"None position found for {symbol}, removing")
                    del positions_to_check[symbol]
                    continue

                # CRITICAL: Skip external/manually managed positions when manual_close_only is ON
                # These positions should only be managed by the user, not the bot
                is_external = getattr(position, 'is_external', False)
                is_manually_managed = getattr(position, 'is_manually_managed', False)

                if (self.auto_close_disabled or (self.manual_close_only and (is_external or is_manually_managed))):
                    self._audit("position_manager", symbol, "hold",
                                "auto_close_disabled",
                                external=is_external, manually_managed=is_manually_managed)
                    # Only update the price for display purposes, but don't take any action
                    if self.data_provider:
                        quote = self.data_provider.get_quote(symbol)
                        if quote:
                            position.current_price = quote.get('last', position.current_price)
                            if position.side == 'short':
                                position.unrealized_pnl = (position.entry_price - position.current_price) * position.quantity
                            else:
                                position.unrealized_pnl = (position.current_price - position.entry_price) * position.quantity
                    continue  # Skip all automatic management for this position

                # Get current price - always use real data when available
                if self.data_provider:
                    quote = self.data_provider.get_quote(symbol)
                    if quote and quote.get('last'):
                        current_price = quote.get('last')
                    else:
                        # Fallback: keep current price if real data unavailable
                        logger.warning(f"Could not get real price for {symbol}, using last known price")
                        current_price = position.current_price
                    
                    # Update position
                    old_price = position.current_price
                    position.current_price = current_price
                    if position.side == 'short':
                        position.unrealized_pnl = (position.entry_price - current_price) * position.quantity
                    else:  # long
                        position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
                    
                    # Calculate P&L percentage
                    if position.side == 'short':
                        pnl_pct = ((position.entry_price - current_price) / position.entry_price) * 100
                    else:
                        pnl_pct = ((current_price - position.entry_price) / position.entry_price) * 100
                    
                    # Alert on significant losses
                    if position.unrealized_pnl < 0:
                        loss_pct = abs(pnl_pct)
                        
                        # Check if we should alert (avoid spamming)
                        should_alert = False
                        alert_info = self.loss_alert_tracker.get(symbol, {})
                        last_alert_time = alert_info.get('last_alert_time')
                        last_loss_pct = alert_info.get('last_loss_pct', 0)
                        
                        # Alert if: first time, or loss increased by 0.5%, or 5 minutes passed
                        if (not last_alert_time or 
                            loss_pct >= last_loss_pct + 0.5 or
                            (datetime.now() - last_alert_time).total_seconds() > 300):
                            should_alert = True
                        
                        if should_alert:
                            # Critical alert for large losses
                            if loss_pct >= 3.0:  # 3% or more loss
                                self.commentary.add_commentary(TradingCommentary(
                                    timestamp=datetime.now(),
                                    type=CommentaryType.WARNING,
                                    symbol=symbol,
                                    title=f"🚨 CRITICAL LOSS: {symbol}",
                                    message=f"Position down {loss_pct:.1f}% (${abs(position.unrealized_pnl):.2f})\n"
                                            f"IMMEDIATE ACTION REQUIRED!\n"
                                            f"Stop loss: ${position.stop_loss:.2f}",
                                    data={
                                        'current_price': current_price,
                                        'entry_price': position.entry_price,
                                        'unrealized_pnl': position.unrealized_pnl,
                                        'pnl_percentage': pnl_pct,
                                        'distance_to_stop': abs((current_price - position.stop_loss) / current_price * 100)
                                    },
                                    importance=10
                                ))
                            elif loss_pct >= 1.0:  # 1% or more loss
                                self.commentary.add_commentary(TradingCommentary(
                                    timestamp=datetime.now(),
                                    type=CommentaryType.WARNING,
                                    symbol=symbol,
                                    title=f"⚠️ Position Losing: {symbol}",
                                    message=f"Down {loss_pct:.1f}% (${abs(position.unrealized_pnl):.2f})\n"
                                            f"Stop loss at ${position.stop_loss:.2f}",
                                    data={
                                        'current_price': current_price,
                                        'entry_price': position.entry_price,
                                        'unrealized_pnl': position.unrealized_pnl,
                                        'pnl_percentage': pnl_pct,
                                        'distance_to_stop': abs((current_price - position.stop_loss) / current_price * 100)
                                    },
                                    importance=7 if loss_pct < 2 else 9
                                ))
                            
                            # Update tracker
                            self.loss_alert_tracker[symbol] = {
                                'last_alert_time': datetime.now(),
                                'last_loss_pct': loss_pct
                            }
                    
                    # Regular position updates for smaller moves
                    price_change_pct = ((current_price - old_price) / old_price) * 100
                    if abs(price_change_pct) > 0.05 and abs(pnl_pct) >= 0.5:  # Lower threshold
                        self.commentary.add_commentary(TradingCommentary(
                            timestamp=datetime.now(),
                            type=CommentaryType.MARKET_ANALYSIS,
                            symbol=symbol,
                            title=f"📊 Position Update: {symbol}",
                            message=f"Position {'profitable' if position.unrealized_pnl > 0 else 'losing'} "
                                f"by {abs(pnl_pct):.1f}%",
                            data={
                                'current_price': current_price,
                                'entry_price': position.entry_price,
                                'unrealized_pnl': position.unrealized_pnl,
                                'pnl_percentage': pnl_pct,
                                'distance_to_stop': ((current_price - position.stop_loss) / current_price) * 100,
                                'distance_to_target': ((position.take_profit - current_price) / current_price) * 100
                            },
                            importance=4
                        ))
                    
                    # ================================================================
                    # ================================================================
                    # ATR CALCULATION (used by scale-trail + pro exit manager)
                    # ================================================================
                    current_atr = None
                    market_df = None
                    if self.data_provider and not getattr(position, 'is_long_term', False):
                        try:
                            raw_data = self.data_provider.get_market_data(symbol)
                            if not raw_data.empty and len(raw_data) >= 14:
                                high = raw_data['High'].values
                                low = raw_data['Low'].values
                                close = raw_data['Close'].values
                                tr = np.maximum(
                                    high[1:] - low[1:],
                                    np.maximum(
                                        np.abs(high[1:] - close[:-1]),
                                        np.abs(low[1:] - close[:-1])
                                    )
                                )
                                current_atr = float(np.mean(tr[-14:]))
                                market_df = raw_data.rename(columns={
                                    'Open': 'open', 'High': 'high', 'Low': 'low',
                                    'Close': 'close', 'Volume': 'volume'
                                })
                        except Exception as e:
                            logger.debug(f"ATR calc error for {symbol}: {e}")

                    # ================================================================
                    # SCALE-OUT AT 1R + ATR TRAILING STOP
                    # ================================================================
                    if not getattr(position, 'is_long_term', False) and not self.auto_close_disabled:
                        # --- 1R Partial Exit ---
                        partial = self.scale_trail.check_partial_exit(position, current_price)
                        if partial is not None and partial.exit_qty > 0:
                            position.quantity -= partial.exit_qty
                            position.scaled_out = True
                            position.stop_loss = partial.new_stop
                            self._audit("scale_out", symbol, "partial_exit_1R",
                                        "1R_reached",
                                        exit_qty=partial.exit_qty,
                                        remaining=position.quantity,
                                        new_stop=round(partial.new_stop, 2),
                                        pnl_locked=round(partial.exit_qty * abs(current_price - position.entry_price), 2))
                            self.commentary.add_commentary(TradingCommentary(
                                timestamp=datetime.now(),
                                type=CommentaryType.DECISION,
                                symbol=symbol,
                                title=f"🎯 1R Reached — Scaling Out 50%",
                                message=(
                                    f"Price hit 1R at ${current_price:.2f}. "
                                    f"Closed {partial.exit_qty} shares, keeping {position.quantity}. "
                                    f"Stop moved to breakeven at ${partial.new_stop:.2f}."
                                ),
                                importance=9,
                            ))

                        # --- ATR Trailing Stop ---
                        if current_atr is not None:
                            new_trail = self.scale_trail.update_trailing_stop(
                                position, current_price, current_atr,
                            )
                            if new_trail is not None:
                                old_trail = position.trailing_stop
                                position.trailing_stop = new_trail
                                self._audit("trailing_stop", symbol, "update",
                                            "atr_trail",
                                            old=round(old_trail or 0, 2),
                                            new=round(new_trail, 2),
                                            atr=round(current_atr, 3))

                            if self.scale_trail.is_trailing_stop_hit(position, current_price):
                                self._audit("trailing_stop", symbol, "exit",
                                            "atr_trail_hit",
                                            price=round(current_price, 2),
                                            trail=round(position.trailing_stop, 2))
                                self.commentary.add_commentary(TradingCommentary(
                                    timestamp=datetime.now(),
                                    type=CommentaryType.DECISION,
                                    symbol=symbol,
                                    title=f"📉 Trailing Stop Hit",
                                    message=(
                                        f"ATR trail at ${position.trailing_stop:.2f} breached "
                                        f"at ${current_price:.2f}. Locking in gains."
                                    ),
                                    importance=9,
                                ))
                                await self._close_position_with_commentary(
                                    position, "trailing_stop_atr"
                                )
                                continue

                    # ================================================================
                    # PROFESSIONAL EXIT MANAGER
                    # ================================================================
                    if self.pro_trading_wrapper is not None and not getattr(position, 'is_long_term', False):
                        try:

                            # Check for exit signals from professional exit manager
                            pro_exit_signals = self.pro_trading_wrapper.update_positions(
                                current_prices={symbol: current_price},
                                market_data={symbol: market_df} if market_df is not None else None,
                                atrs={symbol: current_atr} if current_atr else None
                            )

                            # Process any exit signals
                            for exit_sig in pro_exit_signals:
                                if exit_sig.symbol == symbol:
                                    self.commentary.add_commentary(TradingCommentary(
                                        timestamp=datetime.now(),
                                        type=CommentaryType.DECISION,
                                        symbol=symbol,
                                        title=f"🏛️ Pro Exit: {exit_sig.exit_reason.value.upper()}",
                                        message=f"{exit_sig.message}\n"
                                               f"Exiting {exit_sig.exit_size_pct:.0%} @ ${exit_sig.exit_price:.2f}",
                                        data={
                                            'r_multiple': exit_sig.r_multiple,
                                            'exit_reason': exit_sig.exit_reason.value,
                                            'exit_size_pct': exit_sig.exit_size_pct
                                        },
                                        importance=9
                                    ))

                                    # Handle partial exit vs full exit
                                    if exit_sig.exit_size_pct < 1.0:
                                        # Partial exit - scale out
                                        exit_quantity = int(position.quantity * exit_sig.exit_size_pct)
                                        if exit_quantity > 0 and not self.auto_close_disabled:
                                            # Execute partial close
                                            position.quantity -= exit_quantity
                                            self.commentary.add_commentary(TradingCommentary(
                                                timestamp=datetime.now(),
                                                type=CommentaryType.DECISION,
                                                symbol=symbol,
                                                title=f"📊 Scaled Out {exit_sig.exit_size_pct:.0%}",
                                                message=f"Locked in profit at {exit_sig.r_multiple:.1f}R. "
                                                       f"Remaining: {position.quantity} shares",
                                                importance=8
                                            ))
                                    else:
                                        # Full exit
                                        if not self.auto_close_disabled:
                                            await self._close_position_with_commentary(
                                                position,
                                                f"pro_exit_{exit_sig.exit_reason.value}"
                                            )
                                            continue  # Position closed, move to next

                        except Exception as e:
                            logger.error(f"Professional exit manager error for {symbol}: {e}")
                            # Continue with normal exit logic on error

                    # Check exit conditions with reasoning (skip if long-term)
                    if not getattr(position, 'is_long_term', False):
                        should_exit, exit_reason = await self._evaluate_exit_conditions(position, current_price)
                        
                        if should_exit:
                            # Check if manual close only is enabled
                            if self.auto_close_disabled:
                                self.commentary.add_commentary(TradingCommentary(
                                    timestamp=datetime.now(),
                                    type=CommentaryType.INFO,
                                    symbol=position.symbol,
                                    title=f"📍 Exit Signal (Manual Mode)",
                                    message=f"Exit condition met ({exit_reason}) but manual close only is ON. "
                                           f"Current P&L: ${position.unrealized_pnl:.2f}. YOU must close manually.",
                                    importance=7
                                ))
                            else:
                                await self._close_position_with_commentary(position, exit_reason)
                    else:
                        # Log that we're skipping long-term position
                        if not hasattr(self, '_long_term_logged') or symbol not in self._long_term_logged:
                            if not hasattr(self, '_long_term_logged'):
                                self._long_term_logged = set()
                            self._long_term_logged.add(symbol)
                            
                            self.commentary.add_commentary(TradingCommentary(
                                timestamp=datetime.now(),
                                type=CommentaryType.DECISION,
                                symbol=symbol,
                                title=f"🔒 Long-Term Hold Protected",
                                message=f"{symbol} is marked as long-term hold - automatic exit conditions disabled",
                                importance=5
                            ))
                
            except Exception as e:
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.WARNING,
                    symbol=symbol,
                    title=f"⚠️ Error Managing {symbol}",
                    message=str(e),
                    importance=6
                ))
    async def _evaluate_exit_conditions(self, position, current_price: float) -> Tuple[bool, str]:
        """Evaluate exit conditions with detailed reasoning"""
        # CRITICAL: Never auto-exit external/manually managed positions
        is_external = getattr(position, 'is_external', False)
        is_manually_managed = getattr(position, 'is_manually_managed', False)

        if is_external or is_manually_managed:
            # External positions are NEVER auto-managed
            return False, ""

        # Check if stop loss / take profit are hit based on position side.
        # Both use the same simple price comparison — no indicator logic.
        # The dynamic exit manager below still runs for softer/partial exits
        # but hard SL/TP must fire regardless of regime.
        stop_loss_hit = False
        take_profit_hit = False
        if position.side == 'short':
            stop_loss_hit = current_price >= position.stop_loss
            take_profit_hit = (
                position.take_profit is not None
                and current_price <= position.take_profit
            )
        else:  # long
            stop_loss_hit = current_price <= position.stop_loss
            take_profit_hit = (
                position.take_profit is not None
                and current_price >= position.take_profit
            )

        # If manual close only mode, only check hard stops
        if self.auto_close_disabled:
            # Still check stop loss for safety
            if stop_loss_hit:
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.WARNING,
                    symbol=position.symbol,
                    title=f"⚠️ Stop Loss Triggered",
                    message=f"Price hit stop loss at ${current_price:.2f}. Manual close required.",
                    importance=9
                ))
                # Don't auto-close, just warn
                return False, ""
            return False, ""
        # First check hard stops
        if stop_loss_hit:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.RISK_ASSESSMENT,
                symbol=position.symbol,
                title=f"🛑 Stop Loss Hit",
                message=f"Stopped out at ${current_price:.2f}. Part of the game - on to the next one.",
                importance=9
            ))
            return True, "stop_loss"

        # Hard take-profit check — fires on the next bar whose price
        # reaches the target, independent of indicators. Previous code
        # let the dynamic exit manager swallow this case and sometimes
        # held past the target.
        if take_profit_hit:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.DECISION,
                symbol=position.symbol,
                title=f"🎯 Take-Profit Hit",
                message=(
                    f"Target reached at ${current_price:.2f} "
                    f"(target ${position.take_profit:.2f}). Closing."
                ),
                importance=9,
            ))
            return True, "take_profit"
        
        # Get current indicators for dynamic exit
        if self.data_provider:
            data = self.data_provider.get_market_data(position.symbol)
            if not data.empty:
                indicators = await self.technical_analyzer.analyze_with_commentary(data, position.symbol)
                
                # Use dynamic exit manager
                should_exit, reason, exit_portion = await self.exit_manager.evaluate_exit(
                    position, current_price, indicators
                )
                
                if should_exit:
                    if exit_portion < 1.0:
                        # Partial exit - reduce position
                        position.quantity = int(position.quantity * (1 - exit_portion))
                        self.commentary.add_commentary(TradingCommentary(
                            timestamp=datetime.now(),
                            type=CommentaryType.DECISION,
                            symbol=position.symbol,
                            title=f"📊 Partial Exit",
                            message=f"Taking {exit_portion*100:.0f}% off the table. Letting the rest ride.",
                            importance=8
                        ))
                        if position.quantity > 0:
                            return False, ""  # Keep remaining position
                    
                    return True, reason
        
        return False, ""
    
    async def _close_position_with_commentary(self, position, reason: str):
        """Close position with detailed commentary"""
        # CRITICAL: Never auto-close external/manually managed positions (except for explicit manual_override)
        is_external = getattr(position, 'is_external', False)
        is_manually_managed = getattr(position, 'is_manually_managed', False)

        if (is_external or is_manually_managed) and reason != "manual_override":
            logger.warning(f"Blocked auto-close of EXTERNAL position {position.symbol}. Reason: {reason}")
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.WARNING,
                symbol=position.symbol,
                title=f"🚫 External Position Protected",
                message=f"Cannot auto-close {position.symbol} - this is an EXTERNAL position.\n"
                       f"Current P&L: ${position.unrealized_pnl:.2f}\n"
                       f"Use Schwab directly or the manual close button.",
                importance=9
            ))
            return  # NEVER auto-close external positions

        # CRITICAL: Check if manual close only is enabled (except for manual_override)
        if self.auto_close_disabled and reason != "manual_override":
            self._audit("position_manager", position.symbol, "skip_close",
                        "auto_close_disabled",
                        requested_reason=reason,
                        pnl=round(float(getattr(position, "unrealized_pnl", 0)), 2))
            logger.warning(f"Attempted to auto-close {position.symbol} but manual_close_only is ON. Reason: {reason}")
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.WARNING,
                symbol=position.symbol,
                title=f"⚠️ Auto-Close Blocked",
                message=f"Attempted to close {position.symbol} ({reason}) but manual close only is ON. "
                       f"Current P&L: ${position.unrealized_pnl:.2f}. Use manual close button.",
                importance=8
            ))
            return  # Don't close the position
        
        # Debug log
        logger.info(f"Attempting to close position: {position.symbol}, mode: {self.mode}")
        # ADD CONFIRMATION HERE
        # Check if confirmation is needed
        if self.mode == TradingMode.LIVE and self.require_confirmations:
            # Calculate P&L percentage
            pnl = position.unrealized_pnl
            pnl_percent = abs((pnl / (position.entry_price * position.quantity)) * 100)
            
            # Determine if confirmation is needed based on settings
            needs_confirmation = True
            
            # Skip confirmation for profits if only confirming losses
            if self.confirm_only_losses and pnl > 0:
                needs_confirmation = False
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.DECISION,
                    symbol=position.symbol,
                    title=f"✅ Auto-closing Profit",
                    message=f"Confirmations only required for losses. Closing profitable position.",
                    importance=6
                ))
            
            # Skip confirmation if below threshold
            elif self.confirm_threshold_percent > 0 and pnl_percent < self.confirm_threshold_percent:
                needs_confirmation = False
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.DECISION,
                    symbol=position.symbol,
                    title=f"✅ Auto-closing Small Position",
                    message=f"P&L below {self.confirm_threshold_percent}% threshold. No confirmation needed.",
                    importance=6
                ))
            
            # Request confirmation if needed
            if needs_confirmation:
                if not await self._get_close_confirmation(position, reason):
                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.DECISION,
                        symbol=position.symbol,
                        title=f"🚫 Close Cancelled",
                        message=f"User cancelled position close for {position.symbol}",
                        importance=7
                    ))
                    return
        
        # Get the most current price for accurate P&L calculation
        exit_price = position.current_price
        try:
            if self.data_provider:
                quote = self.data_provider.get_quote(position.symbol)
                if quote and 'last' in quote:
                    exit_price = quote['last']
        except Exception as e:
            logger.debug(f"Could not get current quote for {position.symbol}: {e}")
        
        # Calculate P&L using the most current price
        if position.side == 'short':
            pnl = (position.entry_price - exit_price) * position.quantity
        else:  # long
            pnl = (exit_price - position.entry_price) * position.quantity
        
        roi = (pnl / (position.entry_price * position.quantity)) * 100
        
        self.commentary.add_commentary(TradingCommentary(
            timestamp=datetime.now(),
            type=CommentaryType.DECISION,
            symbol=position.symbol,
            title=f"{'💰' if pnl > 0 else '💸'} Position Closed: {position.symbol}",
            message=f"Closing position with {'profit' if pnl > 0 else 'loss'} of ${abs(pnl):.2f} ({abs(roi):.1f}%)",
            data={
                'entry_price': position.entry_price,
                'exit_price': exit_price,
                'quantity': position.quantity,
            'pnl': pnl,
                'roi_percentage': roi,
                'exit_reason': reason,
                'holding_period': (datetime.now() - position.entry_time).total_seconds() / 3600
            },
            importance=9
        ))
        
	# Learn from this trade
        memory = self.brain.remember_trade(
            symbol=position.symbol,
            pattern=position.reasoning.get('strategy', 'unknown'),
            outcome='win' if pnl > 0 else 'loss',
            pnl_percent=roi,
            context={
                'exit_reason': reason,
                'holding_time': (datetime.now() - position.entry_time).total_seconds() / 60,
                'max_profit': getattr(position, 'max_unrealized_pnl', pnl),
                'market_conditions': self.market_state
            }
        )
        
        # Add the lesson learned
        self.commentary.add_commentary(TradingCommentary(
            timestamp=datetime.now(),
            type=CommentaryType.PSYCHOLOGY,
            symbol=position.symbol,
            title=f"📚 Lesson Learned",
            message=memory.lesson,
            importance=7
        ))
        
        # Track strategy performance
        strategy = position.reasoning.get('strategy', 'unknown')
        if not hasattr(self, 'strategy_performance'):
            self.strategy_performance = {}

        if strategy not in self.strategy_performance:
            self.strategy_performance[strategy] = {'wins': 0, 'losses': 0, 'total': 0}

        if pnl > 0:
            self.strategy_performance[strategy]['wins'] += 1
        else:
            self.strategy_performance[strategy]['losses'] += 1
        self.strategy_performance[strategy]['total'] += 1

        # Disable underperforming strategies
        if self.strategy_performance[strategy]['total'] > 20:
            win_rate = self.strategy_performance[strategy]['wins'] / self.strategy_performance[strategy]['total']
            if win_rate < 0.35:
                logger.warning(f"Disabling {strategy} - win rate {win_rate:.1%}")
        # Save state after each trade
        self._save_state()

        # Post-trade analysis
        self.commentary.add_commentary(TradingCommentary(
            timestamp=datetime.now(),
            type=CommentaryType.PSYCHOLOGY,
            symbol=position.symbol,
            title=f"📝 Post-Trade Analysis",
            message=self._generate_post_trade_analysis(position, reason, pnl),
            importance=6
        ))
        
        # At the end of the method, before removing from positions dict:
        if self.mode == TradingMode.LIVE:
        # Close real position
            success = await self._close_real_position(position)
            if not success:
                return  # Don't remove from tracking if close failed
        
        # Update risk manager only after successful close
        # Note: P&L will be updated from Schwab on next sync
        if pnl < 0:
            self.risk_manager.consecutive_losses += 1
        else:
            self.risk_manager.consecutive_losses = 0
        
        # Remove from positions
        if self.mode == TradingMode.SIMULATION_WITH_COMMENTARY:
            del self.simulated_positions[position.symbol]
        else:
            del self.positions[position.symbol]
        
        # Update trade history
        trade_record = {
            'symbol': position.symbol,
            'entry_time': position.entry_time.isoformat(),
            'exit_time': datetime.now().isoformat(),
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'quantity': position.quantity,
            'pnl': pnl,
            'reason': reason,  # Changed from exit_reason to match frontend
            'reasoning': position.reasoning
        }
        self.trade_history.append(trade_record)

        # Persist to Postgres for SQL queryability
        if self.db_logger is not None:
            pnl_pct = ((exit_price - position.entry_price) / position.entry_price * 100
                        if position.side == "long"
                        else (position.entry_price - exit_price) / position.entry_price * 100)
            try:
                asyncio.get_event_loop().create_task(
                    self.db_logger.log_trade(
                        symbol=position.symbol,
                        side=position.side or "long",
                        strategy=(position.reasoning or {}).get("strategy"),
                        entry_time=position.entry_time,
                        exit_time=datetime.now(),
                        entry_price=position.entry_price,
                        exit_price=exit_price,
                        quantity=position.quantity,
                        pnl=pnl,
                        pnl_pct=round(pnl_pct, 4),
                        exit_reason=reason,
                        atr_at_entry=(position.reasoning or {}).get("atr"),
                        stop_loss=position.stop_loss,
                        take_profit=position.take_profit,
                        confidence=getattr(position, "confidence", None),
                        scaled_out=getattr(position, "scaled_out", False),
                        mode=self.mode.value,
                        reasoning=position.reasoning,
                    )
                )
            except Exception:
                pass

        # Broadcast trade update immediately if we have a connection manager
        if hasattr(self, 'connection_manager') and self.connection_manager:
            try:
                asyncio.create_task(self._broadcast_trade_update(trade_record))
            except Exception as e:
                logger.debug(f"Could not broadcast trade update: {e}")
    
    async def _get_close_confirmation(self, position, reason: str) -> bool:
        """Get user confirmation before closing position"""
        # Initialize pending requests dict if it doesn't exist
        if not hasattr(self, 'pending_close_requests'):
            self.pending_close_requests = {}
        # Store pending close request
        close_request_id = f"close_{position.symbol}_{int(time.time())}"
        self.pending_close_requests = getattr(self, 'pending_close_requests', {})
        
        # Create confirmation request
        self.pending_close_requests[close_request_id] = {
            'position': position,
            'reason': reason,
            'timestamp': datetime.now(),
            'confirmed': None
        }
        
        # Send confirmation request to UI
        pnl = position.unrealized_pnl
        roi = (pnl / (position.entry_price * position.quantity)) * 100
        # Log that we're requesting confirmation
        logger.info(f"Requesting close confirmation for {position.symbol}")
        # Send confirmation request to UI
        try:
            await self.connection_manager.broadcast({
                'type': 'close_confirmation_request',
                'data': {
                    'request_id': close_request_id,
                    'symbol': position.symbol,
                    'quantity': position.quantity,
                    'entry_price': position.entry_price,
                    'current_price': position.current_price,
                    'pnl': pnl,
                    'roi': roi,
                    'reason': reason,
                    'message': f"Close {position.symbol} with {'profit' if pnl > 0 else 'loss'} of ${abs(pnl):.2f} ({abs(roi):.1f}%)?"
                }
            })
        except Exception as e:
            logger.error(f"Error broadcasting close confirmation: {e}")
            return False
        
        # Wait for user response (with timeout)
        timeout = 30  # 30 seconds timeout
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if close_request_id in self.pending_close_requests:
                if self.pending_close_requests[close_request_id]['confirmed'] is not None:
                    confirmed = self.pending_close_requests[close_request_id]['confirmed']
                    del self.pending_close_requests[close_request_id]
                    return confirmed
            await asyncio.sleep(0.1)
        
        # Timeout - default to not closing
        self.commentary.add_commentary(TradingCommentary(
            timestamp=datetime.now(),
            type=CommentaryType.WARNING,
            symbol=position.symbol,
            title=f"⏱️ Close Confirmation Timeout",
            message=f"No response received for closing {position.symbol}. Position remains open.",
            importance=8
        ))
        
        if close_request_id in self.pending_close_requests:
            del self.pending_close_requests[close_request_id]
        
        return False
    def _generate_post_trade_analysis(self, position, exit_reason: str, pnl: float) -> str:
        """Generate insightful post-trade analysis"""
        analysis = []
        
        if pnl > 0:
            analysis.append("✅ What went right:")
            analysis.append(f"- The {position.reasoning.get('strategy', 'strategy')} correctly identified the opportunity")
            analysis.append(f"- Entry timing was good at ${position.entry_price:.2f}")
            if exit_reason == "take_profit":
                analysis.append("- The trade reached its full potential as planned")
        else:
            analysis.append("❌ What could be improved:")
            if exit_reason == "stop_loss":
                analysis.append("- The market moved against our analysis")
                analysis.append("- Consider if the stop was too tight or entry was premature")
            analysis.append("- Review if market conditions changed after entry")
        
        analysis.append("\n💡 Key lessons:")
        analysis.append("- Risk management worked as designed")
        analysis.append(f"- The {abs((position.current_price - position.entry_price) / position.entry_price * 100):.1f}% move took {(datetime.now() - position.entry_time).total_seconds() / 3600:.1f} hours")
        
        return "\n".join(analysis)
    
    async def _get_real_account_info(self) -> Dict[str, float]:
        """Get real account information directly from Schwab - NO CALCULATIONS"""
        if not self.schwab_client or not self.account_id:
            return {}
    
        try:
            # Get account with positions to ensure we have all data
            from schwab.client import Client
            response = self.schwab_client.get_account(
                self.account_hash,
                fields=[Client.Account.Fields.POSITIONS]
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to get account info from Schwab. Status: {response.status_code}")
                return {}
            
            data = response.json()
            account = data.get('securitiesAccount', {})
            current_balances = account.get('currentBalances', {})
            
            # ONLY use Schwab's reported values - NO CALCULATIONS
            current_value = current_balances.get('liquidationValue', 0)
            current_cash = current_balances.get('cashBalance', 0)
            buying_power = current_balances.get('buyingPower', 0)
            
            # Get today's P&L from positions ONLY (most reliable)
            day_pnl = 0
            if 'positions' in account:
                for position in account.get('positions', []):
                    # Schwab provides currentDayProfitLoss for each position
                    position_day_pnl = position.get('currentDayProfitLoss', 0)
                    if position_day_pnl == 0:
                        # Fallback: Check for other P&L fields
                        position_day_pnl = position.get('dayGainLoss', 0)
                    day_pnl += position_day_pnl
            
            logger.debug(f"Schwab Direct Values - Balance: ${current_value:.2f}, P&L: ${day_pnl:.2f}, Cash: ${current_cash:.2f}")
            
            return {
                'balance': current_value,
                'buying_power': buying_power,
                'day_trades_remaining': account.get('roundTrips', 3),
                'cash': current_cash,
                'day_pnl': day_pnl  # Direct from Schwab positions
            }
        except Exception as e:
            logger.error(f"Account info error: {e}")
    
        return {}

    def _is_market_open(self) -> bool:
        """Check if market is open"""
        now = datetime.now()
        if now.weekday() >= 5:
            return False
        
        # Pre-market: 4:00 AM - 9:30 AM EST
        pre_market_start = now.replace(hour=4, minute=0, second=0)
    
        # After-hours end: 8:00 PM EST
        after_hours_end = now.replace(hour=20, minute=0, second=0)
    
        return pre_market_start <= now < after_hours_end
    
    def _get_next_market_open(self) -> str:
        """Get next market open time"""
        now = datetime.now()
        
        if now.weekday() < 5 and now.hour < 9 or (now.hour == 9 and now.minute < 30):
            next_open = now.replace(hour=9, minute=30, second=0)
        elif now.weekday() >= 4:
            days_until_monday = (7 - now.weekday()) % 7
            if days_until_monday == 0:
                days_until_monday = 7
            next_open = now + timedelta(days=days_until_monday)
            next_open = next_open.replace(hour=9, minute=30, second=0)
        else:
            next_open = now + timedelta(days=1)
            next_open = next_open.replace(hour=9, minute=30, second=0)
        
        return next_open.strftime("%Y-%m-%d %H:%M:%S")

    # Add these methods inside the TradingEngineWithCommentary class:

    async def get_schwab_positions(self) -> List[Dict]:
        """Get real positions from Schwab account"""
        if not self.schwab_client or not self.account_id:
            return []
        
        try:
            response = self.schwab_client.get_account(
                self.account_hash,
                fields=[self.schwab_client.Account.Fields.POSITIONS]
            )
            
            if response.status_code == 200:
                data = response.json()
                account_data = data.get('securitiesAccount', {})
                positions_data = account_data.get('positions', [])
                
                schwab_positions = []
                
                for pos in positions_data:
                    instrument = pos.get('instrument', {})
                    symbol = instrument.get('symbol', 'UNKNOWN')
                    
                    # Extract position details
                    long_quantity = pos.get('longQuantity', 0)
                    short_quantity = pos.get('shortQuantity', 0)
                    quantity = long_quantity - short_quantity
                    
                    if quantity == 0:
                        continue
                    
                    # Get prices
                    average_price = pos.get('averagePrice', 0)
                    market_value = pos.get('marketValue', 0)
                    
                    # Calculate current price
                    current_price = market_value / quantity if quantity != 0 else 0
                    
                    # Get P&L
                    day_pnl = pos.get('currentDayProfitLoss', 0)
                    total_pnl = pos.get('unrealizedProfitLoss', 0)
                    
                    # Calculate percentages
                    pnl_percent = (total_pnl / (average_price * abs(quantity))) * 100 if average_price > 0 and quantity != 0 else 0
                    
                    position_info = {
                        'symbol': symbol,
                        'quantity': quantity,
                        'side': 'long' if quantity > 0 else 'short',
                        'average_price': average_price,
                        'current_price': current_price,
                        'market_value': market_value,
                        'day_pnl': day_pnl,
                        'total_pnl': total_pnl,
                        'pnl_percent': pnl_percent,
                        'asset_type': instrument.get('assetType', 'EQUITY')
                    }
                    
                    schwab_positions.append(position_info)
                    
                    # Add commentary about the position
                    # self.commentary.add_commentary(TradingCommentary(
                    #     timestamp=datetime.now(),
                    #     type=CommentaryType.MARKET_ANALYSIS,
                    #     symbol=symbol,
                    #     title=f"📊 Existing Position: {symbol}",
                    #     message=f"Found {abs(quantity)} shares {'long' if quantity > 0 else 'short'}\n"
                    #         f"Avg Price: ${average_price:.2f}, Current: ${current_price:.2f}\n"
                    #         f"P&L: ${total_pnl:.2f} ({pnl_percent:.1f}%)",
                    #     data=position_info,
                    #     importance=6
                    # ))
                
                return schwab_positions
                
        except Exception as e:
            logger.error(f"Error fetching Schwab positions: {e}")
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.WARNING,
                symbol=None,
                title="⚠️ Error Fetching Positions",
                message=f"Could not retrieve positions: {str(e)}",
                importance=7
            ))
        
        return []

    async def _update_and_track_real_positions(self):
        """Update and track real positions from Schwab account"""
        if not self.schwab_client or self.mode != TradingMode.LIVE:
            return
        
        try:
            schwab_positions = await self.get_schwab_positions()
            
            # Update existing tracked positions
            for pos_data in schwab_positions:
                symbol = pos_data['symbol']
                
                # Check if we're already tracking this position
                if symbol in self.positions:
                    # Update the tracked position with latest data
                    position = self.positions[symbol]
                    position.current_price = pos_data['current_price']
                    position.unrealized_pnl = pos_data['total_pnl']
                else:
                    # Create a new position object for positions not initiated by bot
                    position = Position(
                        symbol=symbol,
                        entry_price=pos_data['average_price'],
                        current_price=pos_data['current_price'],
                        quantity=pos_data['quantity'],
                        side='long' if pos_data['quantity'] > 0 else 'short',
                        stop_loss=pos_data['average_price'] * (1 - Config().DEFAULT_STOP_LOSS_PCT),
                        take_profit=pos_data['average_price'] * (1 + Config().DEFAULT_TAKE_PROFIT_PCT),
                        entry_time=datetime.now(),  # We don't know actual entry time
                        unrealized_pnl=pos_data['total_pnl'],
                        reasoning={'source': 'existing_position', 'tracked_from': datetime.now().isoformat()}
                    )
                    
                    self.positions[symbol] = position
                    
                    # Initialize exit tracking for existing positions
                    if hasattr(self, 'exit_manager'):
                        self.exit_manager.initialize_position_tracking(
                            symbol,
                            position.entry_price,
                            position.stop_loss,
                            position.take_profit
                        )
                    
                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.MARKET_ANALYSIS,
                        symbol=symbol,
                        title=f"📌 Tracking Existing Position",
                        message=f"Now tracking {symbol} position from your account",
                        data={
                            'shares': position.quantity,
                            'avg_price': position.entry_price,
                            'current_pnl': position.unrealized_pnl
                        },
                        importance=7
                    ))
                    
        except Exception as e:
            logger.error(f"Error updating real positions: {e}")
    
    def _is_news_blackout(self) -> bool:
        """Check if we're in news blackout period"""
        now = datetime.now()
        
        # Economic calendar blackouts (EST)
        blackouts = [
            # (hour, minute, duration_minutes)
            (8, 30, 15),   # CPI/Jobs
            (10, 0, 15),   # Consumer confidence
            (14, 0, 30),   # FOMC
            (14, 30, 15),  # Powell speaks
        ]
        
        for hour, minute, duration in blackouts:
            event_time = now.replace(hour=hour, minute=minute, second=0)
            if event_time <= now <= event_time + timedelta(minutes=duration):
                return True
        return False
