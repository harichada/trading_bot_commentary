#!/usr/bin/env python3
"""
Professional Backtesting Engine
Test strategies on historical data with realistic simulation
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from enum import Enum

from strategy_system import StrategyManager, StrategySignal, SignalType
from ml_model_manager_safe import ModelManager
from advanced_orders import Order, OrderType, OrderSide, OrderStatus

logger = logging.getLogger('BacktestingEngine')

class BacktestMode(Enum):
    FAST = "FAST"  # Skip some calculations for speed
    REALISTIC = "REALISTIC"  # Include slippage, fees, etc.
    DETAILED = "DETAILED"  # Track every tick

@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    start_date: datetime
    end_date: datetime
    initial_capital: float = 100000
    commission: float = 0.001  # 0.1% per trade
    slippage: float = 0.0005  # 0.05% slippage
    mode: BacktestMode = BacktestMode.REALISTIC
    symbols: List[str] = field(default_factory=list)
    timeframe: str = "5min"
    
    # Risk parameters (optimized for best performance)
    max_positions: int = 5
    max_position_size: float = 0.2  # 20% of capital (optimized)
    stop_loss_default: float = 0.02  # 2% (optimized)
    take_profit_default: float = 0.04  # 4% (optimized)
    
    # Advanced features
    use_trailing_stops: bool = True
    use_portfolio_optimization: bool = True
    reinvest_profits: bool = True
    use_consensus: bool = True  # Require 2+ strategies to agree before trading
    exit_on_opposite_signal: bool = True  # Exit when consensus reverses - captures small profits before stops hit
    min_holding_bars: int = 12  # Minimum bars to hold before allowing signal-based exit (12 bars = 1 hour at 5min)

@dataclass
class Trade:
    """Represents a completed trade"""
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    side: str  # 'long' or 'short'
    pnl: float
    pnl_pct: float
    commission: float
    slippage: float
    strategy: str
    exit_reason: str  # 'signal', 'stop_loss', 'take_profit', 'end_of_data'
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BacktestResults:
    """Results from a backtest run"""
    trades: List[Trade]
    equity_curve: pd.Series
    daily_returns: pd.Series
    
    # Performance metrics
    total_return: float
    annual_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    
    total_trades: int
    winning_trades: int
    losing_trades: int
    
    # Risk metrics
    value_at_risk: float  # 95% VaR
    conditional_value_at_risk: float  # CVaR
    beta: float
    alpha: float
    
    # Additional info
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary"""
        return {
            'total_return': self.total_return,
            'annual_return': self.annual_return,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'total_trades': self.total_trades,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss
        }

class Position:
    """Represents an open position during backtesting"""
    def __init__(self, symbol: str, quantity: float, entry_price: float, 
                 entry_time: datetime, strategy: str):
        self.symbol = symbol
        self.quantity = quantity
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.strategy = strategy
        self.stop_loss = None
        self.take_profit = None
        self.trailing_stop_distance = None
        self.highest_price = entry_price  # For trailing stops
        
    def update_trailing_stop(self, current_price: float):
        """Update trailing stop based on current price"""
        if self.trailing_stop_distance and current_price > self.highest_price:
            self.highest_price = current_price
            self.stop_loss = current_price - self.trailing_stop_distance

class BacktestingEngine:
    """Professional backtesting engine with realistic simulation"""

    def __init__(self, config: BacktestConfig, enabled_strategies: Optional[List[str]] = None):
        self.config = config
        self.strategy_manager = StrategyManager()
        self.model_manager = ModelManager()

        # Filter strategies if specified
        if enabled_strategies is not None:
            for key in list(self.strategy_manager.strategy_configs.keys()):
                if key not in enabled_strategies:
                    self.strategy_manager.disable_strategy(key)

        # State tracking
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.cash = config.initial_capital
        self.equity_curve = []
        self.current_time = None

        # Performance tracking
        self.daily_pnl = {}
        self.high_water_mark = config.initial_capital

        # Progress tracking for async UI updates
        self.progress = {
            'status': 'idle',      # idle | running | complete | error
            'percent': 0,
            'message': '',
            'current_step': 0,
            'total_steps': 0
        }

        # Structured logging for UI display
        self.log_entries = []
        
    def load_historical_data(self, data_path: str) -> Dict[str, pd.DataFrame]:
        """Load historical data for backtesting"""
        market_data = {}
        
        for symbol in self.config.symbols:
            file_path = Path(data_path) / f"{symbol}_{self.config.timeframe}.csv"
            if file_path.exists():
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                
                # Filter by date range
                mask = (df.index >= self.config.start_date) & (df.index <= self.config.end_date)
                df = df.loc[mask]
                
                market_data[symbol] = df
                logger.info(f"Loaded {len(df)} bars for {symbol}")
            else:
                logger.warning(f"No data file found for {symbol}")
        
        return market_data

    def _log(self, level: str, category: str, message: str, symbol: str = '', data: Optional[Dict] = None):
        """Add structured log entry for UI display"""
        self.log_entries.append({
            'timestamp': self.current_time.isoformat() if self.current_time else '',
            'level': level,
            'category': category,  # signal, trade_open, trade_close, skip, stop_hit, tp_hit, info, error
            'symbol': symbol,
            'message': message,
            'data': data or {}
        })

    def _precompute_indicators(self, market_data: Dict[str, pd.DataFrame]):
        """Pre-compute all technical indicators once on full DataFrames.

        This avoids O(T×B) recomputation at every timestamp.
        Indicators are computed once, then strategies just look up values.
        """
        for symbol, df in market_data.items():
            if len(df) < 30:
                continue
            close = df['close']
            volume = df['volume']
            high = df['high']
            low = df['low']

            # MA Cross indicators
            df['sma_fast'] = close.rolling(9).mean()
            df['sma_slow'] = close.rolling(21).mean()
            df['volume_sma'] = volume.rolling(20).mean()

            # RSI indicators
            df['rsi'] = ta.momentum.RSIIndicator(close=close, window=14).rsi()
            df['rsi_ma'] = df['rsi'].rolling(5).mean()
            df['price_ma'] = close.rolling(20).mean()

            # Bollinger Bands (matching strategy params: window=20, window_dev=1.8)
            bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=1.8)
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_middle'] = bb.bollinger_mavg()
            df['bb_width'] = bb.bollinger_wband()

            # MACD
            macd_ind = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
            df['macd'] = macd_ind.macd()
            df['macd_signal'] = macd_ind.macd_signal()
            df['macd_histogram'] = macd_ind.macd_diff()

            # Momentum Breakout
            df['roc'] = ((close - close.shift(5)) / close.shift(5) * 100)
            df['volume_ma'] = volume.rolling(20).mean()
            df['volume_surge'] = volume / df['volume_ma']
            df['range_high'] = high.rolling(10).max()
            df['range_low'] = low.rolling(10).min()
            df['short_momentum'] = (close - close.shift(3)) / close.shift(3) * 100
            df['atr'] = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()

            # Simple Price Action
            df['sma_short'] = close.rolling(5).mean()
            df['sma_medium'] = close.rolling(15).mean()
            df['sma_20'] = close.rolling(20).mean()
            df['ema_9'] = close.ewm(span=9).mean()
            df['candle_dir'] = np.where(close > df['open'], 1, -1)

            # Enhanced Strategy Indicators (for new strategies)
            df['ema_fast'] = close.ewm(span=12).mean()
            df['ema_slow'] = close.ewm(span=26).mean()

            # ADX with +DI and -DI (for trend strength and direction)
            adx_ind = ta.trend.ADXIndicator(high=high, low=low, close=close, window=14)
            df['adx'] = adx_ind.adx()
            df['di_plus'] = adx_ind.adx_pos()
            df['di_minus'] = adx_ind.adx_neg()

            # Volume analysis (for breakout detection)
            df['volume_std'] = volume.rolling(20).std()
            df['high_20'] = high.rolling(20).max()
            df['low_20'] = low.rolling(20).min()

    def run(self, market_data: Dict[str, pd.DataFrame]) -> BacktestResults:
        """Run the backtest"""
        self.progress = {'status': 'running', 'percent': 0, 'message': 'Starting backtest...', 'current_step': 0, 'total_steps': 0}
        self._log('info', 'info', f"Starting backtest from {self.config.start_date} to {self.config.end_date}")
        logger.info(f"Starting backtest from {self.config.start_date} to {self.config.end_date}")

        for symbol, df in market_data.items():
            logger.info(f"Backtest data: {symbol} has {len(df)} bars, columns={list(df.columns)}, "
                        f"range={df.index[0]} to {df.index[-1]}")

        # Pre-compute all indicators once (major performance optimization)
        self.progress['message'] = 'Pre-computing indicators...'
        self._log('info', 'info', 'Pre-computing technical indicators on full data')
        self._precompute_indicators(market_data)

        # Get all unique timestamps
        all_timestamps = set()
        for df in market_data.values():
            all_timestamps.update(df.index)

        timestamps = sorted(all_timestamps)
        self.progress['total_steps'] = len(timestamps)
        logger.info(f"Backtest: {len(timestamps)} total timestamps to process")
        self._log('info', 'info', f"Processing {len(timestamps)} timestamps across {len(market_data)} symbols")

        # Main backtest loop
        signal_count = 0
        for i, timestamp in enumerate(timestamps):
            self.current_time = timestamp

            # Update progress
            self.progress['current_step'] = i
            self.progress['percent'] = int((i / len(timestamps)) * 100)
            if i % 100 == 0:  # Update message every 100 steps to avoid overhead
                self.progress['message'] = f'Processing {timestamp.strftime("%Y-%m-%d %H:%M")} ({i}/{len(timestamps)})'

            # Get current market snapshot
            current_data = {}
            for symbol, df in market_data.items():
                if timestamp in df.index:
                    current_data[symbol] = df.loc[:timestamp]

            # Update trailing stops
            self._update_trailing_stops(current_data)

            # Check stop losses and take profits
            self._check_exits(current_data)

            # Generate signals
            signals = self._generate_signals(current_data)
            signal_count += len(signals)

            # Execute signals
            for signal in signals:
                self._process_signal(signal, current_data)

            # Record equity
            equity = self._calculate_equity(current_data)
            self.equity_curve.append({
                'timestamp': timestamp,
                'equity': equity,
                'cash': self.cash,
                'positions_value': equity - self.cash
            })

            # Update high water mark
            if equity > self.high_water_mark:
                self.high_water_mark = equity

        # Close all remaining positions
        self._close_all_positions(market_data)

        logger.info(f"Backtest complete: {signal_count} total signals generated, "
                    f"{len(self.trades)} trades executed, "
                    f"{len(self.positions)} positions still open")
        self._log('info', 'info', f"Backtest complete: {signal_count} signals, {len(self.trades)} trades")

        # Calculate results
        results = self._calculate_results()

        # Save report to disk for comparison
        self._save_report(results)

        self.progress = {'status': 'complete', 'percent': 100, 'message': 'Backtest complete', 'current_step': len(timestamps), 'total_steps': len(timestamps)}

        return results

    def _save_report(self, results: BacktestResults):
        """Save backtest report to disk for later comparison"""
        try:
            reports_dir = Path('backtest_reports')
            reports_dir.mkdir(exist_ok=True)

            # Generate filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            symbols_str = '_'.join(self.config.symbols[:3])  # First 3 symbols
            if len(self.config.symbols) > 3:
                symbols_str += f'_+{len(self.config.symbols)-3}'
            filename = f"backtest_{timestamp}_{symbols_str}.json"

            # Build report data
            report = {
                'meta': {
                    'timestamp': datetime.now().isoformat(),
                    'filename': filename
                },
                'config': {
                    'symbols': self.config.symbols,
                    'start_date': self.config.start_date.isoformat(),
                    'end_date': self.config.end_date.isoformat(),
                    'initial_capital': self.config.initial_capital,
                    'timeframe': self.config.timeframe,
                    'mode': self.config.mode.value,
                    'max_positions': self.config.max_positions,
                    'max_position_size': self.config.max_position_size,
                    'stop_loss_default': self.config.stop_loss_default,
                    'take_profit_default': self.config.take_profit_default,
                    'commission': self.config.commission,
                    'slippage': self.config.slippage,
                    'use_consensus': self.config.use_consensus,
                    'exit_on_opposite_signal': self.config.exit_on_opposite_signal,
                    'strategies_enabled': list(self.strategy_manager.strategies.keys())
                },
                'metrics': {
                    'total_return': float(results.total_return),
                    'annual_return': float(results.annual_return),
                    'sharpe_ratio': float(results.sharpe_ratio),
                    'sortino_ratio': float(results.sortino_ratio),
                    'max_drawdown': float(results.max_drawdown),
                    'max_drawdown_duration': int(results.max_drawdown_duration),
                    'win_rate': float(results.win_rate),
                    'profit_factor': float(results.profit_factor),
                    'total_trades': int(results.total_trades),
                    'winning_trades': int(results.winning_trades),
                    'losing_trades': int(results.losing_trades),
                    'avg_win': float(results.avg_win),
                    'avg_loss': float(results.avg_loss),
                    'largest_win': float(results.largest_win),
                    'largest_loss': float(results.largest_loss),
                    'initial_capital': float(results.initial_capital),
                    'final_capital': float(results.final_capital),
                    'value_at_risk': float(results.value_at_risk) if not np.isnan(results.value_at_risk) else 0,
                    'conditional_var': float(results.conditional_value_at_risk) if not np.isnan(results.conditional_value_at_risk) else 0
                },
                'trades': [
                    {
                        'symbol': t.symbol,
                        'entry_time': t.entry_time.isoformat(),
                        'exit_time': t.exit_time.isoformat(),
                        'entry_price': float(t.entry_price),
                        'exit_price': float(t.exit_price),
                        'quantity': float(t.quantity),
                        'side': t.side,
                        'pnl': float(t.pnl),
                        'pnl_pct': float(t.pnl_pct),
                        'commission': float(t.commission),
                        'strategy': t.strategy,
                        'exit_reason': t.exit_reason
                    }
                    for t in results.trades
                ],
                'equity_curve': [
                    {'date': d.isoformat(), 'equity': float(v)}
                    for d, v in results.equity_curve.items()
                ][-500:],  # Last 500 points
                'logs': self.log_entries[-500:]  # Last 500 log entries
            }

            # Save to file
            filepath = reports_dir / filename
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)

            logger.info(f"Backtest report saved to {filepath}")
            self._log('info', 'info', f"Report saved: {filename}")

        except Exception as e:
            logger.error(f"Failed to save backtest report: {e}")
    
    def _generate_signals(self, market_data: Dict[str, pd.DataFrame]) -> List[StrategySignal]:
        """Generate trading signals from strategies using regime-aware consensus filtering"""
        all_signals = []

        for symbol, data in market_data.items():
            if len(data) < 30:  # Need minimum data for indicators
                continue

            # Ensure index.name is set so strategies know the symbol
            data.index.name = symbol

            # Run strategies with regime awareness
            try:
                raw_signals, regime = self.strategy_manager.analyze_with_regime(data, self.positions)
            except Exception as e:
                logger.debug(f"Strategy error for {symbol} (len={len(data)}): {e}")
                # Fallback to standard analysis if regime-aware fails
                try:
                    raw_signals = self.strategy_manager.analyze_all(data, self.positions)
                    regime = None
                except Exception as e2:
                    logger.debug(f"Fallback strategy error for {symbol}: {e2}")
                    continue

            # Fix symbol on signals
            for sig in raw_signals:
                if sig.symbol == 'UNKNOWN' or not sig.symbol:
                    sig.symbol = symbol

            # USE CONSENSUS FILTERING - require 2+ strategies to agree
            if self.config.use_consensus:
                consensus_signal = self.strategy_manager.get_consensus_signal(raw_signals)
                if consensus_signal:
                    signals = [consensus_signal]
                    regime_info = regime.regime.value if regime else 'unknown'
                    self._log('info', 'signal',
                        f"CONSENSUS {consensus_signal.signal_type.value} from {consensus_signal.metadata.get('strategies', [])} "
                        f"(strength={consensus_signal.strength:.2f}, regime={regime_info})", symbol, {
                        'strategy': 'Consensus',
                        'signal_type': consensus_signal.signal_type.value,
                        'strength': consensus_signal.strength,
                        'price': consensus_signal.entry_price,
                        'agreeing_strategies': consensus_signal.metadata.get('strategies', []),
                        'regime': regime_info,
                        'regime_confidence': regime.confidence if regime else 0
                    })
                else:
                    signals = []  # No consensus = no trade
                    if raw_signals:
                        self._log('debug', 'skip', f"No consensus from {len(raw_signals)} signals (regime={regime.regime.value if regime else 'unknown'})", symbol)
            else:
                # Legacy mode: process all signals (causes strategy conflicts!)
                signals = raw_signals
                if signals:
                    logger.info(f"Backtest: {len(signals)} signals for {symbol} at {data.index[-1]}")
                    for sig in signals:
                        self._log('info', 'signal', f"{sig.signal_type.value} signal from {sig.strategy_name} (strength={sig.strength:.2f})", symbol, {
                            'strategy': sig.strategy_name,
                            'signal_type': sig.signal_type.value,
                            'strength': sig.strength,
                            'price': sig.entry_price
                        })

            # Filter with ML models if configured
            if signals and self.model_manager.active_model:
                signals = self._filter_signals_with_ml(signals, data)

            all_signals.extend(signals)

        # Apply portfolio-level filters
        if self.config.use_portfolio_optimization:
            all_signals = self._optimize_portfolio_signals(all_signals)

        return all_signals
    
    def _filter_signals_with_ml(self, signals: List[StrategySignal], 
                               data: pd.DataFrame) -> List[StrategySignal]:
        """Filter signals using ML predictions"""
        filtered = []
        
        for signal in signals:
            # Prepare features
            features = self._prepare_ml_features(data)
            if features is not None:
                # Get prediction
                try:
                    proba = self.model_manager.predict_proba(features)
                    confidence = proba[0][1] if signal.signal_type == SignalType.BUY else proba[0][0]
                    
                    if confidence > 0.6:  # Confidence threshold
                        signal.strength *= confidence
                        signal.metadata['ml_confidence'] = confidence
                        filtered.append(signal)
                except:
                    # If ML fails, use original signal
                    filtered.append(signal)
            else:
                filtered.append(signal)
        
        return filtered
    
    def _process_signal(self, signal: StrategySignal, market_data: Dict[str, pd.DataFrame]):
        """Process a trading signal"""
        symbol = signal.symbol

        # Check if we already have a position
        if symbol in self.positions and signal.signal_type in [SignalType.BUY]:
            logger.debug(f"Backtest: Skipping {symbol} BUY - already have position")
            self._log('debug', 'skip', f"Skipping BUY - already have position", symbol)
            return

        # Check position limits
        if len(self.positions) >= self.config.max_positions:
            logger.debug(f"Backtest: Skipping {symbol} - max positions ({self.config.max_positions}) reached")
            self._log('debug', 'skip', f"Skipping - max positions ({self.config.max_positions}) reached", symbol)
            return

        # Get current price
        if symbol not in market_data or len(market_data[symbol]) == 0:
            logger.debug(f"Backtest: Skipping {symbol} - no market data")
            self._log('debug', 'skip', f"Skipping - no market data", symbol)
            return

        current_bar = market_data[symbol].iloc[-1]

        if signal.signal_type == SignalType.BUY:
            # Calculate position size
            position_size = self._calculate_position_size(signal, current_bar['close'])
            
            if position_size > 0:
                # Apply slippage
                entry_price = current_bar['close'] * (1 + self.config.slippage)
                
                # Calculate commission
                commission = position_size * entry_price * self.config.commission
                
                # Check if we have enough cash
                total_cost = position_size * entry_price + commission
                if total_cost <= self.cash:
                    # Open position
                    position = Position(
                        symbol=symbol,
                        quantity=position_size,
                        entry_price=entry_price,
                        entry_time=self.current_time,
                        strategy=signal.strategy_name
                    )
                    
                    # Set stop loss and take profit (always use config values for consistent backtesting)
                    position.stop_loss = entry_price * (1 - self.config.stop_loss_default)
                    position.take_profit = entry_price * (1 + self.config.take_profit_default)
                    
                    # Set trailing stop if configured (uses stop_loss_default as distance)
                    if self.config.use_trailing_stops:
                        position.trailing_stop_distance = entry_price * self.config.stop_loss_default
                    
                    self.positions[symbol] = position
                    self.cash -= total_cost

                    logger.debug(f"Opened {symbol} position: {position_size} @ {entry_price:.2f}")
                    self._log('info', 'trade_open', f"Opened {position_size:.2f} shares @ ${entry_price:.2f} (strategy: {signal.strategy_name})", symbol, {
                        'quantity': position_size,
                        'entry_price': entry_price,
                        'stop_loss': position.stop_loss,
                        'take_profit': position.take_profit,
                        'strategy': signal.strategy_name,
                        'cost': total_cost
                    })
        
        elif signal.signal_type in [SignalType.SELL, SignalType.CLOSE_LONG]:
            if symbol in self.positions:
                position = self.positions[symbol]

                # Check minimum holding period (to prevent over-trading)
                if self.current_time and position.entry_time:
                    bars_held = (self.current_time - position.entry_time).total_seconds() / 300  # 5-min bars
                    if bars_held < self.config.min_holding_bars:
                        self._log('debug', 'skip', f'Ignoring exit signal - only held {bars_held:.0f} bars (min: {self.config.min_holding_bars})', symbol)
                        return  # Don't exit yet

                if self.config.exit_on_opposite_signal:
                    # Close on opposite signal
                    self._close_position(symbol, market_data, 'signal')
                else:
                    # Skip closing - let stop loss/take profit manage exit
                    self._log('debug', 'skip', f'Ignoring opposite signal - waiting for SL/TP exit', symbol)
    
    def _check_exits(self, market_data: Dict[str, pd.DataFrame]):
        """Check for stop loss and take profit exits"""
        positions_to_close = []
        
        for symbol, position in self.positions.items():
            if symbol not in market_data or len(market_data[symbol]) == 0:
                continue
            
            current_bar = market_data[symbol].iloc[-1]
            current_price = current_bar['close']
            
            # Check stop loss
            if position.stop_loss and current_price <= position.stop_loss:
                positions_to_close.append((symbol, 'stop_loss'))
                self._log('warning', 'stop_hit', f"Stop loss hit at ${current_price:.2f} (stop was ${position.stop_loss:.2f})", symbol)

            # Check take profit
            elif position.take_profit and current_price >= position.take_profit:
                positions_to_close.append((symbol, 'take_profit'))
                self._log('info', 'tp_hit', f"Take profit hit at ${current_price:.2f} (target was ${position.take_profit:.2f})", symbol)
        
        # Close positions
        for symbol, reason in positions_to_close:
            self._close_position(symbol, market_data, reason)
    
    def _update_trailing_stops(self, market_data: Dict[str, pd.DataFrame]):
        """Update trailing stops for all positions"""
        for symbol, position in self.positions.items():
            if position.trailing_stop_distance and symbol in market_data:
                current_price = market_data[symbol].iloc[-1]['close']
                position.update_trailing_stop(current_price)
    
    def _close_position(self, symbol: str, market_data: Dict[str, pd.DataFrame], 
                       reason: str = 'signal'):
        """Close a position and record the trade"""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        
        # Get exit price
        if symbol in market_data and len(market_data[symbol]) > 0:
            exit_price = market_data[symbol].iloc[-1]['close']
        else:
            exit_price = position.entry_price  # Fallback
        
        # Apply slippage for market exit
        exit_price *= (1 - self.config.slippage)
        
        # Calculate P&L
        if position.quantity > 0:  # Long position
            pnl = (exit_price - position.entry_price) * position.quantity
        else:  # Short position
            pnl = (position.entry_price - exit_price) * abs(position.quantity)
        
        # Calculate commission
        commission = abs(position.quantity) * exit_price * self.config.commission
        
        # Net P&L
        net_pnl = pnl - commission
        pnl_pct = net_pnl / (position.entry_price * abs(position.quantity))
        
        # Record trade
        trade = Trade(
            symbol=symbol,
            entry_time=position.entry_time,
            exit_time=self.current_time,
            entry_price=position.entry_price,
            exit_price=exit_price,
            quantity=position.quantity,
            side='long' if position.quantity > 0 else 'short',
            pnl=net_pnl,
            pnl_pct=pnl_pct,
            commission=commission,
            slippage=abs(position.quantity) * exit_price * self.config.slippage,
            strategy=position.strategy,
            exit_reason=reason
        )
        
        self.trades.append(trade)
        self.cash += abs(position.quantity) * exit_price - commission

        # Remove position
        del self.positions[symbol]

        logger.debug(f"Closed {symbol} position: P&L = ${net_pnl:.2f} ({pnl_pct:.2%})")
        pnl_sign = '+' if net_pnl >= 0 else ''
        self._log('info', 'trade_close', f"Closed @ ${exit_price:.2f} | P&L: {pnl_sign}${net_pnl:.2f} ({pnl_pct:+.2%}) | Reason: {reason}", symbol, {
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'quantity': position.quantity,
            'pnl': net_pnl,
            'pnl_pct': pnl_pct,
            'exit_reason': reason,
            'strategy': position.strategy
        })
    
    def _close_all_positions(self, market_data: Dict[str, pd.DataFrame]):
        """Close all remaining positions at end of backtest"""
        symbols_to_close = list(self.positions.keys())
        for symbol in symbols_to_close:
            self._close_position(symbol, market_data, 'end_of_data')
    
    def _calculate_position_size(self, signal: StrategySignal, current_price: float) -> float:
        """Calculate position size based on risk management rules"""
        # Maximum position value
        max_position_value = self.cash * self.config.max_position_size
        
        # Adjust by signal strength
        position_value = max_position_value * signal.strength
        
        # Kelly Criterion (optional)
        if hasattr(signal, 'win_probability') and hasattr(signal, 'win_loss_ratio'):
            kelly_fraction = self._calculate_kelly_fraction(
                signal.win_probability, 
                signal.win_loss_ratio
            )
            position_value *= min(kelly_fraction, 0.25)  # Cap at 25%
        
        # Calculate shares
        shares = position_value / current_price
        
        return round(shares, 2)
    
    def _calculate_kelly_fraction(self, win_prob: float, win_loss_ratio: float) -> float:
        """Calculate optimal position size using Kelly Criterion"""
        if win_loss_ratio <= 0:
            return 0
        
        kelly = (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio
        return max(0, min(kelly, 1))  # Bound between 0 and 1
    
    def _calculate_equity(self, market_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate current account equity"""
        positions_value = 0
        
        for symbol, position in self.positions.items():
            if symbol in market_data and len(market_data[symbol]) > 0:
                current_price = market_data[symbol].iloc[-1]['close']
                if position.quantity > 0:
                    positions_value += position.quantity * current_price
                else:
                    # Short position
                    positions_value += position.quantity * current_price
                    positions_value += 2 * abs(position.quantity) * position.entry_price
        
        return self.cash + positions_value
    
    def _prepare_ml_features(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Prepare features for ML prediction"""
        if len(data) < 100:
            return None
        
        features = pd.DataFrame(index=[data.index[-1]])
        
        # Price features
        features['returns_1'] = data['close'].pct_change(1).iloc[-1]
        features['returns_5'] = data['close'].pct_change(5).iloc[-1]
        features['returns_20'] = data['close'].pct_change(20).iloc[-1]
        
        # Technical indicators
        features['rsi'] = ta.momentum.RSIIndicator(close=data['close']).rsi().iloc[-1]
        
        # Moving averages
        features['sma_ratio'] = data['close'].iloc[-1] / data['close'].rolling(20).mean().iloc[-1]
        
        # Volume
        features['volume_ratio'] = data['volume'].iloc[-1] / data['volume'].rolling(20).mean().iloc[-1]
        
        return features.fillna(0)
    
    def _optimize_portfolio_signals(self, signals: List[StrategySignal]) -> List[StrategySignal]:
        """Optimize signal selection for portfolio-level risk"""
        if len(signals) <= self.config.max_positions:
            return signals
        
        # Score signals by multiple factors
        scored_signals = []
        for signal in signals:
            score = signal.strength
            
            # Diversification bonus
            symbol_count = sum(1 for s in signals if s.symbol == signal.symbol)
            score *= (1 / symbol_count)  # Penalize multiple signals for same symbol
            
            # Add to list
            scored_signals.append((score, signal))
        
        # Sort by score and take top N
        scored_signals.sort(key=lambda x: x[0], reverse=True)
        return [signal for _, signal in scored_signals[:self.config.max_positions]]
    
    def _calculate_results(self) -> BacktestResults:
        """Calculate comprehensive backtest results"""
        # Convert equity curve to series
        equity_df = pd.DataFrame(self.equity_curve)
        equity_series = pd.Series(
            equity_df['equity'].values,
            index=equity_df['timestamp']
        )
        
        # Calculate returns
        returns = equity_series.pct_change().dropna()
        daily_returns = returns.resample('D').sum()
        
        # Basic metrics
        total_return = (equity_series.iloc[-1] - self.config.initial_capital) / self.config.initial_capital
        
        # Annualized return
        days = (equity_series.index[-1] - equity_series.index[0]).days
        annual_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
        
        # Risk metrics
        sharpe_ratio = self._calculate_sharpe_ratio(daily_returns)
        sortino_ratio = self._calculate_sortino_ratio(daily_returns)
        max_dd, max_dd_duration = self._calculate_max_drawdown(equity_series)
        
        # Trade statistics
        if self.trades:
            winning_trades = [t for t in self.trades if t.pnl > 0]
            losing_trades = [t for t in self.trades if t.pnl <= 0]
            
            win_rate = len(winning_trades) / len(self.trades)
            
            avg_win = sum(t.pnl for t in winning_trades) / len(winning_trades) if winning_trades else 0
            avg_loss = sum(t.pnl for t in losing_trades) / len(losing_trades) if losing_trades else 0
            
            profit_factor = abs(sum(t.pnl for t in winning_trades) / sum(t.pnl for t in losing_trades)) if losing_trades and sum(t.pnl for t in losing_trades) != 0 else 0
            
            largest_win = max((t.pnl for t in self.trades), default=0)
            largest_loss = min((t.pnl for t in self.trades), default=0)
        else:
            win_rate = avg_win = avg_loss = profit_factor = largest_win = largest_loss = 0
        
        # Risk metrics
        var_95 = self._calculate_var(daily_returns, 0.95)
        cvar_95 = self._calculate_cvar(daily_returns, 0.95)
        
        # Market correlation (simplified - assume SPY as benchmark)
        beta = 1.0  # Placeholder
        alpha = annual_return - beta * 0.10  # Assume 10% market return
        
        return BacktestResults(
            trades=self.trades,
            equity_curve=equity_series,
            daily_returns=daily_returns,
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_dd,
            max_drawdown_duration=max_dd_duration,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            total_trades=len(self.trades),
            winning_trades=len([t for t in self.trades if t.pnl > 0]),
            losing_trades=len([t for t in self.trades if t.pnl <= 0]),
            value_at_risk=var_95,
            conditional_value_at_risk=cvar_95,
            beta=beta,
            alpha=alpha,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            initial_capital=self.config.initial_capital,
            final_capital=equity_series.iloc[-1]
        )
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0 or returns.std() == 0:
            return 0
        
        excess_returns = returns - risk_free_rate / 252
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    
    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        if len(returns) == 0:
            return 0
        
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0
        
        return np.sqrt(252) * excess_returns.mean() / downside_returns.std()
    
    def _calculate_max_drawdown(self, equity_series: pd.Series) -> Tuple[float, int]:
        """Calculate maximum drawdown and duration"""
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max
        
        max_drawdown = drawdown.min()
        
        # Calculate duration
        drawdown_start = None
        max_duration = 0
        current_duration = 0
        
        for i, dd in enumerate(drawdown):
            if dd < 0:
                if drawdown_start is None:
                    drawdown_start = i
                current_duration = i - drawdown_start
            else:
                if current_duration > max_duration:
                    max_duration = current_duration
                drawdown_start = None
                current_duration = 0
        
        return abs(max_drawdown), max_duration
    
    def _calculate_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Value at Risk"""
        if len(returns) == 0:
            return 0
        
        return np.percentile(returns, (1 - confidence) * 100)
    
    def _calculate_cvar(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        var = self._calculate_var(returns, confidence)
        return returns[returns <= var].mean()

class BacktestReport:
    """Generate comprehensive backtest reports"""
    
    @staticmethod
    def generate_html_report(results: BacktestResults, output_path: str = "backtest_report.html"):
        """Generate HTML report with charts and metrics"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Backtest Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
                .header {{ background: #333; color: white; padding: 20px; border-radius: 8px; }}
                .metrics-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0; }}
                .metric-card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #333; }}
                .metric-label {{ color: #666; font-size: 14px; }}
                .chart {{ background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .positive {{ color: #4CAF50; }}
                .negative {{ color: #f44336; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Backtest Report</h1>
                <p>{start_date} to {end_date}</p>
            </div>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Total Return</div>
                    <div class="metric-value {return_class}">{total_return:.2%}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Annual Return</div>
                    <div class="metric-value {return_class}">{annual_return:.2%}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Sharpe Ratio</div>
                    <div class="metric-value">{sharpe_ratio:.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Max Drawdown</div>
                    <div class="metric-value negative">{max_drawdown:.2%}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Win Rate</div>
                    <div class="metric-value">{win_rate:.1%}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Profit Factor</div>
                    <div class="metric-value">{profit_factor:.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Total Trades</div>
                    <div class="metric-value">{total_trades}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Average Win/Loss</div>
                    <div class="metric-value">${avg_win:.2f} / ${avg_loss:.2f}</div>
                </div>
            </div>
            
            <div class="chart">
                <div id="equity-curve"></div>
            </div>
            
            <div class="chart">
                <div id="drawdown-chart"></div>
            </div>
            
            <div class="chart">
                <div id="returns-dist"></div>
            </div>
            
            <script>
                {charts_js}
            </script>
        </body>
        </html>
        """
        
        # Generate JavaScript for charts
        equity_data = results.equity_curve
        dates = [d.strftime('%Y-%m-%d') for d in equity_data.index]
        
        charts_js = f"""
        // Equity Curve
        var equityTrace = {{
            x: {dates},
            y: {equity_data.values.tolist()},
            type: 'scatter',
            mode: 'lines',
            name: 'Equity'
        }};
        
        var equityLayout = {{
            title: 'Equity Curve',
            xaxis: {{ title: 'Date' }},
            yaxis: {{ title: 'Equity ($)' }}
        }};
        
        Plotly.newPlot('equity-curve', [equityTrace], equityLayout);
        
        // Drawdown Chart
        var drawdownTrace = {{
            x: {dates},
            y: {((equity_data / equity_data.expanding().max() - 1) * 100).values.tolist()},
            type: 'scatter',
            mode: 'lines',
            fill: 'tozeroy',
            name: 'Drawdown'
        }};
        
        var drawdownLayout = {{
            title: 'Drawdown',
            xaxis: {{ title: 'Date' }},
            yaxis: {{ title: 'Drawdown (%)' }}
        }};
        
        Plotly.newPlot('drawdown-chart', [drawdownTrace], drawdownLayout);
        
        // Returns Distribution
        var returnsTrace = {{
            x: {results.daily_returns.values.tolist()},
            type: 'histogram',
            name: 'Daily Returns'
        }};
        
        var returnsLayout = {{
            title: 'Returns Distribution',
            xaxis: {{ title: 'Daily Return' }},
            yaxis: {{ title: 'Frequency' }}
        }};
        
        Plotly.newPlot('returns-dist', [returnsTrace], returnsLayout);
        """
        
        # Fill template
        html = html_template.format(
            start_date=results.start_date.strftime('%Y-%m-%d'),
            end_date=results.end_date.strftime('%Y-%m-%d'),
            total_return=results.total_return,
            annual_return=results.annual_return,
            sharpe_ratio=results.sharpe_ratio,
            max_drawdown=results.max_drawdown,
            win_rate=results.win_rate,
            profit_factor=results.profit_factor,
            total_trades=results.total_trades,
            avg_win=results.avg_win,
            avg_loss=abs(results.avg_loss),
            return_class='positive' if results.total_return > 0 else 'negative',
            charts_js=charts_js
        )
        
        # Save report
        with open(output_path, 'w') as f:
            f.write(html)
        
        logger.info(f"Backtest report saved to {output_path}")
    
    @staticmethod
    def generate_json_report(results: BacktestResults, output_path: str = "backtest_results.json"):
        """Save results as JSON for further analysis"""
        report_data = {
            'summary': results.to_dict(),
            'trades': [
                {
                    'symbol': t.symbol,
                    'entry_time': t.entry_time.isoformat(),
                    'exit_time': t.exit_time.isoformat(),
                    'entry_price': t.entry_price,
                    'exit_price': t.exit_price,
                    'quantity': t.quantity,
                    'pnl': t.pnl,
                    'pnl_pct': t.pnl_pct,
                    'strategy': t.strategy,
                    'exit_reason': t.exit_reason
                }
                for t in results.trades
            ],
            'equity_curve': [
                {
                    'date': date.isoformat(),
                    'equity': float(value)
                }
                for date, value in results.equity_curve.items()
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")

# Import required for features
try:
    import ta
except ImportError:
    logger.warning("ta library not found, installing...")
    import os
    os.system("pip install ta")
    import ta