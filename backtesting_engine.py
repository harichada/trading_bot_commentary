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
    
    # Risk parameters
    max_positions: int = 5
    max_position_size: float = 0.1  # 10% of capital
    stop_loss_default: float = 0.02  # 2%
    take_profit_default: float = 0.05  # 5%
    
    # Advanced features
    use_trailing_stops: bool = True
    use_portfolio_optimization: bool = True
    reinvest_profits: bool = True

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
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.strategy_manager = StrategyManager()
        self.model_manager = ModelManager()
        
        # State tracking
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.cash = config.initial_capital
        self.equity_curve = []
        self.current_time = None
        
        # Performance tracking
        self.daily_pnl = {}
        self.high_water_mark = config.initial_capital
        
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
    
    def run(self, market_data: Dict[str, pd.DataFrame]) -> BacktestResults:
        """Run the backtest"""
        logger.info(f"Starting backtest from {self.config.start_date} to {self.config.end_date}")
        
        # Get all unique timestamps
        all_timestamps = set()
        for df in market_data.values():
            all_timestamps.update(df.index)
        
        timestamps = sorted(all_timestamps)
        
        # Main backtest loop
        for timestamp in timestamps:
            self.current_time = timestamp
            
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
        
        # Calculate results
        results = self._calculate_results()
        
        return results
    
    def _generate_signals(self, market_data: Dict[str, pd.DataFrame]) -> List[StrategySignal]:
        """Generate trading signals from strategies"""
        all_signals = []
        
        for symbol, data in market_data.items():
            if len(data) < 100:  # Need minimum data
                continue
            
            # Run strategies
            signals = self.strategy_manager.analyze_all(data, self.positions)
            
            # Filter with ML models if configured
            if self.model_manager.active_model:
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
            return
        
        # Check position limits
        if len(self.positions) >= self.config.max_positions:
            return
        
        # Get current price
        if symbol not in market_data or len(market_data[symbol]) == 0:
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
                    
                    # Set stop loss and take profit
                    position.stop_loss = signal.stop_loss or entry_price * (1 - self.config.stop_loss_default)
                    position.take_profit = signal.take_profit or entry_price * (1 + self.config.take_profit_default)
                    
                    # Set trailing stop if configured
                    if self.config.use_trailing_stops:
                        position.trailing_stop_distance = entry_price * 0.02  # 2% trailing
                    
                    self.positions[symbol] = position
                    self.cash -= total_cost
                    
                    logger.debug(f"Opened {symbol} position: {position_size} @ {entry_price:.2f}")
        
        elif signal.signal_type in [SignalType.SELL, SignalType.CLOSE_LONG]:
            if symbol in self.positions:
                self._close_position(symbol, market_data, 'signal')
    
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
            
            # Check take profit
            elif position.take_profit and current_price >= position.take_profit:
                positions_to_close.append((symbol, 'take_profit'))
        
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
    raise ImportError(
        "Required package 'ta' is not installed. "
        "Install it with: pip install ta"
    )