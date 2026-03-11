import logging
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger('TradingBot')


@dataclass
class BacktestResult:
    """Results from backtesting"""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    consecutive_wins: int
    consecutive_losses: int
    equity_curve: List[float]
    trade_history: List[Dict]
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float

class BacktestEngine:
    """Comprehensive backtesting engine"""

    def __init__(self, config: dict):
        self.config = config
        self.initial_capital = config.get('backtesting', {}).get('initial_capital', 100000)
        self.commission = config.get('backtesting', {}).get('commission', 0.005)
        self.slippage = config.get('backtesting', {}).get('slippage', 0.001)
        self.current_capital = self.initial_capital
        self.positions = {}
        self.trade_history = []
        self.equity_curve = [self.initial_capital]
        self.current_date = None

    def run_backtest(self, data: Dict[str, pd.DataFrame], strategy,
                    start_date: str, end_date: str) -> BacktestResult:
        """Run backtest on historical data"""
        logger.info(f"Starting backtest from {start_date} to {end_date}")

        # Prepare data
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        # Get all unique dates
        all_dates = set()
        for symbol, df in data.items():
            df['date'] = pd.to_datetime(df.index)
            df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]
            all_dates.update(df['date'].dt.date)

        all_dates = sorted(list(all_dates))

        # Run simulation
        for date in all_dates:
            self.current_date = date
            self._process_date(date, data, strategy)
            self._update_equity_curve()

        # Calculate results
        return self._calculate_results(start_dt, end_dt)

    def _process_date(self, date, data: Dict[str, pd.DataFrame], strategy):
        """Process a single date"""
        # Update positions with current prices
        for symbol, position in self.positions.items():
            if symbol in data:
                df = data[symbol]
                current_price = df[df['date'].dt.date == date]['close'].iloc[0]
                position['current_price'] = current_price
                position['unrealized_pnl'] = (
                    current_price - position['entry_price']
                ) * position['quantity']

        # Generate signals
        for symbol, df in data.items():
            if symbol in df and df[df['date'].dt.date == date].shape[0] > 0:
                current_data = df[df['date'].dt.date == date].iloc[0]
                signal = strategy.generate_signal(current_data)

                if signal:
                    self._execute_signal(signal, current_data)

    def _execute_signal(self, signal, current_data):
        """Execute a trading signal"""
        from core.models import SignalType

        symbol = signal.symbol
        price = current_data['close']

        # Apply slippage
        if signal.signal_type == SignalType.BUY:
            execution_price = price * (1 + self.slippage)
        else:
            execution_price = price * (1 - self.slippage)

        # Calculate position size
        position_value = signal.position_size * execution_price
        commission_cost = position_value * self.commission

        if signal.signal_type == SignalType.BUY:
            if self.current_capital >= position_value + commission_cost:
                self.positions[symbol] = {
                    'entry_price': execution_price,
                    'quantity': signal.position_size,
                    'entry_date': self.current_date,
                    'current_price': execution_price,
                    'unrealized_pnl': 0
                }
                self.current_capital -= (position_value + commission_cost)
                self._record_trade(symbol, 'BUY', execution_price, signal.position_size, commission_cost)

        elif signal.signal_type == SignalType.SELL and symbol in self.positions:
            position = self.positions[symbol]
            exit_value = position['quantity'] * execution_price
            commission_cost = exit_value * self.commission
            pnl = (execution_price - position['entry_price']) * position['quantity'] - commission_cost

            self.current_capital += (exit_value - commission_cost)
            del self.positions[symbol]
            self._record_trade(symbol, 'SELL', execution_price, position['quantity'], commission_cost, pnl)

    def _record_trade(self, symbol: str, side: str, price: float, quantity: int,
                     commission: float, pnl: float = 0):
        """Record a trade"""
        trade = {
            'date': self.current_date,
            'symbol': symbol,
            'side': side,
            'price': price,
            'quantity': quantity,
            'commission': commission,
            'pnl': pnl,
            'capital': self.current_capital
        }
        self.trade_history.append(trade)

    def _update_equity_curve(self):
        """Update equity curve"""
        total_value = self.current_capital
        for position in self.positions.values():
            total_value += position['unrealized_pnl']
        self.equity_curve.append(total_value)

    def _calculate_results(self, start_date: datetime, end_date: datetime) -> BacktestResult:
        """Calculate comprehensive backtest results"""
        equity_series = pd.Series(self.equity_curve)
        returns = equity_series.pct_change().dropna()

        # Basic metrics
        total_return = (equity_series.iloc[-1] - equity_series.iloc[0]) / equity_series.iloc[0]
        days = (end_date - start_date).days
        annualized_return = (1 + total_return) ** (365 / days) - 1

        # Risk metrics
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        sortino_ratio = self._calculate_sortino_ratio(returns)
        max_drawdown, max_drawdown_duration = self._calculate_max_drawdown(equity_series)

        # Trade metrics
        trades_df = pd.DataFrame(self.trade_history)
        if len(trades_df) > 0:
            winning_trades = trades_df[trades_df['pnl'] > 0]
            losing_trades = trades_df[trades_df['pnl'] < 0]

            win_rate = len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0
            profit_factor = abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if len(losing_trades) > 0 else float('inf')

            average_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
            average_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
            largest_win = winning_trades['pnl'].max() if len(winning_trades) > 0 else 0
            largest_loss = losing_trades['pnl'].min() if len(losing_trades) > 0 else 0
        else:
            win_rate = profit_factor = average_win = average_loss = largest_win = largest_loss = 0

        # Consecutive trades
        consecutive_wins, consecutive_losses = self._calculate_consecutive_trades(trades_df)

        return BacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_drawdown_duration,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(trades_df),
            winning_trades=len(winning_trades) if len(trades_df) > 0 else 0,
            losing_trades=len(losing_trades) if len(trades_df) > 0 else 0,
            average_win=average_win,
            average_loss=average_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            consecutive_wins=consecutive_wins,
            consecutive_losses=consecutive_losses,
            equity_curve=self.equity_curve,
            trade_history=self.trade_history,
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_capital=equity_series.iloc[-1]
        )

    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0:
            return 0
        risk_free_rate = self.config.get('performance_metrics', {}).get('risk_free_rate', 0.02) / 252
        excess_returns = returns - risk_free_rate
        return np.sqrt(252) * excess_returns.mean() / returns.std() if returns.std() > 0 else 0

    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio"""
        if len(returns) == 0:
            return 0
        risk_free_rate = self.config.get('performance_metrics', {}).get('risk_free_rate', 0.02) / 252
        excess_returns = returns - risk_free_rate
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std()
        return np.sqrt(252) * excess_returns.mean() / downside_std if downside_std > 0 else 0

    def _calculate_max_drawdown(self, equity_series: pd.Series) -> Tuple[float, int]:
        """Calculate maximum drawdown and duration"""
        peak = equity_series.expanding().max()
        drawdown = (equity_series - peak) / peak
        max_drawdown = drawdown.min()

        # Calculate duration
        peak_idx = equity_series.idxmax()
        bottom_idx = drawdown.idxmin()
        if peak_idx < bottom_idx:
            max_drawdown_duration = (bottom_idx - peak_idx).days
        else:
            max_drawdown_duration = 0

        return max_drawdown, max_drawdown_duration

    def _calculate_consecutive_trades(self, trades_df: pd.DataFrame) -> Tuple[int, int]:
        """Calculate consecutive wins and losses"""
        if len(trades_df) == 0:
            return 0, 0

        consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0

        for pnl in trades_df['pnl']:
            if pnl > 0:
                consecutive_wins += 1
                consecutive_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
            else:
                consecutive_losses += 1
                consecutive_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)

        return max_consecutive_wins, max_consecutive_losses


class PerformanceAnalyzer:
    """Comprehensive performance analysis"""

    def __init__(self, config: dict):
        self.config = config
        self.metrics_history = []

    def calculate_metrics(self, positions: List['Position'], trade_history: List[Dict]) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        if not trade_history:
            return self._get_empty_metrics()

        trades_df = pd.DataFrame(trade_history)
        returns = trades_df['pnl'].sum() / 100000  # Assuming 100k account

        metrics = {
            'total_return': returns,
            'total_trades': len(trades_df),
            'win_rate': self._calculate_win_rate(trades_df),
            'profit_factor': self._calculate_profit_factor(trades_df),
            'average_win': trades_df[trades_df['pnl'] > 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] > 0]) > 0 else 0,
            'average_loss': trades_df[trades_df['pnl'] < 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] < 0]) > 0 else 0,
            'largest_win': trades_df['pnl'].max(),
            'largest_loss': trades_df['pnl'].min(),
            'sharpe_ratio': self._calculate_sharpe_ratio(trades_df),
            'sortino_ratio': self._calculate_sortino_ratio(trades_df),
            'max_drawdown': self._calculate_max_drawdown(trades_df),
            'calmar_ratio': self._calculate_calmar_ratio(trades_df),
            'var_95': self._calculate_var(trades_df, 0.95),
            'cvar_95': self._calculate_cvar(trades_df, 0.95),
            'kelly_criterion': self._calculate_kelly_criterion(trades_df),
            'expectancy': self._calculate_expectancy(trades_df),
            'risk_reward_ratio': self._calculate_risk_reward_ratio(trades_df)
        }

        self.metrics_history.append({
            'timestamp': datetime.now(),
            'metrics': metrics
        })

        return metrics

    def get_win_rate(self) -> float:
        """Get the latest win rate from metrics history"""
        if self.metrics_history:
            return self.metrics_history[-1]['metrics'].get('win_rate', 0.0)
        return 0.0

    def _get_empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics structure"""
        return {
            'total_return': 0.0,
            'total_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'average_win': 0.0,
            'average_loss': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            'calmar_ratio': 0.0,
            'var_95': 0.0,
            'cvar_95': 0.0,
            'kelly_criterion': 0.0,
            'expectancy': 0.0,
            'risk_reward_ratio': 0.0
        }

    def _calculate_win_rate(self, trades_df: pd.DataFrame) -> float:
        """Calculate win rate"""
        if len(trades_df) == 0:
            return 0.0
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        return winning_trades / len(trades_df)

    def _calculate_profit_factor(self, trades_df: pd.DataFrame) -> float:
        """Calculate profit factor"""
        winning_trades = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        losing_trades = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
        return winning_trades / losing_trades if losing_trades > 0 else float('inf')

    def _calculate_sharpe_ratio(self, trades_df: pd.DataFrame) -> float:
        """Calculate Sharpe ratio"""
        if len(trades_df) == 0:
            return 0.0
        returns = trades_df['pnl'] / 100000  # Normalize to account size
        risk_free_rate = self.config.get('performance_metrics', {}).get('risk_free_rate', 0.02) / 252
        excess_returns = returns - risk_free_rate
        return np.sqrt(252) * excess_returns.mean() / returns.std() if returns.std() > 0 else 0

    def _calculate_sortino_ratio(self, trades_df: pd.DataFrame) -> float:
        """Calculate Sortino ratio"""
        if len(trades_df) == 0:
            return 0.0
        returns = trades_df['pnl'] / 100000
        risk_free_rate = self.config.get('performance_metrics', {}).get('risk_free_rate', 0.02) / 252
        excess_returns = returns - risk_free_rate
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std()
        return np.sqrt(252) * excess_returns.mean() / downside_std if downside_std > 0 else 0

    def _calculate_max_drawdown(self, trades_df: pd.DataFrame) -> float:
        """Calculate maximum drawdown"""
        if len(trades_df) == 0:
            return 0.0
        cumulative_pnl = trades_df['pnl'].cumsum()
        peak = cumulative_pnl.expanding().max()
        drawdown = (cumulative_pnl - peak) / 100000  # Normalize to account size
        return abs(drawdown.min())

    def _calculate_calmar_ratio(self, trades_df: pd.DataFrame) -> float:
        """Calculate Calmar ratio"""
        if len(trades_df) == 0:
            return 0.0
        total_return = trades_df['pnl'].sum() / 100000
        max_drawdown = self._calculate_max_drawdown(trades_df)
        return total_return / max_drawdown if max_drawdown > 0 else 0

    def _calculate_var(self, trades_df: pd.DataFrame, confidence: float) -> float:
        """Calculate Value at Risk"""
        if len(trades_df) == 0:
            return 0.0
        returns = trades_df['pnl'] / 100000
        return np.percentile(returns, (1 - confidence) * 100)

    def _calculate_cvar(self, trades_df: pd.DataFrame, confidence: float) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        if len(trades_df) == 0:
            return 0.0
        returns = trades_df['pnl'] / 100000
        var = self._calculate_var(trades_df, confidence)
        return returns[returns <= var].mean()

    def _calculate_kelly_criterion(self, trades_df: pd.DataFrame) -> float:
        """Calculate Kelly Criterion"""
        if len(trades_df) == 0:
            return 0.0
        win_rate = self._calculate_win_rate(trades_df)
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] > 0]) > 0 else 0
        avg_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].mean()) if len(trades_df[trades_df['pnl'] < 0]) > 0 else 1

        if avg_loss == 0:
            return 0.0

        return (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win if avg_win > 0 else 0

    def _calculate_expectancy(self, trades_df: pd.DataFrame) -> float:
        """Calculate expectancy per trade"""
        if len(trades_df) == 0:
            return 0.0
        win_rate = self._calculate_win_rate(trades_df)
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] > 0]) > 0 else 0
        avg_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].mean()) if len(trades_df[trades_df['pnl'] < 0]) > 0 else 0

        return win_rate * avg_win - (1 - win_rate) * avg_loss

    def _calculate_risk_reward_ratio(self, trades_df: pd.DataFrame) -> float:
        """Calculate risk-reward ratio"""
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] > 0]) > 0 else 0
        avg_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].mean()) if len(trades_df[trades_df['pnl'] < 0]) > 0 else 1

        return avg_win / avg_loss if avg_loss > 0 else 0
