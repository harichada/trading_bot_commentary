#!/usr/bin/env python3
"""
Performance Analytics and Reporting System
Comprehensive performance tracking, analysis, and reporting
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
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

logger = logging.getLogger('PerformanceAnalytics')

class PerformancePeriod(Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    ALL_TIME = "all_time"

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    # Returns
    total_return: float
    annual_return: float
    monthly_return: float
    daily_return: float
    
    # Risk metrics
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Drawdown metrics
    max_drawdown: float
    max_drawdown_duration: int
    current_drawdown: float
    avg_drawdown: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    profit_factor: float
    expectancy: float
    
    # Risk-adjusted metrics
    risk_adjusted_return: float
    information_ratio: float
    treynor_ratio: float
    
    # Additional metrics
    trades_per_day: float
    avg_holding_period: float
    kelly_percentage: float
    recovery_factor: float
    payoff_ratio: float

@dataclass
class StrategyPerformance:
    """Performance metrics for individual strategies"""
    strategy_name: str
    metrics: PerformanceMetrics
    contribution_to_pnl: float
    signal_accuracy: float
    best_market_conditions: Dict[str, Any]
    worst_market_conditions: Dict[str, Any]

@dataclass
class SymbolPerformance:
    """Performance metrics for individual symbols"""
    symbol: str
    metrics: PerformanceMetrics
    total_pnl: float
    best_trade: Dict[str, Any]
    worst_trade: Dict[str, Any]
    avg_spread_cost: float
    avg_slippage: float

class PerformanceAnalyzer:
    """Analyzes trading performance with detailed metrics"""
    
    def __init__(self):
        self.trades: List[Dict[str, Any]] = []
        self.equity_curve: pd.Series = pd.Series()
        self.positions_history: List[Dict[str, Any]] = []
        self.daily_pnl: pd.Series = pd.Series()
        
    def add_trade(self, trade: Dict[str, Any]):
        """Add a completed trade for analysis"""
        self.trades.append(trade)
        
    def update_equity(self, timestamp: datetime, equity: float):
        """Update equity curve"""
        if self.equity_curve.empty:
            self.equity_curve = pd.Series([equity], index=[timestamp])
        else:
            self.equity_curve[timestamp] = equity
    
    def calculate_metrics(self, period: PerformancePeriod = PerformancePeriod.ALL_TIME) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        if not self.trades or self.equity_curve.empty:
            return self._empty_metrics()
        
        # Filter data by period
        start_date = self._get_period_start_date(period)
        period_trades = [t for t in self.trades if t['exit_time'] >= start_date]
        period_equity = self.equity_curve[self.equity_curve.index >= start_date]
        
        if not period_trades or period_equity.empty:
            return self._empty_metrics()
        
        # Calculate returns
        returns = period_equity.pct_change().dropna()
        total_return = (period_equity.iloc[-1] - period_equity.iloc[0]) / period_equity.iloc[0]
        
        # Annualized metrics
        days = (period_equity.index[-1] - period_equity.index[0]).days
        annual_factor = 365 / days if days > 0 else 0
        annual_return = (1 + total_return) ** annual_factor - 1
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(252)
        sharpe = self._calculate_sharpe_ratio(returns)
        sortino = self._calculate_sortino_ratio(returns)
        calmar = self._calculate_calmar_ratio(annual_return, period_equity)
        
        # Drawdown analysis
        dd_stats = self._calculate_drawdown_stats(period_equity)
        
        # Trade statistics
        trade_stats = self._calculate_trade_stats(period_trades)
        
        # Risk-adjusted returns
        risk_adjusted = annual_return / volatility if volatility > 0 else 0
        info_ratio = self._calculate_information_ratio(returns)
        treynor = self._calculate_treynor_ratio(returns, annual_return)
        
        # Additional metrics
        trades_per_day = len(period_trades) / max(days, 1)
        avg_holding = self._calculate_avg_holding_period(period_trades)
        kelly = self._calculate_kelly_percentage(trade_stats)
        recovery = self._calculate_recovery_factor(total_return, dd_stats['max_drawdown'])
        
        return PerformanceMetrics(
            total_return=total_return,
            annual_return=annual_return,
            monthly_return=total_return / max(days / 30, 1),
            daily_return=returns.mean(),
            volatility=volatility,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=dd_stats['max_drawdown'],
            max_drawdown_duration=dd_stats['max_duration'],
            current_drawdown=dd_stats['current_drawdown'],
            avg_drawdown=dd_stats['avg_drawdown'],
            **trade_stats,
            risk_adjusted_return=risk_adjusted,
            information_ratio=info_ratio,
            treynor_ratio=treynor,
            trades_per_day=trades_per_day,
            avg_holding_period=avg_holding,
            kelly_percentage=kelly,
            recovery_factor=recovery,
            payoff_ratio=trade_stats['avg_win'] / abs(trade_stats['avg_loss']) if trade_stats['avg_loss'] != 0 else 0
        )
    
    def analyze_by_strategy(self) -> Dict[str, StrategyPerformance]:
        """Analyze performance by strategy"""
        strategy_trades = {}
        
        # Group trades by strategy
        for trade in self.trades:
            strategy = trade.get('strategy', 'unknown')
            if strategy not in strategy_trades:
                strategy_trades[strategy] = []
            strategy_trades[strategy].append(trade)
        
        results = {}
        total_pnl = sum(t['pnl'] for t in self.trades)
        
        for strategy, trades in strategy_trades.items():
            # Create temporary analyzer for this strategy
            temp_analyzer = PerformanceAnalyzer()
            temp_analyzer.trades = trades
            
            # Calculate metrics
            metrics = temp_analyzer.calculate_metrics()
            
            # Calculate contribution
            strategy_pnl = sum(t['pnl'] for t in trades)
            contribution = strategy_pnl / total_pnl if total_pnl != 0 else 0
            
            # Analyze market conditions
            best_conditions = self._analyze_best_conditions(trades)
            worst_conditions = self._analyze_worst_conditions(trades)
            
            # Signal accuracy
            accuracy = len([t for t in trades if t['pnl'] > 0]) / len(trades) if trades else 0
            
            results[strategy] = StrategyPerformance(
                strategy_name=strategy,
                metrics=metrics,
                contribution_to_pnl=contribution,
                signal_accuracy=accuracy,
                best_market_conditions=best_conditions,
                worst_market_conditions=worst_conditions
            )
        
        return results
    
    def analyze_by_symbol(self) -> Dict[str, SymbolPerformance]:
        """Analyze performance by symbol"""
        symbol_trades = {}
        
        # Group trades by symbol
        for trade in self.trades:
            symbol = trade['symbol']
            if symbol not in symbol_trades:
                symbol_trades[symbol] = []
            symbol_trades[symbol].append(trade)
        
        results = {}
        
        for symbol, trades in symbol_trades.items():
            # Create temporary analyzer
            temp_analyzer = PerformanceAnalyzer()
            temp_analyzer.trades = trades
            
            # Calculate metrics
            metrics = temp_analyzer.calculate_metrics()
            
            # Total P&L
            total_pnl = sum(t['pnl'] for t in trades)
            
            # Best and worst trades
            best_trade = max(trades, key=lambda t: t['pnl'])
            worst_trade = min(trades, key=lambda t: t['pnl'])
            
            # Transaction costs
            avg_spread = np.mean([t.get('spread_cost', 0) for t in trades])
            avg_slippage = np.mean([t.get('slippage', 0) for t in trades])
            
            results[symbol] = SymbolPerformance(
                symbol=symbol,
                metrics=metrics,
                total_pnl=total_pnl,
                best_trade=best_trade,
                worst_trade=worst_trade,
                avg_spread_cost=avg_spread,
                avg_slippage=avg_slippage
            )
        
        return results
    
    def generate_report(self, output_path: str = "performance_report.html"):
        """Generate comprehensive HTML performance report"""
        metrics = self.calculate_metrics()
        strategy_performance = self.analyze_by_strategy()
        symbol_performance = self.analyze_by_symbol()
        
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Trading Performance Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; }}
                .section {{ background: white; padding: 25px; margin: 20px 0; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }}
                .metric-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; }}
                .metric-value {{ font-size: 28px; font-weight: bold; margin: 10px 0; }}
                .metric-label {{ color: #666; font-size: 14px; }}
                .positive {{ color: #4CAF50; }}
                .negative {{ color: #f44336; }}
                .chart-container {{ height: 400px; margin: 20px 0; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f8f9fa; font-weight: 600; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .strategy-card {{ background: #f8f9fa; padding: 20px; margin: 10px 0; border-radius: 8px; }}
                .strategy-name {{ font-size: 18px; font-weight: bold; margin-bottom: 10px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Trading Performance Report</h1>
                <p>Generated on {report_date}</p>
                <p>Period: {period_start} to {period_end}</p>
            </div>
            
            <div class="section">
                <h2>Performance Summary</h2>
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
                        <div class="metric-label">Expectancy</div>
                        <div class="metric-value {expectancy_class}">${expectancy:.2f}</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>Equity Curve</h2>
                <div id="equity-curve" class="chart-container"></div>
            </div>
            
            <div class="section">
                <h2>Drawdown Analysis</h2>
                <div id="drawdown-chart" class="chart-container"></div>
            </div>
            
            <div class="section">
                <h2>Monthly Returns Heatmap</h2>
                <div id="monthly-heatmap" class="chart-container"></div>
            </div>
            
            <div class="section">
                <h2>Strategy Performance</h2>
                {strategy_performance_html}
            </div>
            
            <div class="section">
                <h2>Symbol Performance</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Symbol</th>
                            <th>Total P&L</th>
                            <th>Win Rate</th>
                            <th>Avg Win</th>
                            <th>Avg Loss</th>
                            <th>Sharpe Ratio</th>
                            <th>Total Trades</th>
                        </tr>
                    </thead>
                    <tbody>
                        {symbol_performance_rows}
                    </tbody>
                </table>
            </div>
            
            <div class="section">
                <h2>Trade Distribution</h2>
                <div id="trade-distribution" class="chart-container"></div>
            </div>
            
            <div class="section">
                <h2>Risk Analysis</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-label">Volatility (Annual)</div>
                        <div class="metric-value">{volatility:.2%}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Sortino Ratio</div>
                        <div class="metric-value">{sortino_ratio:.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Calmar Ratio</div>
                        <div class="metric-value">{calmar_ratio:.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Risk-Adjusted Return</div>
                        <div class="metric-value">{risk_adjusted_return:.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Kelly %</div>
                        <div class="metric-value">{kelly_percentage:.1%}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Recovery Factor</div>
                        <div class="metric-value">{recovery_factor:.2f}</div>
                    </div>
                </div>
            </div>
            
            <script>
                {chart_scripts}
            </script>
        </body>
        </html>
        """
        
        # Generate chart scripts
        chart_scripts = self._generate_chart_scripts()
        
        # Generate strategy performance HTML
        strategy_html = self._generate_strategy_performance_html(strategy_performance)
        
        # Generate symbol performance rows
        symbol_rows = self._generate_symbol_performance_rows(symbol_performance)
        
        # Fill template
        html = html_template.format(
            report_date=datetime.now().strftime('%Y-%m-%d %H:%M'),
            period_start=self.equity_curve.index[0].strftime('%Y-%m-%d') if not self.equity_curve.empty else 'N/A',
            period_end=self.equity_curve.index[-1].strftime('%Y-%m-%d') if not self.equity_curve.empty else 'N/A',
            total_return=metrics.total_return,
            annual_return=metrics.annual_return,
            sharpe_ratio=metrics.sharpe_ratio,
            max_drawdown=metrics.max_drawdown,
            win_rate=metrics.win_rate,
            profit_factor=metrics.profit_factor,
            total_trades=metrics.total_trades,
            expectancy=metrics.expectancy,
            volatility=metrics.volatility,
            sortino_ratio=metrics.sortino_ratio,
            calmar_ratio=metrics.calmar_ratio,
            risk_adjusted_return=metrics.risk_adjusted_return,
            kelly_percentage=metrics.kelly_percentage,
            recovery_factor=metrics.recovery_factor,
            return_class='positive' if metrics.total_return > 0 else 'negative',
            expectancy_class='positive' if metrics.expectancy > 0 else 'negative',
            strategy_performance_html=strategy_html,
            symbol_performance_rows=symbol_rows,
            chart_scripts=chart_scripts
        )
        
        # Save report
        with open(output_path, 'w') as f:
            f.write(html)
        
        logger.info(f"Performance report saved to {output_path}")
    
    def _get_period_start_date(self, period: PerformancePeriod) -> datetime:
        """Get start date for performance period"""
        if not self.equity_curve.empty:
            end_date = self.equity_curve.index[-1]
        else:
            end_date = datetime.now()
        
        if period == PerformancePeriod.DAILY:
            return end_date - timedelta(days=1)
        elif period == PerformancePeriod.WEEKLY:
            return end_date - timedelta(weeks=1)
        elif period == PerformancePeriod.MONTHLY:
            return end_date - timedelta(days=30)
        elif period == PerformancePeriod.QUARTERLY:
            return end_date - timedelta(days=90)
        elif period == PerformancePeriod.YEARLY:
            return end_date - timedelta(days=365)
        else:  # ALL_TIME
            return datetime.min
    
    def _empty_metrics(self) -> PerformanceMetrics:
        """Return empty metrics when no data available"""
        return PerformanceMetrics(
            total_return=0, annual_return=0, monthly_return=0, daily_return=0,
            volatility=0, sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0,
            max_drawdown=0, max_drawdown_duration=0, current_drawdown=0, avg_drawdown=0,
            total_trades=0, winning_trades=0, losing_trades=0, win_rate=0,
            avg_win=0, avg_loss=0, largest_win=0, largest_loss=0,
            profit_factor=0, expectancy=0, risk_adjusted_return=0,
            information_ratio=0, treynor_ratio=0, trades_per_day=0,
            avg_holding_period=0, kelly_percentage=0, recovery_factor=0, payoff_ratio=0
        )
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0 or returns.std() == 0:
            return 0
        
        excess_returns = returns - risk_free_rate / 252
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    
    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio"""
        if len(returns) == 0:
            return 0
        
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0
        
        return np.sqrt(252) * excess_returns.mean() / downside_returns.std()
    
    def _calculate_calmar_ratio(self, annual_return: float, equity_curve: pd.Series) -> float:
        """Calculate Calmar ratio"""
        if equity_curve.empty:
            return 0
        
        max_dd = self._calculate_drawdown_stats(equity_curve)['max_drawdown']
        
        if max_dd == 0:
            return 0
        
        return annual_return / abs(max_dd)
    
    def _calculate_drawdown_stats(self, equity_curve: pd.Series) -> Dict[str, Any]:
        """Calculate drawdown statistics"""
        if equity_curve.empty:
            return {'max_drawdown': 0, 'max_duration': 0, 'current_drawdown': 0, 'avg_drawdown': 0}
        
        # Calculate rolling maximum
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max
        
        # Max drawdown
        max_drawdown = drawdown.min()
        
        # Current drawdown
        current_drawdown = drawdown.iloc[-1]
        
        # Average drawdown
        avg_drawdown = drawdown[drawdown < 0].mean() if len(drawdown[drawdown < 0]) > 0 else 0
        
        # Max duration
        duration = 0
        max_duration = 0
        
        for dd in drawdown:
            if dd < 0:
                duration += 1
                max_duration = max(max_duration, duration)
            else:
                duration = 0
        
        return {
            'max_drawdown': abs(max_drawdown),
            'max_duration': max_duration,
            'current_drawdown': abs(current_drawdown),
            'avg_drawdown': abs(avg_drawdown)
        }
    
    def _calculate_trade_stats(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate trade statistics"""
        if not trades:
            return {
                'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0,
                'win_rate': 0, 'avg_win': 0, 'avg_loss': 0,
                'largest_win': 0, 'largest_loss': 0,
                'profit_factor': 0, 'expectancy': 0
            }
        
        pnls = [t['pnl'] for t in trades]
        winning_trades = [p for p in pnls if p > 0]
        losing_trades = [p for p in pnls if p <= 0]
        
        total_trades = len(trades)
        num_winners = len(winning_trades)
        num_losers = len(losing_trades)
        
        win_rate = num_winners / total_trades if total_trades > 0 else 0
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
        
        largest_win = max(pnls) if pnls else 0
        largest_loss = min(pnls) if pnls else 0
        
        gross_profit = sum(winning_trades) if winning_trades else 0
        gross_loss = abs(sum(losing_trades)) if losing_trades else 0
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        expectancy = np.mean(pnls) if pnls else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': num_winners,
            'losing_trades': num_losers,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'profit_factor': profit_factor,
            'expectancy': expectancy
        }
    
    def _calculate_information_ratio(self, returns: pd.Series, benchmark_returns: Optional[pd.Series] = None) -> float:
        """Calculate Information ratio"""
        if benchmark_returns is None:
            # Use 0 as benchmark (absolute returns)
            active_returns = returns
        else:
            active_returns = returns - benchmark_returns
        
        if len(active_returns) == 0 or active_returns.std() == 0:
            return 0
        
        return np.sqrt(252) * active_returns.mean() / active_returns.std()
    
    def _calculate_treynor_ratio(self, returns: pd.Series, annual_return: float, beta: float = 1.0) -> float:
        """Calculate Treynor ratio"""
        if beta == 0:
            return 0
        
        risk_free_rate = 0.02  # 2% annual
        return (annual_return - risk_free_rate) / beta
    
    def _calculate_avg_holding_period(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate average holding period in hours"""
        if not trades:
            return 0
        
        holding_periods = []
        for trade in trades:
            if 'entry_time' in trade and 'exit_time' in trade:
                period = (trade['exit_time'] - trade['entry_time']).total_seconds() / 3600
                holding_periods.append(period)
        
        return np.mean(holding_periods) if holding_periods else 0
    
    def _calculate_kelly_percentage(self, trade_stats: Dict[str, Any]) -> float:
        """Calculate Kelly percentage for position sizing"""
        win_rate = trade_stats['win_rate']
        avg_win = trade_stats['avg_win']
        avg_loss = abs(trade_stats['avg_loss'])
        
        if avg_loss == 0:
            return 0
        
        win_loss_ratio = avg_win / avg_loss
        kelly = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        
        # Apply Kelly fraction for safety (typically 25% of full Kelly)
        return max(0, min(kelly * 0.25, 0.25))
    
    def _calculate_recovery_factor(self, total_return: float, max_drawdown: float) -> float:
        """Calculate recovery factor"""
        if max_drawdown == 0:
            return 0
        
        return total_return / max_drawdown
    
    def _analyze_best_conditions(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze market conditions for best trades"""
        if not trades:
            return {}
        
        # Sort trades by P&L
        sorted_trades = sorted(trades, key=lambda t: t['pnl'], reverse=True)
        
        # Analyze top 20% of trades
        top_trades = sorted_trades[:max(1, len(sorted_trades) // 5)]
        
        # Extract common conditions (simplified)
        conditions = {
            'avg_volatility': np.mean([t.get('volatility', 0) for t in top_trades]),
            'common_time_of_day': self._most_common_hour([t['entry_time'].hour for t in top_trades if 'entry_time' in t]),
            'avg_holding_period': np.mean([(t['exit_time'] - t['entry_time']).total_seconds() / 3600 
                                         for t in top_trades if 'entry_time' in t and 'exit_time' in t])
        }
        
        return conditions
    
    def _analyze_worst_conditions(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze market conditions for worst trades"""
        if not trades:
            return {}
        
        # Sort trades by P&L (ascending for worst)
        sorted_trades = sorted(trades, key=lambda t: t['pnl'])
        
        # Analyze bottom 20% of trades
        bottom_trades = sorted_trades[:max(1, len(sorted_trades) // 5)]
        
        # Extract common conditions
        conditions = {
            'avg_volatility': np.mean([t.get('volatility', 0) for t in bottom_trades]),
            'common_time_of_day': self._most_common_hour([t['entry_time'].hour for t in bottom_trades if 'entry_time' in t]),
            'avg_holding_period': np.mean([(t['exit_time'] - t['entry_time']).total_seconds() / 3600 
                                         for t in bottom_trades if 'entry_time' in t and 'exit_time' in t])
        }
        
        return conditions
    
    def _most_common_hour(self, hours: List[int]) -> int:
        """Find most common hour"""
        if not hours:
            return 0
        
        return max(set(hours), key=hours.count)
    
    def _generate_chart_scripts(self) -> str:
        """Generate JavaScript for charts"""
        if self.equity_curve.empty:
            return ""
        
        # Prepare data
        dates = [d.strftime('%Y-%m-%d') for d in self.equity_curve.index]
        equity_values = self.equity_curve.values.tolist()
        
        # Calculate drawdown
        rolling_max = self.equity_curve.expanding().max()
        drawdown = ((self.equity_curve - rolling_max) / rolling_max * 100).values.tolist()
        
        # Monthly returns for heatmap
        monthly_returns = self._calculate_monthly_returns_matrix()
        
        # Trade distribution
        trade_pnls = [t['pnl'] for t in self.trades]
        
        scripts = f"""
        // Equity Curve
        var equityTrace = {{
            x: {dates},
            y: {equity_values},
            type: 'scatter',
            mode: 'lines',
            name: 'Equity',
            line: {{ color: '#667eea' }}
        }};
        
        var equityLayout = {{
            title: 'Equity Curve',
            xaxis: {{ title: 'Date' }},
            yaxis: {{ title: 'Equity ($)' }},
            showlegend: false
        }};
        
        Plotly.newPlot('equity-curve', [equityTrace], equityLayout);
        
        // Drawdown Chart
        var drawdownTrace = {{
            x: {dates},
            y: {drawdown},
            type: 'scatter',
            mode: 'lines',
            fill: 'tozeroy',
            name: 'Drawdown',
            line: {{ color: '#f44336' }}
        }};
        
        var drawdownLayout = {{
            title: 'Drawdown %',
            xaxis: {{ title: 'Date' }},
            yaxis: {{ title: 'Drawdown (%)' }},
            showlegend: false
        }};
        
        Plotly.newPlot('drawdown-chart', [drawdownTrace], drawdownLayout);
        
        // Monthly Heatmap
        var heatmapData = {{
            z: {monthly_returns['values']},
            x: {monthly_returns['months']},
            y: {monthly_returns['years']},
            type: 'heatmap',
            colorscale: 'RdYlGn',
            zmid: 0
        }};
        
        var heatmapLayout = {{
            title: 'Monthly Returns Heatmap',
            xaxis: {{ title: 'Month' }},
            yaxis: {{ title: 'Year' }}
        }};
        
        Plotly.newPlot('monthly-heatmap', [heatmapData], heatmapLayout);
        
        // Trade Distribution
        var tradeTrace = {{
            x: {trade_pnls},
            type: 'histogram',
            name: 'Trade P&L',
            marker: {{ color: '#667eea' }}
        }};
        
        var tradeLayout = {{
            title: 'Trade P&L Distribution',
            xaxis: {{ title: 'P&L ($)' }},
            yaxis: {{ title: 'Frequency' }},
            showlegend: false
        }};
        
        Plotly.newPlot('trade-distribution', [tradeTrace], tradeLayout);
        """
        
        return scripts
    
    def _calculate_monthly_returns_matrix(self) -> Dict[str, Any]:
        """Calculate monthly returns for heatmap"""
        if self.equity_curve.empty:
            return {'years': [], 'months': [], 'values': [[]]}
        
        # Resample to monthly
        monthly_equity = self.equity_curve.resample('M').last()
        monthly_returns = monthly_equity.pct_change().dropna()
        
        # Create matrix
        years = sorted(list(set(d.year for d in monthly_returns.index)))
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        matrix = []
        for year in years:
            year_returns = []
            for month in range(1, 13):
                try:
                    ret = monthly_returns[
                        (monthly_returns.index.year == year) & 
                        (monthly_returns.index.month == month)
                    ].iloc[0] * 100
                except:
                    ret = 0
                year_returns.append(ret)
            matrix.append(year_returns)
        
        return {
            'years': years,
            'months': months,
            'values': matrix
        }
    
    def _generate_strategy_performance_html(self, strategy_performance: Dict[str, StrategyPerformance]) -> str:
        """Generate HTML for strategy performance section"""
        if not strategy_performance:
            return "<p>No strategy data available</p>"
        
        html_parts = []
        
        for strategy_name, perf in strategy_performance.items():
            html = f"""
            <div class="strategy-card">
                <div class="strategy-name">{strategy_name}</div>
                <div class="metrics-grid">
                    <div>
                        <strong>Total Return:</strong> 
                        <span class="{'positive' if perf.metrics.total_return > 0 else 'negative'}">
                            {perf.metrics.total_return:.2%}
                        </span>
                    </div>
                    <div><strong>Win Rate:</strong> {perf.metrics.win_rate:.1%}</div>
                    <div><strong>Sharpe Ratio:</strong> {perf.metrics.sharpe_ratio:.2f}</div>
                    <div><strong>P&L Contribution:</strong> {perf.contribution_to_pnl:.1%}</div>
                    <div><strong>Signal Accuracy:</strong> {perf.signal_accuracy:.1%}</div>
                    <div><strong>Total Trades:</strong> {perf.metrics.total_trades}</div>
                </div>
            </div>
            """
            html_parts.append(html)
        
        return '\n'.join(html_parts)
    
    def _generate_symbol_performance_rows(self, symbol_performance: Dict[str, SymbolPerformance]) -> str:
        """Generate table rows for symbol performance"""
        if not symbol_performance:
            return "<tr><td colspan='7'>No symbol data available</td></tr>"
        
        rows = []
        
        for symbol, perf in symbol_performance.items():
            row = f"""
            <tr>
                <td>{symbol}</td>
                <td class="{'positive' if perf.total_pnl > 0 else 'negative'}">${perf.total_pnl:.2f}</td>
                <td>{perf.metrics.win_rate:.1%}</td>
                <td>${perf.metrics.avg_win:.2f}</td>
                <td>${abs(perf.metrics.avg_loss):.2f}</td>
                <td>{perf.metrics.sharpe_ratio:.2f}</td>
                <td>{perf.metrics.total_trades}</td>
            </tr>
            """
            rows.append(row)
        
        return '\n'.join(rows)

class PerformanceTracker:
    """Real-time performance tracking"""
    
    def __init__(self, analyzer: PerformanceAnalyzer):
        self.analyzer = analyzer
        self.real_time_metrics = {}
        self.alerts = []
        
    def update_metrics(self):
        """Update real-time performance metrics"""
        # Calculate current metrics
        metrics = self.analyzer.calculate_metrics(PerformancePeriod.DAILY)
        
        # Check for alerts
        self._check_performance_alerts(metrics)
        
        # Update real-time metrics
        self.real_time_metrics = {
            'daily_pnl': metrics.daily_return,
            'daily_trades': metrics.trades_per_day,
            'current_drawdown': metrics.current_drawdown,
            'win_rate_today': metrics.win_rate,
            'sharpe_30d': self.analyzer.calculate_metrics(PerformancePeriod.MONTHLY).sharpe_ratio
        }
    
    def _check_performance_alerts(self, metrics: PerformanceMetrics):
        """Check for performance-based alerts"""
        # Drawdown alert
        if metrics.current_drawdown > 0.05:  # 5% drawdown
            self.alerts.append({
                'type': 'drawdown',
                'message': f'Current drawdown: {metrics.current_drawdown:.2%}',
                'severity': 'high',
                'timestamp': datetime.now()
            })
        
        # Low win rate alert
        if metrics.win_rate < 0.4 and metrics.total_trades > 10:
            self.alerts.append({
                'type': 'win_rate',
                'message': f'Low win rate: {metrics.win_rate:.1%}',
                'severity': 'medium',
                'timestamp': datetime.now()
            })
        
        # Negative expectancy alert
        if metrics.expectancy < 0:
            self.alerts.append({
                'type': 'expectancy',
                'message': f'Negative expectancy: ${metrics.expectancy:.2f}',
                'severity': 'high',
                'timestamp': datetime.now()
            })
    
    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """Get metrics for dashboard display"""
        return {
            'real_time': self.real_time_metrics,
            'alerts': self.alerts[-10:],  # Last 10 alerts
            'summary': {
                'today': self.analyzer.calculate_metrics(PerformancePeriod.DAILY),
                'week': self.analyzer.calculate_metrics(PerformancePeriod.WEEKLY),
                'month': self.analyzer.calculate_metrics(PerformancePeriod.MONTHLY),
                'year': self.analyzer.calculate_metrics(PerformancePeriod.YEARLY)
            }
        }