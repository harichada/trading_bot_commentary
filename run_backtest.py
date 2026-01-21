#!/usr/bin/env python3
"""
Backtest Runner for Profitable Strategies
==========================================
Runs comprehensive backtests on the trading strategies using
simulated market data with realistic characteristics.

Usage:
    python run_backtest.py                    # Run default backtest
    python run_backtest.py --days 90          # Backtest last 90 days
    python run_backtest.py --symbols AAPL MSFT GOOGL  # Specific symbols
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import json
import argparse
import sys

# Try to import strategies
try:
    from profitable_strategies import (
        AdaptiveTrendStrategy,
        MeanReversionWithRegimeStrategy,
        OpeningRangeBreakoutStrategy,
        VWAPReversionStrategy,
        StrategyEnsemble,
        SignalType,
        StrategySignal,
        create_profitable_strategies
    )
    STRATEGIES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import strategies: {e}")
    STRATEGIES_AVAILABLE = False


@dataclass
class BacktestTrade:
    """Record of a completed trade"""
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: int
    side: str  # 'long' or 'short'
    pnl: float
    pnl_pct: float
    strategy: str
    exit_reason: str


@dataclass
class BacktestResults:
    """Complete backtest results"""
    trades: List[BacktestTrade]
    equity_curve: List[Tuple[datetime, float]]

    # Performance metrics
    total_return: float
    annual_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float

    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float

    total_trades: int
    winning_trades: int
    losing_trades: int

    # Period info
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float


class MarketDataGenerator:
    """
    Generates realistic simulated market data with:
    - Trending periods
    - Mean-reverting periods
    - Volatility clustering
    - Opening gaps
    - Volume patterns
    """

    def __init__(self, seed: int = 42):
        np.random.seed(seed)

    def generate(self, symbol: str, start_date: datetime, end_date: datetime,
                 timeframe_minutes: int = 5) -> pd.DataFrame:
        """Generate realistic OHLCV data"""

        # Calculate number of bars (only market hours: 9:30 AM - 4:00 PM = 6.5 hours)
        market_hours_per_day = 6.5
        bars_per_day = int(market_hours_per_day * 60 / timeframe_minutes)

        # Get trading days
        all_dates = pd.date_range(start_date, end_date, freq='B')  # Business days
        total_bars = len(all_dates) * bars_per_day

        # Generate base price with regime changes
        prices = self._generate_price_series(total_bars, base_price=150.0)

        # Generate OHLC from prices
        opens = prices[:-1]
        closes = prices[1:]

        # Add intrabar volatility
        bar_volatility = np.abs(np.random.randn(len(closes))) * 0.002 * closes
        highs = np.maximum(opens, closes) + bar_volatility
        lows = np.minimum(opens, closes) - bar_volatility

        # Generate volume with patterns
        volume = self._generate_volume(len(closes), bars_per_day)

        # Create timestamps
        timestamps = []
        for date in all_dates[:len(closes) // bars_per_day + 1]:
            market_open = datetime.combine(date.date(), datetime.strptime("09:30", "%H:%M").time())
            for i in range(bars_per_day):
                if len(timestamps) >= len(closes):
                    break
                timestamps.append(market_open + timedelta(minutes=i * timeframe_minutes))

        timestamps = timestamps[:len(closes)]

        df = pd.DataFrame({
            'open': opens[:len(timestamps)],
            'high': highs[:len(timestamps)],
            'low': lows[:len(timestamps)],
            'close': closes[:len(timestamps)],
            'volume': volume[:len(timestamps)]
        }, index=pd.DatetimeIndex(timestamps))

        df.attrs['symbol'] = symbol
        return df

    def _generate_price_series(self, n_bars: int, base_price: float) -> np.ndarray:
        """Generate price series with regime changes"""
        prices = [base_price]

        # Regime parameters
        regime = 'trending'
        regime_duration = 0
        trend_direction = 1

        for i in range(n_bars):
            # Check for regime change
            regime_duration += 1
            if regime_duration > np.random.randint(50, 200):
                regime = np.random.choice(['trending', 'mean_reverting', 'volatile'])
                regime_duration = 0
                trend_direction = np.random.choice([-1, 1])

            # Generate return based on regime
            if regime == 'trending':
                drift = 0.0001 * trend_direction
                vol = 0.001
            elif regime == 'mean_reverting':
                # Mean revert to moving average
                ma = np.mean(prices[-20:]) if len(prices) > 20 else prices[-1]
                drift = 0.01 * (ma - prices[-1]) / prices[-1]
                vol = 0.0015
            else:  # volatile
                drift = 0
                vol = 0.003

            ret = drift + vol * np.random.randn()
            new_price = prices[-1] * (1 + ret)
            prices.append(new_price)

        return np.array(prices)

    def _generate_volume(self, n_bars: int, bars_per_day: int) -> np.ndarray:
        """Generate volume with U-shape pattern"""
        volume = []

        for day in range(n_bars // bars_per_day + 1):
            for bar in range(bars_per_day):
                if len(volume) >= n_bars:
                    break

                # U-shape: high at open and close
                time_of_day = bar / bars_per_day
                u_shape = 2 - 4 * abs(time_of_day - 0.5)  # 2 at edges, 0 at middle

                base_volume = 500000 * (1 + u_shape)
                vol = base_volume * (0.5 + np.random.rand())
                volume.append(int(vol))

        return np.array(volume[:n_bars])


class BacktestEngine:
    """
    Simple but realistic backtesting engine.
    """

    def __init__(self, initial_capital: float = 100000,
                 commission_pct: float = 0.001,
                 slippage_pct: float = 0.0005):
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct

    def run(self, strategy, market_data: Dict[str, pd.DataFrame],
            max_positions: int = 3,
            position_size_pct: float = 0.1) -> BacktestResults:
        """Run backtest for a strategy on market data"""

        cash = self.initial_capital
        positions = {}  # symbol -> {quantity, entry_price, entry_time, stop_loss, take_profit, strategy}
        trades = []
        equity_curve = []

        # Get all timestamps across all symbols
        all_timestamps = set()
        for df in market_data.values():
            all_timestamps.update(df.index.tolist())
        timestamps = sorted(all_timestamps)

        # Main loop
        for i, timestamp in enumerate(timestamps):
            # Calculate current equity
            position_value = sum(
                pos['quantity'] * market_data[sym].loc[:timestamp, 'close'].iloc[-1]
                for sym, pos in positions.items()
                if timestamp in market_data[sym].index or any(t <= timestamp for t in market_data[sym].index)
            )
            equity = cash + position_value
            equity_curve.append((timestamp, equity))

            # Process each symbol
            for symbol, df in market_data.items():
                if timestamp not in df.index:
                    continue

                current_price = df.loc[timestamp, 'close']

                # Check stop loss and take profit for existing positions
                if symbol in positions:
                    pos = positions[symbol]

                    # Check stop loss
                    if pos['stop_loss'] and current_price <= pos['stop_loss']:
                        exit_price = pos['stop_loss'] * (1 - self.slippage_pct)
                        pnl = (exit_price - pos['entry_price']) * pos['quantity']
                        pnl -= abs(pnl) * self.commission_pct

                        trades.append(BacktestTrade(
                            symbol=symbol,
                            entry_time=pos['entry_time'],
                            exit_time=timestamp,
                            entry_price=pos['entry_price'],
                            exit_price=exit_price,
                            quantity=pos['quantity'],
                            side='long',
                            pnl=pnl,
                            pnl_pct=pnl / (pos['entry_price'] * pos['quantity']),
                            strategy=pos['strategy'],
                            exit_reason='stop_loss'
                        ))

                        cash += pos['quantity'] * exit_price
                        del positions[symbol]
                        continue

                    # Check take profit
                    if pos['take_profit'] and current_price >= pos['take_profit']:
                        exit_price = pos['take_profit'] * (1 - self.slippage_pct)
                        pnl = (exit_price - pos['entry_price']) * pos['quantity']
                        pnl -= abs(pnl) * self.commission_pct

                        trades.append(BacktestTrade(
                            symbol=symbol,
                            entry_time=pos['entry_time'],
                            exit_time=timestamp,
                            entry_price=pos['entry_price'],
                            exit_price=exit_price,
                            quantity=pos['quantity'],
                            side='long',
                            pnl=pnl,
                            pnl_pct=pnl / (pos['entry_price'] * pos['quantity']),
                            strategy=pos['strategy'],
                            exit_reason='take_profit'
                        ))

                        cash += pos['quantity'] * exit_price
                        del positions[symbol]
                        continue

                # Get enough history for analysis
                if i < 50:
                    continue

                history = df.loc[:timestamp].tail(250)  # Last 250 bars
                history.attrs['symbol'] = symbol

                # Generate signals
                try:
                    signals = strategy.analyze(history, positions)
                except Exception as e:
                    continue

                for signal in signals:
                    if signal.signal_type == SignalType.BUY and symbol not in positions:
                        if len(positions) >= max_positions:
                            continue

                        # Calculate position size
                        position_value = equity * position_size_pct
                        entry_price = current_price * (1 + self.slippage_pct)
                        quantity = int(position_value / entry_price)

                        if quantity <= 0 or quantity * entry_price > cash:
                            continue

                        # Enter position
                        cost = quantity * entry_price * (1 + self.commission_pct)
                        cash -= cost

                        positions[symbol] = {
                            'quantity': quantity,
                            'entry_price': entry_price,
                            'entry_time': timestamp,
                            'stop_loss': signal.stop_loss,
                            'take_profit': signal.take_profit,
                            'strategy': signal.strategy_name
                        }

                    elif signal.signal_type == SignalType.CLOSE_LONG and symbol in positions:
                        pos = positions[symbol]
                        exit_price = current_price * (1 - self.slippage_pct)
                        pnl = (exit_price - pos['entry_price']) * pos['quantity']
                        pnl -= abs(pnl) * self.commission_pct

                        trades.append(BacktestTrade(
                            symbol=symbol,
                            entry_time=pos['entry_time'],
                            exit_time=timestamp,
                            entry_price=pos['entry_price'],
                            exit_price=exit_price,
                            quantity=pos['quantity'],
                            side='long',
                            pnl=pnl,
                            pnl_pct=pnl / (pos['entry_price'] * pos['quantity']),
                            strategy=pos['strategy'],
                            exit_reason='signal'
                        ))

                        cash += pos['quantity'] * exit_price
                        del positions[symbol]

        # Close remaining positions at last price
        final_timestamp = timestamps[-1]
        for symbol, pos in list(positions.items()):
            df = market_data[symbol]
            exit_price = df.iloc[-1]['close'] * (1 - self.slippage_pct)
            pnl = (exit_price - pos['entry_price']) * pos['quantity']
            pnl -= abs(pnl) * self.commission_pct

            trades.append(BacktestTrade(
                symbol=symbol,
                entry_time=pos['entry_time'],
                exit_time=final_timestamp,
                entry_price=pos['entry_price'],
                exit_price=exit_price,
                quantity=pos['quantity'],
                side='long',
                pnl=pnl,
                pnl_pct=pnl / (pos['entry_price'] * pos['quantity']),
                strategy=pos['strategy'],
                exit_reason='end_of_backtest'
            ))

            cash += pos['quantity'] * exit_price

        # Calculate metrics
        final_capital = cash

        return self._calculate_results(
            trades, equity_curve,
            self.initial_capital, final_capital,
            timestamps[0], timestamps[-1]
        )

    def _calculate_results(self, trades: List[BacktestTrade],
                          equity_curve: List[Tuple[datetime, float]],
                          initial_capital: float,
                          final_capital: float,
                          start_date: datetime,
                          end_date: datetime) -> BacktestResults:
        """Calculate performance metrics"""

        total_trades = len(trades)

        if total_trades == 0:
            return BacktestResults(
                trades=[], equity_curve=equity_curve,
                total_return=0, annual_return=0,
                sharpe_ratio=0, sortino_ratio=0, max_drawdown=0,
                win_rate=0, profit_factor=0, avg_win=0, avg_loss=0,
                total_trades=0, winning_trades=0, losing_trades=0,
                start_date=start_date, end_date=end_date,
                initial_capital=initial_capital, final_capital=final_capital
            )

        # Win/loss stats
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl <= 0]

        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0

        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0

        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Returns
        total_return = (final_capital - initial_capital) / initial_capital

        # Annualized return
        days = (end_date - start_date).days
        years = days / 365
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # Equity curve metrics
        equities = [e[1] for e in equity_curve]
        daily_returns = np.diff(equities) / equities[:-1]

        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0

        # Sortino ratio
        negative_returns = daily_returns[daily_returns < 0]
        downside_std = np.std(negative_returns) if len(negative_returns) > 0 else 0.0001
        sortino_ratio = np.mean(daily_returns) / downside_std * np.sqrt(252)

        # Max drawdown
        peak = equities[0]
        max_dd = 0
        for equity in equities:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak
            if dd > max_dd:
                max_dd = dd

        return BacktestResults(
            trades=trades,
            equity_curve=equity_curve,
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_dd,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            total_trades=total_trades,
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            final_capital=final_capital
        )


def print_results(results: BacktestResults, strategy_name: str):
    """Print backtest results in a nice format"""

    print(f"\n{'='*60}")
    print(f"  BACKTEST RESULTS: {strategy_name}")
    print(f"{'='*60}")

    print(f"\n  Period: {results.start_date.strftime('%Y-%m-%d')} to {results.end_date.strftime('%Y-%m-%d')}")
    print(f"  Initial Capital: ${results.initial_capital:,.2f}")
    print(f"  Final Capital:   ${results.final_capital:,.2f}")

    print(f"\n  RETURNS")
    print(f"  {'-'*40}")
    print(f"  Total Return:    {results.total_return:>10.2%}")
    print(f"  Annual Return:   {results.annual_return:>10.2%}")
    print(f"  Max Drawdown:    {results.max_drawdown:>10.2%}")

    print(f"\n  RISK-ADJUSTED")
    print(f"  {'-'*40}")
    print(f"  Sharpe Ratio:    {results.sharpe_ratio:>10.2f}")
    print(f"  Sortino Ratio:   {results.sortino_ratio:>10.2f}")

    print(f"\n  TRADE STATISTICS")
    print(f"  {'-'*40}")
    print(f"  Total Trades:    {results.total_trades:>10}")
    print(f"  Winning Trades:  {results.winning_trades:>10}")
    print(f"  Losing Trades:   {results.losing_trades:>10}")
    print(f"  Win Rate:        {results.win_rate:>10.1%}")
    print(f"  Profit Factor:   {results.profit_factor:>10.2f}")
    print(f"  Avg Win:         ${results.avg_win:>9,.2f}")
    print(f"  Avg Loss:        ${results.avg_loss:>9,.2f}")

    # Show some trades
    if results.trades:
        print(f"\n  SAMPLE TRADES")
        print(f"  {'-'*40}")
        for trade in results.trades[:5]:
            direction = "+" if trade.pnl > 0 else ""
            print(f"  {trade.symbol}: {trade.entry_time.strftime('%m/%d')} -> {trade.exit_time.strftime('%m/%d')} | "
                  f"P&L: {direction}${trade.pnl:,.2f} ({trade.exit_reason})")

    print(f"\n{'='*60}\n")


def run_comprehensive_backtest():
    """Run comprehensive backtest on all strategies"""

    print("\n" + "="*60)
    print("  COMPREHENSIVE STRATEGY BACKTEST")
    print("="*60)
    print(f"\n  Starting backtest at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if not STRATEGIES_AVAILABLE:
        print("\n  ERROR: Strategies not available. Install numpy/pandas first.")
        print("  Run: pip install numpy pandas")
        return None

    # Generate market data
    print("\n  Generating simulated market data...")
    generator = MarketDataGenerator(seed=42)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)  # 3 months

    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']

    market_data = {}
    for symbol in symbols:
        market_data[symbol] = generator.generate(symbol, start_date, end_date)
        print(f"    Generated {len(market_data[symbol])} bars for {symbol}")

    # Initialize backtest engine
    engine = BacktestEngine(
        initial_capital=100000,
        commission_pct=0.001,
        slippage_pct=0.0005
    )

    # Test individual strategies
    strategies = [
        ('Adaptive Trend', AdaptiveTrendStrategy()),
        ('Mean Reversion', MeanReversionWithRegimeStrategy()),
        ('VWAP Reversion', VWAPReversionStrategy()),
    ]

    all_results = {}

    for name, strategy in strategies:
        print(f"\n  Running backtest: {name}...")
        try:
            results = engine.run(strategy, market_data, max_positions=3, position_size_pct=0.10)
            all_results[name] = results
            print_results(results, name)
        except Exception as e:
            print(f"    ERROR: {e}")

    # Test ensemble strategy
    print("\n  Running backtest: Strategy Ensemble...")
    try:
        ensemble = create_profitable_strategies()
        ensemble.min_consensus = 2  # Require 2 strategies to agree
        results = engine.run(ensemble, market_data, max_positions=3, position_size_pct=0.10)
        all_results['Ensemble'] = results
        print_results(results, 'Strategy Ensemble (Consensus)')
    except Exception as e:
        print(f"    ERROR: {e}")

    # Summary comparison
    print("\n" + "="*60)
    print("  STRATEGY COMPARISON SUMMARY")
    print("="*60)
    print(f"\n  {'Strategy':<20} {'Return':>10} {'Sharpe':>8} {'MaxDD':>8} {'WinRate':>8} {'Trades':>8}")
    print(f"  {'-'*20} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for name, results in all_results.items():
        print(f"  {name:<20} {results.total_return:>10.1%} {results.sharpe_ratio:>8.2f} "
              f"{results.max_drawdown:>8.1%} {results.win_rate:>8.1%} {results.total_trades:>8}")

    print(f"\n{'='*60}")

    # Recommendation
    best_sharpe = max(all_results.items(), key=lambda x: x[1].sharpe_ratio)
    print(f"\n  RECOMMENDATION: {best_sharpe[0]}")
    print(f"  Reason: Highest risk-adjusted return (Sharpe: {best_sharpe[1].sharpe_ratio:.2f})")
    print(f"\n{'='*60}\n")

    return all_results


def main():
    parser = argparse.ArgumentParser(description='Run strategy backtest')
    parser.add_argument('--days', type=int, default=90, help='Number of days to backtest')
    parser.add_argument('--capital', type=float, default=100000, help='Initial capital')
    parser.add_argument('--symbols', nargs='+', default=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'],
                        help='Symbols to test')
    args = parser.parse_args()

    results = run_comprehensive_backtest()

    if results:
        # Save results to file
        output = {
            name: {
                'total_return': r.total_return,
                'annual_return': r.annual_return,
                'sharpe_ratio': r.sharpe_ratio,
                'sortino_ratio': r.sortino_ratio,
                'max_drawdown': r.max_drawdown,
                'win_rate': r.win_rate,
                'profit_factor': r.profit_factor,
                'total_trades': r.total_trades,
                'final_capital': r.final_capital
            }
            for name, r in results.items()
        }

        with open('backtest_results.json', 'w') as f:
            json.dump(output, f, indent=2)

        print("Results saved to backtest_results.json")


if __name__ == "__main__":
    main()
