#!/usr/bin/env python3
"""
Backtesting framework for trading strategies
Tests strategies on historical data before live trading
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import yfinance as yf
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class StrategyBacktester:
    """Backtest trading strategies with realistic constraints"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}
        self.trade_history = []
        self.daily_trades = 0
        self.daily_pnl = 0
        self.consecutive_losses = 0
        
    def download_data(self, symbol: str, period: str = "1mo") -> pd.DataFrame:
        """Download historical data for backtesting"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval="5m")
            
            if data.empty:
                print(f"No data available for {symbol}")
                return pd.DataFrame()
            
            # Calculate technical indicators
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['SMA_50'] = data['Close'].rolling(window=50).mean()
            data['RSI'] = self.calculate_rsi(data['Close'])
            data['ATR'] = self.calculate_atr(data)
            
            # Bollinger Bands
            data['BB_middle'] = data['Close'].rolling(window=20).mean()
            bb_std = data['Close'].rolling(window=20).std()
            data['BB_upper'] = data['BB_middle'] + (bb_std * 2)
            data['BB_lower'] = data['BB_middle'] - (bb_std * 2)
            
            return data
            
        except Exception as e:
            print(f"Error downloading data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr
    
    def check_entry_signal(self, row: pd.Series, strategy: str) -> Tuple[bool, str]:
        """Check if entry conditions are met for given strategy"""
        
        # Skip if missing data
        if pd.isna(row['RSI']) or pd.isna(row['BB_upper']):
            return False, ""
        
        # Price filter - no penny stocks
        if row['Close'] < 5:
            return False, ""
        
        # Time filter - no trades in first 15 min or last 30 min
        hour = row.name.hour
        minute = row.name.minute
        if (hour == 9 and minute < 45) or (hour == 15 and minute >= 30):
            return False, ""
        
        if strategy == "momentum":
            # Momentum: Price above SMA20, RSI > 50 but < 70
            if (row['Close'] > row['SMA_20'] and 
                50 < row['RSI'] < 70 and
                row['Close'] > row['Open']):
                return True, "BUY"
                
        elif strategy == "mean_reversion":
            # Mean Reversion: Price below BB lower, RSI < 30
            if (row['Close'] < row['BB_lower'] and 
                row['RSI'] < 30):
                return True, "BUY"
            # Short opportunity
            elif (row['Close'] > row['BB_upper'] and 
                  row['RSI'] > 70):
                return True, "SELL"
                
        elif strategy == "breakout":
            # Breakout: Price breaks above BB upper with volume
            if (row['Close'] > row['BB_upper'] and 
                row['Volume'] > row['Volume'].rolling(20).mean() * 1.5):
                return True, "BUY"
        
        return False, ""
    
    def calculate_position_size(self, price: float) -> int:
        """Calculate position size with risk management"""
        # Risk 2% per trade
        risk_amount = self.capital * 0.02
        
        # Penny stock adjustment
        if price < 5:
            risk_amount *= 0.5
        
        # Position size based on stop loss distance (3% default)
        stop_distance = price * 0.03
        shares = int(risk_amount / stop_distance)
        
        # Max position value check
        max_position = self.capital * 0.25
        max_shares = int(max_position / price)
        
        return min(shares, max_shares, 1000)  # Cap at 1000 shares
    
    def execute_backtest(self, symbol: str, strategy: str, data: pd.DataFrame) -> Dict:
        """Run backtest for a specific strategy"""
        results = {
            'symbol': symbol,
            'strategy': strategy,
            'trades': [],
            'metrics': {}
        }
        
        position = None
        
        for idx, row in data.iterrows():
            # Reset daily counters at market open
            if row.name.hour == 9 and row.name.minute == 30:
                self.daily_trades = 0
                self.daily_pnl = 0
            
            # Check daily loss limit (5%)
            if self.daily_pnl < -self.initial_capital * 0.05:
                continue
            
            # Position management
            if position:
                # Check exit conditions
                exit_price = None
                exit_reason = ""
                
                # Stop loss
                if position['side'] == 'long' and row['Low'] <= position['stop_loss']:
                    exit_price = position['stop_loss']
                    exit_reason = "stop_loss"
                
                # Take profit
                elif position['side'] == 'long' and row['High'] >= position['take_profit']:
                    exit_price = position['take_profit']
                    exit_reason = "take_profit"
                
                # Time exit - close before market close
                elif row.name.hour == 15 and row.name.minute >= 45:
                    exit_price = row['Close']
                    exit_reason = "eod_exit"
                
                # Exit if conditions met
                if exit_price:
                    pnl = (exit_price - position['entry_price']) * position['shares']
                    self.capital += pnl
                    self.daily_pnl += pnl
                    
                    # Track consecutive losses
                    if pnl < 0:
                        self.consecutive_losses += 1
                    else:
                        self.consecutive_losses = 0
                    
                    # Record trade
                    trade = {
                        'entry_time': position['entry_time'],
                        'exit_time': idx,
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'shares': position['shares'],
                        'pnl': pnl,
                        'exit_reason': exit_reason
                    }
                    results['trades'].append(trade)
                    position = None
            
            # Check for new entry
            if not position and self.daily_trades < 10:  # Max 10 trades per day
                signal, direction = self.check_entry_signal(row, strategy)
                
                if signal and self.consecutive_losses < 5:
                    shares = self.calculate_position_size(row['Close'])
                    
                    # Calculate dynamic stop loss and take profit
                    if row['Close'] < 5:
                        stop_pct = 0.03
                        profit_pct = 0.05
                    elif row['Close'] < 50:
                        stop_pct = 0.04
                        profit_pct = 0.04
                    else:
                        stop_pct = 0.03
                        profit_pct = 0.03
                    
                    position = {
                        'entry_time': idx,
                        'entry_price': row['Close'],
                        'shares': shares,
                        'side': 'long' if direction == 'BUY' else 'short',
                        'stop_loss': row['Close'] * (1 - stop_pct),
                        'take_profit': row['Close'] * (1 + profit_pct)
                    }
                    
                    self.capital -= position['entry_price'] * shares
                    self.daily_trades += 1
        
        # Calculate metrics
        if results['trades']:
            trades_df = pd.DataFrame(results['trades'])
            winning_trades = trades_df[trades_df['pnl'] > 0]
            losing_trades = trades_df[trades_df['pnl'] <= 0]
            
            results['metrics'] = {
                'total_trades': len(trades_df),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': len(winning_trades) / len(trades_df),
                'total_pnl': trades_df['pnl'].sum(),
                'avg_win': winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0,
                'avg_loss': losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0,
                'profit_factor': winning_trades['pnl'].sum() / abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0,
                'final_capital': self.capital
            }
        
        return results

def run_backtest():
    """Run backtests for multiple symbols and strategies"""
    
    # Test symbols (liquid stocks only)
    test_symbols = ['NVDA', 'TSLA', 'AMD', 'AAPL', 'MSFT']
    strategies = ['momentum', 'mean_reversion', 'breakout']
    
    all_results = []
    
    for symbol in test_symbols:
        print(f"\nBacktesting {symbol}...")
        
        for strategy in strategies:
            print(f"  Testing {strategy} strategy...")
            
            # Initialize backtester
            backtester = StrategyBacktester(initial_capital=100000)
            
            # Download data
            data = backtester.download_data(symbol, period="1mo")
            
            if not data.empty:
                # Run backtest
                results = backtester.execute_backtest(symbol, strategy, data)
                all_results.append(results)
    
    # Display results
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS SUMMARY")
    print("=" * 80)
    
    # Sort by total P&L
    sorted_results = sorted(all_results, 
                          key=lambda x: x['metrics'].get('total_pnl', 0), 
                          reverse=True)
    
    print(f"\n{'Symbol':<8} {'Strategy':<15} {'Trades':<8} {'Win Rate':<10} {'Total P&L':<12} {'Profit Factor':<12}")
    print("-" * 80)
    
    for result in sorted_results:
        if result['metrics']:
            m = result['metrics']
            print(f"{result['symbol']:<8} {result['strategy']:<15} "
                  f"{m['total_trades']:<8} {m['win_rate']*100:<9.1f}% "
                  f"${m['total_pnl']:<11.2f} {m['profit_factor']:<12.2f}")
    
    # Best performing combination
    if sorted_results and sorted_results[0]['metrics']:
        best = sorted_results[0]
        print(f"\n✅ Best Strategy: {best['strategy']} on {best['symbol']}")
        print(f"   Total P&L: ${best['metrics']['total_pnl']:.2f}")
        print(f"   Win Rate: {best['metrics']['win_rate']*100:.1f}%")
        print(f"   Profit Factor: {best['metrics']['profit_factor']:.2f}")
    
    # Save detailed results
    output_path = Path("backtest_results.json")
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n💾 Detailed results saved to: {output_path}")

if __name__ == "__main__":
    print("Starting strategy backtesting...")
    print("This will test strategies on historical data with realistic constraints")
    print("-" * 80)
    
    try:
        run_backtest()
    except Exception as e:
        print(f"Error during backtest: {e}")
        print("Make sure you have yfinance installed: pip install yfinance")