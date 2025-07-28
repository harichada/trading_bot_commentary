#!/usr/bin/env python3
"""
Test backtesting in isolation to find the issue
"""

print("Testing backtesting components...")

# First import our safety wrapper
import safe_imports

print("\n1. Testing imports...")
try:
    from backtesting_engine import BacktestingEngine, BacktestConfig
    print("✓ Backtesting engine imported")
except Exception as e:
    print(f"✗ Backtesting engine import failed: {e}")
    import traceback
    traceback.print_exc()

print("\n2. Testing basic backtest configuration...")
try:
    from datetime import datetime, timedelta
    config = BacktestConfig(
        start_date=datetime.now() - timedelta(days=30),
        end_date=datetime.now(),
        initial_capital=100000,
        symbols=['SPY', 'QQQ'],
        commission=0.001
    )
    print(f"✓ Config created: {config.symbols}")
except Exception as e:
    print(f"✗ Config creation failed: {e}")

print("\n3. Testing backtesting engine initialization...")
try:
    engine = BacktestingEngine(config)
    print("✓ Engine initialized")
except Exception as e:
    print(f"✗ Engine initialization failed: {e}")
    import traceback
    traceback.print_exc()

print("\n4. Testing sample data generation...")
try:
    import pandas as pd
    import numpy as np
    
    # Generate simple test data
    dates = pd.date_range(start=config.start_date, end=config.end_date, freq='5min')
    market_data = {}
    
    for symbol in config.symbols:
        prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)
        market_data[symbol] = pd.DataFrame({
            'open': prices,
            'high': prices + 0.5,
            'low': prices - 0.5,
            'close': prices,
            'volume': 1000000
        }, index=dates)
    
    print(f"✓ Generated data for {len(market_data)} symbols")
except Exception as e:
    print(f"✗ Data generation failed: {e}")

print("\n5. Testing backtest run...")
try:
    results = engine.run(market_data)
    print(f"✓ Backtest completed: {results.total_trades} trades")
except Exception as e:
    print(f"✗ Backtest run failed: {e}")
    import traceback
    traceback.print_exc()

print("\nTest complete!")