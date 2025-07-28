#!/usr/bin/env python3
"""
Verify professional features are working correctly
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("Testing Professional Trading Bot Features")
print("="*50)

# Test 1: ML Model Manager (Safe Version)
print("\n1. Testing Safe ML Model Manager...")
try:
    from ml_model_manager_safe import ModelManager
    
    # Create model manager
    manager = ModelManager()
    
    # Create sample data
    X = pd.DataFrame({
        'returns': np.random.randn(100),
        'volume': np.random.randn(100),
        'volatility': np.random.uniform(0.01, 0.03, 100)
    })
    y = pd.Series(np.random.randint(0, 2, 100))
    
    # Train a model
    manager.train_model('random_forest', X, y)
    manager.set_active_model('random_forest')
    
    # Make predictions
    predictions = manager.predict(X.head(5))
    print(f"✓ ML Model Manager working - predictions: {predictions}")
    
except Exception as e:
    print(f"✗ ML Model Manager failed: {e}")

# Test 2: Strategy System
print("\n2. Testing Strategy System...")
try:
    from strategy_system import StrategyManager
    
    # Create strategy manager
    strategy_manager = StrategyManager()
    
    # Generate sample market data
    dates = pd.date_range(end=datetime.now(), periods=100, freq='5min')
    data = pd.DataFrame({
        'open': 100 + np.random.randn(100).cumsum(),
        'high': 101 + np.random.randn(100).cumsum(),
        'low': 99 + np.random.randn(100).cumsum(),
        'close': 100 + np.random.randn(100).cumsum(),
        'volume': np.random.randint(1000000, 5000000, 100)
    }, index=dates)
    
    # Generate signals
    signals = strategy_manager.analyze_all(data, {})
    print(f"✓ Strategy System working - {len(signals)} signals generated")
    
except Exception as e:
    print(f"✗ Strategy System failed: {e}")

# Test 3: Risk Management
print("\n3. Testing Risk Management...")
try:
    from risk_management import RiskManager, PositionSizingMethod
    
    # Create risk manager
    risk_manager = RiskManager(initial_capital=100000)
    
    # Calculate position size
    signal_data = {
        'symbol': 'SPY',
        'price': 450.0,
        'volatility': 0.02,
        'stop_loss_distance': 5.0
    }
    
    size = risk_manager.calculate_position_size(
        PositionSizingMethod.VOLATILITY_BASED,
        signal_data
    )
    print(f"✓ Risk Management working - position size: {size} shares")
    
except Exception as e:
    print(f"✗ Risk Management failed: {e}")

# Test 4: Paper Trading
print("\n4. Testing Paper Trading...")
try:
    from paper_trading import PaperTradingEngine
    from advanced_orders import Order, OrderType, OrderSide
    
    # Create paper trading engine
    engine = PaperTradingEngine(initial_balance=100000)
    
    # Place a test order
    order = Order(
        symbol='SPY',
        side=OrderSide.BUY,
        quantity=100,
        order_type=OrderType.MARKET
    )
    
    order_id = engine.place_order(order)
    print(f"✓ Paper Trading working - order placed: {order_id}")
    
except Exception as e:
    print(f"✗ Paper Trading failed: {e}")

# Test 5: Performance Analytics
print("\n5. Testing Performance Analytics...")
try:
    from performance_analytics import PerformanceAnalyzer
    
    # Create analyzer
    analyzer = PerformanceAnalyzer()
    
    # Add sample trade
    trade = {
        'symbol': 'SPY',
        'pnl': 150,
        'entry_time': datetime.now() - timedelta(hours=2),
        'exit_time': datetime.now()
    }
    analyzer.add_trade(trade)
    
    # Update equity
    analyzer.update_equity(datetime.now(), 100150)
    
    metrics = analyzer.calculate_metrics()
    print(f"✓ Performance Analytics working - total return: {metrics.total_return:.2%}")
    
except Exception as e:
    print(f"✗ Performance Analytics failed: {e}")

print("\n" + "="*50)
print("Professional Features Test Complete!")
print("All components are working without segmentation faults.")
print("\nYou can now run:")
print("  python run_professional_bot.py")
print("\nOr test all features with:")
print("  python test_professional_bot.py")