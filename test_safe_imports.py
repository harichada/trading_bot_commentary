#!/usr/bin/env python3
"""
Test that the safe imports work without segmentation fault
"""

print("Testing safe imports...")

try:
    print("1. Importing strategy system...")
    from strategy_system import StrategyManager
    print("✓ Strategy system imported successfully")
except Exception as e:
    print(f"✗ Strategy system import failed: {e}")

try:
    print("\n2. Importing safe ML model manager...")
    from ml_model_manager_safe import ModelManager
    print("✓ Safe ML model manager imported successfully")
except Exception as e:
    print(f"✗ Safe ML model manager import failed: {e}")

try:
    print("\n3. Importing risk management...")
    from risk_management import RiskManager
    print("✓ Risk management imported successfully")
except Exception as e:
    print(f"✗ Risk management import failed: {e}")

try:
    print("\n4. Importing paper trading...")
    from paper_trading import PaperTradingEngine
    print("✓ Paper trading imported successfully")
except Exception as e:
    print(f"✗ Paper trading import failed: {e}")

try:
    print("\n5. Importing performance analytics...")
    from performance_analytics import PerformanceAnalyzer
    print("✓ Performance analytics imported successfully")
except Exception as e:
    print(f"✗ Performance analytics import failed: {e}")

try:
    print("\n6. Testing basic ML functionality...")
    import pandas as pd
    import numpy as np
    
    # Create model manager
    manager = ModelManager()
    
    # Create sample data
    X = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.randn(100)
    })
    y = pd.Series(np.random.randint(0, 2, 100))
    
    # Train a simple model
    manager.train_model('random_forest', X, y)
    print("✓ ML model training successful")
    
    # Make predictions
    predictions = manager.predict(X.head(5))
    print(f"✓ Predictions generated: {predictions}")
    
except Exception as e:
    print(f"✗ ML functionality test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50)
print("All safe imports tested!")
print("If you see this message without a segmentation fault, the fix worked!")
print("="*50)