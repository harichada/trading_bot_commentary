#!/usr/bin/env python3
"""
Test components in isolation to identify segmentation fault source
"""

import sys
import traceback

def test_imports():
    """Test each import separately to find the problematic one"""
    
    print("Testing basic imports...")
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
    except Exception as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✓ Pandas imported successfully")
    except Exception as e:
        print(f"✗ Pandas import failed: {e}")
        return False
    
    try:
        import sklearn
        from sklearn.ensemble import RandomForestClassifier
        print("✓ Scikit-learn imported successfully")
    except Exception as e:
        print(f"✗ Scikit-learn import failed: {e}")
        return False
    
    # Test potentially problematic imports
    print("\nTesting ML library imports...")
    
    try:
        import xgboost as xgb
        print("✓ XGBoost imported successfully")
    except Exception as e:
        print(f"✗ XGBoost import failed: {e}")
        print("  This might be causing the segmentation fault")
    
    try:
        import lightgbm as lgb
        print("✓ LightGBM imported successfully")
    except Exception as e:
        print(f"✗ LightGBM import failed: {e}")
        print("  This might be causing the segmentation fault")
    
    try:
        import torch
        print("✓ PyTorch imported successfully")
    except Exception as e:
        print(f"✗ PyTorch import failed: {e}")
        print("  This is optional, so it's OK if it fails")
    
    print("\nTesting FastAPI imports...")
    try:
        from fastapi import FastAPI
        import uvicorn
        print("✓ FastAPI imported successfully")
    except Exception as e:
        print(f"✗ FastAPI import failed: {e}")
        return False
    
    print("\nTesting custom module imports...")
    try:
        from strategy_system import StrategyManager
        print("✓ Strategy system imported successfully")
    except Exception as e:
        print(f"✗ Strategy system import failed: {e}")
        print(f"  Error details: {traceback.format_exc()}")
    
    return True

def test_basic_functionality():
    """Test basic functionality without ML libraries"""
    print("\n" + "="*50)
    print("Testing basic functionality...")
    print("="*50)
    
    try:
        # Test strategy system without ML
        print("\nTesting strategy system...")
        from strategy_system import StrategyManager
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        # Create sample data
        dates = pd.date_range(end=datetime.now(), periods=100, freq='5min')
        data = pd.DataFrame({
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 101,
            'low': np.random.randn(100).cumsum() + 99,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates)
        
        manager = StrategyManager()
        signals = manager.analyze_all(data, {})
        print(f"✓ Strategy system working: {len(signals)} signals generated")
        
    except Exception as e:
        print(f"✗ Strategy system failed: {e}")
        traceback.print_exc()
    
    try:
        # Test paper trading without ML
        print("\nTesting paper trading...")
        from paper_trading import PaperTradingEngine
        from advanced_orders import Order, OrderType, OrderSide
        
        engine = PaperTradingEngine()
        print("✓ Paper trading engine created successfully")
        
    except Exception as e:
        print(f"✗ Paper trading failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    print("Professional Trading Bot - Component Isolation Test")
    print("="*50)
    
    # First test imports
    if test_imports():
        print("\nBasic imports successful!")
    
    # Then test functionality
    test_basic_functionality()
    
    print("\n" + "="*50)
    print("Test complete. Check above for any failures.")
    print("If XGBoost or LightGBM failed, we'll create a version without them.")