#!/usr/bin/env python3
"""Test the ML predictor async/await fix"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import the trading bot components
from trading_bot_commentary_updated import MLPredictor, CommentarySystem, Config

def create_test_data():
    """Create sample market data for testing"""
    dates = pd.date_range(end=datetime.now(), periods=100, freq='5min')
    data = pd.DataFrame({
        'open': np.random.uniform(100, 110, 100),
        'high': np.random.uniform(110, 120, 100),
        'low': np.random.uniform(90, 100, 100),
        'close': np.random.uniform(95, 115, 100),
        'volume': np.random.uniform(1000, 10000, 100)
    }, index=dates)
    return data

def test_ml_predictor():
    """Test the ML predictor predict_with_commentary method"""
    print("Testing ML Predictor fix...")
    
    # Create instances
    commentary = CommentarySystem()
    ml_predictor = MLPredictor(commentary)
    
    # Create test data
    test_data = create_test_data()
    test_indicators = {
        'rsi': 45.5,
        'macd': 0.5,
        'macd_signal': 0.3,
        'bb_position': 0.4
    }
    
    try:
        # This should now work without async/await
        signal, explanation = ml_predictor.predict_with_commentary(
            test_indicators, 
            'TEST',
            test_data
        )
        
        print(f"✅ ML Predictor called successfully!")
        print(f"   Signal: {signal}")
        print(f"   Explanation keys: {list(explanation.keys())}")
        print(f"   Method: {explanation.get('method', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error calling ML predictor: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_async_context():
    """Test that the fix works in an async context"""
    print("\nTesting in async context...")
    
    commentary = CommentarySystem()
    ml_predictor = MLPredictor(commentary)
    
    test_data = create_test_data()
    test_indicators = {'rsi': 50, 'macd': 0}
    
    try:
        # Should work without await
        signal, explanation = ml_predictor.predict_with_commentary(
            test_indicators,
            'ASYNC_TEST', 
            test_data
        )
        print("✅ Works correctly in async context!")
        return True
    except Exception as e:
        print(f"❌ Error in async context: {e}")
        return False

if __name__ == "__main__":
    print("ML Predictor Async/Await Fix Test")
    print("==================================")
    
    # Test synchronous call
    sync_result = test_ml_predictor()
    
    # Test in async context  
    async_result = asyncio.run(test_async_context())
    
    print("\n" + "="*40)
    if sync_result and async_result:
        print("✅ All tests passed! The fix is working correctly.")
    else:
        print("❌ Some tests failed. Check the implementation.")