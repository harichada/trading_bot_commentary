"""
Example integration of the Scalping ML Model with the existing trading bot
Shows how to use the new ML model as a drop-in replacement
"""

import asyncio
import pandas as pd
from datetime import datetime
from typing import Dict, Optional

# Import the new scalping ML components
from scalping_ml_integration import create_scalping_ml_predictor
from scalping_ml_training import run_training_pipeline
from scalping_ml_online_updater import create_model_updater

# Example of how to integrate with existing trading bot
class EnhancedTradingEngine:
    """Example of integrating scalping ML with existing engine"""
    
    def __init__(self, data_provider, commentary_system):
        self.data_provider = data_provider
        self.commentary = commentary_system
        
        # Initialize scalping ML predictor
        self.ml_predictor = create_scalping_ml_predictor(commentary_system)
        
        # Initialize online updater
        self.model_updater = create_model_updater(
            self.ml_predictor.model, 
            commentary_system
        )
        
        # Trading symbols for scalping
        self.scalping_symbols = ['SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA']
        
    async def analyze_symbol(self, symbol: str, 
                           market_data: pd.DataFrame,
                           quote_data: Optional[Dict] = None) -> Dict:
        """Analyze symbol using scalping ML model"""
        
        # Get ML prediction with enhanced features
        signal, prediction_details = await self.ml_predictor.predict_with_commentary(
            indicators={},  # Not needed with new model
            symbol=symbol,
            market_data=market_data,
            quote_data=quote_data
        )
        
        # Get adaptive prediction with online learning
        if hasattr(market_data, 'values'):
            features = self.ml_predictor.model.feature_engineer.extract_features(
                market_data, quote_data
            )
            feature_array = [features[name] for name in self.ml_predictor.model.feature_names]
            
            # Add to online updater
            self.model_updater.add_prediction(
                feature_array, 
                prediction_details,
                symbol
            )
            
            # Get enhanced prediction
            enhanced_prediction = self.model_updater.get_adaptive_prediction(
                feature_array,
                prediction_details
            )
            
            return {
                'symbol': symbol,
                'signal': signal,
                'confidence': prediction_details['confidence'],
                'details': enhanced_prediction,
                'timestamp': datetime.now()
            }
        
        return {
            'symbol': symbol,
            'signal': signal,
            'confidence': prediction_details['confidence'],
            'details': prediction_details,
            'timestamp': datetime.now()
        }
    
    async def update_with_trade_result(self, symbol: str, 
                                     prediction_id: str,
                                     entry_price: float,
                                     exit_price: float,
                                     success: bool):
        """Update model with actual trade results"""
        
        # Calculate price change
        price_change = (exit_price - entry_price) / entry_price
        
        # Update ML predictor
        await self.ml_predictor.update_with_result(
            symbol, 
            1 if price_change > 0 else -1,
            entry_price,
            exit_price,
            success
        )
        
        # Update online learner
        self.model_updater.add_outcome(
            prediction_id,
            price_change,
            success
        )
        
        # Check if retrain needed
        if self.ml_predictor.should_retrain():
            asyncio.create_task(self._retrain_model())
    
    async def _retrain_model(self):
        """Retrain the scalping model"""
        await self.ml_predictor.retrain_model(
            self.data_provider,
            self.scalping_symbols,
            lookback_days=30
        )
    
    def get_model_stats(self) -> Dict:
        """Get comprehensive model statistics"""
        ml_metrics = self.ml_predictor.get_model_metrics()
        online_stats = self.model_updater.get_update_stats()
        
        return {
            'ml_model': ml_metrics,
            'online_learning': online_stats,
            'last_update': datetime.now()
        }


# Example usage script
async def main():
    """Example of using the scalping ML model"""
    
    # Mock data provider
    class MockDataProvider:
        def get_market_data(self, symbol, **kwargs):
            # Return mock data
            import numpy as np
            dates = pd.date_range(end=datetime.now(), periods=1000, freq='5min')
            
            # Generate realistic OHLCV data
            close_prices = 100 + np.cumsum(np.random.randn(1000) * 0.1)
            
            df = pd.DataFrame({
                'Open': close_prices + np.random.rand(1000) * 0.1,
                'High': close_prices + np.abs(np.random.rand(1000) * 0.5),
                'Low': close_prices - np.abs(np.random.rand(1000) * 0.5),
                'Close': close_prices,
                'Volume': np.random.randint(1000, 10000, 1000)
            }, index=dates)
            
            return df
    
    # Mock commentary system
    class MockCommentary:
        def add_commentary(self, comment):
            print(f"[{comment.get('type', 'INFO')}] {comment.get('title', '')}: {comment.get('message', '')}")
    
    # Initialize components
    data_provider = MockDataProvider()
    commentary = MockCommentary()
    
    print("=== Scalping ML Model Example ===\n")
    
    # 1. Train the model (if needed)
    print("1. Training scalping ML model...")
    trainer = run_training_pipeline(
        data_provider,
        ['SPY', 'QQQ', 'AAPL'],
        commentary
    )
    print("   Training complete!\n")
    
    # 2. Initialize trading engine with ML
    print("2. Initializing enhanced trading engine...")
    engine = EnhancedTradingEngine(data_provider, commentary)
    print("   Engine ready!\n")
    
    # 3. Make predictions
    print("3. Making scalping predictions...")
    
    for symbol in ['SPY', 'QQQ', 'AAPL']:
        # Get market data
        market_data = data_provider.get_market_data(symbol)
        
        # Mock quote data
        quote_data = {
            'bid': market_data['Close'].iloc[-1] - 0.01,
            'ask': market_data['Close'].iloc[-1] + 0.01,
            'last': market_data['Close'].iloc[-1],
            'volume': market_data['Volume'].iloc[-1]
        }
        
        # Analyze
        result = await engine.analyze_symbol(symbol, market_data, quote_data)
        
        print(f"\n   {symbol}:")
        print(f"   Signal: {result['signal']} ({result['confidence']:.1%} confidence)")
        print(f"   Method: {result['details'].get('method', 'scalping_ml')}")
        
        if 'top_features' in result['details']:
            print("   Top factors:")
            for feature, value in list(result['details']['top_features'].items())[:3]:
                print(f"     - {feature}: {value:.3f}")
    
    # 4. Simulate trade and update
    print("\n4. Simulating trade result...")
    
    # Simulate a winning trade
    await engine.update_with_trade_result(
        'SPY',
        'SPY_123456',
        entry_price=100.00,
        exit_price=100.50,
        success=True
    )
    print("   Trade result recorded\n")
    
    # 5. Show model statistics
    print("5. Model Statistics:")
    stats = engine.get_model_stats()
    
    print(f"   ML Model:")
    print(f"     - Predictions made: {stats['ml_model'].get('predictions_made', 0)}")
    print(f"     - Model trained: {stats['ml_model'].get('model_trained', False)}")
    print(f"     - Features selected: {stats['ml_model'].get('features_selected', 0)}")
    
    print(f"\n   Online Learning:")
    print(f"     - Buffer size: {stats['online_learning'].get('buffer_size', 0)}")
    print(f"     - Drift detected: {stats['online_learning'].get('drift_detected', False)}")
    
    print("\n=== Example Complete ===")


# Integration with existing bot - drop-in replacement
def integrate_with_existing_bot(trading_engine):
    """
    Example of how to integrate with existing trading bot
    
    Replace this line in trading_bot_commentary_updated.py:
        self.ml_model = MLPredictorWithCommentary(self.commentary)
    
    With:
        from scalping_ml_integration import create_scalping_ml_predictor
        self.ml_model = create_scalping_ml_predictor(self.commentary)
    
    The new model has the same interface but with enhanced features!
    """
    
    # The new predictor has the same predict_with_commentary interface
    # but uses advanced scalping features and online learning
    
    old_predictor = trading_engine.ml_model
    new_predictor = create_scalping_ml_predictor(trading_engine.commentary)
    
    # Replace the predictor
    trading_engine.ml_model = new_predictor
    
    print("Scalping ML model integrated successfully!")
    return trading_engine


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())