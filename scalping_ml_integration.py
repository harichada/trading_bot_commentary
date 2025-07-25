"""
Integration module for Scalping ML Model with existing trading bot
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
import asyncio
from pathlib import Path

from scalping_ml_model import (
    ScalpingMLModel, 
    ScalpingFeatureEngineer,
    ScalpingFeatureConfig,
    ModelValidator
)

# Import TradingCommentary from the main bot
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger(__name__)

class ScalpingMLPredictor:
    """Drop-in replacement for existing ML predictor with scalping optimization"""
    
    def __init__(self, commentary_system=None):
        self.commentary = commentary_system
        self._commentary_available = False
        self._check_commentary_imports()
        self.model = ScalpingMLModel(commentary_system)
        self.validator = ModelValidator()
        self.feature_config = ScalpingFeatureConfig()
        
        # Performance tracking
        self.recent_predictions = []
        self.performance_window = 100
        
        # Model state
        self.last_training_data = None
        self.training_queue = []
        self.min_training_samples = 500
        
        # Load existing model
        self._initialize()
        
        # Brain integration (will be set by trading engine)
        self.brain = None
        
    def _check_commentary_imports(self):
        """Check if TradingCommentary is available"""
        try:
            from trading_bot_commentary_updated import TradingCommentary, CommentaryType
            self.TradingCommentary = TradingCommentary
            self.CommentaryType = CommentaryType
            self._commentary_available = True
        except ImportError:
            logger.warning("TradingCommentary not available, using dict-based commentary")
            self._commentary_available = False
    
    def _add_commentary(self, comment_data: Dict):
        """Add commentary with proper object type"""
        if not self.commentary:
            return
            
        if self._commentary_available:
            # Convert dict to TradingCommentary object
            from trading_bot_commentary_updated import TradingCommentary, CommentaryType
            
            # Map string type to CommentaryType enum
            type_mapping = {
                'MODEL_INIT': CommentaryType.MARKET_ANALYSIS,
                'ML_SIGNAL': CommentaryType.SIGNAL_GENERATION,
                'ML_RESULT': CommentaryType.DECISION,
                'MODEL_TRAINING': CommentaryType.MARKET_ANALYSIS,
                'DRIFT_DETECTION': CommentaryType.WARNING,
                'RETRAIN_SCHEDULED': CommentaryType.MARKET_ANALYSIS,
            }
            
            comment_type = type_mapping.get(
                comment_data.get('type', ''), 
                CommentaryType.MARKET_ANALYSIS
            )
            
            commentary = TradingCommentary(
                timestamp=comment_data.get('timestamp', datetime.now()),
                type=comment_type,
                symbol=comment_data.get('symbol'),
                title=comment_data.get('title', ''),
                message=comment_data.get('message', ''),
                data=comment_data.get('data', {}),
                confidence=comment_data.get('confidence'),
                importance=comment_data.get('importance', 5)
            )
            
            self.commentary.add_commentary(commentary)
        else:
            # Fallback: just pass the dict if commentary system accepts it
            self.commentary.add_commentary(comment_data)
    
    def set_brain(self, brain):
        """Set the trading brain for integration"""
        self.brain = brain
        self.model.brain = brain
        
    def _initialize(self):
        """Initialize the model"""
        if self.model.load_model():
            logger.info("Loaded existing scalping ML model")
            if self.commentary:
                self._add_commentary({
                    'timestamp': datetime.now(),
                    'type': 'MODEL_INIT',
                    'title': '🤖 Scalping ML Model Loaded',
                    'message': f'Model trained on {len(self.model.feature_names)} features',
                    'importance': 7
                })
        else:
            logger.info("No existing model, will train on first data batch")
    
    def predict_with_commentary(self, indicators: Dict[str, float], 
                                    symbol: str, 
                                    market_data: Optional[pd.DataFrame] = None,
                                    quote_data: Optional[Dict] = None) -> Tuple[int, Dict]:
        """
        Make prediction with detailed explanation - compatible with existing interface
        
        Args:
            indicators: Pre-calculated indicators (for compatibility)
            symbol: Trading symbol
            market_data: Raw OHLCV DataFrame or dict with DataFrames
            quote_data: Real-time quote data with bid/ask
            
        Returns:
            Tuple of (signal, explanation_dict)
        """
        # Handle different input formats
        df = None
        multi_tf_data = None
        
        if isinstance(market_data, pd.DataFrame):
            df = market_data
        elif isinstance(market_data, dict):
            # Multi-timeframe data
            df = market_data.get('5min', market_data.get('1min'))
            multi_tf_data = market_data
        
        # Fallback if no market data
        if df is None or df.empty:
            return self._fallback_prediction(indicators, symbol)
        
        # Make scalping prediction
        signal, confidence, explanation = self.model.predict(
            df, quote_data, multi_tf_data
        )
        
        # Track prediction
        self.recent_predictions.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'signal': signal,
            'confidence': confidence
        })
        
        # Trim prediction history
        if len(self.recent_predictions) > self.performance_window:
            self.recent_predictions = self.recent_predictions[-self.performance_window:]
        
        # Add commentary
        self._add_prediction_commentary(signal, confidence, explanation, symbol)
        
        # Convert to expected format
        result = {
            'prediction': signal,
            'confidence': confidence,
            'method': 'scalping_ml',
            'signal_strength': explanation.get('signal_strength', 'moderate'),
            **explanation
        }
        
        # Add training data to queue
        self._queue_training_data(df, quote_data, symbol)
        
        return signal, result
    
    def _fallback_prediction(self, indicators: Dict[str, float], symbol: str) -> Tuple[int, Dict]:
        """Fallback prediction when ML model not available"""
        # Simple scalping rules
        rsi = indicators.get('rsi', 50)
        volume_ratio = indicators.get('volume_ratio', 1)
        
        signal = 0
        confidence = 0.5
        
        if rsi < 30 and volume_ratio > 1.5:
            signal = 1  # Buy
            confidence = 0.65
            reasoning = "Oversold with volume surge"
        elif rsi > 70 and volume_ratio > 1.5:
            signal = -1  # Sell
            confidence = 0.65
            reasoning = "Overbought with volume surge"
        else:
            reasoning = "No clear scalping opportunity"
        
        return signal, {
            'prediction': signal,
            'confidence': confidence,
            'method': 'fallback_scalping',
            'signal_strength': 'weak',
            'reasoning': reasoning
        }
    
    def _add_prediction_commentary(self, signal: int, confidence: float, 
                                  explanation: Dict, symbol: str):
        """Add detailed commentary for prediction"""
        if not self.commentary:
            return
        
        # Build message based on signal
        if signal == 1:
            action = "BUY"
            emoji = "🟢"
        elif signal == -1:
            action = "SELL"
            emoji = "🔴"
        else:
            action = "HOLD"
            emoji = "⏸️"
        
        # Extract key insights
        insights = []
        
        # Check microstructure features
        if 'top_features' in explanation:
            features = explanation['top_features']
            
            if features.get('order_flow_imbalance', 0) > 0.3:
                insights.append("Strong buy pressure")
            elif features.get('order_flow_imbalance', 0) < -0.3:
                insights.append("Strong sell pressure")
                
            if features.get('spread_relative', 0) > 0.002:
                insights.append("Wide spread (low liquidity)")
            elif features.get('spread_relative', 0) < 0.0005:
                insights.append("Tight spread (high liquidity)")
                
            if features.get('volatility_ratio', 1) > 2:
                insights.append("Elevated short-term volatility")
        
        # Add specific scalping insights
        if explanation.get('insight'):
            insights.append(explanation['insight'])
        
        self._add_commentary({
            'timestamp': datetime.now(),
            'type': 'ML_SIGNAL',
            'symbol': symbol,
            'title': f"{emoji} Scalping Signal: {action} {symbol}",
            'message': f"Confidence: {confidence:.1%}\n" + 
                      f"Signal Strength: {explanation.get('signal_strength', 'moderate')}\n" +
                      f"Key Factors: {', '.join(insights) if insights else 'Multiple technical factors'}",
            'data': {
                'signal': signal,
                'confidence': confidence,
                'model_votes': explanation.get('model_votes', {}),
                'top_features': explanation.get('top_features', {})
            },
            'confidence': confidence,
            'importance': 8 if confidence > 0.7 else 6
        })
    
    def _queue_training_data(self, df: pd.DataFrame, quote_data: Optional[Dict], symbol: str):
        """Queue data for future training"""
        self.training_queue.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'data': df.copy(),
            'quote': quote_data
        })
        
        # Limit queue size
        if len(self.training_queue) > 1000:
            self.training_queue = self.training_queue[-1000:]
    
    def update_with_result(self, symbol: str, signal: int, 
                               entry_price: float, exit_price: float,
                               success: bool):
        """Update model with trade result"""
        # Calculate reward
        if signal == 1:  # Buy
            pnl_pct = (exit_price - entry_price) / entry_price
        elif signal == -1:  # Sell
            pnl_pct = (entry_price - exit_price) / entry_price
        else:
            pnl_pct = 0
        
        reward = pnl_pct * 100  # Convert to percentage
        
        # Find corresponding prediction
        for pred in reversed(self.recent_predictions):
            if pred['symbol'] == symbol and pred['signal'] == signal:
                # Update model online if supported
                # For now, just track performance
                pred['result'] = {
                    'success': success,
                    'reward': reward,
                    'exit_time': datetime.now()
                }
                break
        
        # Log performance
        if self.commentary:
            emoji = "✅" if success else "❌"
            self._add_commentary({
                'timestamp': datetime.now(),
                'type': 'ML_RESULT',
                'symbol': symbol,
                'title': f"{emoji} Scalping Result: {symbol}",
                'message': f"PnL: {reward:.2f}%",
                'data': {
                    'signal': signal,
                    'pnl_pct': reward,
                    'success': success
                },
                'importance': 5
            })
    
    def retrain_model(self, data_provider, symbols: List[str], 
                          lookback_days: int = 30):
        """Retrain the scalping model with recent data"""
        if self.commentary:
            self._add_commentary({
                'timestamp': datetime.now(),
                'type': 'MODEL_TRAINING',
                'title': '🔧 Retraining Scalping Model',
                'message': f'Collecting data from {len(symbols)} symbols',
                'importance': 8
            })
        
        # Collect training data
        all_X = []
        all_y = []
        
        # Use ThreadPoolExecutor for parallel data fetching
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def fetch_symbol_data(symbol):
            """Fetch data for a single symbol"""
            try:
                # Fetch both timeframes in parallel
                with ThreadPoolExecutor(max_workers=2) as executor:
                    future_1min = executor.submit(
                        data_provider.get_market_data,
                        symbol, period_type='day', period=lookback_days,
                        frequency_type='minute', frequency=1
                    )
                    future_5min = executor.submit(
                        data_provider.get_market_data,
                        symbol, period_type='day', period=lookback_days,
                        frequency_type='minute', frequency=5
                    )
                    
                    timeframes = {
                        '1min': future_1min.result(),
                        '5min': future_5min.result()
                    }
                
                # Generate training samples
                X_symbol, y_symbol = self._generate_training_samples(
                    timeframes, symbol
                )
                
                return symbol, X_symbol, y_symbol
                
            except Exception as e:
                logger.error(f"Error collecting data for {symbol}: {e}")
                return symbol, [], []
        
        # Fetch all symbols in parallel
        with ThreadPoolExecutor(max_workers=len(symbols)) as executor:
            futures = {executor.submit(fetch_symbol_data, symbol): symbol 
                      for symbol in symbols}
            
            for future in as_completed(futures):
                symbol, X_symbol, y_symbol = future.result()
                if X_symbol:
                    all_X.extend(X_symbol)
                    all_y.extend(y_symbol)
                    logger.info(f"Collected {len(X_symbol)} samples from {symbol}")
        
        if len(all_X) >= self.min_training_samples:
            # Train model
            X = np.array(all_X)
            y = np.array(all_y)
            
            success = self.model.train(X, y, optimize_hyperparams=False)
            
            if success and self.commentary:
                self._add_commentary({
                    'timestamp': datetime.now(),
                    'type': 'MODEL_TRAINING',
                    'title': '✅ Scalping Model Retrained',
                    'message': f'Trained on {len(X)} samples with {len(self.model.selected_features)} features',
                    'data': {
                        'samples': len(X),
                        'features': len(self.model.selected_features),
                        'top_features': list(self.model.feature_importance.keys())[:5]
                    },
                    'importance': 8
                })
        else:
            logger.warning(f"Insufficient training data: {len(all_X)} samples")
    
    def _generate_training_samples(self, timeframes: Dict[str, pd.DataFrame], 
                                 symbol: str) -> Tuple[List, List]:
        """Generate training samples from historical data"""
        X = []
        y = []
        
        # Use 5min data as primary timeframe
        primary_df = timeframes.get('5min')
        if primary_df is None or len(primary_df) < 100:
            return X, y
        
        # Generate samples with sliding window
        for i in range(50, len(primary_df) - 10):
            try:
                # Get data window
                window_data = primary_df.iloc[:i+1]
                
                # Extract features
                features = self.model.feature_engineer.extract_features(
                    window_data,
                    quote_data=None,  # Historical data won't have quotes
                    market_data=timeframes
                )
                
                # Create label based on future price movement
                current_price = primary_df['Close'].iloc[i]
                future_prices = primary_df['Close'].iloc[i+1:i+11]  # Next 10 bars
                
                # Calculate max favorable movement
                max_gain = (future_prices.max() - current_price) / current_price
                max_loss = (current_price - future_prices.min()) / current_price
                
                # Label based on risk-reward for scalping
                if max_gain > 0.003 and max_gain > max_loss * 1.5:  # 0.3% gain with 1.5:1 RR
                    label = 2  # Buy
                elif max_loss > 0.003 and max_loss > max_gain * 1.5:
                    label = 0  # Sell
                else:
                    label = 1  # Hold
                
                # Convert features to array
                feature_array = [features[name] for name in self.model.feature_names]
                X.append(feature_array)
                y.append(label)
                
            except Exception as e:
                logger.debug(f"Error generating sample: {e}")
                continue
        
        return X, y
    
    def get_model_metrics(self) -> Dict[str, Any]:
        """Get current model performance metrics"""
        if not self.recent_predictions:
            return {}
        
        # Calculate recent performance
        results = [p for p in self.recent_predictions if 'result' in p]
        
        if not results:
            return {
                'predictions_made': len(self.recent_predictions),
                'awaiting_results': len(self.recent_predictions)
            }
        
        # Calculate metrics
        successes = sum(1 for r in results if r['result']['success'])
        total_pnl = sum(r['result']['reward'] for r in results)
        
        metrics = {
            'predictions_made': len(self.recent_predictions),
            'results_available': len(results),
            'win_rate': successes / len(results) if results else 0,
            'avg_pnl': total_pnl / len(results) if results else 0,
            'total_pnl': total_pnl,
            'model_trained': self.model.is_trained,
            'last_training': self.model.last_retrain_time,
            'features_selected': len(self.model.selected_features) if self.model.selected_features else 0
        }
        
        # Add feature importance
        if self.model.feature_importance:
            metrics['top_features'] = list(self.model.feature_importance.keys())[:10]
        
        return metrics
    
    def should_retrain(self) -> bool:
        """Check if model should be retrained"""
        if not self.model.is_trained:
            return True
        
        # Time-based check
        if self.model.last_retrain_time:
            days_since = (datetime.now() - self.model.last_retrain_time).days
            if days_since > 3:  # Retrain every 3 days for scalping
                return True
        
        # Performance-based check
        metrics = self.get_model_metrics()
        if metrics.get('results_available', 0) > 50:
            if metrics.get('win_rate', 0) < 0.45:  # Below 45% win rate
                return True
            if metrics.get('avg_pnl', 0) < -0.1:  # Negative average PnL
                return True
        
        return False


# Factory function for easy integration
def create_scalping_ml_predictor(commentary_system=None) -> ScalpingMLPredictor:
    """Create scalping ML predictor instance"""
    return ScalpingMLPredictor(commentary_system)


# Async training function
async def train_scalping_model_async(predictor: ScalpingMLPredictor, 
                                   data_provider, 
                                   symbols: List[str],
                                   lookback_days: int = 30):
    """Async function to train scalping model"""
    await predictor.retrain_model(data_provider, symbols, lookback_days)