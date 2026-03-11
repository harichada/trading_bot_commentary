import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd

from core.models import CommentaryType, SignalType
from core.commentary import TradingCommentary
from ml.models import IntegratedMLModel
from ml.features import MLPrediction

logger = logging.getLogger('TradingBot')


class MLPredictorWithCommentary:
    """Drop-in replacement for existing ML predictor with proper integration"""

    def __init__(self, commentary_system):
        self.commentary = commentary_system
        self.model = None  # Will be set to IntegratedMLModel
        self.brain = None  # Will be set by trading engine
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the integrated ML model"""
        self.model = IntegratedMLModel(
            brain=self.brain,
            commentary_system=self.commentary
        )

        # Try to load existing model
        if not self.model.is_trained:
            logger.info("ML model not trained, will use fallback rules until training")

    def predict_with_commentary(self, indicators: Dict[str, float],
                                    symbol: str,
                                    market_data: Optional[pd.DataFrame] = None) -> Tuple[int, Dict]:
        """
        Make prediction with detailed explanation

        Args:
            indicators: Pre-calculated indicators (for compatibility)
            symbol: Trading symbol
            market_data: Raw OHLCV DataFrame (preferred) or dict with DataFrames

        Returns:
            Tuple of (signal, explanation_dict)
        """
        # Handle different input formats
        df = None

        if isinstance(market_data, pd.DataFrame):
            # Direct DataFrame - best case
            df = market_data
        elif isinstance(market_data, dict):
            # Dict of DataFrames - extract 5min data
            df = market_data.get('5min', market_data.get('1min'))

        # If no market data but have indicators, use fallback
        if df is None or df.empty:
            return self._fallback_prediction(indicators, symbol)

        # Make ML prediction
        ml_prediction = self.model.predict(df, symbol)

        # Add commentary
        self._add_prediction_commentary(ml_prediction, symbol)

        # Convert to expected format
        explanation = {
            'prediction': ml_prediction.signal,
            'confidence': ml_prediction.confidence,
            'method': 'ml_model',
            'signal_strength': self._get_signal_strength(ml_prediction.confidence),
            **ml_prediction.reasoning
        }

        return ml_prediction.signal, explanation

    def _fallback_prediction(self, indicators: Dict[str, float], symbol: str) -> Tuple[int, Dict]:
        """Fallback prediction using simple rules when ML not available"""
        rsi = indicators.get('rsi', 50)
        macd = indicators.get('macd', 0)
        macd_signal = indicators.get('macd_signal', 0)

       # Simple rules
        if rsi < 30 and macd > macd_signal:
            signal = 1
            confidence = 0.65
            reasoning = "Oversold with bullish MACD crossover"
        elif rsi > 70 and macd < macd_signal:
            signal = -1  # SELL/SHORT signal
            confidence = 0.65
            reasoning = "Overbought with bearish MACD crossover"
        else:
            signal = 0
            confidence = 0.5
            reasoning = "No clear signal"

        if self.commentary:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.MARKET_ANALYSIS,
                symbol=symbol,
                title=f"📊 Fallback Analysis: {symbol}",
                message=f"Using rule-based analysis: {reasoning}",
                data={'rsi': rsi, 'macd_vs_signal': macd > macd_signal},
                confidence=confidence,
                importance=5
            ))

        return signal, {
            'prediction': signal,
            'confidence': confidence,
            'method': 'fallback',
            'signal_strength': self._get_signal_strength(confidence),
            'reasoning': reasoning
        }

    def _add_prediction_commentary(self, prediction: MLPrediction, symbol: str):
        """Add commentary for ML prediction"""
        if not self.commentary:
            return

        # Build message
        if prediction.signal == 1:
            action = "BUY"
            emoji = "🟢"
        else:
            action = "HOLD"
            emoji = "⏸️"

        # Explain top features
        feature_explanation = []
        for feature, importance in prediction.feature_importance.items():
            if 'rsi' in feature and prediction.reasoning.get('rsi', 50) < 30:
                feature_explanation.append(f"RSI oversold ({prediction.reasoning['rsi']:.0f})")
            elif 'volume_ratio' in feature and prediction.reasoning.get('volume_ratio', 1) > 1.5:
                feature_explanation.append(f"Volume surge ({prediction.reasoning['volume_ratio']:.1f}x)")
            elif 'bb_position' in feature and prediction.reasoning.get('bb_position', 0.5) < 0.2:
                feature_explanation.append("Near lower Bollinger Band")

        self.commentary.add_commentary(TradingCommentary(
            timestamp=datetime.now(),
            type=CommentaryType.MARKET_ANALYSIS,
            symbol=symbol,
            title=f"{emoji} ML Signal: {action} {symbol}",
            message=f"Confidence: {prediction.confidence:.1%}\n" +
                   f"Key factors: {', '.join(feature_explanation) if feature_explanation else 'Multiple technical factors'}",
            data={
                'signal': prediction.signal,
                'confidence': prediction.confidence,
                'top_features': prediction.feature_importance
            },
            confidence=prediction.confidence,
            importance=7
        ))

    def _get_signal_strength(self, confidence: float) -> str:
        """Convert confidence to signal strength"""
        if confidence > 0.75:
            return "strong"
        elif confidence > 0.60:
            return "moderate"
        else:
            return "weak"

    async def retrain_model(self, data_provider, symbols: List[str]):
        """Retrain the ML model with recent data"""
        if self.commentary:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.MARKET_ANALYSIS,
                symbol=None,
                title="🔧 ML Model Retraining",
                message=f"Starting model retraining with {len(symbols)} symbols",
                importance=8
            ))

        # Collect training data
        X, y = self.model.collect_training_data(data_provider, symbols)

        if len(X) > 0:
            # Train model
            success = self.model.train(X, y)

            if success and self.brain:
                # Record in brain
                self.brain.remember_trade(
                    symbol="MODEL",
                    pattern="retraining",
                    outcome="completed",
                    pnl_percent=0,
                    context={
                        'samples': len(X),
                        'timestamp': datetime.now().isoformat()
                    }
                )
        else:
            if self.commentary:
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.WARNING,
                    symbol=None,
                    title="⚠️ Retraining Failed",
                    message="Could not collect sufficient training data",
                    importance=8
                ))

    def set_brain(self, brain: 'TradingBrain'):
        """Set reference to brain for learning integration"""
        self.brain = brain
        if self.model:
            self.model.brain = brain
