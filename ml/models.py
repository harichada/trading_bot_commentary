import logging
import pickle
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from ml.features import (
    AdvancedFeatureEngineer, MarketRegimeDetector, ConceptDriftDetector,
    MLFeatureExtractor, MLTrainingRecord, MLPrediction
)

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

logger = logging.getLogger('TradingBot')


class HybridTradingModel:
    """Production-ready hybrid ML model with adaptive retraining"""

    def __init__(self, commentary_system=None):
        self.commentary = commentary_system
        self.feature_engineer = AdvancedFeatureEngineer()
        self.regime_detector = MarketRegimeDetector()
        self.drift_detector = ConceptDriftDetector()

        # Model ensemble
        self.models = {
            'rf': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=50,
                min_samples_leaf=20,
                random_state=42,
                n_jobs=-1
            ),
            'xgb': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            ),
            'lgb': lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                random_state=42,
                verbose=-1
            )
        }

        # Meta learner
        self.meta_model = VotingClassifier(
            estimators=[(name, model) for name, model in self.models.items()],
            voting='soft',
            weights=[0.4, 0.4, 0.2]  # RF and XGB weighted more
        )

        self.scaler = RobustScaler()  # Robust to outliers
        self.is_trained = False  # Only true after actual training
        self.last_training_time = None
        self.training_history = []
        self.feature_importance = {}

        # Performance tracking
        self.performance_metrics = {
            'accuracy': [],
            'sharpe_ratio': [],
            'max_drawdown': [],
            'win_rate': []
        }

    def should_retrain(self, current_metrics: Dict[str, float]) -> bool:
        """Determine if model should be retrained based on multiple factors"""
        if not self.last_training_time:
            return True

        # Time-based retraining
        regime = self.regime_detector.current_regime
        if regime == "high_volatility":
            retrain_interval = timedelta(hours=6)  # Retrain every 6 hours
        elif regime == "trending":
            retrain_interval = timedelta(days=1)
        else:
            retrain_interval = timedelta(days=3)

        if datetime.now() - self.last_training_time > retrain_interval:
            return True

        # Performance-based retraining
        if len(self.performance_metrics['sharpe_ratio']) >= 5:
            recent_sharpe = np.mean(self.performance_metrics['sharpe_ratio'][-5:])
            if recent_sharpe < 0.5:
                return True

        if len(self.performance_metrics['accuracy']) >= 10:
            recent_accuracy = np.mean(self.performance_metrics['accuracy'][-10:])
            if recent_accuracy < 0.55:
                return True

        # Drift-based retraining
        if self.drift_detector.drift_detected:
            return True

        return False

    def train(self, training_data: Dict[str, pd.DataFrame], labels: pd.Series,
              market_conditions: Dict[str, float]):
        """Train the hybrid model with multi-timeframe data"""
        if self.commentary:
            from core.commentary import TradingCommentary
            from core.models import CommentaryType
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.MARKET_ANALYSIS,
                symbol=None,
                title="🎯 Training Advanced ML Model",
                message="Extracting multi-timeframe features and training ensemble...",
                importance=8
            ))

        # Extract features for all samples
        X = []
        valid_indices = []

        for i in range(len(labels)):
            try:
                # Get historical data for each timeframe
                sample_data = {}
                for tf in self.feature_engineer.timeframes:
                    if tf in training_data:
                        # Get data up to index i
                        sample_data[tf] = training_data[tf].iloc[max(0, i-100):i+1]

                if all(tf in sample_data and len(sample_data[tf]) > 20
                       for tf in ['1min', '5min']):  # Minimum required
                    features = self.feature_engineer.extract_multi_timeframe_features(sample_data)
                    X.append(features)
                    valid_indices.append(i)
            except Exception as e:
                continue

        if len(X) < 100:
            if self.commentary:
                from core.commentary import TradingCommentary
                from core.models import CommentaryType
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.WARNING,
                    symbol=None,
                    title="⚠️ Insufficient Training Data",
                    message=f"Only {len(X)} valid samples available",
                    importance=8
                ))
            return

        X = np.array(X)
        y = labels.iloc[valid_indices].values

        # Handle any remaining NaN or inf values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []

        for train_idx, val_idx in tscv.split(X_scaled):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Train individual models
            for name, model in self.models.items():
                model.fit(X_train, y_train)

            # Train meta model
            self.meta_model.fit(X_train, y_train)

            # Validate
            val_pred = self.meta_model.predict(X_val)
            accuracy = accuracy_score(y_val, val_pred)
            cv_scores.append(accuracy)

        # Final training on all data
        self.meta_model.fit(X_scaled, y)

        # Calculate feature importance
        self._calculate_feature_importance(X, y)

        # Update training metadata
        self.is_trained = True
        self.last_training_time = datetime.now()
        self.training_history.append({
            'timestamp': self.last_training_time,
            'samples': len(X),
            'cv_accuracy': np.mean(cv_scores),
            'regime': self.regime_detector.current_regime
        })

        if self.commentary:
            from core.commentary import TradingCommentary
            from core.models import CommentaryType
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.MARKET_ANALYSIS,
                symbol=None,
                title="✅ Model Training Complete",
                message=f"Trained on {len(X)} samples with {np.mean(cv_scores):.1%} CV accuracy",
                data={
                    'models': list(self.models.keys()),
                    'features': len(X[0]),
                    'regime': self.regime_detector.current_regime
                },
                confidence=np.mean(cv_scores),
                importance=8
            ))

    def predict(self, market_data: Dict[str, pd.DataFrame], symbol: str) -> Tuple[int, Dict[str, Any]]:
        """Make prediction with confidence and explanation"""
        if not self.is_trained:
            return 0, {'error': 'Model not trained'}

        try:
            # Extract features
            features = self.feature_engineer.extract_multi_timeframe_features(market_data)
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            X = features.reshape(1, -1)
            X_scaled = self.scaler.transform(X)

            # Get predictions from all models
            predictions = {}
            probabilities = {}

            for name, model in self.models.items():
                pred = model.predict(X_scaled)[0]
                prob = model.predict_proba(X_scaled)[0]
                predictions[name] = pred
                probabilities[name] = max(prob)

            # Meta prediction
            final_prediction = self.meta_model.predict(X_scaled)[0]
            final_probability = self.meta_model.predict_proba(X_scaled)[0]
            confidence = max(final_probability)

            # Generate explanation
            explanation = self._generate_prediction_explanation(
                features, predictions, probabilities, final_prediction, confidence
            )

            # Update drift detector
            # Note: In production, you'd check actual outcome later
            self.drift_detector.update(True)  # Placeholder

            return int(final_prediction), explanation

        except Exception as e:
            if self.commentary:
                from core.commentary import TradingCommentary
                from core.models import CommentaryType
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.WARNING,
                    symbol=symbol,
                    title="⚠️ Prediction Error",
                    message=str(e),
                    importance=7
                ))
            return 0, {'error': str(e)}

    def _calculate_feature_importance(self, X: np.ndarray, y: np.ndarray):
        """Calculate and store feature importance"""
        # Use RandomForest feature importance as primary
        rf_importance = self.models['rf'].feature_importances_

        # Get XGBoost importance
        xgb_importance = self.models['xgb'].feature_importances_

        # Average importances
        avg_importance = (rf_importance + xgb_importance) / 2

        # Store top features
        top_indices = np.argsort(avg_importance)[-20:][::-1]
        self.feature_importance = {
            f'feature_{i}': float(avg_importance[i])
            for i in top_indices
        }

    def _generate_prediction_explanation(self, features: np.ndarray,
                                       predictions: Dict[str, int],
                                       probabilities: Dict[str, float],
                                       final_prediction: int,
                                       confidence: float) -> Dict[str, Any]:
        """Generate human-readable explanation for prediction"""
        explanation = {
            'prediction': int(final_prediction),
            'confidence': float(confidence),
            'model_votes': predictions,
            'model_confidences': probabilities,
            'regime': self.regime_detector.current_regime,
            'top_features': self.feature_importance,
            'signal_strength': 'strong' if confidence > 0.7 else 'moderate' if confidence > 0.55 else 'weak'
        }

        # Add regime-specific insights
        if self.regime_detector.current_regime == "high_volatility":
            explanation['note'] = "High volatility regime - position size should be reduced"
        elif self.regime_detector.current_regime == "trending":
            explanation['note'] = "Trending market - momentum strategies preferred"

        return explanation

    def update_performance(self, prediction_correct: bool, pnl: float):
        """Update model performance metrics"""
        self.performance_metrics['accuracy'].append(1 if prediction_correct else 0)

        # Update other metrics (simplified)
        if len(self.performance_metrics['accuracy']) > 100:
            # Keep only recent history
            for key in self.performance_metrics:
                self.performance_metrics[key] = self.performance_metrics[key][-100:]

    def save_model(self, path: Path):
        """Save model and metadata"""
        model_data = {
            'models': self.models,
            'meta_model': self.meta_model,
            'scaler': self.scaler,
            'feature_importance': self.feature_importance,
            'training_history': self.training_history,
            'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None
        }

        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

    def load_model(self, path: Path):
        """Load model and metadata"""
        if path.exists():
            with open(path, 'rb') as f:
                model_data = pickle.load(f)

            self.models = model_data['models']
            self.meta_model = model_data['meta_model']
            self.scaler = model_data['scaler']
            self.feature_importance = model_data.get('feature_importance', {})
            self.training_history = model_data.get('training_history', [])

            if model_data.get('last_training_time'):
                self.last_training_time = datetime.fromisoformat(model_data['last_training_time'])

            self.is_trained = True


class IntegratedMLModel:
    """ML model that integrates seamlessly with existing bot architecture"""

    def __init__(self, brain: 'TradingBrain' = None, commentary_system = None):
        self.brain = brain
        self.commentary = commentary_system
        self.feature_extractor = MLFeatureExtractor()

        # Model components
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False

        # Model metadata
        self.model_version = "2.0"
        self.last_training_time = None
        self.training_samples = 0
        self.feature_names = self.feature_extractor.feature_names
        self.performance_history = []

        # Paths
        self.model_path = Path("ml_model_integrated.pkl")
        self.training_data_path = Path("ml_training_data.json")

        # Load existing model
        self._load_model()

    def collect_training_data(self, data_provider, symbols: List[str],
                            lookback_days: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """Collect training data from historical market data"""
        if self.commentary:
            from core.commentary import TradingCommentary
            from core.models import CommentaryType
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.MARKET_ANALYSIS,
                symbol=None,
                title="📊 Collecting Training Data",
                message=f"Gathering {lookback_days} days of data from {len(symbols)} symbols",
                importance=7
            ))

        all_features = []
        all_labels = []

        for symbol in symbols:
            try:
                # Get historical data
                df = data_provider.get_market_data(
                    symbol,
                    period_type='month',
                    period=1,
                    frequency_type='minute',
                    frequency=5
                )

                if df.empty or len(df) < 100:
                    continue

                # Generate training samples using sliding window
                for i in range(50, len(df) - 20):  # Need history and future
                    # Extract features from data up to point i
                    window_data = df.iloc[:i+1]
                    features = self.feature_extractor.extract_from_dataframe(window_data)

                    # Create label: 1 if price increases by 0.3% in next 10 bars
                    current_price = df['Close'].iloc[i]
                    future_price = df['Close'].iloc[i + 10]
                    price_change = (future_price - current_price) / current_price

                    # Label based on whether move was profitable
                    # 3-class labels: 2=BUY, 1=HOLD, 0=SELL
                    if price_change > 0.003:
                        label = 2  # BUY
                    elif price_change < -0.003:
                        label = 0  # SELL (SHORT)
                    else:
                        label = 1  # HOLD

                    # Convert features to array
                    feature_array = [features[name] for name in self.feature_names]
                    all_features.append(feature_array)
                    all_labels.append(label)

                    # Store for brain learning if available
                    if self.brain and label == 1:
                        self.brain.remember_trade(
                            symbol=symbol,
                            pattern="ml_training",
                            outcome="win" if price_change > 0.003 else "loss",
                            pnl_percent=price_change * 100,
                            context={'features': features, 'training': True}
                        )

            except Exception as e:
                logger.error(f"Error collecting data for {symbol}: {e}")
                continue

        if self.commentary:
            from core.commentary import TradingCommentary
            from core.models import CommentaryType
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.MARKET_ANALYSIS,
                symbol=None,
                title="✅ Training Data Collected",
                message=f"Collected {len(all_features)} training samples",
                importance=6
            ))

        return np.array(all_features), np.array(all_labels)

    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the ML model with proper validation"""
        if len(X) < 100:
            if self.commentary:
                from core.commentary import TradingCommentary
                from core.models import CommentaryType
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.WARNING,
                    symbol=None,
                    title="⚠️ Insufficient Training Data",
                    message=f"Need at least 100 samples, have {len(X)}",
                    importance=8
                ))
            return False

        try:
            # Scale features
            X_scaled = self.scaler.fit_transform(X)

            # Train XGBoost model
            self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='mlogloss',  # Multi-class logloss
            use_label_encoder=False,
            # early_stopping_rounds moved to fit() call
            num_class=3,  # 3 classes
            objective='multi:softprob'  # Multi-class classification
        )

            # Train with validation split
            split_idx = int(0.8 * len(X))
            X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=10,
                verbose=False
            )

            # Calculate validation accuracy
            val_pred = self.model.predict(X_val)
            accuracy = np.mean(val_pred == y_val)

            # Update metadata
            self.is_trained = True
            self.last_training_time = datetime.now()
            self.training_samples = len(X)

            # Save model
            self._save_model()

            if self.commentary:
                from core.commentary import TradingCommentary
                from core.models import CommentaryType
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.MARKET_ANALYSIS,
                    symbol=None,
                    title="✅ Model Training Complete",
                    message=f"Trained on {len(X)} samples\nValidation accuracy: {accuracy:.1%}",
                    data={
                        'samples': len(X),
                        'features': len(self.feature_names),
                        'accuracy': accuracy,
                        'model': 'XGBoost'
                    },
                    confidence=accuracy,
                    importance=8
                ))

            return True

        except Exception as e:
            logger.error(f"Training error: {e}")
            if self.commentary:
                from core.commentary import TradingCommentary
                from core.models import CommentaryType
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.WARNING,
                    symbol=None,
                    title="❌ Training Failed",
                    message=str(e),
                    importance=9
                ))
            return False

    def predict(self, df: pd.DataFrame, symbol: str) -> MLPrediction:
        """Make prediction using same data format as training"""
        # Default prediction if not trained
        if not self.is_trained:
            return MLPrediction(
                signal=0,
                confidence=0.5,
                reasoning={'method': 'not_trained'},
                feature_importance={},
                timestamp=datetime.now()
            )

        try:
            # Extract features from dataframe
            features = self.feature_extractor.extract_from_dataframe(df)

            # Convert to array
            X = np.array([features[name] for name in self.feature_names]).reshape(1, -1)

            # Scale
            X_scaled = self.scaler.transform(X)

            # Predict
            prediction = self.model.predict(X_scaled)[0]
            probabilities = self.model.predict_proba(X_scaled)[0]
            confidence = float(max(probabilities))

            # Get feature importance
            if hasattr(self.model, 'feature_importances_'):
                importance_dict = {
                    name: float(imp)
                    for name, imp in zip(self.feature_names, self.model.feature_importances_)
                }
                # Get top 5 important features
                top_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:5]
            else:
                top_features = []

            # Generate reasoning
            reasoning = {
                'signal': int(prediction),
                'confidence': confidence,
                'top_features': dict(top_features),
                'rsi': features.get('rsi', 0) * 100,  # Convert back to 0-100
                'volume_ratio': features.get('volume_ratio', 1),
                'bb_position': features.get('bb_position', 0.5)
            }
            # Map XGBoost output back to trading signals
            prediction_map = {0: -1, 1: 0, 2: 1}  # 0->SELL, 1->HOLD, 2->BUY
            mapped_prediction = prediction_map[int(prediction)]
            return MLPrediction(
                signal=int(prediction),
                confidence=confidence,
                reasoning=reasoning,
                feature_importance=dict(top_features) if top_features else {},
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.error(f"Prediction error for {symbol}: {e}")
            return MLPrediction(
                signal=0,
                confidence=0.5,
                reasoning={'error': str(e)},
                feature_importance={},
                timestamp=datetime.now()
            )

    def should_retrain(self, performance_metrics: Dict[str, float]) -> bool:
        """Determine if model needs retraining"""
        if not self.is_trained:
            return True

        # Time-based retraining
        if self.last_training_time:
            days_since_training = (datetime.now() - self.last_training_time).days
            if days_since_training > 7:  # Weekly retraining
                return True

        # Performance-based retraining
        if 'accuracy' in performance_metrics and performance_metrics['accuracy'] < 0.55:
            return True

        if 'sharpe_ratio' in performance_metrics and performance_metrics['sharpe_ratio'] < 0.5:
            return True

        return False

    def _save_model(self):
        """Save model and metadata"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_version': self.model_version,
            'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
            'training_samples': self.training_samples,
            'performance_history': self.performance_history[-100:]  # Keep last 100
        }

        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)

    def _load_model(self):
        """Load existing model"""
        if not self.model_path.exists():
            return

        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)

            # Check version compatibility
            if model_data.get('model_version') != self.model_version:
                logger.warning(f"Model version mismatch, retraining needed")
                return

            loaded_model = model_data['model']

            from sklearn.utils.validation import check_is_fitted
            from sklearn.exceptions import NotFittedError
            try:
                check_is_fitted(loaded_model)
            except NotFittedError:
                logger.warning("Loaded ML model is not fitted; ignoring stale pickle and using fallback rules")
                return

            self.model = loaded_model
            self.scaler = model_data['scaler']
            self.feature_names = model_data.get('feature_names', self.feature_names)
            self.training_samples = model_data.get('training_samples', 0)
            self.performance_history = model_data.get('performance_history', [])

            if model_data.get('last_training_time'):
                self.last_training_time = datetime.fromisoformat(model_data['last_training_time'])

            self.is_trained = True
            logger.info(f"Loaded ML model v{self.model_version} trained on {self.training_samples} samples")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
