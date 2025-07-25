"""
Online Learning and Adaptive Model Update System for Scalping ML
Implements incremental learning, drift detection, and adaptive retraining
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Deque
from datetime import datetime, timedelta
from collections import deque, defaultdict
import logging
from pathlib import Path
import pickle
import json
import threading
import asyncio
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from river import drift, stats, metrics as river_metrics
import warnings
warnings.filterwarnings('ignore')

from scalping_ml_model import ScalpingMLModel, ScalpingFeatureEngineer

logger = logging.getLogger(__name__)

class OnlineLearningBuffer:
    """Manages online learning data buffer with efficient storage"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.features_buffer: Deque[np.ndarray] = deque(maxlen=max_size)
        self.labels_buffer: Deque[int] = deque(maxlen=max_size)
        self.metadata_buffer: Deque[Dict] = deque(maxlen=max_size)
        self.timestamps: Deque[datetime] = deque(maxlen=max_size)
        
        # Performance tracking
        self.predictions: Deque[Dict] = deque(maxlen=max_size)
        self.outcomes: Deque[Dict] = deque(maxlen=max_size)
        
    def add_sample(self, features: np.ndarray, prediction: Dict, metadata: Dict):
        """Add new prediction sample"""
        self.features_buffer.append(features)
        self.predictions.append(prediction)
        self.metadata_buffer.append(metadata)
        self.timestamps.append(datetime.now())
    
    def add_outcome(self, prediction_id: str, label: int, reward: float):
        """Add actual outcome for a prediction"""
        # Find matching prediction
        for i, pred in enumerate(self.predictions):
            if pred.get('id') == prediction_id:
                # Update with outcome
                self.labels_buffer.append(label)
                self.outcomes.append({
                    'prediction_id': prediction_id,
                    'label': label,
                    'reward': reward,
                    'timestamp': datetime.now()
                })
                break
    
    def get_training_batch(self, batch_size: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Get batch of samples with labels for training"""
        # Get samples that have outcomes
        X, y = [], []
        
        for i in range(min(batch_size, len(self.labels_buffer))):
            if i < len(self.features_buffer) and i < len(self.labels_buffer):
                X.append(self.features_buffer[-(i+1)])
                y.append(self.labels_buffer[-(i+1)])
        
        return np.array(X) if X else np.array([]), np.array(y) if y else np.array([])
    
    def get_recent_performance(self, window: int = 100) -> Dict[str, float]:
        """Calculate recent performance metrics"""
        recent_outcomes = list(self.outcomes)[-window:]
        
        if not recent_outcomes:
            return {'samples': 0}
        
        rewards = [o['reward'] for o in recent_outcomes]
        
        return {
            'samples': len(recent_outcomes),
            'avg_reward': np.mean(rewards),
            'win_rate': sum(1 for r in rewards if r > 0) / len(rewards),
            'sharpe': np.sqrt(252) * np.mean(rewards) / (np.std(rewards) + 1e-10)
        }


class AdaptiveDriftDetector:
    """Advanced drift detection for scalping models"""
    
    def __init__(self):
        # Multiple drift detectors for robustness
        self.adwin = drift.ADWIN()  # Adaptive Windowing
        self.ddm = drift.DDM()  # Drift Detection Method
        self.eddm = drift.EDDM()  # Early Drift Detection Method
        
        # Feature drift tracking
        self.feature_stats = defaultdict(lambda: stats.Mean())
        self.feature_drift = defaultdict(bool)
        
        # Performance drift
        self.performance_tracker = stats.RollingMean(window_size=100)
        self.baseline_performance = None
        
    def update(self, features: np.ndarray, prediction_correct: bool, reward: float):
        """Update drift detectors with new observation"""
        # Update accuracy-based drift detectors
        self.adwin.update(int(prediction_correct))
        self.ddm.update(int(prediction_correct))
        self.eddm.update(int(prediction_correct))
        
        # Update feature statistics
        for i, value in enumerate(features):
            self.feature_stats[i].update(value)
        
        # Update performance tracker
        self.performance_tracker.update(reward)
        
    def check_drift(self) -> Dict[str, Any]:
        """Check for various types of drift"""
        drift_info = {
            'concept_drift': False,
            'feature_drift': False,
            'performance_drift': False,
            'drift_type': None,
            'confidence': 0.0
        }
        
        # Check concept drift
        if self.adwin.drift_detected:
            drift_info['concept_drift'] = True
            drift_info['drift_type'] = 'sudden'
            drift_info['confidence'] = 0.9
        elif self.ddm.drift_detected:
            drift_info['concept_drift'] = True
            drift_info['drift_type'] = 'gradual'
            drift_info['confidence'] = 0.8
        elif self.eddm.drift_detected:
            drift_info['concept_drift'] = True
            drift_info['drift_type'] = 'early_warning'
            drift_info['confidence'] = 0.6
        
        # Check performance drift
        if self.baseline_performance and self.performance_tracker.get():
            current_perf = self.performance_tracker.get()
            if current_perf < self.baseline_performance * 0.7:  # 30% degradation
                drift_info['performance_drift'] = True
                drift_info['confidence'] = max(drift_info['confidence'], 0.85)
        
        # Check feature drift (simplified)
        drift_features = sum(1 for d in self.feature_drift.values() if d)
        if drift_features > len(self.feature_drift) * 0.2:  # 20% of features drifted
            drift_info['feature_drift'] = True
            drift_info['confidence'] = max(drift_info['confidence'], 0.7)
        
        return drift_info
    
    def reset(self):
        """Reset drift detectors after retraining"""
        self.adwin = drift.ADWIN()
        self.ddm = drift.DDM()
        self.eddm = drift.EDDM()
        self.feature_drift.clear()
        
        # Set new baseline
        if self.performance_tracker.get():
            self.baseline_performance = self.performance_tracker.get()


class OnlineScalpingModel:
    """Online learning wrapper for scalping ML model"""
    
    def __init__(self, base_model: ScalpingMLModel):
        self.base_model = base_model
        self.online_model = None
        self.is_initialized = False
        
        # Initialize online learner
        self._initialize_online_learner()
        
    def _initialize_online_learner(self):
        """Initialize fast online learning model"""
        self.online_model = SGDClassifier(
            loss='log',  # Logistic regression
            penalty='l2',
            alpha=0.001,
            learning_rate='adaptive',
            eta0=0.01,
            max_iter=1,  # Single pass for online learning
            warm_start=True,
            random_state=42
        )
        
    def partial_fit(self, X: np.ndarray, y: np.ndarray):
        """Incrementally update the online model"""
        if not self.is_initialized:
            # First fit needs all classes
            self.online_model.partial_fit(X, y, classes=[0, 1, 2])
            self.is_initialized = True
        else:
            self.online_model.partial_fit(X, y)
    
    def predict_ensemble(self, X: np.ndarray) -> Tuple[int, float, Dict]:
        """Ensemble prediction combining base and online models"""
        # Get base model prediction
        base_pred = self.base_model.predict(pd.DataFrame(), None, None)
        
        if self.is_initialized:
            # Get online model prediction
            online_pred = self.online_model.predict(X)[0]
            online_proba = self.online_model.predict_proba(X)[0]
            
            # Weighted ensemble (base model gets more weight initially)
            base_weight = 0.7
            online_weight = 0.3
            
            # Combine probabilities
            if isinstance(base_pred[2], dict) and 'confidence' in base_pred[2]:
                base_confidence = base_pred[2]['confidence']
                ensemble_confidence = base_weight * base_confidence + online_weight * np.max(online_proba)
                
                # Weighted vote for final prediction
                if base_pred[0] == online_pred:
                    final_pred = base_pred[0]
                else:
                    # Disagreement - use higher confidence
                    final_pred = base_pred[0] if base_confidence > np.max(online_proba) else online_pred
                
                return final_pred, ensemble_confidence, {
                    'base_pred': base_pred[0],
                    'online_pred': online_pred,
                    'ensemble_method': 'weighted',
                    'weights': {'base': base_weight, 'online': online_weight}
                }
        
        return base_pred


class ScalpingModelUpdater:
    """Manages online updates and adaptive retraining for scalping model"""
    
    def __init__(self, model: ScalpingMLModel, commentary_system=None):
        self.model = model
        self.commentary = commentary_system
        self._setup_commentary()
        
        # Components
        self.buffer = OnlineLearningBuffer()
        self.drift_detector = AdaptiveDriftDetector()
        self.online_model = OnlineScalpingModel(model)
        
        # Configuration
        self.config = {
            'min_samples_for_update': 50,
            'max_samples_before_retrain': 500,
            'drift_check_interval': 20,
            'performance_window': 100,
            'retrain_on_drift': True,
            'online_learning_enabled': True
        }
        
        # State
        self.samples_since_update = 0
        self.samples_since_retrain = 0
        self.last_update_time = datetime.now()
        self.last_retrain_time = datetime.now()
        
        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        self.update_history = []
        
        # Threading for async updates
        self.update_thread = None
        self.is_running = True
        
    def _setup_commentary(self):
        """Setup commentary helper"""
        self._commentary_available = False
        try:
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from trading_bot_commentary_updated import TradingCommentary, CommentaryType
            self.TradingCommentary = TradingCommentary
            self.CommentaryType = CommentaryType
            self._commentary_available = True
        except ImportError:
            logger.debug("TradingCommentary not available")
    
    def _add_commentary(self, comment_data: Dict):
        """Add commentary with proper object type"""
        if not self.commentary:
            return
            
        if self._commentary_available:
            # Map types
            type_mapping = {
                'DRIFT_DETECTION': self.CommentaryType.WARNING,
                'RETRAIN_SCHEDULED': self.CommentaryType.MARKET_ANALYSIS,
            }
            
            comment_type = type_mapping.get(
                comment_data.get('type', ''), 
                self.CommentaryType.MARKET_ANALYSIS
            )
            
            commentary = self.TradingCommentary(
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
            self.commentary.add_commentary(comment_data)
        
    def add_prediction(self, features: np.ndarray, prediction: Dict, 
                      symbol: str, metadata: Optional[Dict] = None):
        """Add new prediction to buffer"""
        pred_id = f"{symbol}_{datetime.now().timestamp()}"
        prediction['id'] = pred_id
        
        full_metadata = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'prediction_id': pred_id,
            **(metadata or {})
        }
        
        self.buffer.add_sample(features, prediction, full_metadata)
        self.samples_since_update += 1
        
        # Check if update needed
        if self.samples_since_update >= self.config['drift_check_interval']:
            self._check_and_update()
    
    def add_outcome(self, prediction_id: str, actual_price_change: float, 
                   success: bool):
        """Add actual outcome for a prediction"""
        # Determine actual label
        if actual_price_change > 0.003:
            actual_label = 2  # Buy was correct
        elif actual_price_change < -0.003:
            actual_label = 0  # Sell was correct
        else:
            actual_label = 1  # Hold was correct
        
        # Calculate reward (for scalping, we care about quick profits)
        reward = actual_price_change if success else -abs(actual_price_change)
        
        self.buffer.add_outcome(prediction_id, actual_label, reward)
        
        # Update drift detector
        # Note: In production, match features with prediction_id
        self.drift_detector.update(
            np.zeros(50),  # Placeholder
            success,
            reward
        )
    
    def _check_and_update(self):
        """Check for drift and update models if needed"""
        self.samples_since_update = 0
        
        # Check drift
        drift_info = self.drift_detector.check_drift()
        
        if drift_info['concept_drift'] or drift_info['performance_drift']:
            self._handle_drift(drift_info)
        
        # Perform online update if enabled
        if self.config['online_learning_enabled']:
            self._update_online_model()
        
        # Check if full retrain needed
        if self.samples_since_retrain >= self.config['max_samples_before_retrain']:
            self._schedule_retrain('scheduled')
    
    def _handle_drift(self, drift_info: Dict[str, Any]):
        """Handle detected drift"""
        logger.warning(f"Drift detected: {drift_info}")
        
        if self.commentary:
            self._add_commentary({
                'timestamp': datetime.now(),
                'type': 'DRIFT_DETECTION',
                'title': '⚠️ Model Drift Detected',
                'message': f"Type: {drift_info['drift_type']}\n"
                          f"Confidence: {drift_info['confidence']:.1%}",
                'data': drift_info,
                'importance': 8
            })
        
        if self.config['retrain_on_drift'] and drift_info['confidence'] > 0.7:
            self._schedule_retrain('drift')
    
    def _update_online_model(self):
        """Update online model with recent samples"""
        X_batch, y_batch = self.buffer.get_training_batch(
            self.config['min_samples_for_update']
        )
        
        if len(X_batch) > 0:
            try:
                # Update online model
                self.online_model.partial_fit(X_batch, y_batch)
                
                # Track update
                self.update_history.append({
                    'timestamp': datetime.now(),
                    'samples': len(X_batch),
                    'type': 'online_update'
                })
                
                logger.info(f"Online model updated with {len(X_batch)} samples")
                
            except Exception as e:
                logger.error(f"Online update error: {e}")
    
    def _schedule_retrain(self, reason: str):
        """Schedule full model retrain"""
        logger.info(f"Scheduling model retrain due to: {reason}")
        
        if self.commentary:
            self._add_commentary({
                'timestamp': datetime.now(),
                'type': 'RETRAIN_SCHEDULED',
                'title': '🔄 Model Retrain Scheduled',
                'message': f"Reason: {reason}",
                'importance': 7
            })
        
        # In production, this would trigger async retrain job
        # For now, just reset counters
        self.samples_since_retrain = 0
        self.drift_detector.reset()
    
    def get_adaptive_prediction(self, features: np.ndarray, 
                              base_prediction: Dict) -> Dict:
        """Get prediction with online learning adjustments"""
        if not self.config['online_learning_enabled']:
            return base_prediction
        
        # Get ensemble prediction
        pred, conf, info = self.online_model.predict_ensemble(features.reshape(1, -1))
        
        # Enhance base prediction
        enhanced_prediction = base_prediction.copy()
        enhanced_prediction.update({
            'adjusted_signal': pred,
            'adjusted_confidence': conf,
            'online_info': info,
            'drift_status': self.drift_detector.check_drift()
        })
        
        return enhanced_prediction
    
    def get_update_stats(self) -> Dict[str, Any]:
        """Get statistics about model updates"""
        recent_perf = self.buffer.get_recent_performance()
        
        stats = {
            'buffer_size': len(self.buffer.features_buffer),
            'samples_with_outcomes': len(self.buffer.labels_buffer),
            'samples_since_update': self.samples_since_update,
            'samples_since_retrain': self.samples_since_retrain,
            'last_update': self.last_update_time,
            'last_retrain': self.last_retrain_time,
            'online_updates': len(self.update_history),
            'recent_performance': recent_perf,
            'drift_detected': any(self.drift_detector.check_drift().values())
        }
        
        return stats
    
    def save_state(self, path: str = "scalping_online_state.pkl"):
        """Save online learning state"""
        state = {
            'buffer': self.buffer,
            'drift_detector': self.drift_detector,
            'online_model': self.online_model,
            'update_history': self.update_history,
            'performance_history': list(self.performance_history),
            'config': self.config
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    def load_state(self, path: str = "scalping_online_state.pkl"):
        """Load online learning state"""
        if Path(path).exists():
            with open(path, 'rb') as f:
                state = pickle.load(f)
            
            self.buffer = state['buffer']
            self.drift_detector = state['drift_detector']
            self.online_model = state['online_model']
            self.update_history = state['update_history']
            self.performance_history = deque(state['performance_history'], maxlen=1000)
            self.config.update(state['config'])


class AsyncModelRetrainer:
    """Handles asynchronous model retraining"""
    
    def __init__(self, model_path: str = "scalping_ml_model.pkl"):
        self.model_path = model_path
        self.is_retraining = False
        self.retrain_queue = asyncio.Queue()
        
    async def retrain_worker(self, data_provider):
        """Worker process for model retraining"""
        while True:
            try:
                # Get retrain request
                request = await self.retrain_queue.get()
                
                if request['type'] == 'stop':
                    break
                
                self.is_retraining = True
                logger.info(f"Starting model retrain: {request['reason']}")
                
                # Load current model
                model = ScalpingMLModel()
                model.load_model()
                
                # Collect new training data
                symbols = request.get('symbols', ['SPY', 'QQQ', 'AAPL', 'MSFT'])
                X, y = await self._collect_training_data(data_provider, symbols)
                
                if len(X) > 500:
                    # Retrain model
                    success = model.train(X, y, optimize_hyperparams=True)
                    
                    if success:
                        # Save new model
                        model.save_model()
                        logger.info("Model retrain completed successfully")
                    else:
                        logger.error("Model retrain failed")
                
                self.is_retraining = False
                
            except Exception as e:
                logger.error(f"Retrain worker error: {e}")
                self.is_retraining = False
                await asyncio.sleep(60)  # Wait before retry
    
    async def _collect_training_data(self, data_provider, 
                                   symbols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Collect training data asynchronously"""
        # Implementation would collect data similar to training pipeline
        # Simplified for demonstration
        return np.array([]), np.array([])
    
    async def schedule_retrain(self, reason: str, symbols: Optional[List[str]] = None):
        """Schedule a model retrain"""
        await self.retrain_queue.put({
            'type': 'retrain',
            'reason': reason,
            'symbols': symbols,
            'timestamp': datetime.now()
        })
    
    async def stop(self):
        """Stop the retrain worker"""
        await self.retrain_queue.put({'type': 'stop'})


# Integration function
def create_model_updater(model: ScalpingMLModel, 
                        commentary_system=None) -> ScalpingModelUpdater:
    """Create model updater instance"""
    return ScalpingModelUpdater(model, commentary_system)


# Example usage
async def update_model_online(updater: ScalpingModelUpdater,
                            features: np.ndarray,
                            prediction: Dict,
                            symbol: str):
    """Update model with new prediction"""
    # Add prediction
    updater.add_prediction(features, prediction, symbol)
    
    # Get adaptive prediction
    enhanced_pred = updater.get_adaptive_prediction(features, prediction)
    
    return enhanced_pred