#!/usr/bin/env python3
"""
Safe ML Model Manager - Works without XGBoost/LightGBM
Only uses scikit-learn models to avoid segmentation faults
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np
import pandas as pd
from datetime import datetime
import json
import joblib
import logging
from pathlib import Path

# Only use scikit-learn models (no XGBoost/LightGBM)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger('MLModelManager')

@dataclass
class ModelConfig:
    """Configuration for an ML model"""
    name: str
    model_type: str  # 'classifier' or 'regressor'
    algorithm: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    feature_config: Dict[str, Any] = field(default_factory=dict)
    training_config: Dict[str, Any] = field(default_factory=lambda: {
        'test_size': 0.2,
        'n_splits': 5,
        'validation_method': 'time_series_split'
    })
    performance_metrics: Dict[str, float] = field(default_factory=dict)

class BaseMLModel:
    """Base class for all ML models"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.is_trained = False
        
    def create_model(self):
        """Create the underlying ML model"""
        raise NotImplementedError
    
    def train(self, X: pd.DataFrame, y: pd.Series, validation_data: Optional[Tuple] = None):
        """Train the model"""
        if self.model is None:
            self.create_model()
        
        # Preprocess features
        X_scaled = self.preprocess_features(X, fit=True)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        # Evaluate on validation data if provided
        if validation_data:
            X_val, y_val = validation_data
            X_val_scaled = self.preprocess_features(X_val, fit=False)
            metrics = self.evaluate(X_val_scaled, y_val)
            logger.info(f"Validation metrics: {metrics}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        X_scaled = self.preprocess_features(X, fit=False)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities (for classifiers)"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        if self.config.model_type != 'classifier':
            raise ValueError("predict_proba only available for classifiers")
        
        X_scaled = self.preprocess_features(X, fit=False)
        return self.model.predict_proba(X_scaled)
    
    def preprocess_features(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Preprocess features with scaling"""
        if fit:
            scaler_type = self.config.feature_config.get('scaler', 'standard')
            if scaler_type == 'standard':
                self.scaler = StandardScaler()
            elif scaler_type == 'robust':
                self.scaler = RobustScaler()
            elif scaler_type == 'minmax':
                self.scaler = MinMaxScaler()
            else:
                self.scaler = StandardScaler()
            
            X_scaled = self.scaler.fit_transform(X)
            self.feature_names = X.columns.tolist()
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Train model first.")
            X_scaled = self.scaler.transform(X)
        
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model performance"""
        predictions = self.predict(X)
        
        if self.config.model_type == 'classifier':
            accuracy = accuracy_score(y, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(y, predictions, average='weighted', zero_division=0)
            
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        else:
            mse = mean_squared_error(y, predictions)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y - predictions))
            
            metrics = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae
            }
        
        self.config.performance_metrics = metrics
        return metrics
    
    def save(self, filepath: str):
        """Save model to disk"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'config': self.config,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model from disk"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.config = model_data['config']
        self.feature_names = model_data['feature_names']
        self.is_trained = model_data['is_trained']
        logger.info(f"Model loaded from {filepath}")

class RandomForestModel(BaseMLModel):
    """Random Forest implementation"""
    
    def create_model(self):
        """Create Random Forest model"""
        if self.config.model_type == 'classifier':
            self.model = RandomForestClassifier(
                n_estimators=self.config.parameters.get('n_estimators', 100),
                max_depth=self.config.parameters.get('max_depth', None),
                min_samples_split=self.config.parameters.get('min_samples_split', 2),
                min_samples_leaf=self.config.parameters.get('min_samples_leaf', 1),
                max_features=self.config.parameters.get('max_features', 'sqrt'),
                random_state=42,
                n_jobs=-1
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=self.config.parameters.get('n_estimators', 100),
                max_depth=self.config.parameters.get('max_depth', None),
                min_samples_split=self.config.parameters.get('min_samples_split', 2),
                min_samples_leaf=self.config.parameters.get('min_samples_leaf', 1),
                max_features=self.config.parameters.get('max_features', 'sqrt'),
                random_state=42,
                n_jobs=-1
            )
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importances"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        importances = self.model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return feature_importance

class GradientBoostingModel(BaseMLModel):
    """Gradient Boosting implementation (scikit-learn version)"""
    
    def create_model(self):
        """Create Gradient Boosting model"""
        if self.config.model_type == 'classifier':
            self.model = GradientBoostingClassifier(
                n_estimators=self.config.parameters.get('n_estimators', 100),
                max_depth=self.config.parameters.get('max_depth', 3),
                learning_rate=self.config.parameters.get('learning_rate', 0.1),
                subsample=self.config.parameters.get('subsample', 0.8),
                random_state=42
            )
        else:
            from sklearn.ensemble import GradientBoostingRegressor
            self.model = GradientBoostingRegressor(
                n_estimators=self.config.parameters.get('n_estimators', 100),
                max_depth=self.config.parameters.get('max_depth', 3),
                learning_rate=self.config.parameters.get('learning_rate', 0.1),
                subsample=self.config.parameters.get('subsample', 0.8),
                random_state=42
            )

class LogisticRegressionModel(BaseMLModel):
    """Logistic Regression implementation"""
    
    def create_model(self):
        """Create Logistic Regression model"""
        if self.config.model_type != 'classifier':
            raise ValueError("Logistic Regression is only for classification")
        
        self.model = LogisticRegression(
            C=self.config.parameters.get('C', 1.0),
            max_iter=self.config.parameters.get('max_iter', 1000),
            solver=self.config.parameters.get('solver', 'lbfgs'),
            random_state=42
        )

class NeuralNetworkModel(BaseMLModel):
    """Neural Network implementation using sklearn"""
    
    def create_model(self):
        """Create Neural Network model"""
        if self.config.model_type == 'classifier':
            self.model = MLPClassifier(
                hidden_layer_sizes=self.config.parameters.get('hidden_layers', (100, 50)),
                activation=self.config.parameters.get('activation', 'relu'),
                solver=self.config.parameters.get('solver', 'adam'),
                alpha=self.config.parameters.get('alpha', 0.0001),
                learning_rate=self.config.parameters.get('learning_rate', 'constant'),
                learning_rate_init=self.config.parameters.get('learning_rate_init', 0.001),
                max_iter=self.config.parameters.get('max_iter', 200),
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42
            )
        else:
            from sklearn.neural_network import MLPRegressor
            self.model = MLPRegressor(
                hidden_layer_sizes=self.config.parameters.get('hidden_layers', (100, 50)),
                activation=self.config.parameters.get('activation', 'relu'),
                solver=self.config.parameters.get('solver', 'adam'),
                alpha=self.config.parameters.get('alpha', 0.0001),
                learning_rate=self.config.parameters.get('learning_rate', 'constant'),
                learning_rate_init=self.config.parameters.get('learning_rate_init', 0.001),
                max_iter=self.config.parameters.get('max_iter', 200),
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42
            )

class EnsembleModel(BaseMLModel):
    """Ensemble of multiple models"""
    
    def __init__(self, config: ModelConfig, models: List[BaseMLModel]):
        super().__init__(config)
        self.models = models
        self.weights = config.parameters.get('weights', None)
        
    def create_model(self):
        """Not needed for ensemble"""
        pass
    
    def train(self, X: pd.DataFrame, y: pd.Series, validation_data: Optional[Tuple] = None):
        """Train all models in ensemble"""
        for model in self.models:
            logger.info(f"Training {model.config.name}...")
            model.train(X, y, validation_data)
        
        self.is_trained = True
        
        # Evaluate ensemble performance
        if validation_data:
            X_val, y_val = validation_data
            metrics = self.evaluate(X_val, y_val)
            logger.info(f"Ensemble validation metrics: {metrics}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions"""
        if not self.is_trained:
            raise ValueError("Models not trained yet")
        
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Stack predictions
        predictions = np.array(predictions)
        
        if self.config.model_type == 'classifier':
            # Majority voting
            from scipy import stats
            ensemble_pred = stats.mode(predictions, axis=0)[0].flatten()
        else:
            # Average for regression
            if self.weights:
                weights = np.array(self.weights).reshape(-1, 1)
                ensemble_pred = np.average(predictions, axis=0, weights=weights)
            else:
                ensemble_pred = np.mean(predictions, axis=0)
        
        return ensemble_pred
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get ensemble prediction probabilities"""
        if not self.is_trained:
            raise ValueError("Models not trained yet")
        
        if self.config.model_type != 'classifier':
            raise ValueError("predict_proba only available for classifiers")
        
        probas = []
        for model in self.models:
            proba = model.predict_proba(X)
            probas.append(proba)
        
        # Average probabilities
        if self.weights:
            weights = np.array(self.weights).reshape(-1, 1, 1)
            ensemble_proba = np.average(probas, axis=0, weights=weights)
        else:
            ensemble_proba = np.mean(probas, axis=0)
        
        return ensemble_proba

class ModelManager:
    """Manages multiple ML models - Safe version without XGBoost/LightGBM"""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.models: Dict[str, BaseMLModel] = {}
        self.active_model: Optional[str] = None
        self._initialize_default_models()
    
    def _initialize_default_models(self):
        """Initialize default model configurations"""
        default_configs = {
            'random_forest': ModelConfig(
                name='Random Forest Classifier',
                model_type='classifier',
                algorithm='random_forest',
                parameters={
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5
                }
            ),
            'gradient_boosting': ModelConfig(
                name='Gradient Boosting Classifier',
                model_type='classifier',
                algorithm='gradient_boosting',
                parameters={
                    'n_estimators': 100,
                    'max_depth': 3,
                    'learning_rate': 0.1
                }
            ),
            'logistic_regression': ModelConfig(
                name='Logistic Regression',
                model_type='classifier',
                algorithm='logistic_regression',
                parameters={
                    'C': 1.0,
                    'max_iter': 1000
                }
            ),
            'neural_network': ModelConfig(
                name='Neural Network',
                model_type='classifier',
                algorithm='neural_network',
                parameters={
                    'hidden_layers': (100, 50, 25),
                    'activation': 'relu',
                    'learning_rate_init': 0.001
                }
            )
        }
        
        # Create model instances
        model_classes = {
            'random_forest': RandomForestModel,
            'gradient_boosting': GradientBoostingModel,
            'logistic_regression': LogisticRegressionModel,
            'neural_network': NeuralNetworkModel
        }
        
        for key, config in default_configs.items():
            model_class = model_classes.get(config.algorithm)
            if model_class:
                self.models[key] = model_class(config)
    
    def add_model(self, key: str, model: BaseMLModel):
        """Add a new model"""
        self.models[key] = model
        logger.info(f"Added model: {key}")
    
    def remove_model(self, key: str):
        """Remove a model"""
        if key in self.models:
            del self.models[key]
            if self.active_model == key:
                self.active_model = None
            logger.info(f"Removed model: {key}")
    
    def set_active_model(self, key: str):
        """Set the active model for predictions"""
        if key in self.models:
            self.active_model = key
            logger.info(f"Active model set to: {key}")
        else:
            raise ValueError(f"Model {key} not found")
    
    def train_model(self, key: str, X: pd.DataFrame, y: pd.Series, 
                    validation_data: Optional[Tuple] = None):
        """Train a specific model"""
        if key not in self.models:
            raise ValueError(f"Model {key} not found")
        
        model = self.models[key]
        logger.info(f"Training {key}...")
        model.train(X, y, validation_data)
        
        # Save trained model
        model_path = self.model_dir / f"{key}_model.pkl"
        model.save(str(model_path))
    
    def train_all_models(self, X: pd.DataFrame, y: pd.Series, 
                        validation_data: Optional[Tuple] = None):
        """Train all models"""
        for key in self.models:
            try:
                self.train_model(key, X, y, validation_data)
            except Exception as e:
                logger.error(f"Error training {key}: {e}")
    
    def predict(self, X: pd.DataFrame, model_key: Optional[str] = None) -> np.ndarray:
        """Make predictions using specified or active model"""
        model_key = model_key or self.active_model
        if not model_key:
            raise ValueError("No model specified and no active model set")
        
        if model_key not in self.models:
            raise ValueError(f"Model {model_key} not found")
        
        return self.models[model_key].predict(X)
    
    def predict_proba(self, X: pd.DataFrame, model_key: Optional[str] = None) -> np.ndarray:
        """Get prediction probabilities"""
        model_key = model_key or self.active_model
        if not model_key:
            raise ValueError("No model specified and no active model set")
        
        if model_key not in self.models:
            raise ValueError(f"Model {model_key} not found")
        
        return self.models[model_key].predict_proba(X)
    
    def create_ensemble(self, model_keys: List[str], weights: Optional[List[float]] = None) -> EnsembleModel:
        """Create an ensemble from multiple models"""
        models = []
        for key in model_keys:
            if key in self.models:
                models.append(self.models[key])
            else:
                logger.warning(f"Model {key} not found, skipping")
        
        if not models:
            raise ValueError("No valid models found for ensemble")
        
        ensemble_config = ModelConfig(
            name='Ensemble Model',
            model_type=models[0].config.model_type,
            algorithm='ensemble',
            parameters={'weights': weights}
        )
        
        ensemble = EnsembleModel(ensemble_config, models)
        self.models['ensemble'] = ensemble
        
        return ensemble
    
    def compare_models(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Compare performance of all trained models"""
        results = []
        
        for key, model in self.models.items():
            if model.is_trained:
                try:
                    metrics = model.evaluate(X, y)
                    results.append({
                        'model': key,
                        'algorithm': model.config.algorithm,
                        **metrics
                    })
                except Exception as e:
                    logger.error(f"Error evaluating {key}: {e}")
        
        return pd.DataFrame(results).sort_values(
            by='accuracy' if results and 'accuracy' in results[0] else 'rmse',
            ascending=False if results and 'accuracy' in results[0] else True
        )
    
    def save_all_models(self):
        """Save all trained models"""
        for key, model in self.models.items():
            if model.is_trained:
                model_path = self.model_dir / f"{key}_model.pkl"
                model.save(str(model_path))
    
    def load_model(self, key: str, filepath: Optional[str] = None):
        """Load a saved model"""
        if filepath is None:
            filepath = str(self.model_dir / f"{key}_model.pkl")
        
        if key in self.models:
            self.models[key].load(filepath)
        else:
            # Try to determine model type from saved data
            import joblib
            model_data = joblib.load(filepath)
            config = model_data['config']
            
            model_classes = {
                'random_forest': RandomForestModel,
                'gradient_boosting': GradientBoostingModel,
                'logistic_regression': LogisticRegressionModel,
                'neural_network': NeuralNetworkModel
            }
            
            model_class = model_classes.get(config.algorithm)
            if model_class:
                model = model_class(config)
                model.load(filepath)
                self.models[key] = model
    
    def get_model_info(self) -> pd.DataFrame:
        """Get information about all models"""
        info = []
        for key, model in self.models.items():
            info.append({
                'key': key,
                'name': model.config.name,
                'algorithm': model.config.algorithm,
                'type': model.config.model_type,
                'trained': model.is_trained,
                'active': key == self.active_model,
                'performance': model.config.performance_metrics
            })
        
        return pd.DataFrame(info)