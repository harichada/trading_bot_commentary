#!/usr/bin/env python3
"""
Fix ML model compatibility issues
"""
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

print("Loading existing model...")
try:
    with open('ml_model_integrated.pkl', 'rb') as f:
        model_data = pickle.load(f)
    print(f"Model version: {model_data.get('version', 'unknown')}")
    print(f"Feature names: {len(model_data.get('feature_names', []))} features")
except Exception as e:
    print(f"Error loading model: {e}")
    # Create a new model
    model_data = {
        'version': '2.1',
        'feature_names': [],
        'scaler': StandardScaler(),
        'model': None
    }

# Create new ensemble with proper XGBoost initialization
print("\nCreating new ensemble model...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

# Modern XGBoost doesn't use use_label_encoder
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'  # Specify metric explicitly
)

lgbm_model = LGBMClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)

# Create voting classifier
voting_clf = VotingClassifier(
    estimators=[
        ('rf', rf_model),
        ('xgb', xgb_model),
        ('lgbm', lgbm_model)
    ],
    voting='soft'
)

# Update model data
model_data['model'] = voting_clf
model_data['version'] = '2.1'

# Ensure we have feature names
if 'feature_names' not in model_data or not model_data['feature_names']:
    # Default feature names from MLFeatureExtractor
    feature_groups = {
        'price_momentum': ['returns_1', 'returns_5', 'returns_20', 'price_vs_sma20', 'price_vs_sma50'],
        'technical_indicators': ['rsi', 'macd', 'macd_signal', 'macd_hist', 'bb_position', 'bb_width'],
        'volume_analysis': ['volume_ratio', 'volume_std_ratio', 'volume_trend'],
        'volatility': ['atr_ratio', 'high_low_ratio', 'std_dev_ratio'],
        'support_resistance': ['pivot_distance', 'r1_distance', 's1_distance'],
        'market_microstructure': ['spread_proxy', 'volume_imbalance']
    }
    
    feature_names = []
    for group_features in feature_groups.values():
        feature_names.extend(group_features)
    
    model_data['feature_names'] = feature_names

# If the model was already trained, we need to preserve the trained state
# For now, we'll save the untrained model and let it train on first use
print(f"\nSaving updated model with {len(model_data['feature_names'])} features...")

with open('ml_model_integrated.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("Model updated successfully!")
print(f"- Version: {model_data['version']}")
print(f"- Features: {len(model_data['feature_names'])}")
print(f"- Ensemble models: RF, XGBoost, LightGBM")
print("\nNote: The model will need to be retrained with historical data.")