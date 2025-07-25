#!/usr/bin/env python3
"""
Update the ML model to current XGBoost format
"""
import pickle
import xgboost as xgb
from pathlib import Path

# Load the old model
with open('ml_model_integrated.pkl', 'rb') as f:
    model_data = pickle.load(f)

# Extract the voting classifier
voting_clf = model_data['model']

# Update XGBoost model if it exists in the ensemble
for name, estimator in voting_clf.estimators:
    if isinstance(estimator, xgb.XGBClassifier):
        print(f"Found XGBoost model: {name}")
        # Save and reload to update format
        estimator.save_model('temp_xgb_model.json')
        estimator.load_model('temp_xgb_model.json')
        print("Updated XGBoost model format")

# Save the updated model
with open('ml_model_integrated.pkl', 'wb') as f:
    pickle.dump(model_data, f)

# Clean up
Path('temp_xgb_model.json').unlink(missing_ok=True)

print("Model updated successfully!")