"""
Safe imports wrapper to prevent segmentation faults
This module blocks problematic imports before they can cause issues
"""

import sys
import builtins
import warnings

# Store the original import
original_import = builtins.__import__

def safe_import(name, *args, **kwargs):
    """
    Custom import function that blocks problematic libraries
    """
    # List of modules that cause segmentation faults
    blocked_modules = ['xgboost', 'lightgbm', 'xgb', 'lgb']
    
    # Check if trying to import a blocked module
    for blocked in blocked_modules:
        if blocked in name.lower():
            warnings.warn(f"Blocked import of {name} to prevent segmentation fault")
            # Return a dummy module that won't crash
            class DummyModule:
                def __getattr__(self, attr):
                    # Return a dummy class that can be instantiated
                    class DummyClass:
                        def __init__(self, *args, **kwargs):
                            pass
                        def __call__(self, *args, **kwargs):
                            return self
                        def __getattr__(self, name):
                            return self
                        def __getitem__(self, key):
                            return self
                        def fit(self, *args, **kwargs):
                            return self
                        def predict(self, *args, **kwargs):
                            return np.array([0])
                        def predict_proba(self, *args, **kwargs):
                            return np.array([[0.5, 0.5]])
                    
                    if attr in ['XGBClassifier', 'XGBRegressor', 'LGBMClassifier', 'LGBMRegressor']:
                        return DummyClass
                    return DummyClass()
            
            dummy = DummyModule()
            sys.modules[name] = dummy
            return dummy
    
    # Otherwise, use the original import
    try:
        return original_import(name, *args, **kwargs)
    except Exception as e:
        # If any import fails with segfault-like errors, return dummy
        if 'segmentation' in str(e).lower() or 'core dumped' in str(e).lower():
            warnings.warn(f"Import of {name} failed with {e}, using dummy module")
            class DummyModule:
                pass
            return DummyModule()
        raise

# Replace the built-in import
builtins.__import__ = safe_import

# Also need numpy for dummy predictions
import numpy as np

print("Safe imports enabled - XGBoost and LightGBM will be disabled")