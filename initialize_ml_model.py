#!/usr/bin/env python3
"""
Initialize the ML model for the trading bot
"""

import sys
import shutil
from pathlib import Path
from scalping_ml_model import ScalpingMLModel, create_scalping_ml_predictor

def main():
    print("ML Model Initialization")
    print("=" * 50)
    
    # Check for existing models
    scalping_model_path = Path("scalping_ml_model.pkl")
    integrated_model_path = Path("ml_model_integrated.pkl")
    
    print(f"\nChecking existing models:")
    print(f"  scalping_ml_model.pkl exists: {scalping_model_path.exists()}")
    print(f"  ml_model_integrated.pkl exists: {integrated_model_path.exists()}")
    
    if scalping_model_path.exists():
        print("\n✅ Scalping ML model already exists!")
        
        # Try to load it
        model = create_scalping_ml_predictor()
        if model.is_trained:
            print(f"   Model is trained")
            print(f"   Training samples in buffer: {len(model.online_buffer)}")
        else:
            print("   ⚠️ Model exists but is not trained")
    else:
        print("\n❌ Scalping ML model not found")
        
        # Create new empty model (don't copy integrated model as it may be incompatible)
        print("\nCreating new scalping ML model...")
        model = ScalpingMLModel()
        model.save_model()
        print("✅ Created new scalping_ml_model.pkl")
    
    # Check model status
    print("\n" + "=" * 50)
    print("Model Status:")
    model = create_scalping_ml_predictor()
    
    print(f"  Is trained: {model.is_trained}")
    print(f"  Training buffer size: {len(model.online_buffer)}")
    print(f"  Retrain threshold: {model.retrain_threshold} samples")
    
    if not model.is_trained:
        samples_needed = model.retrain_threshold - len(model.online_buffer)
        print(f"\n⚠️ Model needs {samples_needed} more samples before it can make predictions")
        print("  The bot will collect samples as it runs and auto-train when ready.")
    else:
        print("\n✅ Model is ready to make predictions!")
    
    print("\n" + "=" * 50)
    print("\nNOTE: ML predictions require:")
    print("  1. At least 100 training samples (collected during live trading)")
    print("  2. Model to be trained (happens automatically after 100 samples)")
    print("  3. Config().ML_PREDICTION_ENABLED = True")
    
if __name__ == "__main__":
    main()