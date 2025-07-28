#!/usr/bin/env python3
"""
Minimal test to identify segmentation fault source
"""

import sys
print("Python version:", sys.version)

print("\n1. Testing basic imports...")
try:
    import numpy as np
    print("✓ NumPy imported")
except Exception as e:
    print(f"✗ NumPy failed: {e}")

try:
    import pandas as pd
    print("✓ Pandas imported")
except Exception as e:
    print(f"✗ Pandas failed: {e}")

try:
    from sklearn.ensemble import RandomForestClassifier
    print("✓ Scikit-learn imported")
except Exception as e:
    print(f"✗ Scikit-learn failed: {e}")

print("\n2. Testing FastAPI imports...")
try:
    from fastapi import FastAPI
    import uvicorn
    print("✓ FastAPI/Uvicorn imported")
except Exception as e:
    print(f"✗ FastAPI/Uvicorn failed: {e}")

print("\n3. Testing original trading bot imports...")
try:
    # Import just the essential parts
    import asyncio
    import json
    from pathlib import Path
    from datetime import datetime
    print("✓ Basic Python imports successful")
except Exception as e:
    print(f"✗ Basic imports failed: {e}")

print("\n4. Testing trading bot components...")
try:
    from trading_bot_commentary_updated import Config
    print("✓ Config imported")
except Exception as e:
    print(f"✗ Config import failed: {e}")
    import traceback
    traceback.print_exc()

print("\n5. Creating minimal FastAPI app...")
try:
    app = FastAPI()
    
    @app.get("/test")
    async def test():
        return {"status": "ok"}
    
    print("✓ FastAPI app created")
except Exception as e:
    print(f"✗ FastAPI app creation failed: {e}")

print("\n" + "="*50)
print("If you see this message, the basic components work!")
print("Now let's try to import the full trading bot...")
print("="*50)

try:
    print("\n6. Importing full trading bot...")
    from trading_bot_commentary_updated import TradingEngineWithCommentary
    print("✓ TradingEngineWithCommentary imported")
except Exception as e:
    print(f"✗ TradingEngineWithCommentary import failed: {e}")
    import traceback
    traceback.print_exc()

print("\nTest complete!")