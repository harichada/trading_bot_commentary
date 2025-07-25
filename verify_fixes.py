#!/usr/bin/env python3
"""Verify that all ML integration fixes have been applied"""

import os

def check_file_for_pattern(filepath, pattern, should_exist=True):
    """Check if a pattern exists in a file"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            exists = pattern in content
            
        if should_exist and exists:
            return True, f"✅ Found: {pattern[:50]}..."
        elif not should_exist and not exists:
            return True, f"✅ Not found (as expected): {pattern[:50]}..."
        else:
            return False, f"❌ {'Expected but not found' if should_exist else 'Found but should not exist'}: {pattern[:50]}..."
    except Exception as e:
        return False, f"❌ Error reading file: {e}"

print("Verifying ML Integration Fixes...")
print("=" * 60)

# Check 1: LightGBM callbacks fix
print("\n1. Checking LightGBM callbacks fix...")
result, msg = check_file_for_pattern(
    "/home/nvidia/trading_bot_commentary/scalping_ml_model.py",
    "callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]"
)
print(f"   {msg}")

# Check 2: WebSocket commentary type handling
print("\n2. Checking WebSocket commentary type handling...")
result, msg = check_file_for_pattern(
    "/home/nvidia/trading_bot_commentary/trading_bot_commentary_updated.py",
    "if hasattr(commentary, 'to_dict'):"
)
print(f"   {msg}")

# Check 3: Removed await from retrain_model
print("\n3. Checking await removal...")
result, msg = check_file_for_pattern(
    "/home/nvidia/trading_bot_commentary/trading_bot_commentary_updated.py",
    "await self.ml_predictor.retrain_model",
    should_exist=False
)
print(f"   {msg}")

# Check 4: Commentary helper in integration
print("\n4. Checking commentary helpers...")
result, msg = check_file_for_pattern(
    "/home/nvidia/trading_bot_commentary/scalping_ml_integration.py",
    "def _add_commentary(self, comment_data: Dict):"
)
print(f"   {msg}")

print("\n" + "=" * 60)
print("IMPORTANT: If all checks pass but you still see errors:")
print("1. The errors are from a running instance with old code")
print("2. You need to restart the trading bot")
print("3. Kill the current process: pkill -f trading_bot_commentary_updated.py")
print("4. Then restart: python trading_bot_commentary_updated.py")
print("=" * 60)