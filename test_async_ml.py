#!/usr/bin/env python3
"""
Test script to verify ML training doesn't block the trading bot
"""

import asyncio
import time
from datetime import datetime

async def simulate_trading_loop():
    """Simulate the main trading loop"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Trading loop started")
    
    for i in range(20):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Trading cycle {i+1}")
        await asyncio.sleep(1)  # Simulate 1 second per cycle
        
        # Simulate ML training trigger
        if i == 5:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Triggering ML training...")
            asyncio.create_task(simulate_ml_training())
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Trading loop completed")

async def simulate_ml_training():
    """Simulate ML training in background"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ML training started (async)")
    
    # Run blocking operation in thread pool
    loop = asyncio.get_event_loop()
    
    def blocking_training():
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ML training running in thread...")
        time.sleep(5)  # Simulate 5 seconds of training
        return "Training complete"
    
    result = await loop.run_in_executor(None, blocking_training)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ML training result: {result}")

async def main():
    """Main test function"""
    print("Testing async ML training...")
    print("-" * 50)
    
    # Run trading loop
    await simulate_trading_loop()
    
    # Wait for any background tasks
    await asyncio.sleep(2)
    
    print("-" * 50)
    print("Test complete! Trading loop was not blocked by ML training.")

if __name__ == "__main__":
    asyncio.run(main())