#!/usr/bin/env python3
"""
Example showing how the error handling improvements work
"""

from trading_exceptions import *
from circuit_breaker import CircuitBreaker
from error_recovery import ErrorRecoveryManager
import time

# Example 1: Circuit Breaker protecting API calls
@CircuitBreaker(failure_threshold=3, recovery_timeout=30)
def risky_api_call():
    """Simulated API call that might fail"""
    import random
    if random.random() < 0.7:  # 70% chance of failure
        raise NetworkException("Connection timeout")
    return "Success!"

# Example 2: Error recovery in action
def demo_error_recovery():
    recovery = ErrorRecoveryManager()
    
    # Simulate order failure
    try:
        raise InsufficientFundsException(required=1000, available=500)
    except InsufficientFundsException as e:
        # Record the failure
        recovery.record_order_failure(
            order_id="12345",
            symbol="NVDA", 
            error=e,
            order_details={"quantity": 10, "price": 100}
        )
        
        # Get recovery suggestion
        action = recovery.suggest_recovery_action(e)
        print(f"Suggested action: {action}")
        
    # Check for pending retries
    pending = recovery.get_pending_retries()
    print(f"Pending retries: {len(pending)}")
    
    # Get error summary
    summary = recovery.get_error_summary()
    print(f"Error summary: {summary}")

# Example 3: Circuit breaker demo
def demo_circuit_breaker():
    print("\n=== Circuit Breaker Demo ===")
    for i in range(10):
        try:
            result = risky_api_call()
            print(f"Attempt {i+1}: {result}")
        except CircuitBreakerException as e:
            print(f"Attempt {i+1}: Circuit breaker OPEN - {e}")
            break
        except NetworkException as e:
            print(f"Attempt {i+1}: Failed - {e}")
        time.sleep(0.5)

if __name__ == "__main__":
    print("=== Error Recovery Demo ===")
    demo_error_recovery()
    
    print("\n" + "="*50 + "\n")
    
    demo_circuit_breaker()