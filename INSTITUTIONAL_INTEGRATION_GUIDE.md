# Institutional Core Integration Guide

## Quick Start

### 1. Import the Institutional Core

```python
from institutional_core import (
    InstitutionalTradingCore,
    TradingState,
    integrate_with_trading_engine
)
```

### 2. Initialize the Core

```python
# Configuration
config = {
    'state_file': 'trading_state_institutional.json',
    'max_position_value': 50000,
    'max_portfolio_percent': 0.20,
}

# Create institutional core
institutional_core = InstitutionalTradingCore(config)

# Initialize (async)
await institutional_core.initialize()
```

### 3. Integrate with Existing Trading Engine

```python
from trading_bot_commentary_updated import TradingEngineWithCommentary

# Create your trading engine
trading_engine = TradingEngineWithCommentary()

# Integrate institutional features
integrate_with_trading_engine(trading_engine, institutional_core)
```

## Key Features

### State Machine

The system now operates as an explicit state machine:

```
INITIALIZING → CONNECTED → RECONCILING → TRADING → ...
```

Check current state:
```python
current_state = institutional_core.state_machine.state
```

Check if operation is allowed:
```python
if institutional_core.state_machine.is_operation_allowed('place_order'):
    # Execute order
```

### Position Reconciliation

Run reconciliation before trading:

```python
async def fetch_broker_positions():
    # Your broker API call here
    return broker_positions

report = await institutional_core.reconciler.reconcile(
    local_positions=trading_engine.positions,
    fetch_broker_positions=fetch_broker_positions
)

if report.discrepancies:
    logger.warning(f"Found {len(report.discrepancies)} position discrepancies")
```

### Order Idempotency

Orders are automatically deduplicated. The integration helper wraps your order execution:

```python
# This is automatic after calling integrate_with_trading_engine()
# Duplicate orders within the same minute are blocked
```

### Health Monitoring

Check system health:

```python
# Run all health checks
health_results = await institutional_core.health_monitor.run_all_checks()

# Check if healthy
if institutional_core.health_monitor.is_healthy():
    # Continue trading
else:
    # Handle degraded state
```

### Circuit Breakers

Circuit breakers protect against cascading failures:

```python
# Check API circuit breaker
if institutional_core.circuit_breakers['api'].can_execute():
    # Make API call
```

Use as decorator:

```python
@institutional_core.circuit_breakers['api']
async def fetch_market_data():
    # API call here
```

### Position Size Validation

Validate before placing orders:

```python
is_valid, message = institutional_core.position_validator.validate(
    symbol='AAPL',
    quantity=100,
    price=150.0,
    portfolio_value=100000
)

if not is_valid:
    logger.warning(f"Position validation failed: {message}")
```

### Feature Validation (ML)

Validate ML features before prediction:

```python
features = extract_features(market_data)
is_valid, issues, cleaned_features = institutional_core.feature_validator.validate(features)

if not is_valid:
    logger.warning(f"Feature issues: {issues}")
    features = cleaned_features  # Use cleaned version
```

## State Persistence

State is automatically saved atomically with corruption protection:

```python
# Save state periodically
await institutional_core.save_state({
    'positions': trading_engine.positions,
    'trade_history': trading_engine.trade_history
})
```

## Running Tests

```bash
# Run institutional core tests
python -m pytest test_institutional_core.py -v

# Run specific test class
python -m pytest test_institutional_core.py::TestTradingStateMachine -v
```

## Example: Full Integration

```python
import asyncio
from institutional_core import InstitutionalTradingCore, TradingState
from trading_bot_commentary_updated import TradingEngineWithCommentary

async def main():
    # Initialize institutional core
    config = {
        'state_file': 'trading_state_institutional.json',
        'max_position_value': 50000,
    }
    institutional_core = InstitutionalTradingCore(config)
    await institutional_core.initialize()

    # Create trading engine
    trading_engine = TradingEngineWithCommentary()

    # Run reconciliation
    async def fetch_positions():
        return await trading_engine.get_broker_positions()

    report = await institutional_core.reconciler.reconcile(
        trading_engine.positions,
        fetch_positions
    )

    # Transition to trading if no issues
    if not report.discrepancies:
        institutional_core.state_machine.transition_to(
            TradingState.TRADING,
            trigger="reconciliation_complete"
        )

    # Trading loop
    while institutional_core.state_machine.state == TradingState.TRADING:
        # Check health
        if not institutional_core.health_monitor.is_healthy():
            logger.warning("Health check failed")
            break

        # Your trading logic here
        await asyncio.sleep(1)

        # Periodic state save
        await institutional_core.save_state()

if __name__ == '__main__':
    asyncio.run(main())
```

## Migration Checklist

- [ ] Import `institutional_core` module
- [ ] Initialize `InstitutionalTradingCore` with config
- [ ] Call `integrate_with_trading_engine()`
- [ ] Add position reconciliation on startup
- [ ] Add periodic health checks in main loop
- [ ] Add periodic state saves
- [ ] Update error handling to use circuit breakers
- [ ] Add feature validation before ML predictions
- [ ] Test with paper trading first
