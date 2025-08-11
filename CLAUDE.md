# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Running the Application

#### Main Trading Bot
```bash
python trading_bot_commentary_updated.py
```
The application starts a FastAPI server on port 8000 with WebSocket support. Access the web interface at http://localhost:8000.

#### Professional Bot with Enhanced Features
```bash
# With safe imports to prevent segmentation faults
python run_professional_bot.py

# Or using the startup script
./start_professional_bot.sh
```

#### Testing Components
```bash
# Test professional features
python test_professional_bot.py

# Test individual components
python test_components_isolated.py
python test_ml_predictor_fix.py
python test_websocket_fix.py
```

### Installing Dependencies
The application auto-installs missing dependencies when run. Manual installation:
```bash
# Core dependencies
pip install numpy pandas scipy ta scikit-learn xgboost lightgbm shap fastapi uvicorn websockets requests textblob pyyaml rich schwab-py

# Additional for professional features
pip install torch feedparser yfinance beautifulsoup4 aiohttp nltk
```

### Configuration
Edit `Config().yaml` to configure:
- Schwab API credentials (token stored in `token_1.json`)
- Trading parameters (risk limits, position sizes)
- ML model settings
- Commentary verbosity levels
- Professional features (strategies, advanced orders)

## Architecture Overview

### Core Components

1. **TradingEngineWithCommentary** (`trading_bot_commentary_updated.py`) - Main orchestrator
   - Manages positions, orders, and risk
   - Integrates with Schwab API for live trading
   - Implements circuit breakers and error recovery
   - Coordinates all subsystems

2. **TradingBrain & TradingMemory** - Learning system
   - Stores patterns, lessons, and emotional states in `trading_brain.json`
   - Adapts trading behavior based on past performance
   - Uses embeddings for pattern similarity matching
   - Tracks confidence levels and adjusts strategies

3. **CommentarySystem** - Real-time explanation generator
   - Provides detailed rationale for every trading decision
   - Generates educational commentary at different complexity levels
   - Stores history in `trading_commentary.json`
   - Broadcasts via WebSocket to all connected clients

4. **ML Pipeline** - Ensemble prediction system
   - VotingClassifier with RandomForest, XGBoost, and LightGBM
   - Features include 200+ technical indicators, market microstructure, and sentiment
   - Model saved in `ml_model_integrated.pkl`
   - Supports online learning and retraining

5. **WebSocket Server** - Real-time communication
   - FastAPI endpoints for control and monitoring
   - WebSocket for streaming commentary and updates
   - Connection management for multiple clients
   - Handles reconnection and error recovery

### Professional Components

6. **StrategyManager** (`strategy_system.py`) - Modular strategy system
   - Base strategy interface for custom strategies
   - Built-in strategies: MA Cross, RSI Momentum, Bollinger Bands, etc.
   - Consensus voting across multiple strategies
   - Dynamic strategy switching based on market conditions

7. **ModelManager** (`ml_model_manager_safe.py`) - ML model orchestration
   - Support for multiple algorithms (RF, XGBoost, LightGBM, Neural Networks)
   - Feature engineering pipeline with 200+ indicators
   - Model versioning and performance tracking
   - Safe imports to prevent segmentation faults

8. **OrderManager** (`advanced_orders.py`) - Advanced order types
   - Bracket orders (entry + stop loss + take profit)
   - Trailing stops with dynamic adjustment
   - Dollar-cost averaging (DCA)
   - TWAP (Time-weighted average price)
   - Iceberg orders for large positions

9. **BacktestingEngine** (`backtesting_engine.py`) - Historical simulation
   - Accurate order execution simulation
   - Comprehensive performance metrics
   - Walk-forward analysis
   - Parameter optimization

10. **RiskManager** (`risk_management.py`) - Portfolio protection
    - Position sizing methods (Kelly, Fixed Fractional, Volatility-based)
    - Risk limits (max drawdown, exposure, correlation)
    - Dynamic stop loss and take profit
    - Portfolio heat monitoring

### Data Flow

1. Market data → Feature engineering (200+ indicators) → ML predictions
2. Predictions + Risk management + Strategy signals → Trading signals
3. Signals → Order execution (with OCO/bracket orders via Schwab API)
4. All decisions → Commentary generation → WebSocket broadcast
5. Trade results → Brain learning → Memory storage → Strategy adaptation

### State Management

The application maintains several JSON state files:
- `trading_state.json` - Active positions, orders, P&L, and portfolio metrics
- `trading_brain.json` - Learned patterns, adaptive parameters, emotional states
- `trading_commentary.json` - Commentary history with timestamps and importance levels
- `token_1.json` - Schwab OAuth2 authentication token
- `professional_config.json` - Professional feature configuration

### Key Design Patterns

1. **Singleton Pattern** - Config, TradingEngine, Brain, and CommentarySystem instances
2. **Observer Pattern** - WebSocket connections for real-time updates
3. **Strategy Pattern** - Multiple trading strategies with common interface
4. **Circuit Breaker Pattern** - Risk management and API failure recovery
5. **Factory Pattern** - Order creation with different types
6. **Decorator Pattern** - Error recovery and retry logic

### Error Handling

- Custom exception hierarchy in `trading_exceptions.py`
- Circuit breakers (`circuit_breaker.py`) for API failures and risk limits
- Error recovery manager (`error_recovery.py`) with exponential backoff
- Comprehensive try-except blocks with graceful degradation
- Detailed logging to `trading_bot.log*` files with rotation

### Performance Optimizations

- Safe imports (`safe_imports.py`) to prevent segmentation faults
- Efficient numpy operations with proper array handling
- Batch processing for feature engineering
- Connection pooling for API requests
- Caching for frequently accessed data
- Async/await for non-blocking operations

### Testing Considerations

While no formal test suite exists, when adding tests:
1. Mock Schwab API responses for unit tests
2. Use historical data for backtesting strategies  
3. Test risk management limits and circuit breakers
4. Verify commentary generation for edge cases
5. Test WebSocket connection handling and reconnection
6. Validate ML model predictions with known scenarios
7. Test order execution logic with various market conditions

### Common Development Tasks

#### Adding a New Strategy
1. Create a new class inheriting from `BaseStrategy` in `strategy_system.py`
2. Implement `generate_signal()` method
3. Register in `StrategyManager`
4. Add configuration to `Config().yaml`

#### Adding a New ML Model
1. Implement model class in `ml_model_manager_safe.py`
2. Add to `ModelManager` registry
3. Update feature engineering if needed
4. Test with backtesting engine

#### Debugging Tips
- Check `trading_bot.log` for detailed execution trace
- Monitor `trading_state.json` for current positions
- Review `trading_brain.json` for learning patterns
- Use `test_components_isolated.py` to test individual components
- Enable verbose commentary in `Config().yaml` for detailed explanations