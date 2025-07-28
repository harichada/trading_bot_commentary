# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Running the Application
```bash
python trading_bot_commentary_updated.py
```
The application starts a FastAPI server on port 8000 with WebSocket support. Access the web interface at http://localhost:8000.

### Installing Dependencies
The application auto-installs missing dependencies when run. Manual installation:
```bash
pip install numpy pandas scipy ta scikit-learn xgboost lightgbm shap fastapi uvicorn websockets requests textblob pyyaml rich schwab-py
```

### Configuration
Edit `Config().yaml` to configure:
- Schwab API credentials
- Trading parameters (risk limits, position sizes)
- ML model settings
- Commentary verbosity levels

## Architecture Overview

### Core Components

1. **TradingEngine** - Main orchestrator that coordinates all trading activities
   - Manages positions, orders, and risk
   - Integrates with Schwab API for live trading
   - Implements circuit breakers and error recovery

2. **TradingBrain & TradingMemory** - Learning system
   - Stores patterns, lessons, and emotional states in `trading_brain.json`
   - Adapts trading behavior based on past performance
   - Uses embeddings for pattern similarity matching

3. **CommentarySystem** - Real-time explanation generator
   - Provides detailed rationale for every trading decision
   - Generates educational commentary at different complexity levels
   - Stores history in `trading_commentary.json`

4. **ML Pipeline** - Ensemble prediction system
   - Uses VotingClassifier with RandomForest, XGBoost, and LightGBM
   - Features include technical indicators, market microstructure, and sentiment
   - Model saved in `ml_model_integrated.pkl`

5. **WebSocket Server** - Real-time communication
   - FastAPI endpoints for control and monitoring
   - WebSocket for streaming commentary and updates
   - Connection management for multiple clients

### Data Flow

1. Market data → Feature engineering → ML predictions
2. Predictions + Risk management → Trading signals
3. Signals → Order execution (with OCO/bracket orders)
4. All decisions → Commentary generation → WebSocket broadcast
5. Trade results → Brain learning → Memory storage

### State Management

The application maintains several JSON state files:
- `trading_state.json` - Active positions, orders, and P&L
- `trading_brain.json` - Learned patterns and adaptive parameters
- `trading_commentary.json` - Commentary history
- `token_1.json` - Schwab authentication

### Key Design Patterns

1. **Singleton Pattern** - Config, TradingEngine, and Brain instances
2. **Observer Pattern** - WebSocket connections for real-time updates
3. **Strategy Pattern** - Multiple trading strategies (momentum, mean reversion, etc.)
4. **Circuit Breaker Pattern** - Risk management and error recovery

### Error Handling

- Comprehensive try-except blocks with graceful degradation
- Circuit breakers for API failures and risk limits
- Automatic recovery mechanisms for transient errors
- Detailed logging to `trading_bot.log*` files

### Testing Considerations

While no formal test suite exists, when adding tests:
1. Mock Schwab API responses for unit tests
2. Use historical data for backtesting strategies
3. Test risk management limits and circuit breakers
4. Verify commentary generation for edge cases
5. Test WebSocket connection handling