# Trading Bot Refactoring Plan

## Phase 1: Extract Configuration (1 hour)
- [ ] Create `config/` directory
- [ ] Move Config class to `config/settings.py`
- [ ] Add environment variable loading for credentials

## Phase 2: Extract Core Components (2-3 hours)
- [ ] Create `trading/` package
- [ ] Extract TradingEngine → `trading/engine.py`
- [ ] Extract TradingBrain → `trading/brain.py`
- [ ] Extract CommentarySystem → `trading/commentary.py`

## Phase 3: Extract ML Components (2 hours)
- [ ] Create `ml/` package
- [ ] Extract ML models → `ml/models.py`
- [ ] Extract feature engineering → `ml/features.py`
- [ ] Extract predictions → `ml/predictor.py`

## Phase 4: Extract API/WebSocket (1 hour)
- [ ] Create `api/` package
- [ ] Extract WebSocket server → `api/websocket.py`
- [ ] Extract Schwab client → `api/schwab.py`

## Phase 5: Add Tests (2-3 hours)
- [ ] Create `tests/` directory
- [ ] Add unit tests for each module
- [ ] Add integration tests for trading flow
- [ ] Mock Schwab API for testing

## File Structure After Refactoring:
```
trading_bot/
├── config/
│   └── settings.py
├── trading/
│   ├── __init__.py
│   ├── engine.py
│   ├── brain.py
│   └── commentary.py
├── ml/
│   ├── __init__.py
│   ├── models.py
│   ├── features.py
│   └── predictor.py
├── api/
│   ├── __init__.py
│   ├── websocket.py
│   └── schwab.py
├── tests/
│   ├── test_engine.py
│   ├── test_ml.py
│   └── test_api.py
└── main.py
```