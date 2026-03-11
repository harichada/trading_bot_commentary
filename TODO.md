# TODO - Deferred Items

## Consolidation (from P3 audit)
- [ ] Consolidate `backtesting_engine.py` (root) and `risk/backtest.py` into a single backtest system
- [ ] Merge `performance_analytics.py` (root) with `risk/backtest.py` PerformanceAnalyzer
- [ ] Unify `strategy_system.py` (professional) with `strategies/` package
- [ ] Merge `risk_management.py` (professional) with `risk/manager.py`
- [ ] Deprecate `ml_model_manager.py` in favor of `ml_model_manager_safe.py`
- [ ] Unify `infrastructure.py` ConfigurationManager with `core/config.py`

## Placeholder Implementations (from P4 audit)
- [ ] `news_sentiment_widget.py:917` - Integrate Reddit API (PRAW) for real sentiment data
- [ ] `news_sentiment_widget.py:936` - Integrate StockTwits API for real sentiment data
- [ ] `backtesting_engine.py:570` - Calculate actual beta vs market benchmark
- [ ] `AlternativeDataIntegrator` - All methods return simulated random data
- [ ] `institutional_integration.py:456` - Implement correlation calculation

## Security
- [ ] Rotate Schwab API credentials (exposed in git history)
- [ ] Add API authentication to FastAPI endpoints
- [ ] Add WebSocket authentication
- [ ] Encrypt token_1.json on disk
- [ ] Remove `os.system("pip install ...")` calls in setup_wizard.py and backtesting_engine.py

## Architecture
- [ ] Replace JSON state files with SQLite/PostgreSQL
- [ ] Add proper database migrations
- [ ] Implement proper async/await throughout (currently mixing sync/async)
- [ ] Add connection pooling for Schwab API
- [ ] Containerize with Docker

## Testing
- [ ] Convert manual test scripts to pytest
- [ ] Add CI/CD pipeline
- [ ] Add load testing for WebSocket connections
