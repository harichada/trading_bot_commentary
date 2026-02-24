# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Start

```bash
# Main trading bot (port 8000, Schwab API)
python trading_bot_commentary_updated.py

# Claude rules-engine backtest dashboard (port 8001)
python claude_backtest_app.py

# Gap fade bot (port 8002 default, env GAP_FADE_PORT)
python gap_fade_app.py

# Professional bot (wraps main bot with safe_imports to block xgboost/lightgbm)
python run_professional_bot.py
```

### Dependencies

```bash
pip install -r requirements.txt          # Full stack (Schwab, ML, NLP, etc.)
pip install -r requirements-gap-fade.txt # Minimal: gap fade + auth only
```

### Testing

```bash
python -m pytest test_institutional_core.py -v   # Most comprehensive test suite
python -m pytest test_professional_bot.py -v      # Professional bot integration
python -m pytest test_components_isolated.py -v   # Isolated component tests
python -m pytest <test_file>.py -v                # Any individual test file
```

Tests use `pytest` + `pytest-asyncio`. No shared conftest — each `test_*.py` file is self-contained with its own imports and fixtures.

## Architecture

### Three Independent FastAPI Apps

Each app is a standalone monolith (single `.py` file with embedded HTML dashboard). They do not share a web server but share some Python modules:

| App | File | Port | API |
|-----|------|------|-----|
| Main Bot | `trading_bot_commentary_updated.py` (~15K lines) | 8000 | Schwab |
| Backtest Dashboard | `claude_backtest_app.py` (~10K lines) | 8001 | Schwab (optional) |
| Gap Fade | `gap_fade_app.py` (~7.5K lines) | 8002 (env `GAP_FADE_PORT`) | Alpaca |

**Import flow**: `claude_backtest_app.py` imports from `trading_bot_commentary_updated.py` (TradingBrain) and `claude_strategy.py`. `run_professional_bot.py` wraps the main bot. `gap_fade_app.py` is fully standalone.

### Shared Module Graph

- **`strategy_system.py`** — `StrategyManager`, `StrategyConfig`, `StrategySignal` — used by main bot, backtest dashboard, claude_strategy, professional engine, backtesting_engine
- **`claude_strategy.py`** — `ClaudeStrategy` with rules-engine signals — used by backtest dashboard
- **`llm_predictor.py`** — `DataPrep`, `LLMPromptBuilder`, `Predictor`, `BacktestTracker` — used by backtest dashboard
- **`institutional_core.py`** — `TradingStateMachine`, `AtomicStateManager` — used by institutional_trading_system
- **`risk_management.py`** / **`ml_model_manager_safe.py`** / **`paper_trading.py`** / **`advanced_orders.py`** — used by run_professional_bot and test suites

### Gap Fade App Internals

- SQLite price DB: `gap_fade_prices.db` with `daily_bars` table (`WITHOUT ROWID`, PK `(symbol, date)`, index on `(date, symbol)`)
- Data sources: 2006-2015 Yahoo (normalized), 2016+ Alpaca
- Config: `GapFadeConfig` dataclass with 4 feature groups: adaptive stops, market regime filter, re-entry after stop-out, gap-down fading (longs)
- Direction-aware: `direction='short'` for gap-ups, `direction='long'` for gap-downs
- `GapCandidate`, `GapPosition`, `TradeRecord`, `StopOutRecord` — all have `direction` field
- `scan_gaps_sql()`: >2000 symbols uses single-pass scan, <2000 uses chunked IN-clause
- Binding: localhost only by default; set `GAP_FADE_BIND_ALL=1` for all interfaces

### Authentication (Gap Fade frontend)

`auth.py` — OAuth2 (Google/GitHub/Discord) with JWT session cookies via `authlib`. Graceful degradation: fully disabled when `AUTH_JWT_SECRET` env var is unset. Frontend is React/TypeScript in `frontend/src/` (no package.json checked in — built separately).

### safe_imports.py

Monkey-patches `builtins.__import__` to block xgboost/lightgbm (segfault prevention). Returns dummy modules with stub `.fit()`/`.predict()` methods. Must be imported **first** (before any ML imports) — see `run_professional_bot.py` line 8.

### State Files

- `trading_state.json` — positions, orders, P&L
- `trading_brain.json` — learned patterns, emotional states
- `gap_fade_state.json` — gap fade equity, positions, trade history
- `paper_trade_state.json` — live paper trading state
- `token_1.json` — Schwab OAuth2 token (gitignored)

### API Integrations

- **Alpaca**: env vars `ALPACA_API_KEY` + `ALPACA_SECRET_KEY`, paper URL `https://paper-api.alpaca.markets`
- **Schwab**: env vars `SCHWAB_API_KEY` + `SCHWAB_SECRET`, token file `token_1.json`
- **Anthropic**: env var `ANTHROPIC_API_KEY`, used for live_api backtest mode in claude_backtest_app

### Deployment

`gap-fade.service` — systemd unit for gap fade app. Reads `.env` via `EnvironmentFile`, has watchdog (120s), memory cap (2G). Install script: `install-service.sh`.

## Compaction Instructions

When auto-compacting, preserve:
- All modified file paths and line-level changes
- Current config parameter values being tuned
- Backtest results and metric comparisons
- Any bugs found and their fix status
- Database schema and row counts
