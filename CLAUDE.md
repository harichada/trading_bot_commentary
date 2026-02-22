# CLAUDE.md

## Quick Start

```bash
# Main trading bot (port 8000, Schwab API)
python trading_bot_commentary_updated.py

# Claude rules-engine backtest dashboard (port 8001)
python claude_backtest_app.py

# Gap fade bot (port 8003, Alpaca API)
python gap_fade_app.py

# Professional bot (safe imports)
python run_professional_bot.py
```

## Key Applications

### Gap Fade Bot (`gap_fade_app.py`, ~6300 lines)
- **Port 8003**, API key auth (`GAP_FADE_API_KEY` env var)
- Scans for gap-up/gap-down stocks, fades them intraday
- SQLite price DB: `gap_fade_prices.db` (25M rows, 12K+ symbols, 2006-2026)
- Data sources: 2006-2015 Yahoo (normalized), 2016+ Alpaca
- Config: `GapFadeConfig` dataclass with 4 feature groups:
  - **Adaptive stops**: `adaptive_stops`, `stop_gap_fraction`, `stop_min/max_pct`
  - **Market regime filter**: `regime_filter`, `regime_spy_gap_limit/block_pct`
  - **Re-entry after stop-out**: `reentry_enabled`, cooldown/max/trigger params
  - **Gap-down fading (longs)**: `trade_gap_downs`, thresholds
- Direction-aware: `_direction_pnl()`, `_stop_hit()`, `_target_hit()` helpers
- Live trading via Alpaca API, backtesting via `GapFadeBacktester`
- `scan_gaps_sql()`: >2000 symbols uses single-pass (1s), <2000 uses chunked IN-clause

### Claude Backtest Dashboard (`claude_backtest_app.py`)
- **Port 8001**, rules-engine backtesting with embedded HTML dashboard
- `evaluate_rules_signal(params, arrays, i, pos) → int` — shared by optimizer, paper trader, dashboard
- Walk-forward validation, pattern scorecard, parameter optimization
- Paper trading with live Alpaca ticks

### Main Trading Bot (`trading_bot_commentary_updated.py`)
- **Port 8000**, Schwab API integration, ML ensemble predictions
- Commentary system, WebSocket broadcast, learning brain

## Architecture Essentials

### State Files
- `trading_state.json` — positions, orders, P&L
- `trading_brain.json` — learned patterns, emotional states
- `gap_fade_state.json` — gap fade equity, positions, trade history
- `gap_fade_prices.db` — SQLite with `daily_bars` table (symbol, date, OHLCV)
- `paper_trade_state.json` — live paper trading state
- `token_1.json` — Schwab OAuth2 token

### API Integrations
- **Alpaca**: env vars `ALPACA_API_KEY` + `ALPACA_SECRET_KEY`, paper URL `https://paper-api.alpaca.markets`, data URL `https://data.alpaca.markets`
- **Schwab**: env vars `SCHWAB_API_KEY` + `SCHWAB_SECRET`, token file `token_1.json`
- **Anthropic**: env var `ANTHROPIC_API_KEY`, used for live_api backtest mode

### Key Patterns
- Direction-aware trading: `direction='short'` for gap-ups, `direction='long'` for gap-downs
- `GapCandidate`, `GapPosition`, `TradeRecord`, `StopOutRecord` — all have `direction` field
- `WITHOUT ROWID` SQLite table with PK `(symbol, date)`, index on `(date, symbol)`
- `fetch_alpaca_assets()` gets full tradeable universe (~12K symbols), cached 24h
- `fetch_alpaca_bars_multi()` for batch historical data with pagination + retry

## Compaction Instructions
When auto-compacting, preserve:
- All modified file paths and line-level changes
- Current config parameter values being tuned
- Backtest results and metric comparisons
- Any bugs found and their fix status
- Database schema and row counts
