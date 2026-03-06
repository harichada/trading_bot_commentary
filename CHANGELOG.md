# Changelog

All notable changes to the Rudra Trading Engine.

## v12.1 — 2026-03-05
- **Migration**: Trader state persistence moved from `gap_fade_state.json` to PostgreSQL — zero disk writes

## v12.0 — 2026-03-05
- **Fix**: `evaluate_exit()` bug — duplicate price arg broke pullback exit management (caused -$261 TTD loss)
- **Fix**: Raise `min_avg_volume` 5K → 50K — filters illiquid stocks (e.g. PKX at 1.8K vol)
- **Fix**: Lower `max_consec_losses` 3 → 2 — prevents cascading losses
- **Fix**: Add `intraday_max_position_pct=0.20` — caps intraday positions at 20% of equity (was uncapped)
- **Migration**: Full SQLite → PostgreSQL (25M+ rows, all 13 tables, psycopg2 driver)
- **Migration**: `gap_fade_backtester.py`, `bt_backtest.py` ported to psycopg2
- **Migration**: `migrate_sqlite_to_pg.py` one-time COPY-based migration script

## v11.0 — 2026-02-28
- **Rebrand**: "Gap Fade" → "Rudra Trading Engine" across all user-facing strings
- Internal code identifiers, file names, env vars, and strategy names unchanged

## v10.5 — 2026-02-27
- **Fix**: Sharpe ratio — use daily equity returns instead of per-trade returns
- **Fix**: Max drawdown — use equity curve peak-to-trough instead of cumsum
- **Fix**: Re-entry stop — actually check `_stop_hit()` on re-entry trades
- **Fix**: Re-entry borrow cost — deduct before `pnl_pct` calculation
- **Fix**: Regime-skipped days missing from equity curve
- **Fix**: Position lock contention — release lock during LLM calls (5-30s)
- **Fix**: XSS — apply `escHtml()` to all innerHTML with user/LLM content
- **Fix**: `closePosition`/`promptStopAdjust` — use `api()` helper with auth headers
- **Fix**: Chat/review config actions — use `validate_config()` with range checks
- **Fix**: Chat pause/resume/reset — use proper methods with broadcasts
- **Fix**: Rudra JSON parse — treat text-only LLM responses as wait/no-action

## v10.4
- Rudra can now switch strategies via chat (e.g. "switch to classic gap fade")
- Unrealized P&L shown in dashboard metrics bar alongside Total P&L
- Rudra messages no longer truncated in activity feed — full responses visible

## v10.2
- Fix backtest feature toggles not sending `false` when unchecked
- Adaptive stops, regime filter, re-entry, gap-downs now explicitly disabled when unchecked
- Previously, unchecked toggles silently used config defaults (e.g. `adaptive_stops=True`)

## v10.1
- Rudra LLM supervisor now aware of drawdown circuit breakers
- Live DD% and tier status shown in Rudra chat and review context
- Rudra can proactively recommend de-risking as DD approaches thresholds
- DD config fields (thresholds, scales) settable via Rudra chat commands
- `peak_equity` and DD config added to LLM state dict

## v10.0
- Drawdown circuit breakers — graduated Tier 1/Tier 2/Hard Stop response
- Position size reduction during drawdowns (backtest + live)
- Circuit breaker stats in backtest results (days in tier, trades skipped)
- New UI toggle and config fields for DD thresholds and scales
- DD metric aligned: circuit breaker and dashboard Max DD use same formula

## v9.0
- Vectorbt-powered backtesting module with bulk SQLite data loading
- Vectorized gap scanning with pandas — 10x faster than row-by-row
- SimulationState engine with Kelly sizing, adaptive stops, regime filter
- Strategy plugin support in backtester (classic, VWAP, confluence, Minervini)
- MetricsAdapter for dashboard-compatible result format
- Full test suite (43 tests)

## v8.0
- Multi-provider LLM support (Ollama, OpenAI-compatible: GPT, Groq, Together)
- Remove LLM budget rate-limiting for faster autonomous decisions
- Provider-specific API key and endpoint configuration

## v7.0
- Upgrade Rudra LLM to `gpt-oss:20b` with full persona
- Gap-down fading strategy (long entries on gap-downs)
- Direction-aware position sizing, stops, and targets

## v6.0
- Mobile OAuth flow with dynamic redirect URL detection
- Responsive auth UI for mobile devices

## v5.0
- Fix LLM supervisor not initializing from saved config
- Status badge now updates for all LLM actions (wait, monitor)
- Trading loop resilience — single errors no longer kill the loop
- Double-start guard prevents duplicate trading loops
- Relaxed standdown rules — consecutive losses alone don't halt trading
- Version number displayed in UI from git tags

## v4.0
- Real-time position tracker with candlestick charts
- Watchlist with live price streaming
- Per-position controls and autonomous LLM review loop
- LLM learning from trades: metadata tracking, dynamic lessons
- Full zero-human-intervention automation

## v3.0
- LLM chat interface for interactive trading conversations
- Ask Rudra questions, trigger actions via natural language

## v2.0
- LLM supervisor (Rudra) for autonomous trading decisions
- Scan, enter, monitor, standdown — all LLM-driven
- Circuit breaker with fallback to rules-based schedule

## v1.0
- Gap fade strategy dashboard with embedded UI
- OAuth authentication (Google/GitHub/Discord)
- Alpaca paper trading integration
- SQLite price database with historical backtesting
- Catalyst detection (earnings, FDA, M&A filtering)
