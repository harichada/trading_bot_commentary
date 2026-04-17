# Trading Bot — Operator's Reference

Single-source answer to the onboarding questions. Generated 2026-04-16.

---

## 1. How the bot runs today

- **Architecture:** single Python process — FastAPI + asyncio event loop + background trading loop. One PID, one port (`9000`), no workers, no Celery.
- **Start:** `python trading_bot_commentary_updated.py` from the project dir. This boots uvicorn on `0.0.0.0:9000` but does **not** begin trading — trading starts when something hits `POST /api/start`.
  - Current launch pattern: `nohup python trading_bot_commentary_updated.py > trading_bot.log 2>&1 &`
  - No systemd unit for this bot yet. (There is a `gap-fade.service` for a sibling project, but not for this one.)
- **Stop:** `POST /api/stop` sets `trading_engine.is_running = False` → the loop exits on its next tick but the server keeps running. To fully halt: `pkill -TERM -f "python trading_bot_commentary_updated"`. `SIGKILL` is only needed if the process doesn't respond to SIGTERM.
- **Logs:** all three, but the canonical one is the file.
  - `trading_bot.log` — rotating file, configured in `core/config.py:429` (`RotatingFileHandler`). Structured JSON lines for all engine events (`engine_decision component=… action=… reason=… key=value …`).
  - stdout — same content, duplicated via `StreamHandler` at `core/config.py:438`.
  - Postgres — **not written to by the live bot.** Postgres is read-only from the bot's perspective (ML training + backtests). Trade/position state persists to JSON files (§3).

---

## 2. Postgres schema

### Connection
- DB: `rudra_dev`
- Host: `localhost:5432`
- User: `rudra`, password hardcoded in `train_ml_model.py:42` (flagged in security review; env override `POSTGRES_DSN` takes precedence)
- The live bot **doesn't read Postgres** — only training/analysis scripts do (see §4)

### All tables in `rudra_dev` (23 total, plus a parallel `mt.*` schema with the same names, all empty)

| Table | Rows | Bot uses it? |
|---|---|---|
| `minute_bars` | **228,642,032** | **Yes** — ML training, backtests, shadow outcome tracker |
| `daily_bars` | **25,439,850** | **Yes** — daily-bar feature backfills |
| `trade_ticks` | 1,393,888,792 | No — unused by this bot |
| `trades` | 327 | No — sibling project's trade log |
| `journal_entries` | 1,570 | No |
| `signals_intraday` | 461 | No |
| Empty stubs | 0 each | `aegis_bars`, `api_calls`, `candidates_rejected`, `config_history`, `config_profiles`, `earnings_calendar`, `fundamental_cache`, `llm_calls`, `market_events`, `news_alerts`, `news_settings`, `performance_snapshots`, `subscriptions`, `trader_state`, `user_configs`, `user_credentials`, `users` |

### Schemas of the tables that matter

**`minute_bars`** — 1-min OHLCV, the workhorse table
```
Column  Type                       Notes
symbol  text                       NOT NULL
ts      timestamp (without tz)     NOT NULL; America/New_York naive
open    double
high    double
low     double
close   double
volume  bigint                     default 0
```
Indexes:
- `minute_bars_pkey` PRIMARY KEY (symbol, ts)
- `idx_minute_bars_sym_ts` (symbol, ts) — redundant with PK
- `idx_minute_bars_ts` (ts) — for cross-symbol time queries

**`daily_bars`**
```
Column  Type              Notes
symbol  text              NOT NULL
date    text              NOT NULL (stored as string, not a date type — legacy)
open    double
high    double
low     double
close   double
volume  double
```
Indexes: PK `(symbol, date)`, `idx_bars_date_symbol (date, symbol)`

**`trades`** — sibling project's schema, not written to by this bot. Kept for reference in case you point a messenger query at it later.
```
trade_id      serial PK
symbol        text
entry_price   double
exit_price    double
shares        integer
pnl           double
pnl_pct       double
entry_time    text                (ISO string)
exit_time     text                (ISO string)
exit_reason   text
holding_minutes  integer
side          text                default 'short'
gap_pct       double
vol_ratio     double
score         double
catalyst      text
strategy_id   text
setup_type    text
user_id       uuid                FK → users
```
Unique: `(symbol, entry_time, exit_time, shares, exit_reason)`

### Where this bot's own state actually lives

**Not in Postgres.** On disk as JSON:
| File | Contents |
|---|---|
| `trading_state.json` | Open positions, pending orders, realized P&L, simulated positions, bot config |
| `trading_brain.json` | Learned patterns, behavioral memories |
| `trading_commentary.json` | Rolling commentary feed shown on the dashboard |
| `paper_trade_state.json` | Paper-trading state (separate from sim) |
| `token_1.json` | Schwab OAuth token (gitignored) |

So if a messenger wants current positions, it should **not** query Postgres — see §3.

---

## 3. How to ask the bot "what's open right now?"

Four ways, in order of goodness:

1. **HTTP endpoint** — `GET http://localhost:9000/api/status` returns `{bot_running, mode, positions_count, schwab_connected, uptime}`. See `api/routes.py:526`.
2. **Richer HTTP** — `GET /api/account-stats` returns `{balance, buying_power, daily_pnl, total_pnl, position_count, cash}` and pulls live from Schwab when `mode=LIVE`. See `api/routes.py:538`.
3. **State file** — read `trading_state.json` directly. Gives the full position detail the bot is tracking. Updated every trading loop iteration (every 60s).
4. **WebSocket** — `ws://localhost:9000/ws` streams commentary + position updates to the dashboard in real time. The React/HTML dashboard at `http://localhost:9000` uses this.

There's also the structured audit log (`trading_bot.log`) — every decision is logged with a grep-friendly prefix like `engine_decision component=signal_router …`. That's the audit trail, not a live state source.

### All HTTP endpoints

Quick-reference list:

```
Bot control:
  POST /api/start             start trading with {"mode": "simulation"|"live"}
  POST /api/stop              pause loop (server keeps running)
  POST /api/toggle-mode       flip sim ↔ live
  POST /api/toggle-confirmations
  GET  /api/status            minimal status
  GET  /api/account-stats     P&L, buying power, positions count
  POST /api/refresh-positions sync with Schwab

Settings:
  GET  /api/settings          all config
  GET  /api/settings/{cat}
  POST /api/settings/reset

Professional mode (ML/models):
  GET  /api/professional/models
  POST /api/professional/models/{key}/select
  GET  /api/professional/risk
  GET  /api/professional/backtest/report/latest
  POST /api/professional/backtest
  GET  /api/professional/strategies
  GET  /api/professional/performance
  GET  /api/professional/paper/account
  GET  /api/professional/paper/positions
  GET  /api/professional/config
  POST /api/professional/risk/settings
  POST /api/professional/strategies/{key}/toggle

Analytics (stub endpoints — return empty success envelopes):
  GET  /api/analytics/{summary|trades|decisions|performance|daily-stats}
  POST /api/analytics/export/{category}

Sentiment:
  GET  /api/sentiment/{update|market|headlines|alerts|earnings}
  GET  /api/sentiment/symbol/{symbol}
  GET  /api/sentiment/correlation/{symbol}
  GET  /api/sentiment/social/{symbol}
  POST /api/sentiment/alerts/{id}/acknowledge

Backtest:
  POST /api/backtest/run
  GET  /api/backtest/status
```

---

## 4. Main loop (one-pass flow)

Lives in `core/engine.py` inside `TradingEngineWithCommentary._run_commentary_cycle`, started by `POST /api/start`. Runs forever until `is_running=False`.

Every tick (60-second sleep at the end):

```
while self.is_running:
    1. is_market_hours() + _in_warmup_window() → decide if tradable
       - not tradable → _update_position_prices() only (keeps dashboard alive),
                        then sleep 60s and continue
    2. analysis_count += 1
    3. Every 5 iterations → sync with Schwab (account info + positions)
    4. _update_position_prices()           update quotes for all open positions
    5. _assess_market_conditions()         VIX, breadth, commentary
    6. _analyze_premarket_gaps()           gap scan
    7. _evaluate_trading_conditions()      should_trade? (risk, DD, hours)
    8. if should_trade: _analyze_markets_with_commentary()
       — iterates self.dynamic_watchlist (default 10 symbols),
         fetches OHLCV from Schwab, runs each strategy,
         fires _process_signal_with_commentary(signal, ml_signal, ...)
         which in turn calls _shadow_meta_evaluate(signal) in a thread,
         then gates/sizes/submits the order.
    9. _manage_positions_with_commentary()  SL/TP/trailing exits
    10. Emergency stop check (Schwab daily P&L guard)
    11. asyncio.sleep(60)
```

**Hot-path latency per tick:** mostly 60s sleep. Actual work = quote fetches (async via httpx), strategy evaluation (in-process, milliseconds), ML predict (<1ms), Schwab order submission (100–500ms per order).

**Safe hook points for a messenger or observer:**
- After `_manage_positions_with_commentary()` (line ~2086) — you have fresh state, no order being placed.
- Read-only via HTTP/WS — zero interference.
- Tail `trading_bot.log` — structured JSON, grep for `action=received`, `action=filled`, `component=meta_shadow`.

---

## 5. Existing control surface

Three ways to steer the bot at runtime:

| Mechanism | What it controls | How |
|---|---|---|
| HTTP endpoints | Start/stop, mode toggle, confirmations, settings, strategy enable/disable, backtest triggers | `curl` or the React dashboard |
| Env vars | `ENABLE_MOMENTUM=1` re-enables momentum strategy; `POSTGRES_DSN` overrides DB; `ALPACA_API_KEY/SECRET` for data ingest; `SCHWAB_API_KEY/SECRET` for broker | Set before `python trading_bot_commentary_updated.py` |
| State files | Direct edits to `trading_state.json` — risky but possible (bot reloads on next tick) | Stop bot, edit, start bot |

**No signal files (`/tmp/halt`), no Postgres control rows, no CLI.** `POST /api/stop` is the graceful-halt primitive.

**In practice for a halt:**
1. Graceful pause: `curl -X POST http://localhost:9000/api/stop` → loop exits, server stays up, positions untouched
2. Hard stop: `pkill -TERM -f trading_bot_commentary_updated` → process exits, positions stay in Schwab, state file persists
3. Full kill: `pkill -KILL -f trading_bot_commentary_updated` — use only if TERM didn't work in ~5 seconds

---

## 6. Symbol / volume scale today

**Default watchlist (hardcoded in `core/engine.py:141`):** `['NVDA', 'TSLA', 'PLTR']` — only 3 symbols.

**Dynamic watchlist (auto-refreshed by screener):** `engine.py:2403` — up to **10 symbols** at a time. The list is rebuilt from Schwab's top-mover screener periodically.

```python
self.dynamic_watchlist = list(set(screener_symbols + ['NVDA', 'TSLA', 'PLTR']))[:10]
```

**Not 2,433.** That number was the ML training universe scale (historical bars across many symbols). The live bot scans only the watchlist it picks each cycle — typically under 10.

**Trading frequency observed today in sim:**
- `signal_router received` events: ~50–80/day across all strategies
- Actual orders placed (after risk + ML gating): dramatically fewer — single digits per day in sim currently
- Meta-shadow evals on mean-reversion: 40–50/day

**Ingest scale (separate from trading):**
- `ingest_alpaca_bars.py --top-n 150` pulls the top 150 most-traded symbols from `minute_bars`
- ML training also uses this 150-symbol universe
- Postgres has 228M minute bars total, so scale is historical-research heavy, live-trade light.

---

## TL;DR for a messenger/observer integration

- **Ask for state:** `GET /api/account-stats` or read `trading_state.json`
- **Observe decisions:** tail `trading_bot.log`, grep `engine_decision`
- **Halt trading:** `POST /api/stop`
- **Don't write to Postgres** — the bot doesn't own that surface; stick to JSON + HTTP + WebSocket
- **Don't assume 2,433 symbols at runtime** — today's watchlist is ≤10
