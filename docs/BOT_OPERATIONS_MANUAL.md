# Trading Bot Operations Manual

## About this document

This manual describes how the Schwab-connected trading bot in this repository actually operates: how the process boots, how it finds candidates, how a signal becomes an order, how positions are protected and closed, what the machine-learning and data layers contribute, how operators interact with it, and what its trading characteristics are in practice.

It was produced by reading the code on branch `feature/newsbus-autonomy-decision-card` at commit `01bc27c` (2026-09-24) together with the live configuration, state files, logs, and shadow ledgers present on the production host. Every claim carries a `file.py:line` reference so it can be checked. Defaults quoted are code defaults in `core/config.py`; where `Config().yaml` or `.env` overrides one, the effective value is stated. Secrets are never reproduced.

It complements, and where they disagree supersedes, `docs/ARCHITECTURE.md` (diagrams and loop cadences), `docs/BOT_CAPABILITIES.md` (plain-English overview), and `BOT_CONTEXT.md` (Postgres schema). It does not replace `OPERATIONS_RUNBOOK.md`, but that runbook describes an earlier system and is stale; the runbook essentials in Section 6 reflect the current code.

## Executive summary

The bot is a single Python process: FastAPI plus uvicorn on port 9000, with the trading engine running on its own asyncio loop in a daemon thread. Trading starts only when an operator calls `POST /api/start`, and the engine starts in simulation until switched to live.

It is a **long-only, intraday momentum day-trader on liquid US equities**. It trades regular hours only, blocks entries in the first five minutes and after 15:30 ET, flattens day trades at 15:00 ET, and carries no overnight risk from its own lanes. Position notional is small (about $6k, roughly 14% of equity) because day-trade sizing is halved and a 20%-of-buying-power cap binds. Every position is protected at the broker by a Schwab OCO bracket (limit take-profit plus stop-limit), with software breakeven, ATR trailing, proactive momentum-failure exits, and a flatten-hour rule layered on top.

The design is **shadow-first**. Every lane and most gates ship with a live/shadow flag pair and an append-only NDJSON ledger. Promotion from shadow to live is a scorecard decision against fixed Stage A floors, not a code change. As of this snapshot, taken after the close on 2026-09-24, the day-trade momentum long lane was switched from live to shadow, so **no lane can open a new live position**; exit management, gates, and about ten shadow measurers remain active. The free-news lane is the only one with no live-pause flag, but it has produced no live signals in the logged sessions.

Safety is layered: an account-level circuit (5% bot-only daily loss, three consecutive losses, buying power floor), 38 ordered entry gates, a position state machine with per-position locks, a task supervisor with crash isolation and circuit-breaking, a stream watchdog with a dead-man switch, and broker-authoritative exit handling. Every gate and exit emits one structured `engine_decision` audit line, mirrored to Postgres.

The weakest parts are the machine-learning veto (a class-index mapping defect and a five-month-old model), the backtest realism (no costs), the test suite's coverage of the money paths, and operational drift across three checkouts. These are catalogued in Section 8 so that readers weigh the rest of the manual accordingly.

## System at a glance

| Aspect | Value |
|---|---|
| Process | One Python process, FastAPI + uvicorn on `0.0.0.0:9000`, engine on a daemon thread with its own asyncio loop |
| Broker | Charles Schwab via `schwab-py` (REST + LEVELONE_EQUITIES stream); OCO brackets at the broker |
| Data | Schwab 5-min bars and quotes live; Alpaca 1-min bars into Postgres `minute_bars` for training and backtests; Yahoo/Google/MarketWatch RSS plus Alpaca news into an in-process NewsBus |
| Universe | 8 pinned mega-caps plus 12 dollar-volume-ranked movers, rebuilt every 120 s |
| Lanes | Day-trade momentum long (shadow since 09-24 close), day-trade momentum short (shadow), mean-reversion long/short (shadow), ORB (shadow), free-news (registered, heavily gated), breakout and legacy momentum (off) |
| Sizing | 1.5% of balance at risk per trade, Kelly-scaled, 0.5× for day trades, capped at min($10k, 20% of buying power) |
| Risk halts | 5% bot-only daily loss, 3 consecutive losses, buying power < $100, max 5 concurrent bot positions |
| Exits | Broker OCO (TP limit + stop-limit), breakeven at +0.5R, ATR trail from 2.0 ATR, proactive MACD/RSI failure exits, day-trade flatten at 15:00 ET |
| State | `trading_state.json`, `trading_brain.json`, `trading_commentary.json`, Postgres `bot_*` tables, NDJSON shadow ledgers |
| Interfaces | ~80 HTTP endpoints, WebSocket at 4 Hz, legacy HTML dashboard at `/`, flag-gated Activity page, separate Next.js dashboard |
| Account (snapshot) | About $43.5k balance, $29.8k buying power, four external hands-off positions, no bot-managed positions open |

## 1. Runtime lifecycle

### 1.1 Process model

`trading_bot_commentary_updated.py` is a thin re-export shim; `main()` lives in `api/routes.py:4099`. It installs SIGINT/SIGTERM handlers (`api/routes.py:4166`) and runs uvicorn on `0.0.0.0:$TRADING_BOT_PORT` (default **9000**, `api/routes.py:4106`; the CLAUDE.md mention of port 8000 is stale). Booting the process does **not** start trading.

Trading begins on `POST /api/start` (`api/routes.py:559`). On first call the handler builds `TradingEngineWithCommentary` in the default thread executor (the constructor loads the ML model and brain synchronously, ~13 s) and wires a cross-thread commentary queue so engine-side commentary is broadcast from the FastAPI loop (`api/routes.py:585-661`). It then launches `asyncio.run(trading_engine.start())` on a **daemon thread** (`api/routes.py:666`). The engine therefore has its own asyncio event loop, separate from uvicorn's. Cross-loop hops are explicit: `engine._ws_broadcast_loop` and `db_logger.set_owner_loop()` point back at the FastAPI loop (`api/routes.py:601-609`, `core/engine.py:3696`). `POST /api/stop` only sets `is_running = False` (`api/routes.py:676`). The endpoint takes no body; the mode is not selectable at start (BOT_CONTEXT.md's `{"mode": ...}` is not read).

### 1.2 Engine construction (`core/engine.py:80`)

`__init__` runs in this order:

1. Reads extended-hours flags from `Config().yaml`; creates `QuoteCache`, installs the singleton `PriceBook` (stale threshold 600 s) (`core/engine.py:105-118`).
2. `CommentarySystem`, `TradingBrain` (loads `trading_brain.json`), `ErrorRecoveryManager` (loads `error_recovery_state.json`).
3. `_init_schwab_client()` (`core/engine.py:4038`): requires `SCHWAB_API_KEY`/`SECRET`; loads `token_1.json`, runs `check_token_health()` (7-day refresh-token TTL warning only, `core/token_health.py:28`), builds the client, calls `get_account_numbers`/`get_accounts`, captures `account_id`/`account_hash`, and sets `SchwabDataProvider`. On any failure it falls back to yfinance `RealTimeDataProvider` and the bot runs simulation-only.
4. Optional shadow components (side classifier, regime allocator, conviction sizer, symbol-intel hub, meta model) — each wrapped so failure logs and continues.
5. `RiskManagerWithCommentary` (balance placeholder 100 000 until Schwab sync), `MLPredictorWithCommentary` (model file `ml_model_integrated.pkl`, `ml/models.py:399`), `DbLogger` (Postgres; disabled with a warning if unreachable), `StockScreener`, strategy list (Breakout, OversoldBounceV2/MeanReversion, FreeNews, optional Momentum, DayTradeMomentum long/short, ORB).
6. **A second `TradingBrain()` is created** at `core/engine.py:314`, replacing the first (the `ml_predictor.set_brain` reference still points at the first instance). `brain.reset_for_new_session()` pulls greed/fear 60 % toward neutral.
7. `_load_state()` (`core/engine.py:896`) restores from `trading_state.json`: `trade_history`, `consecutive_losses` (reset to 0 if `last_save` is a prior calendar day), `last_loss_date`, `positions_data` (stashed in `_saved_positions_meta` for ownership restore), and full **simulated positions** including FSM state, bracket IDs and trailing stops.
8. `TradingLossBreaker`, performance/behavioral analyzers, initial commentary card.

Default mode is `TradingMode.SIMULATION_WITH_COMMENTARY`; `/api/start` never passes another value.

### 1.3 `start()` sequence (`core/engine.py:4703`)

1. `is_running = True`, `OrderMonitor(self)` created, `start_time` set.
2. Postgres: `ensure_snapshot_table()` awaited; ledger-integrity column migration fired as a background task.
3. If a Schwab client exists (regardless of mode): account sync into risk manager, `_update_and_track_real_positions()`, seed `_schwab_positions_cache`, `_save_state()`, then `bootstrap_orphan_brackets()` attaches WORKING OCO orders to bot-managed positions that lost their `bracket_order_id`.
4. `_recover_inflight_exits()` (`core/engine.py:3008`): positions persisted in `EXITING` are never re-closed. Live: if Schwab no longer holds the symbol → `CLOSED`; if it does (or the check fails) → `ZOMBIE` for operator review. Sim: reverted to `LIVE`.
5. Per-position asyncio locks created; if the ML model is untrained, a blocking retrain on PLTR/NVDA/TSLA runs.
6. In LIVE mode, a second `_update_and_track_real_positions()` with commentary.
7. `_build_supervisor()` registers the background tasks, `start_all()`, then the coroutine polls `is_running` every 0.5 s and calls `stop_all()` in `finally` (`core/engine.py:4877-4883`).

### 1.4 Background tasks

All engine tasks run under `TaskSupervisor` (`core/task_supervisor.py`). A task that raises is logged, surfaced as a WARNING commentary card, and restarted with backoff (NORMAL 5/30/60 s, CRITICAL 2/5/15 s). After `TASK_RESTART_MAX_CRASHES` (default 3) within `TASK_RESTART_WINDOW_SEC` (300 s) the task enters `CIRCUIT_BROKEN` and stays down until `reset_circuit()`; if that task is CRITICAL, `_quote_streamer_healthy` is forced False, which gates new entries (`core/engine.py:4036`). Note: the commentary card tells the operator to `POST /api/supervisor/reset/{name}`, but no such route exists in `api/routes.py` — only `/api/system-stats` exposes `supervisor.status()`. Recovery today requires a restart.

| Task | Registered at | Priority | Cadence | Does | On exception |
|---|---|---|---|---|---|
| `schwab_stream` | `engine.py:3761` (only if `schwab` streaming lib + client + account_hash) | NORMAL | event-driven; reconnect backoff 1/5/15/30/60 s | Level-one equity websocket → `QuoteCache`/`PriceBook`; subscription reconciler every 2 s; `StreamWatchdog` heartbeat every 30 s, declares silent death after 120 s without ticks during RTH (60 s connect grace) and forces reconnect; `DeadMansSwitch` calls `os._exit(42)` if the watchdog heartbeat stalls 90 s (`core/schwab_stream.py:465-470`, `core/schwab_stream_watchdog.py`) | Internal reconnect loop; `is_healthy()` False after 5 consecutive failures |
| `quote_streamer` | `engine.py:3773` | NORMAL | `QUOTE_REFRESH_SEC` (3 s) | Registers open-position symbols as PriceBook interest; sets `_quote_streamer_healthy` from stream health. No REST fallback (`engine.py:3250`) | Supervisor restart |
| `position_loop` | `engine.py:3774` | **CRITICAL** | `POSITION_LOOP_SEC` (1 s) | FSM exit dispatcher: for each open position calls `_evaluate_exits_with_cached_quote` (breakeven lift, scale-out, trail, stop/TP) using PriceBook marks; skips stale marks. Per-position errors are caught; structural errors re-raised (`engine.py:3528`) | Supervisor restart (fast backoff) |
| `order_monitor` | `engine.py:3780` | **CRITICAL** | `ORDER_MONITOR_INTERVAL_SEC` (5 s) | Polls Schwab WORKING orders for bot-managed live positions with a `bracket_order_id`; handles stop/TP fills, cancels, rejects, partials, trail-replace, broker-flat detection (`core/order_monitor/monitor.py:57`) | Outer `except` swallows and continues — never reaches supervisor |
| `analysis_loop` | `engine.py:3781` | NORMAL | `ANALYSIS_LOOP_RTH_SEC` 30 s when tradable, `ANALYSIS_LOOP_OFF_HOURS_SEC` 300 s otherwise; wakes early on NewsBus high-impact event (`engine.py:5235-5252`) | Session announcements; every 5th cycle Schwab account/positions sync and P&L check; emergency-stop check; in LIVE `_check_order_status`/`_update_real_positions`/buying-power; every 10th cycle `_save_state()` + `brain.save_memories()`; indicators, ML predict, strategy signals, signal routing | Inner `except` posts an error card and sleeps 60 s (`engine.py:5258-5269`) |
| `market_indices` | `engine.py:3785` | NORMAL | 10 s | Refreshes `MarketIndicesCache` for the dashboard regime strip | Per-iteration debug log |
| `state_persistence` | `engine.py:3792` | NORMAL | 30 s (first save delayed 30 s) | `_save_state()` | Per-iteration warning |
| `screener_loop` | `engine.py:3799` | NORMAL | `SCREENER_LOOP_SEC` (120 s) | Sole writer of `dynamic_watchlist`, `screener.top_movers`, PriceBook "watchlist"/"ticker_tape" scopes (`core/loops/screener_loop.py`) | Per-tick swallowed |
| `news_loop` | `engine.py:3810` | NORMAL | `trading.news_loop_sec` (20 s) | Fetches Yahoo/Google/MarketWatch news for the watchlist in batches of 5, scores each item through `core/sentiment_backend.score` (VADER by default, FinBERT when `SENTIMENT_BACKEND=finbert`), publishes to `NewsBus`; high-impact items set the analysis wake event (`core/loops/news_loop.py:73`) | Per-tick swallowed |
| `alpaca_news_bus` | `engine.py:3830` (if Alpaca keys set) | NORMAL | `ALPACA_NEWS_BUS_INTERVAL_SEC` | Tier-1 Alpaca news into NewsBus | Per-tick swallowed |
| `active_open_desk` | `engine.py:3889` (if `ENABLE_ACTIVE_OPEN_DESK`) | NORMAL | `ACTIVE_OPEN_DESK_INTERVAL_SEC` (5 s) | Continuous sentiment/regime/indicator review of managed open trades (shadow by default) | Supervisor restart |

Not supervised: the FastAPI-side `commentary_broadcaster` task (`api/routes.py:615`) and the stream's own reconciler/watchdog subtasks (which use `supervise_task` done-callbacks for logging only).

### 1.5 Trading modes

`TradingMode` has `PAPER`, `LIVE`, `SIMULATION_WITH_COMMENTARY` (`core/models.py:9`). `PAPER` is defined but never assigned anywhere. The engine starts in SIMULATION; `POST /api/toggle-mode {"mode":"live"}` switches to LIVE (refused without a Schwab client) and runs `_update_real_positions()`; any other value returns to SIMULATION (`api/routes.py:133`). Mode effects:

- **LIVE**: orders go to Schwab via `_execute_real_trade` (decorated with `order_circuit_breaker`, `core/engine.py:1027`) with OCO brackets; `sync_positions_with_schwab` and `_update_real_positions` only run in LIVE (`engine.py:4503`, `2508`); close confirmations (`require_confirmations`) and `manual_close_only` apply only in LIVE (`engine.py:763`, `9015`); live scale-out is bookkeeping-gated by `ENABLE_LIVE_SCALE_OUT` (`engine.py:3433`).
- **SIMULATION**: fills are in-memory in `simulated_positions`; real market data still flows from Schwab/yfinance; market-hours and econ-blackout gates apply identically ("sim previews live").
- **Shadow** is not a mode but a set of per-feature flags (`*_SHADOW`, `*_LIVE_ENTRIES_ENABLED` in `core/config.py`) that log would-be decisions to NDJSON ledgers without placing orders (e.g. day-trade short, ORB, thesis revalidation, regime allocator).

Both position dictionaries persist across a mode toggle; `clear_inactive=true` drops stale sim positions when going live.

### 1.6 Market hours

`is_market_hours()` (`core/engine.py:2604`) classifies ET weekdays as `closed` (<04:00, ≥20:00), `premarket` (04:00-09:30), `regular` (09:30-16:00), `afterhours` (16:00-20:00). Behaviour by phase:

- **Off-hours / pre-market**: the analysis loop never pauses (`engine.py:4894`); it slows to the 300 s cadence and posts a one-shot "Orders Held" card. `_in_warmup_window()` (`WARMUP_MINUTES_BEFORE_OPEN`, default 30) restores the 30 s cadence before the open so indicator state is ready.
- **Entries**: the signal router blocks new entries outside regular hours unless `trading.extended_hours.allow_premarket/afterhours` is set (`engine.py:6439`); extended-hours orders use limit orders (`should_use_limit_order`, `engine.py:2642`).
- **Late day**: no new entries after `LATE_ENTRY_CUTOFF_HOUR:MINUTE` (15:30) (`engine.py:6565`); day-trade momentum entries are blocked from `DAY_TRADE_FLATTEN_HOUR` (15) and existing day-trade positions are force-exited once the ET hour reaches it (`engine.py:8756`). There is no global EOD flatten for other strategies — swing/mean-reversion positions can hold overnight under their Schwab OCO.
- **Econ calendar**: `get_econ_calendar().is_blackout()` (`core/econ_calendar.py`; `DatedEconCalendar` by default, overridable by `trading.econ_calendar_events` or `ECON_CALENDAR_API_URL`) is a soft veto on new entries only; analysis and exits continue in both modes (`engine.py:10193`).
- Exits, `position_loop` and `order_monitor` run at all hours; the stream watchdog only declares silent death during RTH.

### 1.7 State persistence

| Artifact | Writer | When |
|---|---|---|
| `trading_state.json` | `_save_state()` (`engine.py:2879`) | `state_persistence` loop every 30 s; every 10th analysis cycle; after startup sync; in the signal handler. Contains last 100 trades, `schwab_pnl`, `consecutive_losses`, `last_loss_date`, live `positions_data` (ownership, brackets, reasoning), full `simulated_positions` with FSM state |
| Postgres `bot_positions` | `db_logger.sync_positions` scheduled from `_save_state` | Same cadence (live record wins on symbol collision) |
| Postgres `bot_trades`, `bot_decisions`, `bot_decision_snapshots`, `bot_shadow_news_vetoes` | `DbLogger` (`data_providers/db_logger.py`) | On each trade close, decision/audit, feature snapshot, news veto |
| `trading_brain.json` | `brain.save_memories()` (`core/brain.py:124`) | Every 10th analysis cycle, on `/api/reset-brain`, and at shutdown; keeps last 1 000 memories |
| `trading_commentary.json` | `CommentarySystem._save_to_file` (`core/commentary.py:178`) | Rate-limited to one full rewrite per 5 s, truncated to last 1 000 entries |
| `ml_model_integrated.pkl` | `ml_predictor.save_model()` | Shutdown handler only (online learning otherwise in memory) |
| `error_recovery_state.json` | `ErrorRecoveryManager` | On each recorded failure |
| NDJSON ledgers (`symbol_intel_log`, `regime_allocator_shadow`, `conviction_sizer_shadow`, `data/shadow_*.ndjson`, `data/day_trade_short_shadow.ndjson`) | shadow components / strategies | Append-only per evaluation |
| `trading_bot.log` (+ `.1`-`.5`) | `RotatingFileHandler`, 10 MB × 5, JSON lines (`core/config.py:4115`) | Continuous |

On restart, recovered state = `trading_state.json` (positions, counters), `trading_brain.json`, ML pickle, the Schwab account itself (live positions are re-discovered from the broker and ownership restored from `_saved_positions_meta` / today's `bot_trades`, `engine.py:4472`), and orphan OCO brackets. Commentary history and NDJSON ledgers are append-only context, not recovered into decisions.

### 1.8 Shutdown

`shutdown_handler` (`api/routes.py:4115`) sets `is_running = False`, joins the engine thread for up to 15 s, then saves state, brain and ML model and `sys.exit(0)`. If the thread does not stop in time it skips the save (to avoid a torn JSON write racing the periodic saver) and `os._exit(1)`. Inside the engine, `is_running=False` ends every loop's `while` and `stop_all()` cancels the tasks; `SchwabQuoteStream.stop()` (websocket logout) is never called explicitly — the stream task is simply cancelled. **Open positions are left as-is on shutdown**; live positions remain protected only by their Schwab OCO brackets. Explicit liquidation is the operator's job via `POST /api/emergency/close-all {"mode": "...", "confirm": "YES"}` (`api/routes.py:3298`), which closes bot-managed positions through the normal close path and skips external/manually-managed ones.

### 1.9 Failure handling

- **Loss circuit**: every 5th analysis cycle the engine compares P&L to 5 % of account balance. With `ENABLE_BOT_ONLY_PNL_CIRCUIT` (default True) it uses bot-only P&L and rechecks locally; if breached it closes all `self.positions` (or, under `manual_close_only`, leaves them) and sets `is_running=False` — a hard halt that requires a new `/api/start` (`engine.py:5063-5159`). `TradingLossBreaker` (`core/config.py:166`) trips at 10 %/15 % with a 24 h reset. `MAX_CONSECUTIVE_LOSSES` (3) gates entries in the risk manager; the counter resets daily.
- **API/order breakers**: `api_circuit_breaker` (5 failures / 60 s) and `order_circuit_breaker` (3 failures / 300 s) in `circuit_breaker.py:89-98`; only the order breaker is actually applied (`_execute_real_trade`).
- **ErrorRecoveryManager** (`error_recovery.py`) records failed orders for retry; it is referenced only three times in the engine and is largely dormant.
- **Loop policy**: per-iteration `try/except` with log-and-continue in every loop; the analysis loop additionally backs off 60 s; `position_loop` re-raises structural errors so the supervisor restarts it; `order_monitor` never re-raises.
- **Stream loss**: reconnect with backoff, watchdog-forced reconnect on tick silence, dead-man `os._exit(42)` expecting an external process supervisor to restart the bot (none is shipped in-repo; `start_*.sh` scripts are plain launches).
- **Token expiry**: warned at startup only; a mid-session refresh-token expiry surfaces as Schwab call failures, not a controlled halt.

## 2. Universe, market context, and strategies

All references are to `/home/nvidia/claude/grok/trading_bot_commentary`. Defaults quoted are the code defaults in `core/config.py`; where `Config().yaml` overrides one, the yaml value is stated. The running instance is in LIVE mode (`trading_state.json` reports `"mode": "live"`), which matters because several lanes signal in every mode but are blocked from placing LIVE orders by a separate `*_LIVE_ENTRIES_ENABLED` flag.

### 2.1 Universe / watchlist

**Who owns it.** `ScreenerLoop` (`core/loops/screener_loop.py:60-130`) is a supervised task and the sole writer of `engine.dynamic_watchlist`, `screener.top_movers`, and the PriceBook `watchlist`/`ticker_tape` interest scopes. It calls `StockScreener.screen_stocks()` every `SCREENER_LOOP_SEC` (default 120 s, `core/config.py:780`). The analysis loop only reads the list (`core/engine.py:5578`), and skips any symbol that already has a position (`engine.py:5590-5600`). Analysis cadence is `ANALYSIS_LOOP_RTH_SEC` 30 s during RTH / 300 s off-hours, with early wake on a NewsBus high-impact event (`engine.py:5235-5247`).

**Candidate sources** (`analysis/screener.py:96-141`), in order: Yahoo most-active (30 names), Yahoo day_gainers + day_losers, Schwab `EQUITY_ALL` sorted by VOLUME, PERCENT_CHANGE_UP, PERCENT_CHANGE_DOWN, `$SPX` and `$COMPX` up/down leaders, and a hand-curated ~60-name "volatile candidates" list (`screener.py:612-659`, quote volume > 100k). Schwab mover calls are cached 60 s.

**Filters.**
- `_is_tradeable` (`screener.py:937-972`): volume >= 250k, price $3-$1000, spread <= 1% of price, and a hard exclude of `SPY QQQ IWM DIA VXX UVXY TQQQ SQQQ TSLL NVDL TMF TLT`.
- `LEVERAGED_ETF_BLOCKLIST` (`screener.py:20-40`): ~45 leveraged/inverse/volatility ETFs, always rejected.
- Universe quality filter (`ENABLE_UNIVERSE_QUALITY_FILTER` default True; `screener.py:787-861`): blocklist, daily close > SMA50 (60-day yfinance bars cached 30 min), and 10-day return not worse than SPY by 5 pp. Missing data passes through.
- Mover relax (`ENABLE_MOVER_QUALITY_RELAX` default True; `screener.py:863-935`): Yahoo-sourced movers skip the SMA50/RS gates but must have price >= `MIN_MOVER_PRICE` ($5), volume >= 500k, spread <= 1%.

**Scoring** (`screener.py:975-987`): `0.4*min(volatility/5,1) + 0.3*min(volume/10M,1) + 0.3*min(|pct_change|/10,1)`; the top 30 are kept as `top_movers`.

**Watchlist assembly** (`screener_loop.py:145-262`). With `ENABLE_PINNED_WATCHLIST` (default True, yaml `true`), the pinned list `NVDA TSLA META AMZN MSFT GOOGL AVGO AMD` (yaml `pinned_watchlist`, default at `config.py:826-845`) takes the first slots unconditionally (even if not in today's movers), and the remaining `WATCHLIST_SIZE - 8` slots (`WATCHLIST_SIZE` default 20, `config.py:786-797`) are filled by non-pinned movers ranked by dollar-volume (`price * volume`) with the `MIN_DOLLAR_VOLUME` floor (yaml **$25M**; code default $10M). If pinning is disabled, the legacy path takes screener order plus anchors `NVDA TSLA PLTR`. Each refresh posts a "Watchlist Updated" commentary.

### 2.2 Market context inputs

| Module | Computes | Consumers |
|---|---|---|
| `core/market_indices.py` | `MarketIndicesCache`: SPY/DIA/QQQ/IWM/$VIX plus 11 SPDR sector ETFs, refreshed every 10 s by `engine._market_indices_loop` (`engine.py:3229-3245`) | `market_context`, dashboard `/api/market-indices` |
| `core/market_context.py` | `read_market_context(symbol)` (`:361-465`): `spy_direction` (+/-0.3%), `regime` (`:228-260`): **risk_on** if SPY > +0.3% and (VIX chg < 0 or VIX < 16); **risk_off** if SPY < -0.5% and VIX chg > 5%, or VIX chg > 10%; else **mixed**. `time_of_day` buckets (`:263-299`, assumes EDT): off_hours / opening_30 / morning / midday / afternoon / closing_30. `conviction_multiplier` 0.5-1.5 (`:302-359`). `allows_long` (`:177-195`) blocks risk_off and, since 2026-09-24, mixed with SPY red; `allows_short` blocks risk_on. Also `SpyIndexContext`, `RsFromOpenContext`, `InMoveTurnContext` (`:555-1090`) for the day-trade shadow gates. | breakout, mean-rev, news, day-trade long/short, ORB |
| `core/spy_regime.py` | `SpyRegimeCache.ema_slope_pct(9)` over 30 SPY 5-min closes | Only OversoldBounceV2 Gate E. **No producer calls `update()`** in the engine (grep), so it always returns 0.0 ("flat") |
| `core/direction_reader.py` | `read_direction(close, indicators)` (`:351`): five bounded components (EMA stack, price vs SMA50, EMA-20 slope, ADX x MACD-hist sign, OBV slope) -> `direction` -10..+10 and `phase` ranging/early/middle/late/exhausted (`:310-348`: up-phase exhausted at RSI >= 75, late if > 5% above SMA50 or RSI >= 70) | breakout (`builtin.py:268`), mean-rev uptrend pullback (`:641`), day-trade short (`:2790`), news (`news_strategy.py:1451`); flags `ENABLE_DIRECTION_GATE_*` default True (`config.py:499-530`) |
| `core/econ_calendar.py` | `DatedEconCalendar` (default) with real FOMC/CPI/jobs dates; `is_blackout()` | `engine._is_news_blackout` (`engine.py:10193-10223`): soft veto of new entries at the signal router, analysis continues |
| `core/news_bus.py` | `NewsBus`: 4 h TTL store written only by `NewsLoop` (`core/loops/news_loop.py`, `NEWS_LOOP_SEC` 20 s) and `AlpacaNewsBusPublisher` (tier-1, 60 s). `evaluate_gate` (`:372-470`): VETO_NO_CORROBORATION (nothing in bus), VETO_STALE (freshest > `NEWS_GATE_MAX_AGE_SEC` 1800 s), VETO_LOW_TIER (best tier > 2), REDUCED_SIZE 0.5x (single source), FULL_SIZE 1.0x (>= 2 sources or any high-impact) | `FreeNewsSignalStrategy` (wired via `set_news_bus`, `engine.py:3821`), theme-shock shadow check on every signal (`engine.py:5888-5925`), analysis-loop wake |
| `core/symbol_intel.py` | `build_symbol_view` per symbol per pass (`engine.py:5749-5753`): quality score = 50% direction conviction + 25% ADX + 25% volume; appends to `symbol_intel_log.ndjson` (10-min throttle) | Dashboard panel only; no strategy reads it |
| `core/classifier/` | Rule-based + trained side classifier (`rule_based.py`: hard gates on ATR% > 8%, ADV < 500k, 3-day confirmed trend, |sentiment| > 0.6; weighted soft scores, threshold 0.55) | **Inert today**: `USE_SIDE_CLASSIFIER` default False (`config.py:715`) so `engine._classifier_runtime` stays None and `shadow_evaluate` is skipped (`engine.py:162-184`, `:5958`) |

### 2.3 Strategy interface

`TradingStrategyWithCommentary` (`strategies/base.py:15-30`) requires `name` and `async generate_signal_with_commentary(market_data) -> Optional[TradingSignal]`. `MarketData` (`core/models.py:261-270`) carries OHLCV, `timeframe`, and the `indicators` dict. `TradingSignal` (`models.py:46-56`): `symbol, signal_type (BUY=1/SELL=-1/HOLD=0), strength, entry_price, stop_loss, take_profit, position_size (always 0; the risk manager sizes), reasoning: dict, confidence, timestamp`. Size hints travel in `reasoning` (`market_context_conviction`, `mc_size_mult`, `day_trade_size_multiplier`, `orb_size_multiplier`, news gate multiplier, exit hints such as `scale_out_r_override`).

`_log_decision` (`base.py:32-92`) writes one INFO line per evaluation:

```
strategy_decision strategy=<name> symbol=<SYM> action=<signal_buy|signal_sell|skip|error|veto|shadow|hard_veto|news_gate|size_reduced|advisory|bypass> reason=<snake_case> close=<px> k=v ...
```

and, unless `skip_snapshot=True`, emits a `DecisionSnapshot` to `db_logger` for ML training. The engine's own gates log `engine_decision component=... symbol=... action=... reason=... mode=...` (`engine.py:767-800`).

### 2.4 Strategies

Registration order is fixed in `engine.py:311-383`: Breakout, mean-reversion (legacy or v2), Momentum (only if env `ENABLE_MOMENTUM=1`), FreeNews, DayTradeMomentum, DayTradeMomentumShort, ORB (only if enabled). Every strategy uses a 1%-of-price ATR floor (`builtin.py:18-23`).

| Strategy (`name`) | Side | Status today | Controlling flag(s) | Entry gist | Stop / target |
|---|---|---|---|---|---|
| `breakout` (`builtin.py:157`) | long (+short mirror) | **OFF** (returns None at `:173`) | `ENABLE_BREAKOUT_LONG` False; `ENABLE_SHORT_MIRRORS` False | close > high_20 x 1.003, ADX >= 25, close >= SMA20, vol_ratio >= 2.0, direction >= 3 & phase early/middle, context allows_long | 1.5 x ATR; smart target else 2R |
| `momentum` (`builtin.py:1253`) | long (+short) | **OFF** (not registered; also gated) | env `ENABLE_MOMENTUM`; `ENABLE_MOMENTUM_LONG` False | MACD > signal, RSI 55-65, ADX > 25, close > SMA50 | 1.5 x ATR / 3 x ATR |
| `mean_reversion` legacy (`builtin.py:551`) | long; short shadow | **LIVE long, playbook probation 0.5x**; short shadow | `USE_OVERSOLD_BOUNCE_V2` False selects it; env `MEAN_REV_LIVE_ENTRIES_ENABLED=1` (2026-09-28 13:17 ET, §9); playbook `mean_rev_intraday\|long` on probation (`size_mult` 0.5, §9); `ENABLE_MEAN_REV_SHORT` False, yaml `enable_mean_rev_short_shadow: true` | see 2.4.1 | 2.5 x ATR (`MEAN_REV_ATR_STOP_MULTIPLIER`); smart target else 2R |
| `oversold_bounce_v2` (`oversold_bounce_v2.py`) | long | **OFF** | `USE_OVERSOLD_BOUNCE_V2` False (`config.py:2024`) | RSI < 30 & close < BB lower, vol_ratio > 2, hammer/reclaim bar, swing low within 0.5 ATR, SPY slope > -0.10%, no fresh negative news | max(2 x ATR, 0.5 x bar range); 3R; hints scale-out 0.5R, BE at 1.5 ATR, 30-min time stop |
| `news` FreeNews (`news_strategy.py:404`) | long & short | **LIVE** | always registered; `ENABLE_NEWS_VERIFIER` False (advisory), `ENABLE_NEWS_TECHNICAL_CONFIRMATION` True | see 2.4.2 | 1.5 x ATR; BUY smart target else 2R; SELL 2R |
| `day_trade_momentum` (`builtin.py:1435`) | long | **Signals; entries flag on but playbook-suspended** | `ENABLE_DAY_TRADE_MOMENTUM` True; env `DAY_TRADE_LIVE_ENTRIES_ENABLED=true` (2026-09-28 13:17 ET, §9); playbook `dt_momentum\|long` **suspended** (`size_mult` 0.0, §9) — every entry is skipped regardless of the env flag | see 2.4.3 | 1.5 x ATR; 2R (hard-coded `:2537-2541`) |
| `day_trade_momentum_short` (`builtin.py:2653`) | short | **LIVE** (full size; no playbook entry yet) | `ENABLE_DAY_TRADE_SHORT` True; env `DAY_TRADE_SHORT_LIVE_ENTRIES_ENABLED=1` (2026-09-28 13:17 ET, §9); `ENABLE_DAY_TRADE_SHORT_SHADOW` True (shadow ledger still written alongside live orders) | see 2.4.4 | 1.5 x ATR above; 2R below |
| `orb_contraction_rvol` (`builtin.py:3211`) | long & short | **LIVE** (full size; no playbook entry yet) | `ENABLE_ORB_STRATEGY=1` (`config.py:3478`); env `ORB_LIVE_ENTRIES_ENABLED=1` (2026-09-28 13:17 ET, §9) | 15-60 min after open, contraction >= 20%, RVOL >= 1.2, close beyond opening range, hard skip risk_off | 1.0 x ATR capped at range edge +/- 0.5 ATR; 2R; size 0.5x |
| `NewsSignalStrategy` (`builtin.py:92`) | either | **Dead code** | never instantiated | |avg VADER| > 0.3 | 1.5 x ATR; 2R |
| `fit:<family>` (`strategies/fit_lane.py`) | per family (`strategies/families/registry.py:FAMILIES`) | **SHADOW** unless family listed in `FIT_LANE_LIVE_ENTRIES_ENABLED` | `FIT_LANES_ENABLED` (registers the adapter); `FIT_LANE_LIVE_ENTRIES_ENABLED` (LIVE entries); `SWING_ALLOW_OVERNIGHT` (multiday families held across sessions) | family rules from `strategies/families` with per-symbol params from the fit table (`data/strategy_hunt/fit/strategy_fit.json`) | stop/target set by the family's own grid (see `research/strategy_hunt/promote.py`, `research/strategy_hunt/cli.py promote`) |

Note the discrepancy: the engine comment at `engine.py:315` says v2 is "default True", but `config.py:2024` returns False (backtest 2026-05-19 showed v2 PF 1.04 vs legacy 1.72) and the yaml does not set it, so the legacy mean-reversion class is the one running.

#### 2.4.1 Legacy mean reversion (`builtin.py:551-1250`)
Two long entry patterns: **oversold_bounce** = RSI < 30 and close < BB lower; **uptrend_pullback** (`ENABLE_MEAN_REV_UPTREND_PULLBACK` True) = RSI 30-55, close >= SMA50, close within -1%..+8% of BB lower, plus direction >= 2 and phase != exhausted (`:641-655`). Then: market-context `allows_long` (`ENABLE_MARKET_CONTEXT_GATE` True); falling-knife skip if close < SMA50 and MACD < signal and (relaxed) bar is red (`:696-712`); price-direction bar check (`:772-800`): green bar, and for the classic path also vol_ratio >= 1.5 and lower wick > body. Stop 2.5 x ATR, target from `compute_smart_target` (BB upper / high_20 candidates) else 2R. Every long signal is also appended to `data/shadow_long_log.ndjson` (`ENABLE_MEAN_REV_LONG_SHADOW` True, `:932-985`). Short branch (RSI > 70 and close > BB upper, `:1014`) is disabled by `ENABLE_MEAN_REV_SHORT` and instead writes a hypothetical 2.5 x ATR / 2R short to `data/shadow_short_log.ndjson`, throttled to weekdays 10:00-16:00 ET and 30 min per symbol (`:35-72`). In the engine, signals still face `REGIME_GATE_LIVE_MEANREV` (True): the regime allocator vetoes mean-rev when SPY efficiency ratio says trending (`engine.py:6010-6050`), and the LIVE pause at `engine.py:6935`.

#### 2.4.2 Free news (`news_strategy.py:486-1667`)
Per-symbol 20-min cooldown (`:489`). Reads bus items up to 4 h old, falling back to the aggregator; requires >= 5 articles (`:537`). VADER average sentiment fires when |s| > 0.30 with >= 5 articles, |s| > 0.40, or a high-impact keyword (earnings/beat/miss/sec/investigation) with |s| >= 0.20 (`:586-589`); sign picks BUY/SELL. Gates in order: knife filter vs SMA50 + MACD unless conviction bypass (|s| >= 0.55 and >= 8 articles, `:613-650`); price-direction bar check; `extended_already_ran` (RSI > 70 and close > SMA20 x 1.01, `:910-925`); NewsBus thesis gate (veto or 0.5x/1.0x size, `:990-1085`, exception = fail-closed); NewsVerifier in advisory mode but a hard veto if zero fresh articles (`:1268-1290`); technical confirmation for BUY: RSI < 55, <= 5% above SMA20, MACD >= signal, vol_ratio >= 1.2 (mirrored for SELL, `:1375-1400`); direction gate `allows_long_entry`/`allows_short_entry`; market-context gate. Size hints: news gate multiplier and `market_context_conviction`.

#### 2.4.3 Day-trade momentum long (`builtin.py:1467-2650`)
Context (`:1503-1630`): hard block if SPY <= -1.5% and VIX >= +15%; **hard skip in risk_off** (`DAY_TRADE_HARD_SKIP_RISK_OFF` True, else 0.25x); opening_30 -> size 0.5x; off_hours hard skip. RS vs SPY (same-basis via Schwab quote, `ENABLE_SAME_BASIS_RS` True) must be >= `MOMENTUM_MIN_RS_VS_SPY` 0.5 pp, or >= `PINNED_MIN_RS_VS_SPY` 0.25 for pinned names; vol_ratio >= 1.5. Patterns (`:1720-1745`): **breakout** (close > high_20, ADX > 20, < 3% above), **pullback** (RSI 40-60, within 2% of SMA20, MACD > signal), **continuation** (RSI 55-75, ADX > 25, MACD > signal, close > SMA20). Hard vetoes: continuation or breakout with RSI >= 70 (`ENABLE_HARD_VETO_*_RSI70` True, `:2140-2240`). Shadow-only gates logged to `data/shadow_*.ndjson`: `DT_OPEN30_CONT_INDEX_CONFIRM`, `DT_OPEN30_CONT_RS_FROM_OPEN`, `DT_INMOVE_INDEX_TURN` (yaml: measure on, `*_LIVE_ENFORCE` false). `reasoning` carries `is_day_trade`, `day_trade_size_multiplier` 0.5, `flatten_hour` 15, `mc_size_mult`, `entry_pattern`, `session_id`. Engine LIVE-only gates: flatten-hour entry block from 15:00 ET, RSI <= 50 block, RSI >= 70 veto on breakout/continuation (`engine.py:6740-6900`), late-entry gate in shadow (`ENABLE_LATE_ENTRY_GATE` / `LATE_ENTRY_GATE_SHADOW` True).

#### 2.4.4 Day-trade momentum short (`builtin.py:2685-3190`)
Skips `HANDS_OFF_DENYLIST` (`MU HQGE SPCX`); **hard block in risk_on**; off_hours skip; direction `allows_short_entry` and phase != exhausted; symbol must underperform SPY by >= 0.5 pp (`DAY_TRADE_SHORT_MIN_WEAK_RS_VS_SPY`); vol_ratio >= 1.5; RSI within 30-85 (no shorting oversold or parabolic). Patterns: **breakdown** (close < low_20, ADX > 20, < 3% below), **continuation_down** (RSI 30-50, close < SMA20, MACD < signal), **rejection** (within 2% of high_20, MACD < signal, ADX > 20). Flatten-hour block >= 15:00 ET. With `DAY_TRADE_SHORT_LIVE_ENTRIES_ENABLED=1` (§9) it now takes the LIVE path and emits a real signal instead of writing `data/day_trade_short_shadow.ndjson`; that shadow write (same 10:00-16:00 / 30-min throttle) only fires when the LIVE flag is off (`:2997-3100`). `DAY_TRADE_SHORT_MAX_CONCURRENT` 2 applies.

### 2.5 How signals are combined

There is no consensus vote and no first-wins short-circuit. For each watchlist symbol the analysis loop runs an optional ML prediction (`ML_PREDICTION_ENABLED` True, 4 s timeout, used downstream as `ml_signal`) and then calls **every** registered strategy in order (`engine.py:5793-5806`); each non-None signal is passed independently to `_process_signal_with_commentary` along with the ML output. Duplicate exposure is prevented structurally: held symbols are removed from the scan, and the router's own gates (LIVE pause flags, regime gate, econ blackout soft veto, max positions, mean-rev risk budget at `engine.py:7376+`, correlation guard) decide acceptance. `primary_strategy` is not an engine priority setting; it is the attribute of the optional `MetaInference` shadow model (`engine.py:275`, `:5833`) that limits meta-scoring to signals whose `reasoning['strategy']` starts with that name, and it is disabled when `ml_meta_model.pkl` is absent. Regime-allocator and conviction-sizer shadows write `regime_allocator_shadow.ndjson` / `conviction_sizer_shadow.ndjson` for every routed signal without affecting it (except the mean-rev regime gate noted above).

**Net effect today (updated 2026-09-28 13:17 ET, §9):** free-news (long or short), mean-reversion long (playbook probation, 0.5x size), day-trade momentum short, and ORB (long & short) can all open new LIVE positions; day-trade momentum long produces signals but is playbook-suspended (every entry skipped, `size_mult` 0.0) even though its own `*_LIVE_ENTRIES_ENABLED` flag is now on; breakout, momentum, and oversold-v2 remain off.

## 3. From signal to order to exit

This section follows one trade through the engine: the entry gates a strategy signal must clear, how the share count is computed, how the order and its bracket reach Schwab, how brackets and positions are monitored, how exits fire, how the book is reconciled against the broker, and how the outcome is booked and learned from. Line references are to the current branch.

### 3.1 Where a signal enters

The analysis loop (`core/engine.py:4885` `_analysis_loop_body`) first calls `_evaluate_trading_conditions` (`engine.py:5480`), which in LIVE mode defers to `risk_manager.check_trading_allowed()` (`risk/manager.py:401`). That is the account-level circuit: margin call, buying power < $100, daily loss ≤ −`MAX_DAILY_LOSS` (5%) × balance (using bot-only P&L when `ENABLE_BOT_ONLY_PNL_CIRCUIT`, default True), or `consecutive_losses ≥ MAX_CONSECUTIVE_LOSSES` (3, auto-reset at day rollover). If allowed, `_analyze_markets_with_commentary` (`engine.py:5567`) walks the dynamic watchlist (symbols already held are skipped), runs the ML predictor, asks each strategy for a signal, and hands it to `_process_signal_with_commentary` (`engine.py:5927`, called at `:5807`).

Every gate below emits a structured line through `_audit` (`engine.py:767`): `engine_decision component=… symbol=… action=… reason=… mode=… k=v`, mirrored fire-and-forget into Postgres `bot_decisions` (`data_providers/db_logger.py:373`). Four gates additionally write a `DecisionAction.VETO` snapshot to `bot_decision_snapshots` via `_emit_veto_snapshot` (`engine.py:817`, `db_logger.py:921`): regime_gate, mean_rev_quality_gate, econ_blackout, ml_veto.

### 3.2 Entry gates, in execution order

| # | Gate (audit component / reason) | Condition | Config flag (default) | Failure mode | Line |
|---|---|---|---|---|---|
| 0 | `signal_router received` | logs strategy, side, confidence, strength, ml_signal | — | — | 5937 |
| – | shadow: meta-model, side_classifier, `regime_allocator shadow`, `mean_rev_shadow_ledger`, `conviction_sizer shadow` | log only | — | never blocks | 5946–6335 |
| 1 | `regime_gate blocked regime_gate_meanrev` | mean_reversion / oversold_v2 only; SPY 20-day efficiency ratio ≥ 0.30 ("trending") | `REGIME_GATE_LIVE_MEANREV` (True) | fail-open (ER None → allow) | 6017 |
| 2 | `conviction_floor blocked below_meta_floor` | mean-rev family; `meta_proba < CONVICTION_FLOOR_META` | `ENABLE_CONVICTION_FLOOR_MEANREV` (True), floor 0.65 | fail-open (no meta_proba → allow) | 6069 |
| 3 | `mean_rev_quality_gate blocked` | mean-rev family; RSI must be in [15, 35); price ≤ 5% below VWAP | `ENABLE_MEAN_REV_QUALITY_GATE` (True) | fail-open on missing indicators | 6152 |
| 4 | `mean_rev_regime_gate blocked mean_rev_risk_off_blocked` | mean-rev family; `market_context_regime == "risk_off"` | `MEAN_REV_EXCLUDE_RISK_OFF` (True) | fail-open | 6191 |
| 5 | `mean_rev_dedupe_gate blocked` | same symbol mean-rev entry < 15 min ago | `MEAN_REV_DEDUPE_MINUTES` (15) | — | 6222 |
| 6 | `health_gate skip stream_dead` | LIVE only; Schwab quote stream `is_healthy()` False | none | fail-closed | 6371 |
| 7 | `health_gate skip management_unavailable` | LIVE only; any bot-opened position < 5 min old with `managed_by_bot=False` and not external | none | fail-closed | 6415 |
| 8 | `market_hours skip session_*` | outside 09:30–16:00 ET unless `allow_premarket`/`allow_afterhours` (both False, `engine.py:89`) | yaml `trading.extended_hours.*` | both modes | 6446 |
| 9 | `econ_blackout skip blackout_soft_veto` | active high-impact event window (`_get_active_blackout`) | econ calendar | analysis continues, entries blocked | 6474 |
| 10 | `early_session skip first_5min_hard_block` | 09:30–09:34 ET, no bypass | hard-coded | — | 6539 |
| 11 | `late_entry skip after_entry_cutoff` | ET ≥ 15:30 and < 16:00 | `LATE_ENTRY_CUTOFF_HOUR/MINUTE` (15:30) | — | 6573 |
| 12 | `late_entry_gate shadow/skip` | BUY on day_trade_momentum / mean_reversion(_short) / orb; extension ≥ 0.75, price > VWAP + 1.0×ATR, or ≥ 5 bars since impulse | `ENABLE_LATE_ENTRY_GATE` (True), `LATE_ENTRY_GATE_SHADOW` (True) | **shadow today** — logs, does not block | 6640 |
| 13 | `daytrade_live_pause skip live_entries_disabled` | LIVE + day_trade_momentum | `DAY_TRADE_LIVE_ENTRIES_ENABLED` (env `true` since 2026-09-28 13:17 ET → this gate now passes) | this gate no longer blocks day-trade long, but the earlier `playbook` gate (§9, between gate #1 and #2 above) skips every `dt_momentum\|long` entry first (`playbook_suspended`) — the signal never reaches gate 13 today | 7258 |
| 14 | `daytrade_flatten_hour_gate skip` | LIVE + `is_day_trade` + ET hour ≥ flatten hour | `DAY_TRADE_FLATTEN_HOUR_ENTRY_GATE_ENABLED` (True), hour 15 | fail-open on exception | 6750 |
| 15 | `daytrade_rsi_entry_gate skip rsi_below_50_entry_blocked` | LIVE + day-trade BUY + RSI ≤ 50 | `DAY_TRADE_RSI_ENTRY_GATE_ENABLED` (True) | fail-open | 6814 |
| 16 | `daytrade_rsi_high_entry_veto skip` | LIVE + day-trade BUY + breakout/continuation + RSI ≥ 70 | `DAY_TRADE_RSI_HIGH_ENTRY_VETO_ENABLED` (True) | fail-open | 6879 |
| 17 | `meanrev_live_pause skip` | LIVE + mean_reversion(_short) | `MEAN_REV_LIVE_ENTRIES_ENABLED` (env `1` since 2026-09-28 13:17 ET → gate passes; upstream `playbook` gate applies `mean_rev_intraday\|long` probation `size_mult` 0.5 first, §9, but does not skip) | — | 7503 |
| 18 | `orb_live_pause skip` | LIVE + orb_contraction_rvol | `ORB_LIVE_ENTRIES_ENABLED` (env `1` since 2026-09-28 13:17 ET → gate passes; no playbook entry for `orb`, so ungoverned/full size) | — | 7551 |
| 19 | `signal_router skip existing_schwab_position` | LIVE; `get_schwab_positions()` already holds symbol → registers it as `managed_by_bot=False` and skips | — | — | 7067 |
| 20 | `signal_router skip already_tracking` | symbol in `positions` or `simulated_positions` | — | — | 7083 |
| 21 | `signal_router skip anti_pyramid_pending_order_for_symbol` | any `pending_orders` entry for the symbol | — | — | 7100 |
| 22 | `signal_router skip anti_pyramid_recent_attempt` | last accepted attempt < 240 min ago; attempt timestamp is stamped *before* ordering | — | cleared on cancel/reject/broker-flat | 7129 |
| 23 | `loss_cooldown skip recent_loss` | 60 min after a stop-loss on the symbol | — | — | 7157 |
| 24 | `reentry_cooldown skip recent_exit` | 5 min after any exit | — | — | 7184 |
| 25 | `shortable_check skip not_shortable` | SELL that is not a close; Alpaca shortable lookup with `assume_shortable_on_error=False` | — | fail-closed inside checker; a raised exception falls through (`info=None` → proceed) | 7220 |
| 26 | `max_positions_gate skip limit_reached` | distinct(managed_by_bot positions with qty>0 ∪ `PENDING` pending_orders) ≥ cap | `MAX_POSITIONS` (10) | hands-off/external don't consume slots | 7347 |
| 27 | `mean_rev_budget_gate skip` | mean-rev family: denylist symbol; concurrent mean-rev ≥ `MAX_CONCURRENT_MEAN_REV`; shorts ≥ `MAX_CONCURRENT_MEAN_REV_SHORTS`; notional/equity ≥ `MAX_MEAN_REV_RISK_PCT` | `ENABLE_MEAN_REV_RISK_BUDGET` (True) | — | 7383–7500 |
| 28 | `correlation_guard skip correlated_with_<group>` | symbol shares a hard-coded sector set with an open same-mode position; external and hands-off holdings ignored | `CORRELATION_IGNORE_UNMANAGED`, `CORRELATION_IGNORE_HANDS_OFF` (True) | — | 7530 |
| 29 | `risk skip no_buying_power` | LIVE; `_check_buying_power` < `MIN_BUYING_POWER` ($100) | — | fail-closed | 7562 |
| 30 | `signal_router cancel_and_retry / skip existing_orders` | LIVE; open orders on symbol | `AUTO_CANCEL_EXISTING_ORDERS` | — | 7585 |
| 31 | `brain skip experience_says_no` | `brain.should_take_trade` (see §9) | — | — | 7612 |
| 32 | `ml skip ml_veto_high_confidence` / `ml proceed ml_disagreement_low_conf` | ML disagrees with side and ML confidence ≥ `ML_VETO_CONFIDENCE` (0.65) → skip; otherwise confidence ×0.7 and proceed | `ML_PREDICTION_ENABLED` (True) | advisor below threshold | 7697 / 7724 |
| 33 | Against daily trend | 20-day close direction opposes signal → confidence ×0.7 | — | soft | 7741 |
| 34 | Professional filter | `pro_trading_wrapper.evaluate_trade_signal`; may block, or adjust stop/size | — | fail-open on exception | 7774 |
| 35 | Concentration | `size × entry > MAX_POSITION_VALUE_PCT (0.25) × open book value` → shrink | — | — | 7856 |
| 36 | Sizing | `calculate_position_size_with_commentary` returns 0 → silent return | — | — | 7862 |
| 37 | `signal_router accepted` (+ `rr_audit_low` warning if R:R < 1.8) | — | — | — | 7896 |
| 38 | Sim: `_validate_oco_prices` → `reject_oco_invalid_sim`; LIVE: `_execute_real_trade` | — | — | — | 7949 / 7985 |

Net effect today in LIVE (updated 2026-09-28 13:17 ET, §9): gates 13, 17 and 18 no longer block — `DAY_TRADE_LIVE_ENTRIES_ENABLED`, `MEAN_REV_LIVE_ENTRIES_ENABLED`, and `ORB_LIVE_ENTRIES_ENABLED` are all on in `.env`. The playbook gate (upstream of these, between gate #1 and #2) is now the brake for two of them: it suspends every `dt_momentum|long` entry outright (day-trade momentum long effectively still cannot open a position) and halves `mean_rev_intraday|long` size via probation; day-trade momentum short and ORB have no playbook entry yet and trade at full size once they clear their own gates. Free-news remains ungated by any pause flag. Mean-reversion long must additionally pass the regime, conviction, quality, risk-off, dedupe and budget gates upstream.

#### Strategy fit gate

`core/strategy_fit_gate.py`, called from around `core/engine.py:6068`, sits between gate #1 (regime gate) and gate #2 (conviction floor) in the table above — after the mean-rev regime veto, before the conviction-floor check. It runs `evaluate_fit_gate` against `self._strategy_fit`, an in-memory reader over the offline strategy-fit snapshot (`STRATEGY_FIT_PATH`, default `data/strategy_hunt/fit/strategy_fit.json`, reloaded periodically at `core/engine.py:5587`) produced nightly by `research.strategy_hunt` (see `docs/research/strategy_hunt.md` for the cron schedule).

The gate only governs strategies mapped to a fit family in `LANE_MAP` (`core/strategy_fit_gate.py`: `mean_reversion`/`mean_reversion_short`/`oversold_bounce_v2` → `mean_rev_intraday`, `day_trade_momentum`/`day_trade_momentum_short` → `dt_momentum`, `orb_contraction_rvol` → `orb`, `news_sentiment`/`free_news_sentiment` → `news_catalyst`, plus any `fit:<family>` lane strategy). A signal from an ungoverned strategy passes through untouched (`fit_not_governed`, not audited). Every key is the exact `reasoning['strategy']` value a live strategy writes — `strategies/builtin.py` (`mean_reversion`, `mean_reversion_short`, `day_trade_momentum`, `day_trade_momentum_short`, `orb_contraction_rvol`, `news_sentiment`), `strategies/oversold_bounce_v2.py` (`oversold_bounce_v2`), `strategies/news_strategy.py` (`free_news_sentiment`) — and `tests/core/test_strategy_fit_gate.py` greps `strategies/*.py` to keep it that way. Two earlier placeholder keys (`oversold_v2`, `news`) matched no strategy at all and governed nothing; they were removed 2026-09-25.

Flags (`core/config.py`, search `v-strategy-fit-2026-09-25`; `STRATEGY_FIT_REGIME_ROWS`
and the two `SENTIMENT_*` flags are `v-playbook-desk-b1-2026-09-25`):

| Flag | Default | Effect |
|---|---|---|
| `ENABLE_STRATEGY_FIT_GATE` | `False` | Master switch. `False` → gate is fully inert, no audit lines. |
| `STRATEGY_FIT_SHADOW` | `True` | When the gate is enabled, blocked verdicts are logged but never skip the signal. Must be explicitly `False` to let the gate skip entries. |
| `STRATEGY_FIT_PATH` | `data/strategy_hunt/fit/strategy_fit.json` | Snapshot file read by the gate and by `GET /api/strategy-fit`. |
| `STRATEGY_FIT_MAX_AGE_DAYS` | `7` | Snapshot age (days) beyond which it's treated as stale. |
| `STRATEGY_FIT_REGIME_ROWS` | `True` | Gate consults the regime-conditioned rows for the live market-context regime (symbol-regime row → universe-pooled regime row → unconditional row) instead of the unconditional row alone. `0`/`false` restores the pre-B1 lookup. |
| `SENTIMENT_BACKEND` | `vader` | Sentiment scorer for both live news scoring and the research news cache (`core/sentiment_backend.py`). `finbert` uses ProsusAI/finbert on GPU, falling back to VADER with a warning if it can't load. |
| `SENTIMENT_FINBERT_PATH` | `models/finbert` | Local FinBERT checkpoint directory, built once by `python scripts/prepare_finbert.py`. |
| `FIT_LANES_ENABLED` | `""` | Family names to register live fit-lane adapters for; empty means none registered. |
| `FIT_LANE_LIVE_ENTRIES_ENABLED` | `""` | Family names allowed LIVE entries via their fit lane; absent families stay shadow-only. |
| `SWING_ALLOW_OVERNIGHT` | `False` | Allow swing-lane positions to hold across sessions instead of forced flat by EOD. |
| `SWING_MAX_HOLD_SESSIONS` | `10` | Max sessions a swing-lane position may be held before a forced exit. |
| `STRATEGY_HUNT_DATA_ROOT` | `data/strategy_hunt` | Root of the nightly research data tree; the engine splices its `bars5/`/`daily/` Parquet under the broker's 10-day minute window (so `rvol`/`or*_range_avg20` resolve live) and reads `earnings/` for swing exits. |

`ENABLE_STRATEGY_FIT_GATE` and `STRATEGY_FIT_SHADOW` are both effectively off in their protective sense today (gate disabled; and even if enabled, shadow defaults on) — the gate ships inert and must be turned on deliberately in two steps: enable, then disable shadow.

**What step two costs today.** The current fit snapshot (full universe, 2026-09-25, Playbook Desk B1.6 rerun under entry-time regime labels) has **0 PASS rows** — none of its 9,227 unconditional rows, none of its 1,625 symbol-regime rows, none of its 67 universe-pooled regime rows, so `STRATEGY_FIT_REGIME_ROWS` changes nothing about this either (`research/reports/strategy_hunt/2026-09-25T1557-strategy-hunt.md`, narrated in `research/reports/strategy_hunt/2026-09-25-strategy-hunt-b1-5.md` § "B1.6 — regime label fix"). B1.6 also withdrew B1/B1.5's "regime split is real and large" finding: under labels the desk could actually have known at entry, `dt_momentum short`'s risk_off-minus-risk_on expectancy spread goes from +0.63 R to 0.00 R. So turning shadow off right now makes **every governed lane skip every signal** — `fit_fail` where a row exists, `fit_missing_row` where it doesn't — i.e. it switches off `mean_reversion`, `mean_reversion_short`, `oversold_bounce_v2`, `day_trade_momentum`, `day_trade_momentum_short`, `orb_contraction_rvol`, `news_sentiment` and `free_news_sentiment` entirely. State that plainly to anyone asking for enforcing mode: it is a deliberate "governed lanes off until evidence exists" decision, not a tuning knob. Until a nightly run produces PASS rows for the symbols actually traded, shadow mode is the only setting in which those lanes keep trading. `research/strategy_hunt promote` reports the same posture from the other side via its `gate_not_enforcing` blocker (see `docs/research/strategy_hunt.md` § "Operator flow").

Audit line (component `strategy_fit`, one per governed signal):

```
engine_decision component=strategy_fit symbol=SYM action=proceed|shadow|skip reason=REASON mode=MODE family=... verdict=... score=... exp_r=... stability=... source=... age_days=...
```

`action` is `proceed` (verdict PASS and no higher-scoring competing family also passing), `shadow` (every block reason — `fit_watch_shadow`, `fit_fail`, `fit_missing_row`, `fit_missing`, `fit_stale`, `fit_lower_priority` — logs as `shadow` when `STRATEGY_FIT_SHADOW` is on), or `skip` (the same block reasons log as `skip` and block the signal when the gate is enabled and `STRATEGY_FIT_SHADOW` is off). `reason` is one of `fit_pass`, `fit_watch_shadow` (WATCH verdict), `fit_fail` (FAIL verdict), `fit_missing_row` (snapshot loaded but no row for this symbol/family/side), `fit_missing` (snapshot never successfully computed), `fit_stale` (snapshot older than `STRATEGY_FIT_MAX_AGE_DAYS`), or `fit_lower_priority` (a competing governed family has a higher-scoring PASS on the same symbol/side this pass) — `fit_lower_priority` uses the same shadow/skip helper as every other block reason, with no carve-out.

**Regime-aware lookup (2026-09-25, Playbook Desk B1; precedence and labels hardened in B1.6).** When `STRATEGY_FIT_REGIME_ROWS` is `True` (the default) the engine passes the signal's `market_context_regime` into the gate, which then asks the snapshot for the row measured *in that regime* — the symbol's own regime row, else the universe-pooled row for that regime (`symbol="*"`, `source=pooled`) **but only when the symbol's own unconditional row for that lane is absent or is not FAIL**, else the unconditional row, else `fit_missing_row`. A pooled PASS may fill a gap or stand in front of a WATCH; it never overrules the pipeline's own measured FAIL for that symbol and lane. The audit line gains `regime=<risk_on|risk_off|mixed>` and `pooled=True|False`; any other regime value (or none) means "no regime evidence", the `regime` field is omitted and the unconditional row governs exactly as before. Nothing about shadow/enforcing changes: a regime row can only decide *which* verdict the gate reads, never whether it blocks. Sample floors are in `docs/research/strategy_hunt.md` § "Regime and pooled rows" (a regime needs ≥ 60 trades for a symbol row; pooled rows use the stricter `POOLED_FLOORS`).

The research side labels each historical trade with the regime that was **knowable at its entry bar** (B1.6): SPY's day-to-date move through the last 5-minute SPY bar that closed at or before the entry bar's open, plus the *prior* session's VIX close and close-to-close change, run through the same thresholds as `core/market_context.py::_classify_regime`. Before B1.6 the label came from the entry day's SPY/VIX closes, so a regime row was fitted to the rest of the day the desk had not yet seen. The VIX pair is a deliberate lagged proxy (the research store holds daily VIX only), which makes the research label slightly blunter than the live one on days when VIX turns intraday — never more informed.

Fail-closed behavior: a stale or missing snapshot blocks live entries — but, consistent with shadow mode generally, only when `ENABLE_STRATEGY_FIT_GATE` is `True` and `STRATEGY_FIT_SHADOW` is `False`; otherwise it only logs. The gate never flips any lane's own `*_LIVE_ENTRIES_ENABLED`/shadow flag (e.g. gate #17 `MEAN_REV_LIVE_ENTRIES_ENABLED`) — it is an additional check layered on top of, not a replacement for, each lane's existing pause flag.

#### Desk (shadow)

Playbook Desk B2 (2026-09-25; full writeup in `docs/desk.md`). Two more router
hooks bracket the strategy fit gate above, both gated on `ENABLE_DESK`
(`core/config.py` § `v-playbook-desk-b2-2026-09-25`, default `False` — with it
off neither hook body runs and no `desk_loop` is registered):

- **`desk_limit`**, immediately *before* the strategy-fit gate (which itself
  sits between gate #1 regime gate and gate #2 conviction floor in the table
  above — so the live order is now regime gate → `desk_limit` →
  strategy-fit → `playbook` → conviction floor): checks the day's
  `DailyPlan` (built pre-market by `core/loops/desk_loop.py`) for a sit-out
  day, a `max_trades` breach, or a direction the plan disallows.
  `ENABLE_DESK_LIMITS` (`False`) turns a match from `action=shadow`
  (audited, not blocking) into `action=skip` (blocking).
- **`playbook`**, immediately *after* the strategy-fit gate: checks the
  adaptive playbook (`data/desk/playbook.json`, written nightly by
  `research.strategy_hunt playbook` — see `docs/research/strategy_hunt.md`)
  for the signal's `(family, side, regime)` key. `ENABLE_PLAYBOOK_GATE`
  (`False`) turns a `suspended` key's match into `action=skip` and writes
  its `size_mult` into `signal.reasoning["strategy_size_multiplier"]`,
  which `risk/manager.py::playbook_size_multiplier` then applies (clamped
  `[0.0, 1.5]`) right after the per-strategy multiplier in §3.3 below.

Both hook bodies are wrapped in `try/except` with a once-a-minute warning
(`_desk_hook_warn`) — a desk failure can never block or resize a signal by
accident. Separately, a `DeskLoop` (registered under `ENABLE_DESK` in
`_build_supervisor`) builds the day's plan at 09:00 ET, ranks an in-play
list at 09:35, and writes a post-close journal at 16:05 — see `docs/desk.md`
§ "Timeline". With `ENABLE_DESK_WATCHLIST` (`False`) also on, the 09:35
in-play list drives `engine.dynamic_watchlist` instead of the screener's own
output (`core/loops/screener_loop.py::_desk_owns_watchlist`).

**Current status: none of the four flags are set** (absent from `.env`, so
all read their code default of `False`) — the desk is not running at all
yet. Enabling it in shadow (`ENABLE_DESK=1` only, all three sub-flags left
unset) is Task 8 of the Playbook Desk B2 plan
(`.superpowers/sdd/2026-09-25-playbook-desk-b2/`), to be done after the
close. **Research verdict**: no family has a walk-forward PASS in any
regime as of the latest full-universe run (see "What step two costs today"
above) — so `ENABLE_DESK_LIMITS`/`ENABLE_PLAYBOOK_GATE` would today only
enforce defensive risk rules (spend less / trade less in a worse regime),
never a researched edge; `docs/desk.md` § "How a lane would be promoted
from shadow" states this in full.

**Launch fact (verified 2026-09-25 16:14 ET).** `deploy.sh`'s pre-flight
currently aborts against the present `Config().yaml` layout (§7.5 already
notes it "asserts launch dials that differ from the current yaml"), so on
this host the bot is launched with plain `python
trading_bot_commentary_updated.py`, then trading is started by hand via
`POST /api/start` followed by `POST /api/toggle-mode {"mode":"live"}` —
not `./deploy.sh`, notwithstanding §5.6/§5.7's description of it as "the
sanctioned launcher". Any restart to pick up `ENABLE_DESK=1` (Task 8) uses
this same plain-`python` path.

### 3.3 Sizing

`RiskManagerWithCommentary.calculate_position_size_with_commentary` (`risk/manager.py:66`):

```
risk_per_share = max(|price − stop_loss|, 0.005 × price)          # 0.5% floor (manager.py:98)
max_risk       = account_balance × RISK_PER_TRADE_PCT (0.015)
                 × 1.3 if last-5 WR ≥ 0.8, × 0.5 if ≤ 0.2          # reads self.trade_history, which the engine never sets → inert
base           = int(max_risk / risk_per_share)
quality        = 0.5 × strength + 0.5 × confidence                  # SIZING_STRENGTH_WEIGHT
kelly          = clamp(quality − (1 − quality)/ATR_REWARD_RISK_RATIO(2.0), KELLY_FLOOR 0.50, 1.0)
kelly          = min(kelly, 0.20 | 0.35 | 0.60) if strength < 0.25 | 0.45 | 0.70
shares         = base × kelly
               × STRATEGY_SIZE_MULTIPLIER(strategy)
               × market_context_conviction (0.4–1.6, ENABLE_MARKET_CONTEXT_SIZING)
               × news_gate_multiplier (0.1–1.0)
               × day_trade_size_multiplier (0.1–1.0, day trades)
               × LIVE_SIZE_MULTIPLIER (1.0)
cap: shares × price ≤ min(MAX_POSITION_VALUE 25,000, buying_power × 0.20); ≤ 95% of buying power
shares < MIN_POSITION_SIZE (1) → 0
```

It logs one `position_sizing symbol=… shares=… dollar_risk=… kelly=…` line and stores `confidence` and `kelly_fraction` into `signal.reasoning` so the close path can persist them (`manager.py:310`).

Two sizing modules are shadow-only. `sizing/conviction_sizer.py:56` computes `score = 0.5·meta + 0.125·(ml_agrees + allocator_allows + |direction|/10 + (volume_ratio−0.5)/2)` and maps `[0.3, 0.8]` linearly onto a `0.5–1.5×` multiplier; the engine logs it (`conviction_sizer_shadow.ndjson`) but never applies it. `allocators/regime_allocator.py:57` computes the Kaufman efficiency ratio of SPY's last 20 daily closes (cached 4 h); ER ≥ 0.30 → breakout on / mean-rev off, else the reverse, fail-open on missing data. Its only live use is gate #1.

### 3.4 Order execution

`_execute_real_trade` (`engine.py:1028`): validate stop/target against the live bid/ask (`_validate_oco_prices`, `engine.py:1915`, returns True if no quote), check `risk_manager.buying_power ≥ size × ask`, refuse closed sessions. In regular hours it sends a **market** order (`equity_buy_market` / `equity_sell_short_market`); in extended hours a **limit** at `entry × (1 ± LIMIT_ORDER_SLIPPAGE 0.001)`. Duration DAY. On HTTP 200/201 the order id is read from the `Location` header and recorded in `pending_orders[order_id] = {symbol, signal, status: PENDING, placed_time}` (`engine.py:1175`).

`_verify_order_fill` (`engine.py:2097`) polls `get_order` every 2 s up to 30 times (60 s), does one final check, then cancels; REJECTED/CANCELED/timeout clear the anti-pyramid stamp. A FILLED result creates `Position(managed_by_bot=True, managed_source='bot', entry_price=signal.entry_price)` — the signal's price, not the execution price — pops the pending entry, audits `order_fill position_created`, and calls `_place_bracket_orders` (`engine.py:1806`). The bracket is a Schwab OCO of a GTC sell LIMIT at `take_profit` and a GTC STOP_LIMIT with stop `stop_loss` and limit `stop_loss × 0.995`; both legs use `equity_sell_limit`, so the bracket as coded is long-side only. The OCO id is stored on `position.bracket_order_id` / `broker_stop_price`. `_check_order_status` (`engine.py:2187`, every 10 s from the analysis loop at `:5162`) is the fallback sweep: it creates the Position from the actual fill price if the immediate path missed it and drops CANCELED/REJECTED/EXPIRED orders.

In `SIMULATION_WITH_COMMENTARY` the same OCO validation runs, then a `Position(mode="simulation", managed_by_bot=True)` is placed straight into `simulated_positions` (`engine.py:7957`); nothing touches the broker.

### 3.5 Order monitor (`core/order_monitor/`)

`OrderMonitor.run` (`monitor.py:60`) is a CRITICAL supervised task on a 5 s cadence (`ORDER_MONITOR_INTERVAL_SEC`). It monitors only positions with `managed_by_bot=True`, a `bracket_order_id`, not `is_external`/`is_manually_managed`/`is_long_term`, and not in `HANDS_OFF_DENYLIST` (MU, HQGE, SPCX). Once per process it runs `bootstrap_orphan_brackets` (`monitor.py:203`) to attach WORKING OCOs found at Schwab to managed positions missing bracket ids (exact-qty match preferred; ambiguity → `bootstrap_oco ambiguous_skip` and a HIGH alert).

Each tick fetches the last 7 days of WORKING orders. If the parent OCO is still WORKING, `check_trail_replace` (`brackets.py:93`) replaces the broker stop when the software `trailing_stop` is tighter than `broker_stop_price` (cancel OCO → new OCO). Otherwise `handle_bracket_not_working` (`brackets.py:281`) queries the order:

- **FILLED** → `handle_bracket_fill`: identifies the leg (STOP_LIMIT → stop, LIMIT → take_profit). Full fill → `handle_full_bracket_fill`: short-circuits any pending close confirmation, appends a `trade_history` record with `exit_reason=stop_loss_broker|take_profit_broker`, bumps/clears `consecutive_losses`, deletes the position, `_save_state`. (No `db_logger.log_trade` or `brain.remember_trade` in this handler.) Partial fill → qty reduced and `re_bracket_position`.
- **CANCELED/EXPIRED** → `handle_bracket_canceled`: with `REBRACKET_SKIP_IF_FLAT_OR_EXITING` (True) it skips re-bracketing when the FSM is EXITING/CLOSED/ZOMBIE, local qty is 0, or Schwab reports flat (`rebracket_skipped_exit_in_flight`); otherwise re-brackets.
- **REJECTED** → `handle_bracket_rejected`: if the reason matches oversold/overbought/insufficient shares (`broker_flat.py:34`) and the broker shows qty 0, `handle_broker_flat_detected`; else re-bracket, capped at `MAX_RE_BRACKET_ATTEMPTS=3` (`re_bracket_exhausted`).

`re_bracket_position` (`brackets.py:818`) first checks broker qty (0 → broker-flat path; mismatch → local qty synced to broker), then places a new OCO at `trailing_stop or stop_loss`. `handle_broker_flat_detected` (`broker_flat.py:90`) sets `managed_by_bot=False`, cancels working orders, clears bracket ids, clears the anti-pyramid stamp (`CLEAR_PYRAMID_LOCK_ON_BROKER_FLAT`, True), and only removes the position if `ENABLE_GHOST_FLATTEN_AFTER_BROKER_FLAT` is True (default False → alert-only "ghost position").

### 3.6 Position state machine and exits

`core/position_state.py` defines the FSM: `OPENING → LIVE → AT_BREAKEVEN → AT_1R → TRAILING`, any managed state → `EXITING` or `CLOSED` (broker-confirmed flat) or `ZOMBIE`; `EXITING → CLOSED|ZOMBIE`; `ZOMBIE → CLOSED` (operator). `try_transition` (`position_state.py:208`) holds a per-position `asyncio.Lock`, rejects illegal moves with `position_fsm transition_rejected`, and stamps `exiting_started_at`. `ALLOWED_EXITS` (`position_state.py:75`) restricts which rules may run per state; PROACTIVE_EXIT is absent from AT_1R/TRAILING. `TIME_STOP` is declared in the table but no engine code evaluates it.

A position is auto-managed only if `_is_auto_managed` (`engine.py:741`): not `is_long_term`, `managed_by_bot=True`, and not (LIVE and `MANUAL_CLOSE_ONLY`, default False).

Exits are evaluated on two loops. The fast `_position_loop` (`engine.py:3528`, `POSITION_LOOP_SEC` 1.0) runs `_evaluate_exits_with_cached_quote` (`engine.py:3279`) against the PriceBook mark (stale mark → skip). The slower analysis loop runs `_manage_positions_with_commentary` (`engine.py:8067`) which repeats the ladder with fresh indicators and adds the discretionary rules, then calls `_evaluate_exit_conditions` (`engine.py:8745`).

| Rule | Trigger | Where | Live or shadow |
|---|---|---|---|
| Zombie watchdog | EXITING > `EXITING_ZOMBIE_SEC` (60 s, `engine.py:2991`) → ZOMBIE, hands-off | `_promote_zombie_if_stalled` 3073 | live |
| Hard stop | price ≤ stop (long) / ≥ stop (short) → EXITING → close `stop_loss`; sets 60-min loss cooldown | 3335, 8830 | live |
| Take profit | price crosses `take_profit` → close `take_profit` | 3362, 8858 | live |
| Breakeven lift | peak R ≥ `BREAKEVEN_ACTIVATION_R` (0.5) → `stop_loss = entry`, → AT_BREAKEVEN | 3392; `scale_trail_manager.py:147` | live (software stop only; broker stop updated only via trail replace) |
| Scale-out 1R | +1R (or `scale_out_r_override`) → qty −50%, stop→entry, → AT_1R | 3426 | **bookkeeping only; LIVE skipped unless `ENABLE_LIVE_SCALE_OUT` (False)** — audit `scale_out_skipped live_partials_not_armed` |
| ATR trail | activates at entry ± 2.0×ATR (`TRAIL_ACTIVATION_ATR_MULT`), trails 1.5×ATR (`TRAIL_WIDTH_ATR_MULT`), ratchets; breach → close `trailing_stop_atr` | 3480; `scale_trail_manager.py:95` | live; order monitor mirrors the trail to the broker stop |
| Proactive exit | pnl ≤ −0.5R (−0.7R news) and MACD flip / RSI through 50 / ADX collapse; only after per-strategy min age (news 30, mean-rev 10, default 15 min; day-trade via env) unless pnl ≤ `PROACTIVE_EXIT_R_OVERRIDE_THRESHOLD` (−0.3R) | 8543; `scale_trail_manager.py:194` | live (`ENABLE_PROACTIVE_EXIT` True) |
| Thesis re-validation / news thesis flip | verifier re-check; news flip closes | 8604, 8634 | revalidate is shadow by default; news flip live |
| Unified exit (day-trade) | scale/breakeven/trail/time-stop policy | `core/unified_exit.py:185`, engine 8517 | **shadow only** (`DT_UNIFIED_EXIT` False; `should_override_proactive_exit` always False) |
| Pro exit ladder | `pro_trading_wrapper.update_positions` scale-outs | 8649 | live when wrapper present |
| Day-trade flatten hour | `is_day_trade` and ET hour ≥ `DAY_TRADE_FLATTEN_HOUR` (15) → `day_trade_flatten_hour` | 8765 | live |
| Extended target / dynamic trail | `DynamicExitManager.evaluate_exit` (`analysis/exit_managers.py:155`, long-only): +3% → `take_profits_extended`; +1% → stop = entry+0.2%; +1.5% → 1.5×ATR trail; +2% with RSI>75 → exhaustion | 8880 | live |

`smart_target.py` is not an exit rule: strategies call `compute_smart_target` (`core/smart_target.py:152`) at signal time to pick the nearest destination (BB upper, 20-bar high, half 20-bar range, R:R fallback) with R ≥ 1.5, or skip the trade (`ENABLE_SMART_TAKE_PROFIT`).

`_close_position_with_commentary` (`engine.py:8907`) refuses external / manually-managed / long-term / denylist positions unless `reason == "manual_override"`, and refuses when `auto_close_disabled`. In LIVE with `ENABLE_BROKER_LEG_AUTHORITY` (True) it first asks `is_broker_leg_authoritative` (`broker_flat.py:246`): a WORKING or FILLED bracket skips confirmation. Otherwise `REQUIRE_CLOSE_CONFIRMATION` (True) routes through `_get_close_confirmation` (`engine.py:9418`): a dashboard request that waits `CONFIRMATION_TIMEOUT_SEC` (30 s); on timeout the `supervised` profile denies (state reverted to the ladder state, `state_reverted confirmation_denied`) while `autonomous_live` executes. Confirmation is skipped for profits when `CONFIRM_ONLY_LOSSES`, or when |P&L%| < `CONFIRM_THRESHOLD_PERCENT` (5). The close then takes a quote (3 s executor timeout), computes P&L, calls `brain.remember_trade`, sets the 5-min re-entry cooldown, transitions to EXITING, and in LIVE calls `_close_real_position` (`engine.py:2324`): cancel the bracket, send a market close (limit ±0.1% off-hours), record in `pending_orders`, and `_verify_order_fill`. A failed close promotes the position to ZOMBIE (`broker_close_returned_false`). Success bumps `consecutive_losses` (live closes only), moves EXITING → CLOSED, pops the position, appends `trade_history`, and writes the DB row (§8). At startup `_recover_inflight_exits` (`engine.py:3008`) resolves persisted EXITING positions: gone at Schwab → CLOSED, still held → ZOMBIE, sim → back to LIVE.

### 3.7 Position sync with Schwab

Startup (`engine.py:4765–4854`): `_update_and_track_real_positions` (`engine.py:9902`), `_bootstrap_orphan_brackets`, `_recover_inflight_exits`. Every LIVE analysis cycle: `_check_order_status` and `_update_real_positions` (`engine.py:2494`, which delegates discovery to `_update_and_track_real_positions`). Every fifth cycle (`analysis_count % 5`, `engine.py:4969`): `sync_positions_with_schwab` (`engine.py:4472`).

Ownership rules: `HANDS_OFF_DENYLIST` symbols are never restored to `managed_by_bot=True`. A discovered position is claimed as bot-managed if a pending bot order matches (`discovery_rescue`), or if `ENABLE_MANAGED_BY_BOT_PERSIST` (True) and `_ownership_evidence` (`engine.py:4382`) finds it in saved state meta, today's `bot_trades`, `bot_decisions`, or `bot_positions` (`ENABLE_MANAGED_OWNERSHIP_EVIDENCE_BROAD`, True). Saved `managed_by_bot=False` is honored as an operator decision only when `managed_source='operator'`. Anything else becomes `managed_by_bot=False, is_external=True, is_manually_managed=True` with a default 5%/10% stop/target for display (`position_left_external no_bot_record`). Symbols that vanish from Schwab go through `_reconcile_external_closes` (`engine.py:4138`): match today's SELL/BUY_TO_COVER fills, compute P&L against the prior entry, write a broker-orphan / `external_close` trade row, feed `brain.remember_trade`, then drop the position.

### 3.8 P&L, ledger and persistence

`core/pnl.py:23` picks per-position day P&L: Schwab's `currentDayProfitLoss` unless its magnitude exceeds 50% of market value (outlier), then `netChange × qty`. `_compute_bot_daily_pnl` (`engine.py:9592`) sums today's realized `trade_history` entries flagged bot-managed plus unrealized P&L of managed positions, excluding external/manual/long-term/denylist; it feeds `risk_manager.bot_daily_pnl` for the circuit.

Trade rows: `trade_history` (in-memory, last 100 persisted to `trading_state.json`), plus Postgres `bot_trades` via `db_logger.log_trade` (`db_logger.py:462`, columns include strategy, exit_reason, atr_at_entry, confidence, meta_proba, kelly_fraction, scaled_out). With `LEDGER_INTEGRITY` (default False, requires migration) the close path builds a `TradeLedgerEntry` (`core/ledger_integrity.py:419`) and calls `log_trade_v2`; `compute_r_multiple` (`ledger_integrity.py:157`) derives R from `initial_stop` → `stop_loss` → `1.5×ATR`, exit reasons are normalized to a fixed vocabulary, sources are `bot|broker_orphan|external`, and `is_bot_managed_trade` excludes hands-off and external rows from stats. Open positions are mirrored to `bot_positions` by `db_logger.sync_positions` from `_save_state` (`engine.py:2983`). Pending close requests, bracket ids and FSM fields survive restarts through `trading_state.json` (`_state_persistence_loop`, `engine.py:3189`).

### 3.9 Brain

`TradingBrain` (`core/brain.py:25`, `trading_brain.json`) is fed by `remember_trade` (`brain.py:148`) from both the bot close path (`engine.py:9163`) and external-close reconciliation (`engine.py:4354`). It increments per-strategy win/loss counts, appends a memory with a canned lesson, and moves the emotional state: win → confidence +0.02, greed −0.05, fear −0.02; loss → confidence −0.03, fear +0.05; five bad outcomes raise patience and cut risk appetite. `recover_confidence` decays confidence toward 0.7 at 1%/h and greed/fear toward 0.5 at 20%/h; `reset_for_new_session` (called at engine start, `engine.py:394`) pulls greed/fear 60% toward neutral.

Its only feedback into decisions is gate #31, `should_take_trade` (`brain.py:283`): refuse if the symbol's last-20-memory win rate < 20% over ≥ 3 trades; refuse setups with confidence < 0.6 when fear > 0.9 or greed > 0.7; refuse if `(signal.confidence + pattern_confidence)/2 < 0.6 × emotional confidence`, where `pattern_confidence = strategy win-rate × confidence` (0.5 with fewer than 3 samples). Risk appetite and patience are stored but not read by the sizing or exit code.

## 4. Data, machine learning, and research tooling

### 4.1 Data providers (`data_providers/`)

| Provider | File | Used by | Notes |
|---|---|---|---|
| Schwab REST | `data_providers/schwab.py:26` `SchwabDataProvider` | **Live** engine | `get_market_data()` (`schwab.py:35`) defaults to 1 day of 5-min candles via `get_price_history_every_five_minutes`; 60 s in-memory cache; wrapped in `@api_circuit_breaker`. Also `get_quote()` (`:170`), `calculate_market_breadth()` (VIX), `_get_movers()` for the screener. |
| Schwab stream | `core/schwab_stream.py:59` `SchwabQuoteStream` | **Live** quotes | Single `LEVELONE_EQUITIES` websocket subscription feeding the shared `QuoteCache`; dynamic sub/unsub diffing, exponential backoff, and fallback to REST polling after 5 failures. Registered in the engine supervisor at `core/engine.py:3764`. |
| yfinance fallback | `data_providers/realtime.py:11` `RealTimeDataProvider` | Simulation / no-token | Selected at `core/engine.py:4123` and `:4136` when the Schwab token is missing or invalid; 30 s bar cache, 15 s quote cache. |
| Alpaca ingest | `data_providers/alpaca_ingest.py` + `ingest_alpaca_bars.py` | **Training / backtest** | Writes 1-min bars into Postgres `minute_bars` with `INSERT ... ON CONFLICT (symbol, ts) DO NOTHING` (`alpaca_ingest.py:5,267`); checkpointed (`alpaca_ingest_checkpoint.json`), SIP/IEX feed via `ALPACA_DATA_FEED`. Cron: `0 21 * * 1-5 ... ingest_alpaca_bars.py --top-n 150 --days 7 --feed sip`. |
| Postgres bars | `data_providers/postgres.py:28` `PostgresDataProvider` | **Training / backtest** | Read-only; queries `minute_bars`, resamples 1-min to requested frequency, returns `Open/High/Low/Close/Volume`. Sync SQLAlchemy engine. |
| File bars | `data_providers/file_bars.py` | Backtest / cloud | Parquet/CSV per symbol under `BARS_DIR`; same `BarsLoader` signature as Postgres. |
| Yahoo screener | `analysis/screener.py:385,504` | Live universe | Undocumented `query1.finance.yahoo.com/.../screener/predefined/saved` (`most_actives`, `day_gainers`, `day_losers`) prepended to Schwab movers; yfinance batch bars for quality filters (`:714`), cached 30 min. |
| News | `strategies/news_strategy.py:90` RSS (Yahoo, Google, Investing, MarketWatch) via `feedparser` with pre-fetched bytes and hard timeouts; `analysis/news_verifier.py:37` Alpaca news API | Live | `core/news_bus.py` is the single in-process store; only the news loop writes it (~20 s cadence). |

**`db_logger` (`data_providers/db_logger.py:109`)** is the live write path to Postgres. It creates a SQLAlchemy `AsyncEngine` (asyncpg) with `pool_size=5, max_overflow=3, pool_pre_ping, pool_recycle=1800` (`:129`), defaulting to `POSTGRES_DSN_ASYNC` or the hard-coded `rudra_dev` DSN (`:64`). It is a module singleton (`get_shared_db_logger`, `:74`) bound to an owner event loop; cross-loop calls are routed with `run_coroutine_threadsafe` when `DB_LOGGER_CROSS_LOOP_ROUTE` is true (`:60`). Tables it writes: `bot_decisions` (`:434`), `bot_trades` (`:538`, `:1640` v2 with ledger-integrity columns), `bot_positions` (DELETE+INSERT under a lock, `:625`), `bot_shadow_news_vetoes` (`:736`, later re-evaluated at `:800`), and `bot_decision_snapshots` (created if missing at `:1411`, written at `:993`). `log_strategy_decision` (`:1712`) is a sync fire-and-forget shim used by `gpu_news_critic`/`theme_shock_logger` that maps onto `bot_decisions`.

**`MINUTE_BARS_INGEST_INCLUDE_TRADED`** (`core/config.py:4027`, default `True`; env override or yaml `trading.minute_bars_ingest_include_traded`). When on, `ingest_alpaca_bars.py` unions the top-N-by-volume universe with symbols the bot traded or signalled that ET day, resolved by `data_providers/ingest_universe.py:56 traded_symbols_for_day` from `bot_trades`, `bot_positions`, `bot_decisions` (filtered by `MINUTE_BARS_INGEST_TRADED_ACTIONS`, which excludes `shadow_pass`), `bot_decision_snapshots`, and best-effort `signals_intraday`. Each source query is fault-tolerant. CLI flags `--include-traded/--no-include-traded/--traded-date` (`ingest_alpaca_bars.py:97-112`). The flag only affects data collection; the ingest is idempotent, so off-path behaviour is identical to the legacy top-N-only ingest.

### 4.2 Live ML path

- **Predictor**: `ml/predictor.py:16` `MLPredictorWithCommentary` wraps `IntegratedMLModel`. `predict_with_commentary(indicators, symbol, market_data)` (`:36`) uses the raw OHLCV DataFrame if present, else a rule fallback (RSI<30 & MACD cross = 1, RSI>70 = -1, else 0 at 0.65/0.5 confidence, `:78`).
- **Features**: `ml/features.py:309` `MLFeatureExtractor` defines 22 features in six groups (`:315-322`): returns_1/5/20, price_vs_sma20/50, rsi, macd, macd_signal, macd_hist, bb_position, bb_width, volume_ratio, volume_std_ratio, volume_trend, atr_ratio, high_low_ratio, std_dev_ratio, pivot/r1/s1 distance, spread_proxy, volume_imbalance. **Window requirement: at least 50 bars**; fewer returns all-zero features (`:333-347`), and `batch_extract` returns an empty frame (`:427`). The live call passes the Schwab default (1 day of 5-min bars, ~78 rows).
- **Model**: `ml/models.py:378` `IntegratedMLModel` — `StandardScaler` + `xgb.XGBClassifier(n_estimators=100, ...)` (`:511`), 3-class labels `{0: SELL, 1: HOLD, 2: BUY}`, pickled to `ml_model_integrated.pkl` (`:399`, `:670`). On load it checks `model_version == "2.0"` and `check_is_fitted` (`:695-706`); the current log shows `Loaded ML model v2.0 trained on 2730982 samples`.
- **Known mapping issue (factual)**: `predict()` builds `prediction_map = {0: -1, 1: 0, 2: 1}` and computes `mapped_prediction` at `ml/models.py:630-631`, but the returned `MLPrediction` uses `signal=int(prediction)` at `:633`, i.e. the **raw class index (0/1/2)**, and `mapped_prediction` is never used. Downstream, `core/engine.py:7657` compares `ml_signal` against `expected_ml_signal = 1 if BUY else -1`. Under this encoding a raw `1` (HOLD) reads as agreement with BUY, a raw `2` (BUY) reads as disagreement with BUY, and SELL signals (`-1`) can never match a raw `0`.
- **Engine call**: in `_analyze_markets_with_commentary`, when `Config().ML_PREDICTION_ENABLED` (`core/config.py:430`, default True) the predictor runs in the default thread executor with `asyncio.wait_for(..., timeout=4.0)`; any timeout/exception yields `(0, {})` (`core/engine.py:5775-5789`).
- **Veto**: at `core/engine.py:7658-7700`, a disagreeing ML signal logs a SKIP decision; if `ml_explanation["confidence"] >= Config().ML_VETO_CONFIDENCE` (`core/config.py:434`, default 0.65) the trade is skipped in all modes, audited as `ml_veto_high_confidence`, and a feature snapshot is emitted. Below threshold ML is advisory only.
- **Startup**: if the pickle is untrained, `start()` calls `retrain_model(data_provider, ['PLTR','NVDA','TSLA'])` (`core/engine.py:4828-4841`), which uses the in-model fixed-horizon labeller (`ml/models.py:438-455`).

### 4.3 Meta model and side classifier

**Meta model** (`ml/meta_inference.py:36` `MetaInference`) loads a joblib bundle (`classifier`, `scaler`, `feature_names`, `primary_strategy`) and returns `MetaSignal(win_probability, side, would_trade_at_065/070/075)` (`:74`). The engine loads `ml_meta_model.pkl` at startup (`core/engine.py:267-285`, fail-soft). Per mean-reversion signal, `_shadow_meta_evaluate` (`:5861`) fetches 5-min bars, runs `batch_extract`, takes the last row, and audits `component=meta_shadow` with the probability; it also stashes `signal.reasoning["meta_proba"]` (`:5873`). That value is **not shadow-only**: `ENABLE_CONVICTION_FLOOR_MEANREV` (`core/config.py:1540`, default True) blocks mean-rev/oversold_v2 entries with `meta_proba < CONVICTION_FLOOR_META` (0.65, `:1556`), fail-open when no probability exists (`core/engine.py:6053`). It also feeds `sizing/conviction_sizer.py` (meta_proba weighted 0.5, multiplier 0.5-1.5x, `:34,63`), which is **shadow-only** and writes `conviction_sizer_shadow.ndjson`. Note: `ml_meta_model.pkl` is absent from this checkout (only in `/home/nvidia/claude/trading_bot_commentary`), and `trading_bot.log` contains zero `meta_shadow` lines, so in this working tree the meta path is disabled and the conviction floor fails open.

**Side classifier** (`core/classifier/`) decides per symbol which sides {LONG, SHORT} are allowed. `rule_based.py` is the rule layer; `composer.py:45 classify` intersects it with an optional trained FFN (`trained/ffn_predictor.py`), which can only narrow (`:104`). It is imported by the engine only when `USE_SIDE_CLASSIFIER` is True (`core/config.py:715`, default **False**), with per-strategy switches, thresholds, `SIDE_CLASSIFIER_SHADOW_MODE`, and checkpoint path `models/classifier/ffn_watchlist_v1.pt` (`:767`). `runtime.py:78 ClassifierRuntime.shadow_evaluate` fails open when torch or the checkpoint is missing. Research support: `research/features.py`, `research/labels.py` (triple-barrier LONG_WINS/SHORT_WINS/NEITHER), `research/train_classifier.py`, `research/classifier_backtest.py`, `research/session_reporter.py`, `research/drift_monitor.py`.

### 4.4 Training

- **`train_ml_model.py`** (v1, live model): top-150 symbols by volume from `minute_bars` (leveraged ETFs excluded), 180 days, 5-min bars via `PostgresDataProvider`. Labels are fixed-horizon: 10-bar forward return `> +0.3%` = BUY(2), `< -0.3%` = SELL(0), else HOLD(1) (`:140-141`). Walk-forward CV then a final fit through `IntegratedMLModel.train` so the artifact stays compatible (`:285-290`). Output: `ml_model_integrated.pkl`, `ml_training_report.json`.
- **`ml_training_report.json`** claims (2026-04-14): 150 symbols, 2,730,982 samples (26.9% BUY / 45.4% HOLD / 27.7% SELL), 5-fold mean accuracy **0.527 ± 0.025**; per-class F1 SELL 0.39, HOLD 0.69, BUY 0.27 (BUY recall 0.20). HOLD is the majority class, so accuracy is close to the base rate.
- **`train_ml_model_v2.py`**: triple-barrier labels (`ml/labels.py:24 apply_triple_barrier`, PT 2.0 ATR / SL 1.0 ATR / 60-bar time barrier; `compute_uniqueness` weights at `:118`), `ml/purged_cv.py:23 PurgedKFold` with 1% embargo, XGBoost. Writes `ml_model_v2.pkl` / `ml_training_report_v2.json` — **not loaded by the engine** (only `ml_model_integrated.pkl` and `ml_meta_model.pkl` are referenced).
- **`train_meta_model.py`**: replays primary strategy rules vectorised (`ml/strategy_replay.py`) to get event timestamps, labels them side-aware via `ml/meta_labels.py:apply_meta_triple_barrier` (time expiry = loss), trains a binary XGBoost with purged CV and a threshold sweep; outputs `ml_meta_model.pkl` + `ml_meta_training_report.json`.
- **`retrain_weekly.sh`**: backs up to `model_versions/`, runs `train_meta_model.py` then `train_ml_model_v2.py --pt-mult 1.5 --sl-mult 1.5`. Cron `0 2 * * 1` — but its hard-coded `PROJECT_DIR` is `/home/nvidia/claude/trading_bot_commentary`, not this checkout, and it never retrains the v1 model the engine actually loads. Neither `ml_training_report_v2.json` nor `ml_meta_training_report.json` exists here.

### 4.5 Research and backtest tooling

- **`backtest/engine.py`**: replays Postgres (or `BARS_DIR`) 5-min bars through the indicator strategies in isolation — no ML veto, brain gate, or risk manager; news strategy excluded (`:1-9`). Indicators precomputed with `ta`; bars with NaN indicators skipped; `lows_50` uses an exclusive lookback to avoid lookahead (`:178`). One position per strategy per symbol. `backtest/metrics.py:74 summarize` (win rate, per-trade Sharpe/Sortino/Calmar, PF, max DD); `backtest/cli.py` with `--seed`/`--out`; `seed_run` (`engine.py:44`) is defensive, the loop itself is deterministic and a CI test asserts it.
- **Stage A harness**: `backtest/stage_a_scorer.py:22` locks floors (n>=150, >=10 sessions, PF>=1.30 "fees+slip included", WR>=48% with |R|<0.05 scratches excluded, exp>=+0.05R, DD<=6%, max losing day<=2R; hands-off MU/HQGE/SPCX). `backtest/stage_a_cli.py` scores lanes (mean-rev LONG/SHORT, day-trade SHORT) from Postgres, shadow ledgers, or yfinance (marked INCOMPLETE). Reports live in `research/reports/*stage-a-scorecard*`.
- **Shadow ledgers**: strategies emit hypothetical entries to `data/shadow_long_log.ndjson`, `shadow_short_log.ndjson`, `day_trade_short_shadow.ndjson`, `shadow_unified_exit.ndjson`; `research/mean_rev_long_bt_sample.py` and `day_trade_short_bt_sample.py` synthesise the same ledgers from historical bars. Resolvers (`research/shadow_long_resolver.py`, `shadow_short_resolver.py`, `day_trade_short_resolver.py`) replay each entry to stop/target/timeout/flatten and compute R. Root-level `regime_allocator_shadow.ndjson` (`allocators/regime_allocator.py`, SPY 20-day efficiency ratio >= 0.30, gates nothing yet), `conviction_sizer_shadow.ndjson`, and `symbol_intel_log.ndjson` (`core/symbol_intel.py`, heuristic quality score) are evidence ledgers; `tools/bot_monitor.py:474` reports row counts only. Promotion is manual: each module's docstring states its criterion (e.g. allocator: two weeks of agreement with the walk-forward; sizer: gradient reproduced over n>=20 trades), and `research/drift_monitor.py` emits yaml-change recommendations rather than flipping flags.
- **Walk-forward scripts** (`research/*walkforward*.py`): six non-overlapping 60-day windows with era-appropriate universes; `walkforward_report_2026-06-09.json` verdicts are **FRAGILE** for breakout (4/6 PF>1, worst window -56.5%), mean-reversion (3/6, worst -104.7%), and momentum (0 trades). Later studies test regime allocation, gated pullback/mean-rev, and catalyst-momentum v2. `research/weekly_report.py` joins logs, Schwab fills, and `minute_bars` for counterfactuals; `analyze_whipsaws.py` and `analyze_shadow_outcomes.py` (live precision of `meta_shadow` evals per threshold) are one-off diagnostics; `research/eod_runner.sh` chains session report + drift monitor.

### 4.6 Backtest realism caveats (as implemented)

- `backtest/engine.py` charges **no commissions, fees, or slippage**; entry fills at `signal.entry_price` on the signal bar (`:236`), stops/targets fill exactly at the barrier price using the bar's high/low (`:198-213`), stop takes precedence when both are touched in one bar, timeout/EOD exits at bar close. Intrabar path is unknown, so gap-throughs are ignored.
- Shadow resolvers apply a fixed 0.1% adverse slippage on entry (`shadow_long_resolver.py:60,313`) but no fees; the Stage A scorer reads `fees`/`slippage` columns if present but does not itself model them.
- Fixed-horizon v1 labels ignore path dependence (the v2/meta pipelines exist to fix this but are not what trades live). Both trainers use symbol universes selected by today's volume, and `minute_bars` coverage depends on the nightly ingest.

## 5. Interfaces, configuration, and operations

### 5.1 HTTP API

The FastAPI app is created in `api/routes.py:38` (`FastAPI(title="Trading Bot with Commentary API")`). Backtest endpoints are registered from a separate module via `api/backtest.py:222 register(app)`, called at `api/routes.py:93-94`. The server is started by `main()` in `api/routes.py:4100-4170`; the 120-line top-level script `trading_bot_commentary_updated.py:104,120` only imports `app, main` and calls `main()`. It binds `0.0.0.0` on `TRADING_BOT_PORT` (default **9000**, `api/routes.py:4108`) — not 8000 as older docs say; the live process is currently listening on `:9000`.

#### 5.1.1 Auth model

- `api/auth.py` exposes `verify_api_key` (HTTP Bearer dependency) and `verify_ws_token(token)` (`api/auth.py:27,74`). Both compare against env `TRADING_API_KEY` with `secrets.compare_digest`. **If the env var is unset, auth is disabled** and a one-time warning is logged (`api/auth.py:50-57`).
- In practice, HTTP protection is enforced by `APIAuthMiddleware` (`api/routes.py:56-89`), which requires `Authorization: Bearer <key>` on every path starting with `/api/`, and skips `/`, `/docs*`, `/openapi*`, `/redoc*`. Non-`/api/` paths such as `/activity`, `/stream`, `/logos/{symbol}`, and the WebSocket handshakes are not covered by the middleware.
- WebSockets authenticate via the `?token=` query param: `/ws` at `api/routes.py:1613-1616`, `/ws/backtest/{run_id}` at `api/backtest.py:260-263`. Failures close with code 1008.
- `GET /` and `GET /activity` inject the API key into the page (`window.TRADING_API_KEY`, `api/routes.py:234-238` and `2787-2788`) so the browser can call `/api/*` and `/ws`. The code's own SECURITY NOTE (`api/routes.py:231-233`) marks this as single-user localhost only. `TRADING_API_KEY` is set in `.env`.
- CORS (`api/routes.py:40-53`): allow-list of `http://localhost|127.0.0.1` on ports 3000/8000/9000, `allow_credentials=True`, all methods/headers.

#### 5.1.2 Endpoint table

"Mutates" means it changes engine state, config on disk, or broker positions.

**Pages and streams**

| Method | Path | Purpose | Mutates |
|---|---|---|---|
| GET | `/` | Legacy dashboard; re-reads `templates/dashboard.html` per request, injects API key + `UI_ACTIVITY_PAGE` flag (`routes.py:229-262`) | no |
| GET | `/activity` | Activity page (`templates/activity.html`); 404 unless `Config().UI_ACTIVITY_PAGE` (`routes.py:2778-2800`) | no |
| GET | `/stream` | Tick-level stream monitor rendering `schwab_raw` WS messages (`routes.py:263`) | no |
| GET | `/logos/{symbol}` | Cached company logo from `static/logos/` (`routes.py:3554`) | no |
| WS | `/ws` | Dashboard/commentary stream, 4 Hz (`routes.py:1592`) | no |
| WS | `/ws/backtest/{run_id}` | Backtest progress events (`backtest.py:255`) | no |

**Engine control**

| Method | Path | Purpose | Mutates |
|---|---|---|---|
| POST | `/api/start` | Start engine thread asynchronously (`routes.py:559`) | yes |
| POST | `/api/stop` | Sets `trading_engine.is_running=False` (`routes.py:676`) | yes |
| POST | `/api/toggle-mode` | Switch `TradingMode` live/simulation (`routes.py:133`) | yes |
| POST | `/api/toggle-confirmations` | Toggle close-confirmation requirement (`routes.py:206`) | yes |
| POST | `/api/toggle-close-mode` | Manual vs automatic closing (`routes.py:3789`) | yes |
| POST | `/api/toggle-ml-prediction` | Toggle ML veto (`routes.py:3811`) | yes |
| POST | `/api/toggle-managed-by-bot` | Per-position bot management on/off (`routes.py:3698`) | yes |
| POST | `/api/toggle-long-term` | Mark position long-term (`routes.py:3755`) | yes |
| POST | `/api/confirm-close` | Answer a `close_confirmation_request` (`routes.py:1981`) | yes |
| POST | `/api/request-close-position` | Manual close (`routes.py:3266`) | yes (broker) |
| POST | `/api/emergency/close-all` | Close every open position in a mode (`routes.py:3298`) | yes (broker) |
| POST | `/api/refresh-positions` | Force Schwab position resync (`routes.py:1147`) | cache only |
| POST | `/api/reset-pnl` | Reset P&L counters (`routes.py:3234`) | yes |
| POST | `/api/reset-brain` | Reset `trading_brain.json` memories (`routes.py:3834`) | yes |

**Configuration**

| Method | Path | Purpose | Mutates |
|---|---|---|---|
| GET | `/api/settings`, `/api/settings/{category}` | Read yaml config (`routes.py:687,739`) | no |
| PUT | `/api/settings`, `/api/settings/{category}` | `config_manager.update(key, value)` per key — **writes `Config().yaml`** (`routes.py:721-738`, `core/config.py:146-156`) | yes (disk) |
| POST | `/api/settings/reset` | Reset to defaults and save (`routes.py:770`) | yes (disk) |
| POST | `/api/update-config` | Single-key variant of PUT (`routes.py:3861`) | yes (disk) |
| GET | `/api/config` | Current config (`routes.py:3882`) | no |
| GET/PUT | `/api/hands-off-denylist` | Symbols the bot must never manage (`routes.py:806,843`) | PUT yes |
| POST | `/api/professional/risk/settings` | Risk-manager settings (`routes.py:1547`) | yes |
| POST | `/api/professional/strategies/{key}/toggle` | Enable/disable a strategy (`routes.py:1553`) | yes |
| POST | `/api/professional/models/{key}/select` | Choose active ML model (`routes.py:1218`) | yes |

**Status and read-only data**

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/status` | Bot self-report (used by monitor) (`routes.py:942`) |
| GET | `/api/system-stats` | NewsBus, task supervisor, profile (`routes.py:955`) |
| GET | `/api/account-stats` | Balance/buying power; real Schwab data in live mode (`routes.py:1050`) |
| GET | `/api/market-indices` | Regime strip (SPY/VIX etc.) (`routes.py:927`) |
| GET | `/api/positions/db` | Open positions (`routes.py:2860`) |
| GET | `/api/trades` | Trades from Postgres with filters (`routes.py:2010`) |
| GET | `/api/get-trades-today` | Today's fills from Schwab (`routes.py:1997`) |
| GET | `/api/activity` | Paired entry/exit trades + decision reasoning, union of `bot_trades` and broker fills with orphan flags (`routes.py:2292`) |
| GET | `/api/decisions`, `/api/decision-snapshots` | Engine decisions / snapshots from Postgres (`routes.py:2801,2976`) |
| GET | `/api/features/{symbol}` | Latest feature snapshot (`routes.py:3109`) |
| GET | `/api/quote-cache` | PriceBook introspection (`routes.py:1123`) |
| GET | `/api/news-bus/{symbol}` | NewsBus items (`routes.py:1022`) |
| GET | `/api/news-vetoes/report` | News-veto gate scorecard (`routes.py:3629`) |
| GET | `/api/historical-data/{symbol}` | OHLC for chart (`routes.py:3418`) |
| GET | `/api/performance-metrics`, `/api/advanced-analytics` | Metrics (`routes.py:3841,3896`) |
| GET | `/api/professional/{models,risk,strategies,performance,paper/account,paper/positions,config}` | Professional-layer reads (`routes.py:1161-1352`) |
| GET | `/api/sentiment/{update,market,symbol/{s},headlines,alerts,correlation/{s},social/{s},earnings}` | Sentiment widget reads (`routes.py:1421-1530`); `POST .../alerts/{id}/acknowledge` mutates |
| GET | `/api/analytics/*` | **Stubs** — return empty payloads with `"analytics_logger removed"` (`routes.py:1376-1396`) |
| GET | `/api/alternative-data/{s}`, `/api/market-neutral-opportunities`, `/api/data-quality/{s}` | Legacy analytics reads (`routes.py:3942-4020`) |
| POST | `/api/optimize-portfolio`, `/api/record-execution` | Legacy hooks (`routes.py:4047,4072`) |

**Backtest** (`api/backtest.py`): `POST /api/backtest/run` starts a single-flight run and returns `{status, run_id}`; a concurrent request gets 429 with the in-flight `run_id` (`backtest.py:224-244`). `GET /api/backtest/latest` returns `backtest_results/latest_report.json` or 404. Results persist to `backtest_results/run_YYYYMMDD_HHMMSS.json`.

Note: `api/routes.py:3035` carries a hardcoded development Postgres DSN (with credentials) as the fallback when `POSTGRES_DSN` is unset; the value is not reproduced here.

### 5.2 WebSocket messages

`core/websocket_manager.py` holds a `ConnectionManager` with a locked set of sockets; `broadcast()` snapshots the set, sends concurrently with a 2 s per-client timeout, and drops dead sockets (`websocket_manager.py:28-69`). The engine reaches it via `_broadcast_ui` (`core/engine.py:3696`).

Every message is `{"type": ..., "data": ...}`:

| type | Source | Content |
|---|---|---|
| `dashboard_update` | `/ws` loop every 250 ms (`routes.py:1622,1902-1926`) | `account{balance,buying_power,daily_pnl,margin_call,cash}`, `simulated_positions[]`, `real_positions[]` (each with `is_stale`, `managed_by_bot`, `is_long_term`), `trades`, `screener`, `live_quotes`, `symbol_intel` (top 40) |
| `commentary` | subscriber on `CommentarySystem`, queued thread-safely to the loop (`routes.py:595-640`) | one `TradingCommentary.to_dict()` |
| `trade_update` | `engine.py:2865-2876` | `new_trade`, `all_trades` |
| `close_confirmation_request` | `engine.py:9458-9470` | `request_id, symbol, quantity, entry/current price, pnl, roi, reason, profile, timeout_sec` — answered via `POST /api/confirm-close` |
| `sentiment_update` / `sentiment_alert` | `/ws` loop every ~30 ticks (`routes.py:1891,1932,1941`) | sentiment widget payloads |
| `schwab_raw` | stream publisher (`engine.py:3745`) | raw Schwab stream frame, consumed by `/stream` |
| `progress` / `complete` / `error` | `/ws/backtest/{run_id}` (`backtest.py:206-214,270-290`) | backtest events; last event replayed on connect |

The Next.js hook `dashboardUI/hooks/use-bot-data.ts:194,264` consumes only `commentary` and `dashboard_update`.

### 5.3 Commentary system

`core/commentary.py` defines `TradingCommentary` (`timestamp, type, symbol, title, message, data, confidence, importance=5`, `commentary.py:32-41`) and `CommentarySystem` (`commentary.py:75`). `CommentaryType` lives in `core/models.py:15-28`: `market_analysis, signal_generation, risk_assessment, decision, technical, fundamental, psychology, warning, opportunity, anomaly, info, account_update, error`. `commentary_types_reference.md` documents the enum and the mapping from ML-pipeline event names (e.g. `ML_SIGNAL → SIGNAL_GENERATION`, `DRIFT_DETECTION → WARNING`) and notes there is no `TRADE_EXECUTION` type — use `DECISION`.

Importance is an integer 1-10; the engine mostly emits 6-10 (counts in `core/engine.py`: 43× importance=8, 29× 9, 16× 10). Nothing in `commentary.py` filters on importance or `commentary.detail_level`; both are carried to the client for display decisions.

Flow: `add_commentary()` appends to a `deque(maxlen=max_history)` (yaml `commentary.max_history: 100`), prints a Rich panel colour-coded by type (`commentary.py:121-132`), notifies subscribers (WebSocket bridge), and calls `_save_to_file()`. Persistence to `paths.commentary_log` (`trading_commentary.json`) is rate-limited to once per 5 s with a non-blocking lock (`commentary.py:82-84,181-188`), keeps the last 1000 entries (`commentary.py:243-244`), and recovers from a corrupted file by salvaging valid objects and writing a `.json.corrupted` backup (`commentary.py:218,300-346`).

### 5.4 Dashboards

- **Legacy HTML dashboard** — `templates/dashboard.html` (~337 KB), served at `/`. This is what the running bot serves today. It contains the Activity nav link gated by `window.UI_ACTIVITY_PAGE`.
- **Activity page** — `templates/activity.html`, served at `/activity` and backed by `GET /api/activity?date=&strategy=`. Shows paired entry/exit trades with stop/target/R at entry, exit reason, decision reasoning from `bot_decision_snapshots`, and hands-off/external/orphan flags (broker fills without a `bot_trades` row show `is_orphan=true`). Gated by `Config().UI_ACTIVITY_PAGE` (`core/config.py:4007`; env `UI_ACTIVITY_PAGE`, yaml `trading.ui_activity_page`, default `False`; currently `UI_ACTIVITY_PAGE=1` in `.env`).
- **`static/logos/`** — cached PNG logos served by `/logos/{symbol}`; untracked in git.
- **`dashboardUI/`** — a Next.js app (`next dev|build|start`) with a single route `app/page.tsx` composed from `components/trading/`: account-overview, ticker-bar, market-status, positions-table, recent-trades, live-commentary, decision-card, news-sentiment, performance-metrics, portfolio-allocation, price-chart, watchlist-heatmap, quick-trade, command-palette. `lib/api.ts` reads `NEXT_PUBLIC_BOT_URL` (default same origin), sends Bearer auth via `authFetch`, builds `ws://…/ws?token=` (`api.ts:11,18,28`), and fetches `/api/account-stats`, `/api/status`, `/api/market-indices`, `/api/positions/db` (`api.ts:168-171`). It is not served by the Python process; run it separately on port 3000 (CORS-allowed).

### 5.5 Configuration

`core/config.py` loads `.env` via `python-dotenv` at import (`config.py:8-10`), then `ConfigManager` reads `Config().yaml` (literal filename) or writes a default one (`config.py:16-29`). `Config` exposes ~240 `@property` accessors. Precedence for most feature flags is **env var → yaml key → hardcoded default**; a comment at `config.py:1808` records an incident where yaml silently won over env, which is why env is checked first for those flags. `TRADING_PROFILE` (`config.py:359-371`) selects `supervised` (default) or `autonomous_live`, which flips defaults for `CONFIRMATION_TIMEOUT_ACTION` (deny vs execute) and `FLATTEN_ON_CIRCUIT`.

`Config().yaml` sections: `backtesting`, `commentary`, `error_recovery`, `order_management`, `paths`, `performance_metrics`, `schwab` (callback URL, token path only — secrets are env), `technical_analysis`, `trading`.

Selected flags (default = code default; current = yaml or `.env` if set):

| Area | Property (`core/config.py`) | Source key | Default | Current |
|---|---|---|---|---|
| Risk | `MAX_RISK_PER_TRADE` :249 | trading.max_risk_per_trade | 0.02 | 0.02 |
| Risk | `MAX_DAILY_LOSS` :257 | trading.max_daily_loss | 0.05 | 0.05 |
| Risk | `MAX_CONSECUTIVE_LOSSES` :261 | trading.max_consecutive_losses | 3 | 3 |
| Risk | `MAX_POSITIONS` :309 | trading.max_positions | 10 | 5 |
| Risk | `MAX_POSITION_VALUE` :325 / `_PCT` :2078 | trading.max_position_value(_pct) | 25000 / 0.25 | 10000 / 0.25 |
| Risk | `RISK_PER_TRADE_PCT` :2114, `ATR_STOP_MULTIPLIER` :2083, `ATR_REWARD_RISK_RATIO` :2109 | trading.* | 0.015, 1.5, 2.0 | defaults |
| Risk | `LIVE_SIZE_MULTIPLIER` :1046 | trading.live_size_multiplier | 1.0 | default |
| Risk | `ENABLE_BOT_ONLY_PNL_CIRCUIT` :1777 | env/yaml | True | 1 |
| Risk | `FLATTEN_ON_CIRCUIT` :401 | trading.flatten_on_circuit | profile-dependent | default |
| Orders | `REQUIRE_CLOSE_CONFIRMATION` :339, `CONFIRMATION_TIMEOUT_SEC` :375 | order_management.* | True, 30 | True, default |
| Orders | `ENABLE_BROKER_LEG_AUTHORITY` :1139, `REBRACKET_SKIP_IF_FLAT_OR_EXITING` :1112 | env/yaml | True, True | 1, default |
| Orders | `ENABLE_GHOST_FLATTEN_AFTER_BROKER_FLAT` :1165 | env/yaml | False | 0 |
| Orders | `CLEAR_PYRAMID_LOCK_ON_BROKER_FLAT` :1187 | env/yaml | True | 1 |
| Orders | `HANDS_OFF_DENYLIST` :1818 | trading.hands_off_denylist | MU, HQGE, SPCX | default |
| Strategies | `ENABLE_BREAKOUT_LONG` :1964, `ENABLE_MOMENTUM_LONG` :1976 | trading.* | False | default |
| Strategies | `ENABLE_MEAN_REV_SHORT` :1358 / `_SHADOW` :1411 | trading.* | False / False | default / **true** |
| Strategies | `MEAN_REV_LIVE_ENTRIES_ENABLED` :2190 | env/yaml | True | **1** (live since 2026-09-28 13:17 ET; `mean_rev_intraday\|long` playbook probation 0.5x, §9) |
| Strategies | `ENABLE_ORB_STRATEGY` :3478, `ORB_LIVE_ENTRIES_ENABLED` :3501, `ORB_SIM_SHADOW_ENABLED` :3520 | env/yaml | False, False, True | 1, **1** (live since 2026-09-28 13:17 ET, §9), 1 |
| Strategies | `ENABLE_PINNED_WATCHLIST` :800, `PINNED_WATCHLIST` :826 | env/yaml | True, built-in list | 1, NVDA…AMD |
| Day-trade | `ENABLE_DAY_TRADE_MOMENTUM` :2140 | env/yaml | True | default |
| Day-trade | `DAY_TRADE_LIVE_ENTRIES_ENABLED` :2159 | env/yaml | False | **true** (env, live since 2026-09-28 13:17 ET; `dt_momentum\|long` playbook-suspended, §9) |
| Day-trade | `ENABLE_DAY_TRADE_SHORT` :2387 / `DAY_TRADE_SHORT_LIVE_ENTRIES_ENABLED` :2412 / `_SHADOW` :2443 | env/yaml | True / False / True | default / **1** (live since 2026-09-28 13:17 ET, §9) / 1 |
| Day-trade | `DT_OPEN30_CONT_INDEX_CONFIRM` :2908, `DT_OPEN30_CONT_RS_FROM_OPEN` :2988, `DT_INMOVE_INDEX_TURN` :3058 (+ `_LIVE_ENFORCE`) | env/yaml | False | measure=1, enforce=0 (PR102/103 shadow) |
| Day-trade | `DAY_TRADE_FLATTEN_HOUR` :2699, `LATE_ENTRY_CUTOFF_HOUR/MINUTE` :1061 | trading.* | 15, 15:30 | default |
| Day-trade | `ENABLE_LATE_ENTRY_GATE` :2537 / `LATE_ENTRY_GATE_SHADOW` :2560 | env/yaml | True / True | 1 / **0 (enforcing)** |
| Exits | `ENABLE_PROACTIVE_EXIT` :913, `PROACTIVE_EXIT_MIN_AGE_DAYTRADE` :992, `PROACTIVE_EXIT_R_OVERRIDE_THRESHOLD` :1019 | env/yaml | True, 3, -0.3 | default, 3, -0.3 |
| Exits | `DT_UNIFIED_EXIT` :930 / `_LIVE_ENFORCE` :952 | env/yaml | False / False | default |
| Exits | `BREAKEVEN_ACTIVATION_R` :894, `TRAIL_ACTIVATION_ATR_MULT` :1283 | trading.* | 0.5, 2.0 | default |
| ML | `ML_PREDICTION_ENABLED` :430, `ML_VETO_CONFIDENCE` :434 | trading.* | True, 0.65 | true, default |
| ML | `USE_SIDE_CLASSIFIER` :715, `SIDE_CLASSIFIER_SHADOW_MODE` :754 | trading.* | False | default |
| ML | `FEATURE_SNAPSHOT_LOGGING` :3427 / `_INFERENCE` :3443 | env/yaml | True / False | default |
| News | `ENABLE_NEWS_VERIFIER` :584, `NEWS_VERIFIER_ADVISORY` :603 | trading.* | False, True | default |
| News | `ENABLE_ALPACA_NEWS_BUS` :3971 (needs `ALPACA_API_KEY`), `NEWS_LOOP_SEC` :420 | env/yaml | True, 20 | default |
| News | `ENABLE_GPU_NEWS_CRITIC` :3855 / `GPU_NEWS_CRITIC_SHADOW` :3885 | env/yaml | True / True | default |
| News | `GPU_CRITIC_CONTAM_LEXICON_ENABLE` :3926 | env/yaml | False | **true** |
| News | `ENABLE_THEME_SHOCK_LOGGER` :3763 | env/yaml | True | 1 |
| Screener | `SCREENER_LOOP_SEC` :777, `WATCHLIST_SIZE` :786, `MIN_DOLLAR_VOLUME` :2666 | trading.* | 120, 20, 10M | default, default, 25M |
| Loops | `ANALYSIS_LOOP_RTH_SEC` :1230, `POSITION_LOOP_SEC` :1096, `ORDER_MONITOR_INTERVAL_SEC` :1103 | trading.* | 30, 1.0, 5.0 | default |
| Shadow | `ENABLE_REGIME_ALLOCATOR_SHADOW` :1502, `ENABLE_MEAN_REV_LONG_SHADOW` :1446, `ENABLE_ACTIVE_OPEN_DESK` :3658 / `ACTIVE_OPEN_DESK_SHADOW` :3684 | env/yaml | True | 1 / 1 |
| Desk | `ENABLE_DESK` :4284, `ENABLE_DESK_WATCHLIST` :4300, `ENABLE_DESK_LIMITS` :4314, `ENABLE_PLAYBOOK_GATE` :4328 | trading.enable_desk* | False (all four) | default (unset in `.env`; see § "Desk (shadow)") |
| Desk | `DESK_IN_PLAY_N` :4342, `DESK_SCAN_MAX_SYMBOLS` :4355, `DESK_DATA_DIR` :4369 | trading.desk_* | 20, 60, `data/desk` | default |
| Desk | `DESK_PLAN_TIME_ET` :4382, `DESK_REFRESH_TIME_ET` :4395, `DESK_JOURNAL_TIME_ET` :4408, `DESK_CALENDAR_PATH` :4421 | trading.desk_* | `09:00`, `09:35`, `16:05`, `data/desk/calendar.json` | default |
| UI | `UI_ACTIVITY_PAGE` :4007 | env/yaml | False | 1 |

Shadow loops write NDJSON ledgers in the repo root (`regime_allocator_shadow.ndjson`, `conviction_sizer_shadow.ndjson`, `symbol_intel_log.ndjson`).

**Environment variables read by code** (non-secret): `TRADING_PROFILE`, `TRADING_BOT_PORT`, `UI_ACTIVITY_PAGE`, `POSTGRES_DSN` (routes + training scripts), `ECON_CALENDAR_API_URL` (`core/econ_calendar.py:608`), plus the ~60 feature flags above (`ENABLE_*`, `DT_*`, `DAY_TRADE_*`, `ORB_*`, `MEAN_REV_*`, `PROACTIVE_EXIT_*`, `LATE_ENTRY_*`, `PINNED_*`, `CORRELATION_IGNORE_*`, `LEDGER_INTEGRITY`, `MINUTE_BARS_INGEST_*`, `GPU_*`, `THEME_*`, `ACTIVE_OPEN_DESK_*`, `FEATURE_SNAPSHOT_*`). Secrets: `TRADING_API_KEY`, `SCHWAB_API_KEY`, `SCHWAB_SECRET`/`SCHWAB_APP_SECRET`, `SCHWAB_ACCOUNT_NUMBER`, `ALPACA_API_KEY`/`ALPACA_SECRET_KEY`, `ECON_CALENDAR_API_KEY`. `.env.example` also lists `TRADING_ENV`, `PAPER_TRADING`, `LOG_LEVEL`, `AUTO_START`, `TELEGRAM_*`, `MAX_*` — these appear in `.env` but no code under `core/`, `api/`, or `strategies/` reads them.

Seven `.env.bak_*` / `.env.pre_*` snapshots sit in the repo root (taken before each flag change, e.g. `pre_pr102`, `pre_pr103`, `pre_activity`); they contain secrets and are the rollback path for `.env` edits.

### 5.6 Monitoring and operations

**Logging** (`core/config.py:4081-4134`): logger `TradingBot`, `RotatingFileHandler` on `paths.log` (`trading_bot.log`), 10 MB × 5 backups, INFO level, one JSON object per line (`timestamp, level, logger, message, module, function`); console gets a plain formatter. Six files currently exist (`trading_bot.log` … `.log.5`).

**Monitor** (`tools/bot_monitor.py`, `docs/MONITOR.md`): a pure observer that tails the log, reads `trading_state.json` and `token_1.json`, and calls `GET /api/status`, `/api/positions/db`, `/api/market-indices`. It writes `docs/monitor/latest.md` plus `docs/monitor/<date>/<HH-MM>.md` with eight sections (process health incl. token TTL and stream heartbeat, P&L vs circuit breaker, positions, signal activity, direction-gate skips, R:R audit, ERROR/CRITICAL entries, regime). Run via `tools/start_monitor.sh {start|stop|status}` (nohup, pidfile `/tmp/bot_monitor.pid`, stdout `/tmp/bot_monitor.out`) or `python tools/bot_monitor.py --once|--interval N|--window N|--quiet|--no-history`.

**Scripts**

- `deploy.sh` — the sanctioned launcher. Requires the `trading-bot` conda interpreter (refuses others), asserts expected `Config().yaml` launch dials, checks `token_1.json` age against the 7-day refresh TTL (aborts if expired or nearly so), prints the launch plan, then `exec python trading_bot_commentary_updated.py` (`deploy.sh:191`).
- `rollback_launch.sh` — restores `Config_launch_baseline_2026-05-26.yaml` and `git revert`s tag `launch-2026-05-26`; interactive.
- `start_professional_bot.sh` / `start_safe_bot.sh` — legacy launchers for `run_professional_bot.py` (kill anything on :8000, then start); superseded by `deploy.sh`.
- `retrain_weekly.sh` — backs up `ml_meta_model.pkl` into `model_versions/`, runs `train_meta_model.py` and `train_ml_model_v2.py --pt-mult 1.5 --sl-mult 1.5`. Note its `PROJECT_DIR` is `/home/nvidia/claude/trading_bot_commentary`, not this checkout.
- `reauth_schwab.py` — interactive: backs up the old token as `token_1.expired-<ts>.json`, runs `schwab.auth.client_from_manual_flow`, writes the new token, verifies with one API call. Needs `SCHWAB_API_KEY` and `SCHWAB_SECRET` in env.
- `core/token_health.py:29 check_token_health()` — reads `creation_timestamp`, computes 7-day expiry, logs WARNING at <48 h/<24 h and ERROR when expired; called at engine startup (`core/engine.py:4074`). Does not block startup.

**Scheduling**: `crontab -l` shows five jobs. Only one targets this checkout: 21:00 ET Mon-Fri `ingest_alpaca_bars.py --top-n 150 --days 7 --feed sip` → `ingest_cron.log`. The others run in sibling checkouts (`…/singleton/…` daily-bar backfill 17:00, dead-man switch 15:55, Sunday swing scan; `…/claude/trading_bot_commentary/retrain_weekly.sh` Monday 02:00 UTC). Systemd has only `trading-bot-paper1.{service,timer}` (user units) for a paper-runner in `…/trading_bot_enhacne/…`; nothing manages this bot — it is started by hand with `deploy.sh`.

### 5.7 Runbook essentials

- **Start**: `cd` to repo, ensure `.env` and `token_1.json` are present, run `./deploy.sh` (or `python trading_bot_commentary_updated.py` with the `trading-bot` env). Open `http://localhost:9000`. Trading only begins after `POST /api/start` / the UI Start button; mode defaults to simulation until `POST /api/toggle-mode {"mode":"live"}`.
- **Stop**: Ctrl-C or `kill -TERM <pid>`. `shutdown_handler` (`api/routes.py:4115-4160`) sets `is_running=False`, joins the engine thread for 15 s, saves `trading_state.json`, brain memories and the ML model; if the join times out it logs `Engine thread did not stop within 15s … force-exiting` and `os._exit(1)`. `POST /api/stop` stops trading but leaves the server up.
- **Re-auth** (weekly): watch for `token_health: refresh token expires in Nh … Re-auth required TODAY` in the log/monitor; run `python reauth_schwab.py` interactively, then restart the bot. `deploy.sh` refuses to start with an expired token.
- **Health checks**: `curl -H "Authorization: Bearer $TRADING_API_KEY" localhost:9000/api/status` and `/api/system-stats`; `tools/bot_monitor.py --once`; `ss -ltnp | grep 9000`; `tail -f trading_bot.log | jq -r .message`.
- **Common log signatures**: `sync_call_timeout fn=<lambda> after Ns` (Schwab REST call exceeded the executor timeout — benign if sporadic, 58 today); `schwab_stream: connection failed (attempt N…) Reconnecting` (stream watchdog reconnect; see `docs/incidents/2026-05-12-stream-watchdog.md`); `order_monitor: fetch_working_orders error: … handshake operation timed out` / `Error fetching Schwab positions` (TLS timeouts to Schwab); `Refresh token is invalid, expired or revoked` (token expired — re-auth); `watchdog: heartbeat age=Ns` (stream liveness, parsed by the monitor); `close_confirmation_request symbol=…` (a close is waiting for the operator; auto-resolves after `CONFIRMATION_TIMEOUT_SEC` per profile).
- **Config changes**: edit `Config().yaml` or `.env` and restart (Config is read at startup; the `/api/settings` PUT path also rewrites the yaml). Back up `.env` first, matching the existing `.env.bak_pre_*` convention.

## 6. Trading characteristics and current live profile

Snapshot 2026-09-24 ~18:45 ET. Flag precedence in `core/config.py` is **env var → `Config().yaml` → code default**; unprefixed `:NNN` refs below point into `core/config.py`.

### 6.1 Style in one paragraph

A **long-only, intraday momentum day-trader on liquid US equities**: regular hours only, flat by 15:00 ET, no overnight risk. Swing lanes (breakout, legacy momentum, mean-reversion, news) exist in code but are off or shadow-only; shorts exist only as shadow lanes. Profile is `supervised` (`:359`; `TRADING_PROFILE` unset, no `profile:` in yaml): closes need confirmation, 30 s timeout defaults to *deny* (`:375-397`), except when a broker OCO leg is authoritative.

### 6.2 Trading windows and holding period

| Rule | Value | Source |
|---|---|---|
| Session | RTH 09:30–16:00 ET; premarket/afterhours off; off-hours signals hard-skipped | yaml `trading.extended_hours`; `DAY_TRADE_HARD_SKIP_OFF_HOURS` `:2866` |
| First 5 min | hard block (`early_session first_5min_hard_block`) | engine audit, 7 skips in logs |
| Opening 30 min | allowed at 0.5× size; PR102 opening_30 gates measure-only | `:2754`; `DT_OPEN30_CONT_*` `:2908-3017` |
| Late-entry cutoff | 15:30 ET, enforced (env `LATE_ENTRY_GATE_SHADOW=0`) | `LATE_ENTRY_CUTOFF_HOUR/MINUTE` `:1061-1067` |
| Flatten hour | 15:00 ET; entries at/after flatten hour blocked | `DAY_TRADE_FLATTEN_HOUR` `:2699`; `DAY_TRADE_FLATTEN_HOUR_ENTRY_GATE_ENABLED` `:2221` |
| Overnight | not allowed for momentum lane | `MOMENTUM_ALLOW_OVERNIGHT_HOLD=False` `:3305` |
| Risk-off regime | day-trade lane hard-skips (no size-down) | `DAY_TRADE_HARD_SKIP_RISK_OFF` `:2728` |
| Realized hold time (bot trades) | median 21 min, mean 36 min | `trading_state.json` |

### 6.3 Sizing and risk limits

| Limit | Effective value | Source |
|---|---|---|
| Max concurrent bot positions | 5 (hands-off positions do not count) | yaml `trading.max_positions: 5`; default 10 at `core/config.py:309` |
| Risk per trade | 1.5% of account balance ≈ $650 on $43.5k, before Kelly | `RISK_PER_TRADE_PCT` default 0.015 `:2114`; `risk/manager.py:92-93` |
| Kelly scaling | quality-based Kelly, floor 0.50, caps 0.20/0.35/0.60 by signal strength | `risk/manager.py:142-190`; `KELLY_FLOOR` `:1328` |
| Strategy / lane multipliers | day-trade 0.5×; opening_30 0.5×; live multiplier 1.0× | `:2686`, `:2754`, `:1046` |
| Per-position notional cap | min($10,000 absolute, 20% × buying power ≈ $5.9k) → BP cap binds | yaml `max_position_value: 10000`; `MAX_POSITION_VALUE_BP_PCT` 0.20 `:1337`; `risk/manager.py:350-358` |
| Observed notional (bot trades) | median $6.1k, mean $6.4k (≈14–15% of equity) | `trading_state.json` |
| Daily loss halt | 5% of balance, computed on **bot-only** P&L | yaml `max_daily_loss: 0.05`; `ENABLE_BOT_ONLY_PNL_CIRCUIT` `:1777`; `risk/manager.py:440-452` |
| Consecutive-loss halt | 3 losses, auto-reset at calendar-day change | yaml `max_consecutive_losses: 3`; `risk/manager.py:454-470` |
| Portfolio heat cap | 6% | `risk/manager.py:44` |
| Outer circuit breaker | 10% daily / 15% emergency | yaml `error_recovery`; `core/config.py:169` |
| ML veto | enabled; veto confidence 0.65 | `ML_VETO_CONFIDENCE` `:434` |
| Stops/targets | ATR 1.5× stop, 2:1 R:R; breakeven at +0.5R; trail at 2.0 ATR / 1.5 ATR width | `:2083-2109`, `:894`, `:1283-1299` |
| Min R:R | 2.0 | yaml `min_risk_reward_ratio` |

Legacy `.env` keys `MAX_DAILY_LOSS`, `MAX_POSITION_SIZE`, `MAX_POSITIONS`, `MAX_DAILY_LOSS_PERCENT` are **not read** by `core/config.py`; the yaml values govern.

### 6.4 What is enabled right now

`Config().yaml` (16:06) and `.env` (16:29) were edited post-close on 2026-09-24; the pre-pause backup differs only in `day_trade_live_entries_enabled: true → false`. **As of this snapshot no lane can place a new live entry**; the bot is in exit-management + shadow-collection mode.

| Lane / feature | State | Deciding flags (effective value) |
|---|---|---|
| Day-trade momentum LONG (`DayTradeMomentumStrategy`) | **SHADOW** (was LIVE through 2026-09-24 close) | `ENABLE_DAY_TRADE_MOMENTUM` default True (`:2140`); env `DAY_TRADE_LIVE_ENTRIES_ENABLED=false` + yaml false (`:2159`) |
| Day-trade momentum SHORT | SHADOW | `ENABLE_DAY_TRADE_SHORT` default True (`:2387`); env `DAY_TRADE_SHORT_LIVE_ENTRIES_ENABLED=0`; env `ENABLE_DAY_TRADE_SHORT_SHADOW=1` |
| Mean-reversion LONG (legacy `MeanReversionStrategyWithCommentary`; v2 off) | SHADOW | env `MEAN_REV_LIVE_ENTRIES_ENABLED=0` + yaml false (`:2190`); `ENABLE_MEAN_REV_LONG_SHADOW` default True (`:1446`); `USE_OVERSOLD_BOUNCE_V2` default False (`:2017`) |
| Mean-reversion SHORT | SHADOW | `ENABLE_MEAN_REV_SHORT` default False (`:1358`); yaml `enable_mean_rev_short_shadow: true` (`:1411`) |
| ORB contraction/RVOL | SHADOW | env `ENABLE_ORB_STRATEGY=1`, `ORB_LIVE_ENTRIES_ENABLED=0`, `ORB_SIM_SHADOW_ENABLED=1` (`:3478-3520`) |
| Breakout long | OFF | `ENABLE_BREAKOUT_LONG` default False (`:1964`; `strategies/builtin.py:171`) |
| Legacy momentum swing | OFF | env `ENABLE_MOMENTUM` unset (`core/engine.py:325`) |
| Free news signal strategy | registered, heavily gated; no signals in logged sessions | `core/engine.py:327`; gates `:521-664`; verifier OFF `:584` |
| Active Open Desk | SHADOW desk; its RSI-extreme exit is live (fired 3× in Sep) | env `ENABLE_ACTIVE_OPEN_DESK=1`, `ACTIVE_OPEN_DESK_SHADOW=1`; `:3726` |
| Proactive exits (MACD flip / RSI<50) | LIVE | `ENABLE_PROACTIVE_EXIT` default True (`:913`); min age 3 min for day trades (env) |
| Smart take-profit, breakeven, trailing; late-entry gate; mean-rev quality/budget/dedupe gates; RSI≥70 hard vetoes | LIVE | `:450`, `:894`, `:1283`; env `LATE_ENTRY_GATE_SHADOW=0`; `:1576-1728`; `:3368-3394` |
| Live scale-out (partials) | OFF | `ENABLE_LIVE_SCALE_OUT` default False (`:1491`) |
| PR102 open30 index-confirm / RS-from-open; PR103 in-move index turn | SHADOW (measure only) | yaml `*_live_enforce: false` |
| Regime allocator, conviction sizer | SHADOW | `ENABLE_REGIME_ALLOCATOR_SHADOW` default True (`:1502`) |
| GPU news critic + contamination lexicon, theme-shock logger | SHADOW | `:3855-3926`; env `GPU_CRITIC_CONTAM_LEXICON_ENABLE=true` |
| Pinned watchlist (8 mega-caps) + RS soften; broker-leg authority | LIVE | env `ENABLE_PINNED_WATCHLIST=1`, `PINNED_MIN_RS_VS_SPY=0.25`; `:1139` |
| Ghost-flatten; ledger integrity; side classifier; thesis revalidation; news-thesis exit | OFF | env `=0` (`:1165`); `:1889`; `:715`; `:853`; `:674` |
| Alpaca news bus, feature snapshots, Postgres `DbLogger` | LIVE (data plumbing) | `:3971`, `:3427`; `core/engine.py:290-294` |
| Strategy-fit lanes (`fit:<family>`) | **OFF** (no family registered) | `ENABLE_STRATEGY_FIT_GATE` default False; `FIT_LANES_ENABLED` empty; see table below |

The strategy-fit gate and its per-family fit lanes are governed by nine flags, all off/empty by
default (same set documented in §3 "Strategy fit gate"; repeated here since this section is the
current-live-profile snapshot):

| Flag | Default | Effect |
|---|---|---|
| `ENABLE_STRATEGY_FIT_GATE` | `False` | Master switch for `core/strategy_fit_gate.py`; `False` means fully inert. |
| `STRATEGY_FIT_SHADOW` | `True` | Gate logs but never blocks a live entry until this is explicitly `False`. |
| `STRATEGY_FIT_PATH` | `data/strategy_hunt/fit/strategy_fit.json` | Snapshot file read by the gate and `GET /api/strategy-fit`. |
| `STRATEGY_FIT_MAX_AGE_DAYS` | `7` | Snapshot age (days) beyond which it's treated as stale (fail-closed if the gate is enforcing). |
| `STRATEGY_FIT_REGIME_ROWS` | `True` | Gate reads the regime-conditioned row for the live regime (symbol-regime → pooled → unconditional); `0` restores the unconditional-only lookup. |
| `SENTIMENT_BACKEND` | `vader` | Sentiment scorer shared by live news scoring and the research news cache; `finbert` for ProsusAI/finbert on GPU. |
| `SENTIMENT_FINBERT_PATH` | `models/finbert` | Local FinBERT checkpoint directory (`python scripts/prepare_finbert.py`). |
| `FIT_LANES_ENABLED` | `""` | Family names to register live `fit:<family>` adapters for; `promote` prints the line to set. |
| `FIT_LANE_LIVE_ENTRIES_ENABLED` | `""` | Family names allowed LIVE entries via their fit lane; absent families stay shadow-only even if registered. |
| `SWING_ALLOW_OVERNIGHT` | `False` | Allow swing-lane (multiday) fit-lane positions to hold across sessions instead of forced flat by EOD. |
| `SWING_MAX_HOLD_SESSIONS` | `10` | Max sessions a swing-lane position may be held before a forced exit. |
| `STRATEGY_HUNT_DATA_ROOT` | `data/strategy_hunt` | Root of the nightly research data tree the engine reads (`bars5/`, `daily/`, `earnings/`). |

`research/strategy_hunt promote` (see `docs/research/strategy_hunt.md` § "Operator flow") reads them
off `Config()`, cross-references the fit snapshot and each family's Stage A scorecard, and prints
the exact `FIT_LANES_ENABLED` / `FIT_LANE_LIVE_ENTRIES_ENABLED` / `SWING_ALLOW_OVERNIGHT` lines to add
to `.env` — it never edits `.env` or `Config().yaml` itself. It also reports a `gate_not_enforcing`
blocker for every otherwise-promotable family while `ENABLE_STRATEGY_FIT_GATE` is `False` or
`STRATEGY_FIT_SHADOW` is `True`: no lane goes live while the gate itself is inert.

### 6.5 Realized behavior from `trading_state.json`

`trade_history` is a rolling 100-record window (`core/engine.py:2957`). It contains 42 phantom HQGE exits at $0.0003 (Aug 2025), excluded below, leaving 58 real records.

| Cohort | n | P&L | WR | PF | Avg win / loss | Median hold | Median notional |
|---|---|---|---|---|---|---|---|
| Bot-managed (source bot/fallback) | 47 | −$293.87 | 46.8% | 0.75 | +$40.69 / −$51.70 | 21 min | $6,121 |
| …of which Sep 2026 | 34 | −$631.50 | 38.2% | — | — | — | — |
| …of which Aug 2025 | 13 | +$337.63 | — | — | — | — | — |
| External/existing-position reconciles | 11 | −$14,275.68 | 72.7% | 0.14 | +$295.71 / −$5,547 | 728 min | $21,402 |

The external cohort is two 2026-09-16 reconciliation records (SPCX −$11,302, MU −$5,242) on hands-off holdings, not bot decisions.

Bot trades by strategy: `day_trade_momentum` 28 (−$372, 11 W); `unknown` (mostly Aug 2025) 19 (+$78, 11 W). By exit reason: unlabeled 24 (+$107); `day_trade_flatten_hour` 6 (+$69); `proactive_macd_flipped_bearish` 5 (−$322, 0 W); `stop_loss` 4 (−$34); `proactive_rsi_below_50` 3 (−$183, 0 W); `open_desk_rsi_extreme_*` 3 (+$73, 3 W); `time_decay` 2. Sep 2026 names are movers/mega-caps (INTC, MRNA, CRWV, META, TSLA, NVDA, ORCL, SOFI, ONDS, NBIS, CRCL).

### 6.6 Decision audit (`engine_decision` lines)

The six rotating logs cover only **2026-09-22 12:29 → 2026-09-24 18:43** (three sessions; 10 MB rotation), not two weeks: 7,731 `engine_decision` lines.

| Component → action (reason) | Count |
|---|---|
| position_manager hold (not_auto_managed) — external positions | 6,031 |
| scale_out scale_out_skipped (live_partials_not_armed) | 470 |
| price_filter skip (below_min_price) | 281 |
| signal_router received (mean_reversion 146 / day_trade_momentum 62) | 208 |
| regime_allocator shadow (choppy) | 208 |
| mean_rev_quality_gate blocked | 134 |
| conviction_sizer shadow (confluence) | 68 |
| proactive_exit suppressed (below_min_age) | 30 |
| late_entry skip (after_entry_cutoff) | 25 |
| **signal_router accepted (all day_trade_momentum, mode=live)** | **18** |
| order_fill position_created / order_monitor bracket_placed | 18 / 18 |
| ml proceed (ml_disagreement_low_conf) | 16 |
| market_hours orders_held / skip | 11 / 4 |
| daytrade_flatten_hour_gate skip | 8 |
| early_session skip (first_5min_hard_block) | 7 |
| mean_rev_dedupe_gate blocked; meanrev_live_pause skip | 6; 2 |
| signal_router skip (anti_pyramid_recent_attempt) | 3 |
| Exits: stop_loss_broker 3, take_profit_broker 1, hard_stop 3, flatten_hour 3, proactive macd 3 | 13 |

Entries vs vetoes: 18 accepted against ~440 explicit blocks/skips (quality gate 134, price filter 281, late-entry 25, flatten-hour 8, early-session 7, dedupe 6, anti-pyramid 3, live-pause 2); top veto reasons `mean_rev_quality_blocked`, `below_min_price`, `after_entry_cutoff`. Accepted: TSLA, RKLB, CRWV, META, SMCI, MSFT, SOFI (09-22/23); CRWV, CRCL, GOOGL, INTC ×2, NBIS, ONDS ×2, MRNA ×2 (09-24).

### 6.7 Open positions and account scale

| Symbol | Side | Qty | Entry | managed_by_bot |
|---|---|---|---|---|
| META | short | 50 | $757.09 | False (external) |
| MU | long | 40 | $1,065.55 | False (hands-off denylist) |
| HQGE | long | 5,100 | $0.019 | False (hands-off denylist) |
| SPCX | long | 190 | $211.57 | False (hands-off denylist) |

No bot-managed positions open; `consecutive_losses=0`. Last risk-manager sync (18:25 ET): **Balance ≈ $43.5k, Buying Power ≈ $29.8k, account-wide day P&L ≈ −$810** (includes external positions; the bot-only circuit reads a separate figure).

### 6.8 Learning state (`trading_brain.json`)

| Item | Value |
|---|---|
| memories | 198 (2025-07-26 → 2026-09-24); 28 win / 170 loss |
| pattern_success_rates | manual_entry 9/122, unknown 12/51, day_trade_momentum 5/16, free_news_sentiment 0/5, momentum 0/3, mean_reversion_short 1/1 |
| symbol_behaviors | 22 symbols, mostly empty profiles (success_rate 0.5 default) |
| emotional_state | confidence 0.70, risk_appetite 0.20, patience 0.95, fear 0.50, greed 0.50 |

The `manual_entry` 113-loss count is polluted by the HQGE phantom exits, biasing brain confidence pessimistic. Latest lesson (MRNA, 09-24): "Profitable, but left money on the table."

### 6.9 Shadow ledgers and promotion gates

| File | Rows | Range |
|---|---|---|
| `data/shadow_short_log.ndjson` (mean-rev short) | 215 | 09-11 → 09-24 |
| `data/day_trade_short_shadow.ndjson` | 103 | 08-24 → 09-24 |
| `data/shadow_long_log.ndjson` (mean-rev long) | 11 | 09-17 → 09-24 |
| `data/shadow_open30_cont_index_confirm.ndjson` / `shadow_inmove_index_turn.ndjson` | 10 / 7 | 09-24 |
| `regime_allocator_shadow.ndjson` / `conviction_sizer_shadow.ndjson` / `symbol_intel_log.ndjson` | 1,439 / 1,120 / 6,525 | — |

Promotion to live requires Stage-A floors: n≥150 or ≥10 sessions, PF≥1.30, WR≥48%, expectancy≥+0.05R, DD≤6%, max losing day≤2R (`core/config.py:3203-3284`). Latest scorecard (`research/reports/2026-09-17-stage-a-scorecard-r3.md`): mean-rev long **FAIL** (n=9, PF 0.24, exp −0.62R); day-trade short **NO_GO** (n=94, PF 1.35, WR 39.6%, DD 44.7%, max simultaneous 10). Neither promoted.

### 6.10 Distinctive characteristics

- **Shadow-first**: every lane/gate ships with a `*_SHADOW` / `*_LIVE_ENFORCE` pair and an NDJSON ledger; promotion is a scorecard decision, not a code change.
- **Structured audit trail**: every decision is a JSON `engine_decision component=… action=… reason=…` line (`core/engine.py:787`), mirrored to Postgres `bot_decisions`/`bot_trades`/`bot_positions` via `DbLogger`.
- **External position hygiene**: pre-existing holdings are `managed_by_bot=False`, excluded from max_positions, correlation checks and the bot-only P&L circuit; MU/HQGE/SPCX are permanently hands-off.
- **Broker-authoritative exits**: Schwab OCO legs are the source of truth; software exits (proactive, flatten, hard stop) layer on top and skip confirmation when a broker leg is working.
- **Kill-switch layering**: lane switches, time-of-day gates, RSI≥70 vetoes, ML veto, regime hard-skips and the bot-only daily loss circuit all sit between signal and order.
- **Small, fast, half-sized**: ~$6k notional, ~20-minute holds, 0.5× day-trade sizing, no overnight risk.

## 7. Known limitations and caveats

This section collects the defects, drifts, and gaps found while reading the code, so that the operating description in the preceding sections is weighed correctly. Each item was verified against commit `01bc27c` on 2026-09-24 unless stated otherwise. Items are grouped by the layer they affect and ordered by impact within each group.

### 7.1 Decision and execution layer

- **ML confirmation gate is inverted.** `ml/models.py:630-633` computes `mapped_prediction` from the XGBoost class index (`{0: -1, 1: 0, 2: 1}`) and then returns the raw index. `core/engine.py:7657` compares it with `+1` for BUY and `-1` for SELL. A confident BUY class reads as disagreement and is vetoed at confidence ≥ 0.65; a HOLD class reads as agreement. In the three logged sessions this produced 16 `ml_disagreement_low_conf` proceeds and no high-confidence vetoes, so the practical effect today is a confidence haircut on good entries rather than blocked trades.
- **Live ML features are usually zero.** The engine feeds one day of 5-minute bars; the extractor returns all-zero features below 50 bars (`ml/features.py:333-347`), so the model scores a constant vector for the first four hours of each session. The model itself is the 2026-04-14 pickle; the weekly retrain job runs in a different checkout and never retrains this model.
- **Sizing cap can be exceeded.** `risk/manager.py:355-375` computes `position_value` before applying the notional cap, then compares that stale value against buying power and, if it exceeds it, resets `position_size` to 95% of buying power divided by price. That is larger than the cap just applied. It triggers only when the pre-cap notional exceeds buying power, which the 20%-of-buying-power cap makes unlikely but not impossible at small stop distances.
- **Brackets are long-side only.** Both OCO legs in `core/order_monitor/brackets.py:204-215` and `:906-915` use `equity_sell_limit`. A short position would receive a bracket that adds to the short. Shorts are disabled in every live lane, so this is latent.
- **Entry price is the signal price.** The immediate fill path creates the position with `signal.entry_price` rather than the executed price (`core/engine.py:1155` region); stops, targets, and R are off by the slippage until the periodic Schwab sync overwrites the entry.
- **Broker-side bracket fills bypass learning and the ledger.** `handle_full_bracket_fill` appends to `trade_history` and saves state but does not call `db_logger.log_trade` or `brain.remember_trade`. Stop-outs and take-profits executed at Schwab are invisible to Postgres and to the brain unless the external-close reconciler catches them.
- **Dead or inert controls that look live.** `TIME_STOP` is in the FSM table but never evaluated; `SpyRegimeCache` has no producer so OversoldBounceV2 Gate E always sees a flat slope; the risk manager's hot/cold streak multiplier reads a `trade_history` attribute the engine never sets; `max_portfolio_heat` is set and never read; `TradingMode.PAPER` is defined and never assigned; a second `TradingBrain()` is constructed at `core/engine.py:314` and the ML predictor keeps a reference to the first.
- **Supervisor recovery route is missing.** The circuit-broken commentary card points operators at `POST /api/supervisor/reset/{name}`, which does not exist. A circuit-broken critical task requires a process restart.
- **Cancels all working orders for a symbol.** `_cancel_existing_orders` cancels every WORKING order on the symbol before a close, including any the operator placed by hand.

### 7.2 Interfaces and security

- **The API key is injected into unauthenticated HTML.** `GET /` and `GET /activity` embed `window.TRADING_API_KEY` for the browser. The middleware now returns 401 without a bearer token (verified on the running process), but anyone who can fetch `/` on the LAN receives the key, and the server binds `0.0.0.0`. The code's own note limits this to single-user localhost.
- **Config writes are unvalidated and immediate.** `PUT /api/settings*` and `POST /api/update-config` write arbitrary keys to `Config().yaml` without re-running startup validation, and `Config` properties read live. `POST /api/settings/reset` replaces the tuned yaml with generic defaults.
- **`toggle-mode` flips to LIVE before its safety check.** The endpoint sets the mode and then errors if Schwab is not connected, leaving the engine in LIVE.
- **Cross-loop mutation.** Several endpoints await engine coroutines and mutate `positions`, `mode`, and the risk manager from the FastAPI loop while the engine runs on its own loop; per-position locks are bound to the engine loop.
- **Hard-coded database credentials.** `api/routes.py` carries a development Postgres DSN with embedded credentials as the fallback in three places (about lines 2026, 2337, 2815), and the training and ingest scripts carry the same default.
- **Runtime artifacts are tracked in git.** `trading_state.json`, `trading_brain.json`, `trading_bot.log`, and `ml_model_integrated.pkl` are in `git ls-files` despite `.gitignore`; live account state appears in every diff. Seven `.env` backups containing secrets sit in the repo root, untracked.

### 7.3 Data quality

- **Trade history is polluted.** The rolling 100-record `trade_history` contains 42 phantom HQGE exits at $0.0003 from August 2025 on a sub-penny position that never closed and is still held. They inflate loss counts and bias the brain (113 of the 122 `manual_entry` losses).
- **Brain statistics are therefore pessimistic.** `trading_brain.json` shows 28 wins to 170 losses and `risk_appetite` 0.20, largely from the phantom exits; only the `should_take_trade` gate reads the brain, so the impact is on symbol-level refusals.
- **Realized performance is thin and negative.** Bot-managed trades in the window: 47 trades, −$294, 46.8% win rate, profit factor 0.75; September 2026 alone 34 trades, −$632, 38% win rate. The Stage A scorecard fails both shadow lanes evaluated.

### 7.4 Testing and research

- **Test suite is red.** `pytest tests` at this commit: 1,582 passed, 191 failed, 28 errors, 4 skipped. Failures cluster in config patch targets (`core/__init__.py` re-exports the `config` instance and shadows the submodule), Cursor-sandbox `/workspace` paths, a stale determinism stub, and source-window regex assertions. There is no CI.
- **The money paths have no behavioral tests.** No test instantiates the engine with a mocked broker to drive order placement, bracket attachment, fill verification, reconciliation, or restart recovery. About 200 tests are regex checks on `core/engine.py` source text.
- **Backtests are cost-free.** `backtest/engine.py` charges no commissions, fees, or slippage and fills stops exactly at the barrier. Reported drawdowns in `backtest_report*.json` are unsized per-trade percentage sums. The team's own walk-forwards rate all three swing strategies FRAGILE, and the strategy trading live in September (day-trade momentum) has no walk-forward.

### 7.5 Operations

- **Three checkouts run production.** The live bot runs here; the nightly Alpaca ingest cron targets this checkout; the weekly retrain, daily-bar backfill, dead-man switch, and Sunday scan crons run in two sibling checkouts at different commits.
- **Runbook and rollback are stale.** `OPERATIONS_RUNBOOK.md` references modules and a port that no longer exist; `rollback_launch.sh` requires git tags that were never created; `deploy.sh` asserts launch dials that differ from the current yaml; nothing (systemd or otherwise) supervises the process, so the stream dead-man `os._exit(42)` ends the bot rather than restarting it.
- **Shutdown leaves positions open.** Stopping the process does not flatten; live positions rely on their Schwab OCO brackets until the operator acts.
- **`CHANGELOG.md` stops at 2026-06-09** and about 160 later commits (news bus, order monitor, day-trade desk, snapshots, ledger) are undocumented there.

## 8. Appendix

### 8.1 Repository map

| Path | Purpose |
|---|---|
| `trading_bot_commentary_updated.py` | Entry shim; imports `app` and `main` from `api/routes.py` |
| `api/routes.py`, `api/backtest.py`, `api/auth.py` | FastAPI app, all HTTP/WebSocket endpoints, auth middleware, backtest runner |
| `core/engine.py` | `TradingEngineWithCommentary`: loops, signal router, execution, exits, reconciliation |
| `core/config.py` | `ConfigManager` (yaml) and `Config` (about 240 properties, env → yaml → default) |
| `core/position_state.py` | Position FSM and allowed-exit table |
| `core/order_monitor/` | Bracket polling, fill/cancel/reject handling, re-bracket, broker-flat detection |
| `core/task_supervisor.py` | Supervised background tasks with backoff and circuit-breaking |
| `core/schwab_stream.py`, `core/schwab_stream_watchdog.py`, `core/stream_health.py` | Level-one quote stream, watchdog, dead-man switch |
| `core/price_book.py`, `core/quote_cache.py` | Single price source for exits and dashboard |
| `core/market_context.py`, `core/market_indices.py`, `core/direction_reader.py`, `core/spy_regime.py` | Regime, time-of-day, trend direction and phase |
| `core/news_bus.py`, `core/loops/news_loop.py` | News store and its only writer |
| `core/loops/screener_loop.py`, `analysis/screener.py` | Watchlist construction |
| `core/econ_calendar.py` | High-impact event blackout windows |
| `core/brain.py` | Trade memory and emotional state; `should_take_trade` gate |
| `core/commentary.py`, `core/websocket_manager.py` | Commentary cards and WebSocket broadcast |
| `core/pnl.py`, `core/ledger_integrity.py` | Day P&L selection, R-multiple ledger |
| `core/unified_exit.py`, `core/smart_target.py` | Day-trade exit policy (shadow), target selection at signal time |
| `core/decision_snapshot.py`, `core/symbol_intel.py` | Feature snapshots for training, per-symbol quality view |
| `strategies/base.py`, `strategies/builtin.py`, `strategies/news_strategy.py`, `strategies/oversold_bounce_v2.py` | Strategy interface and all lanes |
| `risk/manager.py` | Position sizing and account-level circuit |
| `sizing/conviction_sizer.py`, `allocators/regime_allocator.py` | Shadow sizing and regime allocation |
| `ml/` | Feature extractor, XGBoost model, predictor, meta inference, triple-barrier labels, purged CV |
| `data_providers/` | Schwab, yfinance, Alpaca ingest, Postgres bars, file bars, `DbLogger` |
| `backtest/` | Bar-replay backtester, metrics, Stage A scorer and CLI |
| `research/` | Walk-forwards, shadow resolvers, weekly report, drift monitor, scorecards |
| `tools/bot_monitor.py` | Read-only health monitor writing `docs/monitor/` |
| `templates/dashboard.html`, `templates/activity.html` | Served UI pages |
| `dashboardUI/` | Separate Next.js dashboard, not served by the bot |
| `tests/` | About 870 pytest tests |

### 8.2 Files written at runtime

| File | Content |
|---|---|
| `trading_state.json` | Last 100 trades, `schwab_pnl`, loss counters, live `positions_data`, full simulated positions |
| `trading_brain.json` | Last 1,000 trade memories, pattern success rates, emotional state |
| `trading_commentary.json` | Last 1,000 commentary cards |
| `error_recovery_state.json` | Failed-order retry state |
| `ml_model_integrated.pkl` | Live XGBoost model, rewritten only at shutdown |
| `trading_bot.log` (+ `.1` … `.5`) | JSON-lines log, 10 MB × 5 rotation |
| `regime_allocator_shadow.ndjson`, `conviction_sizer_shadow.ndjson`, `symbol_intel_log.ndjson` | Root-level shadow ledgers |
| `data/shadow_long_log.ndjson`, `data/shadow_short_log.ndjson`, `data/day_trade_short_shadow.ndjson`, `data/shadow_unified_exit.ndjson`, `data/shadow_open30_*.ndjson`, `data/shadow_inmove_index_turn.ndjson` | Per-lane shadow ledgers |
| `backtest_results/run_*.json`, `backtest_results/latest_report.json` | Backtest outputs |
| `docs/monitor/latest.md`, `docs/monitor/<date>/<HH-MM>.md` | Monitor reports |
| Postgres `bot_decisions`, `bot_trades`, `bot_positions`, `bot_decision_snapshots`, `bot_shadow_news_vetoes` | Mirrored audit, trades, positions, features |

### 8.3 Audit line vocabulary

Engine gates and exits log one line each:

```
engine_decision component=<gate> symbol=<SYM> action=<accepted|skip|blocked|hold|shadow|...> reason=<snake_case> mode=<live|simulation> k=v ...
```

Strategies log one line per evaluation:

```
strategy_decision strategy=<name> symbol=<SYM> action=<signal_buy|signal_sell|skip|veto|shadow|hard_veto|news_gate|size_reduced|advisory|bypass|error> reason=<snake_case> close=<px> k=v ...
```

Common `component` values: `signal_router`, `regime_gate`, `conviction_floor`, `mean_rev_quality_gate`, `mean_rev_budget_gate`, `health_gate`, `market_hours`, `econ_blackout`, `early_session`, `late_entry`, `late_entry_gate`, `daytrade_live_pause`, `daytrade_flatten_hour_gate`, `daytrade_rsi_entry_gate`, `max_positions_gate`, `correlation_guard`, `shortable_check`, `loss_cooldown`, `reentry_cooldown`, `brain`, `ml`, `risk`, `order_fill`, `order_monitor`, `position_manager`, `position_fsm`, `scale_out`, `proactive_exit`, `regime_allocator`, `conviction_sizer`, `meta_shadow`.

### 8.4 Glossary

| Term | Meaning |
|---|---|
| Lane | One strategy family with its own live/shadow flags (day-trade long, day-trade short, mean-rev, ORB, news, breakout) |
| Shadow | A lane or gate that evaluates and logs what it would do without placing or blocking orders |
| Live enforce | A shadow gate flipped to actually block entries (`*_LIVE_ENFORCE`) |
| Stage A | The fixed promotion floors a shadow lane must clear before going live |
| R, R-multiple | Profit or loss divided by initial risk (entry minus stop) |
| OCO | One-cancels-other bracket at Schwab: take-profit limit plus stop-limit |
| Managed by bot | Position the engine is allowed to exit; external and hands-off positions are display-only |
| Hands-off denylist | Symbols the bot must never manage (MU, HQGE, SPCX) |
| Broker-leg authority | If a bracket leg is WORKING or FILLED at Schwab, the software close skips operator confirmation |
| ZOMBIE | A position whose exit failed or stalled; requires operator review |
| PriceBook | Single in-process price source built from the quote stream |
| NewsBus | In-process store of scored news items with a 4-hour TTL |
| Efficiency ratio | Kaufman ratio of SPY net move to path length over 20 days; ≥ 0.30 is "trending" |

## 9. Live allocation and the sleeve lanes (2026-09-28)

On 2026-09-28 the operator directed the bot to trade a hard $5,000 slice of the
~$46.5k margin account, funded by margin (no positions sold), running the slow
long-only sleeve (`mr_20d` long, `mom_12_1` long) live. This section documents
that configuration as built in `core/config.py` § `v-go-live-5k-2026-09-28`,
`risk/allocation.py`, `risk/manager.py`, and `strategies/fit_lane_admission.py`.
Implementation detail and the arithmetic behind every number here is in
`.superpowers/sdd/2026-09-28-go-live-5k/task-1-report.md` and
`task-2-report.md`.

**Update, 13:17 ET the same day:** the operator also flipped the four intraday
builtin lanes' `*_LIVE_ENTRIES_ENABLED` flags on in `.env`
(`ORB_LIVE_ENTRIES_ENABLED=1`, `MEAN_REV_LIVE_ENTRIES_ENABLED=1`,
`DAY_TRADE_LIVE_ENTRIES_ENABLED=true`, `DAY_TRADE_SHORT_LIVE_ENTRIES_ENABLED=1`;
see 9.3). These are the `day_trade_momentum`, `day_trade_momentum_short`,
`mean_reversion`, and `orb_contraction_rvol` strategies in section 2.4 — a
separate lane family from the `fit:mr_20d`/`fit:mom_12_1` sleeve this section
otherwise describes. The flag flip does not mean every one of those lanes
trades at full size: `ENABLE_PLAYBOOK_GATE=1` (§9.1 above, §9.5 below) is the
brake. As of this go-live, `data/desk/playbook.json` carries
`dt_momentum|long|unknown` **suspended** (`size_mult` 0.0 — every
`day_trade_momentum` long entry is skipped regardless of the env flag) and
`mean_rev_intraday|long|unknown` on **probation** (`size_mult` 0.5 — every
`mean_reversion` long entry that would otherwise fire is halved). Both map via
`LANE_MAP` in `core/strategy_fit_gate.py` (§2.4, §3.2 gate #12's neighbourhood).
`day_trade_momentum_short` (`dt_momentum|short`) and `orb_contraction_rvol`
(`orb|*`) have no playbook entry yet, so they are ungoverned and trade at full
size once their own gates clear. The day-trade lane's own flatten-hour rule
(`DAY_TRADE_FLATTEN_HOUR`, default `15` i.e. 3 PM ET, `core/config.py:2761`)
is unchanged and still force-flattens/blocks new day-trade entries from that
hour on, independent of the playbook.

### 9.1 What the $5,000 allocation means

`Config().BOT_CAPITAL_ALLOCATION` (`core/config.py:4501`, env `BOT_CAPITAL_ALLOCATION`
wins over yaml `trading.bot_capital_allocation`, default `0.0` = off) is a hard
dollar ceiling the risk manager applies on top of the real Schwab account:

- **Off (default)** — `account_balance` and `buying_power` are the real Schwab
  numbers, exactly as before this feature existed.
- **On (go-live value `5000`)** — every risk decision reads
  `RiskManagerWithCommentary.effective_balance()` / `.effective_buying_power()`
  (`risk/manager.py:185-190`), which clamp the raw Schwab value at the
  allocation via `risk/allocation.py::effective_value` (`risk/allocation.py:92`).
  The raw Schwab numbers are preserved on `schwab_equity` / `schwab_buying_power`
  for logging. The clamp is applied **at read time**, not just at sync time, so
  none of the three live write sites that periodically overwrite
  `account_balance`/`buying_power` with whole-account Schwab numbers (the
  signal router, `_check_buying_power`, and the `/ws` position-cache refresh)
  can defeat it.
- **Position sizing**: the risk budget (`max_risk_amount = effective_balance()
  × RISK_PER_TRADE_PCT`), the buying-power cap, and — new — the **allocation
  headroom**: `risk/allocation.py::allocation_headroom(allocation,
  bot_open_notional)` returns `max(0, allocation − bot_open_notional)`, and no
  new position's notional may exceed it (`apply_allocation_cap`,
  `risk/allocation.py:253`, called from `risk/manager.py:577-579`). If the
  headroom is below the price of one share, size is trimmed to **0** — no
  trade — and the commentary carries `data['reason'] ==
  'allocation_exhausted'` (`risk/allocation.py:49`); a partial trim carries
  `'allocation_cap'` (`risk/allocation.py:53`). `bot_open_notional` is a sum
  over the bot's own managed positions' market value, refreshed roughly every
  5 analysis cycles alongside `bot_daily_pnl` (`core/engine.py`,
  `_compute_bot_open_notional`) — between refreshes the real ceiling on total
  exposure is `MAX_POSITIONS × min(MAX_POSITION_VALUE, allocation ×
  MAX_POSITION_VALUE_BP_PCT)`, which at the go-live settings (5 positions,
  $1,000 cap each) works out to ≈ the allocation itself.
- **Per-position cap tightens.** With the allocation on, `MAX_POSITION_VALUE_BP_PCT`
  (yaml default `0.20`) binds against the *capped* buying power, not the whole
  account: `min(MAX_POSITION_VALUE $10,000, 0.20 × $5,000)` = **$1,000 per
  position** (was `min($10,000, 0.20 × ~$93,000)` = $10,000 before the
  allocation). The bot can hold roughly 5 such positions before the headroom
  check exhausts the allocation — in line with `MAX_POSITIONS`.
- **Daily-loss stop is computed on the allocation, not the account.**
  `Config().MAX_DAILY_LOSS` (`core/config.py:308`) now accepts an env override
  in `(0, 1]` (`_env_fraction`, `core/config.py:164`) specifically so it can be
  tightened without touching `Config().yaml`; the circuit trips at
  `-MAX_DAILY_LOSS × effective_balance()` (`risk/manager.py:646`). At the
  go-live value `MAX_DAILY_LOSS=0.03` against the $5,000 allocation, that is a
  **$150 daily stop** (previously 3% of the whole account, ≈$1,395). The
  `.env` file on this host carried a legacy line `MAX_DAILY_LOSS="500"`
  (dollars, not a fraction) from an earlier configuration; `_env_fraction`
  refuses any value outside `(0, 1]` and falls back to the yaml default, so
  that line could not silently switch the circuit off. It was removed at the
  go-live restart.
- **The ledger stores the capped values; the dashboard tiles show the raw
  account.** `risk/manager.py`'s consumers (sizing, the "Risk Calculation"
  commentary, `log_trade`/`log_performance` records) all read
  `effective_balance()`/`effective_buying_power()`, so trades are recorded
  against the $5,000 sleeve. `api/routes.py`'s two account-display reads
  (`:1248-1249`, the `/api/account-stats` fallback path, and `:1953-1954`,
  the `/ws` `dashboard_update` payload) were deliberately left reading the raw
  `risk_manager.account_balance`/`.buying_power` — they are *display*, not a
  risk decision — so the dashboard will show the account's real ~$46.5k
  balance and ~$93k buying power even while every order the bot places is
  sized against $5,000. This is expected, not a bug: cross-check position
  sizes against this section's table, not the dashboard header.

### 9.2 Sleeve admission rule, in words

`Config().FIT_LANE_LIVE_ALLOWLIST` (`core/config.py:4528`, env
`FIT_LANE_LIVE_ALLOWLIST`, default `""` = off) names `(family, side[, regimes])`
combinations authorised to trade live **even without a walk-forward PASS
row**. Grammar: `family:side[:regime1|regime2]`, e.g.
`mr_20d:long,mom_12_1:long:risk_on|mixed` — `mr_20d` long is allowed in any
regime; `mom_12_1` long only when the live market-context regime reads
`risk_on` or `mixed` (a `mom_12_1` long in a `risk_off` regime is unmeasured
territory and stays shadow). The grammar, matching and the whole admission
rule live in `strategies/fit_lane_admission.py`.

For a given `(family, side)` still gated by `FIT_LANE_LIVE_ENTRIES_ENABLED`,
`admit()` (`strategies/fit_lane_admission.py:202`) returns:

1. **`pass`** — the entry's own symbol/side row is a walk-forward PASS (the
   pre-existing rule; unconditional on the allowlist).
2. **`allowlist`** — the `(family, side)` is allowlisted, the live
   `market_context_regime` (`unknown` normalizes to `mixed`) is permitted by
   the entry, **and** the symbol's own unconditional fit row is not a
   *measured loser*.
3. Otherwise the signal stays shadow (`fit_lane_shadow` if the family isn't
   live at all, `fit_lane_shadow_verdict` if it fails admission).

**Measured-loser exclusion, precisely** (`symbol_check`,
`strategies/fit_lane_admission.py:180`): a row blocks the allowlist path only
when it has `n ≥ SYMBOL_MIN_N` (60) trades **and** `exp_r < 0`. A row that is
**absent** (never measured for that family/side) or **thin** (fewer than 60
trades, or no finite `exp_r`) does **not** block — most of the universe today
falls in this bucket (in the 2026-09-28 snapshot, `mr_20d:long` had 0 measured
rows out of 539, `mom_12_1:long` 0 of 539; everything was `absent` or `thin`),
which is exactly the population the allowlist exists to admit. The label
(`absent`/`thin`/`positive`/`blocked_negative`) is written to
`reasoning["fit_lane_symbol_check"]` on every fit-lane decision line.

**Neutral sizing.** An allowlist-admitted entry has no walk-forward score
behind it, so it is deliberately **not** sized on the score-derived Kelly
haircut (which would collapse it to the weakest 0.2-strength tier). Instead
`entry_to_signal` gives it `strength = confidence =
Config().FIT_LANE_ALLOWLIST_STRENGTH` (`core/config.py:4559`, env/yaml,
default `0.6`, go-live value `0.8`) and
`reasoning["sizing_basis"] = "risk_per_trade"`, putting it on the plain
`RISK_PER_TRADE_PCT`-of-allocation sizing path instead of the fit-score path
(`sizing_basis = "fit_score"` for a `pass`-admitted entry).

**Every downstream gate still applies.** Allowlist admission changes nothing
about desk limits, the playbook gate, the swing overnight-hold gate, risk
sizing, or the allocation cap — it only changes whether the fit lane emits a
live signal in the first place. Desk limits and `MAX_POSITIONS` are the
remaining breadth control, since the per-symbol fit evidence no longer
narrows the sleeve's universe at all once neither row type blocks.

**`STRATEGY_FIT_SHADOW` stays `1` (on) at go-live**, and must. `ENABLE_STRATEGY_FIT_GATE`
+ `STRATEGY_FIT_SHADOW=False` would make the engine-side strategy-fit gate
(`core/config.py:4157`) **enforce** a walk-forward PASS on every entry,
including the allowlisted sleeve's — since no PASS rows exist, that would
block the sleeve entirely. The allowlist in `strategies/fit_lane_admission.py`
*is* the sleeve's gate; the engine-side fit gate stays in shadow (audit-only)
so it does not double-govern the same entries with a stricter, PASS-only rule.

### 9.3 The go-live `.env` block

Applied at the post-close restart on 2026-09-28 (after removing the legacy
`MAX_DAILY_LOSS="500"` line), then extended intraday. This is the block as it
stands in `.env` today (go-live 11:52 ET, streak reset added shortly after,
restart 13:13 ET onto `d66380d`, the four intraday `*_LIVE_ENTRIES_ENABLED`
flags below flipped on at 13:17 ET) — names and values, verbatim:

```
# v-go-live-5k-2026-09-28: live sleeve inside a $5,000 margin-funded allocation (user directive 2026-09-28)
BOT_CAPITAL_ALLOCATION=5000
RISK_PER_TRADE_PCT=0.01
MAX_DAILY_LOSS=0.03
FIT_LANES_ENABLED=dt_momentum,gap_play,vwap_reversion,mean_rev_intraday,mr_20d,mom_12_1
FIT_LANE_LIVE_ENTRIES_ENABLED=mr_20d,mom_12_1
FIT_LANE_LIVE_ALLOWLIST="mr_20d:long,mom_12_1:long:risk_on|mixed"
FIT_LANE_ALLOWLIST_STRENGTH=0.8
SWING_ALLOW_OVERNIGHT=1
SWING_MAX_HOLD_SESSIONS=20
ENABLE_DESK_LIMITS=1
ENABLE_PLAYBOOK_GATE=1
LEDGER_INTEGRITY=1

# streak reset at go-live (user, 2026-09-28): sessions before this date never count as losing days
DESK_STREAK_SINCE=2026-09-28

# intraday builtin lanes taken live 13:17 ET 2026-09-28 (user directive) — see §9.1 update and §2.4
ORB_LIVE_ENTRIES_ENABLED=1
MEAN_REV_LIVE_ENTRIES_ENABLED=1
DAY_TRADE_LIVE_ENTRIES_ENABLED=true
DAY_TRADE_SHORT_LIVE_ENTRIES_ENABLED=1
```

`STRATEGY_FIT_SHADOW` is **not** set (stays at its default `True`) — see 9.2.
`FIT_LANES_ENABLED` lists every family already registered before go-live
(`dt_momentum,gap_play,vwap_reversion,mean_rev_intraday`) plus the two
appended for the sleeve (`mr_20d,mom_12_1`) — none of the pre-existing
families were removed. `SWING_ALLOW_OVERNIGHT=1` / `SWING_MAX_HOLD_SESSIONS=20`
(`core/config.py:4278`, `:4291`) let the sleeve's multi-day holds carry past a
single session, capped at 20 sessions. `DESK_STREAK_SINCE=2026-09-28`
(`core/config.py:4593`, `core/loops/desk_loop.py::_streak_since`) floors the
losing-day-streak lookback at go-live day — sessions before it can never
trigger the desk's `two_losing_days` sit-out rule (see `docs/desk.md`).

The four intraday flags are a separate switch from everything else in this
block: they gate the builtin `day_trade_momentum`, `day_trade_momentum_short`,
`mean_reversion`, and `orb_contraction_rvol` strategies' own
`*_LIVE_ENTRIES_ENABLED` checks (§2.4, §3.2 gates #13/#17/#18), not the
`fit:mr_20d`/`fit:mom_12_1` sleeve. Flipping them on does not by itself size
every lane up — `ENABLE_PLAYBOOK_GATE=1` still suspends `dt_momentum|long`
(day-trade momentum long is skipped outright) and holds
`mean_rev_intraday|long` on 0.5x probation; see the update note at the top of
§9 and §9.5.

**`.env` quoting rule.** `FIT_LANE_LIVE_ALLOWLIST` must be quoted —
`FIT_LANE_LIVE_ALLOWLIST="mr_20d:long,mom_12_1:long:risk_on|mixed"` — because
a bare, unquoted `|` in a value is a pipe to any shell that `source`s `.env`
(e.g. `scripts/nightly_strategy_hunt.sh`, an operator's own `source .env`),
which breaks the export. The bot's own Python env loader was never affected
by the unquoted form (it doesn't shell-interpret the file), but the value was
quoted at go-live anyway so every consumer of `.env` agrees. If you edit this
line, keep the quotes.

### 9.4 Sizing table

Arithmetic for a $100 stock with a 3% stop, allocation `$5,000`,
`RISK_PER_TRADE_PCT=0.01` (→ `max_risk_amount = $50`, `risk_per_share = $3`,
16 base shares), `MAX_POSITION_VALUE_BP_PCT` default `0.20` (→ `$1,000`
per-position cap), nothing else open (verified against the live sizing chain
in `.superpowers/sdd/2026-09-28-go-live-5k/task-2-report.md` fix round 1):

| Signal strength | Kelly fraction | Shares | Notional | Dollar risk |
|---|---|---|---|---|
| NaN (unmeasured, old behaviour) | 0.20 | 3 | $300 | $9 |
| 0.6 (`FIT_LANE_ALLOWLIST_STRENGTH` code default) | 0.50 | 8 | $800 | $24 (0.48% of allocation) |
| 0.7 | 0.55 | 8 | $800 | $24 |
| **0.8 (go-live value)** | 0.70 | **10** (BP-capped) | **$1,000** | **$30 (0.6% of allocation)** |
| 1.0 (a genuine walk-forward PASS, `sizing_basis=fit_score`) | 1.00 | 10 (BP-capped) | $1,000 | $30 |

At `FIT_LANE_ALLOWLIST_STRENGTH=0.8`, an allowlist-admitted entry sizes
identically to a maximal-conviction PASS entry (both hit the $1,000
buying-power cap) — that cap, not the Kelly curve, is what binds at go-live.
Allocation exhaustion: once $4,950 of the $5,000 is already open, headroom is
$50 — a $100 stock cannot fit one share, so the next signal sizes to 0 with
`allocation_exhausted`.

### 9.5 What to watch on the first live day

Grep `trading_bot.log` (JSON-lines) for:

- `fit_lane_admission=allowlist` — a sleeve entry admitted without a PASS row
  (component `fit_lane_live` in the `strategy_decision` line;
  `market_context_regime=` and `fit_lane_symbol_check=` ride along on the same
  line, `strategies/fit_lane.py:327-330`).
- `allocation_exhausted` / `allocation_cap` — the allocation headroom blocked
  or trimmed a position (`risk/allocation.py:49,53`).
- `component=desk_limit` — a sit-out day, `max_trades` breach, or disallowed
  direction now actually skips the entry (`ENABLE_DESK_LIMITS=1` is enforcing,
  not shadow, at go-live).
- `component=playbook` — a suspended `(family, side, regime)` key skips the
  entry, or a probation/enabled key's size multiplier is applied
  (`ENABLE_PLAYBOOK_GATE=1` is enforcing at go-live; see `docs/desk.md`).

Read-only endpoints: `GET /api/desk/plan`, `/api/desk/playbook`,
`/api/desk/journal` (`api/routes.py:1018,1045,1065`) for the day's plan,
current playbook state, and (after 16:05 ET) the rendered journal
`data/desk/journal/<date>.md`. `GET /api/account-stats` and the dashboard's
`account` block still show the **raw** Schwab account (see 9.1) — do not use
them to check whether the allocation is exhausted; use the `allocation_*`
audit lines or a manual sum of open bot-managed notional instead.

### 9.6 How to stop trading, in one step

- **Fastest, no restart:** `POST /api/stop` (`api/routes.py:716-721`) sets
  `trading_engine.is_running = False`. This halts the trading loops
  immediately; the FastAPI server and dashboard stay up, and open positions'
  brackets remain live at the broker (this does not flatten anything).
- **Persistent, requires a restart:** set `FIT_LANE_LIVE_ENTRIES_ENABLED=`
  (empty) in `.env` and restart via `./deploy.sh`. This drops `mr_20d` and
  `mom_12_1` back to shadow-only — they keep evaluating and logging but never
  place a live order — while leaving `BOT_CAPITAL_ALLOCATION` and everything
  else in place, so a later restart with the flag restored resumes exactly
  where the sleeve left off (shadow ledgers are never deleted by this).

### 9.7 Nightly cron

`crontab -l` on this host runs, Monday–Friday at 22:30 ET:

```
30 22 * * 1-5 /home/nvidia/claude/grok/trading_bot_commentary/scripts/nightly_strategy_hunt.sh >> /home/nvidia/claude/grok/trading_bot_commentary/data/strategy_hunt/nightly/cron.log 2>&1
```

`scripts/nightly_strategy_hunt.sh` sources `.env`, resolves `POSTGRES_DSN`,
and runs `python -m research.strategy_hunt all --workers 12 --postgres`,
appending to `data/strategy_hunt/nightly/<date>.log` (its own start/end
banner lines plus the pipeline's own output); `cron.log` above only captures
anything cron itself couldn't route into that per-date log. `all --postgres`
runs `export → news → earnings → context → calibrate → run → run(shuffle) →
score → playbook`, so it also rewrites `data/desk/playbook.json` and
`context/universe_daily.parquet` (see `docs/research/strategy_hunt.md` §
"Scheduling" and § "Universe context" for the `mom_12_1` context-freshness
requirement this run maintains). A test run on 2026-09-28 completed in 17
minutes (rc 0). If `fit:mom_12_1` stops signalling entirely, check the mtime
of `context/universe_daily.parquet` and this job's log first (see 9.8's carry
window).

### 9.8 Watchdog and the wedge fix (landed 2026-09-28)

`scripts/bot_watchdog.sh`, `core/provider_calls.py`, `core/isolated_calls.py`,
and `core/loop_heartbeat.py` are committed and were on the live process from
the go-live restart onward. Landing history: **066174e** (bounded provider
calls, isolated yfinance subprocess, loop heartbeat, external watchdog) →
**feba6ca** (strategy-alias calls bounded, pool rebuild on probe failure,
watchdog checks engine state, screener fails closed) → **56bf7fd** (bounded
broker calls with explicit timeout errors; active-open-desk and desk scan
bounded; fail-closed vetoes on provider timeout) → **a5382bc** (screener
quote uses the 3 s quote budget; close-timeout audit says the position is
unprotected) → tests pinned at **d66380d**, which is the build the bot has
run on since the 13:13 ET restart. A further restart at 13:17 ET (pid
`1301804`, current) picked up the four intraday `*_LIVE_ENTRIES_ENABLED`
flags (§9.1 update, §9.3) — same `d66380d` build, `.env` only. Full facts,
RED/GREEN evidence and round-by-round findings:
`.superpowers/sdd/2026-09-28-go-live-5k/task-4-report.md`.

The trigger: on 2026-09-28 the live process wedged (247 of 251 threads parked
in `futex_wait`, `/api/status` unresponsive for 60+ seconds, no desk plan
built) after a synchronous yfinance/curl_cffi call held the GIL. The fix has
four layers:

1. **Bounded provider calls.** Every synchronous market-data call
   (`self.data_provider.get_quote(...)` / `get_market_data(...)`, plus the
   five `schwab_provider`-alias reads inside `strategies/builtin.py`) reachable
   from the engine's event loop routes through `provider_call` /
   `provider_call_sync` (`core/provider_calls.py`), backed by one dedicated
   `ThreadPoolExecutor` sized `Config().PROVIDER_EXECUTOR_WORKERS` (default
   `8`, `core/config.py:4604`). Quote calls get `Config().QUOTE_FETCH_TIMEOUT_SEC`
   (3 s, `core/config.py:1320`); bar/history calls get
   `Config().PROVIDER_HISTORY_TIMEOUT_SEC` (10 s, `core/config.py:4628`). A
   timeout returns `None`, logs at most once a minute per call site
   (`provider_call_timeout`, below), and the caller keeps its existing
   fallback. `core/engine.py` has zero bare `get_quote`/`get_market_data`
   calls (enforced by a wiring test); the position loop's own quote read is
   the critical one — see the `quote_timeout` playbook entry below.
2. **Bounded broker calls.** The 20 synchronous Schwab order/account calls
   (`place_order`, `get_order`, `cancel_order`, `get_orders_for_account`,
   `get_account`, `get_quotes`) route through `broker_call` /
   `broker_call_sync`, budget `Config().BROKER_CALL_TIMEOUT_SEC` (default
   15 s, `core/config.py:4661` — deliberately longer than the market-data
   budgets, since cutting an order call short leaves its fate at the broker
   unknown). On timeout these **raise** `BrokerTimeoutError` rather than
   returning `None`: the entry path does not retry (the order may already be
   live at Schwab; `_check_order_status` reconciles), and the close path is
   the `zombie_pending`/`broker_timeout_close` case below.
3. **Isolated yfinance subprocess pool** (`core/isolated_calls.py`) — all
   yfinance usage (screener refresh, news lookups, the fallback realtime
   provider) runs in a spawn-context `ProcessPoolExecutor` (2 workers) that is
   **SIGKILLed and rebuilt** on a timeout, so a GIL-holding hang in a child
   process cannot touch the engine's own loop.
   `Config().ISOLATED_YF_ENABLED` (default `True`, `core/config.py:4683`) is
   the operator's escape hatch: set `0` to route the same calls back through
   the bounded thread pool (still timeout-bounded, no longer GIL-proof) if
   the subprocess pool itself becomes the problem.
4. **Loop heartbeat + external watchdog.** `core/loop_heartbeat.py`
   (`Config().LOOP_LAG_WARN_SEC`, default 5 s, `core/config.py:4645`) updates
   a timestamp every second on each loop and logs `loop_lag loop=<name>
   lag=<s> threads=<n> threshold=<s>` when it falls behind — with the caveat
   documented in the module itself: a true GIL-holding hang freezes this
   thread too, which is why an **external** restarter exists:
   `scripts/bot_watchdog.sh`, cron-installed (see below).

#### Operator playbook: reading the new audit/log lines

- **`position_manager … skip … quote_timeout`** (`core/engine.py`, position
  loop). The bot could not get a fresh quote *and* has no last-known price
  for the symbol within this cycle. It skips managing that position for the
  cycle rather than act on a price it does not have — no action needed unless
  this repeats for the same symbol across multiple cycles, which points at a
  broken quote feed for that symbol specifically.
- **`position_manager … zombie_pending … broker_timeout_close`**
  (`core/engine.py:2872`, `BrokerTimeoutError` on the close-order call). This
  is the one that needs a human. The bracket was already cancelled before the
  close was attempted, Schwab did not answer within
  `BROKER_CALL_TIMEOUT_SEC` (15 s), and the bot marks the position ZOMBIE:
  **it has no broker-side protection (no stop, no take-profit) and no
  automatic exits until you resolve it.** OPERATOR ACTION REQUIRED — check
  the order/position directly at Schwab and close or re-protect it manually;
  do not assume the bot will retry.
- **`provider_call_timeout site=<tag> timeout=<s>s suppressed=<n>
  threads=<n> — caller falls back`** (`core/provider_calls.py`, rate-limited
  to one line per site per 60 s). A market-data or `schwab_provider`-alias
  call timed out; the caller degraded per its documented fallback (skip,
  keep last value, fail-closed veto — see the round reports for the full
  per-site table). A single line is normal under a slow provider. Bursts
  across many sites, or the same site repeating every cycle, means the
  provider is genuinely degraded — check Schwab/Yahoo status before touching
  any timeout knob; if provider concurrency legitimately needs more than 8
  in-flight calls, raise `PROVIDER_EXECUTOR_WORKERS` rather than the
  timeouts.
- **`loop_lag loop=<name> lag=<s> threads=<n> threshold=<s>`**
  (`core/loop_heartbeat.py`, WARNING, rate-limited to one line per loop per
  minute). A registered loop (position loop, analysis loop, desk loop, the
  API's own heartbeat) has not pulsed for longer than `LOOP_LAG_WARN_SEC`
  (5 s). This is the **in-process** early warning; if the process is still
  logging at all, this is the loop that is stuck, which narrows where to
  look. It cannot see a GIL-holding hang (the heartbeat thread freezes too in
  that case) — a `loop_lag` line means the process is alive but a loop
  stalled; **silence for minutes with no other log activity** is the
  GIL-wedge signature the external watchdog exists to catch instead.

#### The external watchdog (`scripts/bot_watchdog.sh`)

Cron, installed live (not `--dry-run`) on this host (`crontab -l`):

```
*/5 6-19 * * 1-5 /home/nvidia/claude/grok/trading_bot_commentary/scripts/bot_watchdog.sh >> /home/nvidia/claude/grok/trading_bot_commentary/bot_watchdog.log 2>&1
```

Every 5 minutes, 06:00–19:55 local server time, Monday–Friday (the script
converts to ET internally via `TZ=America/New_York date …` for its own
09:25–16:05 live-window check, so it does not depend on cron's environment):
`curl -m 30 GET /api/status`, parsed. Healthy requires **all**
of: HTTP 200, `"bot_running": true`, and — inside the live window (09:25–16:05
ET, `WATCHDOG_LIVE_FROM`/`WATCHDOG_LIVE_TO`, unless `WATCHDOG_REQUIRE_LIVE=0`
or `=1` override the auto window) — `"mode": "live"`. State
(`bot_watchdog.state`) carries the consecutive-failure count and today's
restart count across cron runs. On the **second consecutive** failure, and
only if fewer than `MAX_RESTARTS_PER_DAY` (default 2) restarts have happened
today, it restarts the bot: TERM → 20 s grace → KILL if still alive, then
`nohup python trading_bot_commentary_updated.py`, poll `/api/status` up to
120 s, `POST /api/start`, sleep 25 s, `POST /api/toggle-mode
{"mode":"live"}`. A restart budget already spent that day, or a restart that
does not come back healthy, logs `OPERATOR ACTION REQUIRED` and exits
non-zero — that is the signal to look at `bot_watchdog.log` and intervene by
hand rather than wait for another cron tick. `--dry-run` reports only (no
signal, no launch, no state write); the API key is read from `.env` only
when not already exported and passed to curl via a mode-600 header file, so
it never appears in `ps` or the log.

**To pause the watchdog** without removing the fix itself:
- Outside RTH but want it to stop expecting live mode:
  `WATCHDOG_REQUIRE_LIVE=0` (exported, or added as a line the script's
  environment picks up) suppresses the `mode_not_live` failure reason so a
  deliberate simulation-mode period does not accumulate failures/restarts.
- To stop it from running at all: remove (or comment out) the `*/5 6-19 * *
  1-5 …` line from `crontab -e`. The script itself is inert until cron (or an
  operator) invokes it — nothing else calls it automatically.

### 9.9 Risk Console

The Risk Console (`core/settings_console.py`, `core/settings_console_store.py`,
`GET/PUT /api/console/*`, the dashboard card) is a live-editable registry of
risk knobs backed entirely by `Config().yaml` — it never writes `.env`. Each
knob (`KNOB_BY_KEY` in `core/settings_console.py`) carries a `kind`
(`money`/`fraction`/`int`/`bool`/`csv`), a valid range or allowlist, and a
`loosen` direction used to classify every change.

**The ratchet rule.** A change is classified `tighten` (more conservative),
`loosen` (more permissive) or `noop` by comparing the new value against the
knob's current effective value (`core/settings_console.py::classify`). A
`tighten` applies immediately — the operator wants the risk envelope smaller
*right now*, so there is no reason to wait. A `loosen` is queued instead: it
is written to `<DESK_DATA_DIR>/pending_settings.json` and only takes effect
at the next session's settings-promotion step, **09:25 ET** by default
(`Config().SETTINGS_PROMOTE_TIME_ET`, yaml `trading.settings_promote_time_et`
— env has no override, deliberately: this is a desk-loop schedule knob, not
something a stray `.env` line should be able to move), fired once per session
day by the desk loop (`core/loops/desk_loop.py`, same `DeskLoop.tick()` that
drives the 09:00/09:35/16:05 desk steps — see `docs/desk.md` § "Timeline").
This gives a human a full session's notice before any risk limit actually
gets more permissive, while a tightening protective change never waits on
anyone.

**What stays file-only, and why.**
- `DESK_STREAK_SINCE` — not a console knob at all. It is read directly from
  the environment by `core/loops/desk_loop.py` to floor the losing-streak
  sit-out lookback (`docs/desk.md`'s flags table); editing the streak-reset
  date from a web form is exactly the kind of one-off, rarely-touched
  operator action that belongs in `.env`, not in a change log meant for
  routine risk tuning.
- `trading.settings_console.allocation_ceiling` — the dollar ceiling the
  `money`-kind knobs (`bot_capital_allocation`) are validated against
  (`core/settings_console.py::_coerce_money`, default `$5,000`
  `DEFAULT_ALLOCATION_CEILING`). It is set once by the migration script (see
  below) or by hand in yaml, never by a console PUT — raising your own
  ceiling from the same console that is supposed to be constrained by it
  would defeat the point.
- `FIT_LANES_ENABLED` — the knob exists in the registry (`restart_required=True`)
  purely so its value is still validated, but the console **refuses writes
  to it** (`core/settings_console_store.py::submit_change` rejects any
  `restart_required` knob with reason `restart_required`). Which fit-lane
  families are loaded at all is a process-start decision
  (`strategies/fit_lane_admission.py` reads it once at boot); changing it
  without a restart would silently desync the running process from what it
  reports.

**The env-shadowing rule.** Every other console knob also has an `.env`
name (`Knob.env_name`). Per this repo's normal precedence (env wins over
yaml wins over the code default, `docs/desk.md` § "The four flags"), an
`.env` line for a console knob **shadows** the console entirely: a PUT to
that key is rejected with `env_shadowed: <NAME> is set in .env; the yaml
value is ignored until it is removed` (`core/settings_console_store.py::submit_change`),
and `GET /api/console` marks that knob `env_shadowed: true` so the dashboard
card can show it as non-editable rather than silently accepting an edit that
would do nothing. The fix is to migrate the value out of `.env` and into
yaml once, so the console becomes the single source of truth for it going
forward.

**Migration.** `scripts/migrate_console_settings.py` does the cutover for
every console knob still set in `.env` (except `FIT_LANES_ENABLED` and
`DESK_STREAK_SINCE` above, and `MAX_DAILY_LOSS` unless its `.env` value
parses as a fraction in `(0, 1]` — this machine's `.env` once held a
*dollar* `MAX_DAILY_LOSS=500`, and migrating that literally into the
fraction-typed yaml key would set the daily-loss circuit to 500x equity, see
`core/config.py::_env_fraction`). Dry-run first, review, then apply:

```bash
python scripts/migrate_console_settings.py --env .env --yaml "Config().yaml"
python scripts/migrate_console_settings.py --env .env --yaml "Config().yaml" --apply
```

Dry-run prints only the knobs it would migrate (`NAME=value → trading.<key>:
<yaml value>`) and the `trading.settings_console.allocation_ceiling` line it
would add — never any other `.env` line, so a secret such as
`TRADING_API_KEY` sitting in `.env` is never echoed. `--apply` backs up both
files (`<file>.bak_console_<YYYYmmdd_HHMMSS>`), writes the yaml keys via
`ConfigManager.update(...)`, and comments out each migrated `.env` line in
place as `# migrated-to-yaml <date>: NAME=<original value>` — every other
`.env` line, migrated or not, is preserved byte-for-byte (every duplicate
declaration of a migrated name is commented out, not just the last one).

**Stop the bot first, then migrate, then restart — as one sequence.** A
running process holds its own in-memory copy of the yaml and rewrites the
whole file on its next `ConfigManager.update()` (a hands-off save, a block
click), which would silently erase the migrated keys while the `.env` lines
are already commented out. `--apply` therefore refuses (exit 4) while
anything answers on the bot port (`--port`, default 9000); `--force`
overrides that and is not recommended. Before writing, the script also runs
`ConfigManager.validate_candidate` on the planned yaml and aborts with exit 3
(no writes, no backups left behind) if the loader would refuse it. The
backups and temp files (`.env.bak_console_*`, `.env.tmp`,
`Config().yaml.*`) hold secrets and are git-ignored.

**Validation band.** Every console write — a tighten applied now or a
loosen promoted later — is checked against the same validation the config
loader runs at start-up (`ConfigManager.validate_candidate`: daily loss
1%–20%, positions 1–50, …) on a copy *before* it is persisted, and the
registry bounds mirror that band (per-trade risk is capped at 5% in the
console; anything larger is file-only). A queued loosen is re-validated
against that day's allocation ceiling and band when it is promoted; if it no
longer passes it is dropped with a `promotion_rejected` audit row rather
than applied or retried forever.

**Promotion needs the desk loop.** Pending loosens are promoted by the desk
loop's 09:25 step, so with `ENABLE_DESK` off they would sit "pending"
forever. `GET /api/console` reports `promotion_enabled` and the card shows a
red banner when it is false.

**The block switch.** `POST /api/console/block-entries` sets yaml-only
`trading.block_new_entries` (`Config().BLOCK_NEW_ENTRIES`, no env override),
an immediate kill switch for *new* entries only — exits and position
management are untouched, checked at the top of the signal router before
any strategy or gate runs. Every signal skipped this way logs:

```
engine_decision component=signal_router symbol=<SYM> action=skip reason=entries_blocked_by_console
```

Un-blocking requires `confirm: "YES"` in the request body
(`core/settings_console_store.py::set_block_new_entries`) — a deliberate
extra step so a misclick cannot silently resume live entries.

**Endpoints** (all under `/api/console`, all file-backed, never a 500 — see
`core/settings_console_store.py`): `GET /api/console` (every knob's current
value, direction it would classify as, and `env_shadowed` status);
`PUT /api/console/{key}` (submit a change — applies if a tighten, queues if
a loosen); `POST /api/console/block-entries` (set/clear the kill switch,
`confirm: "YES"` required to clear); `DELETE /api/console/pending/{id}`
(cancel a queued loosen before it promotes); `GET /api/console/audit` (the
`settings_audit.ndjson` trail — every applied, queued, cancelled, promoted,
rejected and block/unblock event).

### 9.10 Ledger-integrity migration

Applied 2026-09-28, 09:30 ET, during a quiet window, via
`scripts/migrate_ledger_integrity.py` (idempotent `ADD COLUMN IF NOT EXISTS`
with per-column savepoints and lock/statement timeouts) — adds
`initial_stop`, `initial_tp`, `initial_risk_per_share`, `pnl_r`, `setup_type`,
`hold_time_seconds`, `source`, `is_hands_off`, `is_external`,
`exit_reason_raw` to Postgres `bot_trades`. `Config().LEDGER_INTEGRITY`
(`core/config.py:1947`, env/yaml, default `False`) is set to `1` at the
go-live restart, which switches on: broker-orphan reconciliation (fills that
happened at the broker but weren't recorded by the bot), TRUE
initial-stop/TP tracking (immutable, separate from the dynamic trailing
stop), and R computed from `pnl_r` directly rather than derived from a
closing stop that may have moved (see `docs/desk.md` § "Known limitations"
for what that derivation got wrong before the migration). The migration
script's `--dry-run` flag is currently broken (it autocommits inside a
transaction) — noted here so it isn't relied on; the real run was validated
by checking column presence afterward instead.
