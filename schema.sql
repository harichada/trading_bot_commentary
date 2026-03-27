-- =============================================================================
-- Rudra Trading Engine — Schema
-- Source of truth for all table definitions.
-- Applied by: make db-migrate (via scripts/migrate.sh)
-- All statements are idempotent (IF NOT EXISTS / exception handling).
-- =============================================================================

-- Migration tracking
CREATE TABLE IF NOT EXISTS schema_migrations (
    version     TEXT PRIMARY KEY,
    applied_at  TIMESTAMPTZ DEFAULT NOW(),
    description TEXT DEFAULT ''
);

-- =============================================================================
-- DAILY PRICE DATA
-- =============================================================================
CREATE TABLE IF NOT EXISTS daily_bars (
    symbol  TEXT NOT NULL,
    date    TEXT NOT NULL,
    open    DOUBLE PRECISION,
    high    DOUBLE PRECISION,
    low     DOUBLE PRECISION,
    close   DOUBLE PRECISION,
    volume  DOUBLE PRECISION,
    PRIMARY KEY (symbol, date)
);
CREATE INDEX IF NOT EXISTS idx_bars_date_symbol ON daily_bars (date, symbol);

-- =============================================================================
-- TRADES (completed trade log)
-- =============================================================================
CREATE TABLE IF NOT EXISTS trades (
    trade_id        SERIAL PRIMARY KEY,
    symbol          TEXT NOT NULL,
    entry_price     DOUBLE PRECISION NOT NULL,
    exit_price      DOUBLE PRECISION NOT NULL,
    shares          INTEGER NOT NULL,
    pnl             DOUBLE PRECISION NOT NULL,
    pnl_pct         DOUBLE PRECISION NOT NULL,
    entry_time      TEXT NOT NULL,
    exit_time       TEXT NOT NULL,
    exit_reason     TEXT NOT NULL,
    holding_minutes INTEGER DEFAULT 0,
    side            TEXT DEFAULT 'short',
    gap_pct         DOUBLE PRECISION DEFAULT 0.0,
    vol_ratio       DOUBLE PRECISION DEFAULT 0.0,
    score           DOUBLE PRECISION DEFAULT 0.0,
    catalyst        TEXT DEFAULT '',
    strategy_id     TEXT DEFAULT '',
    setup_type      TEXT DEFAULT '',
    UNIQUE(symbol, entry_time, exit_time, shares, exit_reason)
);
CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades (entry_time);
CREATE INDEX IF NOT EXISTS idx_trades_symbol_entry ON trades (symbol, entry_time);

-- =============================================================================
-- JOURNAL ENTRIES
-- =============================================================================
CREATE TABLE IF NOT EXISTS journal_entries (
    id          SERIAL PRIMARY KEY,
    timestamp   TEXT NOT NULL,
    entry_type  TEXT NOT NULL,
    source      TEXT NOT NULL,
    symbol      TEXT DEFAULT '',
    content     TEXT NOT NULL,
    data        TEXT DEFAULT '{}',
    llm_call    BOOLEAN DEFAULT FALSE,
    UNIQUE(timestamp, entry_type, source, symbol, content)
);
CREATE INDEX IF NOT EXISTS idx_journal_ts ON journal_entries (timestamp);
CREATE INDEX IF NOT EXISTS idx_journal_type ON journal_entries (entry_type, timestamp);

-- =============================================================================
-- MARKET EVENTS
-- =============================================================================
CREATE TABLE IF NOT EXISTS market_events (
    id          SERIAL PRIMARY KEY,
    timestamp   TEXT NOT NULL,
    mono_time   DOUBLE PRECISION NOT NULL,
    event_type  TEXT NOT NULL,
    tier        INTEGER NOT NULL,
    symbol      TEXT DEFAULT '',
    description TEXT DEFAULT '',
    data        TEXT DEFAULT '{}',
    dedup_key   TEXT DEFAULT '',
    UNIQUE(timestamp, event_type, symbol, dedup_key)
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON market_events (timestamp);
CREATE INDEX IF NOT EXISTS idx_events_type ON market_events (event_type, timestamp);

-- =============================================================================
-- LLM CALLS
-- =============================================================================
CREATE TABLE IF NOT EXISTS llm_calls (
    id                SERIAL PRIMARY KEY,
    timestamp         TEXT NOT NULL,
    conversation_role TEXT NOT NULL,
    priority          TEXT NOT NULL,
    model             TEXT DEFAULT '',
    prompt_tokens     INTEGER DEFAULT 0,
    response_tokens   INTEGER DEFAULT 0,
    duration_ms       INTEGER DEFAULT 0,
    success           INTEGER DEFAULT 1,
    budget_5min       INTEGER DEFAULT 0,
    budget_1hr        INTEGER DEFAULT 0,
    response_summary  TEXT DEFAULT '',
    UNIQUE(timestamp, conversation_role)
);
CREATE INDEX IF NOT EXISTS idx_llm_ts ON llm_calls (timestamp);

-- =============================================================================
-- CONFIG PROFILES & HISTORY
-- =============================================================================
CREATE TABLE IF NOT EXISTS config_profiles (
    name        TEXT PRIMARY KEY,
    config_json TEXT NOT NULL,
    description TEXT DEFAULT '',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS config_history (
    id              SERIAL PRIMARY KEY,
    timestamp       TEXT NOT NULL,
    source          TEXT NOT NULL,
    changes_json    TEXT NOT NULL,
    config_snapshot TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_config_history_ts ON config_history (timestamp);

-- =============================================================================
-- API CALLS
-- =============================================================================
CREATE TABLE IF NOT EXISTS api_calls (
    id                   SERIAL PRIMARY KEY,
    timestamp            TEXT NOT NULL,
    endpoint             TEXT NOT NULL,
    method               TEXT NOT NULL DEFAULT 'GET',
    symbols              TEXT DEFAULT '',
    params_json          TEXT DEFAULT '{}',
    status_code          INTEGER DEFAULT 0,
    response_time_ms     DOUBLE PRECISION DEFAULT 0,
    error                TEXT DEFAULT '',
    rate_limit_remaining INTEGER DEFAULT -1
);
CREATE INDEX IF NOT EXISTS idx_api_calls_ts ON api_calls (timestamp);
CREATE INDEX IF NOT EXISTS idx_api_calls_endpoint_ts ON api_calls (endpoint, timestamp);

-- =============================================================================
-- PERFORMANCE SNAPSHOTS
-- =============================================================================
CREATE TABLE IF NOT EXISTS performance_snapshots (
    id                      SERIAL PRIMARY KEY,
    timestamp               TEXT NOT NULL,
    interval_type           TEXT NOT NULL DEFAULT 'hourly',
    equity                  DOUBLE PRECISION DEFAULT 0,
    cash                    DOUBLE PRECISION DEFAULT 0,
    open_positions          INTEGER DEFAULT 0,
    unrealized_pnl          DOUBLE PRECISION DEFAULT 0,
    realized_pnl_today      DOUBLE PRECISION DEFAULT 0,
    trades_today            INTEGER DEFAULT 0,
    wins                    INTEGER DEFAULT 0,
    losses                  INTEGER DEFAULT 0,
    win_rate                DOUBLE PRECISION DEFAULT 0,
    max_drawdown            DOUBLE PRECISION DEFAULT 0,
    strategy_breakdown_json TEXT DEFAULT '{}',
    top_symbols_json        TEXT DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_perf_snap_ts ON performance_snapshots (timestamp);
CREATE INDEX IF NOT EXISTS idx_perf_snap_interval_ts ON performance_snapshots (interval_type, timestamp);

-- =============================================================================
-- INTRADAY SIGNALS
-- =============================================================================
CREATE TABLE IF NOT EXISTS signals_intraday (
    id                SERIAL PRIMARY KEY,
    timestamp         TEXT NOT NULL,
    symbol            TEXT NOT NULL,
    strategy_id       TEXT DEFAULT '',
    signal            TEXT DEFAULT '',
    direction         TEXT DEFAULT '',
    confidence        DOUBLE PRECISION DEFAULT 0,
    entry_price       DOUBLE PRECISION DEFAULT 0,
    stop_price        DOUBLE PRECISION DEFAULT 0,
    target_price      DOUBLE PRECISION DEFAULT 0,
    risk_reward       DOUBLE PRECISION DEFAULT 0,
    reason            TEXT DEFAULT '',
    action            TEXT DEFAULT 'pending',
    market_condition  TEXT DEFAULT '',
    indicators_json   TEXT DEFAULT '{}',
    trade_id          TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_signals_ts ON signals_intraday (timestamp);
CREATE INDEX IF NOT EXISTS idx_signals_sym_ts ON signals_intraday (symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_signals_action_ts ON signals_intraday (action, timestamp);

-- =============================================================================
-- REJECTED CANDIDATES
-- =============================================================================
CREATE TABLE IF NOT EXISTS candidates_rejected (
    id                SERIAL PRIMARY KEY,
    timestamp         TEXT NOT NULL,
    symbol            TEXT NOT NULL,
    gap_pct           DOUBLE PRECISION DEFAULT 0,
    vol_ratio         DOUBLE PRECISION DEFAULT 0,
    score             DOUBLE PRECISION DEFAULT 0,
    direction         TEXT DEFAULT '',
    rejection_stage   TEXT NOT NULL,
    rejection_reason  TEXT DEFAULT '',
    catalyst          TEXT DEFAULT '',
    prev_close        DOUBLE PRECISION DEFAULT 0,
    open_price        DOUBLE PRECISION DEFAULT 0,
    strategy_id       TEXT DEFAULT '',
    market_condition  TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_rejected_ts ON candidates_rejected (timestamp);
CREATE INDEX IF NOT EXISTS idx_rejected_stage_ts ON candidates_rejected (rejection_stage, timestamp);
CREATE INDEX IF NOT EXISTS idx_rejected_sym_ts ON candidates_rejected (symbol, timestamp);

-- =============================================================================
-- MINUTE BARS (1-min intraday data for ORB backtesting)
-- =============================================================================
CREATE TABLE IF NOT EXISTS minute_bars (
    symbol  TEXT NOT NULL,
    ts      TIMESTAMP NOT NULL,
    open    DOUBLE PRECISION,
    high    DOUBLE PRECISION,
    low     DOUBLE PRECISION,
    close   DOUBLE PRECISION,
    volume  BIGINT DEFAULT 0,
    PRIMARY KEY (symbol, ts)
);
CREATE INDEX IF NOT EXISTS idx_minute_bars_ts ON minute_bars (ts);
CREATE INDEX IF NOT EXISTS idx_minute_bars_date ON minute_bars (symbol, (ts::date));

-- =============================================================================
-- TRADER STATE (replaces JSON state files)
-- =============================================================================
CREATE TABLE IF NOT EXISTS trader_state (
    key         TEXT PRIMARY KEY,
    state_json  TEXT NOT NULL,
    saved_at    TEXT NOT NULL
);

-- =============================================================================
-- NEWS ALERTS
-- =============================================================================
CREATE TABLE IF NOT EXISTS news_alerts (
    id          TEXT PRIMARY KEY,
    symbol      TEXT NOT NULL DEFAULT '',
    headline    TEXT NOT NULL,
    summary     TEXT DEFAULT '',
    source      TEXT DEFAULT 'alpaca',
    url         TEXT DEFAULT '',
    impact      TEXT DEFAULT 'medium',
    category    TEXT DEFAULT 'general',
    timestamp   TEXT NOT NULL,
    read        BOOLEAN DEFAULT FALSE,
    dismissed   BOOLEAN DEFAULT FALSE,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_news_alerts_ts ON news_alerts (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_news_alerts_symbol ON news_alerts (symbol, timestamp DESC);

-- =============================================================================
-- NEWS SETTINGS (notification preferences)
-- =============================================================================
CREATE TABLE IF NOT EXISTS news_settings (
    key         TEXT PRIMARY KEY,
    value_json  TEXT DEFAULT '{}'
);
