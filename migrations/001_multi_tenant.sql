-- ============================================================================
-- Migration 001: Multi-Tenant Schema
-- ============================================================================
-- Safe to run multiple times (all statements use IF NOT EXISTS / IF EXISTS).
-- Existing data is preserved — new columns are nullable.
-- Run: psql $DATABASE_URL -f migrations/001_multi_tenant.sql
-- ============================================================================

BEGIN;

-- ── 1. Users table ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
    user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT UNIQUE NOT NULL,
    name TEXT DEFAULT '',
    picture TEXT DEFAULT '',
    provider TEXT DEFAULT '',              -- google, github, discord
    tier TEXT DEFAULT 'free' CHECK (tier IN ('free', 'starter', 'pro', 'enterprise')),
    is_active BOOLEAN DEFAULT TRUE,
    is_admin BOOLEAN DEFAULT FALSE,
    -- Billing
    stripe_customer_id TEXT DEFAULT '',
    stripe_subscription_id TEXT DEFAULT '',
    -- Profit tracking
    monthly_pnl NUMERIC DEFAULT 0,         -- running P&L this month
    monthly_pnl_reset_date DATE DEFAULT CURRENT_DATE,  -- 1st of current month
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_login TIMESTAMPTZ DEFAULT NOW(),
    settings_json TEXT DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_users_email ON users (email);
CREATE INDEX IF NOT EXISTS idx_users_tier ON users (tier);

-- ── 2. User credentials (encrypted broker API keys) ────────────────────────
CREATE TABLE IF NOT EXISTS user_credentials (
    id SERIAL PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    broker TEXT NOT NULL CHECK (broker IN ('alpaca', 'oanda', 'ibkr')),
    credentials_encrypted TEXT NOT NULL,   -- Fernet-encrypted JSON blob
    label TEXT DEFAULT 'default',          -- e.g. 'paper', 'live'
    is_paper BOOLEAN DEFAULT TRUE,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, broker, label)
);

CREATE INDEX IF NOT EXISTS idx_user_creds_user ON user_credentials (user_id);

-- ── 3. User configs (per-user GapFadeConfig) ───────────────────────────────
CREATE TABLE IF NOT EXISTS user_configs (
    user_id UUID PRIMARY KEY REFERENCES users(user_id) ON DELETE CASCADE,
    config_json TEXT NOT NULL DEFAULT '{}',
    active_strategy TEXT DEFAULT 'classic_gap_fade',
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ── 4. Subscriptions ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS subscriptions (
    id SERIAL PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    stripe_subscription_id TEXT UNIQUE,
    tier TEXT NOT NULL DEFAULT 'free',
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'past_due', 'canceled', 'trialing')),
    current_period_start TIMESTAMPTZ,
    current_period_end TIMESTAMPTZ,
    monthly_profit_cap NUMERIC DEFAULT 200,  -- $ cap for this tier
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_subscriptions_user ON subscriptions (user_id);

-- ── 5. Add user_id to existing per-user tables ─────────────────────────────
-- All nullable so existing rows aren't broken.

ALTER TABLE trades ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES users(user_id);
ALTER TABLE trader_state ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES users(user_id);
ALTER TABLE journal_entries ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES users(user_id);
ALTER TABLE signals_intraday ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES users(user_id);
ALTER TABLE config_profiles ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES users(user_id);
ALTER TABLE config_history ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES users(user_id);
ALTER TABLE performance_snapshots ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES users(user_id);
ALTER TABLE llm_calls ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES users(user_id);
ALTER TABLE candidates_rejected ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES users(user_id);
ALTER TABLE news_settings ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES users(user_id);

-- ── 6. Indexes for tenant-scoped queries ────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_trades_user ON trades (user_id);
CREATE INDEX IF NOT EXISTS idx_trades_user_time ON trades (user_id, entry_time);
CREATE INDEX IF NOT EXISTS idx_trader_state_user ON trader_state (user_id);
CREATE INDEX IF NOT EXISTS idx_journal_user ON journal_entries (user_id);
CREATE INDEX IF NOT EXISTS idx_signals_user ON signals_intraday (user_id);
CREATE INDEX IF NOT EXISTS idx_config_profiles_user ON config_profiles (user_id);
CREATE INDEX IF NOT EXISTS idx_config_history_user ON config_history (user_id);
CREATE INDEX IF NOT EXISTS idx_perf_snap_user ON performance_snapshots (user_id);
CREATE INDEX IF NOT EXISTS idx_llm_calls_user ON llm_calls (user_id);

-- ── 7. Backfill: create legacy system user for existing data ────────────────
INSERT INTO users (email, name, tier, is_admin, is_active)
VALUES ('system@rudra.local', 'Legacy System', 'enterprise', TRUE, TRUE)
ON CONFLICT (email) DO NOTHING;

-- Backfill existing rows to legacy user
UPDATE trades SET user_id = (SELECT user_id FROM users WHERE email = 'system@rudra.local')
    WHERE user_id IS NULL;
UPDATE trader_state SET user_id = (SELECT user_id FROM users WHERE email = 'system@rudra.local')
    WHERE user_id IS NULL;
UPDATE journal_entries SET user_id = (SELECT user_id FROM users WHERE email = 'system@rudra.local')
    WHERE user_id IS NULL;
UPDATE signals_intraday SET user_id = (SELECT user_id FROM users WHERE email = 'system@rudra.local')
    WHERE user_id IS NULL;
UPDATE config_profiles SET user_id = (SELECT user_id FROM users WHERE email = 'system@rudra.local')
    WHERE user_id IS NULL;
UPDATE config_history SET user_id = (SELECT user_id FROM users WHERE email = 'system@rudra.local')
    WHERE user_id IS NULL;
UPDATE performance_snapshots SET user_id = (SELECT user_id FROM users WHERE email = 'system@rudra.local')
    WHERE user_id IS NULL;
UPDATE llm_calls SET user_id = (SELECT user_id FROM users WHERE email = 'system@rudra.local')
    WHERE user_id IS NULL;
UPDATE candidates_rejected SET user_id = (SELECT user_id FROM users WHERE email = 'system@rudra.local')
    WHERE user_id IS NULL;

-- ── 8. Tier profit caps (default values) ────────────────────────────────────
-- These are enforced in the application, not DB constraints.
-- Free = $200/month, Starter = $1,000/month, Pro = $5,000/month, Enterprise = unlimited
COMMENT ON TABLE subscriptions IS 'Profit caps: free=$200, starter=$1000, pro=$5000, enterprise=unlimited';

-- ── 9. Row-level security (defense in depth) ────────────────────────────────
-- Enable RLS on per-user tables. Policies added in application code.
-- This is an extra safety net — the app already filters by user_id.
ALTER TABLE trades ENABLE ROW LEVEL SECURITY;
ALTER TABLE trader_state ENABLE ROW LEVEL SECURITY;
ALTER TABLE journal_entries ENABLE ROW LEVEL SECURITY;
ALTER TABLE signals_intraday ENABLE ROW LEVEL SECURITY;
ALTER TABLE config_profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE config_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE performance_snapshots ENABLE ROW LEVEL SECURITY;
ALTER TABLE llm_calls ENABLE ROW LEVEL SECURITY;

-- Note: RLS policies are not enforced until we create per-user DB roles.
-- For now the app user (rudra) bypasses RLS as table owner.
-- Phase 2 will add proper policies if needed.

COMMIT;

-- ============================================================================
-- Verification query (run after migration)
-- ============================================================================
-- SELECT table_name, column_name FROM information_schema.columns
-- WHERE column_name = 'user_id' AND table_schema = 'public' ORDER BY table_name;
