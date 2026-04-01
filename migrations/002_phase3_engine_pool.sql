-- ============================================================================
-- Migration 002: Phase 3 — Per-User Engine Pool indexes
-- ============================================================================
-- Additive indexes only. Safe to run multiple times.
-- ============================================================================

BEGIN;

-- Composite index for user-scoped trade history (descending time)
CREATE INDEX IF NOT EXISTS idx_trades_user_entry_desc
    ON trades (user_id, entry_time DESC);

-- Unique constraint on trader_state per user (allows separate state per user)
CREATE UNIQUE INDEX IF NOT EXISTS idx_trader_state_user_unique
    ON trader_state (user_id) WHERE user_id IS NOT NULL;

COMMIT;
