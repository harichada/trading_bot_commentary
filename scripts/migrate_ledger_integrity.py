#!/usr/bin/env python3
"""v-ledger-integrity-2026-09-24: Standalone migration script for ledger integrity columns.

This script adds the ledger integrity columns to bot_trades table:
  - initial_stop: TRUE stop at entry (immutable)
  - initial_tp: TRUE take-profit at entry (immutable)  
  - initial_risk_per_share: |entry - initial_stop|
  - pnl_r: P&L in R-multiples
  - setup_type: breakout/pullback/continuation/etc.
  - hold_time_seconds: time held
  - source: 'bot', 'broker_orphan', 'external'
  - is_hands_off: MU/HQGE/SPCX flag
  - is_external: external/unmanaged flag
  - exit_reason_raw: original exit reason (before normalization)

Run this script during a quiet window (no trading activity) before enabling
LEDGER_INTEGRITY=1.

Usage:
    python scripts/migrate_ledger_integrity.py [--dsn DSN] [--dry-run]

Options:
    --dsn DSN       PostgreSQL connection string (default: from POSTGRES_DSN env)
    --dry-run       Print SQL without executing
    --skip-indexes  Skip index creation (create them manually with CONCURRENTLY later)

Migration Safety:
  - Uses SAVEPOINT per column so one failure doesn't roll back all
  - Uses lock_timeout (3s) and statement_timeout (30s) to avoid blocking
  - Idempotent: uses ADD COLUMN IF NOT EXISTS
  - Index creation uses CONCURRENTLY (requires being outside a transaction)
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import psycopg2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("migrate_ledger_integrity")

DEFAULT_DSN = "postgresql://rudra:rudra_dev_2024@localhost:5432/rudra_dev"

COLUMNS_TO_ADD = [
    ("initial_stop", "REAL"),
    ("initial_tp", "REAL"),
    ("initial_risk_per_share", "REAL"),
    ("pnl_r", "REAL"),
    ("setup_type", "TEXT"),
    ("hold_time_seconds", "INTEGER"),
    ("source", "TEXT DEFAULT 'bot'"),
    ("is_hands_off", "BOOLEAN DEFAULT FALSE"),
    ("is_external", "BOOLEAN DEFAULT FALSE"),
    ("exit_reason_raw", "TEXT"),
]

INDEXES_TO_CREATE = [
    ("idx_bot_trades_source", "source"),
    ("idx_bot_trades_setup_type", "setup_type"),
    ("idx_bot_trades_pnl_r", "pnl_r"),
    ("idx_bot_trades_is_external", "is_external"),
]


def run_migration(dsn: str, dry_run: bool = False, skip_indexes: bool = False) -> bool:
    """Run the ledger integrity migration.
    
    Returns True if migration succeeded, False if any errors occurred.
    """
    logger.info("Starting ledger integrity migration...")
    logger.info("DSN: %s", dsn.split("@")[-1] if "@" in dsn else dsn)
    logger.info("Dry run: %s", dry_run)
    logger.info("Skip indexes: %s", skip_indexes)
    
    if dry_run:
        logger.info("=== DRY RUN MODE - SQL will be printed but not executed ===")
    
    errors = []
    
    try:
        conn = psycopg2.connect(dsn)
        conn.autocommit = False
    except Exception as e:
        logger.error("Failed to connect to database: %s", e)
        return False
    
    try:
        with conn.cursor() as cur:
            cur.execute("SET LOCAL lock_timeout = '3s'")
            cur.execute("SET LOCAL statement_timeout = '30s'")
            
            for col_name, col_type in COLUMNS_TO_ADD:
                sql = f"ALTER TABLE bot_trades ADD COLUMN IF NOT EXISTS {col_name} {col_type}"
                logger.info("Adding column: %s %s", col_name, col_type)
                
                if dry_run:
                    logger.info("  [DRY RUN] %s", sql)
                    continue
                
                savepoint_name = f"sp_{col_name}"
                try:
                    cur.execute(f"SAVEPOINT {savepoint_name}")
                    cur.execute(sql)
                    cur.execute(f"RELEASE SAVEPOINT {savepoint_name}")
                    logger.info("  SUCCESS: %s added", col_name)
                except psycopg2.Error as e:
                    cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
                    if "already exists" in str(e).lower():
                        logger.info("  SKIP: %s already exists", col_name)
                    else:
                        logger.warning("  FAILED: %s - %s", col_name, e)
                        errors.append((col_name, str(e)))
            
            if not dry_run:
                conn.commit()
                logger.info("Column migration committed.")
    
    except Exception as e:
        logger.error("Column migration failed: %s", e)
        conn.rollback()
        errors.append(("columns", str(e)))
    
    if not skip_indexes:
        conn.autocommit = True
        
        for idx_name, idx_col in INDEXES_TO_CREATE:
            sql = f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {idx_name} ON bot_trades ({idx_col})"
            logger.info("Creating index: %s on %s", idx_name, idx_col)
            
            if dry_run:
                logger.info("  [DRY RUN] %s", sql)
                continue
            
            try:
                with conn.cursor() as cur:
                    cur.execute(f"SET lock_timeout = '10s'")
                    cur.execute(f"SET statement_timeout = '120s'")
                    cur.execute(sql)
                    logger.info("  SUCCESS: %s created", idx_name)
            except psycopg2.Error as e:
                if "already exists" in str(e).lower():
                    logger.info("  SKIP: %s already exists", idx_name)
                else:
                    logger.warning("  FAILED: %s - %s", idx_name, e)
                    errors.append((idx_name, str(e)))
    
    conn.close()
    
    if errors:
        logger.warning("Migration completed with %d errors:", len(errors))
        for name, err in errors:
            logger.warning("  - %s: %s", name, err)
        return False
    
    logger.info("Migration completed successfully!")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Verify columns exist: psql -c '\\d bot_trades'")
    logger.info("  2. Enable the feature: export LEDGER_INTEGRITY=1")
    logger.info("  3. Restart the trading bot")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Migrate bot_trades table for ledger integrity feature",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dsn",
        default=os.environ.get("POSTGRES_DSN", DEFAULT_DSN),
        help="PostgreSQL connection string (default: from POSTGRES_DSN env or localhost)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print SQL without executing",
    )
    parser.add_argument(
        "--skip-indexes",
        action="store_true",
        help="Skip index creation (create them manually with CONCURRENTLY later)",
    )
    
    args = parser.parse_args()
    
    success = run_migration(args.dsn, args.dry_run, args.skip_indexes)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
