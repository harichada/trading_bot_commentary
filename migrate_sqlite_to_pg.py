#!/usr/bin/env python3
"""
One-time migration script: SQLite → PostgreSQL for Rudra Trading Engine.

Reads all 13 tables from gap_fade_prices.db and writes them to PostgreSQL.
Uses COPY for daily_bars (bulk, fast) and execute_values for other tables.

Usage:
    python migrate_sqlite_to_pg.py
"""

import io
import os
import sqlite3
import sys
import time

import psycopg2
import psycopg2.extras

# --- Config ---
SQLITE_PATH = os.path.join(os.path.dirname(__file__) or '.', 'gap_fade_prices.db')
PG_URL = os.environ.get('DATABASE_URL', 'postgresql://rudra:rudra_dev_2024@localhost:5432/rudra_dev')

# Tables with SERIAL PRIMARY KEY (id column auto-generated)
SERIAL_TABLES = [
    'trades', 'journal_entries', 'market_events', 'llm_calls',
    'config_history', 'api_calls', 'performance_snapshots',
    'signals_intraday', 'candidates_rejected',
]

# Tables with natural PK (no auto id)
NATURAL_PK_TABLES = ['daily_bars', 'config_profiles']


def get_table_columns(sqlite_conn, table: str) -> list:
    """Get column names for a SQLite table."""
    cur = sqlite_conn.execute(f'PRAGMA table_info({table})')
    return [row[1] for row in cur.fetchall()]


def migrate_daily_bars(sqlite_conn, pg_conn):
    """Migrate daily_bars using COPY (fastest for large tables)."""
    print("  Counting rows...", end=' ', flush=True)
    count = sqlite_conn.execute('SELECT COUNT(*) FROM daily_bars').fetchone()[0]
    print(f"{count:,} rows")

    if count == 0:
        print("  Skipping (empty)")
        return 0

    # Check if PG already has data (idempotent re-run)
    cur_check = pg_conn.cursor()
    cur_check.execute('SELECT COUNT(*) FROM daily_bars')
    pg_count = cur_check.fetchone()[0]
    if pg_count >= count:
        print(f"  Already populated ({pg_count:,} rows in PG). Skipping.")
        return pg_count

    print("  Streaming via COPY...", flush=True)
    t0 = time.time()

    # Read from SQLite, write to a StringIO buffer, then COPY to PG
    cur_pg = pg_conn.cursor()

    # Use a chunked approach to avoid loading 25M rows into RAM at once
    chunk_size = 500_000
    total_copied = 0
    offset = 0

    while True:
        rows = sqlite_conn.execute(
            'SELECT symbol, date, open, high, low, close, volume '
            'FROM daily_bars ORDER BY symbol, date LIMIT ? OFFSET ?',
            (chunk_size, offset)
        ).fetchall()

        if not rows:
            break

        buf = io.StringIO()
        for r in rows:
            # Tab-separated, \N for nulls
            line = '\t'.join(str(v) if v is not None else '\\N' for v in r)
            buf.write(line + '\n')
        buf.seek(0)

        cur_pg.copy_expert(
            "COPY daily_bars (symbol, date, open, high, low, close, volume) FROM STDIN "
            "WITH (FORMAT text, NULL '\\N')",
            buf
        )

        total_copied += len(rows)
        elapsed = time.time() - t0
        rate = total_copied / elapsed if elapsed > 0 else 0
        print(f"    {total_copied:>10,} / {count:,}  ({rate:,.0f} rows/sec)", flush=True)

        if len(rows) < chunk_size:
            break
        offset += chunk_size

    pg_conn.commit()
    elapsed = time.time() - t0
    print(f"  Done: {total_copied:,} rows in {elapsed:.1f}s")
    return total_copied


def migrate_serial_table(sqlite_conn, pg_conn, table: str):
    """Migrate a table with SERIAL PK, preserving original IDs."""
    cols = get_table_columns(sqlite_conn, table)
    count = sqlite_conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
    print(f"  {table}: {count:,} rows", end=' ', flush=True)

    if count == 0:
        print("(skipped)")
        return 0

    rows = sqlite_conn.execute(f'SELECT * FROM {table}').fetchall()

    # Build INSERT with all columns including id, ON CONFLICT on PK
    col_list = ', '.join(cols)
    cur_pg = pg_conn.cursor()
    pk_col = cols[0]  # trade_id or id
    sql = f"INSERT INTO {table} ({col_list}) VALUES %s ON CONFLICT ({pk_col}) DO NOTHING"

    try:
        psycopg2.extras.execute_values(cur_pg, sql, rows, page_size=5000)
        pg_conn.commit()
    except psycopg2.errors.UniqueViolation:
        pg_conn.rollback()
        # Unique constraint on non-PK columns — insert one-by-one, skipping dupes
        print("(dedup) ", end='', flush=True)
        inserted = 0
        for row in rows:
            try:
                placeholders = ','.join(['%s'] * len(row))
                cur_pg.execute(
                    f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) "
                    f"ON CONFLICT DO NOTHING", row
                )
                if cur_pg.rowcount > 0:
                    inserted += 1
            except psycopg2.Error:
                pg_conn.rollback()
                continue
        pg_conn.commit()
        count = inserted  # actual inserted

    # Reset sequence to max(id) + 1
    id_col = pk_col
    cur_pg.execute(f"SELECT MAX({id_col}) FROM {table}")
    max_id = cur_pg.fetchone()[0]
    if max_id is not None:
        seq_name = f"{table}_{id_col}_seq"
        try:
            cur_pg.execute(f"SELECT setval('{seq_name}', {max_id})")
            pg_conn.commit()
        except psycopg2.Error:
            pg_conn.rollback()
            for pattern in [f"{table}_id_seq", f"{table}_{id_col}_seq"]:
                try:
                    cur_pg.execute(f"SELECT setval('{pattern}', {max_id})")
                    pg_conn.commit()
                    break
                except psycopg2.Error:
                    pg_conn.rollback()

    print(f"-> migrated")
    return count


def migrate_natural_pk_table(sqlite_conn, pg_conn, table: str):
    """Migrate a table with natural PK (no auto-increment)."""
    if table == 'daily_bars':
        return migrate_daily_bars(sqlite_conn, pg_conn)

    cols = get_table_columns(sqlite_conn, table)
    count = sqlite_conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
    print(f"  {table}: {count:,} rows", end=' ', flush=True)

    if count == 0:
        print("(skipped)")
        return 0

    rows = sqlite_conn.execute(f'SELECT * FROM {table}').fetchall()
    col_list = ', '.join(cols)
    pk_col = cols[0]  # natural PK
    sql = f"INSERT INTO {table} ({col_list}) VALUES %s ON CONFLICT ({pk_col}) DO NOTHING"

    cur_pg = pg_conn.cursor()
    psycopg2.extras.execute_values(cur_pg, sql, rows, page_size=5000)
    pg_conn.commit()

    print(f"-> migrated")
    return count


def verify_counts(sqlite_conn, pg_conn):
    """Print row count comparison for all tables."""
    print("\n=== Row Count Verification ===")
    print(f"{'Table':<25} {'SQLite':>12} {'PostgreSQL':>12} {'Match':>7}")
    print("-" * 60)

    all_match = True
    tables = NATURAL_PK_TABLES + SERIAL_TABLES
    for table in tables:
        try:
            sq_count = sqlite_conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
        except sqlite3.OperationalError:
            sq_count = 0  # table may not exist in SQLite

        cur_pg = pg_conn.cursor()
        try:
            cur_pg.execute(f'SELECT COUNT(*) FROM {table}')
            pg_count = cur_pg.fetchone()[0]
        except psycopg2.Error:
            pg_conn.rollback()
            pg_count = 0

        match = 'OK' if sq_count == pg_count else 'MISMATCH'
        if sq_count != pg_count:
            all_match = False
        print(f"  {table:<23} {sq_count:>12,} {pg_count:>12,} {match:>7}")

    print("-" * 60)
    if all_match:
        print("All tables match!")
    else:
        print("WARNING: Some tables have mismatched row counts.")


def main():
    if not os.path.exists(SQLITE_PATH):
        print(f"SQLite database not found: {SQLITE_PATH}")
        sys.exit(1)

    print(f"SQLite: {SQLITE_PATH}")
    print(f"PostgreSQL: {PG_URL.split('@')[1] if '@' in PG_URL else PG_URL}")
    print()

    # Connect
    sqlite_conn = sqlite3.connect(SQLITE_PATH)
    pg_conn = psycopg2.connect(PG_URL)
    pg_conn.autocommit = False

    # Ensure PG tables exist (by importing PriceDB)
    print("Creating PostgreSQL tables (via PriceDB)...")
    sys.path.insert(0, os.path.dirname(__file__) or '.')
    from gap_fade_app import PriceDB
    _db = PriceDB()
    print("Tables created.\n")

    # Migrate
    t_start = time.time()
    total_rows = 0

    print("=== Migrating Tables ===")
    for table in NATURAL_PK_TABLES:
        total_rows += migrate_natural_pk_table(sqlite_conn, pg_conn, table)

    for table in SERIAL_TABLES:
        try:
            sqlite_conn.execute(f'SELECT 1 FROM {table} LIMIT 1')
        except sqlite3.OperationalError:
            print(f"  {table}: not in SQLite (skipped)")
            continue
        total_rows += migrate_serial_table(sqlite_conn, pg_conn, table)

    # Run ANALYZE for query planner
    print("\nRunning ANALYZE on all tables...")
    cur_pg = pg_conn.cursor()
    cur_pg.execute('ANALYZE')
    pg_conn.commit()

    elapsed = time.time() - t_start
    print(f"\nMigration complete: {total_rows:,} total rows in {elapsed:.1f}s")

    # Verify
    verify_counts(sqlite_conn, pg_conn)

    sqlite_conn.close()
    pg_conn.close()


if __name__ == '__main__':
    main()
