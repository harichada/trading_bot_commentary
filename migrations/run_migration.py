"""Run database migrations safely.

Usage: python migrations/run_migration.py [--dry-run]
"""

import os
import sys
import psycopg2

def main():
    dry_run = '--dry-run' in sys.argv

    # Load env
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

    db_url = os.environ.get('DATABASE_URL')
    if not db_url:
        print('ERROR: DATABASE_URL not set')
        sys.exit(1)

    migration_file = os.path.join(os.path.dirname(__file__), '001_multi_tenant.sql')
    with open(migration_file) as f:
        sql = f.read()

    print(f'Database: {db_url.split("@")[1] if "@" in db_url else db_url}')
    print(f'Migration: {migration_file}')
    print(f'Mode: {"DRY RUN" if dry_run else "LIVE"}')
    print()

    if dry_run:
        print('SQL to execute:')
        print('-' * 60)
        print(sql[:2000] + '...' if len(sql) > 2000 else sql)
        print('-' * 60)
        print(f'\n{len(sql)} chars of SQL. Run without --dry-run to execute.')
        return

    conn = psycopg2.connect(db_url)
    conn.autocommit = True  # needed for ALTER TABLE + CREATE INDEX CONCURRENTLY
    cur = conn.cursor()

    # Execute migration
    print('Running migration...')
    try:
        cur.execute(sql)
        print('Migration completed successfully.')
    except Exception as e:
        print(f'ERROR: {e}')
        conn.rollback()
        sys.exit(1)

    # Verify
    print('\nVerification:')
    cur.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public' AND table_name IN ('users', 'user_credentials', 'user_configs', 'subscriptions')
        ORDER BY table_name
    """)
    new_tables = [r[0] for r in cur.fetchall()]
    print(f'  New tables: {new_tables}')

    cur.execute("""
        SELECT table_name, column_name FROM information_schema.columns
        WHERE column_name = 'user_id' AND table_schema = 'public'
        ORDER BY table_name
    """)
    uid_cols = [(r[0], r[1]) for r in cur.fetchall()]
    print(f'  Tables with user_id: {[r[0] for r in uid_cols]}')

    cur.execute("SELECT user_id, email, tier, is_admin FROM users")
    users = cur.fetchall()
    print(f'  Users: {len(users)}')
    for u in users:
        print(f'    {u[1]} (tier={u[2]}, admin={u[3]})')

    cur.execute("SELECT count(*) FROM trades WHERE user_id IS NOT NULL")
    backfilled = cur.fetchone()[0]
    print(f'  Trades backfilled: {backfilled}')

    conn.close()
    print('\nDone.')


if __name__ == '__main__':
    main()
