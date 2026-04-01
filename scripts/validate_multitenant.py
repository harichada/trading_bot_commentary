#!/usr/bin/env python
"""Validate multi-tenant infrastructure readiness.

Usage: python scripts/validate_multitenant.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

checks: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = ''):
    checks.append((name, ok, detail))


# ── 1. Environment Variables ────────────────────────────────────────────────

db_url = os.environ.get('DATABASE_URL', '')
check('DATABASE_URL set', bool(db_url), db_url.split('@')[1] if '@' in db_url else '(missing)')

cek = os.environ.get('CREDENTIAL_ENCRYPTION_KEY', '')
check('CREDENTIAL_ENCRYPTION_KEY set', bool(cek), f'{len(cek)} chars' if cek else '(missing)')

if cek:
    try:
        from cryptography.fernet import Fernet
        f = Fernet(cek.encode())
        enc = f.encrypt(b'test')
        dec = f.decrypt(enc)
        check('Fernet key valid (encrypt/decrypt)', dec == b'test')
    except Exception as e:
        check('Fernet key valid', False, str(e))

jwt_secret = os.environ.get('AUTH_JWT_SECRET', '')
check('AUTH_JWT_SECRET set', bool(jwt_secret), f'{len(jwt_secret)} chars' if jwt_secret else '(missing)')

# ── 2. Database Connectivity ────────────────────────────────────────────────

if db_url:
    try:
        import psycopg2
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()

        for table in ['users', 'user_credentials', 'user_configs', 'subscriptions']:
            cur.execute(f'SELECT COUNT(*) FROM {table}')
            count = cur.fetchone()[0]
            check(f"Table '{table}'", True, f'{count} rows')

        cur.execute("SELECT user_id, email, tier FROM users WHERE email = 'system@rudra.local'")
        row = cur.fetchone()
        check('System user exists', row is not None,
              f'tier={row[2]}' if row else '(missing — run migration)')

        cur.execute("SELECT COUNT(*) FROM trades WHERE user_id IS NOT NULL")
        backfilled = cur.fetchone()[0]
        check('Trades have user_id', True, f'{backfilled} backfilled')

        conn.close()
    except Exception as e:
        check('Database connectivity', False, str(e))
else:
    check('Database connectivity', False, 'DATABASE_URL not set')

# ── 3. Module Imports ───────────────────────────────────────────────────────

try:
    from user_management import get_user_repo, get_cred_mgr, get_config_mgr, get_sub_mgr
    check('user_management.py imports', True)
except Exception as e:
    check('user_management.py imports', False, str(e))

try:
    from user_engine import UserEngineManager, ProfitCapChecker, resolve_user_id
    check('user_engine.py imports', True)
except Exception as e:
    check('user_engine.py imports', False, str(e))

try:
    from auth import auth_enabled, get_current_user
    check('auth.py imports', True, f'auth_enabled={auth_enabled()}')
except Exception as e:
    check('auth.py imports', False, str(e))

# ── 4. Credential Manager ──────────────────────────────────────────────────

try:
    mgr = get_cred_mgr()
    has_fernet = mgr._fernet is not None
    check('CredentialManager encryption', has_fernet,
          'Fernet active' if has_fernet else 'DISABLED — set CREDENTIAL_ENCRYPTION_KEY')
except Exception as e:
    check('CredentialManager', False, str(e))

# ── 5. Frontend Build ──────────────────────────────────────────────────────

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dist = os.path.join(base, 'frontend-v2-dist')
idx = os.path.join(dist, 'index.html')
check('frontend-v2-dist exists', os.path.isdir(dist))
check('frontend-v2-dist/index.html', os.path.isfile(idx))
if os.path.isdir(os.path.join(dist, 'assets')):
    assets = os.listdir(os.path.join(dist, 'assets'))
    check('frontend-v2-dist/assets', len(assets) > 0, f'{len(assets)} files')

# ── 6. OAuth Providers ─────────────────────────────────────────────────────

for provider, key_var in [('Google', 'GOOGLE_CLIENT_ID'), ('GitHub', 'GITHUB_CLIENT_ID'), ('Discord', 'DISCORD_CLIENT_ID')]:
    val = os.environ.get(key_var, '')
    check(f'{provider} OAuth', bool(val), 'configured' if val else '(missing)')

# ── Print Results ───────────────────────────────────────────────────────────

print()
print('=' * 60)
print('  Multi-Tenant Validation Report')
print('=' * 60)
print()

passed = 0
failed = 0
for name, ok, detail in checks:
    status = '\033[92mPASS\033[0m' if ok else '\033[91mFAIL\033[0m'
    suffix = f'  ({detail})' if detail else ''
    print(f'  [{status}] {name}{suffix}')
    if ok:
        passed += 1
    else:
        failed += 1

print()
print(f'  {passed} passed, {failed} failed')
print()

if failed == 0:
    print('  All checks passed. Multi-tenant infrastructure is ready.')
else:
    print('  Fix the failing checks above before deploying.')

print()
sys.exit(0 if failed == 0 else 1)
