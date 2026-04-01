"""
User Management Module — Multi-tenant Rudra Trading Engine

Handles user CRUD, encrypted credential storage, per-user config,
and subscription/tier management. All endpoints require JWT auth.
"""

import json
import logging
import os
import threading
from typing import Optional

import psycopg2
import psycopg2.extras
import psycopg2.pool
from cryptography.fernet import Fernet, InvalidToken
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("GapFadeApp.users")

# Startup warning: credential endpoints require auth to be secure
if not os.environ.get("AUTH_JWT_SECRET"):
    logger.warning(
        "AUTH_JWT_SECRET not set — /api/users/* and /api/admin/* endpoints have no auth protection. "
        "Set AUTH_JWT_SECRET to enable authentication."
    )

# ── Database Connection Pool ────────────────────────────────────────────────

_pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None
_pool_lock = threading.Lock()

MAX_NAME_LENGTH = 128
MAX_SETTINGS_KEYS = 20
MAX_SETTINGS_VALUE_LENGTH = 500
ALLOWED_SETTINGS_KEYS = frozenset({"theme", "timezone", "notifications", "dashboard_layout", "default_view"})


def _get_pool() -> psycopg2.pool.ThreadedConnectionPool:
    """Get or create a thread-safe connection pool."""
    global _pool
    with _pool_lock:
        if _pool is None or _pool.closed:
            db_url = os.environ.get("DATABASE_URL", "")
            if not db_url:
                raise RuntimeError("DATABASE_URL not set")
            _pool = psycopg2.pool.ThreadedConnectionPool(1, 10, db_url)
    return _pool


def _execute(sql: str, params: tuple = (), fetch: str = "all") -> list | dict | None:
    """Execute SQL with a connection from the pool. Thread-safe."""
    pool = _get_pool()
    conn = pool.getconn()
    try:
        conn.autocommit = True
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            if fetch == "none":
                return None
            if fetch == "one":
                row = cur.fetchone()
                return dict(row) if row else None
            return [dict(r) for r in cur.fetchall()]
    finally:
        pool.putconn(conn)


# ── Tier Limits ─────────────────────────────────────────────────────────────

TIER_LIMITS = {
    "free": {
        "monthly_profit_cap": 200,
        "max_positions": 3,
        "brokers": ["alpaca"],
        "paper_only": True,
        "features": ["gap_fade"],
    },
    "starter": {
        "monthly_profit_cap": 1000,
        "max_positions": 5,
        "brokers": ["alpaca"],
        "paper_only": False,
        "features": ["gap_fade", "intraday"],
    },
    "pro": {
        "monthly_profit_cap": 5000,
        "max_positions": 10,
        "brokers": ["alpaca", "oanda"],
        "paper_only": False,
        "features": ["gap_fade", "intraday", "llm", "swing"],
    },
    "enterprise": {
        "monthly_profit_cap": None,  # unlimited
        "max_positions": 20,
        "brokers": ["alpaca", "oanda", "ibkr"],
        "paper_only": False,
        "features": ["gap_fade", "intraday", "llm", "swing", "custom"],
    },
}

VALID_TIERS = frozenset(TIER_LIMITS.keys())


# ── UserRepository ──────────────────────────────────────────────────────────

class UserRepository:
    """CRUD operations on the users table."""

    def upsert_on_login(
        self, email: str, name: str = "", picture: str = "", provider: str = ""
    ) -> dict:
        """Create or update user on OAuth login. Returns full user row."""
        return _execute(
            """INSERT INTO users (email, name, picture, provider, last_login)
               VALUES (%s, %s, %s, %s, NOW())
               ON CONFLICT (email) DO UPDATE SET
                   name = EXCLUDED.name,
                   picture = EXCLUDED.picture,
                   provider = EXCLUDED.provider,
                   last_login = NOW()
               RETURNING *""",
            (email.lower(), name, picture, provider),
            fetch="one",
        )

    def get_by_email(self, email: str) -> Optional[dict]:
        return _execute(
            "SELECT * FROM users WHERE email = %s", (email.lower(),), fetch="one"
        )

    def get_by_id(self, user_id: str) -> Optional[dict]:
        return _execute(
            "SELECT * FROM users WHERE user_id = %s", (user_id,), fetch="one"
        )

    def update_profile(self, user_id: str, name: str, settings_json: dict) -> Optional[dict]:
        return _execute(
            """UPDATE users SET name = %s, settings_json = %s
               WHERE user_id = %s RETURNING *""",
            (name, json.dumps(settings_json), user_id),
            fetch="one",
        )

    def list_all(self) -> list[dict]:
        return _execute(
            """SELECT user_id, email, name, picture, provider, tier,
                      is_active, is_admin, created_at, last_login,
                      monthly_pnl, monthly_pnl_reset_date
               FROM users ORDER BY created_at"""
        )

    def update_tier(self, user_id: str, tier: str) -> Optional[dict]:
        if tier not in VALID_TIERS:
            return None
        cap = TIER_LIMITS[tier]["monthly_profit_cap"]
        # Update user tier + upsert subscription
        user = _execute(
            "UPDATE users SET tier = %s WHERE user_id = %s RETURNING *",
            (tier, user_id),
            fetch="one",
        )
        if user:
            _execute(
                """INSERT INTO subscriptions (user_id, tier, monthly_profit_cap)
                   VALUES (%s, %s, %s)
                   ON CONFLICT (user_id) DO UPDATE SET
                       tier = EXCLUDED.tier,
                       monthly_profit_cap = EXCLUDED.monthly_profit_cap,
                       updated_at = NOW()""",
                (user_id, tier, cap or 999999999),
                fetch="none",
            )
        return user

    def update_monthly_pnl(self, user_id: str, pnl_delta: float) -> Optional[dict]:
        """Atomically increment monthly_pnl. Auto-resets if month changed."""
        return _execute(
            """UPDATE users SET
                   monthly_pnl = CASE
                       WHEN EXTRACT(MONTH FROM monthly_pnl_reset_date) != EXTRACT(MONTH FROM CURRENT_DATE)
                            OR EXTRACT(YEAR FROM monthly_pnl_reset_date) != EXTRACT(YEAR FROM CURRENT_DATE)
                       THEN %s
                       ELSE monthly_pnl + %s
                   END,
                   monthly_pnl_reset_date = CURRENT_DATE
               WHERE user_id = %s
               RETURNING monthly_pnl, monthly_pnl_reset_date, tier""",
            (pnl_delta, pnl_delta, user_id),
            fetch="one",
        )

    def get_monthly_pnl(self, user_id: str) -> Optional[dict]:
        """Return monthly P&L info for profit cap checking."""
        return _execute(
            """SELECT monthly_pnl, monthly_pnl_reset_date, tier
               FROM users WHERE user_id = %s""",
            (user_id,),
            fetch="one",
        )


# ── CredentialManager ───────────────────────────────────────────────────────

class CredentialManager:
    """Encrypted broker credential storage using Fernet."""

    def __init__(self):
        key = os.environ.get("CREDENTIAL_ENCRYPTION_KEY", "")
        if key:
            # Fail loudly on invalid key — misconfigured encryption is a deployment error
            self._fernet = Fernet(key.encode() if isinstance(key, str) else key)
            logger.info("Credential encryption enabled")
        else:
            self._fernet = None
            logger.warning("CREDENTIAL_ENCRYPTION_KEY not set — credential storage disabled")

    def _require_fernet(self):
        if not self._fernet:
            raise RuntimeError(
                "Credential encryption not configured. Set CREDENTIAL_ENCRYPTION_KEY env var."
            )

    def store(
        self, user_id: str, broker: str, credentials: dict, label: str = "default", is_paper: bool = True
    ) -> dict:
        self._require_fernet()
        encrypted = self._fernet.encrypt(json.dumps(credentials).encode()).decode()
        return _execute(
            """INSERT INTO user_credentials (user_id, broker, credentials_encrypted, label, is_paper)
               VALUES (%s, %s, %s, %s, %s)
               ON CONFLICT (user_id, broker, label) DO UPDATE SET
                   credentials_encrypted = EXCLUDED.credentials_encrypted,
                   is_paper = EXCLUDED.is_paper,
                   updated_at = NOW()
               RETURNING id, broker, label, is_paper, is_active, created_at""",
            (user_id, broker, encrypted, label, is_paper),
            fetch="one",
        )

    def list_for_user(self, user_id: str) -> list[dict]:
        """List credentials (without secrets)."""
        return _execute(
            """SELECT id, broker, label, is_paper, is_active, created_at, updated_at
               FROM user_credentials WHERE user_id = %s ORDER BY created_at""",
            (user_id,),
        )

    def get_decrypted(self, user_id: str, credential_id: int) -> Optional[dict]:
        """Decrypt and return credentials. Validates tenant ownership."""
        self._require_fernet()
        row = _execute(
            "SELECT * FROM user_credentials WHERE id = %s AND user_id = %s",
            (credential_id, user_id),
            fetch="one",
        )
        if not row:
            return None
        try:
            decrypted = self._fernet.decrypt(row["credentials_encrypted"].encode())
            creds = json.loads(decrypted)
        except (InvalidToken, json.JSONDecodeError):
            logger.error("Failed to decrypt credential id=%s for user=%s", credential_id, user_id)
            return None
        return {
            "id": row["id"],
            "broker": row["broker"],
            "label": row["label"],
            "is_paper": row["is_paper"],
            "credentials": creds,
        }

    def delete(self, user_id: str, credential_id: int) -> bool:
        """Delete a credential. Returns True if deleted."""
        result = _execute(
            "DELETE FROM user_credentials WHERE id = %s AND user_id = %s RETURNING id",
            (credential_id, user_id),
            fetch="one",
        )
        return result is not None

    def get_all_for_user(self, user_id: str) -> list[dict]:
        """Decrypt ALL active credentials for a user. Used by engine provisioning."""
        self._require_fernet()
        rows = _execute(
            """SELECT id, broker, credentials_encrypted, label, is_paper
               FROM user_credentials WHERE user_id = %s AND is_active = TRUE""",
            (user_id,),
        )
        result = []
        for row in (rows or []):
            try:
                decrypted = self._fernet.decrypt(row["credentials_encrypted"].encode())
                creds = json.loads(decrypted)
            except (InvalidToken, json.JSONDecodeError):
                logger.error("Failed to decrypt credential id=%s for user=%s", row["id"], user_id)
                continue
            result.append({
                "id": row["id"],
                "broker": row["broker"],
                "label": row["label"],
                "is_paper": row["is_paper"],
                "credentials": creds,
            })
        return result


# ── ConfigManager ───────────────────────────────────────────────────────────

# Valid GapFadeConfig fields (extracted from dataclass to avoid circular import)
VALID_CONFIG_FIELDS = frozenset({
    "stop_pct", "risk_pct", "max_positions", "initial_capital",
    "gap_threshold", "max_gap_pct", "vol_ratio_max", "daily_loss_limit",
    "max_drawdown", "max_consec_losses", "kelly_fraction",
    "min_avg_volume", "min_price", "max_price", "min_market_cap",
    "adaptive_stops_enabled", "atr_stop_multiplier", "volatility_stop_floor",
    "volatility_stop_cap", "trailing_stop_pct",
    "market_regime_enabled", "spy_sma_period", "vix_threshold",
    "reentry_enabled", "reentry_max_attempts", "reentry_gap_pct_step",
    "reentry_delay_bars",
    "gap_down_long_enabled", "gap_down_threshold", "gap_down_rsi_max",
    "gap_down_stop_pct", "gap_down_target_pct",
    "orb_enabled", "orb_period_minutes", "orb_breakout_pct",
    "pyramid_enabled", "pyramid_max_adds", "pyramid_add_threshold",
    "sweep_enabled", "sweep_min_symbols", "sweep_max_symbols",
    "swing_enabled", "swing_hold_days", "swing_stop_pct",
    "exclude_leveraged",
    "llm_enabled", "llm_model", "llm_timeout",
    "scan_start_hour", "final_scan_hour", "final_scan_min",
})


class ConfigManager:
    """Per-user GapFadeConfig stored as JSON."""

    def get_config(self, user_id: str) -> dict:
        row = _execute(
            "SELECT config_json, active_strategy, updated_at FROM user_configs WHERE user_id = %s",
            (user_id,),
            fetch="one",
        )
        if not row:
            return {"config": {}, "active_strategy": "classic_gap_fade", "updated_at": None}
        try:
            config = json.loads(row["config_json"]) if row["config_json"] else {}
        except json.JSONDecodeError:
            config = {}
        return {
            "config": config,
            "active_strategy": row.get("active_strategy", "classic_gap_fade"),
            "updated_at": str(row["updated_at"]) if row.get("updated_at") else None,
        }

    def update_config(self, user_id: str, config_partial: dict) -> dict:
        """Merge partial config update. Validates field names."""
        invalid = set(config_partial.keys()) - VALID_CONFIG_FIELDS
        if invalid:
            raise ValueError(f"Invalid config fields: {', '.join(sorted(invalid))}")

        current = self.get_config(user_id)
        merged = {**current["config"], **config_partial}

        _execute(
            """INSERT INTO user_configs (user_id, config_json, updated_at)
               VALUES (%s, %s, NOW())
               ON CONFLICT (user_id) DO UPDATE SET
                   config_json = EXCLUDED.config_json,
                   updated_at = NOW()""",
            (user_id, json.dumps(merged)),
            fetch="none",
        )
        return self.get_config(user_id)


# ── SubscriptionManager ────────────────────────────────────────────────────

class SubscriptionManager:

    def get_subscription(self, user_id: str) -> dict:
        row = _execute(
            """SELECT s.*, u.tier as user_tier
               FROM subscriptions s
               JOIN users u ON u.user_id = s.user_id
               WHERE s.user_id = %s""",
            (user_id,),
            fetch="one",
        )
        if not row:
            # Check user tier directly
            user = _execute("SELECT tier FROM users WHERE user_id = %s", (user_id,), fetch="one")
            tier = user["tier"] if user else "free"
            limits = TIER_LIMITS.get(tier, TIER_LIMITS["free"])
            return {
                "tier": tier,
                "status": "active",
                "monthly_profit_cap": limits["monthly_profit_cap"],
                "current_period_end": None,
                "limits": limits,
            }
        tier = row.get("tier", "free")
        limits = TIER_LIMITS.get(tier, TIER_LIMITS["free"])
        return {
            "tier": tier,
            "status": row.get("status", "active"),
            "monthly_profit_cap": row.get("monthly_profit_cap"),
            "current_period_start": str(row["current_period_start"]) if row.get("current_period_start") else None,
            "current_period_end": str(row["current_period_end"]) if row.get("current_period_end") else None,
            "limits": limits,
        }


# ── Singletons ──────────────────────────────────────────────────────────────

_user_repo: Optional[UserRepository] = None
_cred_mgr: Optional[CredentialManager] = None
_config_mgr: Optional[ConfigManager] = None
_sub_mgr: Optional[SubscriptionManager] = None


def get_user_repo() -> UserRepository:
    global _user_repo
    if _user_repo is None:
        _user_repo = UserRepository()
    return _user_repo


def get_cred_mgr() -> CredentialManager:
    global _cred_mgr
    if _cred_mgr is None:
        _cred_mgr = CredentialManager()
    return _cred_mgr


def get_config_mgr() -> ConfigManager:
    global _config_mgr
    if _config_mgr is None:
        _config_mgr = ConfigManager()
    return _config_mgr


def get_sub_mgr() -> SubscriptionManager:
    global _sub_mgr
    if _sub_mgr is None:
        _sub_mgr = SubscriptionManager()
    return _sub_mgr


# ── Auth Helper ─────────────────────────────────────────────────────────────

def _get_user_from_request(request: Request) -> Optional[dict]:
    """Extract authenticated user from JWT. Returns None if not authenticated."""
    from auth import get_current_user, auth_enabled
    if not auth_enabled():
        return None
    return get_current_user(request)


def _require_user(request: Request) -> dict:
    """Extract user or raise 401."""
    user = _get_user_from_request(request)
    if not user:
        raise _http_error(401, "Authentication required")
    return user


def _require_admin(request: Request) -> dict:
    """Extract admin user or raise 403."""
    user = _require_user(request)
    if not user.get("is_admin") and user.get("role") != "admin":
        raise _http_error(403, "Admin access required")
    return user


class _HTTPError(Exception):
    def __init__(self, status: int, message: str):
        self.status = status
        self.message = message


def _http_error(status: int, message: str) -> _HTTPError:
    return _HTTPError(status, message)


# ── JSON Serializer ─────────────────────────────────────────────────────────

def _serialize_user(user: dict) -> dict:
    """Convert DB user row to JSON-safe dict."""
    import datetime
    result = {}
    for k, v in user.items():
        if isinstance(v, datetime.datetime):
            result[k] = v.isoformat()
        elif isinstance(v, datetime.date):
            result[k] = v.isoformat()
        elif hasattr(v, "hex"):  # UUID
            result[k] = str(v)
        else:
            result[k] = v
    # Never expose sensitive fields
    result.pop("settings_json", None)
    return result


# ── API Router: /api/users ──────────────────────────────────────────────────

router = APIRouter(prefix="/api/users", tags=["users"])


@router.get("/me")
async def get_me(request: Request):
    """Get current user profile from DB."""
    try:
        user = _require_user(request)
    except _HTTPError as e:
        return JSONResponse({"error": e.message}, e.status)

    email = user.get("sub", "")
    db_user = get_user_repo().get_by_email(email)
    if not db_user:
        return JSONResponse({"error": "User not found in database"}, 404)

    return _serialize_user(db_user)


@router.put("/me")
async def update_me(request: Request):
    """Update current user profile."""
    try:
        user = _require_user(request)
    except _HTTPError as e:
        return JSONResponse({"error": e.message}, e.status)

    body = await request.json()
    email = user.get("sub", "")
    db_user = get_user_repo().get_by_email(email)
    if not db_user:
        return JSONResponse({"error": "User not found"}, 404)

    # Validate name
    name = body.get("name", db_user.get("name", ""))
    if not isinstance(name, str) or len(name) > MAX_NAME_LENGTH:
        return JSONResponse({"error": f"Name must be a string under {MAX_NAME_LENGTH} chars"}, 400)

    # Validate settings_json: whitelist keys, limit size
    settings = body.get("settings_json", {})
    if not isinstance(settings, dict) or len(settings) > MAX_SETTINGS_KEYS:
        return JSONResponse({"error": f"settings_json must be an object with max {MAX_SETTINGS_KEYS} keys"}, 400)
    invalid_keys = set(settings.keys()) - ALLOWED_SETTINGS_KEYS
    if invalid_keys:
        return JSONResponse({"error": f"Invalid settings keys: {', '.join(sorted(invalid_keys))}"}, 400)
    for v in settings.values():
        if not isinstance(v, (str, int, bool)) or (isinstance(v, str) and len(v) > MAX_SETTINGS_VALUE_LENGTH):
            return JSONResponse({"error": "Settings values must be string/int/bool, strings max 500 chars"}, 400)

    updated = get_user_repo().update_profile(str(db_user["user_id"]), name, settings)
    return _serialize_user(updated)


# ── Credentials ─────────────────────────────────────────────────────────────

@router.post("/me/credentials")
async def store_credential(request: Request):
    """Store encrypted broker credentials."""
    try:
        user = _require_user(request)
    except _HTTPError as e:
        return JSONResponse({"error": e.message}, e.status)

    body = await request.json()
    broker = body.get("broker", "")
    credentials = body.get("credentials", {})
    label = body.get("label", "default")
    is_paper = body.get("is_paper", True)

    if broker not in ("alpaca", "oanda", "ibkr"):
        return JSONResponse({"error": "Invalid broker. Must be: alpaca, oanda, ibkr"}, 400)
    if not credentials:
        return JSONResponse({"error": "credentials object required"}, 400)

    email = user.get("sub", "")
    db_user = get_user_repo().get_by_email(email)
    if not db_user:
        return JSONResponse({"error": "User not found"}, 404)

    try:
        result = get_cred_mgr().store(
            str(db_user["user_id"]), broker, credentials, label, is_paper
        )
    except RuntimeError as e:
        return JSONResponse({"error": str(e)}, 503)

    return JSONResponse(result, 201)


@router.get("/me/credentials")
async def list_credentials(request: Request):
    """List credentials (no secrets returned)."""
    try:
        user = _require_user(request)
    except _HTTPError as e:
        return JSONResponse({"error": e.message}, e.status)

    email = user.get("sub", "")
    db_user = get_user_repo().get_by_email(email)
    if not db_user:
        return JSONResponse({"error": "User not found"}, 404)

    creds = get_cred_mgr().list_for_user(str(db_user["user_id"]))
    return [_serialize_user(c) for c in creds]


@router.delete("/me/credentials/{credential_id}")
async def delete_credential(credential_id: int, request: Request):
    """Delete a stored credential."""
    try:
        user = _require_user(request)
    except _HTTPError as e:
        return JSONResponse({"error": e.message}, e.status)

    email = user.get("sub", "")
    db_user = get_user_repo().get_by_email(email)
    if not db_user:
        return JSONResponse({"error": "User not found"}, 404)

    deleted = get_cred_mgr().delete(str(db_user["user_id"]), credential_id)
    if not deleted:
        return JSONResponse({"error": "Credential not found"}, 404)
    return {"deleted": True}


# ── Config ──────────────────────────────────────────────────────────────────

@router.get("/me/config")
async def get_config(request: Request):
    """Get per-user trading config."""
    try:
        user = _require_user(request)
    except _HTTPError as e:
        return JSONResponse({"error": e.message}, e.status)

    email = user.get("sub", "")
    db_user = get_user_repo().get_by_email(email)
    if not db_user:
        return JSONResponse({"error": "User not found"}, 404)

    return get_config_mgr().get_config(str(db_user["user_id"]))


@router.put("/me/config")
async def update_config(request: Request):
    """Update per-user trading config (partial merge)."""
    try:
        user = _require_user(request)
    except _HTTPError as e:
        return JSONResponse({"error": e.message}, e.status)

    body = await request.json()
    email = user.get("sub", "")
    db_user = get_user_repo().get_by_email(email)
    if not db_user:
        return JSONResponse({"error": "User not found"}, 404)

    try:
        result = get_config_mgr().update_config(str(db_user["user_id"]), body)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, 400)

    return result


# ── Subscription ────────────────────────────────────────────────────────────

@router.get("/me/subscription")
async def get_subscription(request: Request):
    """Get subscription and tier info."""
    try:
        user = _require_user(request)
    except _HTTPError as e:
        return JSONResponse({"error": e.message}, e.status)

    email = user.get("sub", "")
    db_user = get_user_repo().get_by_email(email)
    if not db_user:
        return JSONResponse({"error": "User not found"}, 404)

    return get_sub_mgr().get_subscription(str(db_user["user_id"]))


# ── Admin Router: /api/admin ────────────────────────────────────────────────

admin_router = APIRouter(prefix="/api/admin", tags=["admin"])


@admin_router.get("/users")
async def admin_list_users(request: Request):
    """List all users (admin only)."""
    try:
        _require_admin(request)
    except _HTTPError as e:
        return JSONResponse({"error": e.message}, e.status)

    users = get_user_repo().list_all()
    return [_serialize_user(u) for u in users]


@admin_router.put("/users/{user_id}/tier")
async def admin_update_tier(user_id: str, request: Request):
    """Update a user's tier (admin only)."""
    try:
        _require_admin(request)
    except _HTTPError as e:
        return JSONResponse({"error": e.message}, e.status)

    body = await request.json()
    tier = body.get("tier", "")
    if tier not in VALID_TIERS:
        return JSONResponse(
            {"error": f"Invalid tier. Must be one of: {', '.join(sorted(VALID_TIERS))}"},
            400,
        )

    updated = get_user_repo().update_tier(user_id, tier)
    if not updated:
        return JSONResponse({"error": "User not found"}, 404)

    return {"user_id": str(updated["user_id"]), "tier": updated["tier"], "updated": True}


@admin_router.get("/tiers")
async def get_tiers(request: Request):
    """Get available tiers and their limits."""
    try:
        _require_user(request)
    except _HTTPError as e:
        return JSONResponse({"error": e.message}, e.status)

    return TIER_LIMITS


@admin_router.get("/engines")
async def admin_list_engines(request: Request):
    """List all active per-user trading engines (admin only)."""
    try:
        _require_admin(request)
    except _HTTPError as e:
        return JSONResponse({"error": e.message}, e.status)

    try:
        from user_engine import get_engine_manager, MAX_ENGINES
        mgr = get_engine_manager()
        if mgr is None:
            return {"engines": {}, "count": 0, "max": MAX_ENGINES}
        active = mgr.get_all_active()
        enriched = {}
        for uid, info in active.items():
            user = get_user_repo().get_by_id(uid)
            enriched[uid] = {
                **info,
                "email": user["email"] if user else "unknown",
                "tier": user["tier"] if user else "unknown",
            }
        return {"engines": enriched, "count": mgr.engine_count, "max": MAX_ENGINES}
    except Exception as e:
        logger.error("Failed to list engines: %s", e)
        return JSONResponse({"error": str(e)}, 500)
