"""
Per-User Engine Pool — Multi-tenant Rudra Trading Engine

Manages one GapFadeLiveTrader per authenticated user. Each user gets
their own engine with their decrypted broker credentials and config.
When auth is disabled, all requests map to the global system trader.
"""

import asyncio
import logging
import os
from typing import Any, Optional

logger = logging.getLogger("GapFadeApp.engine_pool")

# ── System User ─────────────────────────────────────────────────────────────

SYSTEM_USER_ID: Optional[str] = None  # resolved at init time

MAX_ENGINES = int(os.environ.get("MAX_USER_ENGINES", "50"))


def _resolve_system_user_id() -> Optional[str]:
    """Look up the system@rudra.local user_id from DB."""
    try:
        from user_management import get_user_repo
        user = get_user_repo().get_by_email("system@rudra.local")
        return str(user["user_id"]) if user else None
    except Exception as e:
        logger.warning("Could not resolve system user: %s", e)
        return None


# ── Profit Cap Checker ──────────────────────────────────────────────────────

class ProfitCapChecker:
    """Check if a user has hit their tier's monthly profit cap."""

    def can_enter(self, user_id: str) -> tuple[bool, str]:
        """Returns (allowed, reason). Called before position entry."""
        if not user_id or user_id == SYSTEM_USER_ID:
            return True, ""  # system user has no cap

        try:
            from user_management import get_user_repo, TIER_LIMITS
            pnl_info = get_user_repo().get_monthly_pnl(user_id)
            if pnl_info is None:
                return True, ""

            tier = pnl_info.get("tier", "free")
            cap = TIER_LIMITS.get(tier, {}).get("monthly_profit_cap")
            if cap is None:  # enterprise = unlimited
                return True, ""

            current_pnl = float(pnl_info.get("monthly_pnl", 0))
            if current_pnl >= cap:
                return False, (
                    f"Monthly profit cap reached (${current_pnl:,.0f}/${cap:,} "
                    f"for {tier} tier). Upgrade to increase limit."
                )
            return True, ""
        except Exception as e:
            logger.error("Profit cap check failed (allowing entry): %s", e)
            return True, ""  # fail-open: don't block trading on DB errors


# ── User Engine Manager ─────────────────────────────────────────────────────

class UserEngineManager:
    """Manages one GapFadeLiveTrader per authenticated user.

    Thread/async safe. Engines are lazy-created on first API request.
    The global trader (system user) is always available as fallback.
    """

    def __init__(self, global_trader: Any):
        self._engines: dict[str, Any] = {}  # user_id -> GapFadeLiveTrader
        self._global_trader = global_trader
        self._lock = asyncio.Lock()
        self._profit_checker = ProfitCapChecker()

    @property
    def global_trader(self) -> Any:
        return self._global_trader

    @property
    def engine_count(self) -> int:
        return len(self._engines)

    async def get_trader(self, user_id: str) -> Any:
        """Get or create a trader for the given user_id.

        Returns the global trader ONLY when auth is disabled (system user).
        For authenticated users without credentials, returns None.
        """
        # Auth disabled or no user -> global trader (backward compat)
        if not user_id or user_id == SYSTEM_USER_ID:
            return self._global_trader

        # Check if already provisioned
        if user_id in self._engines:
            return self._engines[user_id]

        # Provision new engine under lock
        async with self._lock:
            # Double-check after acquiring lock
            if user_id in self._engines:
                return self._engines[user_id]

            if len(self._engines) >= MAX_ENGINES:
                raise RuntimeError(
                    f"Engine pool full ({MAX_ENGINES} max). "
                    "Contact admin to increase capacity."
                )

            engine = await self._provision_engine(user_id)
            if engine is None:
                # No credentials — user needs to add them on Account page
                logger.info("No credentials for user %s — engine not provisioned", user_id[:8])
                return None

            self._engines[user_id] = engine
            logger.info(
                "Provisioned engine for user %s (%d/%d active)",
                user_id[:8], len(self._engines), MAX_ENGINES
            )
            return engine

    async def _provision_engine(self, user_id: str) -> Optional[Any]:
        """Create a GapFadeLiveTrader with user-specific credentials and config."""
        try:
            from user_management import get_cred_mgr, get_config_mgr

            # Get user's broker credentials
            creds = get_cred_mgr().get_all_for_user(user_id)
            alpaca_cred = next(
                (c for c in creds if c["broker"] == "alpaca"), None
            )
            if not alpaca_cred:
                return None  # no Alpaca credentials = can't trade

            # Build alpaca config from user's credentials
            user_creds = alpaca_cred["credentials"]
            alpaca_config = {
                "api_key": user_creds.get("api_key", ""),
                "secret_key": user_creds.get("secret_key", ""),
                "trade_api_key": user_creds.get("trade_api_key", "") or user_creds.get("api_key", ""),
                "trade_secret_key": user_creds.get("trade_secret_key", "") or user_creds.get("secret_key", ""),
                "base_url": "https://paper-api.alpaca.markets" if alpaca_cred["is_paper"]
                            else "https://api.alpaca.markets",
                "data_url": "https://data.alpaca.markets",
            }

            # Get user's config overrides
            config_data = get_config_mgr().get_config(user_id)
            user_overrides = config_data.get("config", {})

            # Build GapFadeConfig with user overrides (whitelist against dataclass fields)
            from gap_fade_app import GapFadeConfig
            from user_management import VALID_CONFIG_FIELDS
            config = GapFadeConfig(**{
                k: v for k, v in user_overrides.items()
                if k in VALID_CONFIG_FIELDS
            })

            # Create trader with per-user config and credentials
            from gap_fade_app import GapFadeLiveTrader
            trader = GapFadeLiveTrader(
                config=config,
                alpaca_config=alpaca_config,
                user_id=user_id,
            )
            return trader

        except Exception as e:
            logger.error("Failed to provision engine for user %s: %s", user_id[:8], e)
            return None

    async def remove_trader(self, user_id: str) -> bool:
        """Stop and remove a user's engine. Returns True if removed."""
        async with self._lock:
            engine = self._engines.pop(user_id, None)
            if engine is None:
                return False
            try:
                await engine.stop()
            except Exception as e:
                logger.warning("Error stopping engine for user %s: %s", user_id[:8], e)
            logger.info("Removed engine for user %s", user_id[:8])
            return True

    def get_trader_sync(self, user_id: str) -> Any:
        """Sync version — returns engine only if already provisioned."""
        if not user_id or user_id == SYSTEM_USER_ID:
            return self._global_trader
        return self._engines.get(user_id, self._global_trader)

    def get_all_active(self) -> dict[str, dict]:
        """Return summary of all active engines (for admin)."""
        result = {}
        for uid, engine in self._engines.items():
            result[uid] = {
                "status": getattr(engine, "status", "unknown"),
                "positions": len(getattr(engine, "engine", None).positions)
                             if getattr(engine, "engine", None) else 0,
            }
        return result


# ── Request Helpers ─────────────────────────────────────────────────────────

def resolve_user_id(request) -> str:
    """Extract user_id from JWT, or return SYSTEM_USER_ID if auth disabled."""
    from auth import get_current_user, auth_enabled
    if not auth_enabled():
        return SYSTEM_USER_ID or ""
    user = get_current_user(request)
    if not user or not user.get("user_id"):
        return SYSTEM_USER_ID or ""
    return user["user_id"]


async def get_trader_for_request(request) -> Any:
    """One-liner for endpoints: resolve user -> get their engine.
    Returns None if user has no broker credentials configured."""
    if _engine_manager is None:
        # Module not initialized — return global trader via import
        from gap_fade_app import live_trader
        return live_trader
    user_id = resolve_user_id(request)
    return await _engine_manager.get_trader(user_id)


# Standard empty state for users without an engine
NO_ENGINE_STATE = {
    "status": "not_configured",
    "equity": 0,
    "daily_pnl": 0,
    "positions": {},
    "candidates": [],
    "messages": [],
    "needs_setup": True,
    "setup_message": "Add your broker credentials on the Account page to start trading.",
}


# ── Module Singleton ────────────────────────────────────────────────────────

_engine_manager: Optional[UserEngineManager] = None


def init_engine_manager(global_trader: Any) -> UserEngineManager:
    """Called once at app startup after live_trader is created."""
    global _engine_manager, SYSTEM_USER_ID
    SYSTEM_USER_ID = _resolve_system_user_id()
    _engine_manager = UserEngineManager(global_trader)
    logger.info(
        "UserEngineManager initialized (system_user=%s, max=%d)",
        SYSTEM_USER_ID[:8] if SYSTEM_USER_ID else "none", MAX_ENGINES
    )
    return _engine_manager


def get_engine_manager() -> Optional[UserEngineManager]:
    return _engine_manager
