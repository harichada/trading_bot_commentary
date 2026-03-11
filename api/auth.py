"""API authentication module for the trading bot.

Provides a FastAPI dependency (verify_api_key) and a WebSocket helper
(verify_ws_token) that both validate against the TRADING_API_KEY environment
variable.  When the env var is unset the module degrades gracefully and
allows all requests — suitable for local development.

Security: all token comparisons use secrets.compare_digest to prevent
timing-based side-channel attacks.
"""

import os
import secrets
import logging
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

logger = logging.getLogger("TradingBot")
security = HTTPBearer(auto_error=False)

# Module-level flag so the "auth disabled" warning is only logged once per
# process lifetime (avoids log spam on every request).
_warned_no_key: bool = False


def verify_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[str]:
    """FastAPI dependency that verifies a Bearer token against TRADING_API_KEY.

    Behaviour:
    - TRADING_API_KEY unset  → auth disabled, all requests allowed (dev mode).
    - Correct Bearer token   → returns the token string.
    - Wrong / missing token  → raises HTTP 401.

    Args:
        credentials: Injected by FastAPI from the Authorization header.

    Returns:
        The validated token string, or None when auth is disabled.

    Raises:
        HTTPException: 401 when auth is enabled and the token is invalid/absent.
    """
    global _warned_no_key  # noqa: PLW0603

    api_key = os.getenv("TRADING_API_KEY")

    if not api_key:
        if not _warned_no_key:
            logger.warning(
                "TRADING_API_KEY not set - API authentication disabled (dev mode)"
            )
            _warned_no_key = True
        return None

    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization header",
        )

    if not secrets.compare_digest(credentials.credentials, api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    return credentials.credentials


def verify_ws_token(token: Optional[str]) -> bool:
    """Verify a WebSocket authentication token.

    Args:
        token: The token string received from the WebSocket client.

    Returns:
        True  when TRADING_API_KEY is unset (dev mode) or the token matches.
        False when TRADING_API_KEY is set but the token is absent or wrong.
    """
    api_key = os.getenv("TRADING_API_KEY")

    if not api_key:
        return True

    if not token:
        return False

    return secrets.compare_digest(token, api_key)
