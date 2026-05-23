"""Schwab token TTL health check.

Schwab issues a refresh_token that expires 7 days after creation.
When it expires, the access_token (which auto-refreshes from it)
can no longer be renewed, and the bot will get invalid_token errors
mid-session.

This module reads token_1.json, compares creation_timestamp to now,
and emits a clear warning at startup if the remaining lifetime is
short. It does NOT block startup — it warns and returns the remaining
TTL so callers can decide what to do.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger("TradingBot")

# Schwab refresh-token lifetime in days. Per Schwab docs, refresh tokens
# expire 7 days after creation (no auto-renewal of the refresh itself).
_REFRESH_TOKEN_DAYS = 7


def check_token_health(token_path: Path) -> Optional[timedelta]:
    """Return remaining lifetime of the Schwab refresh token, or None
    if the token file is missing or malformed.

    Side effect: logs warnings if the remaining lifetime is short.
    Severity thresholds:
        * <24h:  WARNING with explicit re-auth instructions
        * <48h:  INFO note that re-auth is needed within 2 days
        * <72h:  DEBUG/no-op (still healthy)
    """
    if not token_path.exists():
        logger.warning("token_health: token file missing at %s", token_path)
        return None
    try:
        with token_path.open("r") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("token_health: failed to read %s: %s", token_path, exc)
        return None

    creation_ts = data.get("creation_timestamp")
    if not isinstance(creation_ts, (int, float)):
        logger.warning(
            "token_health: %s missing creation_timestamp (got %r)",
            token_path, creation_ts,
        )
        return None

    created = datetime.fromtimestamp(creation_ts)
    expiry = created + timedelta(days=_REFRESH_TOKEN_DAYS)
    remaining = expiry - datetime.now()

    hours = remaining.total_seconds() / 3600
    if remaining.total_seconds() <= 0:
        logger.error(
            "token_health: REFRESH TOKEN EXPIRED at %s. Bot will fail "
            "on first Schwab call. Re-auth required: see Schwab OAuth "
            "flow / setup_wizard.py.",
            expiry.strftime("%Y-%m-%d %H:%M"),
        )
    elif hours < 24:
        logger.warning(
            "token_health: refresh token expires in %.1fh (at %s). "
            "Re-auth required TODAY before bot stops working mid-session.",
            hours, expiry.strftime("%Y-%m-%d %H:%M"),
        )
    elif hours < 48:
        logger.info(
            "token_health: refresh token expires in %.1fh (at %s). "
            "Schedule re-auth within 2 days.",
            hours, expiry.strftime("%Y-%m-%d %H:%M"),
        )
    else:
        logger.info(
            "token_health: refresh token valid for %.1f days (until %s).",
            remaining.total_seconds() / 86400,
            expiry.strftime("%Y-%m-%d %H:%M"),
        )

    return remaining
