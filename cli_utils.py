"""Shared utilities for CLI scripts (stop_trading, emergency_stop, status_trading)."""

import os
from typing import Dict


def auth_headers() -> Dict[str, str]:
    """Build auth headers from TRADING_API_KEY env var if set."""
    key = os.getenv("TRADING_API_KEY", "")
    return {"Authorization": f"Bearer {key}"} if key else {}
