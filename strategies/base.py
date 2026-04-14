"""Base class for trading strategies."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

from core.models import TradingSignal


_logger = logging.getLogger("TradingBot")


class TradingStrategyWithCommentary(ABC):
    """Base class for strategies with commentary."""

    name: str = "strategy"  # subclasses override

    def __init__(self, commentary_system):
        self.commentary = commentary_system

    @abstractmethod
    async def generate_signal_with_commentary(self, market_data) -> Optional[TradingSignal]:
        ...

    def _log_decision(self, market_data, action: str, reason: str, **details: Any) -> None:
        """Structured per-tick decision log.

        Every strategy call should emit exactly one of these so the log file
        is a complete audit trail even when the UI commentary stays quiet.
        action: 'signal_buy' | 'signal_sell' | 'skip' | 'error'
        reason: short snake_case label
        """
        symbol = getattr(market_data, "symbol", "?")
        close = getattr(market_data, "close", None)
        kv = " ".join(f"{k}={v}" for k, v in details.items())
        _logger.info(
            "strategy_decision strategy=%s symbol=%s action=%s reason=%s close=%s %s",
            self.name, symbol, action, reason, close, kv,
        )
