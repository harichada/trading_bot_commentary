"""v-shortable-check-2026-04-29: pre-trade shortable / easy-to-borrow gate.

Before the bot submits a SHORT signal, this checks whether the symbol is
actually shortable at the broker. Avoids the "order rejected — hard to
borrow" runtime error that would surface deep in the live execution path.

Source: Alpaca's free /v2/assets/{symbol} endpoint. Returns:
  shortable        — broker-accepted short (subset of marginable stocks)
  easy_to_borrow   — broker has shares available without HTB borrow fees

Caching:
  In-memory dict keyed by symbol with 6-hour TTL. Status occasionally
  changes intraday (HTB list updates after the open) but most names are
  stable. Cache survives only the process — restart re-fetches.

Failure modes:
  Network error / 404 → assume_shortable_on_error=True (default) so a
  transient API outage doesn't block all shorts. Set False to fail closed
  if you'd rather skip the trade than risk an HTB rejection.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import aiohttp

logger = logging.getLogger("TradingBot")

ALPACA_ASSETS_URL = "https://api.alpaca.markets/v2/assets/{symbol}"


@dataclass
class ShortableInfo:
    symbol: str
    shortable: bool
    easy_to_borrow: bool
    source: str  # 'alpaca' / 'cache' / 'fallback'
    fetched_at: float


class ShortableChecker:
    """Stateless API wrapper with in-memory cache."""

    def __init__(
        self,
        cache_ttl_sec: int = 6 * 3600,
        assume_shortable_on_error: bool = True,
    ) -> None:
        self.cache_ttl_sec = cache_ttl_sec
        self.assume_shortable_on_error = assume_shortable_on_error
        self._cache: dict[str, ShortableInfo] = {}
        self._api_key = os.getenv("ALPACA_API_KEY", "")
        self._api_sec = os.getenv("ALPACA_SECRET_KEY", "")

    async def is_shortable(self, symbol: str) -> ShortableInfo:
        """Returns ShortableInfo for the symbol. Never raises."""
        symbol = symbol.upper()
        now = time.time()

        # Cache hit?
        cached = self._cache.get(symbol)
        if cached is not None and (now - cached.fetched_at) < self.cache_ttl_sec:
            return ShortableInfo(
                symbol=cached.symbol,
                shortable=cached.shortable,
                easy_to_borrow=cached.easy_to_borrow,
                source="cache",
                fetched_at=cached.fetched_at,
            )

        # Need credentials
        if not (self._api_key and self._api_sec):
            return ShortableInfo(
                symbol=symbol,
                shortable=self.assume_shortable_on_error,
                easy_to_borrow=False,
                source="fallback_no_creds",
                fetched_at=now,
            )

        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=4)
            ) as sess:
                async with sess.get(
                    ALPACA_ASSETS_URL.format(symbol=symbol),
                    headers={
                        "APCA-API-KEY-ID": self._api_key,
                        "APCA-API-SECRET-KEY": self._api_sec,
                        "accept": "application/json",
                    },
                ) as r:
                    if r.status == 200:
                        data = await r.json()
                        info = ShortableInfo(
                            symbol=symbol,
                            shortable=bool(data.get("shortable", False)),
                            easy_to_borrow=bool(data.get("easy_to_borrow", False)),
                            source="alpaca",
                            fetched_at=now,
                        )
                        self._cache[symbol] = info
                        return info
                    elif r.status == 404:
                        # Symbol not on Alpaca's universe — likely fine on
                        # Schwab (e.g., some OTC names). Default conservative.
                        info = ShortableInfo(
                            symbol=symbol,
                            shortable=False,
                            easy_to_borrow=False,
                            source="alpaca_404",
                            fetched_at=now,
                        )
                        self._cache[symbol] = info
                        return info
                    else:
                        logger.debug(
                            f"shortable_check {symbol} http {r.status}"
                        )
        except Exception as exc:
            logger.debug(f"shortable_check error for {symbol}: {exc}")

        # Network / unknown failure — apply policy default. Not cached so
        # next call retries (cache only persists known-good answers).
        return ShortableInfo(
            symbol=symbol,
            shortable=self.assume_shortable_on_error,
            easy_to_borrow=False,
            source="fallback_error",
            fetched_at=now,
        )
