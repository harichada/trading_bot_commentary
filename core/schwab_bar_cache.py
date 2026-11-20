"""Live 1-min bar cache fed by Schwab CHART_EQUITY stream.

SchwabStream writes completed 1-min bars here as they arrive from the
broker. load_bars() reads today's intraday slice from here so the shadow
loop always has zero-lag data without polling any external API.

Thread contract
---------------
Writes come from the asyncio event loop (stream handler, via
call_soon_threadsafe). Reads come from the shadow's sync tick loop.
The cache uses a threading.Lock so concurrent reads/writes are safe.
"""
from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd

_ET = ZoneInfo("America/New_York")
_MAX_BARS_PER_SYMBOL = 500   # ~8 trading hours of 1-min bars


class SchwabBarCache:
    """Singleton in-memory store of 1-min OHLCV bars per symbol.

    Bars are keyed by their open-timestamp in milliseconds (from
    Schwab's CHART_TIME_MILLIS field) so back-fills and duplicates are
    silently deduplicated via dict overwrite.
    """

    _instance: Optional["SchwabBarCache"] = None
    _lock = threading.Lock()

    @classmethod
    def instance(cls) -> "SchwabBarCache":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self) -> None:
        # symbol → { chart_time_millis: {open, high, low, close, volume} }
        self._bars: dict[str, dict[int, dict]] = {}
        self._rw_lock = threading.Lock()

    # ── write ──────────────────────────────────────────────────────────

    def upsert(self, symbol: str, chart_time_millis: int,
               open_: float, high: float, low: float,
               close: float, volume: float) -> None:
        """Store or update a single 1-min bar. Called from stream handler."""
        with self._rw_lock:
            sym_bars = self._bars.setdefault(symbol, {})
            sym_bars[chart_time_millis] = {
                "open": open_, "high": high, "low": low,
                "close": close, "volume": volume,
            }
            # Trim to the most recent N bars (by key order).
            if len(sym_bars) > _MAX_BARS_PER_SYMBOL:
                oldest = sorted(sym_bars)[:len(sym_bars) - _MAX_BARS_PER_SYMBOL]
                for k in oldest:
                    del sym_bars[k]

    # ── read ───────────────────────────────────────────────────────────

    def get_today_df(self, symbol: str) -> pd.DataFrame:
        """Return today's RTH 1-min bars as a UTC-aware DataFrame.

        Returns an empty DataFrame if no bars are cached for the symbol.
        Columns: Open, High, Low, Close, Volume. Index: UTC-aware DatetimeIndex.
        """
        with self._rw_lock:
            sym_bars = dict(self._bars.get(symbol, {}))

        if not sym_bars:
            return pd.DataFrame()

        rows = []
        for ts_ms, bar in sorted(sym_bars.items()):
            ts_utc = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
            rows.append({
                "ts": ts_utc,
                "Open": bar["open"],
                "High": bar["high"],
                "Low": bar["low"],
                "Close": bar["close"],
                "Volume": bar["volume"],
            })

        df = pd.DataFrame(rows).set_index("ts")

        # RTH filter: 9:30–15:55 ET
        et = df.index.tz_convert(_ET)
        today = datetime.now(_ET).date()
        minutes = et.hour * 60 + et.minute
        mask = (
            (et.date == today) &  # today only
            (minutes >= 9 * 60 + 30) &
            (minutes <= 15 * 60 + 55)
        )
        return df[mask]

    def has_today_bars(self, symbol: str) -> bool:
        """True if there is at least one RTH bar for today."""
        df = self.get_today_df(symbol)
        return not df.empty

    def symbols(self) -> list[str]:
        """Symbols currently in the cache."""
        with self._rw_lock:
            return list(self._bars.keys())
