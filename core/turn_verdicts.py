"""v-turn-engine-2026-10-05: the latest turn-score verdict per symbol.

The fit lane records a verdict every time `rsi_turn` scores a bar in the
oversold zone (every 5 minutes); the swing exit path reads it to decide
`thesis_failed` (spec §3.6). Thread-safe, bounded, process-wide.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional

MAX_SYMBOLS = 256


@dataclass(frozen=True)
class TurnVerdict:
    symbol: str
    ts: datetime
    score: float
    theta: float
    cvd_60: float
    vetoes: tuple
    detail: Mapping[str, Any] = field(default_factory=dict)

    def age_s(self, now: Optional[datetime] = None) -> float:
        now = now or datetime.now(timezone.utc)
        a, b = self.ts, now
        if (a.tzinfo is None) != (b.tzinfo is None):
            b = b.replace(tzinfo=None) if b.tzinfo is not None else b
            a = a.replace(tzinfo=None) if a.tzinfo is not None else a
        return (b - a).total_seconds()


class TurnVerdicts:
    _singleton: Optional["TurnVerdicts"] = None

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_symbol: Dict[str, TurnVerdict] = {}

    @classmethod
    def instance(cls) -> "TurnVerdicts":
        if cls._singleton is None:
            cls._singleton = TurnVerdicts()
        return cls._singleton

    def record(self, verdict: TurnVerdict) -> None:
        with self._lock:
            if len(self._by_symbol) >= MAX_SYMBOLS and verdict.symbol.upper() not in self._by_symbol:
                oldest = min(self._by_symbol.items(), key=lambda kv: kv[1].ts)[0]
                del self._by_symbol[oldest]
            self._by_symbol[verdict.symbol.upper()] = verdict

    def latest(self, symbol: str) -> Optional[TurnVerdict]:
        with self._lock:
            return self._by_symbol.get((symbol or "").upper())

    def clear(self) -> None:
        with self._lock:
            self._by_symbol.clear()
