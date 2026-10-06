"""v-algorithms-page-2026-10-06: the last decisions of every strategy, in memory.

`log_strategy_decision` (strategies/base.py) already writes one audit line per
evaluation; this keeps the same facts in a bounded ring per strategy so the
dashboard's Algorithms page can show "what did each algorithm decide, and
why" without grepping the log or hitting Postgres. Process-wide, thread-safe,
O(1) per record; ~300 rows x ~25 strategies of small dicts.
"""
from __future__ import annotations

import threading
from collections import Counter, deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional

RING = 300


class DecisionBook:
    _singleton: Optional["DecisionBook"] = None

    def __init__(self, ring: int = RING) -> None:
        self._lock = threading.Lock()
        self._ring = max(10, int(ring))
        self._rows: Dict[str, Deque[dict]] = {}
        self._counts: Dict[str, Counter] = {}        # (strategy) -> Counter of "action:reason" for today
        self._day: Optional[str] = None

    @classmethod
    def instance(cls) -> "DecisionBook":
        if cls._singleton is None:
            cls._singleton = DecisionBook()
        return cls._singleton

    def record(self, strategy: str, symbol: str, action: str, reason: str, close: Any = None,
               ts: Optional[datetime] = None, **details: Any) -> None:
        ts = ts or datetime.now(timezone.utc)
        day = ts.strftime("%Y-%m-%d")
        row = {"ts": ts.isoformat(timespec="seconds"), "symbol": str(symbol or ""), "action": str(action or ""),
               "reason": str(reason or ""), "close": _num(close),
               "details": {k: _jsonable(v) for k, v in list(details.items())[:12]}}
        with self._lock:
            if day != self._day:
                self._day = day
                self._counts = {}
            self._rows.setdefault(strategy, deque(maxlen=self._ring)).append(row)
            self._counts.setdefault(strategy, Counter())[f"{row['action']}:{row['reason']}"] += 1

    def strategies(self) -> List[str]:
        with self._lock:
            return sorted(self._rows)

    def recent(self, strategy: str, limit: int = 25) -> List[dict]:
        with self._lock:
            rows = list(self._rows.get(strategy, ()))
        return list(reversed(rows))[:max(0, int(limit))]

    def counts(self, strategy: str) -> Dict[str, Any]:
        with self._lock:
            c = self._counts.get(strategy, Counter())
            by_action: Counter = Counter()
            for key, n in c.items():
                by_action[key.split(":", 1)[0]] += n
            top = c.most_common(6)
            rows = list(self._rows.get(strategy, ()))
        return {"today_total": sum(c.values()), "by_action": dict(by_action),
                "top_reasons": [{"action": k.split(":", 1)[0], "reason": k.split(":", 1)[1], "n": n} for k, n in top],
                "last_ts": rows[-1]["ts"] if rows else None, "day": self._day}

    def clear(self) -> None:
        with self._lock:
            self._rows.clear()
            self._counts.clear()


def _num(v: Any) -> Optional[float]:
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def _jsonable(v: Any) -> Any:
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    return str(v)[:80]
