"""Fit snapshot reader: strategy-fit verdicts computed offline by the
research pipeline, read here for gating live trading decisions.

The research pipeline periodically writes a JSON snapshot of the form:

    {
        "computed_at": "<ISO timestamp, no timezone>",
        "floors": {...},
        "rows": {
            "<SYMBOL>": {
                "<strategy>:<side>": {
                    "symbol": ..., "strategy": ..., "side": ...,
                    "verdict": "PASS"|"WATCH"|"FAIL", "score": ...,
                    "exp_r": ..., "exp_r_lb": ..., "t_stat": ..., "pf": ...,
                    "wr": ..., "stability": ..., "decay": ..., "n": ...,
                    "source": ..., "params_json": "<json string>",
                    "computed_at": ...
                },
                ...
            },
            ...
        }
    }

`StrategyFit` watches the file's mtime and reloads it on change, never
raising to callers: a missing file, a corrupt file, or a stale snapshot
all degrade to "no lanes allowed" rather than an exception on the hot
trading path. `core/` must not depend on `research/` — this module only
reads the JSON the research pipeline produces.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional


# ────────────────────────────────────────────────────────────────────
# Row model
# ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FitRow:
    """One (symbol, strategy, side) fit verdict from the snapshot.

    Frozen: a row is a point-in-time read from the snapshot and should
    never be mutated by a caller.
    """
    symbol: str
    strategy: str
    side: str
    verdict: str
    exp_r: float
    exp_r_lb: float
    t_stat: float
    pf: float
    wr: float
    stability: float
    decay: float
    n: int
    score: float
    source: str
    params: Dict = field(default_factory=dict)
    computed_at: str = ""

    @staticmethod
    def from_dict(d: Dict) -> "FitRow":
        """Build a FitRow from a raw snapshot row dict.

        Tolerant of missing keys (default to nan/0/"") and of a
        `params_json` field that is a JSON string rather than a dict.
        """
        def _f(key: str, default: float) -> float:
            try:
                return float(d.get(key, default))
            except (TypeError, ValueError):
                return default

        def _i(key: str, default: int) -> int:
            try:
                return int(d.get(key, default))
            except (TypeError, ValueError):
                return default

        def _s(key: str, default: str) -> str:
            v = d.get(key, default)
            return v if isinstance(v, str) else default

        params = d.get("params")
        if not isinstance(params, dict):
            raw = d.get("params_json")
            try:
                parsed = json.loads(raw) if isinstance(raw, str) else {}
                params = parsed if isinstance(parsed, dict) else {}
            except (TypeError, ValueError, json.JSONDecodeError):
                params = {}

        return FitRow(
            symbol=_s("symbol", ""),
            strategy=_s("strategy", ""),
            side=_s("side", ""),
            verdict=_s("verdict", ""),
            exp_r=_f("exp_r", float("nan")),
            exp_r_lb=_f("exp_r_lb", float("nan")),
            t_stat=_f("t_stat", float("nan")),
            pf=_f("pf", float("nan")),
            wr=_f("wr", float("nan")),
            stability=_f("stability", float("nan")),
            decay=_f("decay", float("nan")),
            n=_i("n", 0),
            score=_f("score", float("nan")),
            source=_s("source", ""),
            params=params,
            computed_at=_s("computed_at", ""),
        )

    def to_json_dict(self) -> Dict:
        """JSON-safe dict of this row: every non-finite float becomes None.

        v-strategy-fit-review2-2026-09-25 (Important 5): cold-start rows
        legitimately carry NaN metrics (`from_dict` defaults every missing
        numeric to nan). `json.dumps` emits bare `NaN`, which is not valid
        JSON — FastAPI's encoder rejects it and `GET /api/strategy-fit`
        answered 500 for any symbol with a cold-start row. `None` is the
        honest wire representation of "not measured".
        """
        out: Dict = {}
        for key, value in dataclasses.asdict(self).items():
            if isinstance(value, float) and not math.isfinite(value):
                out[key] = None
            else:
                out[key] = value
        return out


# ────────────────────────────────────────────────────────────────────
# Snapshot reader
# ────────────────────────────────────────────────────────────────────

_PASS_VERDICT = "PASS"


class StrategyFit:
    """Thread-safe reader over the research pipeline's fit snapshot.

    Both the engine loop and the API loop read this concurrently, so a
    single `threading.Lock` guards reload and reads of the row map.
    No public method ever raises — any failure is recorded in
    `last_error` and the previous (possibly empty) row set is kept.
    """

    def __init__(
        self,
        path: str,
        max_age_days: int = 7,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self._path = path
        self._max_age_days = max_age_days
        self._clock = clock or datetime.now
        self._lock = threading.Lock()

        self._rows: Dict[str, Dict[str, FitRow]] = {}
        self._computed_at: Optional[str] = None
        self._mtime: Optional[float] = None
        self._last_error: Optional[str] = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def reload_if_changed(self) -> bool:
        """Reload the snapshot if its mtime changed since the last load.

        Returns True only when a load succeeded and replaced the rows.
        Returns False when unchanged, missing, or the load failed (in
        which case the previous rows are kept and `last_error` is set).
        Never raises.
        """
        with self._lock:
            try:
                mtime = os.stat(self._path).st_mtime
            except OSError as exc:
                self._last_error = f"stat failed: {exc}"
                return False

            if self._mtime is not None and mtime == self._mtime:
                return False

            try:
                with open(self._path, "r") as fh:
                    raw = json.load(fh)
            except json.JSONDecodeError as exc:
                self._last_error = f"json load failed: {exc}"
                return False
            except Exception as exc:  # noqa: BLE001 - last-resort: OSError,
                # UnicodeDecodeError (binary/undecodable file), etc. Never
                # raise out of reload_if_changed.
                self._last_error = f"load failed: {exc}"
                return False

            try:
                rows = self._parse_rows(raw)
            except Exception as exc:  # noqa: BLE001 - never raise to caller
                self._last_error = f"parse failed: {exc}"
                return False

            computed_at_raw = raw.get("computed_at") if isinstance(raw, dict) else None
            self._rows = rows
            self._computed_at = computed_at_raw if isinstance(computed_at_raw, str) else None
            self._mtime = mtime
            self._last_error = None
            return True

    @staticmethod
    def _parse_rows(raw: object) -> Dict[str, Dict[str, FitRow]]:
        rows_raw = raw.get("rows", {}) if isinstance(raw, dict) else {}
        parsed: Dict[str, Dict[str, FitRow]] = {}
        for symbol, lanes in rows_raw.items():
            if not isinstance(lanes, dict):
                continue
            lane_map: Dict[str, FitRow] = {}
            for lane_key, row_dict in lanes.items():
                if not isinstance(row_dict, dict):
                    continue
                lane_map[lane_key] = FitRow.from_dict(row_dict)
            parsed[symbol] = lane_map
        return parsed

    # ------------------------------------------------------------------
    # Age / staleness
    # ------------------------------------------------------------------

    def age_days(self) -> Optional[float]:
        """Age of the loaded snapshot in days, or None if unknown."""
        with self._lock:
            computed_at = self._computed_at
        if not computed_at:
            return None
        try:
            computed_dt = datetime.fromisoformat(computed_at)
        except (TypeError, ValueError):
            return None
        delta = self._clock() - computed_dt
        return delta.total_seconds() / 86400.0

    def is_stale(self) -> bool:
        age = self.age_days()
        return age is None or age > self._max_age_days

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def lookup(self, symbol: str, strategy: str, side: str) -> Optional[FitRow]:
        """Return the row for (symbol, strategy, side), or None when
        stale, missing, or unloaded."""
        if self.is_stale():
            return None
        key = f"{strategy}:{side}"
        with self._lock:
            return self._rows.get(symbol, {}).get(key)

    def allowed_lanes(self, symbol: str) -> List[FitRow]:
        """PASS rows for `symbol`, sorted by score desc then strategy
        asc. Empty when stale, missing, or unloaded."""
        if self.is_stale():
            return []
        with self._lock:
            lanes = list(self._rows.get(symbol, {}).values())
        passing = [r for r in lanes if r.verdict == _PASS_VERDICT]
        return sorted(passing, key=lambda r: (-r.score, r.strategy))

    def rows_for(self, symbol: str) -> List[FitRow]:
        """All rows for `symbol` across every strategy and side,
        regardless of verdict. Empty when stale, missing, or unloaded."""
        if self.is_stale():
            return []
        with self._lock:
            lanes = list(self._rows.get(symbol, {}).values())
        return sorted(lanes, key=lambda r: (r.strategy, r.side))

    def params_for(self, symbol: str, strategy: str, side: str = "long") -> Dict:
        """Chosen params for (symbol, strategy, side), or {} when
        stale, missing, or unloaded."""
        row = self.lookup(symbol, strategy, side)
        return row.params if row is not None else {}

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> Dict:
        with self._lock:
            rows = self._rows
            computed_at = self._computed_at
            last_error = self._last_error

        counts: Dict[str, int] = {}
        sources: Dict[str, int] = {}
        n_symbols = len(rows)
        for lanes in rows.values():
            for row in lanes.values():
                counts[row.verdict] = counts.get(row.verdict, 0) + 1
                sources[row.source] = sources.get(row.source, 0) + 1

        return {
            "path": self._path,
            "computed_at": computed_at,
            "age_days": self.age_days(),
            "stale": self.is_stale(),
            "counts": counts,
            "sources": sources,
            "n_symbols": n_symbols,
            "last_error": last_error,
        }
