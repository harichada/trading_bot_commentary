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

v-playbook-desk-b1-2026-09-25 adds two optional, additive sections:

    "regime_rows": {"<SYMBOL>": {"<strategy>:<side>:<regime>": {...}}},
    "pooled_rows": {"<strategy>:<side>:<regime>": {... "symbol": "*",
                                                   "source": "pooled"}}

A snapshot written before they existed simply has neither key, which reads as
"no regime-conditioned evidence" and falls back to the unconditional rows.

`StrategyFit` watches the file's mtime and reloads it on change, never
raising to callers: a missing file, a corrupt file, or a stale snapshot
all degrade to "no lanes allowed" rather than an exception on the hot
trading path. `core/` must not depend on `research/` — this module only
reads the JSON the research pipeline produces.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from typing import Callable, Dict, List, Optional

from core.strategy_fit_rows import FitRow


# ────────────────────────────────────────────────────────────────────
# Row model — see core/strategy_fit_rows.py
# ────────────────────────────────────────────────────────────────────

# Re-exported: `from core.strategy_fit import FitRow` stays the public path.
__all__ = ["FitRow", "StrategyFit"]


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
        # v-playbook-desk-b1-2026-09-25: regime-conditioned lanes, keyed
        # "<strategy>:<side>:<regime>" (per symbol, and pooled across the universe).
        self._regime_rows: Dict[str, Dict[str, FitRow]] = {}
        self._pooled_rows: Dict[str, FitRow] = {}
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
                regime_rows = self._parse_nested(raw, "regime_rows")
                pooled_rows = self._parse_flat(raw, "pooled_rows")
            except Exception as exc:  # noqa: BLE001 - never raise to caller
                self._last_error = f"parse failed: {exc}"
                return False

            computed_at_raw = raw.get("computed_at") if isinstance(raw, dict) else None
            self._rows = rows
            self._regime_rows = regime_rows
            self._pooled_rows = pooled_rows
            self._computed_at = computed_at_raw if isinstance(computed_at_raw, str) else None
            self._mtime = mtime
            self._last_error = None
            return True

    @staticmethod
    def _parse_rows(raw: object) -> Dict[str, Dict[str, FitRow]]:
        return StrategyFit._parse_nested(raw, "rows")

    @staticmethod
    def _parse_nested(raw: object, key: str) -> Dict[str, Dict[str, FitRow]]:
        """{symbol: {lane_key: row}} for the snapshot section `key`.

        A missing section (older snapshots have no "regime_rows") and any
        non-dict entry inside it are skipped rather than raised on.
        """
        rows_raw = raw.get(key, {}) if isinstance(raw, dict) else {}
        if not isinstance(rows_raw, dict):
            return {}
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

    @staticmethod
    def _parse_flat(raw: object, key: str) -> Dict[str, FitRow]:
        """{lane_key: row} for a flat (symbol-less) snapshot section."""
        rows_raw = raw.get(key, {}) if isinstance(raw, dict) else {}
        if not isinstance(rows_raw, dict):
            return {}
        return {lane_key: FitRow.from_dict(row) for lane_key, row in rows_raw.items()
                if isinstance(row, dict)}

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

    def lookup_regime(self, symbol: str, strategy: str, side: str, regime: str) -> Optional[FitRow]:
        """Best available row for this lane *in this regime*, in precedence order:

        1. the symbol's own row for that regime,
        2. the universe-pooled row for that regime (`symbol == "*"`, `pooled`),
        3. the unconditional row for the lane,
        4. None.

        A missing regime row means "insufficient sample", not "no edge": the scorer only
        writes one once the symbol has traded that regime enough times to measure it
        (`score/regime_rows.REGIME_ROW_MIN_N`), precisely so a thin sample cannot
        manufacture a FAIL that outranks the pooled or unconditional evidence below it.

        A stale, missing or unloaded snapshot yields None, exactly as `lookup` does.
        """
        if self.is_stale():
            return None
        key = f"{strategy}:{side}:{regime}"
        with self._lock:
            row = self._regime_rows.get(symbol, {}).get(key) or self._pooled_rows.get(key)
        if row is not None:
            return row
        return self.lookup(symbol, strategy, side)   # outside the lock: lookup takes it

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
            regime_rows = self._regime_rows
            pooled_rows = self._pooled_rows
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
            "n_regime_rows": sum(len(lanes) for lanes in regime_rows.values()),
            "n_pooled_pass": sum(1 for row in pooled_rows.values() if row.verdict == _PASS_VERDICT),
            "last_error": last_error,
        }
