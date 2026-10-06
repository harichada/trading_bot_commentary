"""The fit lane's shadow/live ledger writer (one ndjson line per kept entry).
Extracted from strategies/fit_lane.py (v-turn-engine-2026-10-05) to keep that
file inside its line budget; behaviour unchanged except the optional
`thesis` factor block the rsi_turn family adds.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("TradingBot")


def append_ledger(path: Path, family_name: str, now_ts, symbol: str, kept: list, rows_by_side: dict,
                  params: dict, thesis: Optional[dict] = None) -> None:
    """Best-effort: a bad ledger_dir (unwritable, or a file where a directory
    is expected) must not stop a live/shadow decision — it is just logged."""
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as fh:
            for entry in kept:
                row = rows_by_side.get(entry.side)
                fh.write(json.dumps({
                    "ts": now_ts.isoformat(),
                    "symbol": symbol,
                    "side": entry.side,
                    "entry_px": entry.entry_px,
                    "stop_px": entry.stop_px,
                    "target_px": entry.target_px,
                    "time_stop": entry.time_stop,
                    "hold_kind": entry.hold_kind,
                    "tag": entry.tag,
                    "params": params,
                    "verdict": str(getattr(row, "verdict", "")),
                    "score": getattr(row, "score", None),
                    **({"thesis": thesis} if thesis is not None else {}),
                }) + "\n")
    except Exception as exc:
        logger.warning("fit_lane: ledger write failed for %s/%s: %s", family_name, symbol, exc)


def append_thesis_only(path: Path, now_ts, symbol: str, thesis: dict) -> None:
    """One line for a scored oversold bar that produced no entry (spec §3.8:
    every occurrence is logged with its factor breakdown)."""
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as fh:
            fh.write(json.dumps({"ts": now_ts.isoformat(), "symbol": symbol, "event": "zone_no_entry",
                                 "thesis": thesis}) + "\n")
    except Exception as exc:
        logger.warning("fit_lane: thesis ledger write failed for %s: %s", symbol, exc)
