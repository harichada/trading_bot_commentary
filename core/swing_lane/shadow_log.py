"""SPEC §7 shadow log: append-only NDJSON, one row per signal/event, every field present."""
from __future__ import annotations

import json
import math
import os
from datetime import date, datetime, timezone
from pathlib import Path

FIELDS = (
    "ts", "run_id", "setup", "sym", "signal_date", "regime_on", "close", "atr14", "rvol", "gap",
    "rng20_atr", "earnings_next", "blackout_skip", "rank", "slot_available", "planned_entry",
    "planned_stop", "planned_target", "planned_shares", "planned_risk_usd", "fill_date",
    "fill_price", "slippage_bps", "stop_updates", "exit_date", "exit_price", "exit_reason",
    "gap_through", "pnl_usd", "R", "mae_R", "mfe_R", "hold_days", "skipped_reason",
    "code_version", "params_hash",
)
EXTRA_FIELDS = ("event",)
EXIT_REASONS = frozenset({"stop", "gapstop", "target", "trail", "time", "earnings"})


def _clean(v):
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    if hasattr(v, "isoformat"):
        return v.isoformat()
    if hasattr(v, "item"):
        return v.item()
    return v


def make_row(**kw) -> dict:
    unknown = set(kw) - set(FIELDS) - set(EXTRA_FIELDS)
    if unknown:
        raise KeyError(f"unknown shadow fields: {sorted(unknown)}")
    if kw.get("exit_reason") is not None and kw["exit_reason"] not in EXIT_REASONS:
        raise ValueError(f"bad exit_reason {kw['exit_reason']!r}")
    row = {f: None for f in FIELDS}
    row["stop_updates"] = []
    row["event"] = None
    row.update(kw)
    if row["ts"] is None:
        row["ts"] = datetime.now(timezone.utc).isoformat()
    return {k: (_clean(v) if k != "stop_updates" else [_clean(x) for x in (v or [])]) for k, v in row.items()}


class ShadowLog:
    def __init__(self, log_dir: str):
        self.path = Path(log_dir) / "swing_shadow.ndjson"

    def append(self, row: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(row, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def read(self) -> list:
        if not self.path.exists():
            return []
        with open(self.path) as f:
            return [json.loads(x) for x in f if x.strip()]
