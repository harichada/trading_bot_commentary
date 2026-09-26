"""Entry: a candidate trade proposed by a family at the open of `entry_ts`. Frozen and validated."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

SIDES = ("long", "short")
# "overnight" (Batch-1 Task 1): entered at 15:55 and closed at the next session's open. It is a
# distinct kind because the simulator exits it at that open, not at a session close like
# "multiday" -- and because the live side must not treat it as a day trade.
HOLD_KINDS = ("intraday", "multiday", "overnight")
# How the simulator is to fill the entry at `entry_ts` (Batch-2 Task B). The style is part of the
# candidate, not of the cost model, because it changes *whether* the trade happens at all:
#   "market"     -- taken at the fill bar's open, paying half-spread + slippage (the Batch-1 lane).
#   "limit_open" -- a resting limit a fixed offset inside the fill bar's open, filled at that price
#                   only if the bar trades through it (otherwise no trade), at maker cost.
ENTRY_STYLES = ("market", "limit_open")


@dataclass(frozen=True)
class Entry:
    symbol: str
    side: str
    entry_ts: pd.Timestamp
    entry_px: float
    stop_px: float
    target_px: float
    time_stop: int
    hold_kind: str
    tag: str = ""
    entry_style: str = "market"

    def __post_init__(self) -> None:
        if self.side not in SIDES:
            raise ValueError(f"side must be one of {SIDES}, got {self.side!r}")
        if self.hold_kind not in HOLD_KINDS:
            raise ValueError(f"hold_kind must be one of {HOLD_KINDS}, got {self.hold_kind!r}")
        if self.entry_style not in ENTRY_STYLES:
            raise ValueError(f"entry_style must be one of {ENTRY_STYLES}, got {self.entry_style!r}")
        if min(self.entry_px, self.stop_px, self.target_px) <= 0:
            raise ValueError("prices must be positive")
        if self.time_stop <= 0:
            raise ValueError("time_stop must be positive")
        ok = (self.stop_px < self.entry_px < self.target_px) if self.side == "long" else (self.target_px < self.entry_px < self.stop_px)
        if not ok:
            raise ValueError("stop/target must bracket entry on the correct side")

    @property
    def risk_per_share(self) -> float:
        return abs(self.entry_px - self.stop_px)
