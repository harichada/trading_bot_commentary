"""Entry: a candidate trade proposed by a family at the open of `entry_ts`. Frozen and validated."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

SIDES = ("long", "short")
HOLD_KINDS = ("intraday", "multiday")


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

    def __post_init__(self) -> None:
        if self.side not in SIDES:
            raise ValueError(f"side must be one of {SIDES}, got {self.side!r}")
        if self.hold_kind not in HOLD_KINDS:
            raise ValueError(f"hold_kind must be one of {HOLD_KINDS}, got {self.hold_kind!r}")
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
