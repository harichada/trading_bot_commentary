"""Domain dataclasses for the backtest package.

Mirrors the types previously defined inline in ``backtest_strategies.py``.
The ``Trade`` type gains an ``entry_indicators`` snapshot dictionary that
the engine populates when a strategy emits a signal — this lets downstream
tooling (e.g. CSV writers, post-hoc analysis) inspect the exact indicator
values at the moment the bot took the trade.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Union

import pandas as pd


@dataclass
class Trade:
    """A single replayed trade.

    Mutable only because it's constructed incrementally in
    ``replay_symbol``; treat as read-only after that. ``entry_indicators``
    is the snapshot of ``MarketData.indicators`` captured at entry, with
    NaNs already filtered out by the engine guard.
    """

    strategy: str
    symbol: str
    side: str  # "long" | "short"
    entry_time: Any
    entry_price: float
    exit_time: Any
    exit_price: float
    exit_reason: str  # "stop" | "target" | "timeout" | "eod"
    hold_bars: int
    entry_indicators: dict = field(default_factory=dict)

    @property
    def return_pct(self) -> float:
        direction = 1 if self.side == "long" else -1
        return direction * (self.exit_price - self.entry_price) / self.entry_price * 100


@dataclass(frozen=True)
class SymbolMetrics:
    """Per-(strategy, symbol) drill-down row."""

    n_trades: int
    win_rate_pct: float
    sum_return_pct: float
    avg_return_pct: float


@dataclass(frozen=True)
class ProgressEvent:
    """Pushed to ``progress_cb`` during a run.

    ``stage`` = ``"loading_data" | "replaying"``.
    """

    stage: str
    symbols_done: int
    symbols_total: int
    current_symbol: str


# Type alias: progress callbacks may be sync or async.
ProgressCb = Callable[[ProgressEvent], Union[None, Awaitable[None]]]
# ``bars_loader`` signature: ``(symbol, days, frequency) -> DataFrame``.
BarsLoader = Callable[[str, int, int], pd.DataFrame]
