"""Persistence helpers for backtest output.

Two writers, separated by concern:

- ``write_report_json(report, path)`` — same JSON shape as the previous
  monolithic ``backtest_strategies.py`` produced, so ``api/backtest.py``
  and the dashboard keep working unchanged.

- ``write_trades_csv(trades, path)`` — new MVP feature. One row per
  trade with all ``Trade`` fields flattened. ``entry_indicators`` is
  flattened with an ``ind_`` prefix (e.g. ``ind_rsi``, ``ind_atr``). The
  union of all indicator keys across all trades is used as the column
  set so a sparse trade simply gets blanks for indicators it didn't
  carry.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable

from backtest.trades import Trade

# Top-level dataclass fields that are always emitted as their own column,
# in this order. Anything else lives under ``entry_indicators`` and is
# flattened with the ``ind_`` prefix.
_TRADE_BASE_FIELDS: tuple[str, ...] = (
    "strategy",
    "symbol",
    "side",
    "entry_time",
    "entry_price",
    "exit_time",
    "exit_price",
    "exit_reason",
    "hold_bars",
)


def write_report_json(report: dict[str, Any], path: str | Path) -> Path:
    """Write the aggregated report dict to ``path`` as JSON.

    Uses ``default=str`` to handle ``datetime``/``Timestamp`` instances
    that come back inside the per-strategy summaries. Matches what
    ``api/backtest.py:_save_report`` does today.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2, default=str))
    return p


def _collect_indicator_columns(trades: Iterable[Trade]) -> list[str]:
    """Return a stable-sorted list of all indicator keys seen across trades."""
    keys: set[str] = set()
    for t in trades:
        keys.update(t.entry_indicators.keys())
    return sorted(keys)


def write_trades_csv(trades: list[Trade], path: str | Path) -> Path:
    """Write a flat CSV: one row per trade, indicators prefixed with ``ind_``.

    Even with zero trades we still write a header row so downstream
    tools can read the file without ``EmptyDataError``. The indicator
    columns will just be absent in that case.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    indicator_keys = _collect_indicator_columns(trades)
    header = list(_TRADE_BASE_FIELDS) + ["return_pct"] + [
        f"ind_{k}" for k in indicator_keys
    ]

    with p.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for t in trades:
            base_row: list[Any] = [getattr(t, f) for f in _TRADE_BASE_FIELDS]
            base_row.append(t.return_pct)
            ind_row = [t.entry_indicators.get(k, "") for k in indicator_keys]
            writer.writerow(base_row + ind_row)
    return p


# ---------------------------------------------------------------------------
# Misc convenience
# ---------------------------------------------------------------------------

def trade_to_dict(trade: Trade) -> dict[str, Any]:
    """Return a flat dict view of a ``Trade`` (for ad-hoc serialisation)."""
    if is_dataclass(trade):
        return asdict(trade)
    return {f: getattr(trade, f) for f in _TRADE_BASE_FIELDS}
