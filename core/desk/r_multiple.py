"""v-playbook-desk-b2-2026-09-25 (Task 3b): one pure R-multiple derivation shared by the
live-trade readers — `research/strategy_hunt/playbook_live.py` and
`data_providers/db_logger.py::get_bot_trades_between` — so both produce the same R
whether Postgres's `bot_trades` carries the migrated `pnl_r` column
(`scripts/migrate_ledger_integrity.py`, gated by `LEDGER_INTEGRITY`, default off) or not.

This deliberately does not reuse `core/ledger_integrity.py::compute_r_multiple`: that
function recomputes pnl-per-share from `entry_price`/`exit_price`/`side` against a
priority-ordered stop (`initial_stop`, then `stop_loss`, then a `1.5*ATR` fallback) — a
different, richer signature the un-migrated schema (no `initial_stop`) doesn't fit. The
ruling here is narrower: trust the row's own realized `pnl`, and fall back only to
`stop_loss` distance — never ATR — so a trade the bot never bracketed reads R `None`,
not a guess.

Lives in `core/` (not `research/`) because `research` may import `core` but never the
reverse, and `data_providers/db_logger.py` — a `core`-side module — needs this too.
"""
from __future__ import annotations

import math
from typing import Any, Optional

__all__ = ["derive_pnl_r"]


def derive_pnl_r(
    pnl_r: Any,
    pnl: Any,
    entry_price: Any,
    stop_loss: Any,
    quantity: Any,
) -> Optional[float]:
    """One R multiple for a `bot_trades` row, on either schema.

    Preference order:
      1. the row's own `pnl_r` (the migrated schema), when present and finite.
      2. `pnl / (abs(entry_price - stop_loss) * quantity)` (the un-migrated schema),
         when `pnl`, `entry_price`, `stop_loss` and `quantity` are all present, finite,
         and the denominator is positive — sign follows `pnl` since the denominator is
         never negative.
      3. `None` otherwise: a degenerate stop (`entry_price == stop_loss`), a missing
         field, or a non-finite value are all "R undefined", not a guess.
    """
    stored = _finite(pnl_r)
    if stored is not None:
        return stored

    realized = _finite(pnl)
    entry = _finite(entry_price)
    stop = _finite(stop_loss)
    qty = _finite(quantity)
    if realized is None or entry is None or stop is None or qty is None:
        return None

    denom = abs(entry - stop) * abs(qty)
    if denom <= 0:
        return None
    return realized / denom


def _finite(value: Any) -> Optional[float]:
    """`value` as a finite float, or None — covers missing fields (`None`), unreadable
    types, and non-finite results (`NaN`/`inf`, which a Postgres null surfaces as once
    it crosses pandas) uniformly."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None
