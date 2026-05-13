"""Pure-function P&L computations (v-day-pnl-intraday-entry-2026-05-12).

Helpers in this module take Schwab position dicts and return scalars.
No I/O, no logger writes by default, no engine state. Lives outside
core/engine.py so the unit tests can exercise the math without
pulling in the engine's heavyweight import chain (scipy, pandas,
schwab-py, etc.).
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger("TradingBot")

# A position whose currentDayProfitLoss magnitude exceeds this fraction
# of its market value is treated as a Schwab-side reporting outlier
# (see the PLTR same-day-lot-transferred case under
# v-schwab-pnl-fix-2026-04-21). 0.5 = 50 % single-day swing.
_OUTLIER_THRESHOLD = 0.5


def compute_position_day_pnl(
    position: Dict[str, Any],
    on_outlier: Optional[Callable[[str, float, float, float], None]] = None,
) -> float:
    """Pick the right day-P&L formula per position.

    Schwab returns two relevant fields per position:

      - ``currentDayProfitLoss``: broker's authoritative day P&L.
        Correct for both intraday entries and overnight positions
        because the broker knows each lot's entry price.
      - ``instrument.netChange``: per-share price move from
        **previous close**. For an intraday entry at price ≠
        today's open, ``netChange × qty`` attributes the pre-entry
        move to your P&L — over- or under-reporting depending on
        whether you entered after a rally or sell-off.

    The original v-schwab-pnl-fix-2026-04-21 preferred
    netChange × qty to dodge a PLTR same-day-lot-transferred outlier
    (currentDayProfitLoss returned +$21,932 on a position worth
    ~$3,000). That solved PLTR but corrupted day P&L for every
    normal intraday entry — observed 2026-05-12 on FCEL/ORCL/CRWV
    after the operator flagged a phantom-loss display.

    Strategy: trust currentDayProfitLoss unless its magnitude
    exceeds 50 % of the position's market value (= the PLTR
    outlier shape). On the outlier branch, fall back to
    netChange × qty and call ``on_outlier`` so callers can log.
    """
    qty = position.get("longQuantity", 0) - position.get("shortQuantity", 0)
    market_value = position.get("marketValue", 0) or 0
    net_change = (position.get("instrument") or {}).get("netChange", 0) or 0
    netchange_pnl = net_change * qty

    broker_pnl = position.get("currentDayProfitLoss", 0) or 0
    if broker_pnl == 0:
        broker_pnl = position.get("dayGainLoss", 0) or 0

    if (
        broker_pnl != 0
        and abs(market_value) > 0
        and abs(broker_pnl) > abs(market_value) * _OUTLIER_THRESHOLD
    ):
        symbol = (position.get("instrument") or {}).get("symbol", "?")
        if on_outlier is not None:
            on_outlier(symbol, broker_pnl, market_value, netchange_pnl)
        return netchange_pnl

    if broker_pnl != 0:
        return broker_pnl

    return netchange_pnl
