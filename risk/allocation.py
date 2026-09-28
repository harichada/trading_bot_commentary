"""v-go-live-5k-2026-09-28 (Task 1): the bot's capital-allocation cap.

`Config.BOT_CAPITAL_ALLOCATION` is the hard dollar slice of a bigger brokerage
account the bot is allowed to work with. Default 0.0 means *off* — and off must
be byte-identical to the pre-allocation behaviour, so every function here is
written so the caller can skip the whole step on `allocation <= 0`.

The pure rules, all total (they never raise, whatever comes in):

  0. `effective_value` — **the** cap: a raw balance as the bot may use it. Read
     at every consumption point (see `RiskManagerWithCommentary
     .effective_balance` / `.effective_buying_power`) so that the live sites
     which overwrite the raw attributes with whole-account Schwab numbers
     cannot defeat it.
  1. `capped_balance_pair` — one balance under the cap: the capped value the
     risk manager uses, the raw Schwab value kept for logging, and whether the
     input was unusable (in which case both previous values survive).
  2. `allocation_headroom` — how many dollars of the allocation are still free
     given the notional the bot already has open.
  3. `allocation_capped_shares` — the largest share count whose notional fits
     in that headroom; 0 when not even one share fits ("allocation exhausted").

And the two steps `risk/manager.py` calls, built out of those rules:

  * `capped_sync_balances` — the whole `sync_with_schwab_data` decision, as new
    values for the caller to assign (nothing is mutated in place).
  * `apply_allocation_cap` — the whole sizing step: the trimmed share count,
    plus the operator-facing log line and commentary.

All of it lives outside `risk/manager.py` to keep that module's edit localized
(two small call sites) and because the rules are then testable without the main
module's Config singleton (`risk.manager.Config` lazily imports
`trading_bot_commentary_updated`).
"""
from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Any, NamedTuple, Optional, Tuple

from core.commentary import TradingCommentary
from core.models import CommentaryType

logger = logging.getLogger('TradingBot')

# Commentary / audit reason emitted when a new position cannot be opened
# because the bot's allocation is fully deployed.
ALLOCATION_EXHAUSTED = "allocation_exhausted"

# Commentary / audit reason emitted when the allocation merely *trims* a
# position that would otherwise have been larger.
ALLOCATION_CAP = "allocation_cap"

# The "off" sentinel. Any value at or below this disables the whole feature.
ALLOCATION_OFF = 0.0


def finite_float(value: Any) -> Optional[float]:
    """`value` as a finite float, or None when it is not usable as a dollar
    amount.

    Rejects None, NaN, ±inf, non-numeric strings, containers — and bools,
    which Python would otherwise happily coerce to 0.0/1.0 and turn a config
    typo into a $1 allocation.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def coerce_allocation(raw: Any) -> float:
    """The configured allocation as a positive float, or `ALLOCATION_OFF`.

    A missing, zero, negative or unusable value all mean "no allocation cap" —
    the flag's default state, in which callers must leave today's behaviour
    untouched. Failing *open* here is deliberate: a malformed allocation must
    not silently become a tiny one that starves the bot of size.
    """
    value = finite_float(raw)
    if value is None or value <= ALLOCATION_OFF:
        return ALLOCATION_OFF
    return value


def effective_value(raw: Any, allocation: float) -> Any:
    """A raw balance as the bot is allowed to *use* it — the read-time cap.

    v-go-live-5k-r1-2026-09-28 (review round 0, critical 1): capping inside
    `sync_with_schwab_data` was not enough. Three live sites overwrite
    `risk_manager.buying_power` / `.account_balance` with whole-account Schwab
    numbers right after a sync — the engine's router before sizing,
    `_check_buying_power`, and the dashboard's `/ws` position-cache refresh
    every 45s — which restored an $18,600 per-position cap and a -$1,395 daily
    stop. So the clamp moved to the read: the plain attributes stay as write
    targets and every *consumer* goes through here, which no write site can
    defeat.

    With the allocation off this returns `raw` **itself**, untouched — not a
    coerced float. That is what keeps the off path byte-identical, including
    for values that would raise downstream today (a `None` buying power still
    raises the same TypeError at the same comparison it always did).

    With the allocation on, an unusable raw value yields 0.0: under a cap,
    "I cannot read the balance" must mean "no capital", never "all of it".
    """
    if allocation <= ALLOCATION_OFF:
        return raw
    value = finite_float(raw)
    if value is None:
        return 0.0
    return min(value, allocation)


def clamp_previous(prev: Any, allocation: float) -> float:
    """A previously stored balance made safe to reuse under the cap.

    `min(prev, allocation)`, falling back to `allocation` when `prev` is
    unusable (a pre-allocation sync may have stored None, and the seeded
    placeholder balance predates any cap). Bounded by the allocation either
    way, so an odd broker payload on the very first capped sync cannot leave
    whole-account equity in place.
    """
    value = finite_float(prev)
    if value is None:
        return allocation
    return min(value, allocation)


def capped_balance_pair(raw: Any, allocation: float, prev_raw: float,
                        prev_capped: float) -> Tuple[float, float, bool]:
    """One balance under the allocation cap: `(capped, raw_kept, odd)`.

    `allocation` is assumed > 0 — the caller handles the off case, which must
    not touch these values at all.

    * `capped` is `min(raw, allocation)`: what sizing and the daily-loss
      circuit read.
    * `raw_kept` is the unclamped Schwab number, kept for logging and the
      dashboard (`schwab_equity` / `schwab_buying_power`).
    * `odd` is True when `raw` was unusable (None/NaN/junk). In that case both
      previous values are returned unchanged so a bad broker payload can never
      zero the bot's balances — the caller logs it once.
    """
    value = finite_float(raw)
    if value is None:
        return prev_capped, prev_raw, True
    return min(value, allocation), value, False


def allocation_headroom(allocation: float, bot_open_notional: Any) -> float:
    """Dollars of the allocation still free: `max(0, allocation - open)`.

    An unusable `bot_open_notional` is treated as "fully deployed" (headroom
    0) rather than as zero: a broken notional must never *widen* the
    allowance. The engine sets the attribute once per cycle; its default of
    0.0 is the honest "nothing open yet" starting point.
    """
    alloc = finite_float(allocation)
    if alloc is None or alloc <= ALLOCATION_OFF:
        return 0.0
    open_notional = finite_float(bot_open_notional)
    if open_notional is None:
        return 0.0
    return max(0.0, alloc - max(0.0, open_notional))


def allocation_capped_shares(position_size: int, price: Any,
                             headroom: Any) -> int:
    """`position_size` trimmed so `shares x price <= headroom`.

    Returns 0 when one share already costs more than the remaining headroom —
    the exhaustion case, which `engine._execute_signal` reads as "no trade".

    An unusable `price` leaves the size alone (the price sanity checks upstream
    own that failure); an unusable `headroom` yields 0, matching
    `allocation_headroom`'s fail-closed stance.
    """
    try:
        shares = int(position_size)
    except (TypeError, ValueError):
        return 0
    if shares <= 0:
        return max(0, shares)

    unit = finite_float(price)
    if unit is None or unit <= 0:
        return shares

    room = finite_float(headroom)
    if room is None:
        return 0
    if room < unit:
        return 0
    return min(shares, int(room // unit))


class SyncedBalances(NamedTuple):
    """New balance values for `RiskManagerWithCommentary` to assign.

    `balance`/`buying_power` are what sizing and the daily-loss circuit read;
    `schwab_equity`/`schwab_buying_power` keep the raw broker numbers for the
    log line and the dashboard.
    """
    balance: float
    buying_power: float
    schwab_equity: float
    schwab_buying_power: float


def capped_sync_balances(raw_balance: Any, raw_buying_power: Any,
                         allocation: float, prev: SyncedBalances) -> SyncedBalances:
    """The capped-sync decision, as new values — nothing is mutated in place.

    `allocation` must be > 0; the caller keeps the pre-allocation assignments
    for the off case so that path stays byte-identical.

    An unusable broker figure (None/NaN/junk) leaves that pair at its previous
    value and is logged once here — a malformed payload must never zero the
    bot's balances, and must never raise inside the sync loop. The previous
    *capped* values are re-clamped to the allocation on the way in, so a bad
    payload on the very first capped sync cannot leave whole-account equity in
    place.
    """
    prev_equity = finite_float(prev.schwab_equity)
    prev_bp_raw = finite_float(prev.schwab_buying_power)
    bal_capped, bal_raw, bal_odd = capped_balance_pair(
        raw_balance, allocation,
        prev_raw=prev_equity if prev_equity is not None else allocation,
        prev_capped=clamp_previous(prev.balance, allocation))
    bp_capped, bp_raw, bp_odd = capped_balance_pair(
        raw_buying_power, allocation,
        prev_raw=prev_bp_raw if prev_bp_raw is not None else allocation,
        prev_capped=clamp_previous(prev.buying_power, allocation))

    if bal_odd or bp_odd:
        logger.warning(
            "allocation_sync_odd_balance balance=%r buying_power=%r — keeping "
            "previous equity=$%.2f buying_power=$%.2f",
            raw_balance, raw_buying_power, bal_raw, bp_raw,
        )

    return SyncedBalances(balance=bal_capped, buying_power=bp_capped,
                          schwab_equity=bal_raw, schwab_buying_power=bp_raw)


def apply_allocation_cap(position_size: int, current_price: float, symbol: str,
                         allocation: float, bot_open_notional: Any,
                         commentary) -> int:
    """The sizing chain's allocation step: the share count, capped.

    `allocation` must be > 0 (the caller skips this step entirely when the flag
    is off, which is what keeps sizing byte-identical by default).

    Returns 0 — read as "no trade" by `engine._execute_signal` — when the
    remaining headroom cannot fund a single share. Emits the operator-facing
    log line and commentary for both the trim and the exhaustion case, so the
    caller is a two-line call site.
    """
    headroom = allocation_headroom(allocation, bot_open_notional)
    new_size = allocation_capped_shares(position_size, current_price, headroom)
    if new_size == position_size:
        return position_size

    if new_size == 0:
        logger.warning(
            "%s symbol=%s allocation=$%.2f open_notional=%s headroom=$%.2f "
            "price=%.2f old=%d new=0",
            ALLOCATION_EXHAUSTED, symbol, allocation, bot_open_notional,
            headroom, current_price, position_size,
        )
        commentary.add_commentary(TradingCommentary(
            timestamp=datetime.now(),
            type=CommentaryType.WARNING,
            symbol=symbol,
            title="⛔ Allocation Exhausted",
            message=(f"The bot's ${allocation:,.0f} allocation is fully deployed "
                     f"(${headroom:,.2f} free) — not enough for one share at "
                     f"${current_price:,.2f}."),
            data={'reason': ALLOCATION_EXHAUSTED,
                  'allocation': allocation,
                  'bot_open_notional': bot_open_notional,
                  'headroom': headroom},
            importance=8,
        ))
        return 0

    logger.info(
        "allocation_size_cap symbol=%s allocation=$%.2f open_notional=%s "
        "headroom=$%.2f old=%d new=%d",
        symbol, allocation, bot_open_notional, headroom, position_size, new_size,
    )
    commentary.add_commentary(TradingCommentary(
        timestamp=datetime.now(),
        type=CommentaryType.RISK_ASSESSMENT,
        symbol=symbol,
        title="⚠️ Position Size Capped by Allocation",
        message=(f"Reduced from {position_size} to {new_size} shares — only "
                 f"${headroom:,.0f} of the ${allocation:,.0f} allocation is free."),
        data={'reason': ALLOCATION_CAP,
              'allocation': allocation,
              'bot_open_notional': bot_open_notional,
              'headroom': headroom},
        importance=7,
    ))
    return new_size
