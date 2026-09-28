"""Live-admission rule for the fit lanes: which entries may become real orders.

v-go-live-5k-2026-09-28 (Task 2). Until now a fit-lane entry could go live only when the
walk-forward pipeline had written a PASS row for that exact (symbol, family, side)
(`live_eligible`, strategies/fit_lane_frames.py). No PASS row exists anywhere in the current fit
snapshot, so every lane was shadow-only in practice — which is the right default, but it leaves
no way to put a *deliberately chosen* sleeve to work on the desk's own authority.

`FIT_LANE_LIVE_ALLOWLIST` is that authority, and nothing more: entries of the form

    family:side[:regime1|regime2]          e.g. "mr_20d:long,mom_12_1:long:risk_on|mixed"

An allowlisted (family, side) is admitted in a permitted regime *unless the pipeline measured
that symbol's lane and failed it* — the allowlist is a statement about a strategy, never a
licence to override evidence about a symbol. It also cannot make a family live: the family must
still be in `FIT_LANE_LIVE_ENTRIES_ENABLED`, and an admitted entry still faces every downstream
gate (desk limits, the playbook, the swing overnight gate, risk sizing and the allocation cap).

Default "" parses to no entries, which reduces `admit()` to exactly the old `live_eligible`
rule, so the whole file is inert until the flag is set.

This module owns the whole admission story, so it reads as one unit: the rule (`admit`), its
inputs (`read_live_regime`, `safe_regime`, `admissible_sides`) and the signal the admitted entry
becomes, carrying the audit fields the decision produced (`entry_to_signal`). Everything but the
regime read is pure. Never imports `research`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Tuple

from core.models import SignalType, TradingSignal
from strategies.families.models import Entry
from strategies.fit_lane_frames import _clamp, live_eligible

logger = logging.getLogger("TradingBot")

__all__ = [
    "ADMIT_ALLOWLIST", "entry_to_signal", "ADMIT_PASS", "AllowEntry", "REGIMES", "SIDES", "admissible_sides",
    "admit", "allowlist_permits", "allowlisted_sides", "normalize_regime", "parse_allowlist",
    "read_live_regime", "safe_regime",
]

SIDES = ("long", "short")
# The market-context regimes core/market_context.py can report. "unknown" is deliberately absent:
# it is not a regime an allowlist entry may name, it is the absence of a reading, and
# `normalize_regime` collapses it to the neutral "mixed" (the same treatment
# MarketContext.allows_long gives an unknown tape: permitted, but never as the risk_on case).
REGIMES = ("risk_on", "risk_off", "mixed")

ADMIT_PASS = "pass"              # the pre-existing rule: a walk-forward PASS row for that side
ADMIT_ALLOWLIST = "allowlist"    # admitted by FIT_LANE_LIVE_ALLOWLIST on a non-FAIL row

_FAIL_VERDICT = "FAIL"
_ANY_REGIME: Tuple[str, ...] = ()


@dataclass(frozen=True)
class AllowEntry:
    """One parsed allowlist entry. `regimes` empty means "any regime"."""

    family: str
    side: str
    regimes: Tuple[str, ...] = _ANY_REGIME

    def permits(self, regime: Any) -> bool:
        return not self.regimes or normalize_regime(regime) in self.regimes


def normalize_regime(regime: Any) -> str:
    """Lower-cased regime, with a missing/`unknown` reading collapsed to `mixed`.

    The market-indices cache reports `unknown` whenever it cannot see SPY/VIX (startup, a quote
    gap). Treating that as its own regime would either strand a `risk_on|mixed` sleeve for the
    first minutes of every session or, if listed, let a blind tape authorise trading — so it maps
    to the neutral middle, which an entry has to name explicitly to permit.
    """
    text = str(regime or "").strip().lower()
    return "mixed" if text in ("", "unknown") else text


_WARNED: set = set()


def _warn_once(raw: str) -> None:
    """One warning per distinct malformed entry: this is read on every analysis tick, so an
    unconditional log would flood the file, and silence would hide a typo that quietly narrows
    (or, worse, is assumed to widen) live trading."""
    if raw in _WARNED:
        return
    _WARNED.add(raw)
    logger.warning(
        "fit_lane allowlist: ignoring malformed entry %r (expected family:side[:regime|regime], "
        "side in %s, regime in %s)", raw, SIDES, REGIMES)


def _parse_entry(raw: str) -> Optional[AllowEntry]:
    parts = [p.strip().lower() for p in str(raw).split(":")]
    if len(parts) not in (2, 3):
        return None
    family, side = parts[0], parts[1]
    if not family or side not in SIDES:
        return None
    if len(parts) == 2:
        return AllowEntry(family, side, _ANY_REGIME)
    regimes = tuple(r for r in (p.strip() for p in parts[2].split("|")) if r)
    if not regimes or any(r not in REGIMES for r in regimes):
        return None
    return AllowEntry(family, side, regimes)


def parse_allowlist(raw: Any) -> Tuple[AllowEntry, ...]:
    """Parse `FIT_LANE_LIVE_ALLOWLIST` (a comma string, or the tuple Config() returns for a yaml
    list) into entries. Malformed entries are skipped with one warning each — a bad entry must
    never be guessed at, and must never stop the good entries (or the lane) from working."""
    if raw is None:
        items: Iterable[Any] = ()
    elif isinstance(raw, (list, tuple, set)):
        items = list(raw)
    else:
        items = str(raw).split(",")
    out = []
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        entry = _parse_entry(text)
        if entry is None:
            _warn_once(text)
            continue
        out.append(entry)
    return tuple(out)


def allowlisted_sides(allowlist: Iterable[AllowEntry], family: str) -> Tuple[str, ...]:
    """Sides `family` is allowlisted for, in any regime. The lane uses this as a cheap pre-check:
    a side listed here is worth computing an entry for even when the fit snapshot has no row for
    it at all (the regime and verdict checks still happen in `admit`)."""
    fam = str(family).strip().lower()
    return tuple(sorted({e.side for e in allowlist or () if e.family == fam}))


def allowlist_permits(allowlist: Iterable[AllowEntry], family: str, side: str, regime: Any) -> bool:
    """True when some entry allowlists this (family, side) for this regime."""
    fam, sd = str(family).strip().lower(), str(side).strip().lower()
    return any(e.family == fam and e.side == sd and e.permits(regime) for e in allowlist or ())


def _is_fail(row: Any) -> bool:
    return row is not None and str(getattr(row, "verdict", "")).strip().upper() == _FAIL_VERDICT


def admit(row: Any, family: str, side: str, regime: Any,
          allowlist: Iterable[AllowEntry]) -> Optional[str]:
    """The audit reason this entry may go live under, or None to keep it in shadow.

    `ADMIT_PASS`      — the pre-existing rule: `row` is a walk-forward PASS for this side.
    `ADMIT_ALLOWLIST` — (family, side) is allowlisted, the regime is permitted, and the symbol's
                        unconditional row is absent or is not FAIL.

    `row` is the symbol's *unconditional* row for this lane (`StrategyFit.lookup`), which is the
    row a FAIL has to come from: a regime-conditioned or pooled row is evidence about a regime or
    about the universe, not the pipeline's verdict on this symbol's lane.
    """
    if live_eligible(row):
        return ADMIT_PASS
    if allowlist_permits(allowlist, family, side, regime) and not _is_fail(row):
        return ADMIT_ALLOWLIST
    return None


# ── the admission rule's inputs ──


def read_live_regime(symbol: str) -> str:
    """The live market-context regime for `symbol` ("risk_on"|"risk_off"|"mixed"|"unknown").

    The fit lane builds signals *before* the router, so nothing has written
    `reasoning["market_context_regime"]` yet — but this rule is regime-conditional, and the
    engine's fit gate and playbook gate both read that key off the emitted signal. Reading it
    here (the same cache-backed, I/O-free `read_market_context` call `strategies/builtin.py`
    makes) and writing it into the reasoning means all three see one reading instead of three.
    "unknown" on any failure — never raises.
    """
    try:
        from core.market_context import read_market_context
        return str(read_market_context(symbol).regime or "unknown")
    except Exception:
        logger.debug("fit_lane: market-context regime unavailable for %s", symbol, exc_info=True)
        return "unknown"


def safe_regime(source: Any, symbol: str) -> str:
    """`source(symbol)` as a regime string, or "unknown" when it is empty or raises. A blind
    regime read must never take a lane down, and "unknown" is the conservative answer: it
    normalizes to `mixed`, which an allowlist entry has to name explicitly."""
    try:
        return str(source(symbol) or "unknown")
    except Exception:
        logger.debug("fit_lane: regime unreadable for %s", symbol, exc_info=True)
        return "unknown"


def admissible_sides(family: str, live_families: Any, allowlist: Any) -> Tuple[str, ...]:
    """Sides the allowlist could admit for `family`, and () unless the family is also in
    `live_families()`.

    The lane uses this as a cheap pre-check before any row lookup or entry computation: a side
    listed here is worth evaluating even when the fit snapshot has no row for it. Empty while
    the flag is "" (its default) or the family is shadow-only, so the lane's pre-existing
    no-row skip is untouched by default. Never raises — an unreadable flag means "no allowlist".
    """
    try:
        if family not in live_families():
            return ()
        return allowlisted_sides(parse_allowlist(allowlist()), family)
    except Exception:
        logger.debug("fit_lane: allowlist unreadable for %s", family, exc_info=True)
        return ()


# ── the signal an admitted entry becomes ──


def entry_to_signal(entry: Entry, family: str, params: dict, verdict: str, score: float, symbol: str,
                    admission: str = ADMIT_PASS, regime: Optional[str] = None) -> TradingSignal:
    """`entry.symbol` falls back to "" when the family's make_entry() call
    didn't pass one explicitly (the common case) — `symbol` is the
    market-data symbol the adapter is actually evaluating, and always wins
    when entry.symbol is empty.

    `admission` and `regime` (v-go-live-5k-2026-09-28) are audit fields: why this entry was
    allowed to be live, and the regime it was admitted on — the latter under the key every
    downstream gate already reads."""
    strength = _clamp(float(score) * 2.0, 0.2, 1.0)
    is_day_trade = entry.hold_kind == "intraday"
    reasoning = {
        "strategy": f"fit:{family}",
        "fit_family": family,
        "fit_params": params,
        "fit_verdict": verdict,
        "fit_lane_admission": admission,
        "market_context_regime": regime,
        "hold_kind": entry.hold_kind,
        "entry_pattern": entry.tag,
        "max_hold_sessions": None if is_day_trade else entry.time_stop,
        "is_day_trade": is_day_trade,
        "day_trade_size_multiplier": 0.5 if is_day_trade else 1.0,
        "flatten_hour": 15 if is_day_trade else None,
        "strategy_size_multiplier": 1.0,
    }
    return TradingSignal(
        symbol=entry.symbol or symbol,
        signal_type=SignalType.BUY if entry.side == "long" else SignalType.SELL,
        strength=strength,
        entry_price=entry.entry_px,
        stop_loss=entry.stop_px,
        take_profit=entry.target_px,
        position_size=0,
        reasoning=reasoning,
        confidence=strength,
    )
