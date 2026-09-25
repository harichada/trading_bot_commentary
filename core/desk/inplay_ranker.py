"""v-playbook-desk-b2-2026-09-25: the in-play score — which stocks are worth
watching today.

One pure function scores an `InPlayRow` from four opening features and another
ranks a list of them. No I/O, no clock, no config: the desk loop gathers the
features (`core/desk/inplay_features.py`), this module only decides the order.

    score = 0.40 * min(rvol_open15 / 3, 1)      # volume vs its own normal
          + 0.20 * min(|gap_pct| / 5, 1)        # overnight repricing
          + 0.20 * catalyst                     # earnings/news, 0 or 1
          + 0.20 * min(|rs_20| / 10, 1)         # relative strength, either way

A feature that is missing (NaN, or a non-finite value, which `InPlayRow`
serializes as null) scores 0 and leaves a `missing:<field>` reason rather than
knocking the symbol off the list: a stock with a real gap and a real catalyst
still ranks even when its volume history is thin. `pinned` symbols are always
in the output, appended after the ranked top_n when they did not make the cut.
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from core.desk.models import InPlayRow

__all__ = ["score_row", "rank_in_play", "RVOL_CAP", "GAP_CAP", "RS_CAP",
           "W_RVOL", "W_GAP", "W_CATALYST", "W_RS", "PINNED_REASON"]

# Saturation points: at or beyond these the component is worth its full weight.
RVOL_CAP = 3.0          # 3x its own opening volume
GAP_CAP = 5.0           # 5 % overnight move
RS_CAP = 10.0           # 10 % 20-day relative strength

W_RVOL, W_GAP, W_CATALYST, W_RS = 0.4, 0.2, 0.2, 0.2

PINNED_REASON = "pinned"
CATALYST_ON = 0.5       # catalyst is 0/1; anything at or above this counts as on

# Scores are rounded so that float noise (0.2 + 0.1 + 0.1 = 0.4000000000000001)
# can never reorder two rows that the formula says are tied, and so the journal
# and the API show a stable number.
SCORE_DIGITS = 6


# ────────────────────────────────────────────────────────────────────
# helpers
# ────────────────────────────────────────────────────────────────────

def _finite(value: Any) -> Optional[float]:
    """The value as a float, or None when it is missing/unusable/non-finite."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _ratio(value: Any, cap: float, absolute: bool) -> Optional[float]:
    """`value / cap` clamped to [0, 1]; None when the value is unusable."""
    number = _finite(value)
    if number is None:
        return None
    if absolute:
        number = abs(number)
    return max(0.0, min(number / cap, 1.0))


def _symbol(value: Any) -> str:
    return value.strip().upper() if isinstance(value, str) else ""


def _with_reason(reasons: List[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _flag_pinned(row: InPlayRow) -> InPlayRow:
    if PINNED_REASON in row.reasons:
        return row
    return replace(row, reasons=row.reasons + (PINNED_REASON,))


def _top_n(value: Any) -> int:
    """A non-negative int; anything unusable means "rank nothing" rather than raising."""
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _as_row(item: Any) -> InPlayRow:
    return item if isinstance(item, InPlayRow) else InPlayRow.from_dict(item)


# ────────────────────────────────────────────────────────────────────
# scoring
# ────────────────────────────────────────────────────────────────────

def score_row(row: InPlayRow) -> InPlayRow:
    """A new row carrying `score` and any `missing:<field>` reasons.

    Never mutates `row` and never raises: every field is coerced, and an
    unusable one contributes 0.
    """
    reasons = list(row.reasons)

    def component(value: Any, field: str, cap: float, absolute: bool) -> float:
        share = _ratio(value, cap, absolute)
        if share is None:
            _with_reason(reasons, f"missing:{field}")
            return 0.0
        return share

    # rvol is one-sided (volume is never negative); gap and rs count both ways.
    rvol = component(row.rvol_open15, "rvol_open15", RVOL_CAP, absolute=False)
    gap = component(row.gap_pct, "gap_pct", GAP_CAP, absolute=True)

    catalyst_value = _finite(row.catalyst)
    if catalyst_value is None:
        _with_reason(reasons, "missing:catalyst")
        catalyst = 0.0
    else:
        catalyst = 1.0 if catalyst_value >= CATALYST_ON else 0.0

    rs = component(row.rs_20, "rs_20", RS_CAP, absolute=True)

    score = W_RVOL * rvol + W_GAP * gap + W_CATALYST * catalyst + W_RS * rs
    return replace(row, score=round(score, SCORE_DIGITS), reasons=tuple(reasons))


# ────────────────────────────────────────────────────────────────────
# ranking
# ────────────────────────────────────────────────────────────────────

def _scored_by_symbol(candidates: Optional[Iterable[Any]]) -> Dict[str, InPlayRow]:
    """Score every candidate, keyed by normalized symbol, best score winning a tie."""
    scored: Dict[str, InPlayRow] = {}
    for item in candidates or ():
        row = _as_row(item)
        symbol = _symbol(row.symbol)
        if not symbol:
            continue
        out = score_row(replace(row, symbol=symbol))
        previous = scored.get(symbol)
        if previous is None or out.score > previous.score:
            scored[symbol] = out
    return scored


def _pinned_symbols(pinned: Optional[Iterable[Any]]) -> List[str]:
    out: List[str] = []
    for item in pinned or ():
        symbol = _symbol(item)
        if symbol and symbol not in out:
            out.append(symbol)
    return out


def rank_in_play(candidates: Sequence[InPlayRow], pinned: Sequence[str],
                 top_n: int) -> Tuple[InPlayRow, ...]:
    """The day's in-play list: the top `top_n` by score, then the pinned leftovers.

    Order is fully determined by the inputs — score descending, then symbol —
    so two runs over the same candidates produce the same list. A pinned symbol
    that is absent from `candidates` still appears, scored 0 with its missing
    features spelled out; every pinned symbol in the output carries `"pinned"`.
    """
    scored = _scored_by_symbol(candidates)
    pins = _pinned_symbols(pinned)

    ranked = sorted(scored.values(), key=lambda r: (-r.score, r.symbol))
    head = ranked[:_top_n(top_n)]
    taken = {row.symbol for row in head}

    kept = [_flag_pinned(row) if row.symbol in pins else row for row in head]
    tail = [_flag_pinned(scored.get(symbol) or score_row(InPlayRow(symbol=symbol)))
            for symbol in pins if symbol not in taken]
    return tuple(kept + tail)
