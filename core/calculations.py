"""v-calc-engine-2026-05-01: stateless P&L computation against PriceBook.

Architectural role: the ONLY place P&L math happens. Position objects no
longer store current_price or unrealized_pnl — those are computed on
demand by these functions, sourcing the mark from the canonical
PriceBook. This forces the reactive-pull pattern: nothing can return a
stale price, because nothing stores one.

If the PriceBook says a symbol is stale or unknown, get_pnl returns a
PnLResult with is_stale=True and no price. Callers must inspect the
flag before acting (e.g. exit-loop skips decisions on stale; dashboard
greys out the cell).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.models import Position
    from core.price_book import PriceObject


@dataclass(frozen=True)
class PnLResult:
    """The complete reactive-pull P&L answer. Frozen so consumers can
    safely cache the snapshot they read."""
    symbol: str
    side: str                          # "long" | "short"
    quantity: float
    entry_price: float
    mark_price: Optional[float]        # None when no quote available
    unrealized_pnl: float              # 0.0 when no quote available
    unrealized_pnl_pct: float          # 0.0 when no quote / entry is 0
    market_value: float                # 0.0 when no quote
    is_stale: bool                     # True if mark is stale or missing
    source: str                        # forwarded from PriceObject.source

    @property
    def has_mark(self) -> bool:
        return self.mark_price is not None and self.mark_price > 0


class CalculationEngine:
    """Static-method facade — no state. The whole module is a function
    library; using a class so call sites read as
    `CalculationEngine.get_pnl(...)` per the architectural directive.
    """

    @staticmethod
    def get_pnl(position, price_object: "PriceObject") -> PnLResult:
        """Compute the position's mark-to-market P&L using the canonical
        price from the PriceBook.

        Side-aware: short positions profit when mark < entry. Returns
        is_stale=True (and zeroed P&L) when the price is missing or
        beyond the staleness threshold — callers must check before
        acting.
        """
        symbol = getattr(position, "symbol", "?") or "?"
        side = getattr(position, "side", "long") or "long"
        qty = float(getattr(position, "quantity", 0) or 0)
        entry = float(getattr(position, "entry_price", 0) or 0)

        # No tradable mark → safe defaults, is_stale=True
        if (price_object is None
                or not price_object.has_price
                or price_object.is_stale):
            return PnLResult(
                symbol=symbol, side=side, quantity=qty,
                entry_price=entry, mark_price=None,
                unrealized_pnl=0.0, unrealized_pnl_pct=0.0,
                market_value=0.0,
                is_stale=True,
                source=(price_object.source if price_object else "none"),
            )

        mark = price_object.price
        # Side-aware computation
        if side == "short":
            unreal = (entry - mark) * abs(qty)
            pct = ((entry - mark) / entry * 100.0) if entry > 0 else 0.0
        else:
            unreal = (mark - entry) * qty
            pct = ((mark - entry) / entry * 100.0) if entry > 0 else 0.0
        return PnLResult(
            symbol=symbol, side=side, quantity=qty,
            entry_price=entry, mark_price=mark,
            unrealized_pnl=unreal, unrealized_pnl_pct=pct,
            market_value=abs(qty) * mark,
            is_stale=False,
            source=price_object.source,
        )
