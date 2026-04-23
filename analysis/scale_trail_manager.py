"""Scale-out at 1R and ATR trailing stop manager.

Pure logic — no commentary, no side effects, no I/O. Returns action
dicts that the engine interprets and logs. Designed so every method
is independently testable with a Position dataclass and a price.

1R partial exit:  When price moves 1×stop_distance in our favor,
                  sell 50% and move stop to breakeven. Guarantees
                  at least half a win on any trade that reaches 1R.

ATR trailing:     After price moves 1×ATR in favor, the trailing
                  stop activates and ratchets at 1×ATR behind price.
                  Only moves in the favorable direction (up for longs,
                  down for shorts). Tighter than the initial 1.5×ATR
                  entry stop — designed to protect running profits.
"""
from __future__ import annotations

from typing import Optional

from core.models import Position


class PartialExitAction:
    __slots__ = ("exit_fraction", "new_stop", "exit_qty")

    def __init__(self, exit_fraction: float, new_stop: float, exit_qty: int) -> None:
        self.exit_fraction = exit_fraction
        self.new_stop = new_stop
        self.exit_qty = exit_qty


class ScaleTrailManager:
    """Stateless manager — all state lives on the Position object."""

    def check_partial_exit(
        self, position: Position, current_price: float
    ) -> Optional[PartialExitAction]:
        """Return a PartialExitAction if 1R has been reached, else None."""
        if position.scaled_out:
            return None

        original_stop = position.original_stop or position.stop_loss
        stop_distance = abs(position.entry_price - original_stop)
        if stop_distance <= 0:
            return None

        exit_qty = int(position.quantity * 0.5)
        if exit_qty < 1:
            return None  # can't split 1 share

        if position.side == "long":
            if current_price >= position.entry_price + stop_distance:
                return PartialExitAction(
                    exit_fraction=0.5,
                    new_stop=position.entry_price,
                    exit_qty=exit_qty,
                )
        else:  # short
            if current_price <= position.entry_price - stop_distance:
                return PartialExitAction(
                    exit_fraction=0.5,
                    new_stop=position.entry_price,
                    exit_qty=exit_qty,
                )
        return None

    def update_trailing_stop(
        self, position: Position, current_price: float, atr: float
    ) -> Optional[float]:
        """Return a new trailing-stop level if the trail should improve, else None.

        v-trail-widen-2026-04-23: activation + width now configurable via
        Config.TRAIL_ACTIVATION_ATR_MULT (default 2.0, was 1.0) and
        Config.TRAIL_WIDTH_ATR_MULT (default 1.5, was 1.0). Old settings
        fired trail exits at micro-profits (often ≤0.3%), crushing R:R.

        Activation:  price must be activation_mult × ATR in favor of position.
        Trail width: width_mult × ATR behind price.
        Ratchet:     only moves in the favorable direction — never widens.
        """
        if atr <= 0:
            return None

        # Lazy import avoids a circular dep at module load; Config singleton
        # cost is negligible vs per-bar price evaluation.
        from core.config import Config
        cfg = Config()
        activation_mult = cfg.TRAIL_ACTIVATION_ATR_MULT
        width_mult = cfg.TRAIL_WIDTH_ATR_MULT

        activation_distance = activation_mult * atr
        trail_distance = width_mult * atr

        if position.side == "long":
            if current_price < position.entry_price + activation_distance:
                return None  # not activated yet
            new_trail = current_price - trail_distance
            if position.trailing_stop is None or new_trail > position.trailing_stop:
                return new_trail
        else:  # short
            if current_price > position.entry_price - activation_distance:
                return None
            new_trail = current_price + trail_distance
            if position.trailing_stop is None or new_trail < position.trailing_stop:
                return new_trail

        return None  # trail wouldn't improve

    def is_trailing_stop_hit(
        self, position: Position, current_price: float
    ) -> bool:
        """True if the position's trailing stop has been breached."""
        if position.trailing_stop is None:
            return False
        if position.side == "long":
            return current_price <= position.trailing_stop
        return current_price >= position.trailing_stop  # short
