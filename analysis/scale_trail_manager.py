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

    def check_proactive_exit(
        self,
        position: Position,
        current_price: float,
        indicators: dict,
    ) -> Optional[str]:
        """v-proactive-exit-2026-04-27: bail when the thesis is deteriorating
        and we're already partway to the stop. Avoids riding losers to the
        full -1.5×ATR stop when momentum has already reversed against us.

        Returns a short reason string if proactive exit is warranted, else None.
        Caller passes ATR-period indicators (macd, macd_signal, rsi, adx).

        Triggers (any one):
          T1: pnl <= -0.5R AND macd flip against position
          T2: pnl <= -0.5R AND rsi crossed back through 50 against us
          T3: pnl <= -0.5R AND adx falling AND below 20 (trend disintegrating)

        -0.5R is the halfway point to the entry stop. Earlier than that,
        normal noise can pull positions to -0.3R; later than that we're
        close enough to the stop to just let it hit. -0.5R is "thesis is
        struggling, get a second opinion from the indicators."
        """
        original_stop = position.original_stop or position.stop_loss
        stop_distance = abs(position.entry_price - original_stop)
        if stop_distance <= 0:
            return None

        # Compute pnl in R-multiples (negative when adverse).
        if position.side == "long":
            pnl_r = (current_price - position.entry_price) / stop_distance
        else:  # short
            pnl_r = (position.entry_price - current_price) / stop_distance

        # Only consider proactive exit when at least halfway to stop.
        if pnl_r > -0.5:
            return None

        macd = float(indicators.get("macd", 0) or 0)
        macd_sig = float(indicators.get("macd_signal", 0) or 0)
        rsi = float(indicators.get("rsi", 50) or 50)
        adx = float(indicators.get("adx", 0) or 0)
        adx_prev = float(indicators.get("adx_prev", adx) or adx)

        if position.side == "long":
            # T1: MACD bearish for a long
            if macd < macd_sig:
                return "macd_flipped_bearish"
            # T2: RSI sliced below 50 — momentum left the building
            if rsi < 50:
                return "rsi_below_50"
        else:  # short
            # T1: MACD bullish for a short
            if macd > macd_sig:
                return "macd_flipped_bullish"
            # T2: RSI crossed above 50 against us
            if rsi > 50:
                return "rsi_above_50"

        # T3: ADX collapsing (trend the position relied on is dying)
        if adx < 20 and adx < adx_prev:
            return "adx_collapsing"

        return None
