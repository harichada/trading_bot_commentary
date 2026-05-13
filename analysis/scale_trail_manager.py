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

    def update_peak_and_breakeven(
        self,
        position: Position,
        current_price: float,
        activation_r: float,
    ) -> Optional[float]:
        """v-breakeven-stop-2026-04-28: track high-water mark in R-multiples
        and lift stop to entry once peak crosses activation_r.

        Returns the new stop level (= entry_price) on the FIRST tick the
        threshold is crossed, else None. Subsequent calls return None
        because position.breakeven_lifted is set on first lift.

        The fix for the "winner-turned-loser" pattern: PLTR long entered
        $141.69, went briefly to +0.3% favor, then reversed to -0.94%. With
        breakeven stop armed at +0.5R, that trade exits at $141.69 (flat)
        instead of $140.34 (-$125). Across many trades this compounds.
        """
        if activation_r <= 0:
            return None  # disabled

        original_stop = position.original_stop or position.stop_loss
        stop_distance = abs(position.entry_price - original_stop)
        if stop_distance <= 0:
            return None

        # Compute current R (positive = in favor)
        if position.side == "long":
            current_r = (current_price - position.entry_price) / stop_distance
        else:
            current_r = (position.entry_price - current_price) / stop_distance

        # Track peak for diagnostics & for any future logic that needs it
        if current_r > position.peak_favorable_r:
            position.peak_favorable_r = current_r

        if position.breakeven_lifted:
            return None  # already moved once

        if position.peak_favorable_r < activation_r:
            return None  # not far enough in favor yet

        # Cross the threshold — lift stop to entry. Use exactly entry; no
        # buffer. A buffer would create slippage that could exit a trade
        # that hasn't actually reversed.
        return position.entry_price

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

        Triggers (any one), AFTER pnl crosses the strategy-specific threshold:
          T1: macd flip against position
          T2: rsi crossed back through 50 against us
          T3: adx falling AND below 20 (trend disintegrating)

        Per-strategy threshold (v-proactive-strategy-thresh-2026-04-30):
          news trades:   require pnl_r <= -0.70  (give it more room)
          other:         require pnl_r <= -0.50  (legacy default)
        Whipsaw analysis 2026-04-30 showed news-strategy trades with 100%
        whipsaw rate on RSI-below-50 and MACD-flip-bullish proactive
        exits — they were just barely past -0.5R when the indicator
        flipped, then fully recovered. Pulling the trigger back to -0.7R
        for news (closer to the genuine stop) cuts whipsaws by giving
        the trade more room to absorb noise.
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

        # Per-strategy threshold. News trades get -0.7 instead of -0.5
        # because the whipsaw analysis showed they were the dominant
        # whipsaw category with 100% recovery rates on -0.5 to -0.6 exits.
        _strategy = (getattr(position, 'reasoning', {}) or {}).get('strategy', '')
        _is_news = 'news' in _strategy.lower() or _strategy == 'free_news_sentiment'
        _threshold = -0.70 if _is_news else -0.50
        if pnl_r > _threshold:
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
