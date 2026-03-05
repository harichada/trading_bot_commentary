"""Classic Gap Fade Strategy — Strategy #1.

Enhanced with profit protection: breakeven stop, trailing stop, and profit
drawdown exit — all computed from position data (no real-time indicators needed).
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .base import ExitSignal, GapFadeStrategy
from .registry import GapFadeStrategyRegistry

# Default config values
_DEFAULTS = {
    'breakeven_trigger_pct': 0.005,       # 0.5% profit -> move stop to breakeven
    'trail_trigger_pct': 0.01,            # 1.0% profit -> start trailing
    'trail_offset_pct': 0.004,            # trail 0.4% behind high water mark
    'profit_protect_trigger_pct': 0.015,  # 1.5% peak profit -> enable drawdown exit
    'profit_drawdown_pct': 0.50,          # exit if 50% of peak profit given back
}


@GapFadeStrategyRegistry.register('classic_gap_fade')
class ClassicGapFadeStrategy(GapFadeStrategy):
    """Original gap fade behavior with real-time profit protection.

    Entry, stop placement, and target logic are unchanged (engine defaults).
    Adds three layers of profit protection that run on every tick:

    1. Breakeven stop: moves stop to entry price once profit reaches threshold
    2. Trailing stop: trails behind high water mark after reaching trigger
    3. Profit drawdown exit: closes position if too much peak profit is given back

    No real-time indicators needed — uses only position tracking data.
    """

    name = 'Classic Gap Fade'
    description = (
        'Original strategy with profit protection: short gap-ups on low volume '
        'at market open (9:31 AM). Fixed % stop, midpoint partial cover, full '
        'target at previous close. Breakeven + trailing stop and profit drawdown '
        'protection on every tick.'
    )
    version = '1.1'

    def get_default_config(self) -> Dict:
        return dict(_DEFAULTS)

    # ── Entry & Targets: unchanged (engine defaults) ────────────────

    # filter_candidate -> (True, '')
    # score_candidate -> None
    # get_entry_window -> None (engine uses 9:31)
    # should_enter_now -> (True, '')
    # compute_stop_price -> None (engine uses fixed % or adaptive)
    # compute_targets -> None (engine uses midpoint / prev_close)
    # get_required_indicators -> [] (no indicator engine needed)

    # ── Profit Protection ───────────────────────────────────────────

    def update_trailing_stop(self, position: dict, price: float,
                             tick_data: Optional[dict] = None,
                             now: Optional[datetime] = None) -> Optional[float]:
        """Breakeven + trailing stop using position high water mark data.

        Phase 1 (breakeven): Once peak P&L >= breakeven_trigger_pct, move stop
        to entry price so the trade can't become a loser.

        Phase 2 (trailing): Once peak P&L >= trail_trigger_pct, trail stop at
        trail_offset_pct behind the high water price, locking in more profit
        as the trade moves in our favor.
        """
        direction = position.get('direction', 'short')
        entry_price = position.get('entry_price', 0)
        current_stop = position.get('stop_price', 0)
        high_water_pnl = position.get('high_water_pnl_pct', 0)
        high_water_price = position.get('high_water_price', 0)

        if entry_price <= 0:
            return None

        breakeven_trigger = self._config.get('breakeven_trigger_pct', 0.005)
        trail_trigger = self._config.get('trail_trigger_pct', 0.01)
        trail_offset = self._config.get('trail_offset_pct', 0.004)

        # Not enough profit yet for any protection
        if high_water_pnl < breakeven_trigger:
            return None

        # Phase 1: Breakeven — move stop to entry price
        if direction == 'short':
            # For shorts: tighter = lower stop price
            new_stop = min(entry_price, current_stop)
        else:
            # For longs: tighter = higher stop price
            new_stop = max(entry_price, current_stop)

        # Phase 2: Trailing — trail behind high water mark
        if high_water_pnl >= trail_trigger and high_water_price > 0:
            if direction == 'short':
                # For shorts, high_water_price is the lowest price (best P&L)
                # Trail stop just above it
                trail_stop = round(high_water_price * (1 + trail_offset), 2)
                if trail_stop < new_stop:
                    new_stop = trail_stop
            else:
                # For longs, high_water_price is the highest price (best P&L)
                # Trail stop just below it
                trail_stop = round(high_water_price * (1 - trail_offset), 2)
                if trail_stop > new_stop:
                    new_stop = trail_stop

        new_stop = round(new_stop, 2)

        # Only return if actually tighter than current stop
        if direction == 'short' and new_stop < current_stop:
            return new_stop
        elif direction == 'long' and new_stop > current_stop:
            return new_stop

        return None

    def evaluate_exit(self, position: dict, price: float, high: float,
                      tick_data: Optional[dict] = None,
                      now: Optional[datetime] = None) -> Optional[ExitSignal]:
        """Profit drawdown exit: close if too much peak profit is given back.

        Only activates after peak P&L exceeds profit_protect_trigger_pct.
        If position then gives back >= profit_drawdown_pct of peak profit, exit.

        Example with defaults: position peaked at +2.0% profit. If current
        profit drops to +1.0% (gave back 50%), close immediately.
        """
        high_water_pnl = position.get('high_water_pnl_pct', 0)
        entry_price = position.get('entry_price', 0)
        direction = position.get('direction', 'short')
        remaining = position.get('remaining_shares', 0)

        if entry_price <= 0 or remaining <= 0:
            return None

        protect_trigger = self._config.get('profit_protect_trigger_pct', 0.015)
        drawdown_pct = self._config.get('profit_drawdown_pct', 0.50)

        # Only activate when peak profit exceeds protection threshold
        if high_water_pnl < protect_trigger:
            return None

        # Compute current unrealized P&L %
        if direction == 'short':
            current_pnl_pct = (entry_price - price) / entry_price
        else:
            current_pnl_pct = (price - entry_price) / entry_price

        # How much of peak profit was given back?
        given_back = (high_water_pnl - current_pnl_pct) / high_water_pnl

        if given_back >= drawdown_pct:
            return ExitSignal(
                action='close',
                reason=(f'profit_drawdown (peak {high_water_pnl:.1%}, '
                        f'now {current_pnl_pct:.1%}, gave back {given_back:.0%})'),
                shares=remaining,
            )

        return None

    # ── UI Config ───────────────────────────────────────────────────

    def get_parameter_schema(self) -> Dict:
        return {
            'breakeven_trigger_pct': {
                'type': 'float', 'label': 'Breakeven Trigger %',
                'default': _DEFAULTS['breakeven_trigger_pct'],
                'min': 0.001, 'max': 0.05, 'step': 0.001,
                'description': 'Move stop to entry price (breakeven) after this % profit is reached',
            },
            'trail_trigger_pct': {
                'type': 'float', 'label': 'Trail Trigger %',
                'default': _DEFAULTS['trail_trigger_pct'],
                'min': 0.002, 'max': 0.10, 'step': 0.001,
                'description': 'Start trailing stop behind high water mark after this % profit',
            },
            'trail_offset_pct': {
                'type': 'float', 'label': 'Trail Offset %',
                'default': _DEFAULTS['trail_offset_pct'],
                'min': 0.001, 'max': 0.02, 'step': 0.001,
                'description': 'Distance behind high water mark for trailing stop',
            },
            'profit_protect_trigger_pct': {
                'type': 'float', 'label': 'Profit Protect Trigger %',
                'default': _DEFAULTS['profit_protect_trigger_pct'],
                'min': 0.005, 'max': 0.10, 'step': 0.001,
                'description': 'Peak profit % required before drawdown exit activates',
            },
            'profit_drawdown_pct': {
                'type': 'float', 'label': 'Max Drawdown from Peak %',
                'default': _DEFAULTS['profit_drawdown_pct'],
                'min': 0.20, 'max': 0.80, 'step': 0.05,
                'description': 'Exit if this fraction of peak profit is given back (0.50 = 50%)',
            },
        }
