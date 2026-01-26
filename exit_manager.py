"""
Exit Manager - Professional Exit Management System
Implements R-multiple scaling, ATR-based stops, time stops, and smart trailing

"Entries are about timing. Exits are about making money."

This is the MOST IMPORTANT module for profitability.
The best traders spend 80% of their time on exits, not entries.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class ExitReason(Enum):
    """Reasons for exiting a position"""
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT_1R = "take_profit_1r"
    TAKE_PROFIT_2R = "take_profit_2r"
    TAKE_PROFIT_3R = "take_profit_3r"
    TRAILING_STOP = "trailing_stop"
    BREAKEVEN_STOP = "breakeven_stop"
    TIME_STOP = "time_stop"
    END_OF_DAY = "end_of_day"
    SUPPORT_BREAK = "support_break"
    RESISTANCE_BREAK = "resistance_break"
    REGIME_CHANGE = "regime_change"
    MANUAL = "manual"
    CIRCUIT_BREAKER = "circuit_breaker"


class TrailMode(Enum):
    """Trailing stop modes"""
    NONE = "none"
    FIXED_ATR = "fixed_atr"        # Trail by ATR distance
    CHANDELIER = "chandelier"      # Highest high minus ATR
    PARABOLIC = "parabolic"        # Accelerating trail
    STRUCTURE = "structure"        # Trail to swing lows/highs


@dataclass
class ExitLevel:
    """A specific exit level with target and size"""
    r_multiple: float  # 1R, 2R, 3R
    price: float
    size_pct: float  # Percentage of position to exit (0.25, 0.50, etc)
    hit: bool = False
    hit_time: Optional[datetime] = None


@dataclass
class PositionState:
    """State tracking for an open position"""
    symbol: str
    direction: str  # 'long' or 'short'
    entry_price: float
    entry_time: datetime
    position_size: float
    original_size: float
    risk_per_share: float  # Entry - Stop = 1R

    # Stop levels
    initial_stop: float
    current_stop: float
    breakeven_stop: Optional[float]
    trailing_stop: Optional[float]

    # Take profit levels
    exit_levels: List[ExitLevel]

    # ATR at entry (for dynamic calculations)
    atr_at_entry: float

    # State
    highest_price: float  # For trailing (longs)
    lowest_price: float   # For trailing (shorts)
    bars_in_trade: int
    current_r: float  # Current unrealized R-multiple
    realized_r: float  # R from partial exits
    is_at_breakeven: bool
    trail_mode: TrailMode


@dataclass
class ExitSignal:
    """Signal to exit a position"""
    symbol: str
    exit_reason: ExitReason
    exit_price: float
    exit_size_pct: float  # What percentage to exit
    r_multiple: float  # R at exit
    message: str
    timestamp: datetime = field(default_factory=datetime.now)


class ProExitManager:
    """
    Professional exit management system.

    Philosophy:
    - Exits make the money, not entries
    - Scale out at R-multiples (50% at 1R, 25% at 2R, 25% trail)
    - Move to breakeven after 1R (free trade)
    - Use ATR-based stops that respect volatility
    - Time stops - cut losers that aren't moving
    - Trail with structure, not arbitrary percentages

    "Cut losers fast. Let winners run. Scale out to lock profits."
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

        # R-Multiple targets (default scale-out plan)
        self.scale_out_plan = self.config.get('scale_out_plan', [
            {'r': 1.0, 'size_pct': 0.50},  # Take 50% off at 1R
            {'r': 2.0, 'size_pct': 0.25},  # Take 25% off at 2R
            {'r': 3.0, 'size_pct': 0.25},  # Final 25% at 3R or trail
        ])

        # Stop loss settings
        self.initial_stop_atr_mult = self.config.get('initial_stop_atr_mult', 1.5)
        self.max_stop_pct = self.config.get('max_stop_pct', 0.03)  # 3% max stop
        self.min_stop_pct = self.config.get('min_stop_pct', 0.005)  # 0.5% min stop

        # Breakeven settings
        self.move_to_breakeven_at_r = self.config.get('move_to_breakeven_at_r', 1.0)
        self.breakeven_buffer_pct = self.config.get('breakeven_buffer_pct', 0.001)  # 0.1% buffer

        # Trailing stop settings
        self.start_trailing_at_r = self.config.get('start_trailing_at_r', 1.5)
        self.trail_atr_mult = self.config.get('trail_atr_mult', 2.0)
        self.default_trail_mode = TrailMode(self.config.get('trail_mode', 'chandelier'))

        # Time stop settings
        self.time_stop_bars = self.config.get('time_stop_bars', 20)  # Exit if flat after N bars
        self.time_stop_min_r = self.config.get('time_stop_min_r', 0.3)  # Must be at least 0.3R to stay

        # End of day settings
        self.close_before_eod_minutes = self.config.get('close_before_eod_minutes', 15)
        self.eod_close_hour = self.config.get('eod_close_hour', 15)  # 3:45 PM
        self.eod_close_minute = self.config.get('eod_close_minute', 45)

        # Position tracking
        self.positions: Dict[str, PositionState] = {}

    def open_position(self, symbol: str, direction: str, entry_price: float,
                      position_size: float, stop_loss: float,
                      atr: Optional[float] = None) -> PositionState:
        """
        Open a new position with proper exit management.

        Args:
            symbol: Trading symbol
            direction: 'long' or 'short'
            entry_price: Entry price
            position_size: Number of shares
            stop_loss: Initial stop loss price
            atr: ATR at entry (optional, for dynamic trailing)

        Returns:
            PositionState with exit levels
        """
        # Calculate risk per share (1R)
        if direction == 'long':
            risk_per_share = entry_price - stop_loss
        else:
            risk_per_share = stop_loss - entry_price

        if risk_per_share <= 0:
            logger.error(f"Invalid stop: risk_per_share <= 0 for {symbol}")
            risk_per_share = entry_price * 0.01  # Default 1%

        # Use ATR if provided, otherwise estimate
        if atr is None:
            atr = risk_per_share  # Assume stop is ~1 ATR

        # Calculate take profit levels
        exit_levels = []
        for plan in self.scale_out_plan:
            if direction == 'long':
                tp_price = entry_price + (risk_per_share * plan['r'])
            else:
                tp_price = entry_price - (risk_per_share * plan['r'])

            exit_levels.append(ExitLevel(
                r_multiple=plan['r'],
                price=tp_price,
                size_pct=plan['size_pct'],
                hit=False,
                hit_time=None
            ))

        # Create position state
        state = PositionState(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            entry_time=datetime.now(),
            position_size=position_size,
            original_size=position_size,
            risk_per_share=risk_per_share,
            initial_stop=stop_loss,
            current_stop=stop_loss,
            breakeven_stop=None,
            trailing_stop=None,
            exit_levels=exit_levels,
            atr_at_entry=atr,
            highest_price=entry_price,
            lowest_price=entry_price,
            bars_in_trade=0,
            current_r=0.0,
            realized_r=0.0,
            is_at_breakeven=False,
            trail_mode=TrailMode.NONE
        )

        self.positions[symbol] = state
        logger.info(f"Position opened: {symbol} {direction} @ {entry_price:.2f}, "
                   f"Stop: {stop_loss:.2f}, 1R: ${risk_per_share:.2f}")

        return state

    def update_position(self, symbol: str, current_price: float,
                        current_atr: Optional[float] = None,
                        ohlcv_data: Optional[pd.DataFrame] = None) -> List[ExitSignal]:
        """
        Update position and check for exit signals.

        Args:
            symbol: Trading symbol
            current_price: Current market price
            current_atr: Current ATR (for dynamic trailing)
            ohlcv_data: Optional OHLCV data for structure-based trailing

        Returns:
            List of ExitSignals (can be multiple for scale-outs)
        """
        if symbol not in self.positions:
            return []

        state = self.positions[symbol]
        signals = []

        # Update price tracking
        state.highest_price = max(state.highest_price, current_price)
        state.lowest_price = min(state.lowest_price, current_price)
        state.bars_in_trade += 1

        # Calculate current R
        if state.direction == 'long':
            state.current_r = (current_price - state.entry_price) / state.risk_per_share
        else:
            state.current_r = (state.entry_price - current_price) / state.risk_per_share

        # 1. Check stop loss
        stop_signal = self._check_stop_loss(state, current_price)
        if stop_signal:
            signals.append(stop_signal)
            return signals  # Stop hit = exit all

        # 2. Check take profit levels (scale out)
        tp_signals = self._check_take_profits(state, current_price)
        signals.extend(tp_signals)

        # 3. Update breakeven stop
        self._update_breakeven(state, current_price)

        # 4. Update trailing stop
        if current_atr:
            self._update_trailing_stop(state, current_price, current_atr, ohlcv_data)

        # 5. Check time stop
        time_signal = self._check_time_stop(state, current_price)
        if time_signal:
            signals.append(time_signal)

        # 6. Check end of day
        eod_signal = self._check_end_of_day(state, current_price)
        if eod_signal:
            signals.append(eod_signal)

        return signals

    def _check_stop_loss(self, state: PositionState, price: float) -> Optional[ExitSignal]:
        """Check if stop loss is hit"""
        stop_hit = False

        if state.direction == 'long':
            stop_hit = price <= state.current_stop
        else:
            stop_hit = price >= state.current_stop

        if stop_hit:
            # Determine stop type
            if state.is_at_breakeven:
                reason = ExitReason.BREAKEVEN_STOP
            elif state.trail_mode != TrailMode.NONE:
                reason = ExitReason.TRAILING_STOP
            else:
                reason = ExitReason.STOP_LOSS

            r_at_exit = state.current_r
            remaining_pct = state.position_size / state.original_size

            logger.info(f"STOP HIT: {state.symbol} @ {price:.2f} ({reason.value}), "
                       f"R: {r_at_exit:+.2f}, Exiting {remaining_pct:.0%}")

            return ExitSignal(
                symbol=state.symbol,
                exit_reason=reason,
                exit_price=price,
                exit_size_pct=1.0,  # Exit all remaining
                r_multiple=r_at_exit,
                message=f"Stop hit at ${price:.2f} ({reason.value}). Final R: {r_at_exit:+.2f}"
            )

        return None

    def _check_take_profits(self, state: PositionState, price: float) -> List[ExitSignal]:
        """Check if any take profit levels are hit"""
        signals = []

        for level in state.exit_levels:
            if level.hit:
                continue

            tp_hit = False
            if state.direction == 'long':
                tp_hit = price >= level.price
            else:
                tp_hit = price <= level.price

            if tp_hit:
                level.hit = True
                level.hit_time = datetime.now()

                # Calculate actual R at this price
                if state.direction == 'long':
                    actual_r = (price - state.entry_price) / state.risk_per_share
                else:
                    actual_r = (state.entry_price - price) / state.risk_per_share

                # Add to realized R
                realized_portion = level.size_pct * actual_r
                state.realized_r += realized_portion

                # Reduce position size
                exit_shares = state.original_size * level.size_pct
                state.position_size -= exit_shares

                reason = self._r_to_exit_reason(level.r_multiple)

                logger.info(f"TP HIT: {state.symbol} {level.r_multiple}R @ {price:.2f}, "
                           f"Exiting {level.size_pct:.0%}, Realized: +{realized_portion:.2f}R")

                signals.append(ExitSignal(
                    symbol=state.symbol,
                    exit_reason=reason,
                    exit_price=price,
                    exit_size_pct=level.size_pct,
                    r_multiple=actual_r,
                    message=f"Take profit {level.r_multiple}R hit! Scaling out {level.size_pct:.0%} @ ${price:.2f}"
                ))

        return signals

    def _r_to_exit_reason(self, r: float) -> ExitReason:
        """Convert R-multiple to exit reason"""
        if r <= 1.0:
            return ExitReason.TAKE_PROFIT_1R
        elif r <= 2.0:
            return ExitReason.TAKE_PROFIT_2R
        else:
            return ExitReason.TAKE_PROFIT_3R

    def _update_breakeven(self, state: PositionState, price: float):
        """Move stop to breakeven after target R is reached"""
        if state.is_at_breakeven:
            return

        if state.current_r >= self.move_to_breakeven_at_r:
            # Move to breakeven with small buffer
            if state.direction == 'long':
                be_price = state.entry_price * (1 + self.breakeven_buffer_pct)
            else:
                be_price = state.entry_price * (1 - self.breakeven_buffer_pct)

            state.breakeven_stop = be_price
            state.current_stop = be_price
            state.is_at_breakeven = True

            logger.info(f"BREAKEVEN: {state.symbol} stop moved to ${be_price:.2f} "
                       f"(was ${state.initial_stop:.2f})")

    def _update_trailing_stop(self, state: PositionState, price: float,
                               atr: float, ohlcv_data: Optional[pd.DataFrame]):
        """Update trailing stop based on mode"""
        if state.current_r < self.start_trailing_at_r:
            return

        # Activate trailing if not already
        if state.trail_mode == TrailMode.NONE:
            state.trail_mode = self.default_trail_mode
            logger.info(f"TRAILING ACTIVATED: {state.symbol} at {state.current_r:.2f}R, "
                       f"mode: {state.trail_mode.value}")

        # Calculate new trailing stop
        if state.trail_mode == TrailMode.CHANDELIER:
            new_trail = self._chandelier_stop(state, atr)
        elif state.trail_mode == TrailMode.FIXED_ATR:
            new_trail = self._fixed_atr_stop(state, price, atr)
        elif state.trail_mode == TrailMode.STRUCTURE:
            new_trail = self._structure_stop(state, ohlcv_data)
        else:
            new_trail = self._fixed_atr_stop(state, price, atr)

        if new_trail is None:
            return

        # Only tighten the stop, never widen
        if state.direction == 'long':
            if new_trail > state.current_stop:
                state.trailing_stop = new_trail
                state.current_stop = new_trail
                logger.debug(f"Trail updated: {state.symbol} stop -> ${new_trail:.2f}")
        else:
            if new_trail < state.current_stop:
                state.trailing_stop = new_trail
                state.current_stop = new_trail
                logger.debug(f"Trail updated: {state.symbol} stop -> ${new_trail:.2f}")

    def _chandelier_stop(self, state: PositionState, atr: float) -> float:
        """Chandelier exit: Highest high minus ATR multiplier"""
        if state.direction == 'long':
            return state.highest_price - (atr * self.trail_atr_mult)
        else:
            return state.lowest_price + (atr * self.trail_atr_mult)

    def _fixed_atr_stop(self, state: PositionState, price: float, atr: float) -> float:
        """Trail by fixed ATR distance from current price"""
        if state.direction == 'long':
            return price - (atr * self.trail_atr_mult)
        else:
            return price + (atr * self.trail_atr_mult)

    def _structure_stop(self, state: PositionState,
                        ohlcv_data: Optional[pd.DataFrame]) -> Optional[float]:
        """Trail to swing lows (longs) or swing highs (shorts)"""
        if ohlcv_data is None or len(ohlcv_data) < 10:
            return None

        lookback = 5

        if state.direction == 'long':
            # Find recent swing low
            recent_lows = ohlcv_data['low'].iloc[-lookback:].values
            swing_low = min(recent_lows)
            # Small buffer below swing
            return swing_low * 0.998
        else:
            # Find recent swing high
            recent_highs = ohlcv_data['high'].iloc[-lookback:].values
            swing_high = max(recent_highs)
            # Small buffer above swing
            return swing_high * 1.002

    def _check_time_stop(self, state: PositionState, price: float) -> Optional[ExitSignal]:
        """Exit if position isn't moving after N bars"""
        if state.bars_in_trade < self.time_stop_bars:
            return None

        # If we're not at least min_r after time_stop_bars, cut it
        if abs(state.current_r) < self.time_stop_min_r:
            logger.info(f"TIME STOP: {state.symbol} flat after {state.bars_in_trade} bars "
                       f"(R: {state.current_r:+.2f})")

            return ExitSignal(
                symbol=state.symbol,
                exit_reason=ExitReason.TIME_STOP,
                exit_price=price,
                exit_size_pct=1.0,
                r_multiple=state.current_r,
                message=f"Time stop: Position flat after {state.bars_in_trade} bars. "
                        f"Not moving = dead money. R: {state.current_r:+.2f}"
            )

        return None

    def _check_end_of_day(self, state: PositionState, price: float) -> Optional[ExitSignal]:
        """Close position before end of day"""
        now = datetime.now()
        eod_cutoff = now.replace(
            hour=self.eod_close_hour,
            minute=self.eod_close_minute,
            second=0,
            microsecond=0
        )

        if now >= eod_cutoff:
            logger.info(f"EOD EXIT: {state.symbol} closing before market close "
                       f"(R: {state.current_r:+.2f})")

            return ExitSignal(
                symbol=state.symbol,
                exit_reason=ExitReason.END_OF_DAY,
                exit_price=price,
                exit_size_pct=1.0,
                r_multiple=state.current_r,
                message=f"End of day exit. Closing {state.symbol} @ ${price:.2f}. "
                        f"Final R: {state.current_r:+.2f}"
            )

        return None

    def close_position(self, symbol: str, exit_reason: ExitReason,
                       exit_price: float) -> Optional[Dict]:
        """
        Fully close a position and return final stats.

        Returns:
            Dict with position summary or None if position not found
        """
        if symbol not in self.positions:
            return None

        state = self.positions[symbol]

        # Calculate final R
        if state.direction == 'long':
            final_r = (exit_price - state.entry_price) / state.risk_per_share
        else:
            final_r = (state.entry_price - exit_price) / state.risk_per_share

        # Account for partial exits
        remaining_pct = state.position_size / state.original_size
        total_r = state.realized_r + (final_r * remaining_pct)

        duration = datetime.now() - state.entry_time

        summary = {
            'symbol': symbol,
            'direction': state.direction,
            'entry_price': state.entry_price,
            'exit_price': exit_price,
            'exit_reason': exit_reason.value,
            'initial_stop': state.initial_stop,
            'risk_per_share': state.risk_per_share,
            'final_r': final_r,
            'total_r': total_r,  # Including partials
            'realized_r': state.realized_r,
            'highest_r': (state.highest_price - state.entry_price) / state.risk_per_share if state.direction == 'long' else (state.entry_price - state.lowest_price) / state.risk_per_share,
            'bars_in_trade': state.bars_in_trade,
            'duration_minutes': int(duration.total_seconds() / 60),
            'was_at_breakeven': state.is_at_breakeven,
            'trail_mode_used': state.trail_mode.value,
        }

        # Remove position
        del self.positions[symbol]

        logger.info(f"POSITION CLOSED: {symbol} {state.direction} "
                   f"Entry: ${state.entry_price:.2f} -> Exit: ${exit_price:.2f} "
                   f"Total R: {total_r:+.2f} ({exit_reason.value})")

        return summary

    def get_position_status(self, symbol: str) -> Optional[Dict]:
        """Get current status of a position"""
        if symbol not in self.positions:
            return None

        state = self.positions[symbol]

        # Find next unfilled exit level
        next_tp = None
        for level in state.exit_levels:
            if not level.hit:
                next_tp = level
                break

        return {
            'symbol': symbol,
            'direction': state.direction,
            'entry_price': f"${state.entry_price:.2f}",
            'current_stop': f"${state.current_stop:.2f}",
            'current_r': f"{state.current_r:+.2f}R",
            'realized_r': f"{state.realized_r:+.2f}R",
            'highest_r': f"{(state.highest_price - state.entry_price) / state.risk_per_share:.2f}R" if state.direction == 'long' else f"{(state.entry_price - state.lowest_price) / state.risk_per_share:.2f}R",
            'position_remaining': f"{state.position_size / state.original_size:.0%}",
            'next_target': f"${next_tp.price:.2f} ({next_tp.r_multiple}R)" if next_tp else "None",
            'bars_in_trade': state.bars_in_trade,
            'is_at_breakeven': state.is_at_breakeven,
            'trail_mode': state.trail_mode.value,
            'risk_per_share': f"${state.risk_per_share:.2f}",
        }

    def get_all_positions_summary(self) -> List[Dict]:
        """Get summary of all open positions"""
        return [self.get_position_status(symbol) for symbol in self.positions]

    def calculate_optimal_stop(self, entry_price: float, atr: float,
                                direction: str, key_level: Optional[float] = None) -> float:
        """
        Calculate optimal initial stop loss.

        Args:
            entry_price: Entry price
            atr: Average True Range
            direction: 'long' or 'short'
            key_level: Optional support/resistance level to place stop beyond

        Returns:
            Optimal stop price
        """
        # ATR-based stop
        atr_stop_distance = atr * self.initial_stop_atr_mult

        # Clamp to min/max percentages
        min_distance = entry_price * self.min_stop_pct
        max_distance = entry_price * self.max_stop_pct
        stop_distance = max(min_distance, min(atr_stop_distance, max_distance))

        if direction == 'long':
            atr_stop = entry_price - stop_distance

            # If key level provided, place stop below it
            if key_level and key_level < entry_price:
                structure_stop = key_level * 0.998  # Just below support
                # Use the tighter of the two (but not too tight)
                return max(atr_stop, structure_stop)

            return atr_stop
        else:
            atr_stop = entry_price + stop_distance

            if key_level and key_level > entry_price:
                structure_stop = key_level * 1.002  # Just above resistance
                return min(atr_stop, structure_stop)

            return atr_stop


# Singleton instance
_exit_manager: Optional[ProExitManager] = None

def get_exit_manager(config: Optional[Dict] = None) -> ProExitManager:
    """Get or create singleton ProExitManager instance"""
    global _exit_manager
    if _exit_manager is None:
        _exit_manager = ProExitManager(config)
    return _exit_manager
