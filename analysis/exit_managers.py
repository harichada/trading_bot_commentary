import logging
from datetime import datetime
from typing import Dict, Tuple, Optional, Any

import numpy as np

from core.models import CommentaryType
from core.commentary import TradingCommentary

logger = logging.getLogger('TradingBot')


class AdvancedExitManager:
    """Advanced exit strategies with dynamic adjustments"""

    def __init__(self, commentary_system):
        self.commentary = commentary_system
        self.position_tracking = {}

    def initialize_trailing_stop(self, symbol: str, entry_price: float,
                                initial_stop: float, trailing_percent: float = 0.02):
        """Initialize trailing stop for a position"""
        self.position_tracking[symbol] = {
            'entry_price': entry_price,
            'initial_stop': initial_stop,
            'current_stop': initial_stop,
            'trailing_percent': trailing_percent,
            'highest_price': entry_price,
            'lowest_price': entry_price
        }

    def update_trailing_stop(self, symbol: str, current_price: float, side: str = 'long') -> float:
        """Update trailing stop based on current price"""
        if symbol not in self.position_tracking:
            return None

        tracking = self.position_tracking[symbol]

        if side == 'long':
            # Update highest price for long positions
            if current_price > tracking['highest_price']:
                tracking['highest_price'] = current_price
                new_stop = current_price * (1 - tracking['trailing_percent'])
                if new_stop > tracking['current_stop']:
                    tracking['current_stop'] = new_stop

                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.TECHNICAL,
                        symbol=symbol,
                        title="Trailing Stop Updated",
                        message=f"Trailing stop moved up to ${new_stop:.2f} (was ${tracking['current_stop']:.2f})",
                        data={'new_stop': new_stop, 'current_price': current_price},
                        importance=5
                    ))
        else:
            # Update lowest price for short positions
            if current_price < tracking['lowest_price']:
                tracking['lowest_price'] = current_price
                new_stop = current_price * (1 + tracking['trailing_percent'])
                if new_stop < tracking['current_stop']:
                    tracking['current_stop'] = new_stop

                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.TECHNICAL,
                        symbol=symbol,
                        title="Trailing Stop Updated",
                        message=f"Trailing stop moved down to ${new_stop:.2f} (was ${tracking['current_stop']:.2f})",
                        data={'new_stop': new_stop, 'current_price': current_price},
                        importance=5
                    ))

        return tracking['current_stop']

    def should_partial_exit(self, position, current_price: float,
                           indicators: Dict[str, float]) -> Tuple[bool, float]:
        """Determine if partial exit should be taken"""
        # Exit 50% if profit target is hit
        if position.side == 'long':
            profit_target = position.entry_price * 1.05  # 5% profit target
            if current_price >= profit_target:
                return True, 0.5  # Exit 50%
        else:
            profit_target = position.entry_price * 0.95  # 5% profit target
            if current_price <= profit_target:
                return True, 0.5  # Exit 50%

        # Exit 25% if RSI is overbought/oversold
        rsi = indicators.get('rsi', 50)
        if position.side == 'long' and rsi > 80:
            return True, 0.25  # Exit 25%
        elif position.side == 'short' and rsi < 20:
            return True, 0.25  # Exit 25%

        return False, 0.0

    def time_based_exit(self, position, max_hold_days: int = 5) -> bool:
        """Exit position based on time held"""
        days_held = (datetime.now() - position.entry_time).days
        return days_held >= max_hold_days

    def volatility_based_exit(self, position, current_volatility: float,
                             entry_volatility: float) -> bool:
        """Exit position based on volatility changes"""
        volatility_change = abs(current_volatility - entry_volatility) / entry_volatility
        return volatility_change > 0.5  # 50% change in volatility


class DynamicExitManager:
    """Manages exits like a human day trader - adapts based on price action"""

    def __init__(self, trading_brain, commentary_system):
        self.brain = trading_brain
        self.commentary = commentary_system
        self.exit_trackers: Dict[str, Dict[str, Any]] = {}

    def initialize_position_tracking(self, symbol: str, entry_price: float,
                                   initial_stop: float, initial_target: float):
        """Start tracking a position for dynamic exit management"""
        self.exit_trackers[symbol] = {
            'entry_price': entry_price,
            'current_stop': initial_stop,
            'current_target': initial_target,
            'highest_price': entry_price,
            'lowest_price': entry_price,
            'time_in_profit': 0,
            'time_in_loss': 0,
            'momentum_shifts': 0,
            'partial_exits': [],
            'trailing_activated': False,
            'scalp_target_hit': False,
            'last_check': datetime.now()
        }

    async def evaluate_exit(self, position, current_price: float,
                          indicators: Dict[str, float]) -> Tuple[bool, str, float]:
        """
        Evaluate if we should exit - returns (should_exit, reason, exit_portion)
        exit_portion: 1.0 for full exit, 0.5 for half, etc.
        """
        symbol = position.symbol
        tracker = self.exit_trackers.get(symbol, {})

        if not tracker:
            return False, "", 0

        # Update tracker
        tracker['highest_price'] = max(tracker['highest_price'], current_price)
        tracker['lowest_price'] = min(tracker['lowest_price'], current_price)

        # Time tracking
        time_elapsed = datetime.now() - tracker['last_check']
        if current_price > position.entry_price:
            tracker['time_in_profit'] += time_elapsed.total_seconds()
        else:
            tracker['time_in_loss'] += time_elapsed.total_seconds()
        tracker['last_check'] = datetime.now()

        pnl_percent = ((current_price - position.entry_price) / position.entry_price) * 100

        # 1. Quick Scalp Exit — DISABLED 2026-04-18
        # Was: exit 50% if up 0.5% in <5 min. Pre-empts 1R partial exit
        # and grabs tiny wins before the thesis can play out.
        # Replaced by: ScaleTrailManager 1R partial exit in engine.py.

        # Progressive profit taking — DISABLED 2026-04-18
        # Was: exit 50% at 1.5% profit (fixed %). Conflicts with ATR-
        # scaled 1R partial which fires at the right volatility-adjusted
        # level instead of a fixed percentage.

        # Extended target — keep this one (3%+ is a genuine outsized move)
        if pnl_percent > 3.0:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.DECISION,
                symbol=symbol,
                title=f"\U0001f3af Extended Target Hit",
                message=f"Full exit at {pnl_percent:.1f}% profit",
                importance=9
            ))
            return True, "take_profits_extended", 1.0

        # 2. Momentum Failure Exit — DISABLED 2026-04-18
        # Was: full exit at 0.3% if MACD/RSI diverge. Killed winners
        # at +$36 avg — the primary cause of the 1.11:1 win/loss ratio.
        # Replaced by: ATR trailing stop which protects profits without
        # requiring indicator confirmation.

        # 3. Time-Based Trailing Stop
        if pnl_percent > 1.0 and not tracker['trailing_activated']:
            new_stop = position.entry_price * 1.002  # Move stop to breakeven + 0.2%
            tracker['current_stop'] = new_stop
            tracker['trailing_activated'] = True

            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.RISK_ASSESSMENT,
                symbol=symbol,
                title=f"\U0001f6e1\ufe0f Protecting Profits",
                message=f"Moving stop to breakeven + 0.2%. Can't let a winner turn into a loser!",
                importance=7
            ))

        # 4. Dynamic Trailing Stop based on ATR
        if tracker['trailing_activated'] and pnl_percent > 1.5:
            atr = indicators.get('atr', current_price * 0.01)
            new_stop = current_price - (1.5 * atr)

            if new_stop > tracker['current_stop']:
                tracker['current_stop'] = new_stop
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.DECISION,
                    symbol=symbol,
                    title=f"\U0001f4c8 Trailing Stop Update",
                    message=f"Profit at {pnl_percent:.1f}%. Moving stop up to ${new_stop:.2f} to lock in more gains.",
                    importance=6
                ))

        # 5. Exhaustion Exit - When move is overextended
        if pnl_percent > 2.0:
            price_extension = ((current_price - tracker['lowest_price']) / tracker['lowest_price']) * 100
            if price_extension > 3.0 and indicators.get('rsi', 50) > 75:
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.OPPORTUNITY,
                    symbol=symbol,
                    title=f"\U0001f3af Exhaustion Top",
                    message=f"Up {pnl_percent:.1f}% and RSI screaming overbought. Taking profits here - pigs get slaughtered!",
                    confidence=0.9,
                    importance=9
                ))
                return True, "exhaustion", 1.0

        # 6. Time Decay Exit — DISABLED 2026-04-17
        # Was: close if flat (< ±0.3%) after 30 min.
        # Problem: meta-model trains with max_holding=300 min (5 hrs).
        # Closing at 30 min pre-empts the thesis before the setup can
        # resolve, corrupting sim P&L honesty. Hard SL/TP handle exits.
        # Re-enable only for scalping strategies where the thesis IS
        # "if it doesn't move in 30 min, it's wrong."
        #
        # position_age_minutes = (datetime.now() - position.entry_time).total_seconds() / 60
        # if position_age_minutes > 30 and abs(pnl_percent) < 0.3:
        #     return True, "time_decay", 1.0

        # 7. Check against dynamic stop
        if current_price <= tracker['current_stop']:
            return True, "trailing_stop", 1.0

        # 8. Human-like Patience Limit — DISABLED 2026-04-17
        # Was: randomly close after 15 min if >0.8% profit.
        # Problem: grabs small wins before TP, destroying the 2:1 R:R
        # that the meta-model was trained to expect. A +0.8% exit when
        # TP is at +1.5% ATR leaves half the expected return on the table.
        #
        # position_age_minutes = (datetime.now() - position.entry_time).total_seconds() / 60
        # if position_age_minutes > 15 and pnl_percent > 0.8:
        #     patience = self.brain.emotional_state['patience']
        #     if patience < 0.5 and np.random.random() > patience:
        #         return True, "impatience", 1.0

        return False, "", 0

        # Short position stop loss (price went up)
        if position.side == 'short' and current_price >= position.stop_loss:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.RISK_ASSESSMENT,
                symbol=position.symbol,
                title=f"\U0001f6d1 Short Stop Loss Hit",
                message=f"Stopped out of short at ${current_price:.2f}",
                importance=9
            ))
            return True, "stop_loss_short"
