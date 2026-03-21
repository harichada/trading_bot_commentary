"""Opening Trend (First 15-Minute Direction) — intraday strategy.

The first 15 minutes of trading (9:30-9:45) are dominated by institutional order
flow. The direction established in this period tends to persist for the rest of
the session. This strategy measures the opening move, establishes a directional
bias, waits for a pullback, and enters in the direction of the trend.

Entry (long): trend_pct >= min_trend_pct, price > VWAP, price < day_high * (1 - pullback),
              EMA9 > EMA20.
Entry (short): trend_pct <= -min_trend_pct, price < VWAP, price > day_low * (1 + pullback),
               EMA9 < EMA20.
Stop: max(entry - atr * 1.5, or_low - buffer) for long; mirror for short.
Target: reward_ratio * risk from entry.
Active window: 9:50 AM to 2:30 PM ET.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .base import ExitSignal
from .intraday_base import IntradaySetup, IntradayStrategy
from .intraday_registry import IntradayStrategyRegistry

logger = logging.getLogger(__name__)


@IntradayStrategyRegistry.register('opening_trend')
class OpeningTrendStrategy(IntradayStrategy):
    """Opening Trend strategy — trade in the direction of the first 15-minute move.

    Measures institutional order flow direction during 9:30-9:45, waits for a
    pullback entry, and rides the trend with ATR-based stops and reward-ratio targets.
    """

    name = 'Opening Trend'
    description = 'First 15-minute directional bias with pullback entry'
    version = '1.0'
    strategy_id = 'opening_trend'

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        # Daily state — reset each day
        self._day_bias: Dict[str, str] = {}         # symbol -> 'long'/'short'
        self._trend_strength: Dict[str, float] = {}  # symbol -> trend_pct
        self._entered_symbols: Dict[str, int] = {}    # symbol -> entry count today
        self._or_prices: Dict[str, Dict] = {}         # symbol -> {open_930, close_945, or_high, or_low}

    # ── Scanning ──────────────────────────────────────────────────────

    def scan_for_setups(self, symbol: str, tick_data: Dict,
                        snapshot: Optional[Dict],
                        now: datetime) -> Optional[IntradaySetup]:
        """Scan for opening trend setups.

        Phase 1 (before 9:45): Gather opening range data, no signals.
        Phase 2 (9:45 onward): Establish bias from first 15-min direction.
        Phase 3 (9:50+): Look for pullback entries in the bias direction.
        """
        if not tick_data:
            return None

        # ── Phase 1: Before 9:45 — collect data, no signals ──────────
        if now.hour < 9 or (now.hour == 9 and now.minute < 45):
            return None

        # ── Phase 2: Establish day bias (once per symbol) ────────────
        if symbol not in self._day_bias:
            bias = self._establish_bias(symbol, tick_data)
            if bias is None:
                return None  # Not enough data yet

        # ── Phase 3: Look for pullback entry ─────────────────────────
        bias = self._day_bias.get(symbol)
        if bias is None:
            return None

        # Check max entries per day
        max_entries = self._config.get('max_entries_per_day', 2)
        if self._entered_symbols.get(symbol, 0) >= max_entries:
            return None

        # Get current price
        price = None
        if snapshot:
            price = snapshot.get('price') or snapshot.get('latestTrade', {}).get('p')
        if price is None:
            return None

        # Required indicators
        vwap = tick_data.get('vwap', 0)
        ema9 = tick_data.get('ema9', 0)
        ema20 = tick_data.get('ema20', 0)
        rsi = tick_data.get('rsi', 50)
        atr = tick_data.get('atr', 0)
        day_high = tick_data.get('day_high', 0)
        day_low = tick_data.get('day_low', float('inf'))
        vol_surge = tick_data.get('volume_surge_ratio', 0)

        # Must have initialized EMAs
        if ema9 <= 0 or ema20 <= 0:
            return None

        pullback_pct = self._config.get('pullback_pct', 0.003)

        # ── Check pullback entry conditions ──────────────────────────
        if bias == 'long':
            # EMA trend confirmation
            if ema9 <= ema20:
                return None

            # VWAP confirmation
            if vwap > 0 and price <= vwap:
                return None

            # Pullback check: price must be below day_high by at least pullback_pct
            if day_high <= 0:
                return None
            if price >= day_high * (1 - pullback_pct):
                return None  # No pullback yet — too close to high

            # Price should still be above VWAP (already checked) and above EMA9
            if price < ema9:
                return None  # Too deep a pullback — trend may be broken

        elif bias == 'short':
            # EMA trend confirmation
            if ema9 >= ema20:
                return None

            # VWAP confirmation
            if vwap > 0 and price >= vwap:
                return None

            # Pullback check: price must be above day_low by at least pullback_pct
            if day_low <= 0 or day_low == float('inf'):
                return None
            if price <= day_low * (1 + pullback_pct):
                return None  # No pullback yet — too close to low

            # Price should still be below VWAP and below EMA9
            if price > ema9:
                return None  # Too deep a pullback

        else:
            return None

        # ── Compute stop, target, R:R ────────────────────────────────
        stop_atr_mult = self._config.get('stop_atr_mult', 1.5)
        min_stop_pct = self._config.get('min_stop_pct', 0.006)
        reward_ratio = self._config.get('reward_ratio', 2.0)

        or_high = self._or_prices.get(symbol, {}).get('or_high', 0)
        or_low = self._or_prices.get(symbol, {}).get('or_low', 0)
        or_buffer = price * 0.001  # 0.1% buffer beyond OR

        if bias == 'long':
            atr_stop = price - (atr * stop_atr_mult) if atr > 0 else 0
            or_stop = (or_low - or_buffer) if or_low > 0 else 0

            # Use the wider (more protective) stop as base
            if atr_stop > 0 and or_stop > 0:
                stop_price = max(atr_stop, or_stop)
            elif atr_stop > 0:
                stop_price = atr_stop
            elif or_stop > 0:
                stop_price = or_stop
            else:
                return None  # Cannot compute stop

            # Enforce minimum stop distance
            min_stop_dist = price * min_stop_pct
            if price - stop_price < min_stop_dist:
                stop_price = price - min_stop_dist

            risk = price - stop_price
            if risk <= 0:
                return None
            target_price = price + (risk * reward_ratio)

        else:  # short
            atr_stop = price + (atr * stop_atr_mult) if atr > 0 else 0
            or_stop = (or_high + or_buffer) if or_high > 0 else 0

            if atr_stop > 0 and or_stop > 0:
                stop_price = min(atr_stop, or_stop)
            elif atr_stop > 0:
                stop_price = atr_stop
            elif or_stop > 0:
                stop_price = or_stop
            else:
                return None

            min_stop_dist = price * min_stop_pct
            if stop_price - price < min_stop_dist:
                stop_price = price + min_stop_dist

            risk = stop_price - price
            if risk <= 0:
                return None
            target_price = price - (risk * reward_ratio)

        rr = abs(target_price - price) / risk if risk > 0 else 0

        # ── Confidence scoring ───────────────────────────────────────
        trend_pct = self._trend_strength.get(symbol, 0)
        confidence = self._compute_confidence(
            tick_data, bias, trend_pct, vol_surge, rsi)

        return IntradaySetup(
            symbol=symbol,
            strategy_id=self.strategy_id,
            direction=bias,
            entry_price=price,
            stop_price=round(stop_price, 2),
            target_price=round(target_price, 2),
            risk_reward=round(rr, 2),
            confidence=round(confidence, 3),
            setup_type='opening_trend',
            indicators={
                'trend_pct': round(trend_pct, 4),
                'bias': bias,
                'ema9': round(ema9, 2),
                'ema20': round(ema20, 2),
                'rsi': round(rsi, 1),
                'volume_surge': round(vol_surge, 2),
                'vwap': round(vwap, 2) if vwap else None,
                'atr': round(atr, 2) if atr else None,
                'or_high': round(or_high, 2) if or_high else None,
                'or_low': round(or_low, 2) if or_low else None,
            },
            timestamp=now,
            notes=(f'Opening trend {bias}, move={trend_pct:.2%}, '
                   f'EMA9={ema9:.1f}/EMA20={ema20:.1f}'),
        )

    def _establish_bias(self, symbol: str, tick_data: Dict) -> Optional[str]:
        """Establish the day's directional bias from the first 15 minutes.

        Uses opening range data (or_high/or_low) if available, falls back to
        bar_history to compute the 9:30-9:45 price change.

        Returns 'long', 'short', or None if the move is too small.
        """
        min_trend_pct = self._config.get('min_trend_pct', 0.003)

        open_930 = None
        close_945 = None
        or_high = None
        or_low = None

        # Method 1: Use opening range data from tick indicators
        or_complete = tick_data.get('or_complete', False)
        if or_complete:
            or_high = tick_data.get('or_high')
            or_low = tick_data.get('or_low')
            or_open = tick_data.get('or_open')  # Some engines track this

            if or_open and or_high and or_low:
                open_930 = or_open
                # Infer close direction from OR midpoint relative to open
                or_mid = (or_high + or_low) / 2
                close_945 = or_mid  # Approximation; bar_history is more precise

        # Method 2: Fall back to bar_history for precise 9:30-9:45 price data
        # bar_history contains FiveMinBar dataclass objects with .bar_time, .open, .close, etc.
        bar_history = tick_data.get('bar_history', [])
        if bar_history and len(bar_history) >= 2:
            day_bars = []
            for bar in bar_history:
                bt = getattr(bar, 'bar_time', None) or (bar.get('bar_time') if isinstance(bar, dict) else None)
                if not bt:
                    continue
                # Parse bar_time 'H:MM' format
                try:
                    parts = bt.split(':')
                    bh, bm = int(parts[0]), int(parts[1])
                except (ValueError, IndexError):
                    continue
                # Accept bars from 9:30-9:40 (first 3 five-min bars: 9:30, 9:35, 9:40)
                bar_min = bh * 60 + bm
                if 570 <= bar_min <= 580:  # 9:30=570, 9:40=580
                    day_bars.append(bar)

            if day_bars:
                # Sort by bar_time
                def _bar_sort_key(b):
                    bt = getattr(b, 'bar_time', '') or (b.get('bar_time', '') if isinstance(b, dict) else '')
                    try:
                        p = bt.split(':')
                        return int(p[0]) * 60 + int(p[1])
                    except (ValueError, IndexError):
                        return 0
                day_bars.sort(key=_bar_sort_key)
                first = day_bars[0]
                last = day_bars[-1]
                open_930 = getattr(first, 'open', None) or (first.get('open') if isinstance(first, dict) else None)
                close_945 = getattr(last, 'close', None) or (last.get('close') if isinstance(last, dict) else None)

        # Method 3: Use OR high/low to infer direction if we have no open/close
        if open_930 is None and or_high and or_low:
            # Use day_high/day_low trend as proxy
            day_open = tick_data.get('day_open', 0)
            if day_open > 0:
                open_930 = day_open
                # Current price at 9:45 is the best we have
                close_945 = (or_high + or_low) / 2

        if open_930 is None or close_945 is None:
            return None
        if open_930 <= 0:
            return None

        trend_pct = (close_945 - open_930) / open_930

        # Store OR prices for stop calculation
        if or_high and or_low:
            self._or_prices[symbol] = {
                'open_930': open_930,
                'close_945': close_945,
                'or_high': or_high,
                'or_low': or_low,
            }
        else:
            self._or_prices[symbol] = {
                'open_930': open_930,
                'close_945': close_945,
                'or_high': max(open_930, close_945),
                'or_low': min(open_930, close_945),
            }

        # Check if move is strong enough
        if abs(trend_pct) < min_trend_pct:
            # Move too small — no bias today for this symbol
            self._day_bias[symbol] = None  # type: ignore[assignment]
            self._trend_strength[symbol] = trend_pct
            return None

        bias = 'long' if trend_pct > 0 else 'short'
        self._day_bias[symbol] = bias
        self._trend_strength[symbol] = trend_pct

        logger.info(
            f"Opening trend bias for {symbol}: {bias} "
            f"(trend={trend_pct:+.2%}, open=${open_930:.2f}, close=${close_945:.2f})"
        )

        return bias

    def _compute_confidence(self, tick_data: Dict, direction: str,
                            trend_pct: float, vol_surge: float,
                            rsi: float) -> float:
        """Compute confidence score (0.0 - 1.0).

        Base: 0.5
        +0.15 if trend_pct > 2x min_trend_pct (strong opening move)
        +0.10 if volume increasing (vol_surge > 1.5)
        +0.10 if RSI confirms direction (>55 for long, <45 for short)
        -0.15 if RSI diverges (>70 for long = overbought, <30 for short = oversold)
        """
        min_trend_pct = self._config.get('min_trend_pct', 0.003)
        score = 0.50

        # Strong opening move
        if abs(trend_pct) > 2 * min_trend_pct:
            score += 0.15

        # Volume confirmation
        if vol_surge >= 1.5:
            score += 0.10

        # RSI directional confirmation
        if direction == 'long' and rsi > 55:
            score += 0.10
        elif direction == 'short' and rsi < 45:
            score += 0.10

        # RSI divergence penalty
        if direction == 'long' and rsi > 70:
            score -= 0.15  # Overbought on a long bias
        elif direction == 'short' and rsi < 30:
            score -= 0.15  # Oversold on a short bias

        return max(0.0, min(score, 0.95))

    def get_watchlist_criteria(self) -> Dict:
        return {
            'min_volume': 500_000,
            'min_price': 10.0,
            'max_price': 500.0,
            'prefer_gappers': True,
        }

    # ── Entry ─────────────────────────────────────────────────────────

    def get_active_window(self) -> Tuple[int, int, int, int]:
        """Active from 9:50 AM to 2:30 PM ET."""
        return (9, 50, 14, 30)

    def record_entry(self, symbol: str):
        """Track that an entry was made for this symbol today."""
        self._entered_symbols[symbol] = self._entered_symbols.get(symbol, 0) + 1

    # ── Exit Management ───────────────────────────────────────────────

    def evaluate_exit(self, position: Dict, price: float,
                      tick_data: Optional[Dict] = None,
                      now: Optional[datetime] = None) -> Optional[ExitSignal]:
        """Exit if trend reversal detected: EMA cross against position direction."""
        if not tick_data:
            return None

        direction = position.get('direction', 'long')
        entry_price = position.get('entry_price', 0)
        ema9 = tick_data.get('ema9', 0)
        ema20 = tick_data.get('ema20', 0)

        if ema9 <= 0 or ema20 <= 0:
            return None

        # EMA crossover against position = trend reversal
        if direction == 'long' and ema9 < ema20:
            return ExitSignal(
                action='close',
                reason=(f'Opening trend reversal: EMA9 ${ema9:.2f} crossed below '
                        f'EMA20 ${ema20:.2f}'),
            )
        if direction == 'short' and ema9 > ema20:
            return ExitSignal(
                action='close',
                reason=(f'Opening trend reversal: EMA9 ${ema9:.2f} crossed above '
                        f'EMA20 ${ema20:.2f}'),
            )

        # Time-based exit: close after max_hold_minutes
        if now is not None and entry_price > 0:
            entry_time = position.get('entry_time')
            if entry_time is not None:
                if isinstance(entry_time, datetime):
                    hold_minutes = (now - entry_time).total_seconds() / 60
                    max_hold = self._config.get('max_hold_minutes', 120)
                    if hold_minutes >= max_hold:
                        return ExitSignal(
                            action='close',
                            reason=f'Max hold time reached ({max_hold} min)',
                        )

        return None

    def update_trailing_stop(self, position: Dict, price: float,
                             tick_data: Optional[Dict] = None,
                             now: Optional[datetime] = None) -> Optional[float]:
        """Trail stop using EMA20 after 1R profit.

        Uses EMA20 (slower) since this is a trend-following strategy
        that aims to hold for extended moves.
        """
        if not tick_data:
            return None

        direction = position.get('direction', 'long')
        entry_price = position.get('entry_price', 0)
        stop_price = position.get('stop_price', 0)
        if entry_price <= 0 or stop_price <= 0:
            return None

        # Calculate current P&L and initial risk
        if direction == 'long':
            initial_risk = entry_price - stop_price
            pnl = price - entry_price
        else:
            initial_risk = stop_price - entry_price
            pnl = entry_price - price

        if initial_risk <= 0:
            return None

        # Only start trailing after 1R profit
        r_multiple = pnl / initial_risk
        if r_multiple < 1.0:
            return None

        # Use EMA20 as trailing stop (slower than EMA9 for trend-following)
        ema20 = tick_data.get('ema20', 0)
        if ema20 <= 0:
            return None

        buffer_pct = 0.0015  # 0.15% buffer
        if direction == 'long':
            new_stop = ema20 * (1 - buffer_pct)
            # Only tighten, never widen
            if new_stop > stop_price:
                return round(new_stop, 2)
        else:
            new_stop = ema20 * (1 + buffer_pct)
            if new_stop < stop_price:
                return round(new_stop, 2)

        return None

    # ── Indicators ────────────────────────────────────────────────────

    def get_required_indicators(self) -> List[str]:
        return ['vwap', 'ema_multi', 'opening_range', 'day_high', 'day_low',
                'rsi', 'atr', 'bar_history']

    def get_ema_periods(self) -> List[int]:
        return [9, 20]

    # ── Config ────────────────────────────────────────────────────────

    def get_parameter_schema(self) -> Dict:
        return {
            'min_trend_pct': {
                'type': 'float', 'label': 'Min Trend %',
                'default': 0.003, 'min': 0.001, 'max': 0.01,
                'description': 'Minimum price change in first 15 min to establish bias (0.3%)'
            },
            'pullback_pct': {
                'type': 'float', 'label': 'Pullback %',
                'default': 0.003, 'min': 0.001, 'max': 0.01,
                'description': 'Required pullback from high/low before entry (0.3%)'
            },
            'stop_atr_mult': {
                'type': 'float', 'label': 'Stop ATR Multiple',
                'default': 1.5, 'min': 1.0, 'max': 3.0,
                'description': 'ATR multiplier for stop placement'
            },
            'min_stop_pct': {
                'type': 'float', 'label': 'Min Stop Distance %',
                'default': 0.006, 'min': 0.003, 'max': 0.02,
                'description': 'Minimum stop distance as % of entry price (0.6%)'
            },
            'reward_ratio': {
                'type': 'float', 'label': 'Reward Ratio',
                'default': 2.0, 'min': 1.5, 'max': 4.0,
                'description': 'Target profit as multiple of risk'
            },
            'max_hold_minutes': {
                'type': 'int', 'label': 'Max Hold Minutes',
                'default': 120, 'min': 30, 'max': 300,
                'description': 'Maximum time to hold a position (minutes)'
            },
            'max_entries_per_day': {
                'type': 'int', 'label': 'Max Entries Per Day',
                'default': 2, 'min': 1, 'max': 5,
                'description': 'Maximum entries per symbol per day'
            },
        }

    def get_default_config(self) -> Dict:
        return {
            'min_trend_pct': 0.003,
            'pullback_pct': 0.003,
            'stop_atr_mult': 1.5,
            'min_stop_pct': 0.006,
            'reward_ratio': 2.0,
            'max_hold_minutes': 120,
            'max_entries_per_day': 2,
            'min_risk_reward': 1.5,
            'min_confidence': 0.5,
        }

    # ── Lifecycle ─────────────────────────────────────────────────────

    def on_day_start(self):
        """Reset all daily state."""
        self._day_bias.clear()
        self._trend_strength.clear()
        self._entered_symbols.clear()
        self._or_prices.clear()

    def on_day_end(self):
        """Cleanup daily state."""
        self._day_bias.clear()
        self._trend_strength.clear()
        self._entered_symbols.clear()
        self._or_prices.clear()
