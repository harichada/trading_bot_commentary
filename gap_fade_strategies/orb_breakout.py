"""Opening Range Breakout (ORB) — intraday strategy.

Calculates the opening range (first 15 minutes, 9:30-9:45) and watches for
price breakouts above OR_high (long) or below OR_low (short) with volume
confirmation.

Entry: after OR is complete, price breaks out with volume >= 1.5x 20-bar avg.
Stop: opposite side of OR + buffer.
Target: 1.5x risk from entry.
Active window: 9:46 AM to 11:30 AM.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .base import ExitSignal
from .intraday_base import IntradaySetup, IntradayStrategy
from .intraday_registry import IntradayStrategyRegistry

logger = logging.getLogger(__name__)


@IntradayStrategyRegistry.register('orb_breakout')
class ORBBreakoutStrategy(IntradayStrategy):
    """Opening Range Breakout strategy.

    Waits for the opening range to form (first 15 minutes), then enters on
    a breakout above/below with volume confirmation. Uses the opposite side
    of the opening range as a stop.
    """

    name = 'ORB Breakout'
    description = 'Opening Range Breakout with volume confirmation'
    version = '1.0'
    strategy_id = 'orb_breakout'

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        # Track daily state per symbol
        self._entered_symbols: Dict[str, int] = {}  # symbol -> entry count today
        self._or_data: Dict[str, Dict] = {}  # symbol -> {or_high, or_low, or_range_pct}

    # ── Scanning ──────────────────────────────────────────────────────

    def scan_for_setups(self, symbol: str, tick_data: Dict,
                        snapshot: Optional[Dict],
                        now: datetime) -> Optional[IntradaySetup]:
        """Scan for ORB breakout setups.

        Requires:
        - Opening range complete (or_complete = True)
        - Price above OR_high (long) or below OR_low (short)
        - Volume >= volume_confirm_ratio x 20-bar average
        - OR range within min/max bounds
        - Not exceeded max entries per session
        """
        if not tick_data:
            return None

        # Check OR is complete
        if not tick_data.get('or_complete', False):
            return None

        or_high = tick_data.get('or_high')
        or_low = tick_data.get('or_low')
        if or_high is None or or_low is None or or_high <= 0 or or_low <= 0:
            return None

        # Cache OR data for this symbol
        if symbol not in self._or_data:
            or_range = or_high - or_low
            or_mid = (or_high + or_low) / 2
            if or_mid <= 0:
                return None
            or_range_pct = or_range / or_mid
            self._or_data[symbol] = {
                'or_high': or_high,
                'or_low': or_low,
                'or_range': or_range,
                'or_range_pct': or_range_pct,
            }

        or_info = self._or_data[symbol]
        or_range_pct = or_info['or_range_pct']

        # Check OR range is within acceptable bounds
        min_range = self._config.get('min_or_range_pct', 0.005)
        max_range = self._config.get('max_or_range_pct', 0.03)
        if or_range_pct < min_range:
            return None  # OR too narrow — not enough range
        if or_range_pct > max_range:
            return None  # OR too wide — risk too large

        # Check max entries
        max_entries = self._config.get('max_entries_per_session', 3)
        entries_today = self._entered_symbols.get(symbol, 0)
        if entries_today >= max_entries:
            return None

        # Get current price from snapshot or tick_data
        price = None
        if snapshot:
            price = snapshot.get('price') or snapshot.get('latestTrade', {}).get('p')
        if price is None:
            # Try to infer from tick_data (current bar close)
            return None

        # Breakout buffer
        breakout_buffer = self._config.get('breakout_buffer_pct', 0.001)

        # Check for breakout
        direction = None
        if price > or_high * (1 + breakout_buffer):
            direction = 'long'
        elif price < or_low * (1 - breakout_buffer):
            direction = 'short'
        else:
            return None  # No breakout

        # Volume confirmation
        volume_confirm = self._config.get('volume_confirm_ratio', 1.5)
        vol_surge = tick_data.get('volume_surge_ratio', 0.0)
        if vol_surge < volume_confirm:
            return None  # Not enough volume

        # VWAP confirmation (optional)
        vwap = tick_data.get('vwap', 0.0)
        if vwap > 0:
            if direction == 'long' and price < vwap:
                return None  # Long breakout should be above VWAP
            if direction == 'short' and price > vwap:
                return None  # Short breakout should be below VWAP

        # Compute stop, target, R:R
        stop_buffer = self._config.get('stop_buffer_pct', 0.002)
        target_rr = self._config.get('target_rr', 1.5)

        if direction == 'long':
            stop_price = or_low * (1 - stop_buffer)
            risk = price - stop_price
            target_price = price + (risk * target_rr)
        else:
            stop_price = or_high * (1 + stop_buffer)
            risk = stop_price - price
            target_price = price - (risk * target_rr)

        if risk <= 0:
            return None

        rr = (abs(target_price - price)) / risk if risk > 0 else 0

        # Confidence scoring
        confidence = self._compute_confidence(tick_data, direction, vol_surge,
                                              or_range_pct, price, vwap)

        return IntradaySetup(
            symbol=symbol,
            strategy_id=self.strategy_id,
            direction=direction,
            entry_price=price,
            stop_price=round(stop_price, 2),
            target_price=round(target_price, 2),
            risk_reward=round(rr, 2),
            confidence=round(confidence, 3),
            setup_type='orb_breakout',
            indicators={
                'or_high': or_high,
                'or_low': or_low,
                'or_range_pct': round(or_range_pct, 4),
                'volume_surge': round(vol_surge, 2),
                'vwap': round(vwap, 2) if vwap else None,
                'rsi': tick_data.get('rsi'),
                'atr': tick_data.get('atr'),
            },
            timestamp=now,
            notes=f'OR ${or_low:.2f}-${or_high:.2f}, vol surge {vol_surge:.1f}x',
        )

    def _compute_confidence(self, tick_data: Dict, direction: str,
                            vol_surge: float, or_range_pct: float,
                            price: float, vwap: float) -> float:
        """Compute confidence score for the setup (0.0 - 1.0)."""
        score = 0.50  # base

        # Volume strength (higher = more confident)
        if vol_surge >= 2.5:
            score += 0.15
        elif vol_surge >= 2.0:
            score += 0.10
        elif vol_surge >= 1.5:
            score += 0.05

        # VWAP alignment
        if vwap > 0:
            if direction == 'long' and price > vwap:
                score += 0.10
            elif direction == 'short' and price < vwap:
                score += 0.10

        # RSI not extreme (not overbought for long, not oversold for short)
        rsi = tick_data.get('rsi', 50)
        if direction == 'long' and 40 <= rsi <= 70:
            score += 0.05
        elif direction == 'short' and 30 <= rsi <= 60:
            score += 0.05

        # OR range quality (not too tight, not too wide)
        if 0.008 <= or_range_pct <= 0.02:
            score += 0.10

        # EMA alignment
        ema9 = tick_data.get('ema9', 0)
        ema20 = tick_data.get('ema20', 0)
        if ema9 > 0 and ema20 > 0:
            if direction == 'long' and ema9 > ema20:
                score += 0.05
            elif direction == 'short' and ema9 < ema20:
                score += 0.05

        return min(score, 0.95)

    def get_watchlist_criteria(self) -> Dict:
        return {
            'min_volume': 500_000,
            'min_price': 10.0,
            'max_price': 500.0,
            'min_adr_pct': 0.02,
            'prefer_gappers': True,
        }

    # ── Entry ─────────────────────────────────────────────────────────

    def get_active_window(self) -> Tuple[int, int, int, int]:
        """ORB active from 9:46 AM to 11:30 AM."""
        return (9, 46, 11, 30)

    def record_entry(self, symbol: str):
        """Track that an entry was made for this symbol today."""
        self._entered_symbols[symbol] = self._entered_symbols.get(symbol, 0) + 1

    # ── Exit Management ───────────────────────────────────────────────

    def evaluate_exit(self, position: Dict, price: float,
                      tick_data: Optional[Dict] = None,
                      now: Optional[datetime] = None) -> Optional[ExitSignal]:
        """ORB-specific exit: close if price falls back inside the OR."""
        if not tick_data:
            return None

        or_high = tick_data.get('or_high')
        or_low = tick_data.get('or_low')
        if or_high is None or or_low is None:
            return None

        direction = position.get('direction', 'long')
        entry_price = position.get('entry_price', 0)

        # If long and price drops back below OR midpoint — failed breakout
        or_mid = (or_high + or_low) / 2
        if direction == 'long' and price < or_mid:
            return ExitSignal(
                action='close',
                reason=f'ORB failed breakout: price ${price:.2f} back below OR midpoint ${or_mid:.2f}',
            )

        # If short and price rallies back above OR midpoint — failed breakdown
        if direction == 'short' and price > or_mid:
            return ExitSignal(
                action='close',
                reason=f'ORB failed breakdown: price ${price:.2f} back above OR midpoint ${or_mid:.2f}',
            )

        return None

    def update_trailing_stop(self, position: Dict, price: float,
                             tick_data: Optional[Dict] = None,
                             now: Optional[datetime] = None) -> Optional[float]:
        """Trail stop using EMA9 after 1R profit."""
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

        # Use EMA9 as trailing stop
        ema9 = tick_data.get('ema9', 0)
        if ema9 <= 0:
            return None

        buffer_pct = 0.001  # 0.1% buffer
        if direction == 'long':
            new_stop = ema9 * (1 - buffer_pct)
            # Only tighten, never widen
            if new_stop > stop_price:
                return round(new_stop, 2)
        else:
            new_stop = ema9 * (1 + buffer_pct)
            if new_stop < stop_price:
                return round(new_stop, 2)

        return None

    # ── Indicators ────────────────────────────────────────────────────

    def get_required_indicators(self) -> List[str]:
        return ['opening_range', 'vwap', 'volume_profile', 'ema', 'ema_multi',
                'rsi', 'day_high', 'day_low', 'bar_history']

    def get_ema_periods(self) -> List[int]:
        return [9, 20]

    # ── Config ────────────────────────────────────────────────────────

    def get_parameter_schema(self) -> Dict:
        return {
            'or_minutes': {
                'type': 'int', 'label': 'Opening Range Minutes',
                'default': 15, 'min': 5, 'max': 30,
                'description': 'Minutes after open to define opening range'
            },
            'breakout_buffer_pct': {
                'type': 'float', 'label': 'Breakout Buffer %',
                'default': 0.001, 'min': 0.0, 'max': 0.01,
                'description': 'Price must exceed OR by this % to trigger'
            },
            'volume_confirm_ratio': {
                'type': 'float', 'label': 'Volume Confirmation Ratio',
                'default': 1.5, 'min': 1.0, 'max': 5.0,
                'description': 'Current bar volume must be this multiple of 20-bar avg'
            },
            'min_or_range_pct': {
                'type': 'float', 'label': 'Min OR Range %',
                'default': 0.005, 'min': 0.001, 'max': 0.02,
                'description': 'Minimum opening range width to consider'
            },
            'max_or_range_pct': {
                'type': 'float', 'label': 'Max OR Range %',
                'default': 0.03, 'min': 0.01, 'max': 0.10,
                'description': 'Maximum opening range width (too wide = too much risk)'
            },
            'stop_buffer_pct': {
                'type': 'float', 'label': 'Stop Buffer %',
                'default': 0.002, 'min': 0.0, 'max': 0.01,
                'description': 'Buffer added beyond opposite side of OR for stop'
            },
            'target_rr': {
                'type': 'float', 'label': 'Target R:R Multiple',
                'default': 1.5, 'min': 1.0, 'max': 3.0,
                'description': 'Target profit as multiple of risk'
            },
            'max_entries_per_session': {
                'type': 'int', 'label': 'Max Entries Per Session',
                'default': 3, 'min': 1, 'max': 10,
                'description': 'Maximum ORB entries per trading day'
            },
        }

    def get_default_config(self) -> Dict:
        return {
            'or_minutes': 15,
            'breakout_buffer_pct': 0.001,
            'volume_confirm_ratio': 1.5,
            'min_or_range_pct': 0.005,
            'max_or_range_pct': 0.03,
            'stop_buffer_pct': 0.002,
            'target_rr': 1.5,
            'max_entries_per_session': 3,
            'min_risk_reward': 1.5,
            'min_confidence': 0.5,
        }

    # ── Lifecycle ─────────────────────────────────────────────────────

    def on_day_start(self):
        """Reset daily tracking."""
        self._entered_symbols.clear()
        self._or_data.clear()

    def on_day_end(self):
        """Cleanup."""
        self._entered_symbols.clear()
        self._or_data.clear()
