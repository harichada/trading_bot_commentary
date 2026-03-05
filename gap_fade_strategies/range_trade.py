"""Range Trade — intraday strategy.

Trades bounces off support/resistance levels within an established range.

Entry (long at support): defined range visible, price at lower boundary,
              RSI < 40, declining volume, bounce candle.
Entry (short at resistance): price at upper boundary, RSI > 60, bounce.
Stop: 0.5% beyond range boundary.
Target: opposite side of range.
Active window: 10:30 AM to 3:00 PM ET (ranges establish after morning action).
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .base import ExitSignal
from .intraday_base import IntradaySetup, IntradayStrategy
from .intraday_registry import IntradayStrategyRegistry

logger = logging.getLogger(__name__)


@IntradayStrategyRegistry.register('range_trade')
class RangeTradeStrategy(IntradayStrategy):
    """Range trading strategy — buy support, sell resistance.

    Identifies intraday ranges using day high/low + VWAP as midpoint,
    then enters at boundaries with RSI/volume confirmation.
    """

    name = 'Range Trade'
    description = 'Support/resistance range bounces'
    version = '1.0'
    strategy_id = 'range_trade'

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self._entered_symbols: Dict[str, int] = {}

    # ── Scanning ──────────────────────────────────────────────────────

    def scan_for_setups(self, symbol: str, tick_data: Dict,
                        snapshot: Optional[Dict],
                        now: datetime) -> Optional[IntradaySetup]:
        """Scan for range trade setups.

        Requires established range (day_high - day_low > min_range_pct)
        and price near a boundary.
        """
        if not tick_data:
            return None

        price = None
        if snapshot:
            price = snapshot.get('price') or snapshot.get('latestTrade', {}).get('p')
        if price is None:
            return None

        max_entries = self._config.get('max_entries_per_session', 3)
        if self._entered_symbols.get(symbol, 0) >= max_entries:
            return None

        day_high = tick_data.get('day_high', 0)
        day_low = tick_data.get('day_low', float('inf'))
        rsi = tick_data.get('rsi', 50)
        rsi_initialized = tick_data.get('rsi_initialized', False)
        vwap = tick_data.get('vwap', 0)
        vol_surge = tick_data.get('volume_surge_ratio', 1.0)
        ema9 = tick_data.get('ema9', 0)
        atr = tick_data.get('atr', 0)

        if day_high <= 0 or day_low <= 0 or day_low >= day_high:
            return None
        if not rsi_initialized:
            return None

        # Check range is established (minimum width)
        range_mid = (day_high + day_low) / 2
        if range_mid <= 0:
            return None
        range_pct = (day_high - day_low) / range_mid
        min_range = self._config.get('min_range_pct', 0.005)
        max_range = self._config.get('max_range_pct', 0.04)
        if range_pct < min_range or range_pct > max_range:
            return None

        # Need enough bars to confirm range (at least 6 five-min bars = 30 min)
        bar_count = tick_data.get('bar_count', 0)
        min_bars = self._config.get('min_bars_for_range', 6)
        if bar_count < min_bars:
            return None

        # Check price proximity to boundaries
        boundary_pct = self._config.get('boundary_proximity_pct', 0.003)
        near_support = (price - day_low) / day_low < boundary_pct if day_low > 0 else False
        near_resistance = (day_high - price) / price < boundary_pct if price > 0 else False

        direction = None
        rsi_support_max = self._config.get('rsi_support_max', 40)
        rsi_resistance_min = self._config.get('rsi_resistance_min', 60)
        max_vol = self._config.get('max_volume_surge', 1.5)

        # ── LONG at support ──
        if near_support and rsi < rsi_support_max and vol_surge < max_vol:
            # Bounce confirmation: price above EMA9
            if ema9 > 0 and price > ema9:
                direction = 'long'

        # ── SHORT at resistance ──
        elif near_resistance and rsi > rsi_resistance_min and vol_surge < max_vol:
            if ema9 > 0 and price < ema9:
                direction = 'short'

        if direction is None:
            return None

        # Compute stop, target
        stop_buffer_pct = self._config.get('stop_buffer_pct', 0.005)

        if direction == 'long':
            stop_price = day_low * (1 - stop_buffer_pct)
            target_price = day_high * (1 - 0.001)  # Just inside resistance
            risk = price - stop_price
        else:
            stop_price = day_high * (1 + stop_buffer_pct)
            target_price = day_low * (1 + 0.001)  # Just inside support
            risk = stop_price - price

        if risk <= 0:
            return None

        rr = abs(target_price - price) / risk if risk > 0 else 0

        # Confidence
        confidence = self._compute_confidence(
            tick_data, direction, vol_surge, rsi, range_pct, price, vwap, range_mid)

        return IntradaySetup(
            symbol=symbol,
            strategy_id=self.strategy_id,
            direction=direction,
            entry_price=price,
            stop_price=round(stop_price, 2),
            target_price=round(target_price, 2),
            risk_reward=round(rr, 2),
            confidence=round(confidence, 3),
            setup_type='range_trade',
            indicators={
                'day_high': round(day_high, 2), 'day_low': round(day_low, 2),
                'range_pct': round(range_pct, 4),
                'rsi': round(rsi, 1), 'vwap': round(vwap, 2) if vwap else None,
                'volume_surge': round(vol_surge, 2),
            },
            timestamp=now,
            notes=f'Range ${day_low:.2f}-${day_high:.2f} ({range_pct:.1%}), '
                  f'{"support" if direction == "long" else "resistance"} bounce',
        )

    def _compute_confidence(self, tick_data, direction, vol_surge, rsi,
                            range_pct, price, vwap, range_mid):
        score = 0.50
        # Volume declining (less selling/buying at boundary)
        if vol_surge < 0.8:
            score += 0.10
        elif vol_surge < 1.2:
            score += 0.05
        # RSI extreme (stronger at boundaries)
        if direction == 'long' and rsi < 30:
            score += 0.10
        elif direction == 'short' and rsi > 70:
            score += 0.10
        # Range quality (wider = more tradeable)
        if 0.01 <= range_pct <= 0.03:
            score += 0.10
        # VWAP as midpoint confirmation
        if vwap > 0 and range_mid > 0:
            vwap_vs_mid = abs(vwap - range_mid) / range_mid
            if vwap_vs_mid < 0.005:
                score += 0.10
        return min(score, 0.95)

    def get_watchlist_criteria(self) -> Dict:
        return {
            'min_volume': 500_000,
            'min_price': 10.0,
            'max_price': 300.0,
            'min_adr_pct': 0.01,
            'prefer_gappers': False,
        }

    def get_active_window(self) -> Tuple[int, int, int, int]:
        return (10, 30, 15, 0)

    def record_entry(self, symbol: str):
        self._entered_symbols[symbol] = self._entered_symbols.get(symbol, 0) + 1

    # ── Exit ──────────────────────────────────────────────────────────

    def evaluate_exit(self, position: Dict, price: float,
                      tick_data: Optional[Dict] = None,
                      now: Optional[datetime] = None) -> Optional[ExitSignal]:
        """Exit if range breaks (price moves beyond range + buffer)."""
        if not tick_data:
            return None
        direction = position.get('direction', 'long')
        day_high = tick_data.get('day_high', 0)
        day_low = tick_data.get('day_low', float('inf'))
        if day_high <= 0 or day_low <= 0:
            return None

        break_pct = 0.008  # Range break threshold
        if direction == 'long' and price < day_low * (1 - break_pct):
            return ExitSignal(action='close',
                              reason=f'Range support broken: ${price:.2f} < ${day_low:.2f}')
        if direction == 'short' and price > day_high * (1 + break_pct):
            return ExitSignal(action='close',
                              reason=f'Range resistance broken: ${price:.2f} > ${day_high:.2f}')
        return None

    def update_trailing_stop(self, position: Dict, price: float,
                             tick_data: Optional[Dict] = None,
                             now: Optional[datetime] = None) -> Optional[float]:
        """Move stop to midpoint after 50% of range captured."""
        if not tick_data:
            return None
        direction = position.get('direction', 'long')
        entry_price = position.get('entry_price', 0)
        stop_price = position.get('stop_price', 0)
        day_high = tick_data.get('day_high', 0)
        day_low = tick_data.get('day_low', float('inf'))
        if entry_price <= 0 or stop_price <= 0 or day_high <= 0 or day_low <= 0:
            return None

        range_mid = (day_high + day_low) / 2

        if direction == 'long':
            # If price has moved past midpoint, tighten stop to entry
            if price > range_mid:
                new_stop = max(entry_price, range_mid * 0.998)
                if new_stop > stop_price:
                    return round(new_stop, 2)
        else:
            if price < range_mid:
                new_stop = min(entry_price, range_mid * 1.002)
                if new_stop < stop_price:
                    return round(new_stop, 2)
        return None

    # ── Indicators ────────────────────────────────────────────────────

    def get_required_indicators(self) -> List[str]:
        return ['vwap', 'ema', 'rsi', 'volume_profile',
                'day_high', 'day_low', 'bar_history']

    def get_ema_periods(self) -> List[int]:
        return [9]

    # ── Config ────────────────────────────────────────────────────────

    def get_parameter_schema(self) -> Dict:
        return {
            'min_range_pct': {
                'type': 'float', 'label': 'Min Range %',
                'default': 0.005, 'min': 0.003, 'max': 0.02,
            },
            'max_range_pct': {
                'type': 'float', 'label': 'Max Range %',
                'default': 0.04, 'min': 0.02, 'max': 0.10,
            },
            'boundary_proximity_pct': {
                'type': 'float', 'label': 'Boundary Proximity %',
                'default': 0.003, 'min': 0.001, 'max': 0.01,
            },
            'stop_buffer_pct': {
                'type': 'float', 'label': 'Stop Buffer %',
                'default': 0.005, 'min': 0.002, 'max': 0.015,
            },
            'rsi_support_max': {
                'type': 'float', 'label': 'RSI Max at Support',
                'default': 40, 'min': 25, 'max': 50,
            },
            'rsi_resistance_min': {
                'type': 'float', 'label': 'RSI Min at Resistance',
                'default': 60, 'min': 50, 'max': 75,
            },
            'max_entries_per_session': {
                'type': 'int', 'label': 'Max Entries', 'default': 3,
                'min': 1, 'max': 10,
            },
        }

    def get_default_config(self) -> Dict:
        return {
            'min_range_pct': 0.005,
            'max_range_pct': 0.04,
            'boundary_proximity_pct': 0.003,
            'stop_buffer_pct': 0.005,
            'rsi_support_max': 40,
            'rsi_resistance_min': 60,
            'max_volume_surge': 1.5,
            'min_bars_for_range': 6,
            'max_entries_per_session': 3,
            'min_risk_reward': 1.0,
            'min_confidence': 0.5,
        }

    def on_day_start(self):
        self._entered_symbols.clear()

    def on_day_end(self):
        self._entered_symbols.clear()
