"""Pullback Entry — intraday strategy.

Enters trend-following pullbacks when a stock in an uptrend pulls back to
support (VWAP or EMA20) with declining volume, then bounces.

Entry (long): EMA20 > EMA50 (uptrend), price pulled back to VWAP/EMA20,
              RSI dropped from >60 to 40-55, declining volume on pullback,
              bounce candle (close > open on current/recent bar).
Stop: below EMA50 or pullback low.
Target: previous swing high (approximated by day high).
Trailing: EMA9 after 1R profit.
Active window: 10:00 AM to 3:00 PM ET.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .base import ExitSignal
from .intraday_base import IntradaySetup, IntradayStrategy
from .intraday_registry import IntradayStrategyRegistry

logger = logging.getLogger(__name__)


@IntradayStrategyRegistry.register('pullback_entry')
class PullbackEntryStrategy(IntradayStrategy):
    """Pullback entry strategy — buy the dip in uptrends.

    Waits for a stock in an established uptrend to pull back to support,
    then enters when it shows signs of resuming the trend.
    """

    name = 'Pullback Entry'
    description = 'Trend pullback to support with bounce confirmation'
    version = '1.0'
    strategy_id = 'pullback_entry'

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self._entered_symbols: Dict[str, int] = {}

    # ── Scanning ──────────────────────────────────────────────────────

    def scan_for_setups(self, symbol: str, tick_data: Dict,
                        snapshot: Optional[Dict],
                        now: datetime) -> Optional[IntradaySetup]:
        """Scan for pullback entry setups.

        Long criteria:
        - EMA20 > EMA50 (established uptrend)
        - Price near VWAP or EMA20 (pulled back to support)
        - RSI 40-55 (pulled back but not broken)
        - Volume declining or low (selling pressure fading)
        - Current bar shows bounce (close > open approximated by price > EMA9)

        Short criteria: mirror conditions for downtrend pullbacks.
        """
        if not tick_data:
            return None

        price = None
        if snapshot:
            price = snapshot.get('price') or snapshot.get('latestTrade', {}).get('p')
        if price is None:
            return None

        max_entries = self._config.get('max_entries_per_session', 2)
        if self._entered_symbols.get(symbol, 0) >= max_entries:
            return None

        # Required indicators
        ema9 = tick_data.get('ema9', 0)
        ema20 = tick_data.get('ema20', 0)
        ema50 = tick_data.get('ema50', 0)
        rsi = tick_data.get('rsi', 50)
        rsi_initialized = tick_data.get('rsi_initialized', False)
        vwap = tick_data.get('vwap', 0)
        atr = tick_data.get('atr', 0)
        vol_surge = tick_data.get('volume_surge_ratio', 1.0)
        day_high = tick_data.get('day_high', 0)
        day_low = tick_data.get('day_low', float('inf'))

        if ema20 <= 0 or ema50 <= 0 or not rsi_initialized:
            return None

        # Detect direction
        direction = None
        pullback_pct = self._config.get('pullback_proximity_pct', 0.01)
        rsi_pb_min = self._config.get('rsi_pullback_min', 40)
        rsi_pb_max = self._config.get('rsi_pullback_max', 55)
        max_vol_surge = self._config.get('max_volume_surge_pullback', 1.5)

        # ── LONG PULLBACK ──
        if ema20 > ema50:
            # Price near VWAP or EMA20 (within pullback_pct)
            near_vwap = vwap > 0 and abs(price - vwap) / vwap < pullback_pct
            near_ema20 = abs(price - ema20) / ema20 < pullback_pct
            if not (near_vwap or near_ema20):
                return None  # Not pulled back enough

            # RSI in pullback zone
            if not (rsi_pb_min <= rsi <= rsi_pb_max):
                return None

            # Volume declining (not surging)
            if vol_surge > max_vol_surge:
                return None  # Volume still high — selling not done

            # Bounce confirmation: price above EMA9 (short-term trend resuming)
            if ema9 > 0 and price > ema9:
                direction = 'long'

        # ── SHORT PULLBACK ──
        elif ema20 < ema50:
            rsi_pb_short_min = self._config.get('rsi_pullback_short_min', 45)
            rsi_pb_short_max = self._config.get('rsi_pullback_short_max', 60)

            near_vwap = vwap > 0 and abs(price - vwap) / vwap < pullback_pct
            near_ema20 = abs(price - ema20) / ema20 < pullback_pct
            if not (near_vwap or near_ema20):
                return None

            if not (rsi_pb_short_min <= rsi <= rsi_pb_short_max):
                return None

            if vol_surge > max_vol_surge:
                return None

            if ema9 > 0 and price < ema9:
                direction = 'short'

        if direction is None:
            return None

        # Compute stop, target, R:R
        target_rr = self._config.get('target_rr', 2.0)

        if direction == 'long':
            # Stop below EMA50 or recent low
            ema50_stop = ema50 * (1 - 0.003)
            low_stop = day_low * (1 - 0.002) if day_low < float('inf') else ema50_stop
            stop_price = max(ema50_stop, low_stop)  # Tighter wins
            risk = price - stop_price
            if risk <= 0:
                return None
            # Target: day high (previous swing high)
            target_price = day_high if day_high > price else price + (risk * target_rr)
        else:
            ema50_stop = ema50 * (1 + 0.003)
            high_stop = day_high * (1 + 0.002) if day_high > 0 else ema50_stop
            stop_price = min(ema50_stop, high_stop)
            risk = stop_price - price
            if risk <= 0:
                return None
            target_price = day_low if day_low < price and day_low > 0 else price - (risk * target_rr)

        rr = abs(target_price - price) / risk if risk > 0 else 0

        # Confidence
        confidence = self._compute_confidence(
            tick_data, direction, vol_surge, rsi, price, vwap, ema9, ema20, ema50)

        return IntradaySetup(
            symbol=symbol,
            strategy_id=self.strategy_id,
            direction=direction,
            entry_price=price,
            stop_price=round(stop_price, 2),
            target_price=round(target_price, 2),
            risk_reward=round(rr, 2),
            confidence=round(confidence, 3),
            setup_type='pullback_entry',
            indicators={
                'ema9': round(ema9, 2), 'ema20': round(ema20, 2),
                'ema50': round(ema50, 2) if ema50 else None,
                'rsi': round(rsi, 1), 'vwap': round(vwap, 2) if vwap else None,
                'volume_surge': round(vol_surge, 2),
            },
            timestamp=now,
            notes=f'Pullback to {"VWAP" if vwap and abs(price - vwap) / vwap < pullback_pct else "EMA20"}, RSI={rsi:.0f}',
        )

    def _compute_confidence(self, tick_data, direction, vol_surge, rsi,
                            price, vwap, ema9, ema20, ema50):
        score = 0.50
        # EMA alignment strength
        if ema20 > 0 and ema50 > 0:
            spread = abs(ema20 - ema50) / ema50
            if spread >= 0.005:
                score += 0.10
        # Volume declining (good for pullback)
        if vol_surge < 0.8:
            score += 0.10
        elif vol_surge < 1.0:
            score += 0.05
        # RSI sweet spot
        if direction == 'long' and 42 <= rsi <= 52:
            score += 0.10
        elif direction == 'short' and 48 <= rsi <= 58:
            score += 0.10
        # VWAP proximity (closer = better support)
        if vwap > 0:
            dist = abs(price - vwap) / vwap
            if dist < 0.003:
                score += 0.10
        # Bounce confirmation strength (price distance from EMA9)
        if ema9 > 0:
            if direction == 'long' and price > ema9 * 1.002:
                score += 0.05
            elif direction == 'short' and price < ema9 * 0.998:
                score += 0.05
        return min(score, 0.95)

    def get_watchlist_criteria(self) -> Dict:
        return {
            'min_volume': 500_000,
            'min_price': 15.0,
            'max_price': 500.0,
            'min_adr_pct': 0.015,
            'prefer_gappers': True,
        }

    def get_active_window(self) -> Tuple[int, int, int, int]:
        return (10, 0, 15, 0)

    def record_entry(self, symbol: str):
        self._entered_symbols[symbol] = self._entered_symbols.get(symbol, 0) + 1

    # ── Exit ──────────────────────────────────────────────────────────

    def evaluate_exit(self, position: Dict, price: float,
                      tick_data: Optional[Dict] = None,
                      now: Optional[datetime] = None) -> Optional[ExitSignal]:
        if not tick_data:
            return None
        direction = position.get('direction', 'long')
        ema20 = tick_data.get('ema20', 0)
        if ema20 <= 0:
            return None
        # Exit if price crosses through EMA20 against direction
        if direction == 'long' and price < ema20 * 0.995:
            return ExitSignal(action='close',
                              reason=f'Pullback failed: price ${price:.2f} < EMA20 ${ema20:.2f}')
        if direction == 'short' and price > ema20 * 1.005:
            return ExitSignal(action='close',
                              reason=f'Pullback failed: price ${price:.2f} > EMA20 ${ema20:.2f}')
        return None

    def update_trailing_stop(self, position: Dict, price: float,
                             tick_data: Optional[Dict] = None,
                             now: Optional[datetime] = None) -> Optional[float]:
        if not tick_data:
            return None
        direction = position.get('direction', 'long')
        entry_price = position.get('entry_price', 0)
        stop_price = position.get('stop_price', 0)
        if entry_price <= 0 or stop_price <= 0:
            return None
        if direction == 'long':
            initial_risk = entry_price - stop_price
            pnl = price - entry_price
        else:
            initial_risk = stop_price - entry_price
            pnl = entry_price - price
        if initial_risk <= 0 or pnl / initial_risk < 1.0:
            return None
        ema9 = tick_data.get('ema9', 0)
        if ema9 <= 0:
            return None
        if direction == 'long':
            new_stop = ema9 * 0.999
            return round(new_stop, 2) if new_stop > stop_price else None
        else:
            new_stop = ema9 * 1.001
            return round(new_stop, 2) if new_stop < stop_price else None

    # ── Indicators ────────────────────────────────────────────────────

    def get_required_indicators(self) -> List[str]:
        return ['vwap', 'ema', 'ema_multi', 'rsi', 'atr',
                'volume_profile', 'day_high', 'day_low', 'bar_history']

    def get_ema_periods(self) -> List[int]:
        return [9, 20, 50]

    # ── Config ────────────────────────────────────────────────────────

    def get_parameter_schema(self) -> Dict:
        return {
            'pullback_proximity_pct': {
                'type': 'float', 'label': 'Pullback Proximity %',
                'default': 0.01, 'min': 0.003, 'max': 0.03,
                'description': 'Price must be within this % of VWAP/EMA20',
            },
            'rsi_pullback_min': {
                'type': 'float', 'label': 'RSI Pullback Min',
                'default': 40, 'min': 25, 'max': 50,
            },
            'rsi_pullback_max': {
                'type': 'float', 'label': 'RSI Pullback Max',
                'default': 55, 'min': 45, 'max': 65,
            },
            'max_volume_surge_pullback': {
                'type': 'float', 'label': 'Max Vol Surge (Pullback)',
                'default': 1.5, 'min': 1.0, 'max': 3.0,
                'description': 'Volume must be below this ratio (selling fading)',
            },
            'target_rr': {
                'type': 'float', 'label': 'Target R:R Multiple',
                'default': 2.0, 'min': 1.5, 'max': 4.0,
            },
            'max_entries_per_session': {
                'type': 'int', 'label': 'Max Entries', 'default': 2,
                'min': 1, 'max': 5,
            },
        }

    def get_default_config(self) -> Dict:
        return {
            'pullback_proximity_pct': 0.01,
            'rsi_pullback_min': 40,
            'rsi_pullback_max': 55,
            'rsi_pullback_short_min': 45,
            'rsi_pullback_short_max': 60,
            'max_volume_surge_pullback': 1.5,
            'target_rr': 2.0,
            'max_entries_per_session': 2,
            'min_risk_reward': 1.5,
            'min_confidence': 0.5,
        }

    def on_day_start(self):
        self._entered_symbols.clear()

    def on_day_end(self):
        self._entered_symbols.clear()
