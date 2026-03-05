"""Momentum Surge — intraday strategy.

Identifies strong momentum moves with aligned technicals and volume confirmation.

Entry (long): price > EMA9 > EMA20, RSI 50-75, volume >= 2x avg, price > VWAP,
              new intraday high.
Entry (short): price < EMA9 < EMA20, RSI 25-50, volume >= 2x avg, price < VWAP,
               new intraday low.
Stop: below EMA20 or 1.5 ATR (whichever is tighter).
Target: 2x risk from entry.
Trailing: EMA9 after 1R profit.
Active window: 9:45 AM to 3:00 PM ET.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .base import ExitSignal
from .intraday_base import IntradaySetup, IntradayStrategy
from .intraday_registry import IntradayStrategyRegistry

logger = logging.getLogger(__name__)


@IntradayStrategyRegistry.register('momentum_surge')
class MomentumSurgeStrategy(IntradayStrategy):
    """Momentum surge strategy — trend + volume + VWAP alignment.

    Catches strong directional moves after EMA alignment and volume confirmation.
    Uses EMA9 trailing stop for trend following.
    """

    name = 'Momentum Surge'
    description = 'Trend-aligned momentum with volume confirmation'
    version = '1.0'
    strategy_id = 'momentum_surge'

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self._entered_symbols: Dict[str, int] = {}
        self._cooldown_symbols: Dict[str, float] = {}  # symbol -> monotonic timestamp

    # ── Scanning ──────────────────────────────────────────────────────

    def scan_for_setups(self, symbol: str, tick_data: Dict,
                        snapshot: Optional[Dict],
                        now: datetime) -> Optional[IntradaySetup]:
        """Scan for momentum surge setups.

        Long criteria:
        - Price > EMA9 > EMA20 (EMA alignment)
        - RSI 50-75 (momentum but not overbought)
        - Volume surge >= 2x 20-bar average
        - Price > VWAP
        - Price near or at intraday high

        Short criteria: mirror conditions.
        """
        if not tick_data:
            return None

        # Get current price
        price = None
        if snapshot:
            price = snapshot.get('price') or snapshot.get('latestTrade', {}).get('p')
        if price is None:
            return None

        # Max entries check
        max_entries = self._config.get('max_entries_per_session', 3)
        if self._entered_symbols.get(symbol, 0) >= max_entries:
            return None

        # Required indicators
        ema9 = tick_data.get('ema9', 0)
        ema20 = tick_data.get('ema20', 0)
        rsi = tick_data.get('rsi', 50)
        rsi_initialized = tick_data.get('rsi_initialized', False)
        vwap = tick_data.get('vwap', 0)
        atr = tick_data.get('atr', 0)
        vol_surge = tick_data.get('volume_surge_ratio', 0)
        day_high = tick_data.get('day_high', 0)
        day_low = tick_data.get('day_low', float('inf'))

        # Must have initialized EMAs and RSI
        if ema9 <= 0 or ema20 <= 0:
            return None
        if not rsi_initialized:
            return None

        # Volume confirmation
        min_vol_surge = self._config.get('volume_surge_ratio', 2.0)
        if vol_surge < min_vol_surge:
            return None

        # Check for long or short setup
        direction = None
        rsi_min_long = self._config.get('rsi_min_long', 50)
        rsi_max_long = self._config.get('rsi_max_long', 75)
        rsi_min_short = self._config.get('rsi_min_short', 25)
        rsi_max_short = self._config.get('rsi_max_short', 50)

        near_high_pct = self._config.get('near_high_pct', 0.005)
        near_low_pct = self._config.get('near_low_pct', 0.005)

        if (price > ema9 > ema20 and
                rsi_min_long <= rsi <= rsi_max_long and
                (vwap <= 0 or price > vwap)):
            # Near intraday high check
            if day_high > 0 and price >= day_high * (1 - near_high_pct):
                direction = 'long'

        elif (price < ema9 < ema20 and
              rsi_min_short <= rsi <= rsi_max_short and
              (vwap <= 0 or price < vwap)):
            # Near intraday low check
            if day_low < float('inf') and price <= day_low * (1 + near_low_pct):
                direction = 'short'

        if direction is None:
            return None

        # Compute stop, target, R:R
        target_rr = self._config.get('target_rr', 2.0)
        atr_stop_mult = self._config.get('atr_stop_mult', 1.5)

        if direction == 'long':
            ema_stop = ema20 * (1 - 0.001)  # Below EMA20
            atr_stop = price - (atr * atr_stop_mult) if atr > 0 else ema_stop
            stop_price = max(ema_stop, atr_stop)  # Tighter stop wins
            risk = price - stop_price
            if risk <= 0:
                return None
            target_price = price + (risk * target_rr)
        else:
            ema_stop = ema20 * (1 + 0.001)  # Above EMA20
            atr_stop = price + (atr * atr_stop_mult) if atr > 0 else ema_stop
            stop_price = min(ema_stop, atr_stop)  # Tighter stop wins
            risk = stop_price - price
            if risk <= 0:
                return None
            target_price = price - (risk * target_rr)

        rr = abs(target_price - price) / risk if risk > 0 else 0

        # Confidence scoring
        confidence = self._compute_confidence(
            tick_data, direction, vol_surge, rsi, price, vwap, ema9, ema20)

        return IntradaySetup(
            symbol=symbol,
            strategy_id=self.strategy_id,
            direction=direction,
            entry_price=price,
            stop_price=round(stop_price, 2),
            target_price=round(target_price, 2),
            risk_reward=round(rr, 2),
            confidence=round(confidence, 3),
            setup_type='momentum_surge',
            indicators={
                'ema9': round(ema9, 2),
                'ema20': round(ema20, 2),
                'rsi': round(rsi, 1),
                'volume_surge': round(vol_surge, 2),
                'vwap': round(vwap, 2) if vwap else None,
                'atr': round(atr, 2) if atr else None,
            },
            timestamp=now,
            notes=(f'EMA9={ema9:.1f}>EMA20={ema20:.1f}, '
                   f'RSI={rsi:.0f}, vol {vol_surge:.1f}x'),
        )

    def _compute_confidence(self, tick_data: Dict, direction: str,
                            vol_surge: float, rsi: float,
                            price: float, vwap: float,
                            ema9: float, ema20: float) -> float:
        """Compute confidence score (0.0 - 1.0)."""
        score = 0.45  # base

        # EMA spread (wider = stronger trend)
        if ema9 > 0 and ema20 > 0:
            ema_spread = abs(ema9 - ema20) / ema20
            if ema_spread >= 0.005:
                score += 0.10
            if ema_spread >= 0.01:
                score += 0.05

        # Volume strength
        if vol_surge >= 3.0:
            score += 0.15
        elif vol_surge >= 2.5:
            score += 0.10
        elif vol_surge >= 2.0:
            score += 0.05

        # RSI sweet spot (not extreme)
        if direction == 'long' and 55 <= rsi <= 70:
            score += 0.10
        elif direction == 'short' and 30 <= rsi <= 45:
            score += 0.10

        # VWAP alignment
        if vwap > 0:
            if direction == 'long' and price > vwap * 1.002:
                score += 0.05
            elif direction == 'short' and price < vwap * 0.998:
                score += 0.05

        # EMA50 trend confirmation (if available)
        ema50 = tick_data.get('ema50', 0)
        if ema50 > 0:
            if direction == 'long' and ema20 > ema50:
                score += 0.05
            elif direction == 'short' and ema20 < ema50:
                score += 0.05

        return min(score, 0.95)

    def get_watchlist_criteria(self) -> Dict:
        return {
            'min_volume': 1_000_000,
            'min_price': 15.0,
            'max_price': 500.0,
            'min_adr_pct': 0.025,
            'prefer_gappers': True,
        }

    # ── Entry ─────────────────────────────────────────────────────────

    def get_active_window(self) -> Tuple[int, int, int, int]:
        """Momentum active from 9:45 AM to 3:00 PM."""
        return (9, 45, 15, 0)

    def record_entry(self, symbol: str):
        self._entered_symbols[symbol] = self._entered_symbols.get(symbol, 0) + 1

    # ── Exit Management ───────────────────────────────────────────────

    def evaluate_exit(self, position: Dict, price: float,
                      tick_data: Optional[Dict] = None,
                      now: Optional[datetime] = None) -> Optional[ExitSignal]:
        """Momentum exit: EMA cross against position direction."""
        if not tick_data:
            return None

        direction = position.get('direction', 'long')
        ema9 = tick_data.get('ema9', 0)

        if ema9 <= 0:
            return None

        # Long: close if price crosses below EMA9
        if direction == 'long' and price < ema9 * 0.998:
            return ExitSignal(
                action='close',
                reason=f'Momentum lost: price ${price:.2f} < EMA9 ${ema9:.2f}',
            )

        # Short: close if price crosses above EMA9
        if direction == 'short' and price > ema9 * 1.002:
            return ExitSignal(
                action='close',
                reason=f'Momentum lost: price ${price:.2f} > EMA9 ${ema9:.2f}',
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

        # Only trail after 1R profit
        if pnl / initial_risk < 1.0:
            return None

        ema9 = tick_data.get('ema9', 0)
        if ema9 <= 0:
            return None

        buffer_pct = 0.001
        if direction == 'long':
            new_stop = ema9 * (1 - buffer_pct)
            if new_stop > stop_price:
                return round(new_stop, 2)
        else:
            new_stop = ema9 * (1 + buffer_pct)
            if new_stop < stop_price:
                return round(new_stop, 2)

        return None

    # ── Indicators ────────────────────────────────────────────────────

    def get_required_indicators(self) -> List[str]:
        return ['vwap', 'ema', 'ema_multi', 'rsi', 'atr',
                'volume_profile', 'day_high', 'day_low', 'bar_history']

    def get_ema_periods(self) -> List[int]:
        return [9, 20]

    # ── Config ────────────────────────────────────────────────────────

    def get_parameter_schema(self) -> Dict:
        return {
            'volume_surge_ratio': {
                'type': 'float', 'label': 'Volume Surge Ratio',
                'default': 2.0, 'min': 1.5, 'max': 5.0,
                'description': 'Minimum volume surge for entry'
            },
            'rsi_min_long': {
                'type': 'float', 'label': 'RSI Min (Long)',
                'default': 50, 'min': 30, 'max': 70,
                'description': 'Minimum RSI for long entries'
            },
            'rsi_max_long': {
                'type': 'float', 'label': 'RSI Max (Long)',
                'default': 75, 'min': 60, 'max': 90,
                'description': 'Maximum RSI for long entries (avoid overbought)'
            },
            'rsi_min_short': {
                'type': 'float', 'label': 'RSI Min (Short)',
                'default': 25, 'min': 10, 'max': 40,
                'description': 'Minimum RSI for short entries'
            },
            'rsi_max_short': {
                'type': 'float', 'label': 'RSI Max (Short)',
                'default': 50, 'min': 30, 'max': 70,
                'description': 'Maximum RSI for short entries (avoid oversold)'
            },
            'atr_stop_mult': {
                'type': 'float', 'label': 'ATR Stop Multiple',
                'default': 1.5, 'min': 1.0, 'max': 3.0,
                'description': 'ATR multiplier for stop placement'
            },
            'target_rr': {
                'type': 'float', 'label': 'Target R:R Multiple',
                'default': 2.0, 'min': 1.5, 'max': 4.0,
                'description': 'Target profit as multiple of risk'
            },
            'near_high_pct': {
                'type': 'float', 'label': 'Near High %',
                'default': 0.005, 'min': 0.001, 'max': 0.02,
                'description': 'Price must be within this % of day high for long'
            },
            'max_entries_per_session': {
                'type': 'int', 'label': 'Max Entries Per Session',
                'default': 3, 'min': 1, 'max': 10,
                'description': 'Maximum momentum entries per trading day'
            },
        }

    def get_default_config(self) -> Dict:
        return {
            'volume_surge_ratio': 2.0,
            'rsi_min_long': 50,
            'rsi_max_long': 75,
            'rsi_min_short': 25,
            'rsi_max_short': 50,
            'atr_stop_mult': 1.5,
            'target_rr': 2.0,
            'near_high_pct': 0.005,
            'near_low_pct': 0.005,
            'max_entries_per_session': 3,
            'min_risk_reward': 1.5,
            'min_confidence': 0.5,
        }

    # ── Lifecycle ─────────────────────────────────────────────────────

    def on_day_start(self):
        self._entered_symbols.clear()
        self._cooldown_symbols.clear()

    def on_day_end(self):
        self._entered_symbols.clear()
        self._cooldown_symbols.clear()
