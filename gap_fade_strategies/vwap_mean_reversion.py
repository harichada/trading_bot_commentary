"""VWAP Mean Reversion — intraday strategy.

When price deviates significantly from VWAP, mean-revert back toward it.
Institutional traders anchor to VWAP, creating a natural gravitational pull
that this strategy exploits once overextension is confirmed by RSI and volume.

Entry:
    - Price > VWAP + threshold_pct  -> SHORT (revert down)
    - Price < VWAP - threshold_pct  -> LONG  (revert up)
    - Volume surge >= 1.2x (activity present, but not extreme — want exhaustion)
    - RSI > 65 for shorts, RSI < 35 for longs (confirms overextension)

Stop:  max(ATR * 2.0, entry * 0.8%) from entry — prevents too-tight stops.
Target: Price returns to VWAP (full mean reversion).
Active window: 10:00 AM to 3:00 PM ET.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .base import ExitSignal
from .intraday_base import IntradaySetup, IntradayStrategy
from .intraday_registry import IntradayStrategyRegistry

logger = logging.getLogger(__name__)


@IntradayStrategyRegistry.register('vwap_mean_reversion')
class VWAPMeanReversionStrategy(IntradayStrategy):
    """VWAP Mean Reversion strategy.

    Enters counter-trend when price is significantly extended from VWAP,
    targeting a return to the volume-weighted average price. Uses RSI to
    confirm overextension and ATR-based stops to manage risk.
    """

    name = 'VWAP Mean Reversion'
    description = 'Mean-revert to VWAP when price is overextended with RSI confirmation'
    version = '1.0'
    strategy_id = 'vwap_mean_reversion'

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        # Track daily state per symbol
        self._entered_symbols: Dict[str, int] = {}   # symbol -> entry count today
        self._entry_times: Dict[str, datetime] = {}   # symbol -> entry timestamp

    # ── Scanning ──────────────────────────────────────────────────────

    def scan_for_setups(self, symbol: str, tick_data: Dict,
                        snapshot: Optional[Dict],
                        now: datetime) -> Optional[IntradaySetup]:
        """Scan for VWAP mean reversion setups.

        Requires:
        - VWAP is valid (cum_volume > 0)
        - Price deviates from VWAP by at least vwap_deviation_pct
        - RSI confirms overextension (>65 for short, <35 for long)
        - Volume surge present but not extreme (exhaustion, not momentum)
        - ATR initialized for stop calculation
        - Not exceeded max entries per day
        """
        if not tick_data:
            return None

        # ── Check max entries for this symbol ──
        max_entries = self._config.get('max_entries_per_day', 4)
        entries_today = self._entered_symbols.get(symbol, 0)
        if entries_today >= max_entries:
            return None

        # ── Get current price ──
        price = None
        if snapshot:
            price = snapshot.get('price') or snapshot.get('latestTrade', {}).get('p')
        if price is None or price <= 0:
            return None

        # ── Validate VWAP ──
        vwap = tick_data.get('vwap', 0.0)
        cum_volume = tick_data.get('cum_volume', 0.0)
        if vwap <= 0 or cum_volume <= 0:
            return None  # VWAP not yet valid

        # ── Check ATR is initialized ──
        atr = tick_data.get('atr', 0.0)
        if atr <= 0:
            return None

        # ── Calculate VWAP deviation ──
        deviation_pct = (price - vwap) / vwap
        threshold = self._config.get('vwap_deviation_pct', 0.015)

        # Determine direction based on deviation
        direction = None
        if deviation_pct > threshold:
            direction = 'short'  # Price too high above VWAP, expect revert down
        elif deviation_pct < -threshold:
            direction = 'long'   # Price too low below VWAP, expect revert up
        else:
            return None  # Not enough deviation

        # ── RSI confirmation ──
        rsi = tick_data.get('rsi')
        if rsi is None:
            return None

        rsi_overbought = self._config.get('rsi_overbought', 65)
        rsi_oversold = self._config.get('rsi_oversold', 35)

        if direction == 'short' and rsi < rsi_overbought:
            return None  # RSI not overbought enough for short
        if direction == 'long' and rsi > rsi_oversold:
            return None  # RSI not oversold enough for long

        # ── Volume confirmation ──
        vol_surge = tick_data.get('volume_surge_ratio', 0.0)
        min_vol_surge = self._config.get('min_volume_surge', 1.2)
        if vol_surge < min_vol_surge:
            return None  # Not enough volume activity

        # ── Compute stop price ──
        stop_atr_mult = self._config.get('stop_atr_mult', 2.0)
        min_stop_pct = self._config.get('min_stop_pct', 0.008)

        atr_stop_distance = atr * stop_atr_mult
        pct_stop_distance = price * min_stop_pct
        stop_distance = max(atr_stop_distance, pct_stop_distance)

        if direction == 'long':
            stop_price = price - stop_distance
        else:
            stop_price = price + stop_distance

        # Sanity: stop must be positive
        if stop_price <= 0:
            return None

        # ── Target = VWAP (mean reversion) ──
        target_price = vwap

        # ── Risk:Reward check ──
        if direction == 'long':
            risk = price - stop_price
            reward = target_price - price
        else:
            risk = stop_price - price
            reward = price - target_price

        if risk <= 0 or reward <= 0:
            return None  # Invalid geometry (target on wrong side or zero risk)

        rr = reward / risk
        min_rr = self._config.get('min_risk_reward', 1.0)
        if rr < min_rr:
            return None  # Risk:reward not attractive enough

        # ── Confidence scoring ──
        confidence = self._compute_confidence(
            tick_data, direction, deviation_pct, rsi, vol_surge, price, vwap
        )

        min_confidence = self._config.get('min_confidence', 0.4)
        if confidence < min_confidence:
            return None

        return IntradaySetup(
            symbol=symbol,
            strategy_id=self.strategy_id,
            direction=direction,
            entry_price=price,
            stop_price=round(stop_price, 2),
            target_price=round(target_price, 2),
            risk_reward=round(rr, 2),
            confidence=round(confidence, 3),
            setup_type='vwap_mean_reversion',
            indicators={
                'vwap': round(vwap, 2),
                'vwap_deviation_pct': round(deviation_pct, 4),
                'rsi': round(rsi, 2),
                'atr': round(atr, 4),
                'volume_surge': round(vol_surge, 2),
                'ema9': tick_data.get('ema9'),
                'ema20': tick_data.get('ema20'),
            },
            timestamp=now,
            notes=(
                f'VWAP ${vwap:.2f}, dev {deviation_pct:+.2%}, '
                f'RSI {rsi:.0f}, vol {vol_surge:.1f}x'
            ),
        )

    def _compute_confidence(self, tick_data: Dict, direction: str,
                            deviation_pct: float, rsi: float,
                            vol_surge: float, price: float,
                            vwap: float) -> float:
        """Compute confidence score for the setup (0.0 - 1.0).

        Base 0.5, adjusted by:
        - RSI extremes (+0.1 if RSI > 75 for short or RSI < 25 for long)
        - Large deviation (+0.1 if deviation > 2x threshold)
        - EMA9 divergence from VWAP (+0.1 if confirms overextension)
        - Strong trend penalty (-0.1 if EMA9 far from EMA20 in deviation direction)
        """
        score = 0.50

        # RSI extreme bonus
        if direction == 'short' and rsi > 75:
            score += 0.10
        elif direction == 'long' and rsi < 25:
            score += 0.10

        # Large deviation bonus
        threshold = self._config.get('vwap_deviation_pct', 0.015)
        if abs(deviation_pct) > threshold * 2:
            score += 0.10

        # EMA9 divergence from VWAP (confirms overextension)
        ema9 = tick_data.get('ema9', 0)
        if ema9 > 0 and vwap > 0:
            ema_vwap_gap = (ema9 - vwap) / vwap
            if direction == 'short' and ema_vwap_gap > 0.005:
                score += 0.10  # EMA9 also above VWAP — overextended
            elif direction == 'long' and ema_vwap_gap < -0.005:
                score += 0.10  # EMA9 also below VWAP — overextended

        # Trending penalty: if EMA9 far from EMA20 in direction of deviation,
        # the stock may be trending and mean reversion is riskier
        ema20 = tick_data.get('ema20', 0)
        if ema9 > 0 and ema20 > 0:
            ema_spread = (ema9 - ema20) / ema20 if ema20 != 0 else 0
            if direction == 'short' and ema_spread > 0.01:
                score -= 0.10  # Strong uptrend — risky to short
            elif direction == 'long' and ema_spread < -0.01:
                score -= 0.10  # Strong downtrend — risky to go long

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
        """Active from 10:00 AM to 3:00 PM ET.

        Skips first 30 min (opening chaos) and last hour (EOD flows).
        """
        return (10, 0, 15, 0)

    def record_entry(self, symbol: str, now: Optional[datetime] = None):
        """Track that an entry was made for this symbol today."""
        self._entered_symbols[symbol] = self._entered_symbols.get(symbol, 0) + 1
        if now is not None:
            self._entry_times[symbol] = now

    # ── Exit Management ───────────────────────────────────────────────

    def evaluate_exit(self, position: Dict, price: float,
                      tick_data: Optional[Dict] = None,
                      now: Optional[datetime] = None) -> Optional[ExitSignal]:
        """VWAP mean reversion exit logic.

        Exit conditions:
        1. Price returns to VWAP (within target_near_vwap_pct) — target hit
        2. Time exit: position held longer than max_hold_minutes
        """
        if not tick_data:
            return None

        direction = position.get('direction', 'long')
        entry_price = position.get('entry_price', 0)
        if entry_price <= 0:
            return None

        vwap = tick_data.get('vwap', 0.0)
        if vwap <= 0:
            return None

        near_vwap_pct = self._config.get('target_near_vwap_pct', 0.002)

        # ── Target exit: price returned to VWAP ──
        if direction == 'long':
            # Long: target is price rising back to VWAP
            if price >= vwap * (1 - near_vwap_pct):
                return ExitSignal(
                    action='close',
                    reason=(
                        f'VWAP target hit: price ${price:.2f} returned to '
                        f'VWAP ${vwap:.2f} (within {near_vwap_pct:.1%})'
                    ),
                )
        else:
            # Short: target is price falling back to VWAP
            if price <= vwap * (1 + near_vwap_pct):
                return ExitSignal(
                    action='close',
                    reason=(
                        f'VWAP target hit: price ${price:.2f} returned to '
                        f'VWAP ${vwap:.2f} (within {near_vwap_pct:.1%})'
                    ),
                )

        # ── Time exit: max hold exceeded ──
        if now is not None:
            entry_time = position.get('entry_time') or position.get('timestamp')
            if entry_time is not None:
                # Handle string timestamps
                if isinstance(entry_time, str):
                    try:
                        entry_time = datetime.fromisoformat(entry_time)
                    except (ValueError, TypeError):
                        entry_time = None

                if entry_time is not None:
                    max_hold = self._config.get('max_hold_minutes', 45)
                    elapsed = (now - entry_time).total_seconds() / 60
                    if elapsed >= max_hold:
                        return ExitSignal(
                            action='close',
                            reason=(
                                f'Time exit: held {elapsed:.0f}m '
                                f'(max {max_hold}m), price ${price:.2f}'
                            ),
                        )

        return None

    def update_trailing_stop(self, position: Dict, price: float,
                             tick_data: Optional[Dict] = None,
                             now: Optional[datetime] = None) -> Optional[float]:
        """Tighten trailing stop as price moves toward VWAP.

        As the trade works (price reverts toward VWAP), we progressively
        tighten the stop to lock in profit. The stop moves proportionally
        to the fraction of the move captured.
        """
        if not tick_data:
            return None

        direction = position.get('direction', 'long')
        entry_price = position.get('entry_price', 0)
        stop_price = position.get('stop_price', 0)
        if entry_price <= 0 or stop_price <= 0:
            return None

        vwap = tick_data.get('vwap', 0.0)
        if vwap <= 0:
            return None

        atr = tick_data.get('atr', 0.0)
        if atr <= 0:
            return None

        # Calculate how much of the move toward VWAP has been captured
        if direction == 'long':
            total_move = vwap - entry_price    # expected move (positive for long)
            captured = price - entry_price     # actual move so far

            if total_move <= 0:
                return None  # Shouldn't happen, but guard

            fraction = captured / total_move
            if fraction < 0.3:
                return None  # Not enough progress to tighten

            # Trail using a fraction of ATR that tightens as we approach target
            # At 30% captured: stop = entry - 1.5*ATR
            # At 50% captured: stop = entry - 1.0*ATR
            # At 75% captured: stop = entry (breakeven)
            trail_mult = max(0.0, 1.5 - (fraction * 2.0))
            new_stop = price - (atr * max(trail_mult, 0.5))

            # Only tighten, never widen
            if new_stop > stop_price:
                return round(new_stop, 2)

        else:  # short
            total_move = entry_price - vwap    # expected move (positive for short)
            captured = entry_price - price     # actual move so far

            if total_move <= 0:
                return None

            fraction = captured / total_move
            if fraction < 0.3:
                return None

            trail_mult = max(0.0, 1.5 - (fraction * 2.0))
            new_stop = price + (atr * max(trail_mult, 0.5))

            # Only tighten (lower for shorts), never widen
            if new_stop < stop_price:
                return round(new_stop, 2)

        return None

    # ── Indicators ────────────────────────────────────────────────────

    def get_required_indicators(self) -> List[str]:
        return ['vwap', 'rsi', 'atr', 'ema_multi', 'volume_profile',
                'day_high', 'day_low']

    def get_ema_periods(self) -> List[int]:
        return [9, 20]

    # ── Config ────────────────────────────────────────────────────────

    def get_parameter_schema(self) -> Dict:
        return {
            'vwap_deviation_pct': {
                'type': 'float', 'label': 'VWAP Deviation %',
                'default': 0.015, 'min': 0.005, 'max': 0.05,
                'description': 'Minimum % deviation from VWAP to trigger entry'
            },
            'rsi_overbought': {
                'type': 'int', 'label': 'RSI Overbought Threshold',
                'default': 65, 'min': 55, 'max': 85,
                'description': 'RSI must exceed this for short entries'
            },
            'rsi_oversold': {
                'type': 'int', 'label': 'RSI Oversold Threshold',
                'default': 35, 'min': 15, 'max': 45,
                'description': 'RSI must be below this for long entries'
            },
            'min_volume_surge': {
                'type': 'float', 'label': 'Min Volume Surge Ratio',
                'default': 1.2, 'min': 1.0, 'max': 3.0,
                'description': 'Minimum volume surge ratio (want activity, not extreme)'
            },
            'stop_atr_mult': {
                'type': 'float', 'label': 'Stop ATR Multiplier',
                'default': 2.0, 'min': 1.0, 'max': 4.0,
                'description': 'ATR multiplier for initial stop distance'
            },
            'min_stop_pct': {
                'type': 'float', 'label': 'Minimum Stop %',
                'default': 0.008, 'min': 0.003, 'max': 0.02,
                'description': 'Minimum stop distance as % of entry price'
            },
            'target_near_vwap_pct': {
                'type': 'float', 'label': 'Target Near VWAP %',
                'default': 0.002, 'min': 0.0, 'max': 0.01,
                'description': 'Close enough to VWAP to count as target hit'
            },
            'max_hold_minutes': {
                'type': 'int', 'label': 'Max Hold Time (minutes)',
                'default': 45, 'min': 10, 'max': 120,
                'description': 'Maximum minutes to hold a position before time exit'
            },
            'max_entries_per_day': {
                'type': 'int', 'label': 'Max Entries Per Day',
                'default': 4, 'min': 1, 'max': 10,
                'description': 'Maximum VWAP reversion entries per trading day'
            },
            'min_risk_reward': {
                'type': 'float', 'label': 'Min Risk:Reward Ratio',
                'default': 1.0, 'min': 0.5, 'max': 3.0,
                'description': 'Minimum R:R ratio to accept a setup'
            },
            'min_confidence': {
                'type': 'float', 'label': 'Min Confidence',
                'default': 0.4, 'min': 0.1, 'max': 0.8,
                'description': 'Minimum confidence score to enter'
            },
        }

    def get_default_config(self) -> Dict:
        return {
            'vwap_deviation_pct': 0.015,
            'rsi_overbought': 65,
            'rsi_oversold': 35,
            'min_volume_surge': 1.2,
            'stop_atr_mult': 2.0,
            'min_stop_pct': 0.008,
            'target_near_vwap_pct': 0.002,
            'max_hold_minutes': 45,
            'max_entries_per_day': 4,
            'min_risk_reward': 1.0,
            'min_confidence': 0.4,
        }

    # ── Lifecycle ─────────────────────────────────────────────────────

    def on_day_start(self):
        """Reset daily tracking."""
        self._entered_symbols.clear()
        self._entry_times.clear()

    def on_day_end(self):
        """Cleanup."""
        self._entered_symbols.clear()
        self._entry_times.clear()
