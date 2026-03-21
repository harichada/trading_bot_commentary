"""First Hour Range Breakout (FHRB) -- intraday strategy.

The first hour of trading (9:30-10:30) establishes a significant range as
institutions position. When price breaks above/below this range later in
the day, it signals directional commitment.

Unlike ORB (first 15 minutes), the full first hour gives more reliable
support/resistance levels with less noise.

Entry: after 10:30, price breaks out of the 1-hour range with volume.
Stop: opposite side of range (or midpoint if range is wide).
Target: measured move = first_hour_range * target_range_mult from entry.
Active window: 10:35 AM to 3:00 PM.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .base import ExitSignal
from .intraday_base import IntradaySetup, IntradayStrategy
from .intraday_registry import IntradayStrategyRegistry

logger = logging.getLogger(__name__)


@IntradayStrategyRegistry.register('first_hour_breakout')
class FirstHourBreakoutStrategy(IntradayStrategy):
    """First Hour Range Breakout strategy.

    Records the high/low of the first trading hour (9:30-10:30), then
    enters on a confirmed breakout above/below that range with volume
    confirmation.  Uses the opposite side of the range (or midpoint for
    wide ranges) as a stop, and targets a measured move equal to the
    first hour range.
    """

    name = 'First Hour Range Breakout'
    description = 'First-hour range breakout with volume confirmation'
    version = '1.0'
    strategy_id = 'first_hour_breakout'

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        # symbol -> {'high', 'low', 'range_pct', 'complete'}
        self._first_hour_range: Dict[str, Dict] = {}
        # symbol -> entry count today
        self._entered_symbols: Dict[str, int] = {}

    # ── First Hour Range Construction ─────────────────────────────────

    def _build_first_hour_range(self, symbol: str, tick_data: Dict) -> Optional[Dict]:
        """Build the first hour range from bar_history.

        Examines bar_history for bars between 09:30 and 10:30 (inclusive).
        The bar_time field uses 'HH:MM' format.

        Returns the cached range dict, or None if data is insufficient.
        """
        # Return cached result if already computed and complete
        if symbol in self._first_hour_range:
            cached = self._first_hour_range[symbol]
            if cached.get('complete'):
                return cached

        bar_history = tick_data.get('bar_history')
        if not bar_history or not isinstance(bar_history, list):
            return None

        fh_high = None
        fh_low = None
        bar_count = 0

        for bar in bar_history:
            # bar_history contains FiveMinBar dataclasses or dicts
            bar_time = getattr(bar, 'bar_time', None) or (bar.get('bar_time', '') if isinstance(bar, dict) else '')
            if not bar_time:
                continue

            # Parse bar_time 'H:MM' or 'HH:MM' into (hour, minute)
            try:
                parts = bar_time.split(':')
                bh, bm = int(parts[0]), int(parts[1])
            except (ValueError, IndexError):
                continue

            # Accept bars from 9:30 to 10:25 (last 5-min bar starting before 10:30)
            bar_minutes = bh * 60 + bm
            if bar_minutes < 9 * 60 + 30 or bar_minutes > 10 * 60 + 25:
                continue

            high = getattr(bar, 'high', None) or (bar.get('high') if isinstance(bar, dict) else None)
            low = getattr(bar, 'low', None) or (bar.get('low') if isinstance(bar, dict) else None)
            if high is None or low is None:
                continue
            if high <= 0 or low <= 0:
                continue

            bar_count += 1
            if fh_high is None or high > fh_high:
                fh_high = high
            if fh_low is None or low < fh_low:
                fh_low = low

        if fh_high is None or fh_low is None or fh_high <= fh_low:
            return None

        # Need at least a few bars to be meaningful
        if bar_count < 2:
            return None

        fh_mid = (fh_high + fh_low) / 2
        if fh_mid <= 0:
            return None

        fh_range_pct = (fh_high - fh_low) / fh_mid

        # Determine if the first hour is complete: need a bar at or after 10:25
        # (last 5-min bar of the first hour), or the current time is past 10:30.
        has_1030_bar = False
        for b in bar_history:
            bt = getattr(b, 'bar_time', None) or (b.get('bar_time', '') if isinstance(b, dict) else '')
            if not bt:
                continue
            try:
                p = bt.split(':')
                h, m = int(p[0]), int(p[1])
                if h * 60 + m >= 10 * 60 + 25:
                    has_1030_bar = True
                    break
            except (ValueError, IndexError):
                continue

        result = {
            'high': fh_high,
            'low': fh_low,
            'mid': fh_mid,
            'range': fh_high - fh_low,
            'range_pct': fh_range_pct,
            'bar_count': bar_count,
            'complete': has_1030_bar,
        }

        self._first_hour_range[symbol] = result
        return result if result['complete'] else None

    # ── Scanning ──────────────────────────────────────────────────────

    def scan_for_setups(self, symbol: str, tick_data: Dict,
                        snapshot: Optional[Dict],
                        now: datetime) -> Optional[IntradaySetup]:
        """Scan for first hour breakout setups.

        Requires:
        - First hour range complete (time >= 10:30)
        - Range within min/max bounds
        - Price breaks above high or below low with buffer
        - Volume surge confirmation
        - Not exceeded max entries for this symbol
        """
        if not tick_data:
            return None

        # Must be past 10:30 to have a complete first hour
        if now.hour < 10 or (now.hour == 10 and now.minute < 30):
            return None

        # Build / retrieve first hour range
        fh = self._build_first_hour_range(symbol, tick_data)
        if fh is None:
            return None

        fh_high = fh['high']
        fh_low = fh['low']
        fh_range_pct = fh['range_pct']

        # Check range is within acceptable bounds
        min_range = self._config.get('min_range_pct', 0.005)
        max_range = self._config.get('max_range_pct', 0.03)
        if fh_range_pct < min_range:
            return None  # Range too narrow -- noise
        if fh_range_pct > max_range:
            return None  # Range too wide -- already moved

        # Check max entries for this symbol
        max_entries = self._config.get('max_entries_per_day', 2)
        entries_today = self._entered_symbols.get(symbol, 0)
        if entries_today >= max_entries:
            return None

        # Get current price
        price = None
        if snapshot:
            price = snapshot.get('price') or snapshot.get('latestTrade', {}).get('p')
        if price is None:
            return None

        # Breakout detection with buffer
        breakout_buffer = self._config.get('breakout_buffer_pct', 0.001)
        direction = None
        if price > fh_high * (1 + breakout_buffer):
            direction = 'long'
        elif price < fh_low * (1 - breakout_buffer):
            direction = 'short'
        else:
            return None  # No breakout

        # Volume confirmation
        min_vol_surge = self._config.get('min_volume_surge', 1.3)
        vol_surge = tick_data.get('volume_surge_ratio', 0.0)
        if vol_surge < min_vol_surge:
            return None  # Not enough volume conviction

        # ── Compute stop price ────────────────────────────────────────

        use_midpoint_stop = self._config.get('use_midpoint_stop', True)
        wide_range_threshold = self._config.get('wide_range_threshold', 0.015)
        min_stop_pct = self._config.get('min_stop_pct', 0.008)
        fh_mid = fh['mid']

        if direction == 'long':
            # Default stop: opposite end of range (first hour low)
            raw_stop = fh_low
            # Use midpoint if range is very wide
            if use_midpoint_stop and fh_range_pct > wide_range_threshold:
                raw_stop = fh_mid
            # Enforce minimum stop distance
            min_stop_dist = price * min_stop_pct
            if price - raw_stop < min_stop_dist:
                raw_stop = price - min_stop_dist
            stop_price = raw_stop
            risk = price - stop_price
        else:
            # Default stop: opposite end of range (first hour high)
            raw_stop = fh_high
            # Use midpoint if range is very wide
            if use_midpoint_stop and fh_range_pct > wide_range_threshold:
                raw_stop = fh_mid
            # Enforce minimum stop distance
            min_stop_dist = price * min_stop_pct
            if raw_stop - price < min_stop_dist:
                raw_stop = price + min_stop_dist
            stop_price = raw_stop
            risk = stop_price - price

        if risk <= 0:
            return None

        # ── Compute target price ──────────────────────────────────────

        target_range_mult = self._config.get('target_range_mult', 1.0)
        fh_range = fh['range']
        measured_move = fh_range * target_range_mult

        if direction == 'long':
            target_price = price + measured_move
        else:
            target_price = price - measured_move

        if target_price <= 0:
            return None

        # Risk-reward check
        reward = abs(target_price - price)
        rr = reward / risk if risk > 0 else 0
        min_rr = self._config.get('min_risk_reward', 1.2)
        if rr < min_rr:
            return None

        # ── Confidence scoring ────────────────────────────────────────

        vwap = tick_data.get('vwap', 0.0)
        confidence = self._compute_confidence(tick_data, direction, vol_surge,
                                              price, vwap)

        min_conf = self._config.get('min_confidence', 0.45)
        if confidence < min_conf:
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
            setup_type='first_hour_breakout',
            indicators={
                'fh_high': round(fh_high, 2),
                'fh_low': round(fh_low, 2),
                'fh_range_pct': round(fh_range_pct, 4),
                'fh_bar_count': fh['bar_count'],
                'volume_surge': round(vol_surge, 2),
                'vwap': round(vwap, 2) if vwap else None,
                'rsi': tick_data.get('rsi'),
                'atr': tick_data.get('atr'),
            },
            timestamp=now,
            notes=(
                f'FH ${fh_low:.2f}-${fh_high:.2f} '
                f'({fh_range_pct:.1%}), '
                f'vol surge {vol_surge:.1f}x'
            ),
        )

    def _compute_confidence(self, tick_data: Dict, direction: str,
                            vol_surge: float, price: float,
                            vwap: float) -> float:
        """Compute confidence score for the setup (0.0 - 1.0)."""
        score = 0.50  # base

        # Volume strength: strong surge adds conviction
        if vol_surge >= 2.0:
            score += 0.10
        elif vol_surge >= 1.5:
            score += 0.05

        # EMA alignment confirms trend direction
        ema9 = tick_data.get('ema9', 0)
        ema20 = tick_data.get('ema20', 0)
        ema50 = tick_data.get('ema50', 0)
        if ema9 > 0 and ema20 > 0 and ema50 > 0:
            if direction == 'long' and ema9 > ema20 > ema50:
                score += 0.10
            elif direction == 'short' and ema9 < ema20 < ema50:
                score += 0.10
            elif direction == 'long' and ema9 > ema20:
                score += 0.05
            elif direction == 'short' and ema9 < ema20:
                score += 0.05

        # RSI: fresh (40-60) is ideal, extreme is a penalty
        rsi = tick_data.get('rsi', 50)
        if rsi is not None:
            if 40 <= rsi <= 60:
                score += 0.10  # Fresh, not overbought/oversold
            elif rsi > 75 or rsi < 25:
                score -= 0.10  # Extreme -- likely to revert

        # VWAP alignment
        if vwap > 0:
            if direction == 'long' and price > vwap:
                score += 0.05
            elif direction == 'short' and price < vwap:
                score += 0.05

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
        """FHRB active from 10:35 AM to 3:00 PM."""
        return (10, 35, 15, 0)

    def record_entry(self, symbol: str):
        """Track that an entry was made for this symbol today."""
        self._entered_symbols[symbol] = self._entered_symbols.get(symbol, 0) + 1

    # ── Exit Management ───────────────────────────────────────────────

    def evaluate_exit(self, position: Dict, price: float,
                      tick_data: Optional[Dict] = None,
                      now: Optional[datetime] = None) -> Optional[ExitSignal]:
        """FHRB-specific exit: close if price falls back inside the range midpoint."""
        symbol = position.get('symbol', '')
        fh = self._first_hour_range.get(symbol)
        if fh is None or not fh.get('complete'):
            return None

        direction = position.get('direction', 'long')
        fh_mid = fh['mid']
        entry_price = position.get('entry_price', 0)

        # Failed breakout: price retreats past the midpoint of the first hour range
        if direction == 'long' and price < fh_mid:
            return ExitSignal(
                action='close',
                reason=(
                    f'FHRB failed breakout: price ${price:.2f} back below '
                    f'FH midpoint ${fh_mid:.2f}'
                ),
            )

        if direction == 'short' and price > fh_mid:
            return ExitSignal(
                action='close',
                reason=(
                    f'FHRB failed breakdown: price ${price:.2f} back above '
                    f'FH midpoint ${fh_mid:.2f}'
                ),
            )

        # Max hold time check
        if now is not None:
            entry_time = position.get('entry_time')
            max_hold = self._config.get('max_hold_minutes', 120)
            if entry_time is not None:
                if isinstance(entry_time, str):
                    try:
                        entry_time = datetime.fromisoformat(entry_time)
                    except (ValueError, TypeError):
                        entry_time = None
                if entry_time is not None:
                    held_minutes = (now - entry_time).total_seconds() / 60
                    if held_minutes >= max_hold:
                        return ExitSignal(
                            action='close',
                            reason=f'FHRB max hold time reached ({max_hold} min)',
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
            if new_stop > stop_price:
                return round(new_stop, 2)
        else:
            new_stop = ema9 * (1 + buffer_pct)
            if new_stop < stop_price:
                return round(new_stop, 2)

        return None

    # ── Indicators ────────────────────────────────────────────────────

    def get_required_indicators(self) -> List[str]:
        return ['vwap', 'ema_multi', 'day_high', 'day_low', 'rsi', 'atr',
                'bar_history', 'volume_profile']

    def get_ema_periods(self) -> List[int]:
        return [9, 20, 50]

    # ── Config ────────────────────────────────────────────────────────

    def get_parameter_schema(self) -> Dict:
        return {
            'min_range_pct': {
                'type': 'float', 'label': 'Min First Hour Range %',
                'default': 0.005, 'min': 0.001, 'max': 0.02,
                'description': 'Minimum first hour range as % of midpoint'
            },
            'max_range_pct': {
                'type': 'float', 'label': 'Max First Hour Range %',
                'default': 0.03, 'min': 0.01, 'max': 0.10,
                'description': 'Maximum first hour range (>3% is too volatile)'
            },
            'breakout_buffer_pct': {
                'type': 'float', 'label': 'Breakout Buffer %',
                'default': 0.001, 'min': 0.0, 'max': 0.01,
                'description': 'Price must exceed FH range by this % to trigger'
            },
            'min_volume_surge': {
                'type': 'float', 'label': 'Min Volume Surge Ratio',
                'default': 1.3, 'min': 1.0, 'max': 5.0,
                'description': 'Breakout bar volume must be this multiple of avg'
            },
            'target_range_mult': {
                'type': 'float', 'label': 'Target Range Multiplier',
                'default': 1.0, 'min': 0.5, 'max': 3.0,
                'description': 'Target = entry + (FH range * this multiplier)'
            },
            'use_midpoint_stop': {
                'type': 'bool', 'label': 'Use Midpoint Stop for Wide Ranges',
                'default': True,
                'description': 'Use range midpoint as stop when range exceeds wide threshold'
            },
            'wide_range_threshold': {
                'type': 'float', 'label': 'Wide Range Threshold %',
                'default': 0.015, 'min': 0.005, 'max': 0.05,
                'description': 'If FH range > this %, use midpoint stop instead of full range'
            },
            'min_stop_pct': {
                'type': 'float', 'label': 'Minimum Stop Distance %',
                'default': 0.008, 'min': 0.002, 'max': 0.02,
                'description': 'Minimum stop distance as % of entry price'
            },
            'max_hold_minutes': {
                'type': 'int', 'label': 'Max Hold Time (minutes)',
                'default': 120, 'min': 30, 'max': 300,
                'description': 'Maximum time to hold a position before forced exit'
            },
            'max_entries_per_day': {
                'type': 'int', 'label': 'Max Entries Per Day',
                'default': 2, 'min': 1, 'max': 10,
                'description': 'Maximum FHRB entries per symbol per day'
            },
            'min_risk_reward': {
                'type': 'float', 'label': 'Min Risk:Reward',
                'default': 1.2, 'min': 0.5, 'max': 3.0,
                'description': 'Minimum risk:reward ratio to accept a setup'
            },
            'min_confidence': {
                'type': 'float', 'label': 'Min Confidence',
                'default': 0.45, 'min': 0.1, 'max': 0.9,
                'description': 'Minimum confidence score to accept a setup'
            },
        }

    def get_default_config(self) -> Dict:
        return {
            'min_range_pct': 0.005,
            'max_range_pct': 0.03,
            'breakout_buffer_pct': 0.001,
            'min_volume_surge': 1.3,
            'target_range_mult': 1.0,
            'use_midpoint_stop': True,
            'wide_range_threshold': 0.015,
            'min_stop_pct': 0.008,
            'max_hold_minutes': 120,
            'max_entries_per_day': 2,
            'min_risk_reward': 1.2,
            'min_confidence': 0.45,
        }

    # ── Lifecycle ─────────────────────────────────────────────────────

    def on_day_start(self):
        """Reset daily tracking."""
        self._first_hour_range.clear()
        self._entered_symbols.clear()

    def on_day_end(self):
        """Cleanup."""
        self._first_hour_range.clear()
        self._entered_symbols.clear()
