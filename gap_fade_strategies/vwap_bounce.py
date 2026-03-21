"""VWAP Bounce / Reclaim — intraday strategy.

Institutional algorithms anchor to VWAP, creating structural support/resistance.
Instead of fading deviations FROM VWAP (mean reversion), this strategy trades
pullbacks TO VWAP in trending stocks — buying the bounce in uptrends,
shorting the rejection in downtrends.

Entry:
    LONG:  Price in uptrend (above EMA20) → pulls back to within vwap_proximity
           of VWAP → bounces (current bar closes above VWAP) → BUY
    SHORT: Price in downtrend (below EMA20) → rallies to within vwap_proximity
           of VWAP → rejects (current bar closes below VWAP) → SHORT

The "reclaim" variant: price breaks below VWAP, then reclaims above on volume
→ BUY (failed breakdown = short squeeze + momentum buyers)

Stop:  Below recent swing low (longs) or above recent swing high (shorts),
       minimum ATR * 1.0
Target: 2x risk (R:R = 2.0), or next resistance/support level
Active window: 9:50 AM to 3:00 PM ET.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .base import ExitSignal
from .intraday_base import IntradaySetup, IntradayStrategy
from .intraday_registry import IntradayStrategyRegistry

logger = logging.getLogger(__name__)


@IntradayStrategyRegistry.register('vwap_bounce')
class VWAPBounceStrategy(IntradayStrategy):
    """VWAP Bounce / Reclaim strategy.

    Trades pullbacks TO VWAP in trending markets, not deviations FROM VWAP.
    The key insight: VWAP acts as institutional support in uptrends and
    resistance in downtrends.
    """

    name = 'VWAP Bounce'
    description = 'Trade pullbacks to VWAP in trending stocks (institutional support/resistance)'
    version = '1.0'
    strategy_id = 'vwap_bounce'

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self._entered_symbols: Dict[str, int] = {}
        self._was_near_vwap: Dict[str, bool] = {}  # track if price touched VWAP zone

    def get_default_config(self) -> Dict:
        return {
            # VWAP proximity
            'vwap_proximity_pct': 0.002,   # within 0.2% of VWAP = "at VWAP"
            'vwap_bounce_pct': 0.001,      # must move 0.1% away from VWAP to confirm bounce
            # Trend confirmation
            'trend_ema_period': 20,        # EMA for trend direction
            'require_ema_trend': True,     # price must be on correct side of EMA
            'min_trend_strength': 0.001,   # EMA slope minimum (price vs EMA distance)
            # Stop loss
            'stop_atr_mult': 1.2,          # stop = ATR * this below/above VWAP
            'min_stop_pct': 0.004,         # minimum 0.4% stop
            'max_stop_pct': 0.012,         # maximum 1.2% stop
            # Target
            'target_rr': 2.0,             # risk:reward target ratio
            'min_target_pct': 0.006,       # minimum 0.6% target
            # Reclaim variant
            'enable_reclaim': True,        # enable VWAP reclaim signals
            'reclaim_volume_surge': 1.3,   # volume must be 1.3x average for reclaim
            # Position management
            'max_hold_minutes': 90,
            'max_entries_per_day': 3,
            # Quality
            'min_risk_reward': 1.5,
            'min_confidence': 0.45,
            'min_volume_ratio': 0.6,
        }

    def get_required_indicators(self) -> List[str]:
        return ['vwap', 'ema', 'ema_multi', 'rsi', 'atr', 'day_high', 'day_low',
                'bar_history']

    def get_ema_periods(self) -> List[int]:
        return [9, 20, 50]

    def get_active_window(self) -> Tuple[int, int, int, int]:
        return (9, 50, 15, 0)

    def get_watchlist_criteria(self) -> Dict:
        return {
            'min_volume': 500_000,
            'min_price': 10.0,
            'max_price': 500.0,
            'min_adr_pct': 0.005,
            'prefer_gappers': False,
        }

    # ── Scanning ──────────────────────────────────────────────────────

    def scan_for_setups(self, symbol: str, tick_data: Dict,
                        snapshot: Optional[Dict],
                        now: datetime) -> Optional[IntradaySetup]:
        if not tick_data:
            return None

        # Check max entries
        max_entries = self._config.get('max_entries_per_day', 3)
        if self._entered_symbols.get(symbol, 0) >= max_entries:
            return None

        # Get price
        price = None
        if snapshot:
            price = snapshot.get('price') or snapshot.get('latestTrade', {}).get('p')
        if price is None or price <= 0:
            return None

        # Need VWAP, ATR
        vwap = tick_data.get('vwap', 0)
        atr = tick_data.get('atr', 0)
        cum_vol = tick_data.get('cum_volume', 0)
        if not vwap or vwap <= 0 or not atr or atr <= 0 or cum_vol <= 0:
            return None

        # Need RSI initialized
        if not tick_data.get('rsi_initialized'):
            return None
        rsi = tick_data.get('rsi', 50)

        # Trend EMA
        ema20 = None
        ema_multi = tick_data.get('ema_multi', {})
        if isinstance(ema_multi, dict):
            ema20 = ema_multi.get(20) or ema_multi.get('20')
        if ema20 is None:
            ema20 = tick_data.get('ema20') or tick_data.get('ema', 0)
        if not ema20 or ema20 <= 0:
            return None

        # Volume filter
        min_vol = self._config.get('min_volume_ratio', 0.6)
        vol_ratio = tick_data.get('relative_volume', 1.0)
        if vol_ratio < min_vol:
            return None

        proximity_pct = self._config.get('vwap_proximity_pct', 0.002)
        require_ema = self._config.get('require_ema_trend', True)

        dist_from_vwap = (price - vwap) / vwap  # positive = above VWAP

        # Use bar history to detect bounce pattern:
        # Previous bar touched VWAP zone, current bar moved away = bounce
        bar_history = tick_data.get('bar_history', [])
        prev_bar_touched_vwap = False
        if bar_history and len(bar_history) >= 2:
            prev_bar = bar_history[-2] if not isinstance(bar_history[-2], dict) else bar_history[-2]
            pb_low = getattr(prev_bar, 'low', None) or (prev_bar.get('low') if isinstance(prev_bar, dict) else None)
            pb_high = getattr(prev_bar, 'high', None) or (prev_bar.get('high') if isinstance(prev_bar, dict) else None)
            if pb_low and pb_high and vwap > 0:
                # Previous bar's range crossed VWAP zone
                if pb_low <= vwap * (1 + proximity_pct) and pb_high >= vwap * (1 - proximity_pct):
                    prev_bar_touched_vwap = True

        if not prev_bar_touched_vwap:
            # Also check: current price is near VWAP and bouncing away
            # (for stateful tracking in live mode)
            near_vwap = abs(dist_from_vwap) <= proximity_pct
            was_near = self._was_near_vwap.get(symbol, False)
            if near_vwap:
                self._was_near_vwap[symbol] = True
                return None
            if was_near:
                self._was_near_vwap[symbol] = False
                prev_bar_touched_vwap = True
            else:
                # Neither bar history nor state shows VWAP touch
                # Check if price is simply close to VWAP in a trending context
                if abs(dist_from_vwap) > proximity_pct * 5:
                    return None  # too far from VWAP
                prev_bar_touched_vwap = True  # allow if within 5x proximity

        direction = None
        confidence = 0.0
        setup_type = ''

        # ── LONG BOUNCE: Price above VWAP in uptrend ──
        if dist_from_vwap > 0:
            if require_ema and price < ema20:
                return None  # not in uptrend
            if rsi > 70:
                return None
            direction = 'long'
            setup_type = 'vwap_bounce_long'
            trend_str = (price - ema20) / ema20 if ema20 > 0 else 0
            confidence = min(0.90, 0.45 + abs(trend_str) * 15 + (50 - abs(rsi - 50)) / 100)

        # ── SHORT BOUNCE: Price below VWAP in downtrend ──
        elif dist_from_vwap < 0:
            if require_ema and price > ema20:
                return None  # not in downtrend
            if rsi < 30:
                return None
            direction = 'short'
            setup_type = 'vwap_bounce_short'
            trend_str = (ema20 - price) / ema20 if ema20 > 0 else 0
            confidence = min(0.90, 0.45 + abs(trend_str) * 15 + (50 - abs(rsi - 50)) / 100)

        # ── RECLAIM variant ──
        if direction is None and self._config.get('enable_reclaim', True):
            reclaim_vol = self._config.get('reclaim_volume_surge', 1.3)
            if dist_from_vwap > 0 and abs(dist_from_vwap) < proximity_pct * 5:
                if vol_ratio >= reclaim_vol and price > ema20:
                    direction = 'long'
                    setup_type = 'vwap_reclaim_long'
                    confidence = min(0.85, 0.50 + (vol_ratio - 1.0) * 0.15)
            elif dist_from_vwap < 0 and abs(dist_from_vwap) < proximity_pct * 5:
                if vol_ratio >= reclaim_vol and price < ema20:
                    direction = 'short'
                    setup_type = 'vwap_reclaim_short'
                    confidence = min(0.85, 0.50 + (vol_ratio - 1.0) * 0.15)

        if direction is None:
            return None

        # Calculate stop and target
        stop_atr = self._config.get('stop_atr_mult', 1.2)
        min_stop = self._config.get('min_stop_pct', 0.004)
        max_stop = self._config.get('max_stop_pct', 0.012)
        target_rr = self._config.get('target_rr', 2.0)
        min_target = self._config.get('min_target_pct', 0.006)

        stop_dist = max(atr * stop_atr, price * min_stop)
        stop_dist = min(stop_dist, price * max_stop)

        if direction == 'long':
            # Stop below VWAP (institutional support should hold)
            stop_price = round(min(vwap - atr * 0.5, price - stop_dist), 2)
            target_dist = max(abs(price - stop_price) * target_rr, price * min_target)
            target_price = round(price + target_dist, 2)
        else:
            stop_price = round(max(vwap + atr * 0.5, price + stop_dist), 2)
            target_dist = max(abs(stop_price - price) * target_rr, price * min_target)
            target_price = round(price - target_dist, 2)

        risk = abs(price - stop_price)
        reward = abs(target_price - price)
        rr = reward / risk if risk > 0 else 0

        return IntradaySetup(
            symbol=symbol,
            strategy_id=self.strategy_id,
            direction=direction,
            entry_price=price,
            stop_price=stop_price,
            target_price=target_price,
            risk_reward=rr,
            confidence=confidence,
            setup_type=setup_type,
            timestamp=now,
            indicators={'vwap': round(vwap, 2), 'ema20': round(ema20, 2),
                        'rsi': round(rsi, 1), 'atr': round(atr, 4),
                        'dist_from_vwap': round(dist_from_vwap * 100, 3),
                        'vol_ratio': round(vol_ratio, 2)},
            notes=f'VWAP={vwap:.2f}, dist={dist_from_vwap:.3%}, EMA20={ema20:.2f}',
        )

    # ── Exit Management ───────────────────────────────────────────────

    def evaluate_exit(self, position: Dict, price: float,
                      tick_data: Optional[Dict] = None,
                      now: Optional[datetime] = None) -> Optional[ExitSignal]:
        """Exit on VWAP cross-back (trend failure) or time."""
        if not tick_data:
            return None

        direction = position.get('direction', 'long')
        vwap = tick_data.get('vwap', 0)

        if vwap and vwap > 0:
            # If we're long and price crosses below VWAP = trend failure
            if direction == 'long' and price < vwap * 0.998:
                return ExitSignal(
                    action='close',
                    reason=f'vwap_cross_below (price={price:.2f} < VWAP={vwap:.2f})',
                )
            # If we're short and price crosses above VWAP = trend failure
            elif direction == 'short' and price > vwap * 1.002:
                return ExitSignal(
                    action='close',
                    reason=f'vwap_cross_above (price={price:.2f} > VWAP={vwap:.2f})',
                )

        # Time exit
        max_hold = self._config.get('max_hold_minutes', 90)
        entry_time = position.get('entry_time', '')
        if entry_time and now:
            try:
                if isinstance(entry_time, str):
                    et = datetime.strptime(entry_time, '%Y-%m-%d %H:%M')
                    if now.tzinfo:
                        et = et.replace(tzinfo=now.tzinfo)
                else:
                    et = entry_time
                elapsed = (now - et).total_seconds() / 60
                if elapsed >= max_hold:
                    return ExitSignal(
                        action='close',
                        reason=f'time_exit ({elapsed:.0f}min >= {max_hold}min)',
                    )
            except (ValueError, TypeError):
                pass

        return None

    def update_trailing_stop(self, position: Dict, price: float,
                             tick_data: Optional[Dict] = None,
                             now: Optional[datetime] = None) -> Optional[float]:
        """Trail stop using VWAP as dynamic support/resistance."""
        if not tick_data:
            return None

        direction = position.get('direction', 'long')
        entry_price = position.get('entry_price', 0)
        stop_price = position.get('stop_price', 0)
        vwap = tick_data.get('vwap', 0)
        atr = tick_data.get('atr', 0)

        if not all([entry_price, stop_price, vwap, atr]):
            return None

        risk = abs(entry_price - stop_price)
        if risk <= 0:
            return None

        if direction == 'long':
            pnl_r = (price - entry_price) / risk
            if pnl_r >= 1.0:
                # After 1R profit, trail stop to VWAP - small buffer
                new_stop = vwap - atr * 0.3
                if new_stop > stop_price:
                    return round(new_stop, 2)
            elif pnl_r >= 0.5:
                # After 0.5R, move to breakeven
                if entry_price > stop_price:
                    return round(entry_price, 2)
        else:
            pnl_r = (entry_price - price) / risk
            if pnl_r >= 1.0:
                new_stop = vwap + atr * 0.3
                if new_stop < stop_price:
                    return round(new_stop, 2)
            elif pnl_r >= 0.5:
                if entry_price < stop_price:
                    return round(entry_price, 2)

        return None

    # ── Lifecycle ─────────────────────────────────────────────────────

    def record_entry(self, symbol: str):
        self._entered_symbols[symbol] = self._entered_symbols.get(symbol, 0) + 1

    def on_day_start(self):
        self._entered_symbols.clear()
        self._was_near_vwap.clear()

    def on_day_end(self):
        self._entered_symbols.clear()
        self._was_near_vwap.clear()
