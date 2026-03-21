"""Catalyst Momentum — intraday strategy.

This is the bread-and-butter strategy of profitable day traders.
Stocks with unusual volume (>2x RVOL) and significant gaps (>0.8%) are
"in play" — institutional interest creates directional momentum that
persists throughout the session.

The edge: 95% of stocks are boring on any given day. By filtering for
the 5% with catalysts (earnings, news, upgrades, sector moves), we trade
where the actual opportunity is — not where the algos have every edge priced.

Entry:
    LONG:  Stock gapped up >0.8%, RVOL >2x, price pulled back to VWAP/EMA9
           support, RSI 40-65 (not overbought), bar closes with strength
    SHORT: Stock gapped down >0.8%, RVOL >2x, price rallied to VWAP/EMA9
           resistance, RSI 35-60 (not oversold), bar closes with weakness

Stop:  Below the pullback low (longs) or above rally high (shorts)
       Minimum 0.3%, maximum 1.2% — always defined by structure
Target: Trail with EMA9 after 0.5R profit. Initial target 2R but let
        winners run. Real edge is in the outlier wins.

Active window: 9:50 AM to 3:30 PM ET (skip first 20 min chaos).
Avoid: 11:30 AM - 1:30 PM (midday chop kills momentum strategies).
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .base import ExitSignal
from .intraday_base import IntradaySetup, IntradayStrategy
from .intraday_registry import IntradayStrategyRegistry

logger = logging.getLogger(__name__)


@IntradayStrategyRegistry.register('catalyst_momentum')
class CatalystMomentumStrategy(IntradayStrategy):
    """Catalyst Momentum — trade stocks in play with pullback entries.

    Only enters when: unusual volume + gap + pullback to support/resistance.
    This is how experienced day traders actually make money — by being
    extremely selective about WHICH stocks to trade.
    """

    name = 'Catalyst Momentum'
    description = 'Gap & volume filtered momentum with pullback entries'
    version = '1.0'
    strategy_id = 'catalyst_momentum'

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self._entered_symbols: Dict[str, int] = {}
        self._day_gaps: Dict[str, float] = {}       # symbol -> gap %
        self._day_rvol: Dict[str, float] = {}        # symbol -> relative volume
        self._prev_close: Dict[str, float] = {}      # symbol -> previous day close
        self._pullback_seen: Dict[str, bool] = {}    # symbol -> True if pullback detected
        self._pullback_low: Dict[str, float] = {}    # symbol -> pullback low (for stop)
        self._pullback_high: Dict[str, float] = {}   # symbol -> pullback high (for short stop)

    def get_default_config(self) -> Dict:
        return {
            # Stock-in-play filters (THIS IS THE EDGE)
            'min_gap_pct': 0.008,          # 0.8% minimum gap
            'min_rvol': 1.5,               # 1.5x relative volume minimum
            # Entry
            'pullback_to_vwap': True,      # require pullback to VWAP zone
            'vwap_zone_pct': 0.004,        # within 0.4% of VWAP = "at VWAP"
            'ema_pullback_zone': 0.003,    # within 0.3% of EMA9 = "at EMA9"
            'rsi_long_min': 35,            # RSI floor for longs (not dying)
            'rsi_long_max': 65,            # RSI ceiling for longs (not overbought)
            'rsi_short_min': 35,           # RSI floor for shorts (not oversold)
            'rsi_short_max': 65,           # RSI ceiling for shorts (not too strong)
            # Stop loss (structure-based)
            'stop_below_pullback_buffer': 0.002,  # 0.2% below pullback low
            'min_stop_pct': 0.003,         # minimum 0.3% stop
            'max_stop_pct': 0.012,         # maximum 1.2% stop
            # Target
            'initial_target_rr': 2.0,      # initial target 2R
            'trail_after_r': 0.5,          # start trailing after 0.5R profit
            # Midday filter
            'avoid_midday': True,          # skip 11:30-1:30
            'midday_start_hour': 11,
            'midday_start_min': 30,
            'midday_end_hour': 13,
            'midday_end_min': 30,
            # Position management
            'max_entries_per_day': 4,      # per symbol
            'max_hold_minutes': 120,
            # Quality
            'min_risk_reward': 1.5,
            'min_confidence': 0.45,
        }

    def get_required_indicators(self) -> List[str]:
        return ['vwap', 'ema', 'ema_multi', 'rsi', 'atr', 'day_high', 'day_low',
                'bar_history', 'volume_profile']

    def get_ema_periods(self) -> List[int]:
        return [9, 20, 50]

    def get_active_window(self) -> Tuple[int, int, int, int]:
        return (9, 50, 15, 30)

    def get_watchlist_criteria(self) -> Dict:
        return {
            'min_volume': 200_000,
            'min_price': 5.0,
            'max_price': 500.0,
            'min_adr_pct': 0.008,
            'prefer_gappers': True,
        }

    # ── Core: Stock-in-Play Detection ──────────────────────────────────

    def _is_stock_in_play(self, symbol: str, tick_data: Dict) -> bool:
        """The #1 filter. Only trade stocks with unusual activity today."""
        # Check gap
        gap_pct = self._day_gaps.get(symbol)
        if gap_pct is None:
            gap_pct = self._detect_gap(symbol, tick_data)
            if gap_pct is not None:
                self._day_gaps[symbol] = gap_pct

        min_gap = self._config.get('min_gap_pct', 0.008)
        if gap_pct is None or abs(gap_pct) < min_gap:
            return False

        # Check relative volume (current volume vs average)
        vol_surge = tick_data.get('volume_surge_ratio', 0)
        # Also check cumulative volume relative to what we'd expect
        cum_vol = tick_data.get('cum_volume', 0)
        vol_avg = tick_data.get('volume_avg_20bar', 0)

        # Use the better of the two volume signals
        rvol = max(vol_surge, 1.0)
        if vol_avg > 0:
            bar_count = tick_data.get('bar_count', 1)
            expected_vol = vol_avg * max(bar_count, 1)
            if expected_vol > 0:
                cum_rvol = cum_vol / expected_vol
                rvol = max(rvol, cum_rvol)

        self._day_rvol[symbol] = rvol
        min_rvol = self._config.get('min_rvol', 1.5)
        return rvol >= min_rvol

    def _detect_gap(self, symbol: str, tick_data: Dict) -> Optional[float]:
        """Detect opening gap from bar history."""
        bar_history = tick_data.get('bar_history', [])
        if not bar_history:
            return None

        # Get first bar of the day
        first_bar = bar_history[0]
        day_open = getattr(first_bar, 'open', None)
        if day_open is None and isinstance(first_bar, dict):
            day_open = first_bar.get('open')

        if not day_open or day_open <= 0:
            return None

        # Use previous close if available, otherwise estimate from day's data
        prev_close = self._prev_close.get(symbol)
        if prev_close and prev_close > 0:
            return (day_open - prev_close) / prev_close

        # Fallback: check if EMA50 is available (rough proxy for "where price was")
        ema50 = tick_data.get('ema50', 0)
        if ema50 > 0 and abs(day_open - ema50) / ema50 > 0.005:
            return (day_open - ema50) / ema50

        return None

    # ── Scanning ──────────────────────────────────────────────────────

    def scan_for_setups(self, symbol: str, tick_data: Dict,
                        snapshot: Optional[Dict],
                        now: datetime) -> Optional[IntradaySetup]:
        if not tick_data:
            return None

        # Midday chop filter — experienced traders know to step away
        if self._config.get('avoid_midday', True):
            mid_start_h = self._config.get('midday_start_hour', 11)
            mid_start_m = self._config.get('midday_start_min', 30)
            mid_end_h = self._config.get('midday_end_hour', 13)
            mid_end_m = self._config.get('midday_end_min', 30)
            now_mins = now.hour * 60 + now.minute
            mid_start = mid_start_h * 60 + mid_start_m
            mid_end = mid_end_h * 60 + mid_end_m
            if mid_start <= now_mins <= mid_end:
                return None

        # Max entries check
        max_entries = self._config.get('max_entries_per_day', 4)
        if self._entered_symbols.get(symbol, 0) >= max_entries:
            return None

        # THE KEY FILTER: Is this stock in play today?
        if not self._is_stock_in_play(symbol, tick_data):
            return None

        # Get price
        price = None
        if snapshot:
            price = snapshot.get('price') or snapshot.get('latestTrade', {}).get('p')
        if price is None or price <= 0:
            return None

        # Required indicators
        vwap = tick_data.get('vwap', 0)
        ema9 = tick_data.get('ema9', 0)
        ema20 = tick_data.get('ema20', 0)
        rsi = tick_data.get('rsi', 50)
        atr = tick_data.get('atr', 0)
        day_high = tick_data.get('day_high', 0)
        day_low = tick_data.get('day_low', float('inf'))

        if not tick_data.get('rsi_initialized'):
            return None
        if ema9 <= 0 or ema20 <= 0 or atr <= 0 or vwap <= 0:
            return None

        gap_pct = self._day_gaps.get(symbol, 0)
        rvol = self._day_rvol.get(symbol, 1.0)

        # Determine direction from gap
        gap_direction = 'long' if gap_pct > 0 else 'short'

        # ── BAR PATTERN CONFIRMATION ──
        # An experienced trader waits for the candle to confirm direction
        bar_history = tick_data.get('bar_history', [])
        if not bar_history or len(bar_history) < 4:
            return None

        last_bar = bar_history[-1]
        lb_open = getattr(last_bar, 'open', None) or (last_bar.get('open') if isinstance(last_bar, dict) else None)
        lb_close = getattr(last_bar, 'close', None) or (last_bar.get('close') if isinstance(last_bar, dict) else None)
        lb_high = getattr(last_bar, 'high', None) or (last_bar.get('high') if isinstance(last_bar, dict) else None)
        lb_low = getattr(last_bar, 'low', None) or (last_bar.get('low') if isinstance(last_bar, dict) else None)
        if not all([lb_open, lb_close, lb_high, lb_low]):
            return None

        bar_range = lb_high - lb_low
        if bar_range <= 0:
            return None

        # Look for pullback entry
        vwap_zone = self._config.get('vwap_zone_pct', 0.004)
        ema_zone = self._config.get('ema_pullback_zone', 0.003)

        dist_from_vwap = (price - vwap) / vwap
        dist_from_ema9 = (price - ema9) / ema9

        direction = None
        setup_type = ''
        stop_ref = 0  # reference point for stop

        if gap_direction == 'long':
            rsi_min = self._config.get('rsi_long_min', 35)
            rsi_max = self._config.get('rsi_long_max', 65)
            if not (rsi_min <= rsi <= rsi_max):
                return None

            # Price must be above VWAP (trend intact)
            if price < vwap * 0.998:
                return None

            # EMA alignment: 9 > 20 (uptrend)
            if ema9 < ema20 * 0.999:
                return None

            # BAR CONFIRMATION: must be green bar closing in upper half
            if lb_close <= lb_open:
                return None  # red bar = no bounce confirmed
            close_pos = (lb_close - lb_low) / bar_range
            if close_pos < 0.4:
                return None  # weak close

            # Pullback detected: price near VWAP or EMA9 from above
            near_vwap = abs(dist_from_vwap) <= vwap_zone
            near_ema9 = abs(dist_from_ema9) <= ema_zone
            pullback_from_high = (day_high - price) / price > 0.003 if day_high > 0 else False

            if (near_vwap or near_ema9) and pullback_from_high:
                direction = 'long'
                setup_type = 'catalyst_pullback_long'
                stop_ref = min(vwap, day_low if day_low < float('inf') else vwap)
            elif near_vwap or near_ema9:
                if rvol >= 3.0:
                    direction = 'long'
                    setup_type = 'catalyst_rvol_long'
                    stop_ref = min(vwap, ema20)

        elif gap_direction == 'short':
            rsi_min = self._config.get('rsi_short_min', 35)
            rsi_max = self._config.get('rsi_short_max', 65)
            if not (rsi_min <= rsi <= rsi_max):
                return None

            # Price must be below VWAP (downtrend intact)
            if price > vwap * 1.002:
                return None

            # EMA alignment: 9 < 20 (downtrend)
            if ema9 > ema20 * 1.001:
                return None

            # BAR CONFIRMATION: must be red bar closing in lower half
            if lb_close >= lb_open:
                return None  # green bar = no rejection confirmed
            close_pos = (lb_high - lb_close) / bar_range
            if close_pos < 0.4:
                return None  # weak rejection

            # Pullback to VWAP or EMA9 from below
            near_vwap = abs(dist_from_vwap) <= vwap_zone
            near_ema9 = abs(dist_from_ema9) <= ema_zone
            pullback_from_low = (price - day_low) / price > 0.003 if day_low > 0 and day_low < float('inf') else False

            if (near_vwap or near_ema9) and pullback_from_low:
                direction = 'short'
                setup_type = 'catalyst_pullback_short'
                stop_ref = max(vwap, day_high if day_high > 0 else vwap)
            elif near_vwap or near_ema9:
                if rvol >= 3.0:
                    direction = 'short'
                    setup_type = 'catalyst_rvol_short'
                    stop_ref = max(vwap, ema20)

        if direction is None:
            return None

        # Calculate stop and target
        buffer = self._config.get('stop_below_pullback_buffer', 0.002)
        min_stop_pct = self._config.get('min_stop_pct', 0.003)
        max_stop_pct = self._config.get('max_stop_pct', 0.012)
        target_rr = self._config.get('initial_target_rr', 2.0)

        if direction == 'long':
            stop_price = stop_ref * (1 - buffer)
            stop_dist = price - stop_price
            # Enforce min/max stop
            stop_dist = max(stop_dist, price * min_stop_pct)
            stop_dist = min(stop_dist, price * max_stop_pct)
            stop_price = round(price - stop_dist, 2)
            target_price = round(price + stop_dist * target_rr, 2)
        else:
            stop_price = stop_ref * (1 + buffer)
            stop_dist = stop_price - price
            stop_dist = max(stop_dist, price * min_stop_pct)
            stop_dist = min(stop_dist, price * max_stop_pct)
            stop_price = round(price + stop_dist, 2)
            target_price = round(price - stop_dist * target_rr, 2)

        risk = abs(price - stop_price)
        reward = abs(target_price - price)
        rr = reward / risk if risk > 0 else 0

        # Confidence: gap size + RVOL + trend alignment + RSI position
        confidence = self._compute_confidence(
            gap_pct, rvol, ema9, ema20, rsi, direction, price, vwap)

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
            indicators={
                'gap_pct': round(gap_pct * 100, 2),
                'rvol': round(rvol, 2),
                'vwap': round(vwap, 2),
                'ema9': round(ema9, 2),
                'ema20': round(ema20, 2),
                'rsi': round(rsi, 1),
                'atr': round(atr, 4),
            },
            notes=f'Gap={gap_pct:+.1%} RVOL={rvol:.1f}x EMA9={ema9:.2f} RSI={rsi:.0f}',
        )

    def _compute_confidence(self, gap_pct: float, rvol: float,
                            ema9: float, ema20: float, rsi: float,
                            direction: str, price: float, vwap: float) -> float:
        score = 0.40  # base

        # Gap size: bigger gap = more institutional interest
        if abs(gap_pct) >= 0.03:
            score += 0.15  # 3%+ gap
        elif abs(gap_pct) >= 0.015:
            score += 0.10  # 1.5%+ gap
        elif abs(gap_pct) >= 0.008:
            score += 0.05

        # RVOL: higher = more conviction
        if rvol >= 4.0:
            score += 0.15
        elif rvol >= 2.5:
            score += 0.10
        elif rvol >= 1.5:
            score += 0.05

        # EMA spread: wider = clearer trend
        if ema9 > 0 and ema20 > 0:
            spread = abs(ema9 - ema20) / ema20
            if spread >= 0.005:
                score += 0.05
            if spread >= 0.01:
                score += 0.05

        # RSI sweet spot (45-55 = pullback zone, ideal)
        if 40 <= rsi <= 55 and direction == 'long':
            score += 0.05
        elif 45 <= rsi <= 60 and direction == 'short':
            score += 0.05

        # Price vs VWAP alignment
        if direction == 'long' and price > vwap:
            score += 0.05
        elif direction == 'short' and price < vwap:
            score += 0.05

        return min(score, 0.95)

    # ── Exit Management ───────────────────────────────────────────────

    def evaluate_exit(self, position: Dict, price: float,
                      tick_data: Optional[Dict] = None,
                      now: Optional[datetime] = None) -> Optional[ExitSignal]:
        """Exit management: VWAP breakdown (structural) + time exit.

        Key learning: EMA cross exits fire too frequently on 5-min bars,
        killing 65%+ of trades prematurely. Only exit on structural breaks
        (VWAP loss) or time. Let trailing stops handle the rest.
        """
        if not tick_data:
            return None

        direction = position.get('direction', 'long')
        vwap = tick_data.get('vwap', 0)
        entry_price = position.get('entry_price', 0)

        # VWAP breakdown: only exit on SIGNIFICANT structural failure
        # Small VWAP violations are noise — let the stop handle it
        if vwap > 0 and entry_price > 0:
            if direction == 'long' and price < vwap * 0.990 and price < entry_price * 0.995:
                return ExitSignal(
                    action='close',
                    reason=f'vwap_breakdown: price={price:.2f} < VWAP={vwap:.2f}',
                )
            elif direction == 'short' and price > vwap * 1.010 and price > entry_price * 1.005:
                return ExitSignal(
                    action='close',
                    reason=f'vwap_breakdown: price={price:.2f} > VWAP={vwap:.2f}',
                )

        # Time exit
        max_hold = self._config.get('max_hold_minutes', 120)
        entry_time = position.get('entry_time')
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
        """Trail with EMA9 after 0.5R profit — let winners run."""
        if not tick_data:
            return None

        direction = position.get('direction', 'long')
        entry_price = position.get('entry_price', 0)
        stop_price = position.get('stop_price', 0)
        ema9 = tick_data.get('ema9', 0)

        if not all([entry_price, stop_price, ema9]):
            return None

        risk = abs(entry_price - stop_price)
        if risk <= 0:
            return None

        trail_after = self._config.get('trail_after_r', 0.3)

        if direction == 'long':
            pnl_r = (price - entry_price) / risk
            if pnl_r >= 1.0:
                # After 1R: trail at EMA9
                new_stop = ema9 * 0.998
                if new_stop > stop_price:
                    return round(new_stop, 2)
            elif pnl_r >= trail_after:
                # After 0.3R: move to breakeven (protect capital early)
                if entry_price > stop_price:
                    return round(entry_price, 2)
        else:
            pnl_r = (entry_price - price) / risk
            if pnl_r >= 1.0:
                new_stop = ema9 * 1.002
                if new_stop < stop_price:
                    return round(new_stop, 2)
            elif pnl_r >= trail_after:
                if entry_price < stop_price:
                    return round(entry_price, 2)

        return None

    # ── Lifecycle ─────────────────────────────────────────────────────

    def record_entry(self, symbol: str):
        self._entered_symbols[symbol] = self._entered_symbols.get(symbol, 0) + 1

    def on_day_start(self):
        self._entered_symbols.clear()
        self._day_gaps.clear()
        self._day_rvol.clear()
        self._pullback_seen.clear()
        self._pullback_low.clear()
        self._pullback_high.clear()

    def on_day_end(self):
        self._entered_symbols.clear()
        self._day_gaps.clear()
        self._day_rvol.clear()
        self._pullback_seen.clear()
        self._pullback_low.clear()
        self._pullback_high.clear()
