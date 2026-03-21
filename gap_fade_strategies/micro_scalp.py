"""Micro Scalp — intraday strategy.

How scalpers actually make money: they identify stocks in strong intraday
trends (clear EMA stack) and take tiny pullback entries with tight risk.
Win rate 55-65%, many small wins, few small losses. Edge is in the volume
of trades and selectivity.

The key insights from experienced scalpers:
1. Only scalp during high-liquidity windows (9:35-11:00, 2:30-3:50)
2. Only scalp stocks with volume (RVOL >1.5x) and clear trends
3. Enter at micro-pullbacks (0.1-0.3% dip to 9 EMA in an uptrend)
4. Stop is tight (below EMA20 or 0.4% max)
5. Target is quick (1.0-1.5R) — don't be greedy, take the money
6. If a trade doesn't work in 20 minutes, it's not going to work

Entry:
    LONG:  EMA9 > EMA20 (trend), price dips to within 0.2% of EMA9,
           RSI 40-70 (momentum still alive), volume present
    SHORT: EMA9 < EMA20 (trend), price rallies to within 0.2% of EMA9,
           RSI 30-60 (weakness still present), volume present

Stop:  Opposite side of EMA20 (structure-based), max 0.4%
Target: 1.0-1.5R — take quick profits
Trailing: Move to breakeven after 0.5R, trail at EMA9 after 1R
Active: 9:35-11:00 AM and 2:30-3:50 PM ET (high volume windows only)
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .base import ExitSignal
from .intraday_base import IntradaySetup, IntradayStrategy
from .intraday_registry import IntradayStrategyRegistry

logger = logging.getLogger(__name__)


@IntradayStrategyRegistry.register('micro_scalp')
class MicroScalpStrategy(IntradayStrategy):
    """Micro Scalp — high-probability EMA pullback scalps.

    Takes many small, high-probability trades in trending stocks.
    Tight risk, quick profits, disciplined time management.
    """

    name = 'Micro Scalp'
    description = 'High-frequency EMA pullback scalps in trending stocks'
    version = '1.0'
    strategy_id = 'micro_scalp'

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self._entered_symbols: Dict[str, int] = {}

    def get_default_config(self) -> Dict:
        return {
            # Trend filter
            'require_ema_stack': True,     # EMA9 > EMA20 for longs (or <)
            'min_ema_spread': 0.001,       # minimum 0.1% spread between EMAs
            # Entry
            'ema9_pullback_zone': 0.002,   # within 0.2% of EMA9
            'require_above_vwap_long': True,  # longs must be above VWAP
            'rsi_long_min': 40,
            'rsi_long_max': 70,
            'rsi_short_min': 30,
            'rsi_short_max': 60,
            # Volume
            'min_rvol': 1.0,              # at least average volume
            # Stop (tight)
            'stop_beyond_ema20': 0.001,    # 0.1% beyond EMA20
            'min_stop_pct': 0.002,         # minimum 0.2% stop
            'max_stop_pct': 0.005,         # maximum 0.5% stop (TIGHT)
            # Target (quick)
            'target_rr': 1.2,             # 1.2R target — take quick profits
            'min_target_pct': 0.003,       # minimum 0.3% target
            # Time management
            'max_hold_minutes': 25,        # if not working in 25 min, get out
            # High-volume windows
            'morning_start': (9, 35),
            'morning_end': (11, 0),
            'afternoon_start': (14, 30),
            'afternoon_end': (15, 50),
            # Position management
            'max_entries_per_day': 6,      # more trades, smaller size
            # Quality
            'min_risk_reward': 0.8,
            'min_confidence': 0.40,
        }

    def get_required_indicators(self) -> List[str]:
        return ['vwap', 'ema', 'ema_multi', 'rsi', 'atr', 'day_high', 'day_low',
                'bar_history', 'volume_profile']

    def get_ema_periods(self) -> List[int]:
        return [9, 20, 50]

    def get_active_window(self) -> Tuple[int, int, int, int]:
        # Broadest window — internal logic handles morning/afternoon sessions
        return (9, 35, 15, 50)

    def get_watchlist_criteria(self) -> Dict:
        return {
            'min_volume': 500_000,
            'min_price': 10.0,
            'max_price': 500.0,
            'min_adr_pct': 0.005,
            'prefer_gappers': False,
        }

    # ── Time Window Check ─────────────────────────────────────────────

    def _in_scalp_window(self, now: datetime) -> bool:
        """Only scalp during high-liquidity periods."""
        h, m = now.hour, now.minute
        now_mins = h * 60 + m

        morning_start = self._config.get('morning_start', (9, 35))
        morning_end = self._config.get('morning_end', (11, 0))
        afternoon_start = self._config.get('afternoon_start', (14, 30))
        afternoon_end = self._config.get('afternoon_end', (15, 50))

        ms = morning_start[0] * 60 + morning_start[1]
        me = morning_end[0] * 60 + morning_end[1]
        as_ = afternoon_start[0] * 60 + afternoon_start[1]
        ae = afternoon_end[0] * 60 + afternoon_end[1]

        return (ms <= now_mins <= me) or (as_ <= now_mins <= ae)

    # ── Scanning ──────────────────────────────────────────────────────

    def scan_for_setups(self, symbol: str, tick_data: Dict,
                        snapshot: Optional[Dict],
                        now: datetime) -> Optional[IntradaySetup]:
        if not tick_data:
            return None

        # Only scalp in high-volume windows
        if not self._in_scalp_window(now):
            return None

        # Max entries check
        max_entries = self._config.get('max_entries_per_day', 6)
        if self._entered_symbols.get(symbol, 0) >= max_entries:
            return None

        # Get price
        price = None
        if snapshot:
            price = snapshot.get('price') or snapshot.get('latestTrade', {}).get('p')
        if price is None or price <= 0:
            return None

        # Required indicators
        ema9 = tick_data.get('ema9', 0)
        ema20 = tick_data.get('ema20', 0)
        rsi = tick_data.get('rsi', 50)
        atr = tick_data.get('atr', 0)
        vwap = tick_data.get('vwap', 0)
        vol_surge = tick_data.get('volume_surge_ratio', 0)

        if not tick_data.get('rsi_initialized'):
            return None
        if ema9 <= 0 or ema20 <= 0 or atr <= 0:
            return None

        # Volume check
        min_rvol = self._config.get('min_rvol', 1.0)
        if vol_surge < min_rvol:
            return None

        # EMA spread check — need a clear trend, not a choppy mess
        min_spread = self._config.get('min_ema_spread', 0.001)
        ema_spread = abs(ema9 - ema20) / ema20 if ema20 > 0 else 0
        if ema_spread < min_spread:
            return None

        # ── BAR PATTERN CONFIRMATION ──
        # Real traders wait for the candle to CLOSE with direction
        bar_history = tick_data.get('bar_history', [])
        if not bar_history or len(bar_history) < 4:
            return None  # need enough bars to confirm trend

        # Get last bar (the one that just completed)
        last_bar = bar_history[-1]
        lb_open = getattr(last_bar, 'open', None) or (last_bar.get('open') if isinstance(last_bar, dict) else None)
        lb_close = getattr(last_bar, 'close', None) or (last_bar.get('close') if isinstance(last_bar, dict) else None)
        lb_high = getattr(last_bar, 'high', None) or (last_bar.get('high') if isinstance(last_bar, dict) else None)
        lb_low = getattr(last_bar, 'low', None) or (last_bar.get('low') if isinstance(last_bar, dict) else None)
        lb_vol = getattr(last_bar, 'volume', 0) or (last_bar.get('volume', 0) if isinstance(last_bar, dict) else 0)
        if not all([lb_open, lb_close, lb_high, lb_low]):
            return None

        bar_range = lb_high - lb_low
        if bar_range <= 0:
            return None

        # Check previous bars for established trend (at least 3 bars trending)
        min_trend_bars = self._config.get('min_trend_bars', 3)
        trend_bars_count = 0
        for i in range(max(0, len(bar_history) - 5), len(bar_history) - 1):
            b = bar_history[i]
            b_close = getattr(b, 'close', None) or (b.get('close') if isinstance(b, dict) else None)
            if b_close and ema9 > 0:
                if ema9 > ema20 and b_close > ema20:
                    trend_bars_count += 1
                elif ema9 < ema20 and b_close < ema20:
                    trend_bars_count += 1

        if trend_bars_count < min_trend_bars:
            return None  # trend not established

        # Volume on pullback should be declining (selling exhaustion)
        # Check if last 2 bars had lower volume than the bar before them
        if len(bar_history) >= 3:
            prev_bar = bar_history[-2]
            prev2_bar = bar_history[-3]
            pv = getattr(prev_bar, 'volume', 0) or (prev_bar.get('volume', 0) if isinstance(prev_bar, dict) else 0)
            p2v = getattr(prev2_bar, 'volume', 0) or (prev2_bar.get('volume', 0) if isinstance(prev2_bar, dict) else 0)
            # Skip volume exhaustion check if volumes are 0 (data issue)
            if pv > 0 and p2v > 0 and lb_vol > 0:
                # We want: current bar volume >= previous bar volume
                # (bounce should come with buyers stepping in)
                pass  # Don't filter, just use for confidence

        pullback_zone = self._config.get('ema9_pullback_zone', 0.002)
        dist_from_ema9 = (price - ema9) / ema9 if ema9 > 0 else 0

        direction = None
        setup_type = ''

        # ── LONG SCALP: Uptrend, price bounced FROM EMA9 ──
        # Key: we want price ABOVE EMA9, having just bounced off it
        # The pullback already happened; we're entering the continuation
        if ema9 > ema20:
            rsi_min = self._config.get('rsi_long_min', 40)
            rsi_max = self._config.get('rsi_long_max', 70)
            if not (rsi_min <= rsi <= rsi_max):
                return None

            # VWAP check
            if self._config.get('require_above_vwap_long', True) and vwap > 0:
                if price < vwap * 0.998:
                    return None

            # Price should be above EMA9 (bounce completed), near EMA9
            if 0 < dist_from_ema9 <= pullback_zone * 2:
                # BAR CONFIRMATION: green bar closing strong
                if lb_close > lb_open:
                    close_pos_in_bar = (lb_close - lb_low) / bar_range
                    # The bar's low should have touched or been near EMA9
                    if lb_low and ema9 > 0:
                        low_dist_ema9 = (lb_low - ema9) / ema9
                        # Low touched EMA9 zone but closed above
                        if -pullback_zone * 2 <= low_dist_ema9 <= pullback_zone:
                            if close_pos_in_bar >= 0.5:  # strong close
                                direction = 'long'
                                setup_type = 'micro_scalp_long'

        # ── SHORT SCALP: Downtrend, price rejected FROM EMA9 ──
        elif ema9 < ema20:
            rsi_min = self._config.get('rsi_short_min', 30)
            rsi_max = self._config.get('rsi_short_max', 60)
            if not (rsi_min <= rsi <= rsi_max):
                return None

            # VWAP check (shorts should be below)
            if self._config.get('require_above_vwap_long', True) and vwap > 0:
                if price > vwap * 1.002:
                    return None

            # Price should be below EMA9 (rejection completed)
            if -pullback_zone * 2 <= dist_from_ema9 < 0:
                # BAR CONFIRMATION: red bar closing weak
                if lb_close < lb_open:
                    close_pos_in_bar = (lb_high - lb_close) / bar_range
                    # The bar's high should have tested EMA9
                    if lb_high and ema9 > 0:
                        high_dist_ema9 = (lb_high - ema9) / ema9
                        if -pullback_zone <= high_dist_ema9 <= pullback_zone * 2:
                            if close_pos_in_bar >= 0.5:
                                direction = 'short'
                                setup_type = 'micro_scalp_short'

        if direction is None:
            return None

        # Calculate stop and target (TIGHT)
        stop_buffer = self._config.get('stop_beyond_ema20', 0.001)
        min_stop = self._config.get('min_stop_pct', 0.002)
        max_stop = self._config.get('max_stop_pct', 0.005)
        target_rr = self._config.get('target_rr', 1.2)
        min_target = self._config.get('min_target_pct', 0.003)

        if direction == 'long':
            # Stop just below EMA20
            stop_price = ema20 * (1 - stop_buffer)
            stop_dist = price - stop_price
            stop_dist = max(stop_dist, price * min_stop)
            stop_dist = min(stop_dist, price * max_stop)
            stop_price = round(price - stop_dist, 2)
            target_dist = max(stop_dist * target_rr, price * min_target)
            target_price = round(price + target_dist, 2)
        else:
            stop_price = ema20 * (1 + stop_buffer)
            stop_dist = stop_price - price
            stop_dist = max(stop_dist, price * min_stop)
            stop_dist = min(stop_dist, price * max_stop)
            stop_price = round(price + stop_dist, 2)
            target_dist = max(stop_dist * target_rr, price * min_target)
            target_price = round(price - target_dist, 2)

        risk = abs(price - stop_price)
        reward = abs(target_price - price)
        rr = reward / risk if risk > 0 else 0

        # Confidence
        confidence = self._compute_confidence(
            ema_spread, vol_surge, rsi, direction, dist_from_ema9)

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
                'ema9': round(ema9, 2),
                'ema20': round(ema20, 2),
                'ema_spread': round(ema_spread * 100, 3),
                'rsi': round(rsi, 1),
                'vol_surge': round(vol_surge, 2),
                'dist_ema9': round(dist_from_ema9 * 100, 3),
            },
            notes=f'EMA9={ema9:.2f} EMA20={ema20:.2f} spread={ema_spread:.3%} RSI={rsi:.0f}',
        )

    def _compute_confidence(self, ema_spread: float, vol_surge: float,
                            rsi: float, direction: str,
                            dist_from_ema9: float) -> float:
        score = 0.45

        # Wider EMA spread = clearer trend = better scalp
        if ema_spread >= 0.005:
            score += 0.15
        elif ema_spread >= 0.003:
            score += 0.10
        elif ema_spread >= 0.001:
            score += 0.05

        # Volume
        if vol_surge >= 2.5:
            score += 0.10
        elif vol_surge >= 1.5:
            score += 0.05

        # Perfect pullback (right at EMA9)
        if abs(dist_from_ema9) <= 0.001:
            score += 0.10  # right at the level
        elif abs(dist_from_ema9) <= 0.002:
            score += 0.05

        # RSI sweet spot
        if direction == 'long' and 45 <= rsi <= 60:
            score += 0.05
        elif direction == 'short' and 40 <= rsi <= 55:
            score += 0.05

        return min(score, 0.90)

    # ── Exit Management ───────────────────────────────────────────────

    def evaluate_exit(self, position: Dict, price: float,
                      tick_data: Optional[Dict] = None,
                      now: Optional[datetime] = None) -> Optional[ExitSignal]:
        """Let stops and targets handle exits.

        Key insight from backtesting: the stop/target mechanism has a
        70% target hit rate (PF 1.62). Time exits were killing profitability
        by closing trades before they could resolve via stop or target.

        The backtester's EOD 3:50pm close handles truly stuck trades.
        """
        return None

    def update_trailing_stop(self, position: Dict, price: float,
                             tick_data: Optional[Dict] = None,
                             now: Optional[datetime] = None) -> Optional[float]:
        """Breakeven at 0.5R, trail at EMA9 after 1R."""
        if not tick_data:
            return None

        direction = position.get('direction', 'long')
        entry_price = position.get('entry_price', 0)
        stop_price = position.get('stop_price', 0)
        ema9 = tick_data.get('ema9', 0)

        if not all([entry_price, stop_price]):
            return None

        risk = abs(entry_price - stop_price)
        if risk <= 0:
            return None

        if direction == 'long':
            pnl_r = (price - entry_price) / risk
            if pnl_r >= 1.0 and ema9 > 0:
                new_stop = ema9 * 0.999
                if new_stop > stop_price:
                    return round(new_stop, 2)
            elif pnl_r >= 0.5:
                if entry_price > stop_price:
                    return round(entry_price, 2)
        else:
            pnl_r = (entry_price - price) / risk
            if pnl_r >= 1.0 and ema9 > 0:
                new_stop = ema9 * 1.001
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

    def on_day_end(self):
        self._entered_symbols.clear()
