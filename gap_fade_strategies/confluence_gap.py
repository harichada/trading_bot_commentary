"""Confluence Gap Strategy — Strategy #3 (Multi-Factor Key Level Approach).

Filters gaps by key support/resistance levels, Bollinger Band breaks,
Fibonacci retracement zones, and intraday trend change confirmation.

Key differences from VWAP strategy:
- Uses historical key levels (swing highs/lows) as primary filter
- Bollinger Band break confirmation adds conviction
- Fibonacci retracement zones identify optimal entry zones
- Intraday trend change confirmation before entry (lower highs/lows for shorts)
- ATR-based stops anchored to key levels
- Confidence scoring system (0.50 minimum, 0.70+ for premium setups)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from .base import ExitSignal, GapFadeStrategy
from .registry import GapFadeStrategyRegistry

logger = logging.getLogger(__name__)

# ── Default config ─────────────────────────────────────────────────────

_DEFAULTS = {
    'lookback_days': 126,            # ~6 months for key levels
    'swing_window': 5,               # bars each side for swing detection
    'zone_merge_pct': 0.015,         # merge levels within 1.5%
    'min_touches': 2,                # minimum touches for key level
    'max_key_levels': 10,            # cap per symbol
    'zone_proximity_pct': 0.02,      # gap must be within 2% of key level
    'bb_period': 20,
    'bb_std_dev': 2.0,
    'fib_lookback_days': 60,
    'min_confidence': 0.50,
    'premium_confidence': 0.70,
    'require_trend_confirmation': True,
    'trend_confirm_bars': 3,
    'entry_after_minute': 40,
    'entry_cutoff_hour': 11,
    'entry_cutoff_minute': 30,
    'stop_atr_multiple': 1.5,
    'stop_key_level_buffer_pct': 0.003,
    'ema_trailing_period': 9,
    'ema_trailing_buffer_pct': 0.001,
}


# ── Data structures ────────────────────────────────────────────────────

@dataclass
class KeyLevel:
    """A support or resistance level derived from swing points."""
    price: float
    level_type: str         # 'support' or 'resistance'
    touch_count: int
    last_touch_date: str
    strength: float         # 0.0-1.0


@dataclass
class ConfluenceAnalysis:
    """Cached daily analysis for a symbol."""
    symbol: str
    computed_date: str
    key_levels: List[KeyLevel]
    bb_upper: float
    bb_lower: float
    bb_middle: float
    fib_levels: Dict[str, float]    # {'0.382': price, '0.5': price, ...}
    fib_swing_high: float
    fib_swing_low: float
    atr_14: float


@dataclass
class PendingSignal:
    """A gap candidate that passed confluence filter, awaiting trend confirmation."""
    symbol: str
    confidence: float
    strategy_type: str      # 'key_level', 'premium', 'gap_fill'
    factors: Dict[str, float]
    nearest_level: KeyLevel
    direction: str
    created_at: datetime
    confirmed: bool = False
    confirm_reason: str = ''


# ── Strategy implementation ────────────────────────────────────────────

@GapFadeStrategyRegistry.register('confluence_gap')
class ConfluenceGapStrategy(GapFadeStrategy):
    """Multi-factor confluence strategy: key levels + BB + Fib + trend confirmation.

    Only enters when a gap opens near a significant support/resistance level,
    with additional confluence from Bollinger Band breaks and Fibonacci zones.
    Waits for intraday trend change confirmation (lower highs/lows for shorts)
    before committing capital.
    """

    name = 'Confluence Gap'
    description = (
        'Multi-factor key level strategy: gaps filtered by S/R levels, '
        'Bollinger Band breaks, Fibonacci zones, and intraday trend confirmation. '
        'ATR-based stops anchored to key levels. Confidence scoring 0.50-0.95.'
    )
    version = '1.0'

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self._cache: Dict[str, ConfluenceAnalysis] = {}
        self._pending: Dict[str, PendingSignal] = {}

    def get_default_config(self) -> Dict:
        return dict(_DEFAULTS)

    # ── Technical analysis helpers ─────────────────────────────────────

    def _compute_analysis(self, symbol: str, as_of_date: str) -> Optional[ConfluenceAnalysis]:
        """Compute or return cached ConfluenceAnalysis for symbol."""
        cached = self._cache.get(symbol)
        if cached and cached.computed_date == as_of_date:
            return cached

        # Lazy import to avoid circular dependency
        from gap_fade_app import get_price_db
        db = get_price_db()

        # Query ~6 months of daily bars
        lookback = self._config.get('lookback_days', 126)
        try:
            from datetime import date
            end_dt = date.fromisoformat(as_of_date)
            start_dt = end_dt - timedelta(days=int(lookback * 1.5))  # calendar days > trading days
            start = start_dt.isoformat()
        except (ValueError, TypeError):
            logger.warning(f"confluence_gap: invalid as_of_date={as_of_date}")
            return None

        df = db.get_bars(symbol, start, as_of_date)
        if df is None or len(df) < 30:
            logger.debug(f"confluence_gap: {symbol} only {len(df) if df is not None else 0} bars, need 30+")
            return None

        key_levels = self._find_key_levels(df)
        bb_upper, bb_middle, bb_lower = self._compute_bollinger(
            df['close'].values,
            period=self._config.get('bb_period', 20),
            std_dev=self._config.get('bb_std_dev', 2.0),
        )
        fib_levels, swing_high, swing_low = self._compute_fibonacci(
            df,
            lookback=self._config.get('fib_lookback_days', 60),
        )
        atr = self._compute_atr(df, period=14)

        analysis = ConfluenceAnalysis(
            symbol=symbol,
            computed_date=as_of_date,
            key_levels=key_levels,
            bb_upper=bb_upper,
            bb_lower=bb_lower,
            bb_middle=bb_middle,
            fib_levels=fib_levels,
            fib_swing_high=swing_high,
            fib_swing_low=swing_low,
            atr_14=atr,
        )
        self._cache[symbol] = analysis
        return analysis

    def _find_key_levels(self, df) -> List[KeyLevel]:
        """Detect swing highs/lows and cluster into key support/resistance levels."""
        window = self._config.get('swing_window', 5)
        zone_merge = self._config.get('zone_merge_pct', 0.015)
        min_touches = self._config.get('min_touches', 2)
        max_levels = self._config.get('max_key_levels', 10)

        highs = df['high'].values
        lows = df['low'].values
        dates = df.index

        swing_prices = []

        # Find swing highs
        for i in range(window, len(highs) - window):
            is_swing = True
            for j in range(i - window, i + window + 1):
                if j != i and highs[j] >= highs[i]:
                    is_swing = False
                    break
            if is_swing:
                swing_prices.append((highs[i], 'resistance', str(dates[i].date()) if hasattr(dates[i], 'date') else str(dates[i])))

        # Find swing lows
        for i in range(window, len(lows) - window):
            is_swing = True
            for j in range(i - window, i + window + 1):
                if j != i and lows[j] <= lows[i]:
                    is_swing = False
                    break
            if is_swing:
                swing_prices.append((lows[i], 'support', str(dates[i].date()) if hasattr(dates[i], 'date') else str(dates[i])))

        if not swing_prices:
            return []

        # Sort by price for clustering
        swing_prices.sort(key=lambda x: x[0])

        # Greedy merge: cluster prices within zone_merge_pct of each other
        zones: List[Dict] = []
        for price, ltype, dt in swing_prices:
            merged = False
            for zone in zones:
                if abs(price - zone['center']) / zone['center'] <= zone_merge:
                    # Merge into this zone
                    zone['touches'] += 1
                    zone['center'] = (zone['center'] * (zone['touches'] - 1) + price) / zone['touches']
                    zone['last_date'] = max(zone['last_date'], dt)
                    # Zone type: support if majority of touches are lows
                    if ltype == 'support':
                        zone['support_count'] += 1
                    else:
                        zone['resistance_count'] += 1
                    merged = True
                    break
            if not merged:
                zones.append({
                    'center': price,
                    'touches': 1,
                    'last_date': dt,
                    'support_count': 1 if ltype == 'support' else 0,
                    'resistance_count': 1 if ltype == 'resistance' else 0,
                })

        # Filter by min touches and build KeyLevel objects
        levels = []
        max_touches = max((z['touches'] for z in zones), default=1)
        for zone in zones:
            if zone['touches'] < min_touches:
                continue
            ltype = 'support' if zone['support_count'] >= zone['resistance_count'] else 'resistance'
            strength = min(zone['touches'] / max(max_touches, 1), 1.0)
            levels.append(KeyLevel(
                price=round(zone['center'], 2),
                level_type=ltype,
                touch_count=zone['touches'],
                last_touch_date=zone['last_date'],
                strength=strength,
            ))

        # Sort by strength (descending), cap at max_levels
        levels.sort(key=lambda lv: lv.strength, reverse=True)
        return levels[:max_levels]

    @staticmethod
    def _compute_bollinger(closes, period: int = 20, std_dev: float = 2.0) -> Tuple[float, float, float]:
        """Compute Bollinger Bands from closing prices.

        Returns (upper, middle, lower) for the most recent bar.
        """
        if len(closes) < period:
            return 0.0, 0.0, 0.0

        recent = closes[-period:]
        middle = sum(recent) / period
        variance = sum((c - middle) ** 2 for c in recent) / period
        std = variance ** 0.5
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        return round(upper, 4), round(middle, 4), round(lower, 4)

    @staticmethod
    def _compute_fibonacci(df, lookback: int = 60) -> Tuple[Dict[str, float], float, float]:
        """Compute Fibonacci retracement levels from recent swing range.

        Returns (fib_dict, swing_high, swing_low).
        """
        recent = df.tail(lookback)
        if len(recent) < 5:
            return {}, 0.0, 0.0

        swing_high = float(recent['high'].max())
        swing_low = float(recent['low'].min())
        diff = swing_high - swing_low
        if diff < 0.01:
            return {}, swing_high, swing_low

        fib_levels = {
            '0.236': round(swing_high - 0.236 * diff, 4),
            '0.382': round(swing_high - 0.382 * diff, 4),
            '0.5': round(swing_high - 0.5 * diff, 4),
            '0.618': round(swing_high - 0.618 * diff, 4),
        }
        return fib_levels, swing_high, swing_low

    @staticmethod
    def _compute_atr(df, period: int = 14) -> float:
        """Compute Average True Range."""
        if len(df) < period + 1:
            return 0.0

        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values

        true_ranges = []
        for i in range(1, len(df)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
            true_ranges.append(tr)

        if len(true_ranges) < period:
            return sum(true_ranges) / len(true_ranges) if true_ranges else 0.0

        # Simple average of last `period` true ranges
        return sum(true_ranges[-period:]) / period

    # ── Confidence scoring ─────────────────────────────────────────────

    def _compute_confidence(self, candidate: dict, analysis: ConfluenceAnalysis
                            ) -> Tuple[float, Dict[str, float], str]:
        """Score a candidate against confluence factors.

        Returns (confidence, factors_dict, strategy_type).
        """
        factors: Dict[str, float] = {}
        score = 0.0

        gap_pct = abs(candidate.get('gap_pct', 0))
        direction = candidate.get('direction', 'short')
        gap_open = candidate.get('open', candidate.get('entry_price', 0))
        prev_close = candidate.get('prev_close', 0)

        # Reference price for proximity checks (gap open or current price)
        ref_price = gap_open if gap_open > 0 else prev_close

        if ref_price <= 0:
            return 0.0, factors, 'none'

        # ── Factor 1: Gap at key level ──
        proximity_pct = self._config.get('zone_proximity_pct', 0.02)
        nearest_level = None
        nearest_dist = float('inf')

        for level in analysis.key_levels:
            dist = abs(ref_price - level.price) / ref_price
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_level = level

        if nearest_level and nearest_dist <= proximity_pct:
            factors['key_level'] = 0.35
            score += 0.35

            # Key level strength bonus
            touches = nearest_level.touch_count
            if touches >= 4:
                factors['level_strength'] = 0.10
                score += 0.10
            elif touches >= 3:
                factors['level_strength'] = 0.07
                score += 0.07
            elif touches >= 2:
                factors['level_strength'] = 0.04
                score += 0.04

        # ── Factor 2: Gap width ──
        if gap_pct >= 0.04:
            factors['gap_width'] = 0.20
            score += 0.20
        elif gap_pct >= 0.02:
            factors['gap_width'] = 0.15
            score += 0.15
        elif gap_pct >= 0.01:
            factors['gap_width'] = 0.10
            score += 0.10
        elif gap_pct >= 0.005:
            factors['gap_width'] = 0.05
            score += 0.05

        # ── Factor 3: Bollinger Band break ──
        if analysis.bb_upper > 0:
            if direction == 'short' and ref_price > analysis.bb_upper:
                factors['bb_break'] = 0.15
                score += 0.15
            elif direction == 'long' and ref_price < analysis.bb_lower:
                factors['bb_break'] = 0.15
                score += 0.15

        # ── Factor 4: Fibonacci zone ──
        if analysis.fib_levels:
            fib_382 = analysis.fib_levels.get('0.382', 0)
            fib_500 = analysis.fib_levels.get('0.5', 0)
            fib_618 = analysis.fib_levels.get('0.618', 0)

            # Golden zone: 50%-61.8% retracement
            if fib_500 > 0 and fib_618 > 0:
                in_golden = min(fib_500, fib_618) <= ref_price <= max(fib_500, fib_618)
                if in_golden:
                    factors['fib_golden_zone'] = 0.15
                    score += 0.15
                elif fib_382 > 0:
                    # Check proximity to 38.2% or 61.8%
                    near_382 = abs(ref_price - fib_382) / ref_price <= proximity_pct
                    near_618 = abs(ref_price - fib_618) / ref_price <= proximity_pct
                    if near_382 or near_618:
                        factors['fib_zone'] = 0.10
                        score += 0.10

        # ── Factor 5: False breakout detection ──
        # Gap opens beyond a key level but price is already retreating back
        if nearest_level and nearest_dist <= proximity_pct:
            if direction == 'short' and ref_price > nearest_level.price and nearest_level.level_type == 'resistance':
                # Gapped above resistance — potential false breakout
                factors['false_breakout'] = 0.10
                score += 0.10
            elif direction == 'long' and ref_price < nearest_level.price and nearest_level.level_type == 'support':
                # Gapped below support — potential false breakdown
                factors['false_breakout'] = 0.10
                score += 0.10

        # Cap at 0.95
        score = min(score, 0.95)

        # Determine strategy type
        has_bb = 'bb_break' in factors
        has_fib = 'fib_golden_zone' in factors or 'fib_zone' in factors
        premium_threshold = self._config.get('premium_confidence', 0.70)
        min_threshold = self._config.get('min_confidence', 0.50)

        if score >= premium_threshold and has_bb and has_fib:
            strategy_type = 'premium'
        elif score >= min_threshold and 'key_level' in factors:
            strategy_type = 'key_level'
        elif score >= min_threshold:
            strategy_type = 'gap_fill'
        else:
            strategy_type = 'none'

        return score, factors, strategy_type

    # ── Trend confirmation ─────────────────────────────────────────────

    def _check_trend_confirmation(self, direction: str, bars: list, price: float
                                  ) -> Tuple[bool, str]:
        """Check intraday trend change using 5-min bar history.

        For shorts: need lower highs + lower lows (bearish reversal).
        For longs: need higher highs + higher lows (bullish reversal).

        Returns (confirmed, reason).
        """
        min_bars = self._config.get('trend_confirm_bars', 3)

        if len(bars) < min_bars:
            return False, f'only {len(bars)} bars, need {min_bars}'

        recent = bars[-min_bars:]

        if direction == 'short':
            # Check for lower highs and lower lows
            lower_highs = all(
                recent[i].high < recent[i - 1].high
                for i in range(1, len(recent))
            )
            lower_lows = all(
                recent[i].low < recent[i - 1].low
                for i in range(1, len(recent))
            )

            if lower_highs and lower_lows:
                return True, f'bearish: {min_bars} bars of lower highs + lower lows'
            if lower_highs:
                # Also check if price is below the opening range low via bar data
                first_bar_low = bars[0].low if bars else 0
                if price < first_bar_low:
                    return True, f'bearish: lower highs + price below first bar low'
                return False, 'lower highs but lows not confirming'

            return False, 'no bearish trend structure'

        else:  # long
            higher_highs = all(
                recent[i].high > recent[i - 1].high
                for i in range(1, len(recent))
            )
            higher_lows = all(
                recent[i].low > recent[i - 1].low
                for i in range(1, len(recent))
            )

            if higher_highs and higher_lows:
                return True, f'bullish: {min_bars} bars of higher highs + higher lows'
            if higher_lows:
                first_bar_high = bars[0].high if bars else float('inf')
                if price > first_bar_high:
                    return True, f'bullish: higher lows + price above first bar high'
                return False, 'higher lows but highs not confirming'

            return False, 'no bullish trend structure'

    # ── Strategy hooks ─────────────────────────────────────────────────

    def filter_candidate(self, candidate: dict) -> Tuple[bool, str]:
        """Pre-filter: compute confluence analysis and score confidence."""
        symbol = candidate.get('symbol', '')
        if not symbol:
            return False, 'no symbol'

        # Determine as_of_date from candidate data
        as_of_date = candidate.get('date', '')
        if not as_of_date:
            as_of_date = datetime.now().strftime('%Y-%m-%d')

        analysis = self._compute_analysis(symbol, as_of_date)
        if analysis is None:
            return False, f'insufficient historical data for {symbol}'

        confidence, factors, strategy_type = self._compute_confidence(candidate, analysis)

        if strategy_type == 'none':
            return False, (
                f'confidence {confidence:.2f} < {self._config.get("min_confidence", 0.50):.2f} '
                f'(factors: {factors})'
            )

        # Find nearest key level for the pending signal
        direction = candidate.get('direction', 'short')
        ref_price = candidate.get('open', candidate.get('entry_price', 0))
        nearest_level = None
        if analysis.key_levels and ref_price > 0:
            nearest_level = min(
                analysis.key_levels,
                key=lambda lv: abs(ref_price - lv.price),
            )

        if nearest_level is None:
            nearest_level = KeyLevel(price=ref_price, level_type='unknown',
                                     touch_count=0, last_touch_date='', strength=0.0)

        # Store pending signal for trend confirmation in should_enter_now
        self._pending[symbol] = PendingSignal(
            symbol=symbol,
            confidence=confidence,
            strategy_type=strategy_type,
            factors=factors,
            nearest_level=nearest_level,
            direction=direction,
            created_at=datetime.now(),
        )

        factor_str = ', '.join(f'{k}={v:.2f}' for k, v in factors.items())
        return True, (
            f'{strategy_type} setup: confidence={confidence:.2f} '
            f'[{factor_str}] near {nearest_level.level_type} '
            f'${nearest_level.price:.2f} ({nearest_level.touch_count} touches)'
        )

    def score_candidate(self, candidate: dict, regime: Optional[dict] = None) -> Optional[float]:
        """Override engine score based on confluence confidence."""
        symbol = candidate.get('symbol', '')
        pending = self._pending.get(symbol)
        if pending is None:
            return None

        base_score = candidate.get('score', 50.0)

        # Scale confidence 0.50-1.0 into +10 to +40 score boost
        conf_range = pending.confidence - 0.50
        boost = 10.0 + (conf_range / 0.50) * 30.0
        boost = max(10.0, min(40.0, boost))

        # Premium setups get extra boost
        if pending.strategy_type == 'premium':
            boost += 15.0

        return base_score + boost

    # ── Entry ──────────────────────────────────────────────────────────

    def get_entry_window(self) -> Optional[Tuple[int, int, int, int]]:
        """Delayed entry at 9:40 for confirmation time."""
        entry_min = self._config.get('entry_after_minute', 40)
        cutoff_h = self._config.get('entry_cutoff_hour', 11)
        cutoff_m = self._config.get('entry_cutoff_minute', 30)
        return (9, entry_min, cutoff_h, cutoff_m)

    def should_enter_now(self, candidate: dict, price: float,
                         tick_data: Optional[dict] = None,
                         now: Optional[datetime] = None) -> Tuple[bool, str]:
        """Gate entry on intraday trend change confirmation."""
        symbol = candidate.get('symbol', '')
        pending = self._pending.get(symbol)
        if pending is None:
            return False, 'no pending confluence signal'

        # Already confirmed earlier this session
        if pending.confirmed:
            return True, f'confirmed: {pending.confirm_reason}'

        # If trend confirmation is disabled, pass through
        if not self._config.get('require_trend_confirmation', True):
            pending.confirmed = True
            pending.confirm_reason = 'trend confirmation disabled'
            return True, pending.confirm_reason

        # Check trend confirmation from bar history
        if tick_data is None:
            # No indicator data (backtest mode) — pass through
            pending.confirmed = True
            pending.confirm_reason = 'no tick data (backtest passthrough)'
            return True, pending.confirm_reason

        bars = tick_data.get('bar_history', [])
        if not bars:
            return False, 'waiting for bar history to build'

        confirmed, reason = self._check_trend_confirmation(
            pending.direction, bars, price
        )

        if confirmed:
            # Also apply trend confirmation as a factor boost
            if 'trend_confirm' not in pending.factors:
                pending.factors['trend_confirm'] = 0.15
                pending.confidence = min(pending.confidence + 0.15, 0.95)
            pending.confirmed = True
            pending.confirm_reason = reason
            return True, f'trend confirmed: {reason} (confidence={pending.confidence:.2f})'

        return False, f'awaiting trend confirmation: {reason}'

    # ── Stop & Targets ─────────────────────────────────────────────────

    def compute_stop_price(self, entry_price: float, candidate: dict,
                           tick_data: Optional[dict] = None) -> Optional[float]:
        """ATR-based stop anchored to key levels."""
        symbol = candidate.get('symbol', '')
        direction = candidate.get('direction', 'short')
        pending = self._pending.get(symbol)
        as_of_date = candidate.get('date', datetime.now().strftime('%Y-%m-%d'))
        analysis = self._cache.get(symbol)

        if analysis is None:
            analysis = self._compute_analysis(symbol, as_of_date)
        if analysis is None or analysis.atr_14 <= 0:
            return None

        atr_mult = self._config.get('stop_atr_multiple', 1.5)
        buffer_pct = self._config.get('stop_key_level_buffer_pct', 0.003)

        # Base ATR stop
        if direction == 'short':
            atr_stop = round(entry_price + analysis.atr_14 * atr_mult, 2)
        else:
            atr_stop = round(entry_price - analysis.atr_14 * atr_mult, 2)

        # Check if a key level provides a tighter stop
        if pending and pending.nearest_level.price > 0:
            level_price = pending.nearest_level.price
            if direction == 'short':
                # Stop above the resistance level
                level_stop = round(level_price * (1 + buffer_pct), 2)
                # Use tighter of ATR stop and level stop (lower value for shorts)
                if level_stop > entry_price:
                    return min(atr_stop, level_stop)
            else:
                # Stop below the support level
                level_stop = round(level_price * (1 - buffer_pct), 2)
                # Use tighter of ATR stop and level stop (higher value for longs)
                if level_stop < entry_price:
                    return max(atr_stop, level_stop)

        return atr_stop

    def compute_targets(self, entry_price: float, candidate: dict,
                        tick_data: Optional[dict] = None) -> Optional[Tuple[float, float]]:
        """R:R-based targets with key level awareness.

        Target 1 (partial): prev_close (gap fill = 1R)
        Target 2 (full): next key level or 2R, whichever is closer to entry
        """
        direction = candidate.get('direction', 'short')
        prev_close = candidate.get('prev_close', 0)
        symbol = candidate.get('symbol', '')

        stop = self.compute_stop_price(entry_price, candidate, tick_data)
        if stop is None:
            return None

        risk = abs(entry_price - stop)
        if risk < 0.01:
            return None

        if direction == 'short':
            # Target 1: gap fill (prev close) or 1R
            half_target = prev_close if prev_close > 0 and prev_close < entry_price else entry_price - risk

            # Target 2: next support level or 2R
            full_target_rr = entry_price - 2.0 * risk
            next_level_target = self._find_next_level_target(symbol, entry_price, direction)
            if next_level_target and next_level_target < entry_price:
                full_target = max(full_target_rr, next_level_target)  # closer to entry
            else:
                full_target = full_target_rr

            # Ensure ordering: half_target >= full_target for shorts
            if half_target < full_target:
                half_target, full_target = full_target, half_target

        else:  # long
            half_target = prev_close if prev_close > 0 and prev_close > entry_price else entry_price + risk

            full_target_rr = entry_price + 2.0 * risk
            next_level_target = self._find_next_level_target(symbol, entry_price, direction)
            if next_level_target and next_level_target > entry_price:
                full_target = min(full_target_rr, next_level_target)
            else:
                full_target = full_target_rr

            # Ensure ordering: half_target <= full_target for longs
            if half_target > full_target:
                half_target, full_target = full_target, half_target

        return round(half_target, 2), round(full_target, 2)

    def _find_next_level_target(self, symbol: str, entry_price: float,
                                direction: str) -> Optional[float]:
        """Find the next key level in the profit direction."""
        analysis = self._cache.get(symbol)
        if analysis is None:
            return None

        if direction == 'short':
            # Look for support levels below entry
            support_levels = [
                lv.price for lv in analysis.key_levels
                if lv.price < entry_price and lv.level_type == 'support'
            ]
            return max(support_levels) if support_levels else None
        else:
            # Look for resistance levels above entry
            resistance_levels = [
                lv.price for lv in analysis.key_levels
                if lv.price > entry_price and lv.level_type == 'resistance'
            ]
            return min(resistance_levels) if resistance_levels else None

    # ── Exit Management ────────────────────────────────────────────────

    def evaluate_exit(self, position: dict, price: float, high: float,
                      tick_data: Optional[dict] = None,
                      now: Optional[datetime] = None) -> Optional[ExitSignal]:
        """Key level invalidation: close if price reclaims the level zone."""
        symbol = position.get('symbol', '')
        direction = position.get('direction', 'short')
        pending = self._pending.get(symbol)

        if pending is None:
            return None

        remaining = position.get('remaining_shares', 0)
        if remaining <= 0:
            return None

        # Check key level invalidation
        level_price = pending.nearest_level.price
        if level_price <= 0:
            return None

        buffer = self._config.get('stop_key_level_buffer_pct', 0.003)

        if direction == 'short':
            # If price reclaims above the resistance level + buffer, the breakout is real
            invalidation_price = level_price * (1 + buffer)
            if price > invalidation_price:
                return ExitSignal(
                    action='close',
                    reason=(
                        f'key_level_invalidation: price ${price:.2f} reclaimed '
                        f'resistance ${level_price:.2f}'
                    ),
                    shares=remaining,
                )
        else:
            # If price drops back below support level - buffer, breakdown is real
            invalidation_price = level_price * (1 - buffer)
            if price < invalidation_price:
                return ExitSignal(
                    action='close',
                    reason=(
                        f'key_level_invalidation: price ${price:.2f} broke '
                        f'support ${level_price:.2f}'
                    ),
                    shares=remaining,
                )

        # EMA cross exit (after partial fill, mirrors VWAP strategy)
        if tick_data and position.get('partial_filled', False):
            ema_period = self._config.get('ema_trailing_period', 9)
            ema_key = f'ema{ema_period}'
            ema_val = tick_data.get(ema_key, 0)
            ema_init = tick_data.get('ema_initialized', False)

            if ema_init and ema_val > 0:
                if direction == 'short' and price > ema_val:
                    return ExitSignal(
                        action='close',
                        reason=f'ema_cross (price ${price:.2f} > EMA ${ema_val:.2f})',
                        shares=remaining,
                    )
                elif direction == 'long' and price < ema_val:
                    return ExitSignal(
                        action='close',
                        reason=f'ema_cross (price ${price:.2f} < EMA ${ema_val:.2f})',
                        shares=remaining,
                    )

        return None

    def update_trailing_stop(self, position: dict, price: float,
                             tick_data: Optional[dict] = None,
                             now: Optional[datetime] = None) -> Optional[float]:
        """9 EMA trailing stop on 5-min candles + buffer (mirrors VWAP pattern)."""
        if tick_data is None:
            return None

        direction = position.get('direction', 'short')
        ema_init = tick_data.get('ema_initialized', False)

        if not ema_init:
            return None

        # Only start trailing after partial fill
        if not position.get('partial_filled', False):
            return None

        if direction == 'short':
            ema_stop = tick_data.get('ema_stop_short')
            if ema_stop and ema_stop > 0:
                current_stop = position.get('stop_price', float('inf'))
                if ema_stop < current_stop:
                    return round(ema_stop, 2)
        else:
            ema_stop = tick_data.get('ema_stop_long')
            if ema_stop and ema_stop > 0:
                current_stop = position.get('stop_price', 0)
                if ema_stop > current_stop:
                    return round(ema_stop, 2)

        return None

    # ── Indicators ─────────────────────────────────────────────────────

    def get_required_indicators(self) -> List[str]:
        return ['vwap', 'ema', 'opening_range', 'day_high', 'day_low', 'bar_history']

    # ── LLM Prompt Overrides ───────────────────────────────────────────

    def get_llm_system_prompt(self) -> Optional[str]:
        return (
            "You are an autonomous trading supervisor for a confluence-based gap fade strategy.\n"
            "The bot fades gaps that occur near key support/resistance levels, confirmed by\n"
            "Bollinger Band breaks, Fibonacci retracement zones, and intraday trend changes.\n\n"
            "Key strategy facts:\n"
            "- Gaps are only faded when they open within 2% of a historically significant S/R level.\n"
            "- Bollinger Band breaks add +15% conviction; Fibonacci golden zone adds +15%.\n"
            "- Entry is delayed to 9:40 AM and requires intraday trend change confirmation\n"
            "  (lower highs/lows for shorts, higher highs/lows for longs on 5-min bars).\n"
            "- Premium setups (confidence >= 0.70 with BB + Fib) get larger position sizing.\n"
            "- ATR-based stops are anchored to the nearest key level for structure.\n"
            "- Targets: gap fill (1R partial), next key level or 2R (full cover).\n"
            "- 9 EMA trailing stop on 5-min candles after partial fill.\n\n"
            "Your responsibilities:\n"
            "1. CANDIDATE SELECTION: Only approve candidates with confidence >= 0.50.\n"
            "2. ENTRY TIMING: Wait for trend confirmation on 5-min bars.\n"
            "3. RISK MANAGEMENT: ATR-based stops mean risk scales with volatility.\n"
            "4. PREMIUM SETUPS: Flag and prioritize when BB + Fib + key level align.\n"
            "5. After 11:30 AM ET, do NOT enter new positions.\n\n"
            "Respond ONLY with valid JSON. No text outside the JSON object."
        )

    def get_llm_profit_prompt(self) -> Optional[str]:
        return (
            "You are a profit-taking advisor for a confluence gap fade strategy.\n"
            "The bot uses key S/R levels, Bollinger Bands, Fibonacci zones, and 9 EMA.\n\n"
            "You evaluate open positions that are IN PROFIT and decide whether to:\n"
            "- CLOSE: Take profit when price reaches a key level target or EMA cross.\n"
            "- HOLD: Let it run. DEFAULT bias. Key levels provide strong reversal zones.\n"
            "- TIGHTEN_STOP: Use 9 EMA or next key level as trailing stop.\n\n"
            "CRITICAL PRINCIPLES:\n"
            "- YOUR DEFAULT SHOULD BE HOLD. The 9 EMA trailing stop handles exits.\n"
            "- If price is between key levels, the trade thesis is intact — HOLD.\n"
            "- If price approaches the next key level, TIGHTEN to that level.\n"
            "- If price crosses above 9 EMA on volume (shorts), CLOSE.\n"
            "- Gap fill < 60%: HOLD.\n"
            "- Gap fill 60-90%: TIGHTEN_STOP to EMA or next level.\n"
            "- Gap fill > 90%: may CLOSE — target nearly reached.\n\n"
            "Respond ONLY with valid JSON. No text outside the JSON object."
        )

    # ── UI / Config ────────────────────────────────────────────────────

    def get_parameter_schema(self) -> Dict:
        return {
            'lookback_days': {
                'type': 'int', 'label': 'Key Level Lookback (days)',
                'default': 126, 'min': 30, 'max': 252,
                'description': 'Trading days of history for finding key S/R levels'
            },
            'swing_window': {
                'type': 'int', 'label': 'Swing Window',
                'default': 5, 'min': 2, 'max': 15,
                'description': 'Bars each side for swing high/low detection'
            },
            'zone_merge_pct': {
                'type': 'float', 'label': 'Zone Merge %',
                'default': 0.015, 'min': 0.005, 'max': 0.05,
                'description': 'Merge key levels within this % of each other'
            },
            'min_touches': {
                'type': 'int', 'label': 'Min Touches',
                'default': 2, 'min': 1, 'max': 10,
                'description': 'Minimum touches to qualify as a key level'
            },
            'zone_proximity_pct': {
                'type': 'float', 'label': 'Gap-to-Level Proximity %',
                'default': 0.02, 'min': 0.005, 'max': 0.05,
                'description': 'Gap must open within this % of a key level'
            },
            'bb_period': {
                'type': 'int', 'label': 'Bollinger Period',
                'default': 20, 'min': 10, 'max': 50,
                'description': 'Bollinger Band lookback period'
            },
            'bb_std_dev': {
                'type': 'float', 'label': 'Bollinger Std Dev',
                'default': 2.0, 'min': 1.0, 'max': 3.0,
                'description': 'Number of standard deviations for Bollinger Bands'
            },
            'fib_lookback_days': {
                'type': 'int', 'label': 'Fib Lookback (days)',
                'default': 60, 'min': 20, 'max': 126,
                'description': 'Days to find swing high/low for Fibonacci retracements'
            },
            'min_confidence': {
                'type': 'float', 'label': 'Min Confidence',
                'default': 0.50, 'min': 0.30, 'max': 0.80,
                'description': 'Minimum confidence score to generate a signal'
            },
            'premium_confidence': {
                'type': 'float', 'label': 'Premium Threshold',
                'default': 0.70, 'min': 0.50, 'max': 0.95,
                'description': 'Confidence threshold for premium setup classification'
            },
            'require_trend_confirmation': {
                'type': 'bool', 'label': 'Require Trend Confirmation',
                'default': True,
                'description': 'Wait for intraday trend change on 5-min bars before entry'
            },
            'trend_confirm_bars': {
                'type': 'int', 'label': 'Trend Confirm Bars',
                'default': 3, 'min': 2, 'max': 8,
                'description': 'Consecutive 5-min bars needed to confirm trend change'
            },
            'entry_after_minute': {
                'type': 'int', 'label': 'Entry After Minute',
                'default': 40, 'min': 31, 'max': 59,
                'description': 'Minute past 9:XX to start entries (e.g. 40 = 9:40 AM)'
            },
            'stop_atr_multiple': {
                'type': 'float', 'label': 'Stop ATR Multiple',
                'default': 1.5, 'min': 0.5, 'max': 3.0,
                'description': 'ATR multiple for stop distance from entry'
            },
            'stop_key_level_buffer_pct': {
                'type': 'float', 'label': 'Stop Level Buffer %',
                'default': 0.003, 'min': 0.001, 'max': 0.01,
                'description': 'Buffer added beyond key level for stop placement'
            },
            'ema_trailing_period': {
                'type': 'int', 'label': 'EMA Trailing Period',
                'default': 9, 'min': 3, 'max': 50,
                'description': 'EMA period for trailing stop (on 5-min candles)'
            },
        }
