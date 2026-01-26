"""
Market Context - Key Levels, Trapped Traders & Liquidity Analysis
Identifies where the "smart money" is positioned and where stops are clustered

A pro trader thinks about WHO is on the other side of the trade.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, date, timedelta
import logging

logger = logging.getLogger(__name__)


class LevelType(Enum):
    """Types of key price levels"""
    PREVIOUS_DAY_HIGH = "pdh"
    PREVIOUS_DAY_LOW = "pdl"
    PREVIOUS_DAY_CLOSE = "pdc"
    WEEKLY_HIGH = "weekly_high"
    WEEKLY_LOW = "weekly_low"
    MONTHLY_HIGH = "monthly_high"
    MONTHLY_LOW = "monthly_low"
    VWAP = "vwap"
    ROUND_NUMBER = "round"
    SWING_HIGH = "swing_high"
    SWING_LOW = "swing_low"
    GAP_FILL = "gap_fill"
    VOLUME_PROFILE_POC = "vpoc"  # Point of Control
    VOLUME_PROFILE_VAH = "vah"   # Value Area High
    VOLUME_PROFILE_VAL = "val"   # Value Area Low


class TrapType(Enum):
    """Types of trader traps"""
    FAILED_BREAKOUT_HIGH = "failed_bo_high"
    FAILED_BREAKOUT_LOW = "failed_bo_low"
    STOP_HUNT_ABOVE = "stop_hunt_above"
    STOP_HUNT_BELOW = "stop_hunt_below"
    LIQUIDITY_GRAB = "liquidity_grab"
    FALSE_REVERSAL = "false_reversal"


class MarketControl(Enum):
    """Who is in control of the market"""
    BUYERS = "buyers"
    SELLERS = "sellers"
    BALANCED = "balanced"
    UNCERTAIN = "uncertain"


@dataclass
class KeyLevel:
    """A significant price level"""
    price: float
    level_type: LevelType
    strength: float  # 0.0 to 1.0 (how many times tested)
    last_tested: Optional[datetime]
    times_tested: int
    times_held: int
    times_broken: int
    is_major: bool  # High importance level
    notes: str = ""


@dataclass
class TrappedTraders:
    """Analysis of where traders might be trapped"""
    trap_type: TrapType
    trap_price: float
    estimated_stops: float  # Price level where stops likely clustered
    estimated_volume: float  # Relative volume that might be trapped
    time_trapped: datetime
    potential_squeeze_target: float  # Where price might go if stops trigger
    confidence: float


@dataclass
class MarketContextState:
    """Complete market context analysis"""
    symbol: str
    current_price: float
    timestamp: datetime

    # Key Levels
    key_levels: List[KeyLevel]
    nearest_resistance: Optional[KeyLevel]
    nearest_support: Optional[KeyLevel]
    distance_to_resistance_pct: float
    distance_to_support_pct: float

    # VWAP Analysis
    vwap: float
    vwap_upper_band: float  # +1 std
    vwap_lower_band: float  # -1 std
    price_vs_vwap: str  # "above", "below", "at"

    # Volume Profile
    volume_poc: float  # Point of Control
    value_area_high: float
    value_area_low: float
    in_value_area: bool

    # Market Control
    control: MarketControl
    control_strength: float
    recent_delta: float  # Buy volume - Sell volume (normalized)

    # Trapped Traders
    trapped_longs: Optional[TrappedTraders]
    trapped_shorts: Optional[TrappedTraders]
    potential_squeeze: str  # "none", "short_squeeze", "long_squeeze"

    # Trade Recommendations
    long_quality: float  # 0-100 score for going long
    short_quality: float  # 0-100 score for going short
    recommended_entry_zone: Tuple[float, float]  # (lower, upper)
    recommended_stop_zone: Tuple[float, float]
    recommendation: str


class MarketContextAnalyzer:
    """
    Analyzes market context like a professional trader.

    Philosophy:
    - Trade with the "smart money", not against it
    - Understand where stops are clustered (liquidity)
    - Failed breakouts often lead to strong moves opposite
    - VWAP is institutional gravity - respect it
    - Previous day high/low are magnets

    "Amateurs look at price. Professionals look at context."
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

        # Level detection parameters
        self.swing_lookback = self.config.get('swing_lookback', 20)
        self.round_number_interval = self.config.get('round_number_interval', 5.0)
        self.level_proximity_pct = self.config.get('level_proximity_pct', 0.3)  # 0.3% = "at level"

        # Volume profile parameters
        self.vp_lookback = self.config.get('volume_profile_lookback', 50)
        self.value_area_pct = self.config.get('value_area_pct', 0.70)  # 70% of volume

        # State tracking per symbol
        self.key_levels: Dict[str, List[KeyLevel]] = {}
        self.daily_data: Dict[str, Dict] = {}  # Previous day OHLC
        self.context_cache: Dict[str, MarketContextState] = {}

    def analyze(self, symbol: str, ohlcv_data: pd.DataFrame,
                current_price: Optional[float] = None) -> MarketContextState:
        """
        Main analysis method. Returns complete market context.

        Args:
            symbol: Trading symbol
            ohlcv_data: DataFrame with open, high, low, close, volume
            current_price: Optional override for current price

        Returns:
            MarketContextState with full analysis
        """
        if len(ohlcv_data) < 20:
            return self._empty_context(symbol, current_price or 0)

        price = current_price or float(ohlcv_data['close'].iloc[-1])

        # Calculate all key levels
        levels = self._calculate_key_levels(symbol, ohlcv_data, price)

        # Find nearest support and resistance
        supports = [l for l in levels if l.price < price]
        resistances = [l for l in levels if l.price > price]

        nearest_support = max(supports, key=lambda x: x.price) if supports else None
        nearest_resistance = min(resistances, key=lambda x: x.price) if resistances else None

        # Distance calculations
        dist_to_resistance = ((nearest_resistance.price - price) / price * 100
                             if nearest_resistance else 999)
        dist_to_support = ((price - nearest_support.price) / price * 100
                          if nearest_support else 999)

        # VWAP Analysis
        vwap, vwap_upper, vwap_lower = self._calculate_vwap(ohlcv_data)
        price_vs_vwap = "above" if price > vwap * 1.001 else ("below" if price < vwap * 0.999 else "at")

        # Volume Profile
        vpoc, vah, val = self._calculate_volume_profile(ohlcv_data)
        in_value_area = val <= price <= vah

        # Market Control Analysis
        control, control_strength, delta = self._analyze_control(ohlcv_data)

        # Trapped Trader Analysis
        trapped_longs, trapped_shorts = self._find_trapped_traders(ohlcv_data, price, levels)
        squeeze = self._detect_potential_squeeze(trapped_longs, trapped_shorts, control)

        # Trade Quality Scores
        long_quality = self._score_long_setup(
            price, nearest_support, nearest_resistance, vwap, control,
            in_value_area, trapped_shorts, dist_to_resistance
        )
        short_quality = self._score_short_setup(
            price, nearest_support, nearest_resistance, vwap, control,
            in_value_area, trapped_longs, dist_to_support
        )

        # Entry and Stop Zones
        entry_zone = self._calculate_entry_zone(price, nearest_support, nearest_resistance,
                                                 long_quality > short_quality)
        stop_zone = self._calculate_stop_zone(price, nearest_support, nearest_resistance,
                                               long_quality > short_quality)

        # Generate recommendation
        recommendation = self._generate_recommendation(
            long_quality, short_quality, control, squeeze, in_value_area, price_vs_vwap
        )

        state = MarketContextState(
            symbol=symbol,
            current_price=price,
            timestamp=datetime.now(),
            key_levels=levels,
            nearest_resistance=nearest_resistance,
            nearest_support=nearest_support,
            distance_to_resistance_pct=dist_to_resistance,
            distance_to_support_pct=dist_to_support,
            vwap=vwap,
            vwap_upper_band=vwap_upper,
            vwap_lower_band=vwap_lower,
            price_vs_vwap=price_vs_vwap,
            volume_poc=vpoc,
            value_area_high=vah,
            value_area_low=val,
            in_value_area=in_value_area,
            control=control,
            control_strength=control_strength,
            recent_delta=delta,
            trapped_longs=trapped_longs,
            trapped_shorts=trapped_shorts,
            potential_squeeze=squeeze,
            long_quality=long_quality,
            short_quality=short_quality,
            recommended_entry_zone=entry_zone,
            recommended_stop_zone=stop_zone,
            recommendation=recommendation
        )

        self.context_cache[symbol] = state
        return state

    def _calculate_key_levels(self, symbol: str, data: pd.DataFrame,
                               current_price: float) -> List[KeyLevel]:
        """Calculate all key price levels"""
        levels = []

        # Previous Day High/Low/Close
        if len(data) > 1:
            # Find where day changes (simplified - assumes daily data or uses last bar as proxy)
            yesterday_idx = -2 if len(data) > 1 else -1

            pdh = float(data['high'].iloc[yesterday_idx])
            pdl = float(data['low'].iloc[yesterday_idx])
            pdc = float(data['close'].iloc[yesterday_idx])

            levels.append(KeyLevel(pdh, LevelType.PREVIOUS_DAY_HIGH, 0.9, None, 0, 0, 0, True, "PDH"))
            levels.append(KeyLevel(pdl, LevelType.PREVIOUS_DAY_LOW, 0.9, None, 0, 0, 0, True, "PDL"))
            levels.append(KeyLevel(pdc, LevelType.PREVIOUS_DAY_CLOSE, 0.7, None, 0, 0, 0, False, "PDC"))

        # Round Numbers
        base_round = int(current_price / self.round_number_interval) * self.round_number_interval
        for offset in [-2, -1, 0, 1, 2]:
            round_level = base_round + offset * self.round_number_interval
            if round_level > 0:
                levels.append(KeyLevel(
                    round_level, LevelType.ROUND_NUMBER, 0.6, None, 0, 0, 0,
                    round_level % 10 == 0,  # $10 levels are major
                    f"Round ${round_level}"
                ))

        # Swing Highs and Lows
        swing_levels = self._find_swing_levels(data)
        levels.extend(swing_levels)

        # Weekly High/Low (if enough data)
        if len(data) >= 5:
            weekly_high = float(data['high'].iloc[-5:].max())
            weekly_low = float(data['low'].iloc[-5:].min())
            levels.append(KeyLevel(weekly_high, LevelType.WEEKLY_HIGH, 0.85, None, 0, 0, 0, True, "Weekly High"))
            levels.append(KeyLevel(weekly_low, LevelType.WEEKLY_LOW, 0.85, None, 0, 0, 0, True, "Weekly Low"))

        # Gap detection (opening gap)
        if len(data) >= 2:
            prev_close = float(data['close'].iloc[-2])
            today_open = float(data['open'].iloc[-1])
            if abs(today_open - prev_close) / prev_close > 0.005:  # 0.5% gap
                gap_fill = prev_close
                levels.append(KeyLevel(gap_fill, LevelType.GAP_FILL, 0.75, None, 0, 0, 0, True, "Gap Fill"))

        # Remove duplicates and sort
        unique_levels = self._deduplicate_levels(levels, current_price)
        return sorted(unique_levels, key=lambda x: x.price)

    def _find_swing_levels(self, data: pd.DataFrame) -> List[KeyLevel]:
        """Find swing highs and lows"""
        levels = []
        highs = data['high'].values
        lows = data['low'].values

        lookback = min(self.swing_lookback, len(data) - 2)

        for i in range(lookback, len(data) - lookback):
            # Swing High: higher than neighbors
            if highs[i] == max(highs[i-lookback:i+lookback+1]):
                levels.append(KeyLevel(
                    float(highs[i]), LevelType.SWING_HIGH, 0.7, None, 1, 0, 0, False,
                    f"Swing High @ bar {i}"
                ))

            # Swing Low: lower than neighbors
            if lows[i] == min(lows[i-lookback:i+lookback+1]):
                levels.append(KeyLevel(
                    float(lows[i]), LevelType.SWING_LOW, 0.7, None, 1, 0, 0, False,
                    f"Swing Low @ bar {i}"
                ))

        return levels

    def _deduplicate_levels(self, levels: List[KeyLevel],
                            current_price: float) -> List[KeyLevel]:
        """Remove levels that are too close together"""
        if not levels:
            return []

        # Sort by price
        sorted_levels = sorted(levels, key=lambda x: x.price)

        # Merge close levels
        unique = [sorted_levels[0]]
        min_gap = current_price * (self.level_proximity_pct / 100)

        for level in sorted_levels[1:]:
            if level.price - unique[-1].price > min_gap:
                unique.append(level)
            elif level.is_major and not unique[-1].is_major:
                # Replace with more important level
                unique[-1] = level
            elif level.strength > unique[-1].strength:
                unique[-1] = level

        return unique

    def _calculate_vwap(self, data: pd.DataFrame) -> Tuple[float, float, float]:
        """Calculate VWAP and standard deviation bands"""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        volume = data['volume']

        # Cumulative VWAP
        cumulative_tp_vol = (typical_price * volume).cumsum()
        cumulative_vol = volume.cumsum()

        vwap = float(cumulative_tp_vol.iloc[-1] / cumulative_vol.iloc[-1])

        # Standard deviation for bands
        squared_diff = ((typical_price - vwap) ** 2 * volume).cumsum()
        variance = squared_diff.iloc[-1] / cumulative_vol.iloc[-1]
        std = np.sqrt(variance)

        return vwap, vwap + std, vwap - std

    def _calculate_volume_profile(self, data: pd.DataFrame) -> Tuple[float, float, float]:
        """Calculate Volume Profile levels (POC, VAH, VAL)"""
        # Simple volume profile using price bins
        n_bins = 20
        price_range = data['close'].max() - data['close'].min()
        if price_range == 0:
            mid = float(data['close'].iloc[-1])
            return mid, mid, mid

        bin_size = price_range / n_bins
        bins = {}

        for i in range(len(data)):
            price = data['close'].iloc[i]
            vol = data['volume'].iloc[i]
            bin_idx = int((price - data['close'].min()) / bin_size)
            bin_idx = min(bin_idx, n_bins - 1)
            bins[bin_idx] = bins.get(bin_idx, 0) + vol

        if not bins:
            mid = float(data['close'].iloc[-1])
            return mid, mid, mid

        # Point of Control (highest volume bin)
        poc_bin = max(bins, key=bins.get)
        poc = float(data['close'].min() + (poc_bin + 0.5) * bin_size)

        # Value Area (70% of volume)
        total_vol = sum(bins.values())
        target_vol = total_vol * self.value_area_pct

        sorted_bins = sorted(bins.items(), key=lambda x: x[1], reverse=True)
        cumulative = 0
        va_bins = []

        for bin_idx, vol in sorted_bins:
            va_bins.append(bin_idx)
            cumulative += vol
            if cumulative >= target_vol:
                break

        vah = float(data['close'].min() + (max(va_bins) + 1) * bin_size)
        val = float(data['close'].min() + min(va_bins) * bin_size)

        return poc, vah, val

    def _analyze_control(self, data: pd.DataFrame) -> Tuple[MarketControl, float, float]:
        """Analyze who is in control of the market"""
        recent = data.tail(10)

        # Simple delta approximation
        up_volume = recent[recent['close'] > recent['open']]['volume'].sum()
        down_volume = recent[recent['close'] <= recent['open']]['volume'].sum()
        total_volume = up_volume + down_volume

        if total_volume == 0:
            return MarketControl.UNCERTAIN, 0.0, 0.0

        delta = (up_volume - down_volume) / total_volume

        # Trend analysis
        closes = recent['close'].values
        if len(closes) >= 3:
            trend = np.polyfit(range(len(closes)), closes, 1)[0]
            normalized_trend = trend / closes[-1] * 100  # As percentage
        else:
            normalized_trend = 0

        # Combine delta and trend
        if delta > 0.2 and normalized_trend > 0:
            control = MarketControl.BUYERS
            strength = min(abs(delta) + abs(normalized_trend) / 2, 1.0)
        elif delta < -0.2 and normalized_trend < 0:
            control = MarketControl.SELLERS
            strength = min(abs(delta) + abs(normalized_trend) / 2, 1.0)
        elif abs(delta) < 0.1 and abs(normalized_trend) < 0.1:
            control = MarketControl.BALANCED
            strength = 1.0 - abs(delta)
        else:
            control = MarketControl.UNCERTAIN
            strength = 0.5

        return control, strength, delta

    def _find_trapped_traders(self, data: pd.DataFrame, current_price: float,
                               levels: List[KeyLevel]) -> Tuple[Optional[TrappedTraders], Optional[TrappedTraders]]:
        """Find where traders might be trapped"""
        trapped_longs = None
        trapped_shorts = None

        recent = data.tail(20)
        recent_high = float(recent['high'].max())
        recent_low = float(recent['low'].min())

        # Check for failed breakout high (trapped longs)
        highs_near_recent_high = recent[recent['high'] > recent_high * 0.995]
        if len(highs_near_recent_high) > 0 and current_price < recent_high * 0.99:
            # We broke above and came back down - longs are trapped
            trap_price = recent_high
            stop_estimate = recent_low * 0.995  # Stops likely below recent low

            trapped_longs = TrappedTraders(
                trap_type=TrapType.FAILED_BREAKOUT_HIGH,
                trap_price=trap_price,
                estimated_stops=stop_estimate,
                estimated_volume=0.5,  # Relative estimate
                time_trapped=datetime.now(),
                potential_squeeze_target=stop_estimate * 0.98,  # Price target if stops hit
                confidence=0.7
            )

        # Check for failed breakout low (trapped shorts)
        lows_near_recent_low = recent[recent['low'] < recent_low * 1.005]
        if len(lows_near_recent_low) > 0 and current_price > recent_low * 1.01:
            # We broke below and came back up - shorts are trapped
            trap_price = recent_low
            stop_estimate = recent_high * 1.005  # Stops likely above recent high

            trapped_shorts = TrappedTraders(
                trap_type=TrapType.FAILED_BREAKOUT_LOW,
                trap_price=trap_price,
                estimated_stops=stop_estimate,
                estimated_volume=0.5,
                time_trapped=datetime.now(),
                potential_squeeze_target=stop_estimate * 1.02,
                confidence=0.7
            )

        return trapped_longs, trapped_shorts

    def _detect_potential_squeeze(self, trapped_longs: Optional[TrappedTraders],
                                   trapped_shorts: Optional[TrappedTraders],
                                   control: MarketControl) -> str:
        """Detect if a squeeze is likely"""
        if trapped_shorts and control == MarketControl.BUYERS:
            return "short_squeeze"
        if trapped_longs and control == MarketControl.SELLERS:
            return "long_squeeze"
        return "none"

    def _score_long_setup(self, price: float, support: Optional[KeyLevel],
                          resistance: Optional[KeyLevel], vwap: float,
                          control: MarketControl, in_value_area: bool,
                          trapped_shorts: Optional[TrappedTraders],
                          dist_to_resistance: float) -> float:
        """Score the quality of a long setup (0-100)"""
        score = 50.0  # Neutral start

        # Near support is good (+15)
        if support and (price - support.price) / price < 0.01:
            score += 15

        # Above VWAP is good for longs (+10)
        if price > vwap:
            score += 10
        else:
            score -= 10

        # Buyers in control (+15)
        if control == MarketControl.BUYERS:
            score += 15
        elif control == MarketControl.SELLERS:
            score -= 15

        # In value area is safer (+5)
        if in_value_area:
            score += 5

        # Trapped shorts = potential squeeze (+10)
        if trapped_shorts:
            score += 10

        # Room to resistance (+10 if >1%, -10 if <0.5%)
        if dist_to_resistance > 1:
            score += 10
        elif dist_to_resistance < 0.5:
            score -= 10

        return max(0, min(100, score))

    def _score_short_setup(self, price: float, support: Optional[KeyLevel],
                           resistance: Optional[KeyLevel], vwap: float,
                           control: MarketControl, in_value_area: bool,
                           trapped_longs: Optional[TrappedTraders],
                           dist_to_support: float) -> float:
        """Score the quality of a short setup (0-100)"""
        score = 50.0

        # Near resistance is good for shorts (+15)
        if resistance and (resistance.price - price) / price < 0.01:
            score += 15

        # Below VWAP is good for shorts (+10)
        if price < vwap:
            score += 10
        else:
            score -= 10

        # Sellers in control (+15)
        if control == MarketControl.SELLERS:
            score += 15
        elif control == MarketControl.BUYERS:
            score -= 15

        # In value area is safer (+5)
        if in_value_area:
            score += 5

        # Trapped longs = potential squeeze down (+10)
        if trapped_longs:
            score += 10

        # Room to support (+10 if >1%, -10 if <0.5%)
        if dist_to_support > 1:
            score += 10
        elif dist_to_support < 0.5:
            score -= 10

        return max(0, min(100, score))

    def _calculate_entry_zone(self, price: float, support: Optional[KeyLevel],
                               resistance: Optional[KeyLevel], favor_long: bool) -> Tuple[float, float]:
        """Calculate optimal entry zone"""
        if favor_long:
            # For longs, ideal entry near support
            lower = support.price if support else price * 0.99
            upper = price * 1.002
        else:
            # For shorts, ideal entry near resistance
            lower = price * 0.998
            upper = resistance.price if resistance else price * 1.01

        return (lower, upper)

    def _calculate_stop_zone(self, price: float, support: Optional[KeyLevel],
                              resistance: Optional[KeyLevel], favor_long: bool) -> Tuple[float, float]:
        """Calculate stop loss zone"""
        if favor_long:
            # For longs, stop below support
            if support:
                lower = support.price * 0.995
                upper = support.price * 0.998
            else:
                lower = price * 0.97
                upper = price * 0.98
        else:
            # For shorts, stop above resistance
            if resistance:
                lower = resistance.price * 1.002
                upper = resistance.price * 1.005
            else:
                lower = price * 1.02
                upper = price * 1.03

        return (lower, upper)

    def _generate_recommendation(self, long_quality: float, short_quality: float,
                                  control: MarketControl, squeeze: str,
                                  in_value_area: bool, price_vs_vwap: str) -> str:
        """Generate actionable recommendation"""
        if squeeze == "short_squeeze":
            return "POTENTIAL SHORT SQUEEZE - Trapped shorts above, buyers in control. Watch for acceleration."
        if squeeze == "long_squeeze":
            return "POTENTIAL LONG SQUEEZE - Trapped longs below, sellers in control. Watch for breakdown."

        if long_quality > 70 and short_quality < 40:
            return f"LONG BIAS - Strong support context (Quality: {long_quality:.0f}). Look for long entries."
        if short_quality > 70 and long_quality < 40:
            return f"SHORT BIAS - Strong resistance context (Quality: {short_quality:.0f}). Look for short entries."

        if control == MarketControl.BALANCED:
            return "BALANCED - No clear directional bias. Wait for breakout or fade extremes."

        if not in_value_area:
            if price_vs_vwap == "above":
                return "EXTENDED ABOVE VALUE - Price above value area and VWAP. Caution for longs, watch for mean reversion."
            else:
                return "EXTENDED BELOW VALUE - Price below value area and VWAP. Caution for shorts, watch for bounce."

        return f"NEUTRAL - Long quality: {long_quality:.0f}, Short quality: {short_quality:.0f}. Wait for clearer setup."

    def _empty_context(self, symbol: str, price: float) -> MarketContextState:
        """Return empty context when insufficient data"""
        return MarketContextState(
            symbol=symbol,
            current_price=price,
            timestamp=datetime.now(),
            key_levels=[],
            nearest_resistance=None,
            nearest_support=None,
            distance_to_resistance_pct=999,
            distance_to_support_pct=999,
            vwap=price,
            vwap_upper_band=price * 1.01,
            vwap_lower_band=price * 0.99,
            price_vs_vwap="at",
            volume_poc=price,
            value_area_high=price * 1.01,
            value_area_low=price * 0.99,
            in_value_area=True,
            control=MarketControl.UNCERTAIN,
            control_strength=0.0,
            recent_delta=0.0,
            trapped_longs=None,
            trapped_shorts=None,
            potential_squeeze="none",
            long_quality=50.0,
            short_quality=50.0,
            recommended_entry_zone=(price * 0.99, price * 1.01),
            recommended_stop_zone=(price * 0.97, price * 0.98),
            recommendation="INSUFFICIENT DATA - Need more bars for analysis"
        )

    def get_context_summary(self, symbol: str) -> Dict:
        """Get human-readable context summary for commentary"""
        if symbol not in self.context_cache:
            return {"symbol": symbol, "message": "No context data available"}

        ctx = self.context_cache[symbol]

        return {
            "symbol": symbol,
            "price": f"${ctx.current_price:.2f}",
            "vwap": f"${ctx.vwap:.2f}",
            "price_vs_vwap": ctx.price_vs_vwap,
            "nearest_support": f"${ctx.nearest_support.price:.2f}" if ctx.nearest_support else "None",
            "nearest_resistance": f"${ctx.nearest_resistance.price:.2f}" if ctx.nearest_resistance else "None",
            "distance_to_support": f"{ctx.distance_to_support_pct:.2f}%",
            "distance_to_resistance": f"{ctx.distance_to_resistance_pct:.2f}%",
            "control": ctx.control.value,
            "control_strength": f"{ctx.control_strength:.1%}",
            "in_value_area": ctx.in_value_area,
            "potential_squeeze": ctx.potential_squeeze,
            "long_quality": f"{ctx.long_quality:.0f}/100",
            "short_quality": f"{ctx.short_quality:.0f}/100",
            "recommendation": ctx.recommendation,
            "key_levels_count": len(ctx.key_levels)
        }


# Singleton instance
_market_context: Optional[MarketContextAnalyzer] = None

def get_market_context_analyzer(config: Optional[Dict] = None) -> MarketContextAnalyzer:
    """Get or create singleton MarketContextAnalyzer instance"""
    global _market_context
    if _market_context is None:
        _market_context = MarketContextAnalyzer(config)
    return _market_context
