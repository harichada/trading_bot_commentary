"""
Regime Detector - Market Regime Detection System
Identifies: Trending, Ranging, Volatile, Choppy markets
Auto-adjusts trading parameters based on regime

A pro trader adapts to the market - they don't force trades in unfavorable conditions.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    CHOPPY = "choppy"
    UNKNOWN = "unknown"


@dataclass
class RegimeState:
    """Current regime state with confidence and parameters"""
    regime: MarketRegime
    confidence: float  # 0.0 to 1.0
    trend_strength: float  # ADX or similar
    volatility_percentile: float  # Where current vol sits historically
    range_width: float  # ATR as % of price
    chop_score: float  # 0 = clean, 100 = maximum chop
    recommended_position_size: float  # Multiplier (0.0 to 1.5)
    recommended_strategy: str  # Which strategy type works best
    avoid_trading: bool  # True if conditions are unfavorable
    regime_age_bars: int  # How long we've been in this regime
    timestamp: datetime


@dataclass
class RegimeParameters:
    """Trading parameters adjusted for current regime"""
    min_confidence_threshold: float
    stop_loss_multiplier: float  # Multiply base stop by this
    take_profit_multiplier: float  # Multiply base target by this
    max_position_size_pct: float  # Max % of account per trade
    min_r_multiple_target: float  # Minimum acceptable R target
    allow_mean_reversion: bool
    allow_trend_following: bool
    allow_breakout: bool
    max_trades_per_day: int
    time_stop_bars: int  # Exit if no movement after N bars


class RegimeDetector:
    """
    Detects market regime and provides trading parameter adjustments.

    Philosophy: The market tells you how to trade it. Don't fight the regime.
    - Trending: Let winners run, tight stops, trend-following strategies
    - Ranging: Mean reversion, fade extremes, tighter targets
    - Volatile: Reduce size, wider stops, wait for clarity
    - Choppy: DON'T TRADE - this is where retail loses money
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

        # Lookback periods
        self.trend_lookback = self.config.get('trend_lookback', 20)
        self.volatility_lookback = self.config.get('volatility_lookback', 20)
        self.chop_lookback = self.config.get('chop_lookback', 14)
        self.regime_confirmation_bars = self.config.get('regime_confirmation_bars', 3)

        # Thresholds
        self.adx_trending_threshold = self.config.get('adx_trending_threshold', 25)
        self.adx_strong_trend_threshold = self.config.get('adx_strong_trend_threshold', 40)
        self.chop_index_threshold = self.config.get('chop_index_threshold', 61.8)  # Fibonacci
        self.volatility_high_percentile = self.config.get('volatility_high_percentile', 80)
        self.volatility_low_percentile = self.config.get('volatility_low_percentile', 20)

        # State tracking
        self.current_regime: Dict[str, RegimeState] = {}  # Per-symbol
        self.regime_history: Dict[str, List[RegimeState]] = {}
        self.volatility_history: Dict[str, List[float]] = {}  # For percentile calc

        # Regime-specific parameters
        self._setup_regime_parameters()

    def _setup_regime_parameters(self):
        """Define trading parameters for each regime"""
        self.regime_params = {
            MarketRegime.TRENDING_UP: RegimeParameters(
                min_confidence_threshold=0.65,
                stop_loss_multiplier=1.0,
                take_profit_multiplier=1.5,  # Let winners run
                max_position_size_pct=1.0,
                min_r_multiple_target=2.0,
                allow_mean_reversion=False,
                allow_trend_following=True,
                allow_breakout=True,
                max_trades_per_day=4,
                time_stop_bars=20
            ),
            MarketRegime.TRENDING_DOWN: RegimeParameters(
                min_confidence_threshold=0.70,  # Higher bar for shorts
                stop_loss_multiplier=1.0,
                take_profit_multiplier=1.3,  # Down moves faster
                max_position_size_pct=0.8,
                min_r_multiple_target=2.0,
                allow_mean_reversion=False,
                allow_trend_following=True,
                allow_breakout=True,
                max_trades_per_day=3,
                time_stop_bars=15
            ),
            MarketRegime.RANGING: RegimeParameters(
                min_confidence_threshold=0.70,
                stop_loss_multiplier=0.8,  # Tighter stops in ranges
                take_profit_multiplier=0.8,  # Take profits quicker
                max_position_size_pct=0.8,
                min_r_multiple_target=1.5,
                allow_mean_reversion=True,
                allow_trend_following=False,
                allow_breakout=False,  # Breakouts fail in ranges
                max_trades_per_day=3,
                time_stop_bars=10
            ),
            MarketRegime.VOLATILE: RegimeParameters(
                min_confidence_threshold=0.80,  # Very high bar
                stop_loss_multiplier=1.5,  # Wider stops needed
                take_profit_multiplier=1.2,
                max_position_size_pct=0.5,  # REDUCE SIZE
                min_r_multiple_target=2.5,
                allow_mean_reversion=False,
                allow_trend_following=True,
                allow_breakout=False,
                max_trades_per_day=2,
                time_stop_bars=30
            ),
            MarketRegime.CHOPPY: RegimeParameters(
                min_confidence_threshold=0.95,  # Essentially block trading
                stop_loss_multiplier=2.0,
                take_profit_multiplier=0.5,
                max_position_size_pct=0.25,
                min_r_multiple_target=3.0,  # Unrealistic = don't trade
                allow_mean_reversion=False,
                allow_trend_following=False,
                allow_breakout=False,
                max_trades_per_day=0,  # NO TRADING
                time_stop_bars=5
            ),
            MarketRegime.UNKNOWN: RegimeParameters(
                min_confidence_threshold=0.75,
                stop_loss_multiplier=1.0,
                take_profit_multiplier=1.0,
                max_position_size_pct=0.5,
                min_r_multiple_target=2.0,
                allow_mean_reversion=False,
                allow_trend_following=True,
                allow_breakout=False,
                max_trades_per_day=2,
                time_stop_bars=15
            )
        }

    def calculate_adx(self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                      period: int = 14) -> Tuple[float, float, float]:
        """
        Calculate ADX, +DI, -DI for trend strength
        ADX > 25 = trending, ADX > 40 = strong trend
        """
        if len(close) < period + 1:
            return 0.0, 0.0, 0.0

        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(np.maximum(tr1, tr2), tr3)

        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        # Smoothed averages (Wilder's smoothing)
        def wilder_smooth(data, period):
            result = np.zeros_like(data)
            result[period-1] = np.sum(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
            return result

        atr = wilder_smooth(tr, period)
        plus_dm_smooth = wilder_smooth(plus_dm, period)
        minus_dm_smooth = wilder_smooth(minus_dm, period)

        # Avoid division by zero
        atr = np.where(atr == 0, 1e-10, atr)

        plus_di = 100 * plus_dm_smooth / atr
        minus_di = 100 * minus_dm_smooth / atr

        # DX and ADX
        di_sum = plus_di + minus_di
        di_sum = np.where(di_sum == 0, 1e-10, di_sum)
        dx = 100 * np.abs(plus_di - minus_di) / di_sum

        adx = wilder_smooth(dx, period) / period

        # Return latest values
        return float(adx[-1]), float(plus_di[-1]), float(minus_di[-1])

    def calculate_choppiness_index(self, high: np.ndarray, low: np.ndarray,
                                    close: np.ndarray, period: int = 14) -> float:
        """
        Choppiness Index: 0-100
        < 38.2: Trending (market is directional)
        > 61.8: Choppy (market is consolidating/noisy)

        CI = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
        """
        if len(close) < period + 1:
            return 50.0  # Neutral

        # Calculate ATR
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(np.maximum(tr1, tr2), tr3)

        # Sum of ATR over period
        atr_sum = np.sum(tr[-period:])

        # Highest high - Lowest low over period
        highest = np.max(high[-period:])
        lowest = np.min(low[-period:])
        range_hl = highest - lowest

        if range_hl == 0:
            return 50.0

        # Choppiness Index
        ci = 100 * np.log10(atr_sum / range_hl) / np.log10(period)

        return float(np.clip(ci, 0, 100))

    def calculate_volatility_percentile(self, symbol: str, current_atr: float,
                                         price: float) -> float:
        """
        Calculate where current volatility sits relative to history.
        Returns percentile (0-100).
        """
        # Normalize ATR as % of price
        atr_pct = (current_atr / price) * 100 if price > 0 else 0

        # Initialize history if needed
        if symbol not in self.volatility_history:
            self.volatility_history[symbol] = []

        # Add current reading
        self.volatility_history[symbol].append(atr_pct)

        # Keep rolling window (100 bars)
        if len(self.volatility_history[symbol]) > 100:
            self.volatility_history[symbol] = self.volatility_history[symbol][-100:]

        # Calculate percentile
        history = self.volatility_history[symbol]
        if len(history) < 10:
            return 50.0  # Not enough data

        percentile = (np.sum(np.array(history) < atr_pct) / len(history)) * 100
        return float(percentile)

    def detect_regime(self, symbol: str, ohlcv_data: pd.DataFrame) -> RegimeState:
        """
        Main regime detection logic.

        Args:
            symbol: Trading symbol
            ohlcv_data: DataFrame with columns: open, high, low, close, volume

        Returns:
            RegimeState with current market regime and parameters
        """
        if len(ohlcv_data) < max(self.trend_lookback, self.volatility_lookback,
                                  self.chop_lookback) + 5:
            return RegimeState(
                regime=MarketRegime.UNKNOWN,
                confidence=0.0,
                trend_strength=0.0,
                volatility_percentile=50.0,
                range_width=0.0,
                chop_score=50.0,
                recommended_position_size=0.5,
                recommended_strategy="wait",
                avoid_trading=True,
                regime_age_bars=0,
                timestamp=datetime.now()
            )

        # Extract price data
        high = ohlcv_data['high'].values
        low = ohlcv_data['low'].values
        close = ohlcv_data['close'].values

        # Calculate indicators
        adx, plus_di, minus_di = self.calculate_adx(high, low, close, 14)
        chop_index = self.calculate_choppiness_index(high, low, close, 14)

        # Calculate ATR for volatility
        tr = np.maximum(
            high[-self.volatility_lookback:] - low[-self.volatility_lookback:],
            np.maximum(
                np.abs(high[-self.volatility_lookback:] - np.roll(close, 1)[-self.volatility_lookback:]),
                np.abs(low[-self.volatility_lookback:] - np.roll(close, 1)[-self.volatility_lookback:])
            )
        )
        atr = np.mean(tr)
        current_price = close[-1]

        # Volatility percentile
        vol_percentile = self.calculate_volatility_percentile(symbol, atr, current_price)

        # Range width as ATR % of price
        range_width = (atr / current_price) * 100 if current_price > 0 else 0

        # Determine regime
        regime = MarketRegime.UNKNOWN
        confidence = 0.0
        avoid_trading = False
        recommended_strategy = "wait"
        position_size_mult = 1.0

        # Decision tree for regime classification
        if chop_index > self.chop_index_threshold:
            # CHOPPY - this is where retail traders lose money
            regime = MarketRegime.CHOPPY
            confidence = min((chop_index - self.chop_index_threshold) / 20, 1.0)
            avoid_trading = True
            recommended_strategy = "NO_TRADE"
            position_size_mult = 0.0
            logger.info(f"{symbol}: CHOPPY regime detected (CI={chop_index:.1f}) - AVOID TRADING")

        elif vol_percentile > self.volatility_high_percentile:
            # HIGH VOLATILITY - reduce size, be careful
            regime = MarketRegime.VOLATILE
            confidence = min((vol_percentile - self.volatility_high_percentile) / 20, 1.0)
            avoid_trading = vol_percentile > 90  # Extreme volatility
            recommended_strategy = "trend_only_reduced"
            position_size_mult = 0.5
            logger.info(f"{symbol}: VOLATILE regime (Vol%={vol_percentile:.1f}) - REDUCE SIZE")

        elif adx > self.adx_trending_threshold:
            # TRENDING
            if plus_di > minus_di:
                regime = MarketRegime.TRENDING_UP
                recommended_strategy = "trend_following_long"
            else:
                regime = MarketRegime.TRENDING_DOWN
                recommended_strategy = "trend_following_short"

            # Confidence based on ADX strength
            confidence = min((adx - self.adx_trending_threshold) / 25, 1.0)
            position_size_mult = 1.0 if adx > self.adx_strong_trend_threshold else 0.8
            logger.info(f"{symbol}: TRENDING {regime.value} (ADX={adx:.1f}, +DI={plus_di:.1f}, -DI={minus_di:.1f})")

        else:
            # RANGING - mean reversion works here
            regime = MarketRegime.RANGING
            confidence = min((self.adx_trending_threshold - adx) / 15, 1.0)
            recommended_strategy = "mean_reversion"
            position_size_mult = 0.8

            # Check if range is tradeable (not too tight)
            if range_width < 0.5:  # Less than 0.5% range - too tight
                avoid_trading = True
                logger.info(f"{symbol}: RANGING but too tight ({range_width:.2f}%) - AVOID")
            else:
                logger.info(f"{symbol}: RANGING regime (ADX={adx:.1f}, width={range_width:.2f}%)")

        # Calculate regime age (how long in this regime)
        regime_age = 0
        if symbol in self.current_regime:
            if self.current_regime[symbol].regime == regime:
                regime_age = self.current_regime[symbol].regime_age_bars + 1

        # Create state
        state = RegimeState(
            regime=regime,
            confidence=confidence,
            trend_strength=adx,
            volatility_percentile=vol_percentile,
            range_width=range_width,
            chop_score=chop_index,
            recommended_position_size=position_size_mult,
            recommended_strategy=recommended_strategy,
            avoid_trading=avoid_trading,
            regime_age_bars=regime_age,
            timestamp=datetime.now()
        )

        # Update state tracking
        self.current_regime[symbol] = state

        # Track history
        if symbol not in self.regime_history:
            self.regime_history[symbol] = []
        self.regime_history[symbol].append(state)
        if len(self.regime_history[symbol]) > 100:
            self.regime_history[symbol] = self.regime_history[symbol][-100:]

        return state

    def get_regime_parameters(self, symbol: str) -> RegimeParameters:
        """Get trading parameters adjusted for current regime"""
        if symbol not in self.current_regime:
            return self.regime_params[MarketRegime.UNKNOWN]

        regime = self.current_regime[symbol].regime
        return self.regime_params.get(regime, self.regime_params[MarketRegime.UNKNOWN])

    def should_trade(self, symbol: str, strategy_type: str) -> Tuple[bool, str]:
        """
        Check if trading is allowed for this symbol given current regime.

        Args:
            symbol: Trading symbol
            strategy_type: One of 'trend_following', 'mean_reversion', 'breakout'

        Returns:
            (allowed, reason)
        """
        if symbol not in self.current_regime:
            return False, "No regime data available"

        state = self.current_regime[symbol]
        params = self.regime_params[state.regime]

        if state.avoid_trading:
            return False, f"Trading blocked in {state.regime.value} regime (chop={state.chop_score:.1f})"

        if params.max_trades_per_day == 0:
            return False, f"No trading allowed in {state.regime.value} regime"

        # Check strategy compatibility
        if strategy_type == 'trend_following' and not params.allow_trend_following:
            return False, f"Trend following not allowed in {state.regime.value}"

        if strategy_type == 'mean_reversion' and not params.allow_mean_reversion:
            return False, f"Mean reversion not allowed in {state.regime.value}"

        if strategy_type == 'breakout' and not params.allow_breakout:
            return False, f"Breakout strategies not allowed in {state.regime.value}"

        # Require minimum regime confidence
        if state.confidence < 0.3:
            return False, f"Regime confidence too low ({state.confidence:.2f})"

        return True, f"Trading allowed: {state.regime.value} regime"

    def get_regime_summary(self, symbol: str) -> Dict:
        """Get human-readable regime summary for commentary"""
        if symbol not in self.current_regime:
            return {"regime": "unknown", "message": "Insufficient data for regime detection"}

        state = self.current_regime[symbol]
        params = self.regime_params[state.regime]

        return {
            "symbol": symbol,
            "regime": state.regime.value,
            "confidence": f"{state.confidence:.1%}",
            "trend_strength": f"{state.trend_strength:.1f}",
            "choppiness": f"{state.chop_score:.1f}",
            "volatility_percentile": f"{state.volatility_percentile:.0f}%",
            "recommended_strategy": state.recommended_strategy,
            "position_size_multiplier": state.recommended_position_size,
            "avoid_trading": state.avoid_trading,
            "regime_age": f"{state.regime_age_bars} bars",
            "max_trades_today": params.max_trades_per_day,
            "stop_multiplier": params.stop_loss_multiplier,
            "target_multiplier": params.take_profit_multiplier,
            "message": self._get_regime_message(state)
        }

    def _get_regime_message(self, state: RegimeState) -> str:
        """Generate trading guidance message"""
        messages = {
            MarketRegime.TRENDING_UP: f"Strong uptrend detected (ADX={state.trend_strength:.1f}). Let winners run, use trend-following strategies.",
            MarketRegime.TRENDING_DOWN: f"Downtrend in progress (ADX={state.trend_strength:.1f}). Fast moves, tight management required.",
            MarketRegime.RANGING: f"Range-bound market. Fade extremes, take profits at range boundaries.",
            MarketRegime.VOLATILE: f"High volatility ({state.volatility_percentile:.0f}th percentile). REDUCE POSITION SIZE, wider stops required.",
            MarketRegime.CHOPPY: f"CHOPPY CONDITIONS (CI={state.chop_score:.1f}). This is where retail loses money. STAND ASIDE.",
            MarketRegime.UNKNOWN: "Insufficient data. Use caution and reduced size."
        }
        return messages.get(state.regime, "Unknown regime")


# Singleton instance
_regime_detector: Optional[RegimeDetector] = None

def get_regime_detector(config: Optional[Dict] = None) -> RegimeDetector:
    """Get or create singleton RegimeDetector instance"""
    global _regime_detector
    if _regime_detector is None:
        _regime_detector = RegimeDetector(config)
    return _regime_detector
