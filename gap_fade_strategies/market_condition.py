"""Market Condition Detector — classifies market state from SPY/QQQ indicators.

Classifies the market into:
- trending_up: EMA alignment bullish, RSI 50-70
- trending_down: EMA alignment bearish, RSI 30-50
- choppy: frequent EMA crosses, RSI oscillating
- sideways: narrow range, RSI 40-60

Used by the Strategy Selector (Phase 3) to pick the best intraday strategy
for current conditions.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MarketCondition:
    """Snapshot of current market conditions."""
    condition: str           # 'trending_up', 'trending_down', 'choppy', 'sideways'
    confidence: float        # 0.0 - 1.0
    spy_change_pct: float    # SPY intraday change %
    ema_alignment: str       # 'bullish', 'bearish', 'mixed'
    rsi: float               # SPY RSI
    volatility: str          # 'low', 'normal', 'high'
    timestamp: Optional[datetime] = None
    notes: str = ''


class MarketConditionDetector:
    """Detects market conditions from SPY indicator data.

    Re-evaluates every `eval_interval` seconds (default 300 = 5 min).
    Caches the last result.
    """

    def __init__(self, eval_interval: int = 300):
        self._eval_interval = eval_interval
        self._last_eval: float = 0.0
        self._last_condition: Optional[MarketCondition] = None
        self._condition_history: List[MarketCondition] = []

    @property
    def current_condition(self) -> Optional[MarketCondition]:
        return self._last_condition

    @property
    def condition_name(self) -> str:
        return self._last_condition.condition if self._last_condition else 'unknown'

    def evaluate(self, spy_tick_data: Dict, now: datetime,
                 force: bool = False) -> MarketCondition:
        """Evaluate market condition from SPY indicator data.

        Args:
            spy_tick_data: TickIndicatorEngine.get_data('SPY')
            now: current datetime
            force: bypass interval throttle

        Returns:
            MarketCondition with classification.
        """
        import time as _time
        now_mono = _time.monotonic()
        if not force and self._last_condition and (now_mono - self._last_eval < self._eval_interval):
            return self._last_condition

        self._last_eval = now_mono

        # Extract indicators
        ema9 = spy_tick_data.get('ema9', 0)
        ema20 = spy_tick_data.get('ema20', 0)
        ema50 = spy_tick_data.get('ema50', 0)
        rsi = spy_tick_data.get('rsi', 50)
        rsi_initialized = spy_tick_data.get('rsi_initialized', False)
        atr = spy_tick_data.get('atr', 0)
        day_high = spy_tick_data.get('day_high', 0)
        day_low = spy_tick_data.get('day_low', float('inf'))
        rsi_history = spy_tick_data.get('rsi_history', [])
        vwap = spy_tick_data.get('vwap', 0)

        # Default condition
        condition = 'sideways'
        confidence = 0.50
        ema_alignment = 'mixed'
        volatility = 'normal'
        notes_parts = []

        # ── EMA Alignment ──
        if ema9 > 0 and ema20 > 0:
            if ema9 > ema20:
                ema_alignment = 'bullish'
                notes_parts.append('EMA9>EMA20')
            elif ema9 < ema20:
                ema_alignment = 'bearish'
                notes_parts.append('EMA9<EMA20')

            # Check EMA50 for stronger trend confirmation
            if ema50 > 0:
                if ema9 > ema20 > ema50:
                    ema_alignment = 'bullish'
                    notes_parts.append('full bullish alignment')
                elif ema9 < ema20 < ema50:
                    ema_alignment = 'bearish'
                    notes_parts.append('full bearish alignment')

        # ── SPY Intraday Change ──
        spy_change = 0.0
        if vwap > 0 and day_high > 0:
            # Approximate using price position relative to day range
            bar_history = spy_tick_data.get('bar_history', [])
            if bar_history:
                first_close = bar_history[0].close if hasattr(bar_history[0], 'close') else 0
                last_close = bar_history[-1].close if hasattr(bar_history[-1], 'close') else 0
                if first_close > 0 and last_close > 0:
                    spy_change = (last_close - first_close) / first_close

        # ── Volatility Assessment ──
        if day_high > 0 and day_low < float('inf') and day_low > 0:
            day_range_pct = (day_high - day_low) / day_low
            if day_range_pct > 0.02:
                volatility = 'high'
                notes_parts.append(f'range {day_range_pct:.1%}')
            elif day_range_pct < 0.005:
                volatility = 'low'
                notes_parts.append(f'range {day_range_pct:.1%}')

        # ── RSI Analysis ──
        rsi_oscillating = False
        if rsi_history and len(rsi_history) >= 5:
            # Count direction changes
            changes = 0
            for i in range(1, len(rsi_history)):
                if (rsi_history[i] - rsi_history[i-1]) * (rsi_history[i-1] - rsi_history[max(0, i-2)]) < 0:
                    changes += 1
            if changes >= len(rsi_history) // 2:
                rsi_oscillating = True
                notes_parts.append('RSI oscillating')

        # ── Classification ──

        # Trending UP: bullish EMA, RSI 50-70, not oscillating
        if ema_alignment == 'bullish' and 50 <= rsi <= 70 and not rsi_oscillating:
            condition = 'trending_up'
            confidence = 0.60
            if ema50 > 0 and ema9 > ema20 > ema50:
                confidence += 0.15  # Full alignment boost
            if spy_change > 0.005:
                confidence += 0.10

        # Trending DOWN: bearish EMA, RSI 30-50, not oscillating
        elif ema_alignment == 'bearish' and 30 <= rsi <= 50 and not rsi_oscillating:
            condition = 'trending_down'
            confidence = 0.60
            if ema50 > 0 and ema9 < ema20 < ema50:
                confidence += 0.15
            if spy_change < -0.005:
                confidence += 0.10

        # CHOPPY: oscillating RSI, mixed EMAs, high volatility
        elif rsi_oscillating and volatility == 'high':
            condition = 'choppy'
            confidence = 0.55
            if ema_alignment == 'mixed':
                confidence += 0.10

        # SIDEWAYS: low volatility, RSI 40-60
        elif volatility == 'low' and 40 <= rsi <= 60:
            condition = 'sideways'
            confidence = 0.55
            notes_parts.append('low vol + neutral RSI')

        # Edge case: strong RSI extremes override
        elif rsi >= 70:
            condition = 'trending_up'
            confidence = 0.55
            notes_parts.append('RSI overbought')
        elif rsi <= 30:
            condition = 'trending_down'
            confidence = 0.55
            notes_parts.append('RSI oversold')

        # Choppy fallback: mixed signals
        elif ema_alignment == 'mixed' or rsi_oscillating:
            condition = 'choppy'
            confidence = 0.45

        confidence = min(confidence, 0.95)

        result = MarketCondition(
            condition=condition,
            confidence=confidence,
            spy_change_pct=round(spy_change, 4),
            ema_alignment=ema_alignment,
            rsi=round(rsi, 1),
            volatility=volatility,
            timestamp=now,
            notes='; '.join(notes_parts),
        )

        self._last_condition = result
        self._condition_history.append(result)
        if len(self._condition_history) > 50:
            self._condition_history = self._condition_history[-30:]

        return result

    def reset_daily(self):
        """Reset at start of new trading day."""
        self._last_condition = None
        self._last_eval = 0.0
        self._condition_history.clear()

    def get_history(self) -> List[MarketCondition]:
        """Return condition history for the day."""
        return list(self._condition_history)
