"""Strategy Selector — maps market conditions to optimal intraday strategies.

Uses MarketConditionDetector output to activate/deactivate strategies:
- Trending up/down → Momentum + Pullback
- Choppy → ORB only (reduced size)
- Sideways → Range
- ORB always available in first 2 hours
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Set

from .market_condition import MarketCondition, MarketConditionDetector
from .intraday_base import IntradayStrategy
from .intraday_registry import IntradayStrategyRegistry

logger = logging.getLogger(__name__)

# Default condition-to-strategy mapping
DEFAULT_STRATEGY_MAP = {
    'trending_up': ['momentum_surge', 'pullback_entry'],
    'trending_down': ['momentum_surge', 'pullback_entry'],
    'choppy': ['orb_breakout'],
    'sideways': ['range_trade'],
}

# ORB is always available during its window (first 2 hours)
ORB_ALWAYS_AVAILABLE_UNTIL_HOUR = 11
ORB_ALWAYS_AVAILABLE_UNTIL_MIN = 30


class StrategySelector:
    """Selects active intraday strategies based on market conditions.

    Re-evaluates periodically using MarketConditionDetector, then enables
    or disables strategies from the available pool.
    """

    def __init__(self, available_strategies: Dict[str, IntradayStrategy],
                 condition_detector: Optional[MarketConditionDetector] = None,
                 strategy_map: Optional[Dict[str, List[str]]] = None):
        """
        Args:
            available_strategies: dict of strategy_id -> IntradayStrategy instances
            condition_detector: MarketConditionDetector instance (creates one if None)
            strategy_map: custom condition -> strategy_ids mapping
        """
        self._all_strategies = available_strategies
        self._detector = condition_detector or MarketConditionDetector()
        self._strategy_map = strategy_map or dict(DEFAULT_STRATEGY_MAP)
        self._active_ids: Set[str] = set()
        self._last_condition: Optional[MarketCondition] = None
        self._size_multiplier: float = 1.0  # Reduce size in choppy markets

    @property
    def active_strategy_ids(self) -> List[str]:
        return sorted(self._active_ids)

    @property
    def active_strategies(self) -> Dict[str, IntradayStrategy]:
        return {sid: self._all_strategies[sid]
                for sid in self._active_ids
                if sid in self._all_strategies}

    @property
    def current_condition(self) -> Optional[MarketCondition]:
        return self._last_condition

    @property
    def size_multiplier(self) -> float:
        return self._size_multiplier

    def update(self, spy_tick_data: Dict, now: datetime,
               force: bool = False) -> Dict[str, IntradayStrategy]:
        """Re-evaluate market condition and update active strategies.

        Args:
            spy_tick_data: TickIndicatorEngine.get_data('SPY')
            now: current datetime
            force: bypass cache

        Returns:
            Dict of active strategy_id -> IntradayStrategy
        """
        condition = self._detector.evaluate(spy_tick_data, now, force=force)
        self._last_condition = condition

        # Determine strategies for this condition
        mapped_ids = set(self._strategy_map.get(condition.condition, []))

        # ORB always available during early session
        if (now.hour < ORB_ALWAYS_AVAILABLE_UNTIL_HOUR or
                (now.hour == ORB_ALWAYS_AVAILABLE_UNTIL_HOUR and
                 now.minute <= ORB_ALWAYS_AVAILABLE_UNTIL_MIN)):
            mapped_ids.add('orb_breakout')

        # Filter to only strategies we actually have instances of
        self._active_ids = mapped_ids & set(self._all_strategies.keys())

        # Adjust position sizing for choppy markets
        if condition.condition == 'choppy':
            self._size_multiplier = 0.5
        elif condition.confidence < 0.5:
            self._size_multiplier = 0.75
        else:
            self._size_multiplier = 1.0

        if self._active_ids:
            logger.debug(f"Strategy selector: {condition.condition} "
                         f"(conf={condition.confidence:.0%}) -> "
                         f"{sorted(self._active_ids)}, size={self._size_multiplier}x")

        return self.active_strategies

    def get_status(self) -> Dict:
        """Get current selector status for API/dashboard."""
        _cond = self._last_condition.condition if self._last_condition else 'unknown'
        return {
            'condition': _cond,
            'market_condition': _cond,  # alias used by strategies page JS
            'condition_confidence': self._last_condition.confidence if self._last_condition else 0,
            'ema_alignment': self._last_condition.ema_alignment if self._last_condition else 'unknown',
            'rsi': self._last_condition.rsi if self._last_condition else 0,
            'volatility': self._last_condition.volatility if self._last_condition else 'unknown',
            'active_strategies': sorted(self._active_ids),
            'size_multiplier': self._size_multiplier,
            'strategy_map': self._strategy_map,
            'notes': self._last_condition.notes if self._last_condition else '',
        }

    def reset_daily(self):
        """Reset at start of new trading day."""
        self._detector.reset_daily()
        self._active_ids.clear()
        self._last_condition = None
        self._size_multiplier = 1.0
