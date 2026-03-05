"""Strategy registry for intraday strategies — discover, list, and instantiate.

Uses a decorator-based registration pattern:

    @IntradayStrategyRegistry.register('orb_breakout')
    class ORBBreakoutStrategy(IntradayStrategy):
        ...
"""

import logging
from typing import Dict, List, Optional, Type

from .intraday_base import IntradayStrategy

logger = logging.getLogger(__name__)

_INTRADAY_REGISTRY: Dict[str, Type[IntradayStrategy]] = {}


class IntradayStrategyRegistry:
    """Central registry for intraday strategies."""

    @staticmethod
    def register(strategy_id: str):
        """Decorator to register an intraday strategy class.

        Usage:
            @IntradayStrategyRegistry.register('orb_breakout')
            class ORBBreakoutStrategy(IntradayStrategy):
                ...
        """
        def decorator(cls):
            if not issubclass(cls, IntradayStrategy):
                raise TypeError(f"{cls.__name__} must subclass IntradayStrategy")
            if strategy_id in _INTRADAY_REGISTRY:
                logger.warning(f"Intraday strategy '{strategy_id}' already registered, overwriting")
            _INTRADAY_REGISTRY[strategy_id] = cls
            logger.debug(f"Registered intraday strategy: {strategy_id} -> {cls.__name__}")
            return cls
        return decorator

    @staticmethod
    def list_strategies() -> List[Dict]:
        """Return metadata for all registered intraday strategies."""
        result = []
        for sid, cls in _INTRADAY_REGISTRY.items():
            try:
                instance = cls()
                result.append({
                    'id': sid,
                    'name': instance.name,
                    'description': instance.description,
                    'version': instance.version,
                    'parameters': instance.get_parameter_schema(),
                    'defaults': instance.get_default_config(),
                    'required_indicators': instance.get_required_indicators(),
                    'ema_periods': instance.get_ema_periods(),
                    'active_window': instance.get_active_window(),
                })
            except Exception as e:
                result.append({
                    'id': sid,
                    'name': cls.__name__,
                    'description': f'Error loading: {e}',
                    'version': '?',
                    'parameters': {},
                    'defaults': {},
                    'required_indicators': [],
                    'ema_periods': [],
                    'active_window': (9, 45, 15, 30),
                })
        return result

    @staticmethod
    def create_strategy(strategy_id: str, config: Optional[Dict] = None) -> IntradayStrategy:
        """Instantiate an intraday strategy by ID.

        Raises:
            KeyError if strategy_id not found
        """
        if strategy_id not in _INTRADAY_REGISTRY:
            available = list(_INTRADAY_REGISTRY.keys())
            raise KeyError(
                f"Unknown intraday strategy '{strategy_id}'. "
                f"Available: {available}"
            )
        cls = _INTRADAY_REGISTRY[strategy_id]
        return cls(config=config)

    @staticmethod
    def get_strategy_class(strategy_id: str) -> Type[IntradayStrategy]:
        """Get the strategy class (not instance) by ID."""
        if strategy_id not in _INTRADAY_REGISTRY:
            raise KeyError(f"Unknown intraday strategy '{strategy_id}'")
        return _INTRADAY_REGISTRY[strategy_id]

    @staticmethod
    def has_strategy(strategy_id: str) -> bool:
        """Check if an intraday strategy is registered."""
        return strategy_id in _INTRADAY_REGISTRY

    @staticmethod
    def get_all_ids() -> List[str]:
        """Return all registered strategy IDs."""
        return list(_INTRADAY_REGISTRY.keys())
