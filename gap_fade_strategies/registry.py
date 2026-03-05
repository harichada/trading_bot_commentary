"""Strategy registry — discover, list, and instantiate strategies.

Uses a decorator-based registration pattern:

    @GapFadeStrategyRegistry.register('my_strategy')
    class MyStrategy(GapFadeStrategy):
        ...
"""

import logging
from typing import Dict, List, Optional, Type

from .base import GapFadeStrategy

logger = logging.getLogger(__name__)

_REGISTRY: Dict[str, Type[GapFadeStrategy]] = {}


class GapFadeStrategyRegistry:
    """Central registry for gap fade strategies."""

    @staticmethod
    def register(strategy_id: str):
        """Decorator to register a strategy class.

        Usage:
            @GapFadeStrategyRegistry.register('vwap_gap_fade')
            class VWAPGapFadeStrategy(GapFadeStrategy):
                ...
        """
        def decorator(cls):
            if not issubclass(cls, GapFadeStrategy):
                raise TypeError(f"{cls.__name__} must subclass GapFadeStrategy")
            if strategy_id in _REGISTRY:
                logger.warning(f"Strategy '{strategy_id}' already registered, overwriting")
            _REGISTRY[strategy_id] = cls
            logger.debug(f"Registered strategy: {strategy_id} -> {cls.__name__}")
            return cls
        return decorator

    @staticmethod
    def list_strategies() -> List[Dict]:
        """Return metadata for all registered strategies."""
        result = []
        for sid, cls in _REGISTRY.items():
            # Instantiate briefly to get metadata
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
                })
        return result

    @staticmethod
    def create_strategy(strategy_id: str, config: Optional[Dict] = None) -> GapFadeStrategy:
        """Instantiate a strategy by ID.

        Args:
            strategy_id: registered strategy identifier
            config: optional strategy-specific config dict

        Returns:
            Strategy instance

        Raises:
            KeyError if strategy_id not found
        """
        if strategy_id not in _REGISTRY:
            available = list(_REGISTRY.keys())
            raise KeyError(
                f"Unknown strategy '{strategy_id}'. "
                f"Available: {available}"
            )
        cls = _REGISTRY[strategy_id]
        return cls(config=config)

    @staticmethod
    def get_strategy_class(strategy_id: str) -> Type[GapFadeStrategy]:
        """Get the strategy class (not instance) by ID."""
        if strategy_id not in _REGISTRY:
            raise KeyError(f"Unknown strategy '{strategy_id}'")
        return _REGISTRY[strategy_id]

    @staticmethod
    def has_strategy(strategy_id: str) -> bool:
        """Check if a strategy is registered."""
        return strategy_id in _REGISTRY

    @staticmethod
    def get_default_id() -> str:
        """Return the default strategy ID."""
        return 'classic_gap_fade'
