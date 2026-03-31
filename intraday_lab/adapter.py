"""Strategy adapter — wraps existing IntradayStrategy classes for replay.

Translates between the replay engine's bar-based data format and the
strategy's scan_for_setups() interface without modifying the original classes.
"""

import sys
import os
import logging
from typing import Dict, List, Optional, Type

# Ensure parent dir is on path so we can import gap_fade_strategies
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger('IntradayLab')


def get_all_strategies() -> Dict[str, dict]:
    """List all available intraday strategies from the registry."""
    try:
        from gap_fade_strategies import IntradayStrategyRegistry
        return IntradayStrategyRegistry.list_strategies()
    except Exception as e:
        logger.warning(f"Failed to load strategy registry: {e}")
        return {}


def create_strategy(strategy_id: str, config: Optional[Dict] = None):
    """Create a strategy instance by ID."""
    try:
        from gap_fade_strategies import IntradayStrategyRegistry
        cls = IntradayStrategyRegistry.get_strategy_class(strategy_id)
        return cls(config=config)
    except Exception as e:
        logger.warning(f"Failed to create strategy '{strategy_id}': {e}")
        return None


def get_strategy_params(strategy_id: str) -> Dict:
    """Get the default config params for a strategy (for parameter grid UI)."""
    strategy = create_strategy(strategy_id)
    if strategy and hasattr(strategy, 'get_default_config'):
        return strategy.get_default_config()
    return {}


def list_strategy_ids() -> List[str]:
    """Get list of all intraday strategy IDs."""
    try:
        from gap_fade_strategies import IntradayStrategyRegistry
        strats = IntradayStrategyRegistry.list_strategies()
        if isinstance(strats, dict):
            return list(strats.keys())
        if isinstance(strats, list):
            return [s.get('id', s.get('strategy_id', '')) for s in strats
                    if isinstance(s, dict)]
        return []
    except Exception as e:
        logger.warning(f"Failed to list strategies: {e}")
        return []
