"""Gap Fade Strategy Plugin System.

Import this package to register all built-in strategies.

Usage:
    from gap_fade_strategies import GapFadeStrategyRegistry

    # List available strategies
    strategies = GapFadeStrategyRegistry.list_strategies()

    # Create a strategy instance
    strategy = GapFadeStrategyRegistry.create_strategy('classic_gap_fade')
    strategy = GapFadeStrategyRegistry.create_strategy('vwap_gap_fade', config={...})

    # Intraday strategies
    from gap_fade_strategies import IntradayStrategyRegistry
    intraday = IntradayStrategyRegistry.list_strategies()
"""

from .base import ExitSignal, GapFadeStrategy
from .indicators import TickIndicatorEngine
from .registry import GapFadeStrategyRegistry
from .intraday_base import IntradaySetup, IntradayStrategy
from .intraday_registry import IntradayStrategyRegistry
from .market_condition import MarketCondition, MarketConditionDetector
from .strategy_selector import StrategySelector

# Import strategy modules to trigger @register decorators
from . import classic_gap_fade  # noqa: F401
from . import vwap_gap_fade     # noqa: F401
from . import confluence_gap    # noqa: F401
from . import minervini_trend   # noqa: F401

# Import intraday strategy modules to trigger @register decorators
from . import orb_breakout      # noqa: F401
from . import momentum_surge    # noqa: F401
from . import pullback_entry    # noqa: F401
from . import range_trade       # noqa: F401

__all__ = [
    'GapFadeStrategy',
    'GapFadeStrategyRegistry',
    'TickIndicatorEngine',
    'ExitSignal',
    'IntradaySetup',
    'IntradayStrategy',
    'IntradayStrategyRegistry',
    'MarketCondition',
    'MarketConditionDetector',
    'StrategySelector',
]
