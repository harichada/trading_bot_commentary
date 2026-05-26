"""Back-compat shim. The implementation now lives in the ``backtest``
package — see ``backtest/engine.py``. This module exists only so
existing imports (``from backtest_strategies import run_backtest``,
``api/backtest.py``, the unit tests, etc.) keep working.
"""
from __future__ import annotations

from backtest.engine import (  # noqa: F401
    DEFAULT_DAYS,
    DEFAULT_DSN,
    DEFAULT_FREQUENCY,
    DEFAULT_TOP_N,
    MAX_HOLD_BARS,
    compute_indicators,
    load_strategies,
    replay_symbol,
    run_backtest,
)
from backtest.metrics import summarize, summarize_per_symbol  # noqa: F401
from backtest.trades import (  # noqa: F401
    BarsLoader,
    ProgressCb,
    ProgressEvent,
    SymbolMetrics,
    Trade,
)
