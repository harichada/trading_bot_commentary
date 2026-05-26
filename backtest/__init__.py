"""Backtest framework MVP — see ``backtest/engine.py`` for the entry point.

Public surface re-exported here so callers can write::

    from backtest import run_backtest, Trade, write_report_json
"""
from __future__ import annotations

from backtest.engine import (
    DEFAULT_DAYS,
    DEFAULT_DSN,
    DEFAULT_FREQUENCY,
    DEFAULT_TOP_N,
    MAX_HOLD_BARS,
    _run_backtest_with_trades,
    compute_indicators,
    load_strategies,
    replay_symbol,
    run_backtest,
    seed_run,
)
from backtest.metrics import summarize, summarize_per_symbol
from backtest.report_writers import write_report_json, write_trades_csv
from backtest.trades import BarsLoader, ProgressCb, ProgressEvent, SymbolMetrics, Trade


# Alias used by tests and downstream callers that want the legacy name.
BacktestReport = dict

__all__ = [
    # Core
    "run_backtest",
    "_run_backtest_with_trades",
    "replay_symbol",
    "compute_indicators",
    "load_strategies",
    "seed_run",
    # Types
    "Trade",
    "SymbolMetrics",
    "ProgressEvent",
    "ProgressCb",
    "BarsLoader",
    "BacktestReport",
    # Metrics
    "summarize",
    "summarize_per_symbol",
    # Writers
    "write_report_json",
    "write_trades_csv",
    # Constants
    "DEFAULT_DSN",
    "DEFAULT_TOP_N",
    "DEFAULT_DAYS",
    "DEFAULT_FREQUENCY",
    "MAX_HOLD_BARS",
]
