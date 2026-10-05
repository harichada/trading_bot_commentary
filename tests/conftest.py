"""v-test-log-isolation-2026-05-13: pytest contamination of
``trading_bot.log`` was making the production log unreliable.

When tests import any core module, ``core/config.py`` runs its
module-level logging setup and attaches a RotatingFileHandler to the
``TradingBot`` logger pointing at ``trading_bot.log``. From that point
on, *every* log call from anywhere in the test process writes to the
live bot's log file — including pytest fixtures that synthesize
silent-death events, dead-man's-switch triggers, etc.

This was observed 2026-05-13 11:01: my watchdog regression tests
emitted CRITICAL lines into the running bot's log, which Monitor saw
and pushed as if the live bot were dying.

Fix: at pytest session start, detach any file handlers that target
trading_bot.log from the TradingBot logger. Tests still log at the
configured level — they just don't poison the production log file.
"""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

import pytest


@pytest.fixture(scope="session", autouse=True)
def _isolate_test_logging():
    """Remove file handlers that write to trading_bot.log from the
    TradingBot logger before any tests run. Idempotent — safe to call
    multiple times even if pytest reloads modules."""
    logger = logging.getLogger("TradingBot")
    removed = []
    for handler in list(logger.handlers):
        baseFilename = getattr(handler, "baseFilename", None)
        if baseFilename and "trading_bot.log" in baseFilename:
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass
            removed.append(baseFilename)
    # Replace with a quiet null sink so test code that logs at
    # WARNING/ERROR still has somewhere for output to go without
    # complaints about missing handlers.
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    yield removed


@pytest.fixture(autouse=True)
def _hands_off_list_is_data(monkeypatch):
    """v-hands-off-data-2026-10-05: the hands-off list is yaml data with no
    code floor, so tests pin it in memory (never on disk) to the two symbols
    the live account protects. Tests that need a different list override
    `core.config.config_manager.config['trading']['hands_off_denylist']`."""
    try:
        from core.config import config_manager
        trading = config_manager.config.setdefault("trading", {})
        monkeypatch.setitem(trading, "hands_off_denylist", ["HQGE", "SPCX"])
    except Exception:
        pass
