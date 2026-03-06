"""Shared fixtures for the comprehensive test suite.

Stubs external dependencies (psycopg2, broker APIs, LLM) so tests run
without network access, real credentials, or a running database.
"""
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Stub heavy dependencies BEFORE importing gap_fade_app
# ---------------------------------------------------------------------------
_fake_pg = MagicMock()
_fake_pg.connect.return_value = MagicMock()
_fake_pg.extensions = MagicMock()
_fake_pg.extensions.ISOLATION_LEVEL_AUTOCOMMIT = 0
sys.modules.setdefault('psycopg2', _fake_pg)
sys.modules.setdefault('psycopg2.extensions', _fake_pg.extensions)
sys.modules.setdefault('psycopg2.extras', MagicMock())

for mod in ('anthropic', 'ollama', 'authlib', 'authlib.integrations',
            'authlib.integrations.starlette_client', 'httpx',
            'starlette.middleware.sessions'):
    sys.modules.setdefault(mod, MagicMock())

os.environ.setdefault('DATABASE_URL', 'postgresql://x:x@localhost/test')
os.environ.setdefault('ALPACA_API_KEY', 'test_key_pk')
os.environ.setdefault('ALPACA_SECRET_KEY', 'test_key_sk')

# Now safe to import
from gap_fade_app import (
    GapFadeConfig, GapFadeEngine, GapCandidate, GapPosition,
    TradeRecord, DailyStats, StopOutRecord, MarketRegime,
    compute_adaptive_stop_pct, _direction_pnl, _stop_hit, _target_hit,
    GapFadeLiveTrader, AlertNotifier, EventBus, MarketEventDetector,
    TradingJournal, ConversationMemory, AnalysisCache,
)

# Timezone
try:
    from zoneinfo import ZoneInfo
    ET = ZoneInfo('America/New_York')
except ImportError:
    import pytz
    ET = pytz.timezone('US/Eastern')


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def make_config(**overrides) -> GapFadeConfig:
    """Create a GapFadeConfig with sensible test defaults."""
    defaults = dict(
        initial_capital=100_000,
        risk_pct=0.02,
        max_positions=5,
        stop_pct=0.015,
        adaptive_stops=True,
        stop_gap_fraction=0.25,
        stop_min_pct=0.015,
        stop_max_pct=0.025,
        gap_threshold=0.07,
        max_gap_pct=0.50,
        vol_ratio_max=3.0,
        min_avg_volume=50_000,
        min_price=10.0,
        entry_cutoff_hour=11,
        entry_cutoff_min=30,
        time_exit_hour=15,
        time_exit_min=0,
        eod_exit_hour=15,
        eod_exit_min=50,
        daily_loss_limit=0.02,
        max_consec_losses=2,
        max_drawdown=0.05,
        reentry_enabled=True,
        reentry_cooldown_minutes=30,
        reentry_max_per_symbol=1,
        reentry_trigger_pct=0.0,
        trade_gap_downs=False,
        kelly_fraction=0.25,
        slippage_pct=0.0015,
        max_notional=50_000,
        max_pct_adv=0.02,
        catalyst_enabled=False,
        auto_start=False,
        llm_enabled=False,
        intraday_enabled=False,
    )
    defaults.update(overrides)
    return GapFadeConfig(**defaults)


def make_engine(backtest_mode: bool = True, **config_overrides) -> GapFadeEngine:
    """Create a GapFadeEngine in backtest mode (skip circuit breakers)."""
    config = make_config(**config_overrides)
    return GapFadeEngine(config, backtest_mode=backtest_mode)


def make_candidate(
    symbol: str = 'TEST',
    gap_pct: float = 0.10,
    prev_close: float = 100.0,
    direction: str = 'short',
    vol_ratio: float = 1.5,
    avg_vol: float = 500_000,
    catalyst: str = '',
    **kwargs,
) -> GapCandidate:
    """Create a GapCandidate for testing."""
    premarket = prev_close * (1 + gap_pct) if direction == 'short' else prev_close * (1 - gap_pct)
    return GapCandidate(
        symbol=symbol,
        gap_pct=gap_pct,
        prev_close=prev_close,
        premarket_price=premarket,
        avg_vol_20d=avg_vol,
        vol_ratio=vol_ratio,
        shortable=True,
        easy_to_borrow=True,
        direction=direction,
        catalyst=catalyst,
        **kwargs,
    )


def make_time(hour: int, minute: int, second: int = 0,
              year: int = 2026, month: int = 3, day: int = 9) -> datetime:
    """Create a timezone-aware ET datetime."""
    return datetime(year, month, day, hour, minute, second, tzinfo=ET)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    return make_config()


@pytest.fixture
def engine():
    return make_engine()


@pytest.fixture
def live_engine():
    """Engine in live mode (circuit breakers active)."""
    return make_engine(backtest_mode=False)
