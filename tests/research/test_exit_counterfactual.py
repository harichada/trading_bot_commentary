"""Tests for exit counterfactual research harness.

v-exit-counterfactual-tests-2026-09-24. Tests the replay logic for
all four exit policies:
  (a) actual — pass-through of ledger exit
  (b) hold_pure — pure hold to SL/TP/flatten
  (c) pro_policy — scale 50% at +1R, breakeven, trail
  (d) pro_no_early — pro policy without early indicator exits
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pandas as pd
import pytest

from research.exit_counterfactual import (
    DayTradeEntry,
    ReplayResult,
    PolicyMetrics,
    replay_policy_actual,
    replay_policy_hold_pure,
    replay_policy_pro,
    compute_policy_metrics,
    extract_day_trade_entries,
    HANDS_OFF_SYMBOLS,
)


def _entry(
    symbol: str = "TEST",
    entry_price: float = 100.0,
    exit_price: float = 102.0,
    stop_loss: float = 97.0,
    take_profit: float = 106.0,
    stop_distance: float = 3.0,
    exit_reason: str = "target",
    entry_pattern: str = "continuation",
    atr: float = 2.0,
) -> DayTradeEntry:
    """Create a test DayTradeEntry."""
    # Use 10:00 AM ET to avoid flatten time checks (3 PM ET)
    now = datetime(2026, 9, 15, 14, 0, 0, tzinfo=timezone.utc)  # 10:00 AM ET
    return DayTradeEntry(
        symbol=symbol,
        entry_time=now,
        exit_time=now + timedelta(minutes=30),
        entry_price=entry_price,
        exit_price=exit_price,
        quantity=100,
        pnl=(exit_price - entry_price) * 100,
        exit_reason=exit_reason,
        strategy="day_trade_momentum",
        entry_pattern=entry_pattern,
        stop_loss=stop_loss,
        take_profit=take_profit,
        stop_distance=stop_distance,
        atr=atr,
        rsi=55.0,
        regime="risk_on",
        time_of_day="opening_30",
        session_id="2026-09-24",
        raw={},
    )


def _make_bars_df(
    start: datetime,
    prices: list[tuple[float, float, float, float]],
    interval_minutes: int = 1,
) -> pd.DataFrame:
    """Create a mock bars DataFrame.
    
    Args:
        start: Start timestamp
        prices: List of (open, high, low, close) tuples
        interval_minutes: Minutes between bars
    
    Timestamps start 1 minute after start and increment by interval_minutes.
    This ensures bars are after the entry time and within market hours.
    """
    # Start at 10:01 AM ET (14:01 UTC) to be well before flatten time
    base_time = datetime(2026, 9, 15, 14, 1, 0, tzinfo=timezone.utc)
    timestamps = [base_time + timedelta(minutes=i * interval_minutes) for i in range(len(prices))]
    df = pd.DataFrame(
        [{"Open": o, "High": h, "Low": l, "Close": c, "Volume": 1000} for o, h, l, c in prices],
        index=pd.DatetimeIndex(timestamps),
    )
    return df


class TestDayTradeEntryParsing:
    """Tests for DayTradeEntry.from_trade_history."""
    
    def test_parses_momentum_trade(self):
        """Basic momentum trade parses correctly."""
        raw = {
            "symbol": "NVDA",
            "entry_time": "2026-09-15T14:30:00",
            "exit_time": "2026-09-15T15:00:00",
            "entry_price": 120.0,
            "exit_price": 122.0,
            "quantity": 50,
            "pnl": 100.0,
            "exit_reason": "target",
            "reasoning": {
                "strategy": "day_trade_momentum",
                "entry_pattern": "breakout",
                "stop_loss": 118.0,
                "take_profit": 124.0,
                "stop_distance": 2.0,
                "atr": 1.5,
                "rsi": 60.0,
                "regime": "risk_on",
            },
        }
        
        entry = DayTradeEntry.from_trade_history(raw)
        
        assert entry is not None
        assert entry.symbol == "NVDA"
        assert entry.entry_price == 120.0
        assert entry.stop_loss == 118.0
        assert entry.strategy == "day_trade_momentum"
    
    def test_excludes_hands_off_symbols(self):
        """MU, HQGE, SPCX are excluded."""
        for symbol in HANDS_OFF_SYMBOLS:
            raw = {
                "symbol": symbol,
                "entry_time": "2026-09-15T14:30:00",
                "reasoning": {"strategy": "day_trade_momentum"},
            }
            assert DayTradeEntry.from_trade_history(raw) is None
    
    def test_excludes_non_day_trade_strategies(self):
        """Non-day-trade strategies are excluded."""
        raw = {
            "symbol": "AAPL",
            "entry_time": "2026-09-15T14:30:00",
            "reasoning": {"strategy": "swing_trade"},
        }
        assert DayTradeEntry.from_trade_history(raw) is None
    
    def test_computes_stop_distance_from_atr(self):
        """Stop distance is computed from ATR when not provided."""
        raw = {
            "symbol": "TSLA",
            "entry_time": "2026-09-15T14:30:00",
            "entry_price": 200.0,
            "reasoning": {
                "strategy": "day_trade_momentum",
                "atr": 4.0,
            },
        }
        
        entry = DayTradeEntry.from_trade_history(raw)
        
        assert entry is not None
        assert entry.stop_distance == 4.0 * 1.5  # ATR_STOP_MULT = 1.5


class TestReplayPolicyActual:
    """Tests for policy (a): actual exit."""
    
    def test_returns_ledger_values(self):
        """Returns the actual exit from the ledger unchanged."""
        entry = _entry(
            entry_price=100.0,
            exit_price=102.0,
            stop_distance=3.0,
            exit_reason="target",
        )
        
        result = replay_policy_actual(entry)
        
        assert result.policy == "actual"
        assert result.exit_price == 102.0
        assert result.exit_reason == "target"
        assert result.r_multiple == pytest.approx(2.0 / 3.0, rel=0.01)
    
    def test_computes_r_multiple_for_loss(self):
        """R-multiple is negative for losses."""
        entry = _entry(
            entry_price=100.0,
            exit_price=97.0,
            stop_distance=3.0,
            exit_reason="stop_loss",
        )
        
        result = replay_policy_actual(entry)
        
        assert result.r_multiple == pytest.approx(-1.0, rel=0.01)


class TestReplayPolicyHoldPure:
    """Tests for policy (b): pure hold to SL/TP/flatten."""
    
    def test_stop_hit_exits_at_stop(self):
        """Trade exits at stop when price hits stop."""
        entry = _entry(
            entry_price=100.0,
            stop_loss=97.0,
            take_profit=106.0,
            stop_distance=3.0,
        )
        
        prices = [
            (100.0, 100.5, 99.5, 100.0),
            (100.0, 100.2, 96.5, 97.0),  # Low < stop
        ]
        df = _make_bars_df(entry.entry_time, prices)
        bars_loader = MagicMock(return_value=df)
        
        result = replay_policy_hold_pure(entry, bars_loader)
        
        assert result.exit_reason == "stop"
        assert result.exit_price == 97.0
        assert result.r_multiple == pytest.approx(-1.0, rel=0.01)
    
    def test_target_hit_exits_at_target(self):
        """Trade exits at target when price hits target."""
        entry = _entry(
            entry_price=100.0,
            stop_loss=97.0,
            take_profit=106.0,
            stop_distance=3.0,
        )
        
        prices = [
            (100.0, 102.0, 100.0, 101.5),
            (101.5, 104.0, 101.0, 103.5),
            (103.5, 107.0, 103.0, 106.0),  # High > target
        ]
        df = _make_bars_df(entry.entry_time, prices)
        bars_loader = MagicMock(return_value=df)
        
        result = replay_policy_hold_pure(entry, bars_loader)
        
        assert result.exit_reason == "target"
        assert result.exit_price == 106.0
        assert result.r_multiple == pytest.approx(2.0, rel=0.01)
    
    def test_no_bars_returns_no_bars_reason(self):
        """Returns no_bars when bars loader returns empty."""
        entry = _entry()
        bars_loader = MagicMock(return_value=pd.DataFrame())
        
        result = replay_policy_hold_pure(entry, bars_loader)
        
        assert result.exit_reason == "no_bars"
        assert result.r_multiple == 0.0


class TestReplayPolicyPro:
    """Tests for policy (c): unified pro policy with scale/trail."""
    
    def test_scale_at_1r_partial_exit(self):
        """Scales out 50% at +1R and moves stop to breakeven."""
        entry = _entry(
            entry_price=100.0,
            stop_loss=97.0,
            take_profit=106.0,
            stop_distance=3.0,
            atr=2.0,
        )
        
        prices = [
            (100.0, 101.0, 100.0, 101.0),
            (101.0, 102.0, 100.5, 102.0),
            (102.0, 103.5, 101.5, 103.0),  # +1R reached (100 + 3 = 103)
            (103.0, 103.0, 100.5, 101.0),  # Retraces but above BE
            (101.0, 101.5, 100.0, 100.5),  # Hits BE, trailing stop fires
        ]
        df = _make_bars_df(entry.entry_time, prices)
        bars_loader = MagicMock(return_value=df)
        
        result = replay_policy_pro(entry, bars_loader)
        
        assert result.scaled_out_at == pytest.approx(103.0, rel=0.01)
        assert result.partial_r == 0.5
    
    def test_target_hit_with_scale(self):
        """Full exit at target still works with scale mechanics."""
        entry = _entry(
            entry_price=100.0,
            stop_loss=97.0,
            take_profit=106.0,
            stop_distance=3.0,
            atr=2.0,
        )
        
        prices = [
            (100.0, 102.0, 100.0, 102.0),
            (102.0, 104.0, 102.0, 104.0),  # +1R
            (104.0, 107.0, 104.0, 106.0),  # Target hit
        ]
        df = _make_bars_df(entry.entry_time, prices)
        bars_loader = MagicMock(return_value=df)
        
        result = replay_policy_pro(entry, bars_loader)
        
        assert result.exit_reason == "target"
        assert result.exit_price == 106.0
    
    def test_policy_names_differ(self):
        """pro_policy and pro_no_early have different policy names."""
        entry = _entry()
        df = _make_bars_df(entry.entry_time, [(100, 101, 99, 100)])
        bars_loader = MagicMock(return_value=df)
        
        result_pro = replay_policy_pro(entry, bars_loader, disable_early_exits=False)
        result_no_early = replay_policy_pro(entry, bars_loader, disable_early_exits=True)
        
        assert result_pro.policy == "pro_policy"
        assert result_no_early.policy == "pro_no_early"


class TestComputePolicyMetrics:
    """Tests for aggregate metrics computation."""
    
    def test_computes_win_rate(self):
        """Win rate is computed correctly."""
        results = [
            ReplayResult(
                symbol="A", entry_time=datetime.now(), entry_price=100,
                exit_time=datetime.now(), exit_price=103, exit_reason="target",
                stop_distance=3, r_multiple=1.0, return_pct=3.0, hold_bars=10,
                policy="test",
            ),
            ReplayResult(
                symbol="B", entry_time=datetime.now(), entry_price=100,
                exit_time=datetime.now(), exit_price=97, exit_reason="stop",
                stop_distance=3, r_multiple=-1.0, return_pct=-3.0, hold_bars=5,
                policy="test",
            ),
        ]
        entries = [_entry(symbol="A"), _entry(symbol="B")]
        
        metrics = compute_policy_metrics(results, entries)
        
        assert metrics.n_trades == 2
        assert metrics.n_wins == 1
        assert metrics.n_losses == 1
        assert metrics.win_rate == 0.5
    
    def test_computes_profit_factor(self):
        """Profit factor is computed correctly."""
        results = [
            ReplayResult(
                symbol="A", entry_time=datetime.now(), entry_price=100,
                exit_time=datetime.now(), exit_price=106, exit_reason="target",
                stop_distance=3, r_multiple=2.0, return_pct=6.0, hold_bars=10,
                policy="test",
            ),
            ReplayResult(
                symbol="B", entry_time=datetime.now(), entry_price=100,
                exit_time=datetime.now(), exit_price=97, exit_reason="stop",
                stop_distance=3, r_multiple=-1.0, return_pct=-3.0, hold_bars=5,
                policy="test",
            ),
        ]
        entries = [_entry(symbol="A"), _entry(symbol="B")]
        
        metrics = compute_policy_metrics(results, entries)
        
        assert metrics.profit_factor == 2.0
        assert metrics.total_r == 1.0
        assert metrics.expectancy_r == 0.5
    
    def test_computes_exit_reason_breakdown(self):
        """Exit reasons are tallied correctly."""
        results = [
            ReplayResult(
                symbol="A", entry_time=datetime.now(), entry_price=100,
                exit_time=datetime.now(), exit_price=106, exit_reason="target",
                stop_distance=3, r_multiple=2.0, return_pct=6.0, hold_bars=10,
                policy="test",
            ),
            ReplayResult(
                symbol="B", entry_time=datetime.now(), entry_price=100,
                exit_time=datetime.now(), exit_price=97, exit_reason="stop",
                stop_distance=3, r_multiple=-1.0, return_pct=-3.0, hold_bars=5,
                policy="test",
            ),
            ReplayResult(
                symbol="C", entry_time=datetime.now(), entry_price=100,
                exit_time=datetime.now(), exit_price=97, exit_reason="stop",
                stop_distance=3, r_multiple=-1.0, return_pct=-3.0, hold_bars=5,
                policy="test",
            ),
        ]
        entries = [_entry(symbol=s) for s in ["A", "B", "C"]]
        
        metrics = compute_policy_metrics(results, entries)
        
        assert metrics.by_exit_reason["target"] == 1
        assert metrics.by_exit_reason["stop"] == 2
    
    def test_excludes_no_bars_from_metrics(self):
        """Trades with no_bars exit are excluded from metrics."""
        results = [
            ReplayResult(
                symbol="A", entry_time=datetime.now(), entry_price=100,
                exit_time=None, exit_price=None, exit_reason="no_bars",
                stop_distance=3, r_multiple=0.0, return_pct=0.0, hold_bars=0,
                policy="test",
            ),
            ReplayResult(
                symbol="B", entry_time=datetime.now(), entry_price=100,
                exit_time=datetime.now(), exit_price=106, exit_reason="target",
                stop_distance=3, r_multiple=2.0, return_pct=6.0, hold_bars=10,
                policy="test",
            ),
        ]
        entries = [_entry(symbol="A"), _entry(symbol="B")]
        
        metrics = compute_policy_metrics(results, entries)
        
        assert metrics.n_trades == 1


class TestExtractDayTradeEntries:
    """Tests for extracting day-trade entries from raw history."""
    
    def test_filters_by_date_range(self):
        """Date range filtering works."""
        raw = [
            {
                "symbol": "A",
                "entry_time": "2026-09-15T10:00:00",
                "reasoning": {"strategy": "day_trade_momentum"},
            },
            {
                "symbol": "B",
                "entry_time": "2026-09-20T10:00:00",
                "reasoning": {"strategy": "day_trade_momentum"},
            },
            {
                "symbol": "C",
                "entry_time": "2026-09-25T10:00:00",
                "reasoning": {"strategy": "day_trade_momentum"},
            },
        ]
        from datetime import date
        
        entries = extract_day_trade_entries(
            raw,
            start_date=date(2026, 9, 18),
            end_date=date(2026, 9, 22),
        )
        
        assert len(entries) == 1
        assert entries[0].symbol == "B"
    
    def test_excludes_hands_off_symbols(self):
        """Hands-off symbols are excluded."""
        raw = [
            {
                "symbol": "MU",
                "entry_time": "2026-09-15T10:00:00",
                "reasoning": {"strategy": "day_trade_momentum"},
            },
            {
                "symbol": "NVDA",
                "entry_time": "2026-09-15T10:00:00",
                "reasoning": {"strategy": "day_trade_momentum"},
            },
        ]
        
        entries = extract_day_trade_entries(raw)
        
        assert len(entries) == 1
        assert entries[0].symbol == "NVDA"
