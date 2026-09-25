"""Tests for exit counterfactual research harness.

v-exit-counterfactual-tests-2026-09-24-r2. Tests the replay logic for
all four exit policies with bug fixes:
  - Timezone: naive timestamps treated as ET, bars filtered to market hours
  - pro_no_early: actually skips MACD/RSI early exits
  - Same-sample comparison: only trades with bars included
  - Exit reason fallback: exit_reason or reason
  - Dedupe on (symbol, entry_time), skip stop_distance<=0
  - Support shorts, plain list ledger
"""
from __future__ import annotations

from datetime import datetime, timedelta, date
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
    load_trading_state,
    run_counterfactual_replay,
    _to_et,
    _is_market_hours,
    _is_flatten_time,
    _is_time_stop,
    _check_early_exit_triggers,
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
    side: str = "long",
    entry_time: datetime = None,
    initial_stop_distance: float = None,
) -> DayTradeEntry:
    """Create a test DayTradeEntry."""
    if entry_time is None:
        entry_time = datetime(2026, 9, 15, 10, 0, 0)
    if initial_stop_distance is None:
        initial_stop_distance = stop_distance
    return DayTradeEntry(
        symbol=symbol,
        entry_time=entry_time,
        exit_time=entry_time + timedelta(minutes=30),
        entry_price=entry_price,
        exit_price=exit_price,
        quantity=100,
        pnl=(exit_price - entry_price) * 100 if side == "long" else (entry_price - exit_price) * 100,
        exit_reason=exit_reason,
        strategy="day_trade_momentum",
        entry_pattern=entry_pattern,
        stop_loss=stop_loss,
        take_profit=take_profit,
        stop_distance=stop_distance,
        initial_stop_distance=initial_stop_distance,
        atr=atr,
        rsi=55.0,
        regime="risk_on",
        time_of_day="opening_30",
        session_id="2026-09-15",
        side=side,
        raw={},
    )


def _make_bars_df(
    start: datetime,
    prices: list[tuple[float, float, float, float]],
    interval_minutes: int = 1,
    with_indicators: bool = False,
) -> pd.DataFrame:
    """Create a mock bars DataFrame.
    
    Args:
        start: Start timestamp (naive, assumed ET)
        prices: List of (open, high, low, close) tuples
        interval_minutes: Minutes between bars
        with_indicators: Add MACD/RSI columns
    """
    base_time = datetime(2026, 9, 15, 10, 1, 0)
    timestamps = [base_time + timedelta(minutes=i * interval_minutes) for i in range(len(prices))]
    df = pd.DataFrame(
        [{"Open": o, "High": h, "Low": l, "Close": c, "Volume": 1000} for o, h, l, c in prices],
        index=pd.DatetimeIndex(timestamps),
    )
    if with_indicators:
        df["macd"] = 0.0
        df["macd_signal"] = 0.0
        df["rsi"] = 50.0
    return df


class TestTimezoneHandling:
    """Tests for timezone handling — naive timestamps treated as ET."""
    
    def test_to_et_naive_is_et(self):
        """Naive timestamps are treated as ET."""
        ts = datetime(2026, 9, 15, 10, 0, 0)
        ts_et = _to_et(ts)
        assert ts_et.hour == 10
        assert ts_et.tzinfo is not None
    
    def test_is_market_hours_within(self):
        """Timestamps within 09:30-16:00 ET are market hours."""
        ts = datetime(2026, 9, 15, 10, 0, 0)
        assert _is_market_hours(ts) is True
    
    def test_is_market_hours_before_open(self):
        """Timestamps before 09:30 ET are not market hours."""
        ts = datetime(2026, 9, 15, 9, 0, 0)
        assert _is_market_hours(ts) is False
    
    def test_is_market_hours_after_close(self):
        """Timestamps after 16:00 ET are not market hours."""
        ts = datetime(2026, 9, 15, 17, 0, 0)
        assert _is_market_hours(ts) is False
    
    def test_is_flatten_time_before(self):
        """10:00 ET is not flatten time."""
        ts = datetime(2026, 9, 15, 10, 0, 0)
        assert _is_flatten_time(ts) is False
    
    def test_is_flatten_time_at_3pm(self):
        """15:00 ET is flatten time."""
        ts = datetime(2026, 9, 15, 15, 0, 0)
        assert _is_flatten_time(ts) is True
    
    def test_is_time_stop_before_cutoff(self):
        """14:30 ET is not time stop (15 min before 15:00)."""
        ts = datetime(2026, 9, 15, 14, 30, 0)
        assert _is_time_stop(ts, minutes_before_flatten=15) is False
    
    def test_is_time_stop_at_cutoff(self):
        """14:45 ET is time stop (15 min before 15:00)."""
        ts = datetime(2026, 9, 15, 14, 45, 0)
        assert _is_time_stop(ts, minutes_before_flatten=15) is True


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
                "reasoning": {"strategy": "day_trade_momentum", "stop_distance": 1.0},
            }
            assert DayTradeEntry.from_trade_history(raw) is None
    
    def test_excludes_non_day_trade_strategies(self):
        """Non-day-trade strategies are excluded."""
        raw = {
            "symbol": "AAPL",
            "entry_time": "2026-09-15T14:30:00",
            "reasoning": {"strategy": "swing_trade", "stop_distance": 1.0},
        }
        assert DayTradeEntry.from_trade_history(raw) is None
    
    def test_excludes_zero_stop_distance(self):
        """Trades with stop_distance<=0 are excluded."""
        raw = {
            "symbol": "AAPL",
            "entry_time": "2026-09-15T14:30:00",
            "reasoning": {"strategy": "day_trade_momentum", "stop_distance": 0},
        }
        assert DayTradeEntry.from_trade_history(raw) is None
    
    def test_exit_reason_fallback(self):
        """exit_reason falls back to reason."""
        raw = {
            "symbol": "TSLA",
            "entry_time": "2026-09-15T14:30:00",
            "reason": "stop_loss",
            "reasoning": {"strategy": "day_trade_momentum", "stop_distance": 2.0},
        }
        entry = DayTradeEntry.from_trade_history(raw)
        assert entry is not None
        assert entry.exit_reason == "stop_loss"
    
    def test_parses_short_trade(self):
        """Short trades are parsed with correct side."""
        raw = {
            "symbol": "TSLA",
            "entry_time": "2026-09-15T14:30:00",
            "entry_price": 200.0,
            "reasoning": {
                "strategy": "day_trade_momentum",
                "side": "short",
                "stop_distance": 5.0,
            },
        }
        entry = DayTradeEntry.from_trade_history(raw)
        assert entry is not None
        assert entry.side == "short"
        assert entry.stop_loss == 205.0  # Above entry for short
        assert entry.take_profit == 190.0  # Below entry for short


class TestExtractDayTradeEntries:
    """Tests for extracting and deduping entries."""
    
    def test_dedupes_on_symbol_entry_time(self):
        """Duplicate (symbol, entry_time) entries are removed."""
        raw = [
            {
                "symbol": "NVDA",
                "entry_time": "2026-09-15T10:00:00",
                "reasoning": {"strategy": "day_trade_momentum", "stop_distance": 2.0},
            },
            {
                "symbol": "NVDA",
                "entry_time": "2026-09-15T10:00:00",
                "reasoning": {"strategy": "day_trade_momentum", "stop_distance": 2.0},
            },
        ]
        
        entries = extract_day_trade_entries(raw)
        
        assert len(entries) == 1
    
    def test_filters_by_date_range(self):
        """Date range filtering works."""
        raw = [
            {
                "symbol": "A",
                "entry_time": "2026-09-15T10:00:00",
                "reasoning": {"strategy": "day_trade_momentum", "stop_distance": 1.0},
            },
            {
                "symbol": "B",
                "entry_time": "2026-09-20T10:00:00",
                "reasoning": {"strategy": "day_trade_momentum", "stop_distance": 1.0},
            },
            {
                "symbol": "C",
                "entry_time": "2026-09-25T10:00:00",
                "reasoning": {"strategy": "day_trade_momentum", "stop_distance": 1.0},
            },
        ]
        
        entries = extract_day_trade_entries(
            raw,
            start_date=date(2026, 9, 18),
            end_date=date(2026, 9, 22),
        )
        
        assert len(entries) == 1
        assert entries[0].symbol == "B"


class TestLoadTradingState:
    """Tests for loading ledger files."""
    
    def test_accepts_plain_list(self, tmp_path):
        """Plain list JSON is accepted."""
        ledger_path = tmp_path / "ledger.json"
        ledger_path.write_text('[{"symbol": "NVDA", "entry_time": "2026-09-15T10:00:00"}]')
        
        result = load_trading_state(ledger_path)
        
        assert len(result) == 1
        assert result[0]["symbol"] == "NVDA"
    
    def test_accepts_trade_history_dict(self, tmp_path):
        """Dict with trade_history key is accepted."""
        ledger_path = tmp_path / "trading_state.json"
        ledger_path.write_text('{"trade_history": [{"symbol": "TSLA"}]}')
        
        result = load_trading_state(ledger_path)
        
        assert len(result) == 1
        assert result[0]["symbol"] == "TSLA"


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
    
    def test_short_r_multiple(self):
        """Short trades compute R correctly."""
        entry = _entry(
            entry_price=100.0,
            exit_price=97.0,
            stop_distance=3.0,
            side="short",
        )
        
        result = replay_policy_actual(entry)
        
        assert result.r_multiple == pytest.approx(1.0, rel=0.01)


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
            (100.0, 100.2, 96.5, 97.0),
        ]
        df = _make_bars_df(entry.entry_time, prices)
        
        result = replay_policy_hold_pure(entry, df)
        
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
            (103.5, 107.0, 103.0, 106.0),
        ]
        df = _make_bars_df(entry.entry_time, prices)
        
        result = replay_policy_hold_pure(entry, df)
        
        assert result.exit_reason == "target"
        assert result.exit_price == 106.0
        assert result.r_multiple == pytest.approx(2.0, rel=0.01)
    
    def test_no_bars_returns_no_bars_reason(self):
        """Returns no_bars when bars loader returns empty."""
        entry = _entry()
        df = pd.DataFrame()
        
        result = replay_policy_hold_pure(entry, df)
        
        assert result.exit_reason == "no_bars"
        assert result.r_multiple == 0.0
    
    def test_flatten_at_3pm(self):
        """Trade flattens at 3 PM ET."""
        entry = _entry()
        
        base_time = datetime(2026, 9, 15, 14, 50, 0)
        timestamps = [base_time + timedelta(minutes=i) for i in range(15)]
        prices = [(100, 100.5, 99.5, 100) for _ in range(15)]
        df = pd.DataFrame(
            [{"Open": o, "High": h, "Low": l, "Close": c, "Volume": 1000} for o, h, l, c in prices],
            index=pd.DatetimeIndex(timestamps),
        )
        
        result = replay_policy_hold_pure(entry, df)
        
        assert result.exit_reason == "flatten"
    
    def test_short_stop_above_entry(self):
        """Short trades exit at stop above entry."""
        entry = _entry(
            entry_price=100.0,
            stop_loss=103.0,
            take_profit=94.0,
            stop_distance=3.0,
            side="short",
        )
        
        prices = [
            (100.0, 101.0, 99.5, 100.5),
            (100.5, 104.0, 100.0, 103.0),
        ]
        df = _make_bars_df(entry.entry_time, prices)
        
        result = replay_policy_hold_pure(entry, df)
        
        assert result.exit_reason == "stop"
        assert result.exit_price == 103.0
        assert result.r_multiple == pytest.approx(-1.0, rel=0.01)


class TestEarlyExitTriggers:
    """Tests for MACD/RSI early exit detection."""
    
    def test_macd_flip_bearish_long(self):
        """MACD flip bearish triggers exit for long."""
        bar = pd.Series({"macd": -0.5, "macd_signal": -0.3, "rsi": 55})
        prev_bar = pd.Series({"macd": 0.2, "macd_signal": 0.1, "rsi": 55})
        
        result = _check_early_exit_triggers(bar, prev_bar, "long")
        
        assert result == "proactive_macd_flipped_bearish"
    
    def test_rsi_below_50_long(self):
        """RSI below 50 triggers exit for long."""
        bar = pd.Series({"macd": 0.5, "macd_signal": 0.3, "rsi": 45})
        prev_bar = pd.Series({"macd": 0.5, "macd_signal": 0.3, "rsi": 55})
        
        result = _check_early_exit_triggers(bar, prev_bar, "long")
        
        assert result == "proactive_rsi_below_50"
    
    def test_macd_flip_bullish_short(self):
        """MACD flip bullish triggers exit for short."""
        bar = pd.Series({"macd": 0.5, "macd_signal": 0.3, "rsi": 45})
        prev_bar = pd.Series({"macd": -0.2, "macd_signal": -0.1, "rsi": 45})
        
        result = _check_early_exit_triggers(bar, prev_bar, "short")
        
        assert result == "proactive_macd_flipped_bullish"
    
    def test_no_trigger_when_holding(self):
        """No trigger when indicators are favorable."""
        bar = pd.Series({"macd": 0.5, "macd_signal": 0.3, "rsi": 55})
        prev_bar = pd.Series({"macd": 0.4, "macd_signal": 0.3, "rsi": 55})
        
        result = _check_early_exit_triggers(bar, prev_bar, "long")
        
        assert result is None


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
            (102.0, 103.5, 101.5, 103.0),
            (103.0, 103.0, 100.5, 101.0),
            (101.0, 101.5, 100.0, 100.5),
        ]
        df = _make_bars_df(entry.entry_time, prices)
        
        result = replay_policy_pro(entry, df, disable_early_exits=True)
        
        assert result.scaled_out_at == pytest.approx(103.0, rel=0.01)
        assert result.partial_r == 0.5
    
    def test_pro_no_early_skips_macd_exit(self):
        """pro_no_early skips MACD early exits."""
        entry = _entry(
            entry_price=100.0,
            stop_loss=97.0,
            take_profit=106.0,
            stop_distance=3.0,
        )
        
        base_time = datetime(2026, 9, 15, 10, 1, 0)
        timestamps = [base_time + timedelta(minutes=i) for i in range(5)]
        data = [
            {"Open": 100, "High": 100.5, "Low": 99, "Close": 99.5, "macd": 0.5, "macd_signal": 0.3, "rsi": 55},
            {"Open": 99.5, "High": 100, "Low": 98.5, "Close": 98.5, "macd": -0.2, "macd_signal": 0.1, "rsi": 48},
            {"Open": 98.5, "High": 99, "Low": 98, "Close": 98, "macd": -0.3, "macd_signal": -0.1, "rsi": 45},
            {"Open": 98, "High": 99, "Low": 97.5, "Close": 98.5, "macd": -0.1, "macd_signal": -0.1, "rsi": 50},
            {"Open": 98.5, "High": 100, "Low": 98, "Close": 99.5, "macd": 0.1, "macd_signal": -0.05, "rsi": 52},
        ]
        df = pd.DataFrame(data, index=pd.DatetimeIndex(timestamps))
        
        result_pro = replay_policy_pro(entry, df, disable_early_exits=False)
        result_no_early = replay_policy_pro(entry, df, disable_early_exits=True)
        
        assert result_pro.policy == "pro_policy"
        assert result_no_early.policy == "pro_no_early"
        
        if "proactive" in result_pro.exit_reason:
            assert result_no_early.exit_reason != result_pro.exit_reason
    
    def test_policy_names_differ(self):
        """pro_policy and pro_no_early have different policy names."""
        entry = _entry()
        df = _make_bars_df(entry.entry_time, [(100, 101, 99, 100)])
        
        result_pro = replay_policy_pro(entry, df, disable_early_exits=False)
        result_no_early = replay_policy_pro(entry, df, disable_early_exits=True)
        
        assert result_pro.policy == "pro_policy"
        assert result_no_early.policy == "pro_no_early"


class TestRunCounterfactualReplay:
    """Tests for the main replay function."""
    
    def test_returns_n_no_bars(self):
        """Returns count of trades without bars."""
        entries = [_entry(symbol="A"), _entry(symbol="B")]
        
        call_count = [0]
        def mock_loader(symbol, start, entry_ts, lookforward_minutes):
            call_count[0] += 1
            if symbol == "A":
                return _make_bars_df(start, [(100, 101, 99, 100)])
            return pd.DataFrame()
        
        results, n_no_bars = run_counterfactual_replay(entries, mock_loader)
        
        assert n_no_bars == 1
        assert len(results["actual"]) == 1
    
    def test_same_sample_all_policies(self):
        """All policies have the same sample (trades with bars)."""
        entries = [_entry(symbol="A"), _entry(symbol="B")]
        
        def mock_loader(symbol, start, entry_ts, lookforward_minutes):
            if symbol == "A":
                return _make_bars_df(start, [(100, 101, 99, 100)])
            return pd.DataFrame()
        
        results, n_no_bars = run_counterfactual_replay(entries, mock_loader)
        
        assert len(results["actual"]) == len(results["hold_pure"])
        assert len(results["actual"]) == len(results["pro_policy"])
        assert len(results["actual"]) == len(results["pro_no_early"])


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
    
    def test_no_max_dd_pct(self):
        """PolicyMetrics no longer has max_dd_pct."""
        results = [
            ReplayResult(
                symbol="A", entry_time=datetime.now(), entry_price=100,
                exit_time=datetime.now(), exit_price=103, exit_reason="target",
                stop_distance=3, r_multiple=1.0, return_pct=3.0, hold_bars=10,
                policy="test",
            ),
        ]
        entries = [_entry(symbol="A")]
        
        metrics = compute_policy_metrics(results, entries)
        
        assert not hasattr(metrics, "max_dd_pct") or "max_dd_pct" not in metrics.__dataclass_fields__
