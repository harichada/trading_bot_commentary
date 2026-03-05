"""Tests for IntradayBacktester — intraday strategy backtesting on daily data."""

import pytest
import asyncio
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo

from gap_fade_backtester import (
    IntradayBarSim, synthesize_intraday_bars,
    IntradayBacktester, IntradayPosition,
)

ET = ZoneInfo('US/Eastern')


class TestSynthesizeBars:
    def test_basic_bar_count(self):
        bars = synthesize_intraday_bars(
            '2024-03-01', 150.0, 155.0, 148.0, 152.0, 1000000, 149.0)
        assert len(bars) == 78  # 6.5 hours * 12 bars/hour

    def test_custom_bar_count(self):
        bars = synthesize_intraday_bars(
            '2024-03-01', 150.0, 155.0, 148.0, 152.0, 1000000, 149.0,
            n_bars=39)
        assert len(bars) == 39

    def test_bars_within_day_range(self):
        bars = synthesize_intraday_bars(
            '2024-03-01', 150.0, 155.0, 148.0, 152.0, 1000000, 149.0)
        for bar in bars:
            assert bar.low >= 148.0 - 0.01, f"Bar low {bar.low} below day low 148.0"
            assert bar.high <= 155.0 + 0.01, f"Bar high {bar.high} above day high 155.0"

    def test_first_bar_opens_at_day_open(self):
        bars = synthesize_intraday_bars(
            '2024-03-01', 150.0, 155.0, 148.0, 152.0, 1000000, 149.0)
        assert bars[0].open == 150.0

    def test_last_bar_closes_at_day_close(self):
        bars = synthesize_intraday_bars(
            '2024-03-01', 150.0, 155.0, 148.0, 152.0, 1000000, 149.0)
        assert bars[-1].close == 152.0

    def test_timestamps_are_sequential(self):
        bars = synthesize_intraday_bars(
            '2024-03-01', 150.0, 155.0, 148.0, 152.0, 1000000, 149.0)
        for i in range(1, len(bars)):
            assert bars[i].timestamp > bars[i-1].timestamp

    def test_timestamps_start_at_930(self):
        bars = synthesize_intraday_bars(
            '2024-03-01', 150.0, 155.0, 148.0, 152.0, 1000000, 149.0)
        assert bars[0].timestamp.hour == 9
        assert bars[0].timestamp.minute == 30

    def test_positive_volume(self):
        bars = synthesize_intraday_bars(
            '2024-03-01', 150.0, 155.0, 148.0, 152.0, 1000000, 149.0)
        for bar in bars:
            assert bar.volume > 0

    def test_deterministic_with_same_date(self):
        bars1 = synthesize_intraday_bars(
            '2024-03-01', 150.0, 155.0, 148.0, 152.0, 1000000, 149.0)
        bars2 = synthesize_intraday_bars(
            '2024-03-01', 150.0, 155.0, 148.0, 152.0, 1000000, 149.0)
        assert [b.close for b in bars1] == [b.close for b in bars2]

    def test_different_dates_differ(self):
        bars1 = synthesize_intraday_bars(
            '2024-03-01', 150.0, 155.0, 148.0, 152.0, 1000000, 149.0)
        bars2 = synthesize_intraday_bars(
            '2024-03-02', 150.0, 155.0, 148.0, 152.0, 1000000, 149.0)
        # Different seeds should produce different paths
        assert [b.close for b in bars1] != [b.close for b in bars2]

    def test_zero_open_returns_empty(self):
        bars = synthesize_intraday_bars(
            '2024-03-01', 0, 155.0, 148.0, 152.0, 1000000, 149.0)
        assert bars == []

    def test_bar_ohlc_valid(self):
        """High >= open,close and Low <= open,close for each bar."""
        bars = synthesize_intraday_bars(
            '2024-03-01', 150.0, 155.0, 148.0, 152.0, 1000000, 149.0)
        for bar in bars:
            assert bar.high >= bar.open, f"High {bar.high} < open {bar.open}"
            assert bar.high >= bar.close, f"High {bar.high} < close {bar.close}"
            assert bar.low <= bar.open, f"Low {bar.low} > open {bar.open}"
            assert bar.low <= bar.close, f"Low {bar.low} > close {bar.close}"


class TestBuildTickData:
    def test_vwap_calculation(self):
        bt = IntradayBacktester()
        bars = synthesize_intraday_bars(
            '2024-03-01', 150.0, 155.0, 148.0, 152.0, 1000000, 149.0)
        tick = bt._build_tick_data(bars, 20, 149.0)
        assert 'vwap' in tick
        assert tick['vwap'] > 0

    def test_or_complete_after_3_bars(self):
        bt = IntradayBacktester()
        bars = synthesize_intraday_bars(
            '2024-03-01', 150.0, 155.0, 148.0, 152.0, 1000000, 149.0)
        tick_2 = bt._build_tick_data(bars, 2, 149.0)
        tick_3 = bt._build_tick_data(bars, 3, 149.0)
        assert tick_2['or_complete'] is True  # 3 bars (0,1,2)
        assert tick_3['or_complete'] is True

    def test_ema_values_present(self):
        bt = IntradayBacktester()
        bars = synthesize_intraday_bars(
            '2024-03-01', 150.0, 155.0, 148.0, 152.0, 1000000, 149.0)
        tick = bt._build_tick_data(bars, 50, 149.0)
        assert 'ema9' in tick
        assert 'ema20' in tick
        assert 'ema50' in tick

    def test_rsi_initialized_after_enough_bars(self):
        bt = IntradayBacktester()
        bars = synthesize_intraday_bars(
            '2024-03-01', 150.0, 155.0, 148.0, 152.0, 1000000, 149.0)
        tick = bt._build_tick_data(bars, 50, 149.0)
        assert tick['rsi_initialized'] is True
        assert 0 <= tick['rsi'] <= 100

    def test_day_high_low_correct(self):
        bt = IntradayBacktester()
        bars = synthesize_intraday_bars(
            '2024-03-01', 150.0, 155.0, 148.0, 152.0, 1000000, 149.0)
        tick = bt._build_tick_data(bars, len(bars) - 1, 149.0)
        # Day high/low should be within the synthesized bars range
        assert tick['day_high'] <= 155.0 + 0.01
        assert tick['day_low'] >= 148.0 - 0.01

    def test_volume_surge_ratio(self):
        bt = IntradayBacktester()
        bars = synthesize_intraday_bars(
            '2024-03-01', 150.0, 155.0, 148.0, 152.0, 1000000, 149.0)
        tick = bt._build_tick_data(bars, 20, 149.0)
        assert 'volume_surge_ratio' in tick
        assert tick['volume_surge_ratio'] > 0

    def test_empty_bars(self):
        bt = IntradayBacktester()
        tick = bt._build_tick_data([], 0, 149.0)
        assert tick == {}


class TestIntradayPosition:
    def test_create_position(self):
        pos = IntradayPosition(
            symbol='AAPL', strategy_id='orb_breakout',
            direction='long', entry_price=150.0,
            stop_price=148.0, target_price=154.0,
            shares=100, entry_time=datetime(2024, 3, 1, 10, 0, tzinfo=ET),
        )
        assert pos.symbol == 'AAPL'
        assert pos.shares == 100


class TestIntradayBacktester:
    def test_init(self):
        bt = IntradayBacktester()
        assert bt.status == 'idle'
        assert bt.progress == 0.0
        assert len(bt.strategy_ids) == 4

    def test_custom_strategies(self):
        bt = IntradayBacktester(strategy_ids=['orb_breakout'])
        assert bt.strategy_ids == ['orb_breakout']

    def test_load_strategies(self):
        bt = IntradayBacktester()
        strategies = bt._load_strategies()
        assert len(strategies) > 0
        assert 'orb_breakout' in strategies

    def test_cancel(self):
        bt = IntradayBacktester()
        bt.cancel()
        assert bt.status == 'cancelled'
        assert bt._cancel is True
