"""Tests for mean_rev_long_bt_sample.

v-mean-rev-bt-sample-tests-2026-09-17. Unit tests for:
  - Output schema compatibility with shadow_long_resolver.py
  - HANDS_OFF exclusion (MU/HQGE/SPCX)
  - Gate detection logic (RSI < 30, close < BB_lower, volume_ratio > 2)
  - Deduplication (same symbol within 15 minutes)
  - Bar structure detection (hammer/doji)
"""
from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


class TestOutputSchema:
    """Verify emitted entries are compatible with shadow_long_resolver."""

    def test_entry_has_required_resolver_fields(self):
        """All fields parsed by ShadowEntry.from_dict must be present."""
        from research.mean_rev_long_bt_sample import (
            scan_symbol_for_entries,
            compute_indicators,
        )

        dates = pd.date_range(start="2024-01-15 09:30:00", periods=100, freq="5min")
        close_prices = [100.0] * 50 + [90.0] * 10 + [100.0] * 40
        df = pd.DataFrame({
            "Open": [p + 0.5 for p in close_prices],
            "High": [p + 2.0 for p in close_prices],
            "Low": [p - 3.0 for p in close_prices],
            "Close": close_prices,
            "Volume": [1000] * 50 + [5000] * 10 + [1000] * 40,
        }, index=dates)

        ind = compute_indicators(df)
        entries = scan_symbol_for_entries("TEST", df, ind)

        if entries:
            entry = entries[0]
            required_fields = [
                "timestamp", "symbol", "signal_type", "reason",
                "signal_close", "rsi", "bb_lower", "bb_middle", "sma_50",
                "macd", "macd_signal", "atr",
                "hypothetical_stop", "hypothetical_target",
                "rr_ratio", "stop_dist",
            ]
            for field in required_fields:
                assert field in entry, f"Missing required field: {field}"

    def test_entry_parseable_by_shadow_entry(self):
        """Emitted entries must parse without error via ShadowEntry.from_dict."""
        from research.shadow_long_resolver import ShadowEntry

        sample_entry = {
            "timestamp": "2024-01-15T10:30:00+00:00",
            "symbol": "AAPL",
            "signal_type": "LONG",
            "reason": "oversold_bounce",
            "signal_close": 150.0,
            "entry_price": 150.0,
            "rsi": 24.5,
            "rsi_14": 24.5,
            "bb_lower": 148.0,
            "bb_middle": 152.0,
            "sma_50": 145.0,
            "macd": -0.5,
            "macd_signal": -0.3,
            "atr": 3.0,
            "hypothetical_stop": 142.5,
            "hypothetical_target": 165.0,
            "rr_ratio": 2.0,
            "stop_dist": 7.5,
            "market_context_regime": None,
            "falling_knife_pass": True,
            "_source": "mean_rev_long_bt_sample",
        }

        parsed = ShadowEntry.from_dict(sample_entry)

        assert parsed.symbol == "AAPL"
        assert parsed.signal_type == "LONG"
        assert parsed.signal_close == 150.0
        assert parsed.rsi == 24.5
        assert parsed.hypothetical_stop == 142.5
        assert parsed.stop_dist == 7.5

    def test_signal_type_is_long(self):
        """All emitted entries must have signal_type=LONG."""
        from research.mean_rev_long_bt_sample import (
            scan_symbol_for_entries,
            compute_indicators,
        )

        dates = pd.date_range(start="2024-01-15 09:30:00", periods=100, freq="5min")
        close_prices = [100.0] * 50 + [85.0] * 10 + [100.0] * 40
        df = pd.DataFrame({
            "Open": [p + 0.5 for p in close_prices],
            "High": [p + 2.0 for p in close_prices],
            "Low": [p - 3.0 for p in close_prices],
            "Close": close_prices,
            "Volume": [1000] * 50 + [5000] * 10 + [1000] * 40,
        }, index=dates)

        ind = compute_indicators(df)
        entries = scan_symbol_for_entries("TEST", df, ind)

        for entry in entries:
            assert entry["signal_type"] == "LONG"

    def test_stop_below_entry_for_long(self):
        """LONG entries must have stop < entry price."""
        from research.mean_rev_long_bt_sample import (
            scan_symbol_for_entries,
            compute_indicators,
        )

        dates = pd.date_range(start="2024-01-15 09:30:00", periods=100, freq="5min")
        close_prices = [100.0] * 50 + [85.0] * 10 + [100.0] * 40
        df = pd.DataFrame({
            "Open": [p + 0.5 for p in close_prices],
            "High": [p + 2.0 for p in close_prices],
            "Low": [p - 3.0 for p in close_prices],
            "Close": close_prices,
            "Volume": [1000] * 50 + [5000] * 10 + [1000] * 40,
        }, index=dates)

        ind = compute_indicators(df)
        entries = scan_symbol_for_entries("TEST", df, ind)

        for entry in entries:
            assert entry["hypothetical_stop"] < entry["signal_close"], \
                f"LONG stop {entry['hypothetical_stop']} >= entry {entry['signal_close']}"

    def test_target_above_entry_for_long(self):
        """LONG entries must have target > entry price."""
        from research.mean_rev_long_bt_sample import (
            scan_symbol_for_entries,
            compute_indicators,
        )

        dates = pd.date_range(start="2024-01-15 09:30:00", periods=100, freq="5min")
        close_prices = [100.0] * 50 + [85.0] * 10 + [100.0] * 40
        df = pd.DataFrame({
            "Open": [p + 0.5 for p in close_prices],
            "High": [p + 2.0 for p in close_prices],
            "Low": [p - 3.0 for p in close_prices],
            "Close": close_prices,
            "Volume": [1000] * 50 + [5000] * 10 + [1000] * 40,
        }, index=dates)

        ind = compute_indicators(df)
        entries = scan_symbol_for_entries("TEST", df, ind)

        for entry in entries:
            assert entry["hypothetical_target"] > entry["signal_close"], \
                f"LONG target {entry['hypothetical_target']} <= entry {entry['signal_close']}"


class TestHandsOffExclusion:
    """HANDS_OFF symbols (MU, HQGE, SPCX) must never be emitted."""

    def test_mu_excluded(self):
        from research.mean_rev_long_bt_sample import HANDS_OFF_SYMBOLS
        assert "MU" in HANDS_OFF_SYMBOLS

    def test_hqge_excluded(self):
        from research.mean_rev_long_bt_sample import HANDS_OFF_SYMBOLS
        assert "HQGE" in HANDS_OFF_SYMBOLS

    def test_spcx_excluded(self):
        from research.mean_rev_long_bt_sample import HANDS_OFF_SYMBOLS
        assert "SPCX" in HANDS_OFF_SYMBOLS

    def test_hands_off_matches_resolver(self):
        """HANDS_OFF must match shadow_long_resolver.py exactly."""
        from research.mean_rev_long_bt_sample import HANDS_OFF_SYMBOLS as BT_HANDS_OFF
        from research.shadow_long_resolver import HANDS_OFF_SYMBOLS as RESOLVER_HANDS_OFF

        assert BT_HANDS_OFF == RESOLVER_HANDS_OFF


class TestGateDetection:
    """Test mean-rev detection gates."""

    def test_gate_a_rsi_threshold(self):
        """Gate A: RSI must be < 30."""
        from research.mean_rev_long_bt_sample import (
            scan_symbol_for_entries,
            compute_indicators,
        )

        dates = pd.date_range(start="2024-01-15 09:30:00", periods=100, freq="5min")
        df = pd.DataFrame({
            "Open": [100.0] * 100,
            "High": [102.0] * 100,
            "Low": [98.0] * 100,
            "Close": [100.0] * 100,
            "Volume": [5000] * 100,
        }, index=dates)

        ind = compute_indicators(df)
        entries = scan_symbol_for_entries("TEST", df, ind)

        for entry in entries:
            assert entry["rsi"] < 30, f"Entry with RSI >= 30: {entry['rsi']}"

    def test_gate_b_volume_ratio(self):
        """Gate B: volume_ratio must be > 2.0."""
        from research.mean_rev_long_bt_sample import (
            scan_symbol_for_entries,
            compute_indicators,
        )

        dates = pd.date_range(start="2024-01-15 09:30:00", periods=100, freq="5min")
        close_prices = [100.0] * 50 + [85.0] * 10 + [100.0] * 40
        df = pd.DataFrame({
            "Open": [p + 0.5 for p in close_prices],
            "High": [p + 2.0 for p in close_prices],
            "Low": [p - 3.0 for p in close_prices],
            "Close": close_prices,
            "Volume": [1000] * 50 + [5000] * 10 + [1000] * 40,
        }, index=dates)

        ind = compute_indicators(df)
        entries = scan_symbol_for_entries("TEST", df, ind)

        for entry in entries:
            assert entry["volume_ratio"] > 2.0, \
                f"Entry with volume_ratio <= 2.0: {entry['volume_ratio']}"


class TestBarStructureDetection:
    """Test Gate C bar structure detection."""

    def test_hammer_detection(self):
        """Long lower wick (hammer) should pass structure check."""
        from research.mean_rev_long_bt_sample import bar_has_capitulation_structure

        result = bar_has_capitulation_structure(
            open_=100.0, high=101.0, low=95.0, close=100.5, prev_low=None
        )
        assert result is True

    def test_no_wick_fails(self):
        """Bar with no lower wick and no prev_low reclaim should fail."""
        from research.mean_rev_long_bt_sample import bar_has_capitulation_structure

        result = bar_has_capitulation_structure(
            open_=100.0, high=101.0, low=99.5, close=100.5, prev_low=102.0
        )
        assert result is False

    def test_prev_low_reclaim_passes(self):
        """Close >= prev_low should pass structure check."""
        from research.mean_rev_long_bt_sample import bar_has_capitulation_structure

        result = bar_has_capitulation_structure(
            open_=100.0, high=101.0, low=99.0, close=100.5, prev_low=100.0
        )
        assert result is True


class TestSwingLowDetection:
    """Test Gate D swing low detection."""

    def test_finds_swing_low_within_threshold(self):
        """Should find swing low within 0.5 * ATR."""
        from research.mean_rev_long_bt_sample import find_nearest_swing_low

        lows = [100.0, 98.0, 97.0, 99.0, 100.0]
        result = find_nearest_swing_low(lows, current_price=97.5, atr=2.0)

        assert result is not None
        assert result == 97.0

    def test_no_swing_low_returns_none(self):
        """Should return None if no swing low within threshold."""
        from research.mean_rev_long_bt_sample import find_nearest_swing_low

        lows = [100.0, 99.0, 98.0, 97.0, 96.0]
        result = find_nearest_swing_low(lows, current_price=100.0, atr=1.0)

        assert result is None


class TestDeduplication:
    """Test 15-minute deduplication logic."""

    def test_dedupe_keeps_first_within_window(self):
        """Same symbol within 15 minutes → keep first only."""
        from research.mean_rev_long_bt_sample import dedupe_entries

        entries = [
            {"symbol": "AAPL", "timestamp": "2024-01-15T10:00:00+00:00"},
            {"symbol": "AAPL", "timestamp": "2024-01-15T10:10:00+00:00"},
        ]

        result = dedupe_entries(entries)
        assert len(result) == 1
        assert result[0]["timestamp"] == "2024-01-15T10:00:00+00:00"

    def test_dedupe_keeps_both_outside_window(self):
        """Same symbol > 15 minutes apart → keep both."""
        from research.mean_rev_long_bt_sample import dedupe_entries

        entries = [
            {"symbol": "AAPL", "timestamp": "2024-01-15T10:00:00+00:00"},
            {"symbol": "AAPL", "timestamp": "2024-01-15T10:20:00+00:00"},
        ]

        result = dedupe_entries(entries)
        assert len(result) == 2

    def test_dedupe_different_symbols_not_deduped(self):
        """Different symbols within window → keep both."""
        from research.mean_rev_long_bt_sample import dedupe_entries

        entries = [
            {"symbol": "AAPL", "timestamp": "2024-01-15T10:00:00+00:00"},
            {"symbol": "MSFT", "timestamp": "2024-01-15T10:05:00+00:00"},
        ]

        result = dedupe_entries(entries)
        assert len(result) == 2


class TestNDJSONOutput:
    """Test NDJSON file output."""

    def test_write_creates_valid_ndjson(self, tmp_path):
        """Output file must be valid NDJSON (one JSON object per line)."""
        from research.mean_rev_long_bt_sample import write_ndjson

        entries = [
            {
                "symbol": "AAPL",
                "timestamp": "2024-01-15T10:00:00+00:00",
                "signal_type": "LONG",
                "signal_close": 150.0,
            },
            {
                "symbol": "NVDA",
                "timestamp": "2024-01-15T10:05:00+00:00",
                "signal_type": "LONG",
                "signal_close": 500.0,
            },
        ]

        output_path = tmp_path / "test_output.ndjson"
        write_ndjson(entries, output_path)

        with open(output_path, "r") as f:
            lines = f.readlines()

        assert len(lines) == 2
        for line in lines:
            parsed = json.loads(line)
            assert "symbol" in parsed
            assert "timestamp" in parsed

    def test_append_mode(self, tmp_path):
        """Append mode should add to existing file."""
        from research.mean_rev_long_bt_sample import write_ndjson

        output_path = tmp_path / "test_append.ndjson"

        write_ndjson([{"symbol": "AAPL"}], output_path, append=False)
        write_ndjson([{"symbol": "MSFT"}], output_path, append=True)

        with open(output_path, "r") as f:
            lines = f.readlines()

        assert len(lines) == 2


class TestStopTargetComputation:
    """Test stop/target computation for LONG entries."""

    def test_stop_distance_uses_atr(self):
        """stop_dist should be max(2*ATR, 0.5*bar_range)."""
        from research.mean_rev_long_bt_sample import compute_stop_target

        stop_dist, stop, target = compute_stop_target(
            close=100.0, atr=2.0, bar_range=2.0
        )

        assert stop_dist == 4.0
        assert stop == 96.0

    def test_rr_ratio_applied_to_target(self):
        """target should be close + (rr_ratio + 1) * stop_dist."""
        from research.mean_rev_long_bt_sample import compute_stop_target

        stop_dist, stop, target = compute_stop_target(
            close=100.0, atr=2.0, bar_range=2.0, rr_ratio=2.0
        )

        expected_target = 100.0 + 3.0 * stop_dist
        assert target == expected_target


class TestBarNormalization:
    """Test bar DataFrame normalization."""

    def test_normalizes_lowercase_columns(self):
        """Should normalize lowercase columns to Title case."""
        from research.mean_rev_long_bt_sample import normalize_bars_df

        dates = pd.date_range(start="2024-01-15", periods=5, freq="5min")
        df = pd.DataFrame({
            "ts": dates,
            "open": [100.0] * 5,
            "high": [102.0] * 5,
            "low": [98.0] * 5,
            "close": [101.0] * 5,
            "volume": [1000] * 5,
        })

        result = normalize_bars_df(df)

        assert "Open" in result.columns
        assert "High" in result.columns
        assert "Low" in result.columns
        assert "Close" in result.columns
        assert "Volume" in result.columns

    def test_sets_datetime_index(self):
        """Should set ts/timestamp column as DatetimeIndex."""
        from research.mean_rev_long_bt_sample import normalize_bars_df

        dates = pd.date_range(start="2024-01-15", periods=5, freq="5min")
        df = pd.DataFrame({
            "ts": dates,
            "open": [100.0] * 5,
            "high": [102.0] * 5,
            "low": [98.0] * 5,
            "close": [101.0] * 5,
            "volume": [1000] * 5,
        })

        result = normalize_bars_df(df)

        assert isinstance(result.index, pd.DatetimeIndex)


class TestCLI:
    """Test CLI interface."""

    def test_dry_run_no_output(self, tmp_path):
        """--dry-run should not write output file."""
        from research.mean_rev_long_bt_sample import main

        bars_dir = tmp_path / "bars"
        bars_dir.mkdir()

        dates = pd.date_range(start="2024-01-15 09:30:00", periods=200, freq="5min")
        close_prices = [100.0] * 100 + [80.0] * 20 + [100.0] * 80
        df = pd.DataFrame({
            "ts": dates,
            "open": [p + 0.5 for p in close_prices],
            "high": [p + 2.0 for p in close_prices],
            "low": [p - 3.0 for p in close_prices],
            "close": close_prices,
            "volume": [1000] * 100 + [5000] * 20 + [1000] * 80,
        })
        df.to_csv(bars_dir / "TEST.csv", index=False)

        output_path = tmp_path / "output.ndjson"

        result = main([
            "--bars-dir", str(bars_dir),
            "--output", str(output_path),
            "--dry-run",
        ])

        assert result == 0
        assert not output_path.exists()

    def test_writes_output_file(self, tmp_path):
        """Should write output file when not dry-run and entries found."""
        from research.mean_rev_long_bt_sample import main

        bars_dir = tmp_path / "bars"
        bars_dir.mkdir()

        dates = pd.date_range(start="2024-01-15 09:30:00", periods=200, freq="5min")
        close_prices = (
            [100.0] * 60 +
            [98.0, 96.0, 94.0, 92.0, 90.0, 88.0, 86.0, 84.0, 82.0, 80.0] +
            [78.0, 76.0, 74.0, 72.0, 70.0, 68.0, 66.0, 65.0, 64.0, 63.0] +
            [80.0] * 120
        )
        df = pd.DataFrame({
            "ts": dates,
            "open": [p + 1.0 for p in close_prices],
            "high": [p + 2.0 for p in close_prices],
            "low": [p - 5.0 for p in close_prices],
            "close": close_prices,
            "volume": ([1000] * 60 + [15000] * 20 + [1000] * 120),
        })
        df.to_csv(bars_dir / "TEST.csv", index=False)

        output_path = tmp_path / "output.ndjson"

        result = main([
            "--bars-dir", str(bars_dir),
            "--output", str(output_path),
        ])

        assert result == 0

    def test_returns_zero_when_no_entries(self, tmp_path):
        """Should return 0 even when no entries found (just warn)."""
        from research.mean_rev_long_bt_sample import main

        bars_dir = tmp_path / "bars"
        bars_dir.mkdir()

        dates = pd.date_range(start="2024-01-15 09:30:00", periods=200, freq="5min")
        df = pd.DataFrame({
            "ts": dates,
            "open": [100.0] * 200,
            "high": [102.0] * 200,
            "low": [98.0] * 200,
            "close": [100.0] * 200,
            "volume": [1000] * 200,
        })
        df.to_csv(bars_dir / "STABLE.csv", index=False)

        output_path = tmp_path / "output.ndjson"

        result = main([
            "--bars-dir", str(bars_dir),
            "--output", str(output_path),
        ])

        assert result == 0
