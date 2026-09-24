"""Tests for shadow_long_resolver.

v-shadow-long-resolver-tests-2026-09-17. Unit tests for:
  - R-multiple math (LONG: positive R = profit)
  - Dedupe logic (same symbol within 15m)
  - Hands-off exclusions (MU/HQGE/SPCX)
  - Barrier resolution (stop/target/timeout/flatten)
"""
from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import pytest


class TestRMath:
    """R-multiple calculation for LONG trades."""

    def test_r_positive_when_price_rises(self):
        """LONG profits when exit > entry → positive R."""
        from research.shadow_long_resolver import ResolvedTrade

        entry = 100.0
        exit = 105.0
        stop_dist = 5.0
        expected_r = 1.0

        trade = ResolvedTrade(
            symbol="TEST",
            entry_time=datetime.now(timezone.utc),
            entry_price=entry,
            exit_time=datetime.now(timezone.utc),
            exit_price=exit,
            exit_reason="target",
            stop=entry - stop_dist,
            target=entry + (2 * stop_dist),
            stop_dist=stop_dist,
            hold_bars=10,
            r_multiple=expected_r,
            return_pct=5.0,
            regime="neutral",
            falling_knife_pass=True,
        )

        assert trade.r_multiple == expected_r

    def test_r_negative_when_price_falls(self):
        """LONG loses when exit < entry → negative R."""
        entry = 100.0
        exit = 95.0
        stop_dist = 5.0
        expected_r = -1.0

        from research.shadow_long_resolver import ResolvedTrade

        trade = ResolvedTrade(
            symbol="TEST",
            entry_time=datetime.now(timezone.utc),
            entry_price=entry,
            exit_time=datetime.now(timezone.utc),
            exit_price=exit,
            exit_reason="stop",
            stop=entry - stop_dist,
            target=entry + (2 * stop_dist),
            stop_dist=stop_dist,
            hold_bars=5,
            r_multiple=expected_r,
            return_pct=-5.0,
            regime="neutral",
            falling_knife_pass=True,
        )

        assert trade.r_multiple == expected_r

    def test_r_calculation_in_resolve_entry(self):
        """Test R calculation inside resolve_entry function."""
        from research.shadow_long_resolver import ShadowEntry, resolve_entry

        entry = ShadowEntry(
            timestamp=datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc),
            symbol="TEST",
            signal_type="LONG",
            reason="oversold_bounce",
            signal_close=100.0,
            rsi=25.0,
            bb_lower=98.0,
            bb_middle=102.0,
            sma_50=105.0,
            macd=-0.5,
            macd_signal=-0.3,
            atr=2.0,
            hypothetical_stop=95.0,
            hypothetical_target=110.0,
            rr_ratio=2.0,
            stop_dist=5.0,
            raw={"falling_knife_pass": True},
        )

        def mock_bars_loader(symbol, start, lookforward_days):
            dates = pd.date_range(start=start + timedelta(minutes=5), periods=10, freq="5min")
            return pd.DataFrame({
                "Open": [100] * 10,
                "High": [111] * 10,
                "Low": [99] * 10,
                "Close": [110] * 10,
            }, index=dates)

        trade = resolve_entry(entry, mock_bars_loader)

        assert trade.exit_reason == "target"
        assert trade.exit_price == 110.0
        assert trade.r_multiple > 0


class TestDedupe:
    """Deduplication: same symbol within 15 minutes → keep first."""

    def test_dedupe_keeps_first_within_window(self):
        from research.shadow_long_resolver import ShadowEntry, filter_and_dedupe

        base = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)
        entries = [
            ShadowEntry(
                timestamp=base,
                symbol="AAPL",
                signal_type="LONG",
                reason="test",
                signal_close=100.0,
                rsi=25, bb_lower=98, bb_middle=102, sma_50=105,
                macd=-0.5, macd_signal=-0.3, atr=2.0,
                hypothetical_stop=95, hypothetical_target=110,
                rr_ratio=2.0, stop_dist=5.0,
            ),
            ShadowEntry(
                timestamp=base + timedelta(minutes=10),
                symbol="AAPL",
                signal_type="LONG",
                reason="test_dupe",
                signal_close=101.0,
                rsi=26, bb_lower=99, bb_middle=103, sma_50=106,
                macd=-0.4, macd_signal=-0.2, atr=2.1,
                hypothetical_stop=96, hypothetical_target=111,
                rr_ratio=2.0, stop_dist=5.0,
            ),
        ]

        result = filter_and_dedupe(entries)

        assert len(result) == 1
        assert result[0].signal_close == 100.0

    def test_dedupe_keeps_both_outside_window(self):
        from research.shadow_long_resolver import ShadowEntry, filter_and_dedupe

        base = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)
        entries = [
            ShadowEntry(
                timestamp=base,
                symbol="AAPL",
                signal_type="LONG",
                reason="test1",
                signal_close=100.0,
                rsi=25, bb_lower=98, bb_middle=102, sma_50=105,
                macd=-0.5, macd_signal=-0.3, atr=2.0,
                hypothetical_stop=95, hypothetical_target=110,
                rr_ratio=2.0, stop_dist=5.0,
            ),
            ShadowEntry(
                timestamp=base + timedelta(minutes=20),
                symbol="AAPL",
                signal_type="LONG",
                reason="test2",
                signal_close=101.0,
                rsi=26, bb_lower=99, bb_middle=103, sma_50=106,
                macd=-0.4, macd_signal=-0.2, atr=2.1,
                hypothetical_stop=96, hypothetical_target=111,
                rr_ratio=2.0, stop_dist=5.0,
            ),
        ]

        result = filter_and_dedupe(entries)

        assert len(result) == 2

    def test_dedupe_different_symbols_not_deduped(self):
        from research.shadow_long_resolver import ShadowEntry, filter_and_dedupe

        base = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)
        entries = [
            ShadowEntry(
                timestamp=base,
                symbol="AAPL",
                signal_type="LONG",
                reason="test1",
                signal_close=100.0,
                rsi=25, bb_lower=98, bb_middle=102, sma_50=105,
                macd=-0.5, macd_signal=-0.3, atr=2.0,
                hypothetical_stop=95, hypothetical_target=110,
                rr_ratio=2.0, stop_dist=5.0,
            ),
            ShadowEntry(
                timestamp=base + timedelta(minutes=5),
                symbol="MSFT",
                signal_type="LONG",
                reason="test2",
                signal_close=200.0,
                rsi=26, bb_lower=197, bb_middle=203, sma_50=206,
                macd=-0.6, macd_signal=-0.4, atr=4.0,
                hypothetical_stop=190, hypothetical_target=220,
                rr_ratio=2.0, stop_dist=10.0,
            ),
        ]

        result = filter_and_dedupe(entries)

        assert len(result) == 2


class TestHandsOffExclusion:
    """Hands-off forever symbols: MU, HQGE, SPCX."""

    def test_mu_excluded(self):
        from research.shadow_long_resolver import ShadowEntry, filter_and_dedupe

        entries = [
            ShadowEntry(
                timestamp=datetime.now(timezone.utc),
                symbol="MU",
                signal_type="LONG",
                reason="test",
                signal_close=100.0,
                rsi=25, bb_lower=98, bb_middle=102, sma_50=105,
                macd=-0.5, macd_signal=-0.3, atr=2.0,
                hypothetical_stop=95, hypothetical_target=110,
                rr_ratio=2.0, stop_dist=5.0,
            ),
        ]

        result = filter_and_dedupe(entries)
        assert len(result) == 0

    def test_snap_not_excluded(self):
        """SNAP removed from permanent hands-off 2026-09-14 — now tradeable."""
        from research.shadow_long_resolver import ShadowEntry, filter_and_dedupe

        entries = [
            ShadowEntry(
                timestamp=datetime.now(timezone.utc),
                symbol="SNAP",
                signal_type="LONG",
                reason="test",
                signal_close=10.0,
                rsi=25, bb_lower=9.8, bb_middle=10.2, sma_50=10.5,
                macd=-0.05, macd_signal=-0.03, atr=0.2,
                hypothetical_stop=9.5, hypothetical_target=11.0,
                rr_ratio=2.0, stop_dist=0.5,
            ),
        ]

        result = filter_and_dedupe(entries)
        assert len(result) == 1, "SNAP should NOT be excluded — removed from denylist 2026-09-14"

    def test_spcx_excluded(self):
        from research.shadow_long_resolver import ShadowEntry, filter_and_dedupe

        entries = [
            ShadowEntry(
                timestamp=datetime.now(timezone.utc),
                symbol="SPCX",
                signal_type="LONG",
                reason="test",
                signal_close=50.0,
                rsi=25, bb_lower=49, bb_middle=51, sma_50=52,
                macd=-0.3, macd_signal=-0.2, atr=1.0,
                hypothetical_stop=47.5, hypothetical_target=55,
                rr_ratio=2.0, stop_dist=2.5,
            ),
        ]

        result = filter_and_dedupe(entries)
        assert len(result) == 0

    def test_hqge_excluded(self):
        from research.shadow_long_resolver import ShadowEntry, filter_and_dedupe

        entries = [
            ShadowEntry(
                timestamp=datetime.now(timezone.utc),
                symbol="HQGE",
                signal_type="LONG",
                reason="test",
                signal_close=25.0,
                rsi=25, bb_lower=24.5, bb_middle=25.5, sma_50=26,
                macd=-0.2, macd_signal=-0.1, atr=0.5,
                hypothetical_stop=23.75, hypothetical_target=27.5,
                rr_ratio=2.0, stop_dist=1.25,
            ),
        ]

        result = filter_and_dedupe(entries)
        assert len(result) == 0

    def test_all_hands_off_excluded(self):
        from research.shadow_long_resolver import ShadowEntry, filter_and_dedupe, HANDS_OFF_SYMBOLS

        base = datetime.now(timezone.utc)
        entries = [
            ShadowEntry(
                timestamp=base + timedelta(minutes=i),
                symbol=sym,
                signal_type="LONG",
                reason="test",
                signal_close=100.0,
                rsi=25, bb_lower=98, bb_middle=102, sma_50=105,
                macd=-0.5, macd_signal=-0.3, atr=2.0,
                hypothetical_stop=95, hypothetical_target=110,
                rr_ratio=2.0, stop_dist=5.0,
            )
            for i, sym in enumerate(HANDS_OFF_SYMBOLS)
        ]
        entries.append(ShadowEntry(
            timestamp=base + timedelta(minutes=10),
            symbol="AAPL",
            signal_type="LONG",
            reason="test",
            signal_close=150.0,
            rsi=25, bb_lower=148, bb_middle=152, sma_50=155,
            macd=-0.5, macd_signal=-0.3, atr=3.0,
            hypothetical_stop=142.5, hypothetical_target=165,
            rr_ratio=2.0, stop_dist=7.5,
        ))

        result = filter_and_dedupe(entries)

        assert len(result) == 1
        assert result[0].symbol == "AAPL"


class TestFixtureNDJSON:
    """Test loading and parsing NDJSON fixture."""

    def test_load_tiny_ndjson(self):
        from research.shadow_long_resolver import load_shadow_log

        with tempfile.NamedTemporaryFile(mode='w', suffix='.ndjson', delete=False) as f:
            lines = [
                {
                    "timestamp": "2024-01-15T10:30:00+00:00",
                    "symbol": "AAPL",
                    "signal_type": "LONG",
                    "reason": "oversold_bounce",
                    "signal_close": 150.0,
                    "rsi": 24.5,
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
                    "falling_knife_pass": True,
                },
                {
                    "timestamp": "2024-01-15T11:00:00+00:00",
                    "symbol": "NVDA",
                    "signal_type": "LONG",
                    "reason": "oversold_bounce",
                    "signal_close": 500.0,
                    "rsi": 22.2,
                    "bb_lower": 490.0,
                    "bb_middle": 510.0,
                    "sma_50": 480.0,
                    "macd": -2.0,
                    "macd_signal": -1.5,
                    "atr": 10.0,
                    "hypothetical_stop": 475.0,
                    "hypothetical_target": 550.0,
                    "rr_ratio": 2.0,
                    "stop_dist": 25.0,
                    "falling_knife_pass": False,
                },
            ]
            for line in lines:
                f.write(json.dumps(line) + "\n")
            f.flush()
            path = Path(f.name)

        try:
            entries = load_shadow_log(path)

            assert len(entries) == 2
            assert entries[0].symbol == "AAPL"
            assert entries[0].signal_close == 150.0
            assert entries[0].hypothetical_stop == 142.5
            assert entries[1].symbol == "NVDA"
            assert entries[1].rsi == 22.2
        finally:
            path.unlink()

    def test_load_handles_malformed_lines(self):
        from research.shadow_long_resolver import load_shadow_log

        with tempfile.NamedTemporaryFile(mode='w', suffix='.ndjson', delete=False) as f:
            f.write('{"symbol": "AAPL", "signal_close": 100, "timestamp": "2024-01-15T10:30:00+00:00"}\n')
            f.write('invalid json line\n')
            f.write('{"symbol": "MSFT", "signal_close": 200, "timestamp": "2024-01-15T11:00:00+00:00"}\n')
            f.write('\n')
            f.flush()
            path = Path(f.name)

        try:
            entries = load_shadow_log(path)
            assert len(entries) == 2
        finally:
            path.unlink()


class TestBarrierResolution:
    """Test stop/target/timeout/flatten exit logic."""

    def test_stop_hit_first(self):
        from research.shadow_long_resolver import ShadowEntry, resolve_entry

        entry = ShadowEntry(
            timestamp=datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc),
            symbol="TEST",
            signal_type="LONG",
            reason="test",
            signal_close=100.0,
            rsi=25, bb_lower=98, bb_middle=102, sma_50=105,
            macd=-0.5, macd_signal=-0.3, atr=2.0,
            hypothetical_stop=95.0,
            hypothetical_target=110.0,
            rr_ratio=2.0, stop_dist=5.0,
        )

        def mock_bars_loader(symbol, start, lookforward_days):
            dates = pd.date_range(start=start + timedelta(minutes=5), periods=10, freq="5min")
            return pd.DataFrame({
                "Open": [100, 99, 98, 97, 96, 95, 94, 93, 92, 91],
                "High": [101, 100, 99, 98, 97, 96, 95, 94, 93, 92],
                "Low": [99, 98, 97, 96, 94, 93, 92, 91, 90, 89],
                "Close": [100, 99, 98, 97, 95, 94, 93, 92, 91, 90],
            }, index=dates)

        trade = resolve_entry(entry, mock_bars_loader)

        assert trade.exit_reason == "stop"
        assert trade.exit_price == 95.0
        assert trade.r_multiple < 0

    def test_target_hit_first(self):
        from research.shadow_long_resolver import ShadowEntry, resolve_entry

        entry = ShadowEntry(
            timestamp=datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc),
            symbol="TEST",
            signal_type="LONG",
            reason="test",
            signal_close=100.0,
            rsi=25, bb_lower=98, bb_middle=102, sma_50=105,
            macd=-0.5, macd_signal=-0.3, atr=2.0,
            hypothetical_stop=95.0,
            hypothetical_target=110.0,
            rr_ratio=2.0, stop_dist=5.0,
        )

        def mock_bars_loader(symbol, start, lookforward_days):
            dates = pd.date_range(start=start + timedelta(minutes=5), periods=10, freq="5min")
            return pd.DataFrame({
                "Open": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
                "High": [101, 102, 103, 104, 105, 111, 112, 113, 114, 115],
                "Low": [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
                "Close": [100, 101, 102, 103, 104, 110, 111, 112, 113, 114],
            }, index=dates)

        trade = resolve_entry(entry, mock_bars_loader)

        assert trade.exit_reason == "target"
        assert trade.exit_price == 110.0
        assert trade.r_multiple > 0

    def test_timeout_when_no_barrier_hit(self):
        from research.shadow_long_resolver import ShadowEntry, resolve_entry, MAX_HOLD_BARS

        entry = ShadowEntry(
            timestamp=datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc),
            symbol="TEST",
            signal_type="LONG",
            reason="test",
            signal_close=100.0,
            rsi=25, bb_lower=98, bb_middle=102, sma_50=105,
            macd=-0.5, macd_signal=-0.3, atr=2.0,
            hypothetical_stop=85.0,
            hypothetical_target=115.0,
            rr_ratio=2.0, stop_dist=15.0,
        )

        def mock_bars_loader(symbol, start, lookforward_days):
            dates = pd.date_range(start=start + timedelta(minutes=5), periods=70, freq="5min")
            return pd.DataFrame({
                "Open": [100] * 70,
                "High": [102] * 70,
                "Low": [98] * 70,
                "Close": [100] * 70,
            }, index=dates)

        trade = resolve_entry(entry, mock_bars_loader)

        assert trade.exit_reason == "timeout"
        assert trade.hold_bars == MAX_HOLD_BARS

    def test_no_bars_returns_no_bars_reason(self):
        from research.shadow_long_resolver import ShadowEntry, resolve_entry

        entry = ShadowEntry(
            timestamp=datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc),
            symbol="TEST",
            signal_type="LONG",
            reason="test",
            signal_close=100.0,
            rsi=25, bb_lower=98, bb_middle=102, sma_50=105,
            macd=-0.5, macd_signal=-0.3, atr=2.0,
            hypothetical_stop=95.0,
            hypothetical_target=110.0,
            rr_ratio=2.0, stop_dist=5.0,
        )

        def mock_bars_loader(symbol, start, lookforward_days):
            return pd.DataFrame()

        trade = resolve_entry(entry, mock_bars_loader)

        assert trade.exit_reason == "no_bars"
        assert trade.hold_bars == 0


class TestMetrics:
    """Test Stage A metrics computation."""

    def test_compute_metrics_basic(self):
        from research.shadow_long_resolver import ResolvedTrade, compute_metrics

        base = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)
        trades = [
            ResolvedTrade(
                symbol="AAPL", entry_time=base, entry_price=100,
                exit_time=base + timedelta(hours=1), exit_price=105,
                exit_reason="target", stop=95, target=110, stop_dist=5,
                hold_bars=12, r_multiple=1.0, return_pct=5.0,
                regime="neutral", falling_knife_pass=True,
            ),
            ResolvedTrade(
                symbol="MSFT", entry_time=base + timedelta(hours=2), entry_price=200,
                exit_time=base + timedelta(hours=3), exit_price=190,
                exit_reason="stop", stop=190, target=220, stop_dist=10,
                hold_bars=6, r_multiple=-1.0, return_pct=-5.0,
                regime="neutral", falling_knife_pass=True,
            ),
        ]

        summary = compute_metrics(trades)

        assert summary.n_trades == 2
        assert summary.n_wins == 1
        assert summary.n_losses == 1
        assert summary.win_rate == 0.5
        assert summary.total_r == 0.0

    def test_exclude_risk_off_for_promotion_book(self):
        from research.shadow_long_resolver import ResolvedTrade, compute_metrics

        base = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)
        trades = [
            ResolvedTrade(
                symbol="AAPL", entry_time=base, entry_price=100,
                exit_time=base + timedelta(hours=1), exit_price=105,
                exit_reason="target", stop=95, target=110, stop_dist=5,
                hold_bars=12, r_multiple=1.0, return_pct=5.0,
                regime="neutral", falling_knife_pass=True,
            ),
            ResolvedTrade(
                symbol="MSFT", entry_time=base + timedelta(hours=2), entry_price=200,
                exit_time=base + timedelta(hours=3), exit_price=190,
                exit_reason="stop", stop=190, target=220, stop_dist=10,
                hold_bars=6, r_multiple=-2.0, return_pct=-5.0,
                regime="risk_off", falling_knife_pass=True,
            ),
        ]

        summary = compute_metrics(trades, exclude_regimes={"risk_off"})

        assert summary.n_trades == 1
        assert summary.n_wins == 1
        assert summary.n_losses == 0
        assert "risk_off" in summary.by_regime
        assert "neutral" in summary.by_regime


class TestFallingKnifePass:
    """Test falling_knife_pass flag handling."""

    def test_reads_falling_knife_from_raw(self):
        from research.shadow_long_resolver import ShadowEntry, resolve_entry

        entry = ShadowEntry(
            timestamp=datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc),
            symbol="TEST",
            signal_type="LONG",
            reason="test",
            signal_close=100.0,
            rsi=25, bb_lower=98, bb_middle=102, sma_50=105,
            macd=-0.5, macd_signal=-0.3, atr=2.0,
            hypothetical_stop=95.0,
            hypothetical_target=110.0,
            rr_ratio=2.0, stop_dist=5.0,
            raw={"falling_knife_pass": False},
        )

        def mock_bars_loader(symbol, start, lookforward_days):
            dates = pd.date_range(start=start + timedelta(minutes=5), periods=2, freq="5min")
            return pd.DataFrame({
                "Open": [100, 100],
                "High": [111, 111],
                "Low": [99, 99],
                "Close": [110, 110],
            }, index=dates)

        trade = resolve_entry(entry, mock_bars_loader)

        assert trade.falling_knife_pass is False

    def test_defaults_to_true_for_legacy(self):
        from research.shadow_long_resolver import ShadowEntry, resolve_entry

        entry = ShadowEntry(
            timestamp=datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc),
            symbol="TEST",
            signal_type="LONG",
            reason="test",
            signal_close=100.0,
            rsi=25, bb_lower=98, bb_middle=102, sma_50=105,
            macd=-0.5, macd_signal=-0.3, atr=2.0,
            hypothetical_stop=95.0,
            hypothetical_target=110.0,
            rr_ratio=2.0, stop_dist=5.0,
            raw={},
        )

        def mock_bars_loader(symbol, start, lookforward_days):
            dates = pd.date_range(start=start + timedelta(minutes=5), periods=2, freq="5min")
            return pd.DataFrame({
                "Open": [100, 100],
                "High": [111, 111],
                "Low": [99, 99],
                "Close": [110, 110],
            }, index=dates)

        trade = resolve_entry(entry, mock_bars_loader)

        assert trade.falling_knife_pass is True


class TestFileBarsLoader:
    """Test file-based bars loader."""

    def test_loads_from_parquet(self, tmp_path):
        from research.shadow_long_resolver import make_file_bars_loader

        dates = pd.date_range(start="2024-01-15 10:35:00+00:00", periods=20, freq="5min")
        df = pd.DataFrame({
            "ts": dates,
            "open": [100.0] * 20,
            "high": [102.0] * 20,
            "low": [98.0] * 20,
            "close": [101.0] * 20,
            "volume": [1000] * 20,
        })
        df.to_parquet(tmp_path / "TEST.parquet", index=False)

        loader = make_file_bars_loader(tmp_path)
        start = datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc)
        result = loader("TEST", start, lookforward_days=1)

        assert not result.empty
        assert "Close" in result.columns
        assert len(result) > 0

    def test_loads_from_csv(self, tmp_path):
        from research.shadow_long_resolver import make_file_bars_loader

        dates = pd.date_range(start="2024-01-15 10:35:00+00:00", periods=20, freq="5min")
        df = pd.DataFrame({
            "ts": dates,
            "open": [100.0] * 20,
            "high": [102.0] * 20,
            "low": [98.0] * 20,
            "close": [101.0] * 20,
            "volume": [1000] * 20,
        })
        df.to_csv(tmp_path / "TEST.csv", index=False)

        loader = make_file_bars_loader(tmp_path)
        start = datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc)
        result = loader("TEST", start, lookforward_days=1)

        assert not result.empty
        assert "Close" in result.columns

    def test_returns_empty_for_missing_symbol(self, tmp_path):
        from research.shadow_long_resolver import make_file_bars_loader

        loader = make_file_bars_loader(tmp_path)
        start = datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc)
        result = loader("MISSING", start, lookforward_days=1)

        assert result.empty
