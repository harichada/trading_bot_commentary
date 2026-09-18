"""Tests for day_trade_short_resolver.

v-day-trade-short-resolver-tests-2026-09-17. Unit tests for:
  - R-multiple math (SHORT: positive R = profit)
  - Dedupe logic (same symbol within 15m)
  - Hands-off exclusions (MU/HQGE/SPCX)
  - Barrier resolution (stop/target/timeout/flatten)
  - Entry pattern breakdown
"""
from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import pytest


class TestRMath:
    """R-multiple calculation for day-trade SHORT trades."""

    def test_r_positive_when_price_drops(self):
        """SHORT profits when exit < entry → positive R."""
        from research.day_trade_short_resolver import ResolvedTrade

        entry = 100.0
        exit = 95.0
        stop_dist = 5.0
        expected_r = 1.0

        trade = ResolvedTrade(
            symbol="TEST",
            entry_time=datetime.now(timezone.utc),
            entry_price=entry,
            exit_time=datetime.now(timezone.utc),
            exit_price=exit,
            exit_reason="target",
            stop=entry + stop_dist,
            target=entry - (2 * stop_dist),
            stop_dist=stop_dist,
            hold_bars=10,
            r_multiple=expected_r,
            return_pct=5.0,
            regime="risk_off",
            entry_pattern="breakdown",
            session_id="2024-01-15",
        )

        assert trade.r_multiple == expected_r

    def test_r_negative_when_price_rises(self):
        """SHORT loses when exit > entry → negative R."""
        entry = 100.0
        exit = 105.0
        stop_dist = 5.0
        expected_r = -1.0

        from research.day_trade_short_resolver import ResolvedTrade

        trade = ResolvedTrade(
            symbol="TEST",
            entry_time=datetime.now(timezone.utc),
            entry_price=entry,
            exit_time=datetime.now(timezone.utc),
            exit_price=exit,
            exit_reason="stop",
            stop=entry + stop_dist,
            target=entry - (2 * stop_dist),
            stop_dist=stop_dist,
            hold_bars=5,
            r_multiple=expected_r,
            return_pct=-5.0,
            regime="risk_off",
            entry_pattern="breakdown",
            session_id="2024-01-15",
        )

        assert trade.r_multiple == expected_r

    def test_r_calculation_in_resolve_entry(self):
        """Test R calculation inside resolve_entry function."""
        from research.day_trade_short_resolver import ShadowEntry, resolve_entry

        entry = ShadowEntry(
            timestamp=datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc),
            symbol="TEST",
            signal_type="SHORT",
            strategy="day_trade_momentum_short",
            entry_pattern="breakdown",
            signal_close=100.0,
            rsi=45.0,
            rs_vs_spy=-2.0,
            volume_ratio=2.0,
            adx=25.0,
            low_20=102.0,
            high_20=110.0,
            sma_20=105.0,
            sma_50=108.0,
            macd=-0.5,
            macd_signal=-0.3,
            atr=2.0,
            hypothetical_stop=105.0,
            hypothetical_target=90.0,
            rr_ratio=2.0,
            stop_dist=5.0,
            market_context_regime="risk_off",
            direction_score=-4.0,
            direction_phase="middle",
            session_id="2024-01-15",
            raw={},
        )

        def mock_bars_loader(symbol, start, lookforward_days):
            dates = pd.date_range(start=start + timedelta(minutes=5), periods=10, freq="5min")
            return pd.DataFrame({
                "Open": [100] * 10,
                "High": [101] * 10,
                "Low": [89] * 10,
                "Close": [90] * 10,
            }, index=dates)

        trade = resolve_entry(entry, mock_bars_loader)

        assert trade.exit_reason == "target"
        assert trade.exit_price == 90.0
        assert trade.r_multiple > 0
        assert trade.entry_pattern == "breakdown"


class TestDedupe:
    """Deduplication: same symbol within 15 minutes → keep first."""

    def test_dedupe_keeps_first_within_window(self):
        from research.day_trade_short_resolver import ShadowEntry, filter_and_dedupe

        base = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)
        entries = [
            ShadowEntry(
                timestamp=base,
                symbol="AAPL",
                signal_type="SHORT",
                strategy="day_trade_momentum_short",
                entry_pattern="breakdown",
                signal_close=100.0,
                rsi=45, rs_vs_spy=-2.0, volume_ratio=2.0, adx=25,
                low_20=102, high_20=110, sma_20=105, sma_50=108,
                macd=-0.5, macd_signal=-0.3, atr=2.0,
                hypothetical_stop=105, hypothetical_target=90,
                rr_ratio=2.0, stop_dist=5.0,
                market_context_regime="risk_off",
                direction_score=-4, direction_phase="middle",
                session_id="2024-01-15",
            ),
            ShadowEntry(
                timestamp=base + timedelta(minutes=10),
                symbol="AAPL",
                signal_type="SHORT",
                strategy="day_trade_momentum_short",
                entry_pattern="continuation_down",
                signal_close=101.0,
                rsi=46, rs_vs_spy=-2.1, volume_ratio=2.1, adx=26,
                low_20=103, high_20=111, sma_20=106, sma_50=109,
                macd=-0.6, macd_signal=-0.4, atr=2.1,
                hypothetical_stop=106, hypothetical_target=91,
                rr_ratio=2.0, stop_dist=5.0,
                market_context_regime="risk_off",
                direction_score=-4.5, direction_phase="middle",
                session_id="2024-01-15",
            ),
        ]

        result = filter_and_dedupe(entries)

        assert len(result) == 1
        assert result[0].signal_close == 100.0

    def test_dedupe_keeps_both_outside_window(self):
        from research.day_trade_short_resolver import ShadowEntry, filter_and_dedupe

        base = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)
        entries = [
            ShadowEntry(
                timestamp=base,
                symbol="AAPL",
                signal_type="SHORT",
                strategy="day_trade_momentum_short",
                entry_pattern="breakdown",
                signal_close=100.0,
                rsi=45, rs_vs_spy=-2.0, volume_ratio=2.0, adx=25,
                low_20=102, high_20=110, sma_20=105, sma_50=108,
                macd=-0.5, macd_signal=-0.3, atr=2.0,
                hypothetical_stop=105, hypothetical_target=90,
                rr_ratio=2.0, stop_dist=5.0,
                market_context_regime="risk_off",
                direction_score=-4, direction_phase="middle",
                session_id="2024-01-15",
            ),
            ShadowEntry(
                timestamp=base + timedelta(minutes=20),
                symbol="AAPL",
                signal_type="SHORT",
                strategy="day_trade_momentum_short",
                entry_pattern="continuation_down",
                signal_close=101.0,
                rsi=46, rs_vs_spy=-2.1, volume_ratio=2.1, adx=26,
                low_20=103, high_20=111, sma_20=106, sma_50=109,
                macd=-0.6, macd_signal=-0.4, atr=2.1,
                hypothetical_stop=106, hypothetical_target=91,
                rr_ratio=2.0, stop_dist=5.0,
                market_context_regime="risk_off",
                direction_score=-4.5, direction_phase="middle",
                session_id="2024-01-15",
            ),
        ]

        result = filter_and_dedupe(entries)

        assert len(result) == 2


class TestHandsOffExclusion:
    """Hands-off forever symbols: MU, HQGE, SPCX."""

    def test_mu_excluded(self):
        from research.day_trade_short_resolver import ShadowEntry, filter_and_dedupe

        entries = [
            ShadowEntry(
                timestamp=datetime.now(timezone.utc),
                symbol="MU",
                signal_type="SHORT",
                strategy="day_trade_momentum_short",
                entry_pattern="breakdown",
                signal_close=100.0,
                rsi=45, rs_vs_spy=-2.0, volume_ratio=2.0, adx=25,
                low_20=102, high_20=110, sma_20=105, sma_50=108,
                macd=-0.5, macd_signal=-0.3, atr=2.0,
                hypothetical_stop=105, hypothetical_target=90,
                rr_ratio=2.0, stop_dist=5.0,
                market_context_regime="risk_off",
                direction_score=-4, direction_phase="middle",
                session_id="2024-01-15",
            ),
        ]

        result = filter_and_dedupe(entries)
        assert len(result) == 0

    def test_spcx_excluded(self):
        from research.day_trade_short_resolver import ShadowEntry, filter_and_dedupe

        entries = [
            ShadowEntry(
                timestamp=datetime.now(timezone.utc),
                symbol="SPCX",
                signal_type="SHORT",
                strategy="day_trade_momentum_short",
                entry_pattern="breakdown",
                signal_close=50.0,
                rsi=45, rs_vs_spy=-2.0, volume_ratio=2.0, adx=25,
                low_20=52, high_20=58, sma_20=55, sma_50=57,
                macd=-0.3, macd_signal=-0.2, atr=1.0,
                hypothetical_stop=52.5, hypothetical_target=45,
                rr_ratio=2.0, stop_dist=2.5,
                market_context_regime="risk_off",
                direction_score=-4, direction_phase="middle",
                session_id="2024-01-15",
            ),
        ]

        result = filter_and_dedupe(entries)
        assert len(result) == 0

    def test_hqge_excluded(self):
        from research.day_trade_short_resolver import ShadowEntry, filter_and_dedupe

        entries = [
            ShadowEntry(
                timestamp=datetime.now(timezone.utc),
                symbol="HQGE",
                signal_type="SHORT",
                strategy="day_trade_momentum_short",
                entry_pattern="breakdown",
                signal_close=25.0,
                rsi=45, rs_vs_spy=-2.0, volume_ratio=2.0, adx=25,
                low_20=26, high_20=30, sma_20=28, sma_50=29,
                macd=-0.2, macd_signal=-0.1, atr=0.5,
                hypothetical_stop=26.25, hypothetical_target=22.5,
                rr_ratio=2.0, stop_dist=1.25,
                market_context_regime="risk_off",
                direction_score=-4, direction_phase="middle",
                session_id="2024-01-15",
            ),
        ]

        result = filter_and_dedupe(entries)
        assert len(result) == 0

    def test_all_hands_off_excluded(self):
        from research.day_trade_short_resolver import ShadowEntry, filter_and_dedupe, HANDS_OFF_SYMBOLS

        base = datetime.now(timezone.utc)
        entries = [
            ShadowEntry(
                timestamp=base + timedelta(minutes=i),
                symbol=sym,
                signal_type="SHORT",
                strategy="day_trade_momentum_short",
                entry_pattern="breakdown",
                signal_close=100.0,
                rsi=45, rs_vs_spy=-2.0, volume_ratio=2.0, adx=25,
                low_20=102, high_20=110, sma_20=105, sma_50=108,
                macd=-0.5, macd_signal=-0.3, atr=2.0,
                hypothetical_stop=105, hypothetical_target=90,
                rr_ratio=2.0, stop_dist=5.0,
                market_context_regime="risk_off",
                direction_score=-4, direction_phase="middle",
                session_id="2024-01-15",
            )
            for i, sym in enumerate(HANDS_OFF_SYMBOLS)
        ]
        entries.append(ShadowEntry(
            timestamp=base + timedelta(minutes=10),
            symbol="AAPL",
            signal_type="SHORT",
            strategy="day_trade_momentum_short",
            entry_pattern="breakdown",
            signal_close=150.0,
            rsi=45, rs_vs_spy=-2.0, volume_ratio=2.0, adx=25,
            low_20=152, high_20=160, sma_20=155, sma_50=158,
            macd=-0.5, macd_signal=-0.3, atr=3.0,
            hypothetical_stop=157.5, hypothetical_target=135,
            rr_ratio=2.0, stop_dist=7.5,
            market_context_regime="risk_off",
            direction_score=-4, direction_phase="middle",
            session_id="2024-01-15",
        ))

        result = filter_and_dedupe(entries)

        assert len(result) == 1
        assert result[0].symbol == "AAPL"


class TestFixtureNDJSON:
    """Test loading and parsing NDJSON fixture."""

    def test_load_tiny_ndjson(self):
        from research.day_trade_short_resolver import load_shadow_log

        with tempfile.NamedTemporaryFile(mode='w', suffix='.ndjson', delete=False) as f:
            lines = [
                {
                    "timestamp": "2024-01-15T10:30:00+00:00",
                    "symbol": "AAPL",
                    "signal_type": "SHORT",
                    "strategy": "day_trade_momentum_short",
                    "entry_pattern": "breakdown",
                    "signal_close": 150.0,
                    "rsi": 45.0,
                    "rs_vs_spy": -2.0,
                    "volume_ratio": 2.0,
                    "adx": 25.0,
                    "low_20": 152.0,
                    "high_20": 160.0,
                    "sma_20": 155.0,
                    "sma_50": 158.0,
                    "macd": -0.5,
                    "macd_signal": -0.3,
                    "atr": 3.0,
                    "hypothetical_stop": 157.5,
                    "hypothetical_target": 135.0,
                    "rr_ratio": 2.0,
                    "stop_dist": 7.5,
                    "market_context_regime": "risk_off",
                    "direction_score": -4.0,
                    "direction_phase": "middle",
                    "session_id": "2024-01-15",
                },
                {
                    "timestamp": "2024-01-15T11:00:00+00:00",
                    "symbol": "NVDA",
                    "signal_type": "SHORT",
                    "strategy": "day_trade_momentum_short",
                    "entry_pattern": "continuation_down",
                    "signal_close": 500.0,
                    "rsi": 42.0,
                    "rs_vs_spy": -3.0,
                    "volume_ratio": 2.5,
                    "adx": 30.0,
                    "low_20": 505.0,
                    "high_20": 520.0,
                    "sma_20": 510.0,
                    "sma_50": 515.0,
                    "macd": -2.0,
                    "macd_signal": -1.5,
                    "atr": 10.0,
                    "hypothetical_stop": 525.0,
                    "hypothetical_target": 450.0,
                    "rr_ratio": 2.0,
                    "stop_dist": 25.0,
                    "market_context_regime": "mixed",
                    "direction_score": -5.0,
                    "direction_phase": "middle",
                    "session_id": "2024-01-15",
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
            assert entries[0].entry_pattern == "breakdown"
            assert entries[1].symbol == "NVDA"
            assert entries[1].entry_pattern == "continuation_down"
        finally:
            path.unlink()

    def test_load_handles_malformed_lines(self):
        from research.day_trade_short_resolver import load_shadow_log

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
        from research.day_trade_short_resolver import ShadowEntry, resolve_entry

        entry = ShadowEntry(
            timestamp=datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc),
            symbol="TEST",
            signal_type="SHORT",
            strategy="day_trade_momentum_short",
            entry_pattern="breakdown",
            signal_close=100.0,
            rsi=45, rs_vs_spy=-2.0, volume_ratio=2.0, adx=25,
            low_20=102, high_20=110, sma_20=105, sma_50=108,
            macd=-0.5, macd_signal=-0.3, atr=2.0,
            hypothetical_stop=105.0,
            hypothetical_target=90.0,
            rr_ratio=2.0, stop_dist=5.0,
            market_context_regime="risk_off",
            direction_score=-4, direction_phase="middle",
            session_id="2024-01-15",
        )

        def mock_bars_loader(symbol, start, lookforward_days):
            dates = pd.date_range(start=start + timedelta(minutes=5), periods=10, freq="5min")
            return pd.DataFrame({
                "Open": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
                "High": [101, 102, 103, 104, 106, 108, 109, 110, 111, 112],
                "Low": [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
                "Close": [100, 101, 102, 103, 105, 107, 108, 109, 110, 111],
            }, index=dates)

        trade = resolve_entry(entry, mock_bars_loader)

        assert trade.exit_reason == "stop"
        assert trade.exit_price == 105.0
        assert trade.r_multiple < 0

    def test_target_hit_first(self):
        from research.day_trade_short_resolver import ShadowEntry, resolve_entry

        entry = ShadowEntry(
            timestamp=datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc),
            symbol="TEST",
            signal_type="SHORT",
            strategy="day_trade_momentum_short",
            entry_pattern="breakdown",
            signal_close=100.0,
            rsi=45, rs_vs_spy=-2.0, volume_ratio=2.0, adx=25,
            low_20=102, high_20=110, sma_20=105, sma_50=108,
            macd=-0.5, macd_signal=-0.3, atr=2.0,
            hypothetical_stop=105.0,
            hypothetical_target=90.0,
            rr_ratio=2.0, stop_dist=5.0,
            market_context_regime="risk_off",
            direction_score=-4, direction_phase="middle",
            session_id="2024-01-15",
        )

        def mock_bars_loader(symbol, start, lookforward_days):
            dates = pd.date_range(start=start + timedelta(minutes=5), periods=10, freq="5min")
            return pd.DataFrame({
                "Open": [100, 99, 98, 97, 96, 95, 94, 93, 92, 91],
                "High": [101, 100, 99, 98, 97, 96, 95, 94, 93, 92],
                "Low": [99, 98, 97, 96, 95, 89, 88, 87, 86, 85],
                "Close": [100, 99, 98, 97, 96, 90, 89, 88, 87, 86],
            }, index=dates)

        trade = resolve_entry(entry, mock_bars_loader)

        assert trade.exit_reason == "target"
        assert trade.exit_price == 90.0
        assert trade.r_multiple > 0

    def test_timeout_when_no_barrier_hit(self):
        from research.day_trade_short_resolver import ShadowEntry, resolve_entry, MAX_HOLD_BARS

        entry = ShadowEntry(
            timestamp=datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc),
            symbol="TEST",
            signal_type="SHORT",
            strategy="day_trade_momentum_short",
            entry_pattern="breakdown",
            signal_close=100.0,
            rsi=45, rs_vs_spy=-2.0, volume_ratio=2.0, adx=25,
            low_20=102, high_20=110, sma_20=105, sma_50=108,
            macd=-0.5, macd_signal=-0.3, atr=2.0,
            hypothetical_stop=110.0,
            hypothetical_target=85.0,
            rr_ratio=2.0, stop_dist=10.0,
            market_context_regime="risk_off",
            direction_score=-4, direction_phase="middle",
            session_id="2024-01-15",
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
        from research.day_trade_short_resolver import ShadowEntry, resolve_entry

        entry = ShadowEntry(
            timestamp=datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc),
            symbol="TEST",
            signal_type="SHORT",
            strategy="day_trade_momentum_short",
            entry_pattern="breakdown",
            signal_close=100.0,
            rsi=45, rs_vs_spy=-2.0, volume_ratio=2.0, adx=25,
            low_20=102, high_20=110, sma_20=105, sma_50=108,
            macd=-0.5, macd_signal=-0.3, atr=2.0,
            hypothetical_stop=105.0,
            hypothetical_target=90.0,
            rr_ratio=2.0, stop_dist=5.0,
            market_context_regime="risk_off",
            direction_score=-4, direction_phase="middle",
            session_id="2024-01-15",
        )

        def mock_bars_loader(symbol, start, lookforward_days):
            return pd.DataFrame()

        trade = resolve_entry(entry, mock_bars_loader)

        assert trade.exit_reason == "no_bars"
        assert trade.hold_bars == 0


class TestMetrics:
    """Test Stage A metrics computation."""

    def test_compute_metrics_basic(self):
        from research.day_trade_short_resolver import ResolvedTrade, compute_metrics

        base = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)
        trades = [
            ResolvedTrade(
                symbol="AAPL", entry_time=base, entry_price=100,
                exit_time=base + timedelta(hours=1), exit_price=95,
                exit_reason="target", stop=105, target=90, stop_dist=5,
                hold_bars=12, r_multiple=1.0, return_pct=5.0,
                regime="risk_off", entry_pattern="breakdown",
                session_id="2024-01-15",
            ),
            ResolvedTrade(
                symbol="MSFT", entry_time=base + timedelta(hours=2), entry_price=200,
                exit_time=base + timedelta(hours=3), exit_price=210,
                exit_reason="stop", stop=210, target=180, stop_dist=10,
                hold_bars=6, r_multiple=-1.0, return_pct=-5.0,
                regime="mixed", entry_pattern="continuation_down",
                session_id="2024-01-15",
            ),
        ]

        summary = compute_metrics(trades)

        assert summary.n_trades == 2
        assert summary.n_wins == 1
        assert summary.n_losses == 1
        assert summary.win_rate == 0.5
        assert summary.total_r == 0.0

    def test_exclude_risk_off_for_promotion_book(self):
        from research.day_trade_short_resolver import ResolvedTrade, compute_metrics

        base = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)
        trades = [
            ResolvedTrade(
                symbol="AAPL", entry_time=base, entry_price=100,
                exit_time=base + timedelta(hours=1), exit_price=95,
                exit_reason="target", stop=105, target=90, stop_dist=5,
                hold_bars=12, r_multiple=1.0, return_pct=5.0,
                regime="mixed", entry_pattern="breakdown",
                session_id="2024-01-15",
            ),
            ResolvedTrade(
                symbol="MSFT", entry_time=base + timedelta(hours=2), entry_price=200,
                exit_time=base + timedelta(hours=3), exit_price=210,
                exit_reason="stop", stop=210, target=180, stop_dist=10,
                hold_bars=6, r_multiple=-2.0, return_pct=-5.0,
                regime="risk_off", entry_pattern="continuation_down",
                session_id="2024-01-15",
            ),
        ]

        summary = compute_metrics(trades, exclude_regimes={"risk_off"})

        assert summary.n_trades == 1
        assert summary.n_wins == 1
        assert summary.n_losses == 0
        assert "risk_off" in summary.by_regime
        assert "mixed" in summary.by_regime

    def test_by_pattern_breakdown(self):
        from research.day_trade_short_resolver import ResolvedTrade, compute_metrics

        base = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)
        trades = [
            ResolvedTrade(
                symbol="AAPL", entry_time=base, entry_price=100,
                exit_time=base + timedelta(hours=1), exit_price=95,
                exit_reason="target", stop=105, target=90, stop_dist=5,
                hold_bars=12, r_multiple=1.0, return_pct=5.0,
                regime="risk_off", entry_pattern="breakdown",
                session_id="2024-01-15",
            ),
            ResolvedTrade(
                symbol="MSFT", entry_time=base + timedelta(hours=2), entry_price=200,
                exit_time=base + timedelta(hours=3), exit_price=190,
                exit_reason="target", stop=210, target=180, stop_dist=10,
                hold_bars=6, r_multiple=1.0, return_pct=5.0,
                regime="risk_off", entry_pattern="continuation_down",
                session_id="2024-01-15",
            ),
            ResolvedTrade(
                symbol="NVDA", entry_time=base + timedelta(hours=4), entry_price=500,
                exit_time=base + timedelta(hours=5), exit_price=520,
                exit_reason="stop", stop=525, target=450, stop_dist=25,
                hold_bars=8, r_multiple=-0.8, return_pct=-4.0,
                regime="risk_off", entry_pattern="breakdown",
                session_id="2024-01-15",
            ),
        ]

        summary = compute_metrics(trades)

        assert "breakdown" in summary.by_pattern
        assert "continuation_down" in summary.by_pattern
        assert summary.by_pattern["breakdown"]["n"] == 2
        assert summary.by_pattern["continuation_down"]["n"] == 1


class TestEntryPattern:
    """Test entry pattern preservation through resolution."""

    def test_entry_pattern_preserved(self):
        from research.day_trade_short_resolver import ShadowEntry, resolve_entry

        entry = ShadowEntry(
            timestamp=datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc),
            symbol="TEST",
            signal_type="SHORT",
            strategy="day_trade_momentum_short",
            entry_pattern="continuation_down",
            signal_close=100.0,
            rsi=42, rs_vs_spy=-2.5, volume_ratio=2.0, adx=25,
            low_20=102, high_20=110, sma_20=105, sma_50=108,
            macd=-0.5, macd_signal=-0.3, atr=2.0,
            hypothetical_stop=105.0,
            hypothetical_target=90.0,
            rr_ratio=2.0, stop_dist=5.0,
            market_context_regime="risk_off",
            direction_score=-4, direction_phase="middle",
            session_id="2024-01-15",
        )

        def mock_bars_loader(symbol, start, lookforward_days):
            dates = pd.date_range(start=start + timedelta(minutes=5), periods=2, freq="5min")
            return pd.DataFrame({
                "Open": [100, 100],
                "High": [101, 101],
                "Low": [89, 89],
                "Close": [90, 90],
            }, index=dates)

        trade = resolve_entry(entry, mock_bars_loader)

        assert trade.entry_pattern == "continuation_down"
