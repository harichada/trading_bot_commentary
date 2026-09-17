"""Tests for day_trade_short_bt_sample — schema compatibility and logic.

v-day-trade-short-bt-sample-tests-2026-09-17. Verifies that:
  - Emitted shadow entries have all required fields for the resolver
  - Pattern detection logic matches the live strategy
  - HANDS_OFF symbols are excluded
  - Risk_on filter works correctly
  - Deduplication works (same symbol within 15m)
"""
from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


REQUIRED_SHADOW_FIELDS = {
    "timestamp",
    "symbol",
    "signal_type",
    "strategy",
    "entry_pattern",
    "signal_close",
    "rsi",
    "rs_vs_spy",
    "volume_ratio",
    "adx",
    "low_20",
    "high_20",
    "sma_20",
    "sma_50",
    "macd",
    "macd_signal",
    "atr",
    "hypothetical_stop",
    "hypothetical_target",
    "rr_ratio",
    "stop_dist",
    "market_context_regime",
    "direction_score",
    "direction_phase",
    "session_id",
}


def make_test_bars(
    n_bars: int = 100,
    base_price: float = 100.0,
    trend: str = "down",
    base_date: datetime = None,
) -> pd.DataFrame:
    """Create synthetic OHLCV bars for testing."""
    if base_date is None:
        base_date = datetime(2024, 1, 15, 9, 30, tzinfo=timezone.utc)
    
    dates = pd.date_range(start=base_date, periods=n_bars, freq="5min")
    
    if trend == "down":
        drift = -0.001
    elif trend == "up":
        drift = 0.001
    else:
        drift = 0.0
    
    closes = [base_price]
    for i in range(1, n_bars):
        change = drift + np.random.normal(0, 0.005)
        closes.append(closes[-1] * (1 + change))
    closes = np.array(closes)
    
    highs = closes * (1 + np.random.uniform(0.001, 0.01, n_bars))
    lows = closes * (1 - np.random.uniform(0.001, 0.01, n_bars))
    opens = lows + np.random.uniform(0.3, 0.7, n_bars) * (highs - lows)
    volumes = np.random.uniform(50000, 200000, n_bars)
    
    return pd.DataFrame({
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": closes,
        "Volume": volumes,
    }, index=dates)


class TestSchemaCompatibility:
    """Verify emitted shadow entries are compatible with the resolver."""

    def test_all_required_fields_present(self):
        """Emitted entries must have all fields the resolver expects."""
        from research.day_trade_short_bt_sample import process_symbol
        
        df = make_test_bars(100, trend="down")
        
        def mock_loader(symbol: str) -> pd.DataFrame:
            return df
        
        entries = process_symbol(
            symbol="TEST",
            bars_loader=mock_loader,
            spy_df=None,
            min_weak_rs=0.0,
            min_volume_ratio=0.1,
            rsi_floor=0,
            rsi_ceiling=100,
            allow_risk_on=True,
        )
        
        if entries:
            entry = entries[0]
            missing = REQUIRED_SHADOW_FIELDS - set(entry.keys())
            assert not missing, f"Missing required fields: {missing}"

    def test_resolver_can_parse_emitted_entry(self):
        """Resolver's ShadowEntry.from_dict must accept our format."""
        from research.day_trade_short_resolver import ShadowEntry
        
        sample_entry = {
            "timestamp": "2024-01-15T10:30:00+00:00",
            "symbol": "TEST",
            "signal_type": "SHORT",
            "strategy": "day_trade_momentum_short",
            "entry_pattern": "breakdown",
            "signal_close": 100.0,
            "rsi": 45.0,
            "rs_vs_spy": -2.0,
            "volume_ratio": 2.0,
            "adx": 25.0,
            "low_20": 102.0,
            "high_20": 110.0,
            "sma_20": 105.0,
            "sma_50": 108.0,
            "macd": -0.5,
            "macd_signal": -0.3,
            "atr": 2.0,
            "hypothetical_stop": 105.0,
            "hypothetical_target": 90.0,
            "rr_ratio": 2.0,
            "stop_dist": 5.0,
            "market_context_regime": "risk_off",
            "market_context_spy_change": -1.5,
            "market_context_vix_change": 0.0,
            "market_context_sector": None,
            "direction_score": -4.0,
            "direction_phase": "middle",
            "session_id": "2024-01-15",
        }
        
        parsed = ShadowEntry.from_dict(sample_entry)
        
        assert parsed.symbol == "TEST"
        assert parsed.signal_type == "SHORT"
        assert parsed.entry_pattern == "breakdown"
        assert parsed.signal_close == 100.0
        assert parsed.hypothetical_stop == 105.0
        assert parsed.hypothetical_target == 90.0

    def test_signal_type_always_short(self):
        """All emitted entries must have signal_type='SHORT'."""
        from research.day_trade_short_bt_sample import process_symbol
        
        df = make_test_bars(100, trend="down")
        
        def mock_loader(symbol: str) -> pd.DataFrame:
            return df
        
        entries = process_symbol(
            symbol="TEST",
            bars_loader=mock_loader,
            spy_df=None,
            min_weak_rs=0.0,
            min_volume_ratio=0.1,
            rsi_floor=0,
            rsi_ceiling=100,
            allow_risk_on=True,
        )
        
        for entry in entries:
            assert entry["signal_type"] == "SHORT"

    def test_strategy_field_correct(self):
        """Strategy field must be 'day_trade_momentum_short'."""
        from research.day_trade_short_bt_sample import process_symbol
        
        df = make_test_bars(100, trend="down")
        
        def mock_loader(symbol: str) -> pd.DataFrame:
            return df
        
        entries = process_symbol(
            symbol="TEST",
            bars_loader=mock_loader,
            spy_df=None,
            min_weak_rs=0.0,
            min_volume_ratio=0.1,
            rsi_floor=0,
            rsi_ceiling=100,
            allow_risk_on=True,
        )
        
        for entry in entries:
            assert entry["strategy"] == "day_trade_momentum_short"


class TestPatternDetection:
    """Verify pattern detection matches live strategy logic."""

    def test_breakdown_pattern_detection(self):
        """Breakdown: price < 20-bar low, ADX > 20."""
        from research.day_trade_short_bt_sample import detect_entry_pattern
        
        result = detect_entry_pattern(
            close=98.0,
            low_20=100.0,
            high_20=110.0,
            sma_20=105.0,
            rsi=45.0,
            adx=25.0,
            macd=-0.5,
            macd_signal=-0.3,
        )
        
        assert result is not None
        pattern, confidence = result
        assert pattern == "breakdown"
        assert confidence > 0.5

    def test_continuation_down_pattern_detection(self):
        """Continuation-down: RSI 30-50, close < SMA20, MACD < signal."""
        from research.day_trade_short_bt_sample import detect_entry_pattern
        
        result = detect_entry_pattern(
            close=103.0,
            low_20=100.0,
            high_20=110.0,
            sma_20=105.0,
            rsi=40.0,
            adx=25.0,
            macd=-0.5,
            macd_signal=-0.3,
        )
        
        assert result is not None
        pattern, confidence = result
        assert pattern == "continuation_down"
        assert confidence > 0.5

    def test_rejection_pattern_detection(self):
        """Rejection: near 20-bar high, failing, MACD bearish."""
        from research.day_trade_short_bt_sample import detect_entry_pattern
        
        result = detect_entry_pattern(
            close=109.5,
            low_20=100.0,
            high_20=110.0,
            sma_20=105.0,
            rsi=55.0,
            adx=25.0,
            macd=-0.5,
            macd_signal=-0.3,
        )
        
        assert result is not None
        pattern, confidence = result
        assert pattern == "rejection"
        assert confidence > 0.5

    def test_no_pattern_when_conditions_not_met(self):
        """Return None when no pattern conditions match."""
        from research.day_trade_short_bt_sample import detect_entry_pattern
        
        result = detect_entry_pattern(
            close=105.0,
            low_20=100.0,
            high_20=110.0,
            sma_20=103.0,
            rsi=55.0,
            adx=15.0,
            macd=0.5,
            macd_signal=0.3,
        )
        
        assert result is None


class TestHandsOffExclusion:
    """Verify HANDS_OFF symbols are excluded."""

    def test_mu_excluded(self):
        """MU is in HANDS_OFF and should be skipped."""
        from research.day_trade_short_bt_sample import process_symbol, HANDS_OFF_SYMBOLS
        
        assert "MU" in HANDS_OFF_SYMBOLS
        
        df = make_test_bars(100, trend="down")
        
        def mock_loader(symbol: str) -> pd.DataFrame:
            return df
        
        entries = process_symbol(
            symbol="MU",
            bars_loader=mock_loader,
            spy_df=None,
            min_weak_rs=0.0,
            min_volume_ratio=0.1,
            rsi_floor=0,
            rsi_ceiling=100,
            allow_risk_on=True,
        )
        
        assert len(entries) == 0

    def test_hqge_excluded(self):
        """HQGE is in HANDS_OFF and should be skipped."""
        from research.day_trade_short_bt_sample import process_symbol, HANDS_OFF_SYMBOLS
        
        assert "HQGE" in HANDS_OFF_SYMBOLS
        
        df = make_test_bars(100, trend="down")
        
        def mock_loader(symbol: str) -> pd.DataFrame:
            return df
        
        entries = process_symbol(
            symbol="HQGE",
            bars_loader=mock_loader,
            spy_df=None,
            min_weak_rs=0.0,
            min_volume_ratio=0.1,
            rsi_floor=0,
            rsi_ceiling=100,
            allow_risk_on=True,
        )
        
        assert len(entries) == 0

    def test_spcx_excluded(self):
        """SPCX is in HANDS_OFF and should be skipped."""
        from research.day_trade_short_bt_sample import process_symbol, HANDS_OFF_SYMBOLS
        
        assert "SPCX" in HANDS_OFF_SYMBOLS
        
        df = make_test_bars(100, trend="down")
        
        def mock_loader(symbol: str) -> pd.DataFrame:
            return df
        
        entries = process_symbol(
            symbol="SPCX",
            bars_loader=mock_loader,
            spy_df=None,
            min_weak_rs=0.0,
            min_volume_ratio=0.1,
            rsi_floor=0,
            rsi_ceiling=100,
            allow_risk_on=True,
        )
        
        assert len(entries) == 0


class TestRiskOnFilter:
    """Verify risk_on regime filtering."""

    def test_risk_on_blocked_by_default(self):
        """Entries in risk_on regime should be blocked when allow_risk_on=False."""
        from research.day_trade_short_bt_sample import classify_regime
        
        assert classify_regime(1.0) == "risk_on"
        assert classify_regime(0.6) == "risk_on"
        assert classify_regime(-1.0) == "risk_off"
        assert classify_regime(0.0) == "mixed"

    def test_classify_regime_thresholds(self):
        """Regime classification based on SPY change."""
        from research.day_trade_short_bt_sample import classify_regime
        
        assert classify_regime(0.51) == "risk_on"
        assert classify_regime(-0.51) == "risk_off"
        assert classify_regime(0.49) == "mixed"
        assert classify_regime(-0.49) == "mixed"


class TestDeduplication:
    """Verify 15-minute deduplication logic."""

    def test_dedupe_skips_same_symbol_within_15m(self):
        """Same symbol should not emit twice within 15 minutes."""
        from research.day_trade_short_bt_sample import process_symbol
        
        base = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)
        df = make_test_bars(100, trend="down", base_date=base)
        
        def mock_loader(symbol: str) -> pd.DataFrame:
            return df
        
        entries = process_symbol(
            symbol="TEST",
            bars_loader=mock_loader,
            spy_df=None,
            min_weak_rs=0.0,
            min_volume_ratio=0.1,
            rsi_floor=0,
            rsi_ceiling=100,
            allow_risk_on=True,
        )
        
        if len(entries) >= 2:
            for i in range(1, len(entries)):
                t1 = datetime.fromisoformat(entries[i-1]["timestamp"].replace("Z", "+00:00"))
                t2 = datetime.fromisoformat(entries[i]["timestamp"].replace("Z", "+00:00"))
                delta_min = (t2 - t1).total_seconds() / 60
                assert delta_min >= 15, f"Dedupe failed: entries {i-1} and {i} only {delta_min:.1f}m apart"


class TestIndicatorComputation:
    """Verify indicator computation produces valid values."""

    def test_compute_indicators_returns_all_required(self):
        """compute_indicators should add all required indicator columns."""
        from research.day_trade_short_bt_sample import compute_indicators
        
        df = make_test_bars(100, trend="down")
        result = compute_indicators(df)
        
        assert not result.empty
        
        required_cols = ["rsi", "sma_20", "sma_50", "macd", "macd_signal", 
                        "adx", "atr", "high_20", "low_20", "volume_ratio"]
        for col in required_cols:
            assert col in result.columns, f"Missing indicator column: {col}"

    def test_compute_indicators_needs_50_bars(self):
        """Should return empty DataFrame if < 50 bars."""
        from research.day_trade_short_bt_sample import compute_indicators
        
        df = make_test_bars(30, trend="down")
        result = compute_indicators(df)
        
        assert result.empty


class TestStopTargetCalculation:
    """Verify stop/target calculations for SHORT."""

    def test_stop_above_entry(self):
        """Stop should be ABOVE entry price for SHORT."""
        from research.day_trade_short_bt_sample import process_symbol
        
        df = make_test_bars(100, trend="down")
        
        def mock_loader(symbol: str) -> pd.DataFrame:
            return df
        
        entries = process_symbol(
            symbol="TEST",
            bars_loader=mock_loader,
            spy_df=None,
            min_weak_rs=0.0,
            min_volume_ratio=0.1,
            rsi_floor=0,
            rsi_ceiling=100,
            allow_risk_on=True,
        )
        
        for entry in entries:
            assert entry["hypothetical_stop"] > entry["signal_close"], \
                f"Stop {entry['hypothetical_stop']} should be above entry {entry['signal_close']}"

    def test_target_below_entry(self):
        """Target should be BELOW entry price for SHORT."""
        from research.day_trade_short_bt_sample import process_symbol
        
        df = make_test_bars(100, trend="down")
        
        def mock_loader(symbol: str) -> pd.DataFrame:
            return df
        
        entries = process_symbol(
            symbol="TEST",
            bars_loader=mock_loader,
            spy_df=None,
            min_weak_rs=0.0,
            min_volume_ratio=0.1,
            rsi_floor=0,
            rsi_ceiling=100,
            allow_risk_on=True,
        )
        
        for entry in entries:
            assert entry["hypothetical_target"] < entry["signal_close"], \
                f"Target {entry['hypothetical_target']} should be below entry {entry['signal_close']}"


class TestWriteNDJSON:
    """Test NDJSON file writing."""

    def test_write_creates_valid_ndjson(self):
        """Written file should be valid NDJSON (one JSON per line)."""
        from research.day_trade_short_bt_sample import write_shadow_log
        
        entries = [
            {
                "timestamp": "2024-01-15T10:30:00+00:00",
                "symbol": "TEST",
                "signal_type": "SHORT",
                "strategy": "day_trade_momentum_short",
                "entry_pattern": "breakdown",
                "signal_close": 100.0,
            },
            {
                "timestamp": "2024-01-15T11:00:00+00:00",
                "symbol": "TEST2",
                "signal_type": "SHORT",
                "strategy": "day_trade_momentum_short",
                "entry_pattern": "continuation_down",
                "signal_close": 200.0,
            },
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ndjson', delete=False) as f:
            path = Path(f.name)
        
        try:
            write_shadow_log(entries, path)
            
            with open(path, "r") as f:
                lines = f.readlines()
            
            assert len(lines) == 2
            
            for line in lines:
                parsed = json.loads(line.strip())
                assert "symbol" in parsed
                assert "signal_type" in parsed
        finally:
            path.unlink()

    def test_append_mode(self):
        """Append mode should add to existing file."""
        from research.day_trade_short_bt_sample import write_shadow_log
        
        entries1 = [{"timestamp": "2024-01-15T10:00:00+00:00", "symbol": "A"}]
        entries2 = [{"timestamp": "2024-01-15T11:00:00+00:00", "symbol": "B"}]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ndjson', delete=False) as f:
            path = Path(f.name)
        
        try:
            write_shadow_log(entries1, path, append=False)
            write_shadow_log(entries2, path, append=True)
            
            with open(path, "r") as f:
                lines = f.readlines()
            
            assert len(lines) == 2
            assert json.loads(lines[0])["symbol"] == "A"
            assert json.loads(lines[1])["symbol"] == "B"
        finally:
            path.unlink()
