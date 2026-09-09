"""Tests for v-feature-snapshot-2026-09-09: DecisionSnapshot schema and persistence.

Tests cover:
  - Schema construction and serialization
  - PriceVolumeFeatures from indicators
  - NewsAggregate from gate result
  - build_snapshot factory function
  - Skip path emits a snapshot (via strategy _log_decision)
  - Snapshot ID determinism
"""
import json
import pytest
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch

from core.decision_snapshot import (
    DecisionAction,
    DecisionSnapshot,
    PriceVolumeFeatures,
    NewsAggregate,
    RegimeContext,
    build_snapshot,
    is_snapshot_logging_enabled,
    is_snapshot_inference_enabled,
    FEATURE_SNAPSHOT_LOGGING_ENABLED,
    FEATURE_SNAPSHOT_INFERENCE_ENABLED,
)


class TestPriceVolumeFeatures:
    """Tests for PriceVolumeFeatures dataclass."""

    def test_from_indicators_basic(self):
        """from_indicators builds features from typical indicator dict."""
        indicators = {
            "rsi": 35,
            "volume_ratio": 1.5,
            "macd": 0.5,
            "macd_signal": 0.3,
            "macd_histogram": 0.2,
            "bb_position": 0.3,
            "bb_width": 0.04,
            "atr": 2.0,
        }
        pv = PriceVolumeFeatures.from_indicators(indicators, price=100.0)
        
        assert pv.price == 100.0
        assert pv.rsi == 0.35  # Normalized to 0-1
        assert pv.volume_ratio == 1.5
        assert pv.macd == 0.005  # Normalized by price
        assert pv.atr_ratio == 0.02  # ATR / price

    def test_from_indicators_empty(self):
        """from_indicators handles empty indicators gracefully."""
        pv = PriceVolumeFeatures.from_indicators({}, price=50.0)
        
        assert pv.price == 50.0
        assert pv.rsi == 0.5  # Default
        assert pv.volume_ratio == 1.0  # Default

    def test_empty_factory(self):
        """empty() creates zero-initialized features."""
        pv = PriceVolumeFeatures.empty(price=75.0)
        
        assert pv.price == 75.0
        assert pv.returns_1 == 0.0
        assert pv.rsi == 0.5

    def test_to_vector(self):
        """to_vector produces correct-length feature vector."""
        pv = PriceVolumeFeatures.empty()
        vec = pv.to_vector()
        
        assert len(vec) == 17
        assert all(isinstance(x, float) for x in vec)


class TestNewsAggregate:
    """Tests for NewsAggregate dataclass."""

    def test_from_gate_result_full_size(self):
        """from_gate_result handles FULL_SIZE gate action."""
        mock_gate = MagicMock()
        mock_gate.action.value = "full_size"
        mock_gate.size_multiplier = 1.0
        mock_gate.news_age_sec = 120.0
        mock_gate.source_tier_min = 1
        mock_gate.corroboration_n = 3
        
        aggregate = {
            "article_count": 5,
            "avg_sentiment": 0.45,
            "high_impact_count": 1,
        }
        
        news = NewsAggregate.from_gate_result(mock_gate, aggregate)
        
        assert news.article_count == 5
        assert news.avg_sentiment == 0.45
        assert news.gate_action == "full_size"
        assert news.gate_size_mult == 1.0
        assert news.corroboration_n == 3

    def test_empty_factory(self):
        """empty() creates default news aggregate."""
        news = NewsAggregate.empty()
        
        assert news.article_count == 0
        assert news.gate_action == "no_news"
        assert news.gate_size_mult == 0.0


class TestRegimeContext:
    """Tests for RegimeContext dataclass."""

    def test_from_context_with_allocator(self):
        """from_context captures allocator tape/ER."""
        mock_alloc = MagicMock()
        mock_alloc.tape = "choppy"
        mock_alloc.er = 0.35
        
        regime = RegimeContext.from_context(
            regime="oversold",
            spy_slope_pct=-0.02,
            vix=18.5,
            allocator_result=mock_alloc,
        )
        
        assert regime.regime == "oversold"
        assert regime.spy_slope_pct == -0.02
        assert regime.vix == 18.5
        assert regime.tape == "choppy"
        assert regime.er == 0.35

    def test_empty_factory(self):
        """empty() creates unknown regime."""
        regime = RegimeContext.empty()
        
        assert regime.regime == "unknown"
        assert regime.spy_slope_pct == 0.0
        assert regime.tape is None


class TestDecisionSnapshot:
    """Tests for DecisionSnapshot dataclass and serialization."""

    def test_compute_id_deterministic(self):
        """compute_id is deterministic for same inputs."""
        ts = datetime(2026, 9, 9, 12, 0, 0)
        id1 = DecisionSnapshot.compute_id("TSLA", ts, "oversold_v2", "skip")
        id2 = DecisionSnapshot.compute_id("TSLA", ts, "oversold_v2", "skip")
        
        assert id1 == id2
        assert len(id1) == 16  # SHA256 truncated to 16 chars

    def test_compute_id_varies_by_inputs(self):
        """compute_id differs for different inputs."""
        ts = datetime(2026, 9, 9, 12, 0, 0)
        id1 = DecisionSnapshot.compute_id("TSLA", ts, "oversold_v2", "skip")
        id2 = DecisionSnapshot.compute_id("NVDA", ts, "oversold_v2", "skip")
        id3 = DecisionSnapshot.compute_id("TSLA", ts, "breakout", "skip")
        
        assert id1 != id2
        assert id1 != id3

    def test_to_dict_serializable(self):
        """to_dict produces JSON-serializable output."""
        snapshot = build_snapshot(
            symbol="AAPL",
            strategy_id="test_strategy",
            action=DecisionAction.SKIP,
            reason="gate_a_not_extreme",
            mode="simulation",
        )
        
        d = snapshot.to_dict()
        
        # Should be JSON-serializable
        json_str = json.dumps(d)
        assert json_str
        
        # Round-trip
        parsed = json.loads(json_str)
        assert parsed["symbol"] == "AAPL"
        assert parsed["action"] == "skip"
        assert parsed["reason"] == "gate_a_not_extreme"

    def test_to_json(self):
        """to_json produces valid JSON string."""
        snapshot = build_snapshot(
            symbol="MSFT",
            strategy_id="momentum",
            action=DecisionAction.SIGNAL_BUY,
            reason="momentum_breakout",
            confidence=0.75,
        )
        
        json_str = snapshot.to_json()
        parsed = json.loads(json_str)
        
        assert parsed["symbol"] == "MSFT"
        assert parsed["confidence"] == 0.75

    def test_from_dict_roundtrip(self):
        """from_dict reconstructs snapshot from dict."""
        original = build_snapshot(
            symbol="GOOG",
            strategy_id="mean_rev",
            action=DecisionAction.VETO,
            reason="ml_veto_high_confidence",
            gate_name="ml_veto",
            confidence=0.9,
            would_entry_price=150.0,
            would_stop_loss=145.0,
            would_take_profit=165.0,
        )
        
        d = original.to_dict()
        reconstructed = DecisionSnapshot.from_dict(d)
        
        assert reconstructed.symbol == original.symbol
        assert reconstructed.action == original.action
        assert reconstructed.reason == original.reason
        assert reconstructed.gate_name == original.gate_name
        assert reconstructed.confidence == original.confidence
        assert reconstructed.would_entry_price == original.would_entry_price


class TestBuildSnapshot:
    """Tests for build_snapshot factory function."""

    def test_build_with_minimal_args(self):
        """build_snapshot works with minimal required args."""
        snapshot = build_snapshot(
            symbol="AMD",
            strategy_id="test",
            action=DecisionAction.SKIP,
            reason="no_setup",
        )
        
        assert snapshot.symbol == "AMD"
        assert snapshot.strategy_id == "test"
        assert snapshot.action == DecisionAction.SKIP
        assert snapshot.reason == "no_setup"
        assert snapshot.mode == "simulation"  # Default
        assert snapshot.snapshot_id  # Auto-generated

    def test_build_with_indicators(self):
        """build_snapshot extracts PriceVolumeFeatures from indicators."""
        indicators = {"rsi": 25, "volume_ratio": 2.5, "atr": 1.5}
        
        snapshot = build_snapshot(
            symbol="PLTR",
            strategy_id="oversold_v2",
            action=DecisionAction.SIGNAL_BUY,
            reason="all_gates_passed",
            indicators=indicators,
            price=20.0,
        )
        
        assert snapshot.price_vol.price == 20.0
        assert snapshot.price_vol.rsi == 0.25
        assert snapshot.price_vol.volume_ratio == 2.5
        assert snapshot.price_vol.atr_ratio == 0.075

    def test_build_with_sizing(self):
        """build_snapshot captures would-be sizing."""
        snapshot = build_snapshot(
            symbol="COIN",
            strategy_id="breakout",
            action=DecisionAction.VETO,
            reason="regime_gate",
            would_entry_price=85.0,
            would_stop_loss=80.0,
            would_take_profit=100.0,
            would_size_shares=50,
            would_size_mult=0.5,
        )
        
        assert snapshot.would_entry_price == 85.0
        assert snapshot.would_stop_loss == 80.0
        assert snapshot.would_take_profit == 100.0
        assert snapshot.would_size_shares == 50
        assert snapshot.would_size_mult == 0.5

    def test_build_with_extra(self):
        """build_snapshot captures extra strategy-specific data."""
        snapshot = build_snapshot(
            symbol="RIVN",
            strategy_id="news",
            action=DecisionAction.SKIP,
            reason="insufficient_news",
            extra={"fresh_count": 0, "source_tier": 3},
        )
        
        assert snapshot.extra["fresh_count"] == 0
        assert snapshot.extra["source_tier"] == 3


class TestFeatureFlags:
    """Tests for feature flags."""

    def test_logging_enabled_by_default(self):
        """Snapshot logging is enabled by default."""
        assert FEATURE_SNAPSHOT_LOGGING_ENABLED is True
        assert is_snapshot_logging_enabled() is True

    def test_inference_disabled_by_default(self):
        """Snapshot inference is disabled by default."""
        assert FEATURE_SNAPSHOT_INFERENCE_ENABLED is False
        assert is_snapshot_inference_enabled() is False


class TestSkipPathEmitsSnapshot:
    """Tests that skip paths emit snapshots via _log_decision.
    
    Uses direct imports from strategies.base to avoid full strategy dependency chain.
    """

    @pytest.fixture
    def mock_engine(self):
        """Create a mock engine with db_logger."""
        engine = MagicMock()
        engine.mode = MagicMock()
        engine.mode.value = "simulation"
        engine.db_logger = MagicMock()
        engine.db_logger.log_decision_snapshot = AsyncMock()
        return engine

    @pytest.fixture
    def mock_market_data(self):
        """Create mock market data."""
        data = MagicMock()
        data.symbol = "TSLA"
        data.close = 250.0
        data.high = 252.0
        data.low = 248.0
        data.open = 249.0
        data.indicators = {"rsi": 45, "volume_ratio": 1.2, "atr": 3.0}
        data.timestamp = datetime.now()
        return data

    @pytest.mark.asyncio
    async def test_strategy_skip_emits_snapshot(self, mock_engine, mock_market_data):
        """Strategy _log_decision on skip path triggers snapshot emission.
        
        This test creates a minimal strategy subclass to verify the base
        class _emit_snapshot functionality works correctly.
        """
        # Import just the base class module directly
        import importlib.util
        import sys
        
        # Load base module without triggering full strategy imports
        spec = importlib.util.spec_from_file_location(
            "strategies_base", 
            "/workspace/strategies/base.py"
        )
        base_module = importlib.util.module_from_spec(spec)
        
        # Need core modules for the import
        sys.modules['strategies_base'] = base_module
        spec.loader.exec_module(base_module)
        
        TradingStrategyWithCommentary = base_module.TradingStrategyWithCommentary
        
        # Create a concrete strategy subclass for testing
        class TestStrategy(TradingStrategyWithCommentary):
            name = "test_strategy"
            
            async def generate_signal_with_commentary(self, market_data):
                return None
        
        strategy = TestStrategy(MagicMock())
        strategy._engine_ref = mock_engine
        
        # Mock the event loop
        with patch('asyncio.get_event_loop') as mock_loop:
            mock_loop.return_value.create_task = MagicMock()
            
            # Call _log_decision (which calls _emit_snapshot internally)
            strategy._log_decision(
                mock_market_data,
                "skip",
                "gate_a_not_extreme",
                gate_name="gate_a_statistical_extreme",
                rsi=45,
            )
            
            # Verify snapshot was tracked
            assert strategy._last_snapshot_id is not None
            assert strategy._last_snapshot_ts is not None
            
            # Verify async task was created for db write
            mock_loop.return_value.create_task.assert_called_once()

    def test_get_snapshot_metadata(self, mock_engine, mock_market_data):
        """get_snapshot_metadata returns tracked snapshot info."""
        import importlib.util
        import sys
        
        spec = importlib.util.spec_from_file_location(
            "strategies_base", 
            "/workspace/strategies/base.py"
        )
        base_module = importlib.util.module_from_spec(spec)
        sys.modules['strategies_base'] = base_module
        spec.loader.exec_module(base_module)
        
        TradingStrategyWithCommentary = base_module.TradingStrategyWithCommentary
        
        class TestStrategy(TradingStrategyWithCommentary):
            name = "test_strategy"
            
            async def generate_signal_with_commentary(self, market_data):
                return None
        
        strategy = TestStrategy(MagicMock())
        strategy._engine_ref = mock_engine
        
        # Before any snapshot
        assert strategy.get_snapshot_metadata() == {}
        
        # After emitting a snapshot
        with patch('asyncio.get_event_loop') as mock_loop:
            mock_loop.return_value.create_task = MagicMock()
            
            strategy._log_decision(
                mock_market_data,
                "skip",
                "gate_b_no_capitulation",
                gate_name="gate_b_capitulation_volume",
            )
        
        metadata = strategy.get_snapshot_metadata()
        assert "snapshot_id" in metadata
        assert "snapshot_ts" in metadata
        assert metadata["snapshot_strategy"] == "test_strategy"


# Skip the OversoldBounceV2 integration test since it requires full strategy chain
# The unit tests above verify the base functionality works correctly.
