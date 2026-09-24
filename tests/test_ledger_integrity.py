"""v-ledger-integrity-2026-09-24 (PR1): Tests for trade ledger integrity.

Tests cover:
  1. R-multiple computation with initial_stop, stop_loss, ATR fallback
  2. Setup type inference from reasoning
  3. Exit reason normalization
  4. Broker orphan entry creation
  5. Trade ledger entry building
  6. Hands-off and external position exclusion
  7. Trade statistics computation
  8. Config flag behavior
"""
import pytest
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from typing import Dict, Any, Optional
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


class TestComputeRMultiple:
    """Tests for compute_r_multiple function."""

    def test_long_with_initial_stop(self):
        """R = (exit - entry) / (entry - initial_stop) for longs."""
        from core.ledger_integrity import compute_r_multiple
        
        pnl_r, risk = compute_r_multiple(
            side="long",
            entry_price=100.0,
            exit_price=104.0,
            initial_stop=98.0,
            stop_loss=None,
            atr=None,
        )
        
        assert risk == 2.0  # 100 - 98
        assert pnl_r == 2.0  # 4 / 2

    def test_long_losing_trade(self):
        """Losing long trade should have negative R."""
        from core.ledger_integrity import compute_r_multiple
        
        pnl_r, risk = compute_r_multiple(
            side="long",
            entry_price=100.0,
            exit_price=98.0,
            initial_stop=98.0,
            stop_loss=None,
            atr=None,
        )
        
        assert risk == 2.0
        assert pnl_r == -1.0  # Lost exactly 1R (hit stop)

    def test_short_with_initial_stop(self):
        """R = (entry - exit) / (initial_stop - entry) for shorts."""
        from core.ledger_integrity import compute_r_multiple
        
        pnl_r, risk = compute_r_multiple(
            side="short",
            entry_price=100.0,
            exit_price=96.0,
            initial_stop=102.0,
            stop_loss=None,
            atr=None,
        )
        
        assert risk == 2.0  # 102 - 100
        assert pnl_r == 2.0  # 4 / 2

    def test_stop_loss_fallback_when_initial_stop_none(self):
        """Falls back to stop_loss when initial_stop is None."""
        from core.ledger_integrity import compute_r_multiple
        
        pnl_r, risk = compute_r_multiple(
            side="long",
            entry_price=100.0,
            exit_price=103.0,
            initial_stop=None,
            stop_loss=97.0,
            atr=None,
        )
        
        assert risk == 3.0  # 100 - 97
        assert pnl_r == 1.0  # 3 / 3

    def test_atr_fallback_when_no_stop(self):
        """Falls back to 1.5*ATR when no stop is available."""
        from core.ledger_integrity import compute_r_multiple
        
        pnl_r, risk = compute_r_multiple(
            side="long",
            entry_price=100.0,
            exit_price=103.0,
            initial_stop=None,
            stop_loss=None,
            atr=2.0,
        )
        
        assert risk == 3.0  # 1.5 * 2
        assert pnl_r == 1.0  # 3 / 3

    def test_stop_equals_entry_uses_atr_fallback(self):
        """When stop_loss == entry (bug), falls back to ATR."""
        from core.ledger_integrity import compute_r_multiple
        
        pnl_r, risk = compute_r_multiple(
            side="long",
            entry_price=100.0,
            exit_price=103.0,
            initial_stop=None,
            stop_loss=100.0,  # Same as entry - invalid
            atr=2.0,
        )
        
        assert risk == 3.0  # 1.5 * ATR
        assert pnl_r == 1.0

    def test_no_stop_no_atr_returns_none(self):
        """Returns (None, None) when R cannot be computed."""
        from core.ledger_integrity import compute_r_multiple
        
        pnl_r, risk = compute_r_multiple(
            side="long",
            entry_price=100.0,
            exit_price=103.0,
            initial_stop=None,
            stop_loss=None,
            atr=None,
        )
        
        assert pnl_r is None
        assert risk is None


class TestInferSetupType:
    """Tests for setup type inference."""

    def test_direct_setup_type_field(self):
        """Direct setup_type field is used."""
        from core.ledger_integrity import infer_setup_type
        
        result = infer_setup_type({"setup_type": "breakout"})
        assert result == "breakout"

    def test_entry_pattern_breakout(self):
        """entry_pattern containing breakout."""
        from core.ledger_integrity import infer_setup_type
        
        result = infer_setup_type({"entry_pattern": "20_bar_breakout"})
        assert result == "breakout"

    def test_entry_pattern_pullback(self):
        """entry_pattern containing pullback."""
        from core.ledger_integrity import infer_setup_type
        
        result = infer_setup_type({"entry_pattern": "pullback_to_vwap"})
        assert result == "pullback"

    def test_entry_pattern_continuation(self):
        """entry_pattern containing continuation."""
        from core.ledger_integrity import infer_setup_type
        
        result = infer_setup_type({"entry_pattern": "trend_continuation"})
        assert result == "continuation"

    def test_strategy_mean_reversion(self):
        """Strategy field with mean_reversion."""
        from core.ledger_integrity import infer_setup_type
        
        result = infer_setup_type({"strategy": "mean_reversion"})
        assert result == "mean_reversion"

    def test_empty_reasoning_returns_unknown(self):
        """Empty or None reasoning returns unknown."""
        from core.ledger_integrity import infer_setup_type
        
        assert infer_setup_type({}) == "unknown"
        assert infer_setup_type(None) == "unknown"


class TestNormalizeExitReason:
    """Tests for exit reason normalization."""

    def test_direct_match(self):
        """Exact exit reason strings pass through."""
        from core.ledger_integrity import normalize_exit_reason
        
        assert normalize_exit_reason("take_profit") == "take_profit"
        assert normalize_exit_reason("stop_loss") == "stop_loss"

    def test_broker_stop(self):
        """Broker stop variations normalize to broker_stop."""
        from core.ledger_integrity import normalize_exit_reason
        
        assert normalize_exit_reason("stop_loss_broker") == "broker_stop"
        assert normalize_exit_reason("broker_stop_filled") == "broker_stop"

    def test_proactive_macd(self):
        """MACD variations normalize to proactive_macd."""
        from core.ledger_integrity import normalize_exit_reason
        
        assert normalize_exit_reason("proactive_macd_flipped_bearish") == "proactive_macd"
        assert normalize_exit_reason("macd_exit") == "proactive_macd"

    def test_flatten_hour(self):
        """Flatten variations normalize to flatten_hour."""
        from core.ledger_integrity import normalize_exit_reason
        
        assert normalize_exit_reason("day_trade_flatten_hour") == "flatten_hour"
        assert normalize_exit_reason("flatten_15") == "flatten_hour"

    def test_external_close(self):
        """External close variations."""
        from core.ledger_integrity import normalize_exit_reason
        
        assert normalize_exit_reason("external_close") == "external_close"
        assert normalize_exit_reason("external") == "external_close"


class TestBuildBrokerOrphanEntry:
    """Tests for broker orphan entry creation."""

    def test_basic_orphan(self):
        """Basic broker orphan entry creation."""
        from core.ledger_integrity import build_broker_orphan_entry
        
        entry = build_broker_orphan_entry(
            symbol="CRCL",
            side="long",
            entry_price=10.0,
            exit_price=9.0,
            quantity=100,
            entry_time=datetime.now() - timedelta(minutes=30),
            exit_time=datetime.now(),
            exit_reason="external_reconcile",
        )
        
        assert entry.symbol == "CRCL"
        assert entry.source == "broker_orphan"
        assert entry.pnl == -100.0  # (9-10) * 100
        assert entry.pnl_r == -1.0  # Convention for orphan stops
        assert entry.setup_type == "unknown"
        assert entry.is_hands_off is False

    def test_hands_off_symbol(self):
        """Hands-off symbols are flagged."""
        from core.ledger_integrity import build_broker_orphan_entry
        
        entry = build_broker_orphan_entry(
            symbol="MU",
            side="long",
            entry_price=100.0,
            exit_price=105.0,
            quantity=10,
            entry_time=datetime.now() - timedelta(hours=1),
            exit_time=datetime.now(),
        )
        
        assert entry.is_hands_off is True

    def test_atr_for_r_computation(self):
        """ATR is used for R computation in orphans."""
        from core.ledger_integrity import build_broker_orphan_entry
        
        entry = build_broker_orphan_entry(
            symbol="NVDA",
            side="long",
            entry_price=100.0,
            exit_price=106.0,
            quantity=10,
            entry_time=datetime.now() - timedelta(hours=1),
            exit_time=datetime.now(),
            atr=2.0,
        )
        
        # risk = 1.5 * 2 = 3, pnl = 6, R = 2
        assert entry.pnl_r == 2.0


class TestBuildTradeLedgerEntry:
    """Tests for TradeLedgerEntry building from positions."""

    @dataclass
    class MockPosition:
        symbol: str = "TSLA"
        side: str = "long"
        entry_price: float = 200.0
        quantity: int = 10
        stop_loss: float = 195.0
        take_profit: float = 210.0
        entry_time: datetime = None
        reasoning: Dict[str, Any] = None
        scaled_out: bool = False
        managed_by_bot: bool = True
        initial_stop: Optional[float] = None
        initial_tp: Optional[float] = None
        initial_risk_per_share: Optional[float] = None
        
        def __post_init__(self):
            if self.entry_time is None:
                self.entry_time = datetime.now() - timedelta(hours=1)
            if self.reasoning is None:
                self.reasoning = {"strategy": "day_trade_momentum", "atr": 3.0}

    def test_basic_entry(self):
        """Basic trade ledger entry creation."""
        from core.ledger_integrity import build_trade_ledger_entry
        
        pos = self.MockPosition()
        entry = build_trade_ledger_entry(
            position=pos,
            exit_price=208.0,
            exit_time=datetime.now(),
            exit_reason="take_profit",
            pnl=80.0,
            pnl_pct=4.0,
        )
        
        assert entry.symbol == "TSLA"
        assert entry.source == "bot"
        assert entry.pnl == 80.0
        assert entry.exit_reason == "take_profit"
        assert entry.is_hands_off is False
        assert entry.is_external is False

    def test_uses_initial_stop_for_r(self):
        """Uses initial_stop field for R computation."""
        from core.ledger_integrity import build_trade_ledger_entry
        
        pos = self.MockPosition()
        pos.initial_stop = 197.0  # 3.0 risk per share
        pos.stop_loss = 199.0  # Current stop (after breakeven lift)
        
        entry = build_trade_ledger_entry(
            position=pos,
            exit_price=206.0,  # +6 profit per share
            exit_time=datetime.now(),
            exit_reason="take_profit",
            pnl=60.0,
            pnl_pct=3.0,
        )
        
        # R = 6 / 3 = 2.0 (uses initial_stop, not current stop_loss)
        assert entry.pnl_r == 2.0
        assert entry.initial_stop == 197.0
        assert entry.initial_risk_per_share == 3.0

    def test_hands_off_denylist(self):
        """Hands-off symbols are flagged correctly."""
        from core.ledger_integrity import build_trade_ledger_entry
        
        pos = self.MockPosition()
        pos.symbol = "HQGE"
        
        entry = build_trade_ledger_entry(
            position=pos,
            exit_price=200.5,
            exit_time=datetime.now(),
            exit_reason="stop_loss",
            pnl=-5.0,
            pnl_pct=-0.25,
        )
        
        assert entry.is_hands_off is True

    def test_external_position_flagged(self):
        """External/unmanaged positions are flagged."""
        from core.ledger_integrity import build_trade_ledger_entry
        
        pos = self.MockPosition()
        pos.managed_by_bot = False
        pos.reasoning = {"source": "external"}
        
        entry = build_trade_ledger_entry(
            position=pos,
            exit_price=205.0,
            exit_time=datetime.now(),
            exit_reason="external_close",
            pnl=50.0,
            pnl_pct=2.5,
        )
        
        assert entry.is_external is True


class TestIsBotManagedTrade:
    """Tests for bot-managed trade filtering."""

    def test_normal_bot_trade_included(self):
        """Normal bot trades are included in stats."""
        from core.ledger_integrity import TradeLedgerEntry, is_bot_managed_trade
        
        entry = TradeLedgerEntry(
            symbol="TSLA",
            side="long",
            strategy="day_trade_momentum",
            entry_time=datetime.now(),
            exit_time=datetime.now(),
            entry_price=200.0,
            exit_price=210.0,
            quantity=10,
            pnl=100.0,
            pnl_pct=5.0,
            pnl_r=2.0,
            exit_reason="take_profit",
            setup_type="continuation",
            source="bot",
            hold_time_seconds=3600,
            initial_stop=195.0,
            initial_tp=210.0,
            initial_risk_per_share=5.0,
            atr_at_entry=3.0,
            stop_loss=195.0,
            take_profit=210.0,
            confidence=0.75,
            meta_proba=0.65,
            kelly_fraction=0.15,
            scaled_out=False,
            mode="live",
            is_hands_off=False,
            is_external=False,
            reasoning={},
        )
        
        assert is_bot_managed_trade(entry) is True

    def test_hands_off_excluded(self):
        """Hands-off trades are excluded from stats."""
        from core.ledger_integrity import TradeLedgerEntry, is_bot_managed_trade
        
        entry = TradeLedgerEntry(
            symbol="MU",
            side="long",
            strategy=None,
            entry_time=datetime.now(),
            exit_time=datetime.now(),
            entry_price=80.0,
            exit_price=85.0,
            quantity=100,
            pnl=500.0,
            pnl_pct=6.25,
            pnl_r=None,
            exit_reason="external_close",
            setup_type="unknown",
            source="external",
            hold_time_seconds=86400,
            initial_stop=None,
            initial_tp=None,
            initial_risk_per_share=None,
            atr_at_entry=None,
            stop_loss=None,
            take_profit=None,
            confidence=None,
            meta_proba=None,
            kelly_fraction=None,
            scaled_out=False,
            mode="live",
            is_hands_off=True,
            is_external=True,
            reasoning={},
        )
        
        assert is_bot_managed_trade(entry) is False

    def test_external_excluded(self):
        """External trades are excluded from stats."""
        from core.ledger_integrity import TradeLedgerEntry, is_bot_managed_trade
        
        entry = TradeLedgerEntry(
            symbol="META",
            side="short",
            strategy=None,
            entry_time=datetime.now(),
            exit_time=datetime.now(),
            entry_price=500.0,
            exit_price=480.0,
            quantity=5,
            pnl=100.0,
            pnl_pct=4.0,
            pnl_r=None,
            exit_reason="external_close",
            setup_type="unknown",
            source="external",
            hold_time_seconds=0,
            initial_stop=None,
            initial_tp=None,
            initial_risk_per_share=None,
            atr_at_entry=None,
            stop_loss=None,
            take_profit=None,
            confidence=None,
            meta_proba=None,
            kelly_fraction=None,
            scaled_out=False,
            mode="live",
            is_hands_off=False,
            is_external=True,
            reasoning={},
        )
        
        assert is_bot_managed_trade(entry) is False

    def test_broker_orphan_included(self):
        """Broker orphans ARE included in stats (they were bot trades)."""
        from core.ledger_integrity import TradeLedgerEntry, is_bot_managed_trade
        
        entry = TradeLedgerEntry(
            symbol="CRCL",
            side="long",
            strategy="day_trade_momentum",
            entry_time=datetime.now(),
            exit_time=datetime.now(),
            entry_price=10.0,
            exit_price=9.0,
            quantity=100,
            pnl=-100.0,
            pnl_pct=-10.0,
            pnl_r=-1.0,
            exit_reason="external_reconcile",
            setup_type="unknown",
            source="broker_orphan",
            hold_time_seconds=300,
            initial_stop=None,
            initial_tp=None,
            initial_risk_per_share=None,
            atr_at_entry=None,
            stop_loss=None,
            take_profit=None,
            confidence=None,
            meta_proba=None,
            kelly_fraction=None,
            scaled_out=False,
            mode="live",
            is_hands_off=False,
            is_external=False,
            reasoning={},
        )
        
        assert is_bot_managed_trade(entry) is True


class TestComputeTradeStats:
    """Tests for aggregate trade statistics."""

    def test_empty_trades(self):
        """Empty trade list returns zeros."""
        from core.ledger_integrity import compute_trade_stats
        
        stats = compute_trade_stats([])
        
        assert stats["n"] == 0
        assert stats["win_rate"] == 0.0
        assert stats["exp_r"] == 0.0

    def test_basic_stats(self):
        """Basic statistics computation."""
        from core.ledger_integrity import TradeLedgerEntry, compute_trade_stats
        
        def make_entry(symbol, pnl_r, pnl):
            return TradeLedgerEntry(
                symbol=symbol,
                side="long",
                strategy="test",
                entry_time=datetime.now(),
                exit_time=datetime.now(),
                entry_price=100.0,
                exit_price=100.0 + pnl/10,
                quantity=10,
                pnl=pnl,
                pnl_pct=pnl/10,
                pnl_r=pnl_r,
                exit_reason="test",
                setup_type="unknown",
                source="bot",
                hold_time_seconds=100,
                initial_stop=None,
                initial_tp=None,
                initial_risk_per_share=None,
                atr_at_entry=None,
                stop_loss=None,
                take_profit=None,
                confidence=None,
                meta_proba=None,
                kelly_fraction=None,
                scaled_out=False,
                mode="live",
                is_hands_off=False,
                is_external=False,
                reasoning={},
            )
        
        entries = [
            make_entry("A", 2.0, 200),   # Winner
            make_entry("B", -1.0, -100), # Loser
            make_entry("C", 1.5, 150),   # Winner
            make_entry("D", -0.5, -50),  # Loser
            make_entry("E", 0.02, 2),    # Scratch
        ]
        
        stats = compute_trade_stats(entries)
        
        assert stats["n"] == 5
        assert stats["winners"] == 2
        assert stats["losers"] == 2
        assert stats["scratches"] == 1
        # Win rate = 2/(2+2) = 50%
        assert stats["win_rate"] == 50.0
        # Total R = 2 - 1 + 1.5 - 0.5 + 0.02 = 2.02
        assert abs(stats["total_r"] - 2.02) < 0.01
        # Total PnL = 200 - 100 + 150 - 50 + 2 = 202
        assert stats["total_pnl"] == 202.0

    def test_excludes_hands_off(self):
        """Hands-off trades are excluded from stats."""
        from core.ledger_integrity import TradeLedgerEntry, compute_trade_stats
        
        entries = [
            TradeLedgerEntry(
                symbol="MU",
                side="long",
                strategy=None,
                entry_time=datetime.now(),
                exit_time=datetime.now(),
                entry_price=80.0,
                exit_price=90.0,
                quantity=100,
                pnl=1000.0,
                pnl_pct=12.5,
                pnl_r=5.0,
                exit_reason="take_profit",
                setup_type="unknown",
                source="external",
                hold_time_seconds=86400,
                initial_stop=None,
                initial_tp=None,
                initial_risk_per_share=None,
                atr_at_entry=None,
                stop_loss=None,
                take_profit=None,
                confidence=None,
                meta_proba=None,
                kelly_fraction=None,
                scaled_out=False,
                mode="live",
                is_hands_off=True,  # Excluded
                is_external=True,
                reasoning={},
            )
        ]
        
        stats = compute_trade_stats(entries)
        
        assert stats["n"] == 0  # MU excluded


class TestConfigFlag:
    """Tests for LEDGER_INTEGRITY config flag."""

    def test_flag_exists(self):
        """LEDGER_INTEGRITY config property exists."""
        from core.config import Config
        
        cfg = Config()
        # Should not raise
        value = cfg.LEDGER_INTEGRITY
        assert isinstance(value, bool)

    def test_default_on(self):
        """LEDGER_INTEGRITY defaults to True (safe for read-mostly)."""
        import os
        from core.config import Config
        
        # Clear any env override
        old_val = os.environ.pop("LEDGER_INTEGRITY", None)
        try:
            cfg = Config()
            # Default should be True per the docstring
            assert cfg.LEDGER_INTEGRITY is True
        finally:
            if old_val is not None:
                os.environ["LEDGER_INTEGRITY"] = old_val


class TestPositionModelFields:
    """Tests for Position model ledger integrity fields."""

    def test_initial_fields_exist(self):
        """Position model has initial_stop, initial_tp, initial_risk_per_share."""
        from core.models import Position
        from datetime import datetime
        
        pos = Position(
            symbol="TEST",
            entry_price=100.0,
            quantity=10,
            side="long",
            stop_loss=95.0,
            take_profit=110.0,
            entry_time=datetime.now(),
            initial_stop=95.0,
            initial_tp=110.0,
            initial_risk_per_share=5.0,
        )
        
        assert pos.initial_stop == 95.0
        assert pos.initial_tp == 110.0
        assert pos.initial_risk_per_share == 5.0

    def test_initial_fields_default_none(self):
        """Initial fields default to None for backward compat."""
        from core.models import Position
        from datetime import datetime
        
        pos = Position(
            symbol="TEST",
            entry_price=100.0,
            quantity=10,
            side="long",
            stop_loss=95.0,
            take_profit=110.0,
            entry_time=datetime.now(),
        )
        
        assert pos.initial_stop is None
        assert pos.initial_tp is None
        assert pos.initial_risk_per_share is None


class TestDbLoggerMethods:
    """Tests for db_logger ledger integrity methods."""

    def test_ensure_ledger_integrity_columns_exists(self):
        """ensure_ledger_integrity_columns method exists."""
        from data_providers.db_logger import DbLogger
        
        assert hasattr(DbLogger, "ensure_ledger_integrity_columns")

    def test_log_trade_v2_exists(self):
        """log_trade_v2 method exists with new parameters."""
        from data_providers.db_logger import DbLogger
        import inspect
        
        assert hasattr(DbLogger, "log_trade_v2")
        sig = inspect.signature(DbLogger.log_trade_v2)
        params = list(sig.parameters.keys())
        
        assert "pnl_r" in params
        assert "setup_type" in params
        assert "source" in params
        assert "hold_time_seconds" in params
        assert "initial_stop" in params
        assert "initial_tp" in params
        assert "initial_risk_per_share" in params
        assert "is_hands_off" in params
        assert "is_external" in params


class TestComputeHoldTime:
    """Tests for hold time computation."""

    def test_basic_hold_time(self):
        """Basic hold time computation."""
        from core.ledger_integrity import compute_hold_time_seconds
        
        entry = datetime(2026, 9, 24, 9, 30, 0)
        exit = datetime(2026, 9, 24, 10, 30, 0)
        
        hold = compute_hold_time_seconds(entry, exit)
        
        assert hold == 3600  # 1 hour

    def test_none_times_return_zero(self):
        """None entry/exit times return 0."""
        from core.ledger_integrity import compute_hold_time_seconds
        
        assert compute_hold_time_seconds(None, datetime.now()) == 0
        assert compute_hold_time_seconds(datetime.now(), None) == 0


class TestExtractInitialValues:
    """Tests for extracting initial values from position."""

    def test_position_attributes(self):
        """Uses position attributes first."""
        from core.ledger_integrity import extract_initial_values_from_position
        from dataclasses import dataclass
        
        @dataclass
        class MockPos:
            initial_stop: float = 95.0
            initial_tp: float = 110.0
            initial_risk_per_share: float = 5.0
            entry_price: float = 100.0
            side: str = "long"
        
        initial_stop, initial_tp, initial_risk = extract_initial_values_from_position(MockPos())
        
        assert initial_stop == 95.0
        assert initial_tp == 110.0
        assert initial_risk == 5.0

    def test_reasoning_fallback(self):
        """Falls back to reasoning dict."""
        from core.ledger_integrity import extract_initial_values_from_position
        from dataclasses import dataclass
        
        @dataclass
        class MockPos:
            initial_stop: float = None
            initial_tp: float = None
            initial_risk_per_share: float = None
            entry_price: float = 100.0
            side: str = "long"
            stop_loss: float = 95.0
            take_profit: float = 110.0
            reasoning: dict = None
        
        pos = MockPos()
        pos.reasoning = {"stop_distance": 5.0}
        
        initial_stop, initial_tp, initial_risk = extract_initial_values_from_position(pos)
        
        # stop_loss used as fallback
        assert initial_stop == 95.0
        # risk computed from stop
        assert initial_risk == 5.0


class TestApiTradesEndpoint:
    """Tests for /api/trades endpoint updates."""

    def test_sql_includes_new_columns(self):
        """The /api/trades SQL query includes ledger integrity columns."""
        from pathlib import Path
        
        routes_src = (REPO_ROOT / "api" / "routes.py").read_text()
        
        # Find the /api/trades endpoint SQL
        assert "pnl_r" in routes_src
        assert "setup_type" in routes_src
        assert "source" in routes_src
        assert "hold_time_seconds" in routes_src
        assert "initial_stop" in routes_src
        assert "is_hands_off" in routes_src
        assert "is_external" in routes_src
