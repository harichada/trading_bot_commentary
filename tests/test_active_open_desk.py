"""Tests for the Active Open Desk continuous position monitor.

v-active-open-desk-2026-09-14: tests covering:
  - Config flags: ENABLE_ACTIVE_OPEN_DESK, ACTIVE_OPEN_DESK_SHADOW, ACTIVE_OPEN_DESK_INTERVAL_SEC
  - Flag off = no desk activity (should_start_desk returns False)
  - Flag on + shadow = would-be actions logged (no broker calls)
  - HANDS_OFF_DENYLIST symbols skipped
  - Parallel asyncio.gather invoked for multiple positions
  - DeskDecision dataclass serialization
"""
import asyncio
import pytest
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import dataclass

from analysis.active_open_desk import (
    ActiveOpenDesk,
    DeskAction,
    DeskDecision,
    should_start_desk,
)


@dataclass
class MockPosition:
    """Minimal Position mock for testing."""
    symbol: str
    entry_price: float
    stop_loss: float
    take_profit: float
    side: str
    quantity: int = 100
    managed_by_bot: bool = True
    is_external: bool = False
    is_manually_managed: bool = False
    is_long_term: bool = False
    current_price: float = 0.0
    
    def __post_init__(self):
        if self.current_price == 0.0:
            self.current_price = self.entry_price


@pytest.fixture
def mock_engine():
    """Create a mock engine with the required attributes."""
    engine = MagicMock()
    engine.is_running = True
    engine.positions = {}
    engine.simulated_positions = {}
    engine.data_provider = None
    engine.technical_analyzer = MagicMock()
    engine.db_logger = None
    return engine


class TestShouldStartDesk:
    """Test the should_start_desk() configuration gate."""

    def test_returns_true_by_default(self, monkeypatch):
        """Default config: ENABLE_ACTIVE_OPEN_DESK=True → should_start_desk=True.
        
        Default True + ACTIVE_OPEN_DESK_SHADOW=True enables evidence soak.
        """
        monkeypatch.delenv("ENABLE_ACTIVE_OPEN_DESK", raising=False)
        assert should_start_desk() is True

    def test_env_1_enables(self, monkeypatch):
        """env ENABLE_ACTIVE_OPEN_DESK=1 → should_start_desk=True."""
        monkeypatch.setenv("ENABLE_ACTIVE_OPEN_DESK", "1")
        assert should_start_desk() is True

    def test_env_true_enables(self, monkeypatch):
        """env ENABLE_ACTIVE_OPEN_DESK=true → should_start_desk=True."""
        monkeypatch.setenv("ENABLE_ACTIVE_OPEN_DESK", "true")
        assert should_start_desk() is True

    def test_env_0_disables_safe_off_path(self, monkeypatch):
        """SAFE OFF-PATH: env ENABLE_ACTIVE_OPEN_DESK=0 → should_start_desk=False.
        
        This is the escape hatch to disable entirely and preserve today's
        bracket + hard-stop behavior unchanged.
        """
        monkeypatch.setenv("ENABLE_ACTIVE_OPEN_DESK", "0")
        assert should_start_desk() is False

    def test_env_false_disables_safe_off_path(self, monkeypatch):
        """SAFE OFF-PATH: env ENABLE_ACTIVE_OPEN_DESK=false → should_start_desk=False."""
        monkeypatch.setenv("ENABLE_ACTIVE_OPEN_DESK", "false")
        assert should_start_desk() is False


class TestConfigFlags:
    """Test all Active Open Desk config flags."""

    def test_shadow_default_true(self, monkeypatch):
        """Default: ACTIVE_OPEN_DESK_SHADOW=True (safe shadow mode)."""
        monkeypatch.delenv("ACTIVE_OPEN_DESK_SHADOW", raising=False)
        from core.config import Config
        assert Config().ACTIVE_OPEN_DESK_SHADOW is True

    def test_shadow_env_0_disables(self, monkeypatch):
        """env ACTIVE_OPEN_DESK_SHADOW=0 → False (enables live actuators)."""
        monkeypatch.setenv("ACTIVE_OPEN_DESK_SHADOW", "0")
        from core.config import Config
        assert Config().ACTIVE_OPEN_DESK_SHADOW is False

    def test_interval_default(self, monkeypatch):
        """Default interval is 5.0 seconds."""
        monkeypatch.delenv("ACTIVE_OPEN_DESK_INTERVAL_SEC", raising=False)
        from core.config import Config
        assert Config().ACTIVE_OPEN_DESK_INTERVAL_SEC == 5.0

    def test_interval_env_override(self, monkeypatch):
        """env ACTIVE_OPEN_DESK_INTERVAL_SEC=10 → 10.0."""
        monkeypatch.setenv("ACTIVE_OPEN_DESK_INTERVAL_SEC", "10")
        from core.config import Config
        assert Config().ACTIVE_OPEN_DESK_INTERVAL_SEC == 10.0


class TestGetMonitoredPositions:
    """Test position filtering logic."""

    def test_empty_when_no_positions(self, mock_engine):
        """Returns empty list when no positions exist."""
        desk = ActiveOpenDesk(mock_engine)
        assert desk.get_monitored_positions() == []

    def test_includes_managed_by_bot_positions(self, mock_engine):
        """Includes positions with managed_by_bot=True."""
        pos = MockPosition(
            symbol="AAPL",
            entry_price=150.0,
            stop_loss=145.0,
            take_profit=160.0,
            side="long",
            managed_by_bot=True,
        )
        mock_engine.positions = {"AAPL": pos}
        
        desk = ActiveOpenDesk(mock_engine)
        result = desk.get_monitored_positions()
        
        assert len(result) == 1
        assert result[0][0] == "AAPL"
        assert result[0][1] is pos

    def test_excludes_non_managed_positions(self, mock_engine):
        """Excludes positions with managed_by_bot=False."""
        pos = MockPosition(
            symbol="AAPL",
            entry_price=150.0,
            stop_loss=145.0,
            take_profit=160.0,
            side="long",
            managed_by_bot=False,
        )
        mock_engine.positions = {"AAPL": pos}
        
        desk = ActiveOpenDesk(mock_engine)
        result = desk.get_monitored_positions()
        
        assert len(result) == 0

    def test_excludes_hands_off_denylist(self, mock_engine):
        """Excludes symbols in HANDS_OFF_DENYLIST (MU, HQGE, SPCX)."""
        pos_mu = MockPosition(
            symbol="MU",
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            side="long",
            managed_by_bot=True,
        )
        pos_hqge = MockPosition(
            symbol="HQGE",
            entry_price=50.0,
            stop_loss=45.0,
            take_profit=60.0,
            side="long",
            managed_by_bot=True,
        )
        pos_spcx = MockPosition(
            symbol="SPCX",
            entry_price=25.0,
            stop_loss=22.0,
            take_profit=30.0,
            side="long",
            managed_by_bot=True,
        )
        pos_aapl = MockPosition(
            symbol="AAPL",
            entry_price=150.0,
            stop_loss=145.0,
            take_profit=160.0,
            side="long",
            managed_by_bot=True,
        )
        mock_engine.positions = {
            "MU": pos_mu,
            "HQGE": pos_hqge,
            "SPCX": pos_spcx,
            "AAPL": pos_aapl,
        }
        
        desk = ActiveOpenDesk(mock_engine)
        result = desk.get_monitored_positions()
        
        symbols = [r[0] for r in result]
        assert "MU" not in symbols
        assert "HQGE" not in symbols
        assert "SPCX" not in symbols
        assert "AAPL" in symbols
        assert len(result) == 1

    def test_excludes_external_positions(self, mock_engine):
        """Excludes positions marked as external."""
        pos = MockPosition(
            symbol="AAPL",
            entry_price=150.0,
            stop_loss=145.0,
            take_profit=160.0,
            side="long",
            managed_by_bot=True,
            is_external=True,
        )
        mock_engine.positions = {"AAPL": pos}
        
        desk = ActiveOpenDesk(mock_engine)
        result = desk.get_monitored_positions()
        
        assert len(result) == 0

    def test_excludes_manually_managed_positions(self, mock_engine):
        """Excludes positions marked as manually managed."""
        pos = MockPosition(
            symbol="AAPL",
            entry_price=150.0,
            stop_loss=145.0,
            take_profit=160.0,
            side="long",
            managed_by_bot=True,
            is_manually_managed=True,
        )
        mock_engine.positions = {"AAPL": pos}
        
        desk = ActiveOpenDesk(mock_engine)
        result = desk.get_monitored_positions()
        
        assert len(result) == 0

    def test_excludes_long_term_positions(self, mock_engine):
        """Excludes positions marked as long-term."""
        pos = MockPosition(
            symbol="AAPL",
            entry_price=150.0,
            stop_loss=145.0,
            take_profit=160.0,
            side="long",
            managed_by_bot=True,
            is_long_term=True,
        )
        mock_engine.positions = {"AAPL": pos}
        
        desk = ActiveOpenDesk(mock_engine)
        result = desk.get_monitored_positions()
        
        assert len(result) == 0

    def test_includes_simulated_positions(self, mock_engine):
        """Includes positions from simulated_positions container."""
        pos = MockPosition(
            symbol="TSLA",
            entry_price=200.0,
            stop_loss=190.0,
            take_profit=220.0,
            side="long",
            managed_by_bot=True,
        )
        mock_engine.simulated_positions = {"TSLA": pos}
        
        desk = ActiveOpenDesk(mock_engine)
        result = desk.get_monitored_positions()
        
        assert len(result) == 1
        assert result[0][0] == "TSLA"


class TestDeskDecision:
    """Test DeskDecision dataclass."""

    def test_to_dict_serialization(self):
        """DeskDecision.to_dict() produces valid JSON-serializable output."""
        decision = DeskDecision(
            symbol="AAPL",
            action=DeskAction.WOULD_TIGHTEN,
            reason="regime_risk_off_tighten",
            current_stop=145.0,
            suggested_stop=147.5,
            current_price=150.0,
            regime="risk_off",
            sentiment_score=-0.5,
            rsi=65.0,
            macd_signal="bearish",
        )
        
        d = decision.to_dict()
        
        assert d["symbol"] == "AAPL"
        assert d["action"] == "WOULD_TIGHTEN"
        assert d["reason"] == "regime_risk_off_tighten"
        assert d["current_stop"] == 145.0
        assert d["suggested_stop"] == 147.5
        assert d["current_price"] == 150.0
        assert d["regime"] == "risk_off"
        assert d["sentiment_score"] == -0.5
        assert d["rsi"] == 65.0
        assert d["macd_signal"] == "bearish"
        assert "timestamp" in d

    def test_no_action_decision(self):
        """NO_ACTION decision is created with minimal fields."""
        decision = DeskDecision(
            symbol="AAPL",
            action=DeskAction.NO_ACTION,
            reason="",
        )
        
        assert decision.action == DeskAction.NO_ACTION
        assert decision.suggested_stop is None
        assert decision.suggested_exit_price is None


class TestEvaluatePosition:
    """Test position evaluation logic."""

    @pytest.mark.asyncio
    async def test_risk_off_triggers_tighten_for_long(self, mock_engine):
        """Long position in risk_off regime → WOULD_TIGHTEN."""
        pos = MockPosition(
            symbol="AAPL",
            entry_price=150.0,
            stop_loss=145.0,
            take_profit=160.0,
            side="long",
            current_price=152.0,
        )
        
        desk = ActiveOpenDesk(mock_engine)
        
        with patch.object(desk, '_get_market_context') as mock_ctx, \
             patch.object(desk, '_get_news_sentiment', new_callable=AsyncMock) as mock_news, \
             patch.object(desk, '_get_indicators', new_callable=AsyncMock) as mock_ind:
            
            mock_ctx.return_value = MagicMock(regime="risk_off")
            mock_news.return_value = 0.0
            mock_ind.return_value = {"rsi": 50.0, "macd": 0.0, "macd_signal": 0.0}
            
            decision = await desk._evaluate_position("AAPL", pos)
        
        assert decision.action == DeskAction.WOULD_TIGHTEN
        assert "risk_off" in decision.reason

    @pytest.mark.asyncio
    async def test_negative_sentiment_triggers_tighten_for_long(self, mock_engine):
        """Long position with negative sentiment → WOULD_TIGHTEN."""
        pos = MockPosition(
            symbol="AAPL",
            entry_price=150.0,
            stop_loss=145.0,
            take_profit=160.0,
            side="long",
            current_price=152.0,
        )
        
        desk = ActiveOpenDesk(mock_engine)
        
        with patch.object(desk, '_get_market_context') as mock_ctx, \
             patch.object(desk, '_get_news_sentiment', new_callable=AsyncMock) as mock_news, \
             patch.object(desk, '_get_indicators', new_callable=AsyncMock) as mock_ind:
            
            mock_ctx.return_value = MagicMock(regime="mixed")
            mock_news.return_value = -0.5
            mock_ind.return_value = {"rsi": 50.0, "macd": 0.0, "macd_signal": 0.0}
            
            decision = await desk._evaluate_position("AAPL", pos)
        
        assert decision.action == DeskAction.WOULD_TIGHTEN
        assert "negative_sentiment" in decision.reason

    @pytest.mark.asyncio
    async def test_extreme_rsi_triggers_exit_for_long(self, mock_engine):
        """Long position with RSI >= 80 → WOULD_EXIT."""
        pos = MockPosition(
            symbol="AAPL",
            entry_price=150.0,
            stop_loss=145.0,
            take_profit=160.0,
            side="long",
            current_price=158.0,
        )
        
        desk = ActiveOpenDesk(mock_engine)
        
        with patch.object(desk, '_get_market_context') as mock_ctx, \
             patch.object(desk, '_get_news_sentiment', new_callable=AsyncMock) as mock_news, \
             patch.object(desk, '_get_indicators', new_callable=AsyncMock) as mock_ind:
            
            mock_ctx.return_value = MagicMock(regime="risk_on")
            mock_news.return_value = 0.0
            mock_ind.return_value = {"rsi": 82.0, "macd": 0.5, "macd_signal": 0.3}
            
            decision = await desk._evaluate_position("AAPL", pos)
        
        assert decision.action == DeskAction.WOULD_EXIT
        assert "rsi_extreme_overbought" in decision.reason

    @pytest.mark.asyncio
    async def test_risk_on_triggers_tighten_for_short(self, mock_engine):
        """Short position in risk_on regime → WOULD_TIGHTEN."""
        pos = MockPosition(
            symbol="AAPL",
            entry_price=150.0,
            stop_loss=155.0,
            take_profit=140.0,
            side="short",
            current_price=148.0,
        )
        
        desk = ActiveOpenDesk(mock_engine)
        
        with patch.object(desk, '_get_market_context') as mock_ctx, \
             patch.object(desk, '_get_news_sentiment', new_callable=AsyncMock) as mock_news, \
             patch.object(desk, '_get_indicators', new_callable=AsyncMock) as mock_ind:
            
            mock_ctx.return_value = MagicMock(regime="risk_on")
            mock_news.return_value = 0.0
            mock_ind.return_value = {"rsi": 50.0, "macd": 0.0, "macd_signal": 0.0}
            
            decision = await desk._evaluate_position("AAPL", pos)
        
        assert decision.action == DeskAction.WOULD_TIGHTEN
        assert "risk_on" in decision.reason

    @pytest.mark.asyncio
    async def test_no_action_when_conditions_normal(self, mock_engine):
        """Position in normal conditions → NO_ACTION."""
        pos = MockPosition(
            symbol="AAPL",
            entry_price=150.0,
            stop_loss=145.0,
            take_profit=160.0,
            side="long",
            current_price=151.0,
        )
        
        desk = ActiveOpenDesk(mock_engine)
        
        with patch.object(desk, '_get_market_context') as mock_ctx, \
             patch.object(desk, '_get_news_sentiment', new_callable=AsyncMock) as mock_news, \
             patch.object(desk, '_get_indicators', new_callable=AsyncMock) as mock_ind:
            
            mock_ctx.return_value = MagicMock(regime="mixed")
            mock_news.return_value = 0.0
            mock_ind.return_value = {"rsi": 50.0, "macd": 0.1, "macd_signal": 0.05}
            
            decision = await desk._evaluate_position("AAPL", pos)
        
        assert decision.action == DeskAction.NO_ACTION


class TestParallelEvaluation:
    """Test that positions are evaluated in parallel via asyncio.gather."""

    @pytest.mark.asyncio
    async def test_multiple_positions_evaluated_in_parallel(self, mock_engine):
        """Multiple positions are evaluated via asyncio.gather."""
        pos1 = MockPosition(
            symbol="AAPL",
            entry_price=150.0,
            stop_loss=145.0,
            take_profit=160.0,
            side="long",
        )
        pos2 = MockPosition(
            symbol="TSLA",
            entry_price=200.0,
            stop_loss=190.0,
            take_profit=220.0,
            side="long",
        )
        mock_engine.positions = {"AAPL": pos1, "TSLA": pos2}
        
        desk = ActiveOpenDesk(mock_engine)
        
        evaluated_symbols = []
        
        async def mock_evaluate(symbol, position):
            evaluated_symbols.append(symbol)
            await asyncio.sleep(0.01)
            return DeskDecision(
                symbol=symbol,
                action=DeskAction.NO_ACTION,
                reason="",
            )
        
        with patch.object(desk, '_evaluate_position', side_effect=mock_evaluate):
            await desk._tick()
        
        assert "AAPL" in evaluated_symbols
        assert "TSLA" in evaluated_symbols
        assert len(evaluated_symbols) == 2


class TestShadowLogging:
    """Test shadow mode logging behavior."""

    @pytest.mark.asyncio
    async def test_shadow_mode_logs_decision(self, mock_engine, monkeypatch, caplog):
        """Shadow mode logs WOULD_* decisions without broker calls."""
        monkeypatch.setenv("ENABLE_ACTIVE_OPEN_DESK", "1")
        monkeypatch.setenv("ACTIVE_OPEN_DESK_SHADOW", "1")
        
        pos = MockPosition(
            symbol="AAPL",
            entry_price=150.0,
            stop_loss=145.0,
            take_profit=160.0,
            side="long",
            current_price=152.0,
        )
        mock_engine.positions = {"AAPL": pos}
        
        desk = ActiveOpenDesk(mock_engine)
        
        decision = DeskDecision(
            symbol="AAPL",
            action=DeskAction.WOULD_TIGHTEN,
            reason="test_reason",
            current_stop=145.0,
            suggested_stop=147.5,
            current_price=152.0,
            regime="risk_off",
            sentiment_score=-0.3,
            rsi=55.0,
            macd_signal="bearish",
        )
        
        import logging
        with caplog.at_level(logging.INFO):
            desk._log_shadow_decision(decision)
        
        assert "active_open_desk" in caplog.text
        assert "WOULD_TIGHTEN" in caplog.text
        assert "AAPL" in caplog.text
        assert "test_reason" in caplog.text


class TestComputeTighterStop:
    """Test stop tightening calculations."""

    def test_tighter_stop_for_profitable_long(self, mock_engine):
        """Long position in profit → stop moves up."""
        pos = MockPosition(
            symbol="AAPL",
            entry_price=150.0,
            stop_loss=145.0,
            take_profit=160.0,
            side="long",
        )
        
        desk = ActiveOpenDesk(mock_engine)
        new_stop = desk._compute_tighter_stop(pos, current_price=154.0)
        
        assert new_stop > pos.stop_loss
        assert new_stop == 152.0

    def test_tighter_stop_for_profitable_short(self, mock_engine):
        """Short position in profit → stop moves down."""
        pos = MockPosition(
            symbol="AAPL",
            entry_price=150.0,
            stop_loss=155.0,
            take_profit=140.0,
            side="short",
        )
        
        desk = ActiveOpenDesk(mock_engine)
        new_stop = desk._compute_tighter_stop(pos, current_price=146.0)
        
        assert new_stop < pos.stop_loss
        assert new_stop == 148.0

    def test_stop_unchanged_when_not_in_profit(self, mock_engine):
        """Position not in profit → stop unchanged."""
        pos = MockPosition(
            symbol="AAPL",
            entry_price=150.0,
            stop_loss=145.0,
            take_profit=160.0,
            side="long",
        )
        
        desk = ActiveOpenDesk(mock_engine)
        new_stop = desk._compute_tighter_stop(pos, current_price=148.0)
        
        assert new_stop == pos.stop_loss


class TestGetLastDecisions:
    """Test decision history access."""

    @pytest.mark.asyncio
    async def test_get_last_decisions_returns_cached(self, mock_engine):
        """get_last_decisions() returns the cached decisions."""
        desk = ActiveOpenDesk(mock_engine)
        
        decision = DeskDecision(
            symbol="AAPL",
            action=DeskAction.WOULD_TIGHTEN,
            reason="test",
        )
        desk._last_decisions["AAPL"] = decision
        
        result = desk.get_last_decisions()
        
        assert "AAPL" in result
        assert result["AAPL"].action == DeskAction.WOULD_TIGHTEN
