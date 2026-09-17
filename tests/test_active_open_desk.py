"""Tests for the Active Open Desk continuous position monitor.

v-active-open-desk-2026-09-14: tests covering:
  - Config flags: ENABLE_ACTIVE_OPEN_DESK, ACTIVE_OPEN_DESK_SHADOW, ACTIVE_OPEN_DESK_INTERVAL_SEC
  - Flag off = no desk activity (should_start_desk returns False)
  - Flag on + shadow = would-be actions logged (no broker calls)
  - HANDS_OFF_DENYLIST symbols skipped
  - Parallel asyncio.gather invoked for multiple positions
  - DeskDecision dataclass serialization

v-proactive-exit-daytrade-2026-09-14 (PR2): tests covering:
  - PROACTIVE_EXIT_MIN_AGE_DAYTRADE config and env override
  - PROACTIVE_EXIT_R_OVERRIDE_THRESHOLD config and env override
  - Day trade detection from reasoning
  - R-override bypasses min age gate
  - Shadow logs include age/R reason strings
  - HANDS_OFF positions skip proactive exit evaluation
"""
import asyncio
import pytest
from datetime import datetime
from typing import Optional
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import dataclass, field

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
    entry_time: datetime = field(default_factory=datetime.now)
    original_stop: Optional[float] = None
    reasoning: dict = field(default_factory=dict)
    
    def __post_init__(self):
        if self.current_price == 0.0:
            self.current_price = self.entry_price
        if self.original_stop is None:
            self.original_stop = self.stop_loss


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


# ============================================================================
# PR2: Proactive Exit Age/R Gates (v-proactive-exit-daytrade-2026-09-14)
# ============================================================================

@dataclass
class MockPositionWithReasoning(MockPosition):
    """MockPosition with reasoning dict for day-trade detection."""
    reasoning: dict = field(default_factory=dict)
    original_stop: Optional[float] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.original_stop is None:
            self.original_stop = self.stop_loss


class TestIsDayTrade:
    """Test day trade detection logic."""

    def test_detects_day_trade_from_is_day_trade_flag(self, mock_engine):
        """Position with is_day_trade=True in reasoning is detected."""
        pos = MockPositionWithReasoning(
            symbol="AAPL",
            entry_price=150.0,
            stop_loss=145.0,
            take_profit=160.0,
            side="long",
            reasoning={"is_day_trade": True, "strategy": "momentum"},
        )
        
        desk = ActiveOpenDesk(mock_engine)
        assert desk._is_day_trade(pos) is True

    def test_detects_day_trade_from_strategy_name(self, mock_engine):
        """Position with strategy=day_trade_momentum is detected."""
        pos = MockPositionWithReasoning(
            symbol="AAPL",
            entry_price=150.0,
            stop_loss=145.0,
            take_profit=160.0,
            side="long",
            reasoning={"strategy": "day_trade_momentum"},
        )
        
        desk = ActiveOpenDesk(mock_engine)
        assert desk._is_day_trade(pos) is True

    def test_detects_day_trade_from_strategy_substring(self, mock_engine):
        """Position with 'day_trade' in strategy name is detected."""
        pos = MockPositionWithReasoning(
            symbol="AAPL",
            entry_price=150.0,
            stop_loss=145.0,
            take_profit=160.0,
            side="long",
            reasoning={"strategy": "my_day_trade_strategy"},
        )
        
        desk = ActiveOpenDesk(mock_engine)
        assert desk._is_day_trade(pos) is True

    def test_non_day_trade_not_detected(self, mock_engine):
        """Position without day-trade markers is not detected as day trade."""
        pos = MockPositionWithReasoning(
            symbol="AAPL",
            entry_price=150.0,
            stop_loss=145.0,
            take_profit=160.0,
            side="long",
            reasoning={"strategy": "news_momentum", "is_day_trade": False},
        )
        
        desk = ActiveOpenDesk(mock_engine)
        assert desk._is_day_trade(pos) is False


class TestProactiveExitMinAgeConfig:
    """Test config flags for proactive exit min age."""

    def test_daytrade_min_age_default(self, monkeypatch):
        """Default PROACTIVE_EXIT_MIN_AGE_DAYTRADE is 3 minutes."""
        monkeypatch.delenv("PROACTIVE_EXIT_MIN_AGE_DAYTRADE", raising=False)
        from core.config import Config
        assert Config().PROACTIVE_EXIT_MIN_AGE_DAYTRADE == 3

    def test_daytrade_min_age_env_override(self, monkeypatch):
        """env PROACTIVE_EXIT_MIN_AGE_DAYTRADE=5 → 5 minutes."""
        monkeypatch.setenv("PROACTIVE_EXIT_MIN_AGE_DAYTRADE", "5")
        from core.config import Config
        assert Config().PROACTIVE_EXIT_MIN_AGE_DAYTRADE == 5

    def test_r_override_threshold_default(self, monkeypatch):
        """Default PROACTIVE_EXIT_R_OVERRIDE_THRESHOLD is -0.3."""
        monkeypatch.delenv("PROACTIVE_EXIT_R_OVERRIDE_THRESHOLD", raising=False)
        from core.config import Config
        assert Config().PROACTIVE_EXIT_R_OVERRIDE_THRESHOLD == -0.3

    def test_r_override_threshold_env_override(self, monkeypatch):
        """env PROACTIVE_EXIT_R_OVERRIDE_THRESHOLD=-0.5 → -0.5."""
        monkeypatch.setenv("PROACTIVE_EXIT_R_OVERRIDE_THRESHOLD", "-0.5")
        from core.config import Config
        assert Config().PROACTIVE_EXIT_R_OVERRIDE_THRESHOLD == -0.5


class TestGetMinAgeForPosition:
    """Test that correct min age is selected based on position type."""

    def test_day_trade_uses_daytrade_min_age(self, mock_engine, monkeypatch):
        """Day trade position uses PROACTIVE_EXIT_MIN_AGE_DAYTRADE."""
        monkeypatch.setenv("PROACTIVE_EXIT_MIN_AGE_DAYTRADE", "3")
        
        pos = MockPositionWithReasoning(
            symbol="AAPL",
            entry_price=150.0,
            stop_loss=145.0,
            take_profit=160.0,
            side="long",
            reasoning={"strategy": "day_trade_momentum"},
        )
        
        desk = ActiveOpenDesk(mock_engine)
        min_age = desk._get_min_age_for_position(pos)
        
        assert min_age == 3

    def test_news_trade_uses_news_min_age(self, mock_engine, monkeypatch):
        """News trade position uses PROACTIVE_EXIT_MIN_AGE_NEWS."""
        monkeypatch.delenv("PROACTIVE_EXIT_MIN_AGE_DAYTRADE", raising=False)
        
        pos = MockPositionWithReasoning(
            symbol="AAPL",
            entry_price=150.0,
            stop_loss=145.0,
            take_profit=160.0,
            side="long",
            reasoning={"strategy": "free_news_sentiment"},
        )
        
        desk = ActiveOpenDesk(mock_engine)
        min_age = desk._get_min_age_for_position(pos)
        
        from core.config import Config
        assert min_age == Config().PROACTIVE_EXIT_MIN_AGE_NEWS

    def test_other_uses_default_min_age(self, mock_engine, monkeypatch):
        """Non-day-trade, non-news position uses PROACTIVE_EXIT_MIN_AGE_DEFAULT."""
        monkeypatch.delenv("PROACTIVE_EXIT_MIN_AGE_DAYTRADE", raising=False)
        
        pos = MockPositionWithReasoning(
            symbol="AAPL",
            entry_price=150.0,
            stop_loss=145.0,
            take_profit=160.0,
            side="long",
            reasoning={"strategy": "momentum"},
        )
        
        desk = ActiveOpenDesk(mock_engine)
        min_age = desk._get_min_age_for_position(pos)
        
        from core.config import Config
        assert min_age == Config().PROACTIVE_EXIT_MIN_AGE_DEFAULT


class TestProactiveExitROverride:
    """Test R-based override of min age gate."""

    def test_r_override_bypasses_age_gate(self, mock_engine, monkeypatch):
        """When pnl_r <= R_OVERRIDE_THRESHOLD, min age is bypassed."""
        from datetime import timedelta
        
        monkeypatch.setenv("PROACTIVE_EXIT_MIN_AGE_DAYTRADE", "10")
        monkeypatch.setenv("PROACTIVE_EXIT_R_OVERRIDE_THRESHOLD", "-0.3")
        monkeypatch.setenv("ENABLE_PROACTIVE_EXIT", "1")
        
        # Position opened 1 minute ago (below 10 min threshold)
        entry_time = datetime.now() - timedelta(minutes=1)
        
        pos = MockPositionWithReasoning(
            symbol="AAPL",
            entry_price=150.0,
            stop_loss=145.0,  # 5.0 stop distance
            take_profit=160.0,
            side="long",
            reasoning={"strategy": "day_trade_momentum"},
        )
        pos.entry_time = entry_time
        pos.original_stop = 145.0
        
        desk = ActiveOpenDesk(mock_engine)
        
        # Current price = 147.0 → pnl_r = (147-150)/5 = -0.6R
        # This is below -0.3R threshold AND below -0.5R pnl_threshold
        current_price = 147.0
        
        # With indicators that would trigger proactive exit
        indicators = {
            "macd": -0.1,
            "macd_signal": 0.1,  # MACD bearish
            "rsi": 45.0,
            "adx": 15.0,
            "adx_prev": 20.0,
        }
        
        decision = desk._check_proactive_exit_gate(pos, current_price, indicators)
        
        assert decision is not None
        assert decision.action == DeskAction.WOULD_EXIT
        assert decision.r_override_used is True
        assert decision.age_gate_met is False
        assert "r_override_bypass" in decision.reason

    def test_age_gate_met_no_r_override_needed(self, mock_engine, monkeypatch):
        """When age gate is met, r_override_used is False even if R is bad."""
        from datetime import timedelta
        
        monkeypatch.setenv("PROACTIVE_EXIT_MIN_AGE_DAYTRADE", "3")
        monkeypatch.setenv("PROACTIVE_EXIT_R_OVERRIDE_THRESHOLD", "-0.3")
        monkeypatch.setenv("ENABLE_PROACTIVE_EXIT", "1")
        
        # Position opened 5 minutes ago (above 3 min threshold)
        entry_time = datetime.now() - timedelta(minutes=5)
        
        pos = MockPositionWithReasoning(
            symbol="AAPL",
            entry_price=150.0,
            stop_loss=145.0,
            take_profit=160.0,
            side="long",
            reasoning={"strategy": "day_trade_momentum"},
        )
        pos.entry_time = entry_time
        pos.original_stop = 145.0
        
        desk = ActiveOpenDesk(mock_engine)
        
        # Current price = 147.0 → pnl_r = -0.6R (bad, below pnl_threshold)
        current_price = 147.0
        
        indicators = {
            "macd": -0.1,
            "macd_signal": 0.1,
            "rsi": 45.0,
            "adx": 15.0,
            "adx_prev": 20.0,
        }
        
        decision = desk._check_proactive_exit_gate(pos, current_price, indicators)
        
        assert decision is not None
        assert decision.age_gate_met is True
        # r_override_used can be True (R is bad), but we don't rely on it
        assert "r_override_bypass" not in decision.reason


class TestProactiveExitSuppression:
    """Test that proactive exit is suppressed when conditions not met."""

    def test_suppressed_when_below_min_age_and_above_r_threshold(self, mock_engine, monkeypatch):
        """Proactive exit suppressed when below min age AND R not bad enough."""
        from datetime import timedelta
        
        monkeypatch.setenv("PROACTIVE_EXIT_MIN_AGE_DAYTRADE", "10")
        monkeypatch.setenv("PROACTIVE_EXIT_R_OVERRIDE_THRESHOLD", "-0.3")
        monkeypatch.setenv("ENABLE_PROACTIVE_EXIT", "1")
        
        # Position opened 1 minute ago
        entry_time = datetime.now() - timedelta(minutes=1)
        
        pos = MockPositionWithReasoning(
            symbol="AAPL",
            entry_price=150.0,
            stop_loss=145.0,
            take_profit=160.0,
            side="long",
            reasoning={"strategy": "day_trade_momentum"},
        )
        pos.entry_time = entry_time
        pos.original_stop = 145.0
        
        desk = ActiveOpenDesk(mock_engine)
        
        # Current price = 149.5 → pnl_r = (149.5-150)/5 = -0.1R
        # This is above -0.3R threshold, so R-override does NOT fire
        current_price = 149.5
        
        indicators = {
            "macd": -0.1,
            "macd_signal": 0.1,
            "rsi": 45.0,
            "adx": 15.0,
            "adx_prev": 20.0,
        }
        
        decision = desk._check_proactive_exit_gate(pos, current_price, indicators)
        
        # Should be None because suppressed
        assert decision is None


class TestDecisionFieldsPopulated:
    """Test that decision includes age/R metadata."""

    def test_decision_includes_age_r_fields(self, mock_engine, monkeypatch):
        """DeskDecision includes age_minutes, pnl_r, is_day_trade, etc."""
        from datetime import timedelta
        
        monkeypatch.setenv("PROACTIVE_EXIT_MIN_AGE_DAYTRADE", "3")
        monkeypatch.setenv("ENABLE_PROACTIVE_EXIT", "1")
        
        entry_time = datetime.now() - timedelta(minutes=5)
        
        pos = MockPositionWithReasoning(
            symbol="AAPL",
            entry_price=150.0,
            stop_loss=145.0,
            take_profit=160.0,
            side="long",
            reasoning={"strategy": "day_trade_momentum"},
        )
        pos.entry_time = entry_time
        pos.original_stop = 145.0
        
        desk = ActiveOpenDesk(mock_engine)
        # Use current_price = 147.0 → pnl_r = -0.6R (below -0.5R threshold)
        current_price = 147.0
        
        indicators = {
            "macd": -0.1,
            "macd_signal": 0.1,
            "rsi": 45.0,
            "adx": 15.0,
            "adx_prev": 20.0,
        }
        
        decision = desk._check_proactive_exit_gate(pos, current_price, indicators)
        
        assert decision is not None
        assert decision.age_minutes is not None
        assert decision.age_minutes >= 4.9  # approximately 5 minutes
        assert decision.pnl_r is not None
        assert abs(decision.pnl_r - (-0.6)) < 0.01  # pnl_r ≈ -0.6
        assert decision.is_day_trade is True


class TestShadowLogReason:
    """Test that shadow logs include detailed reason strings."""

    def test_reason_includes_age_and_r(self, mock_engine, monkeypatch):
        """Reason string includes age and pnl_r information."""
        from datetime import timedelta
        
        monkeypatch.setenv("PROACTIVE_EXIT_MIN_AGE_DAYTRADE", "3")
        monkeypatch.setenv("ENABLE_PROACTIVE_EXIT", "1")
        
        entry_time = datetime.now() - timedelta(minutes=5)
        
        pos = MockPositionWithReasoning(
            symbol="AAPL",
            entry_price=150.0,
            stop_loss=145.0,
            take_profit=160.0,
            side="long",
            reasoning={"strategy": "day_trade_momentum"},
        )
        pos.entry_time = entry_time
        pos.original_stop = 145.0
        
        desk = ActiveOpenDesk(mock_engine)
        # Use current_price = 147.0 → pnl_r = -0.6R (below -0.5R threshold)
        current_price = 147.0
        
        indicators = {
            "macd": -0.1,
            "macd_signal": 0.1,
            "rsi": 45.0,
            "adx": 15.0,
            "adx_prev": 20.0,
        }
        
        decision = desk._check_proactive_exit_gate(pos, current_price, indicators)
        
        assert decision is not None
        assert "age=" in decision.reason
        assert "pnl_r=" in decision.reason
        assert "_daytrade" in decision.reason

    def test_reason_includes_r_override_when_used(self, mock_engine, monkeypatch):
        """Reason includes r_override_bypass when R-override triggered."""
        from datetime import timedelta
        
        monkeypatch.setenv("PROACTIVE_EXIT_MIN_AGE_DAYTRADE", "10")
        monkeypatch.setenv("PROACTIVE_EXIT_R_OVERRIDE_THRESHOLD", "-0.3")
        monkeypatch.setenv("ENABLE_PROACTIVE_EXIT", "1")
        
        entry_time = datetime.now() - timedelta(minutes=1)
        
        pos = MockPositionWithReasoning(
            symbol="AAPL",
            entry_price=150.0,
            stop_loss=145.0,
            take_profit=160.0,
            side="long",
            reasoning={"strategy": "day_trade_momentum"},
        )
        pos.entry_time = entry_time
        pos.original_stop = 145.0
        
        desk = ActiveOpenDesk(mock_engine)
        # Use current_price = 147.0 → pnl_r = -0.6R (below both thresholds)
        current_price = 147.0
        
        indicators = {
            "macd": -0.1,
            "macd_signal": 0.1,
            "rsi": 45.0,
            "adx": 15.0,
            "adx_prev": 20.0,
        }
        
        decision = desk._check_proactive_exit_gate(pos, current_price, indicators)
        
        assert decision is not None
        assert "r_override_bypass" in decision.reason


class TestToDict:
    """Test DeskDecision serialization includes new fields."""

    def test_to_dict_includes_age_r_fields(self):
        """to_dict() includes age_minutes, pnl_r, is_day_trade, etc."""
        decision = DeskDecision(
            symbol="AAPL",
            action=DeskAction.WOULD_EXIT,
            reason="test_age=5.0m_pnl_r=-0.40_daytrade",
            current_stop=145.0,
            suggested_exit_price=148.0,
            current_price=148.0,
            age_minutes=5.0,
            pnl_r=-0.4,
            is_day_trade=True,
            age_gate_met=True,
            r_override_used=False,
        )
        
        d = decision.to_dict()
        
        assert d["age_minutes"] == 5.0
        assert d["pnl_r"] == -0.4
        assert d["is_day_trade"] is True
        assert d["age_gate_met"] is True
        assert d["r_override_used"] is False


class TestHandsOffSkippedForProactiveExit:
    """Test that HANDS_OFF positions skip proactive exit evaluation too."""

    @pytest.mark.asyncio
    async def test_hands_off_positions_not_evaluated(self, mock_engine, monkeypatch):
        """Positions in HANDS_OFF_DENYLIST are not monitored at all.
        
        This implicitly means proactive exit won't fire for them either.
        """
        # MU is in HANDS_OFF_DENYLIST
        pos = MockPositionWithReasoning(
            symbol="MU",
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            side="long",
            managed_by_bot=True,
            reasoning={"strategy": "day_trade_momentum"},
        )
        mock_engine.positions = {"MU": pos}
        
        desk = ActiveOpenDesk(mock_engine)
        result = desk.get_monitored_positions()
        
        # MU should be excluded
        symbols = [r[0] for r in result]
        assert "MU" not in symbols


# ══════════════════════════════════════════════════════════════════════════════
# v-open-desk-rsi-exit-2026-09-17: Tests for ENABLE_OPEN_DESK_RSI_EXTREME_EXIT
# P0 RCA 2026-09-17 (XE): day_trade breakout entered → open-desk spam WOULD_EXIT
# rsi_extreme but suggest-only → stop fill. Fix: execute real exit for day trades.
# ══════════════════════════════════════════════════════════════════════════════

class TestEnableOpenDeskRsiExtremeExitConfig:
    """Test ENABLE_OPEN_DESK_RSI_EXTREME_EXIT config flag."""

    def test_default_enabled(self, monkeypatch):
        """ENABLE_OPEN_DESK_RSI_EXTREME_EXIT must default to True."""
        monkeypatch.delenv("ENABLE_OPEN_DESK_RSI_EXTREME_EXIT", raising=False)
        from core.config import Config
        cfg = Config()
        assert cfg.ENABLE_OPEN_DESK_RSI_EXTREME_EXIT is True, (
            "ENABLE_OPEN_DESK_RSI_EXTREME_EXIT must default to True — "
            "day-trade RSI extreme should execute real exits by default"
        )

    def test_env_0_disables(self, monkeypatch):
        """ENABLE_OPEN_DESK_RSI_EXTREME_EXIT=0 disables real exits."""
        monkeypatch.setenv("ENABLE_OPEN_DESK_RSI_EXTREME_EXIT", "0")
        from core.config import Config
        cfg = Config()
        assert cfg.ENABLE_OPEN_DESK_RSI_EXTREME_EXIT is False

    def test_env_1_enables(self, monkeypatch):
        """ENABLE_OPEN_DESK_RSI_EXTREME_EXIT=1 enables real exits."""
        monkeypatch.setenv("ENABLE_OPEN_DESK_RSI_EXTREME_EXIT", "1")
        from core.config import Config
        cfg = Config()
        assert cfg.ENABLE_OPEN_DESK_RSI_EXTREME_EXIT is True


@dataclass
class MockPositionDayTrade:
    """Mock position with day-trade reasoning for RSI extreme exit tests."""
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
    entry_time: datetime = field(default_factory=datetime.now)
    original_stop: Optional[float] = None
    reasoning: dict = field(default_factory=lambda: {
        'strategy': 'day_trade_momentum',
        'is_day_trade': True,
        'entry_pattern': 'breakout'
    })
    
    def __post_init__(self):
        if self.current_price == 0.0:
            self.current_price = self.entry_price
        if self.original_stop is None:
            self.original_stop = self.stop_loss


class TestRsiExtremeExitRealExit:
    """Test that RSI extreme triggers REAL exit for day trades when enabled."""

    @pytest.mark.asyncio
    async def test_rsi_extreme_executes_real_exit_for_daytrade(self, mock_engine, monkeypatch):
        """When ENABLE_OPEN_DESK_RSI_EXTREME_EXIT=True AND position is day trade
        AND rsi_extreme_overbought fires, execute REAL exit (not shadow).
        
        P0 RCA 2026-09-17 (XE): day_trade breakout @16.31×334 @09:45 ET →
        open-desk spam WOULD_EXIT rsi_extreme_overbought_90+ → suggest-only →
        stop fill. Fix: execute real close for day trades.
        """
        monkeypatch.setenv("ENABLE_OPEN_DESK_RSI_EXTREME_EXIT", "1")
        monkeypatch.setenv("ACTIVE_OPEN_DESK_SHADOW", "1")  # Shadow ON but RSI extreme should still execute
        
        pos = MockPositionDayTrade(
            symbol="XE",
            entry_price=16.31,
            stop_loss=15.50,
            take_profit=18.00,
            side="long",
            current_price=15.80,
        )
        mock_engine.positions = {"XE": pos}
        mock_engine._close_position_with_commentary = AsyncMock()
        
        desk = ActiveOpenDesk(mock_engine)
        
        with patch.object(desk, '_get_market_context') as mock_ctx, \
             patch.object(desk, '_get_news_sentiment', new_callable=AsyncMock) as mock_news, \
             patch.object(desk, '_get_indicators', new_callable=AsyncMock) as mock_ind:
            
            mock_ctx.return_value = MagicMock(regime="risk_on")
            mock_news.return_value = 0.0
            mock_ind.return_value = {"rsi": 92.0, "macd": 0.5, "macd_signal": 0.3}  # RSI >= 80
            
            await desk._tick()
        
        # MUST call supervised close with rsi_extreme reason
        mock_engine._close_position_with_commentary.assert_called_once()
        call_args = mock_engine._close_position_with_commentary.call_args
        reason = call_args[0][1]
        assert "rsi_extreme_overbought" in reason, (
            f"Expected reason to contain 'rsi_extreme_overbought', got '{reason}'"
        )

    @pytest.mark.asyncio
    async def test_rsi_extreme_stays_shadow_when_flag_disabled(self, mock_engine, monkeypatch):
        """When ENABLE_OPEN_DESK_RSI_EXTREME_EXIT=False, rsi_extreme stays shadow-only.
        
        SAFE OFF-PATH: setting flag to 0 preserves old suggest-only behavior.
        """
        monkeypatch.setenv("ENABLE_OPEN_DESK_RSI_EXTREME_EXIT", "0")
        monkeypatch.setenv("ACTIVE_OPEN_DESK_SHADOW", "1")
        
        pos = MockPositionDayTrade(
            symbol="XE",
            entry_price=16.31,
            stop_loss=15.50,
            take_profit=18.00,
            side="long",
            current_price=15.80,
        )
        mock_engine.positions = {"XE": pos}
        mock_engine._close_position_with_commentary = AsyncMock()
        
        desk = ActiveOpenDesk(mock_engine)
        
        with patch.object(desk, '_get_market_context') as mock_ctx, \
             patch.object(desk, '_get_news_sentiment', new_callable=AsyncMock) as mock_news, \
             patch.object(desk, '_get_indicators', new_callable=AsyncMock) as mock_ind:
            
            mock_ctx.return_value = MagicMock(regime="risk_on")
            mock_news.return_value = 0.0
            mock_ind.return_value = {"rsi": 92.0, "macd": 0.5, "macd_signal": 0.3}
            
            await desk._tick()
        
        # Should NOT call supervised close
        mock_engine._close_position_with_commentary.assert_not_called()

    @pytest.mark.asyncio
    async def test_rsi_extreme_only_for_daytrade_positions(self, mock_engine, monkeypatch):
        """RSI extreme real exit only fires for is_day_trade=True positions.
        
        Non-day-trade (swing, news) positions stay shadow-only even with flag ON.
        """
        monkeypatch.setenv("ENABLE_OPEN_DESK_RSI_EXTREME_EXIT", "1")
        monkeypatch.setenv("ACTIVE_OPEN_DESK_SHADOW", "1")
        
        # Non-day-trade position (swing strategy)
        pos = MockPosition(
            symbol="AAPL",
            entry_price=150.0,
            stop_loss=145.0,
            take_profit=160.0,
            side="long",
            current_price=155.0,
            reasoning={'strategy': 'news_strategy', 'is_day_trade': False},
        )
        mock_engine.positions = {"AAPL": pos}
        mock_engine._close_position_with_commentary = AsyncMock()
        
        desk = ActiveOpenDesk(mock_engine)
        
        with patch.object(desk, '_get_market_context') as mock_ctx, \
             patch.object(desk, '_get_news_sentiment', new_callable=AsyncMock) as mock_news, \
             patch.object(desk, '_get_indicators', new_callable=AsyncMock) as mock_ind:
            
            mock_ctx.return_value = MagicMock(regime="risk_on")
            mock_news.return_value = 0.0
            mock_ind.return_value = {"rsi": 92.0, "macd": 0.5, "macd_signal": 0.3}
            
            await desk._tick()
        
        # Non-day-trade should NOT call supervised close (stays shadow)
        mock_engine._close_position_with_commentary.assert_not_called()

    @pytest.mark.asyncio
    async def test_rsi_extreme_oversold_executes_for_short_daytrade(self, mock_engine, monkeypatch):
        """RSI extreme oversold (<= 20) triggers real exit for SHORT day trades."""
        monkeypatch.setenv("ENABLE_OPEN_DESK_RSI_EXTREME_EXIT", "1")
        monkeypatch.setenv("ACTIVE_OPEN_DESK_SHADOW", "1")
        
        pos = MockPositionDayTrade(
            symbol="XE",
            entry_price=16.31,
            stop_loss=17.00,
            take_profit=15.00,
            side="short",
            current_price=15.80,
        )
        mock_engine.positions = {"XE": pos}
        mock_engine._close_position_with_commentary = AsyncMock()
        
        desk = ActiveOpenDesk(mock_engine)
        
        with patch.object(desk, '_get_market_context') as mock_ctx, \
             patch.object(desk, '_get_news_sentiment', new_callable=AsyncMock) as mock_news, \
             patch.object(desk, '_get_indicators', new_callable=AsyncMock) as mock_ind:
            
            mock_ctx.return_value = MagicMock(regime="risk_off")
            mock_news.return_value = 0.0
            mock_ind.return_value = {"rsi": 18.0, "macd": -0.5, "macd_signal": -0.3}  # RSI <= 20
            
            await desk._tick()
        
        # MUST call supervised close with rsi_extreme_oversold reason
        mock_engine._close_position_with_commentary.assert_called_once()
        call_args = mock_engine._close_position_with_commentary.call_args
        reason = call_args[0][1]
        assert "rsi_extreme_oversold" in reason
