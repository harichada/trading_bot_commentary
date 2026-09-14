"""Tests for the ThemeShock shadow logger.

v-theme-shock-logger-2026-09-14: tests covering:
  - Config flags: ENABLE_THEME_SHOCK_LOGGER default True, env=0 disables
  - Theme config loading from YAML
  - Theme tagger: match headline+summary → theme_ids
  - Shadow action determination for basket/watch_only/HANDS_OFF_DENYLIST
  - Shadow logging: logs would-hard-skip / would-exit / alert-only
  - HANDS_OFF_DENYLIST (MU/HQGE/SPCX) never action beyond alert_only
  - set_news_bus wired (tested via engine integration)
  - Alpaca publish path (mocked)
"""
import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import dataclass

from analysis.theme_shock_logger import (
    ThemeShockLogger,
    ThemeConfigLoader,
    ThemeTagger,
    ThemeConfig,
    ThemeMatch,
    ThemeEvent,
    ShadowAction,
    get_theme_shock_logger,
    reset_theme_shock_logger,
)


@pytest.fixture
def reset_singleton():
    """Reset the theme shock logger singleton before each test."""
    reset_theme_shock_logger()
    yield
    reset_theme_shock_logger()


@pytest.fixture
def sample_scored_item():
    """Create a sample ScoredNewsItem-like object for testing."""
    @dataclass
    class MockScoredNewsItem:
        id: str
        symbol: str
        headline: str
        summary: str
        source: str
        source_tier: int
        url: str
        published_time: datetime
        fetched_at: datetime
        sentiment_score: float
        sentiment_confidence: float
        
        def age_sec(self) -> float:
            return (datetime.now() - self.published_time).total_seconds()
    
    return MockScoredNewsItem(
        id="test123",
        symbol="NVDA",
        headline="Anthropic CEO Dario Amodei warns about AI compute pace",
        summary="The frontier of AI is slowing down as compute becomes scarce.",
        source="Reuters",
        source_tier=1,
        url="https://reuters.com/test",
        published_time=datetime.now() - timedelta(minutes=30),
        fetched_at=datetime.now(),
        sentiment_score=-0.5,
        sentiment_confidence=0.5,
    )


class TestConfigFlags:
    """Test ENABLE_THEME_SHOCK_LOGGER config flag."""

    def test_default_enabled(self, monkeypatch, reset_singleton):
        """Default: ENABLE_THEME_SHOCK_LOGGER=True."""
        monkeypatch.delenv("ENABLE_THEME_SHOCK_LOGGER", raising=False)
        from core.config import Config
        assert Config().ENABLE_THEME_SHOCK_LOGGER is True

    def test_env_1_enables(self, monkeypatch, reset_singleton):
        """env ENABLE_THEME_SHOCK_LOGGER=1 → enabled."""
        monkeypatch.setenv("ENABLE_THEME_SHOCK_LOGGER", "1")
        from core.config import Config
        assert Config().ENABLE_THEME_SHOCK_LOGGER is True

    def test_env_0_disables_safe_off_path(self, monkeypatch, reset_singleton):
        """SAFE OFF-PATH: env ENABLE_THEME_SHOCK_LOGGER=0 → disabled."""
        monkeypatch.setenv("ENABLE_THEME_SHOCK_LOGGER", "0")
        from core.config import Config
        assert Config().ENABLE_THEME_SHOCK_LOGGER is False

    def test_env_false_disables(self, monkeypatch, reset_singleton):
        """env ENABLE_THEME_SHOCK_LOGGER=false → disabled."""
        monkeypatch.setenv("ENABLE_THEME_SHOCK_LOGGER", "false")
        from core.config import Config
        assert Config().ENABLE_THEME_SHOCK_LOGGER is False


class TestThemeConfigLoader:
    """Test theme config loading from YAML."""

    def test_loads_themes_from_data_themes(self, reset_singleton):
        """Should load themes from data/themes/ directory."""
        loader = ThemeConfigLoader()
        assert len(loader.themes) > 0
        
        theme_ids = [t.theme_id for t in loader.themes]
        assert "ai_compute" in theme_ids
        assert "ai_mega_cap" in theme_ids
        assert "memory_hbm" in theme_ids
        assert "fed_risk_off" in theme_ids
        assert "earnings_chip" in theme_ids

    def test_theme_structure(self, reset_singleton):
        """Each theme should have required fields."""
        loader = ThemeConfigLoader()
        for theme in loader.themes:
            assert theme.theme_id
            assert isinstance(theme.match, list)
            assert isinstance(theme.basket, list)
            assert isinstance(theme.watch_only, list)
            assert isinstance(theme.actions_shadow, list)

    def test_ai_compute_theme_content(self, reset_singleton):
        """ai_compute theme should have expected content."""
        loader = ThemeConfigLoader()
        ai_compute = next(t for t in loader.themes if t.theme_id == "ai_compute")
        
        assert "anthropic" in ai_compute.match
        assert "amodei" in ai_compute.match
        assert "NVDA" in ai_compute.basket
        assert "AMD" in ai_compute.basket
        assert "MU" in ai_compute.watch_only
        assert ShadowAction.HARD_SKIP_ENTRIES in ai_compute.actions_shadow
        assert ShadowAction.THESIS_EXIT in ai_compute.actions_shadow


class TestThemeTagger:
    """Test theme tagging from headlines."""

    def test_matches_anthropic_ai_compute(self, reset_singleton):
        """'anthropic' keyword should match ai_compute theme."""
        loader = ThemeConfigLoader()
        tagger = ThemeTagger(loader)
        
        matches = tagger.tag(
            headline="Anthropic CEO warns about frontier AI pace",
            summary="",
        )
        
        assert len(matches) >= 1
        theme_ids = [m.theme_id for m in matches]
        assert "ai_compute" in theme_ids

    def test_matches_fed_risk_off(self, reset_singleton):
        """'fomc' keyword should match fed_risk_off theme."""
        loader = ThemeConfigLoader()
        tagger = ThemeTagger(loader)
        
        matches = tagger.tag(
            headline="FOMC decision: Fed holds rates steady",
            summary="",
        )
        
        assert len(matches) >= 1
        theme_ids = [m.theme_id for m in matches]
        assert "fed_risk_off" in theme_ids

    def test_case_insensitive_matching(self, reset_singleton):
        """Matching should be case-insensitive."""
        loader = ThemeConfigLoader()
        tagger = ThemeTagger(loader)
        
        matches_lower = tagger.tag("anthropic slows down", "")
        matches_upper = tagger.tag("ANTHROPIC SLOWS DOWN", "")
        matches_mixed = tagger.tag("AnThRoPiC SlOwS DoWn", "")
        
        assert len(matches_lower) > 0
        assert len(matches_upper) > 0
        assert len(matches_mixed) > 0

    def test_no_match_unrelated_headline(self, reset_singleton):
        """Unrelated headline should not match any theme."""
        loader = ThemeConfigLoader()
        tagger = ThemeTagger(loader)
        
        matches = tagger.tag(
            headline="Apple announces new iPhone 16 Pro",
            summary="New features include improved camera.",
        )
        
        assert len(matches) == 0


class TestShadowActionDetermination:
    """Test shadow action determination based on symbol type."""

    def test_hands_off_denylist_gets_alert_only(self, reset_singleton):
        """HANDS_OFF_DENYLIST symbols should only get alert_only action."""
        logger = ThemeShockLogger()
        
        match = ThemeMatch(
            theme_id="ai_compute",
            matched_text="anthropic",
            symbols_tradable=["NVDA", "AMD"],
            symbols_watch_only=["MU"],
            actions_shadow=[
                ShadowAction.HARD_SKIP_ENTRIES,
                ShadowAction.THESIS_EXIT,
            ],
        )
        
        action_mu = logger._determine_action_for_symbol("MU", match, is_managed_open=False)
        action_hqge = logger._determine_action_for_symbol("HQGE", match, is_managed_open=False)
        action_spcx = logger._determine_action_for_symbol("SPCX", match, is_managed_open=False)
        
        assert action_mu == ShadowAction.ALERT_ONLY
        assert action_hqge == ShadowAction.ALERT_ONLY
        assert action_spcx == ShadowAction.ALERT_ONLY

    def test_watch_only_gets_alert_only(self, reset_singleton):
        """watch_only symbols should only get alert_only action."""
        logger = ThemeShockLogger()
        
        match = ThemeMatch(
            theme_id="ai_compute",
            matched_text="anthropic",
            symbols_tradable=["NVDA", "AMD"],
            symbols_watch_only=["MU"],
            actions_shadow=[
                ShadowAction.HARD_SKIP_ENTRIES,
                ShadowAction.THESIS_EXIT,
            ],
        )
        
        action = logger._determine_action_for_symbol("MU", match, is_managed_open=False)
        assert action == ShadowAction.ALERT_ONLY

    def test_tradable_entry_gets_hard_skip(self, reset_singleton):
        """Tradable basket symbol on entry should get hard_skip_entries."""
        logger = ThemeShockLogger()
        
        match = ThemeMatch(
            theme_id="ai_compute",
            matched_text="anthropic",
            symbols_tradable=["NVDA", "AMD"],
            symbols_watch_only=["MU"],
            actions_shadow=[
                ShadowAction.HARD_SKIP_ENTRIES,
                ShadowAction.THESIS_EXIT,
            ],
        )
        
        action = logger._determine_action_for_symbol("NVDA", match, is_managed_open=False)
        assert action == ShadowAction.HARD_SKIP_ENTRIES

    def test_tradable_open_gets_thesis_exit(self, reset_singleton):
        """Tradable basket symbol on open position should get thesis_exit."""
        logger = ThemeShockLogger()
        
        match = ThemeMatch(
            theme_id="ai_compute",
            matched_text="anthropic",
            symbols_tradable=["NVDA", "AMD"],
            symbols_watch_only=["MU"],
            actions_shadow=[
                ShadowAction.HARD_SKIP_ENTRIES,
                ShadowAction.THESIS_EXIT,
            ],
        )
        
        action = logger._determine_action_for_symbol("NVDA", match, is_managed_open=True)
        assert action == ShadowAction.THESIS_EXIT


class TestThemeEventLogging:
    """Test theme event logging."""

    def test_log_theme_event_creates_event(self, reset_singleton, sample_scored_item):
        """log_theme_event should create and return ThemeEvent."""
        logger = ThemeShockLogger()
        
        match = ThemeMatch(
            theme_id="ai_compute",
            matched_text="anthropic",
            symbols_tradable=["NVDA"],
            symbols_watch_only=["MU"],
            actions_shadow=[ShadowAction.HARD_SKIP_ENTRIES],
        )
        
        event = logger.log_theme_event(
            item=sample_scored_item,
            match=match,
            action=ShadowAction.HARD_SKIP_ENTRIES,
            symbol="NVDA",
        )
        
        assert event is not None
        assert event.theme_id == "ai_compute"
        assert event.action_shadow == ShadowAction.HARD_SKIP_ENTRIES
        assert "NVDA" in event.symbols_tradable
        assert event.news_age_sec > 0

    def test_event_to_dict_serialization(self, reset_singleton, sample_scored_item):
        """ThemeEvent.to_dict() should produce valid dict."""
        logger = ThemeShockLogger()
        
        match = ThemeMatch(
            theme_id="ai_compute",
            matched_text="anthropic",
            symbols_tradable=["NVDA"],
            symbols_watch_only=["MU"],
            actions_shadow=[ShadowAction.HARD_SKIP_ENTRIES],
        )
        
        event = logger.log_theme_event(
            item=sample_scored_item,
            match=match,
            action=ShadowAction.HARD_SKIP_ENTRIES,
            symbol="NVDA",
        )
        
        d = event.to_dict()
        
        assert d["theme_id"] == "ai_compute"
        assert d["action_shadow"] == "hard_skip_entries"
        assert d["symbols_tradable"] == ["NVDA"]
        assert d["symbols_watch_only"] == ["MU"]
        assert d["news_age_sec"] is not None
        assert "event_id" in d

    def test_deduplication_within_window(self, reset_singleton, sample_scored_item):
        """Duplicate events within window should be deduped."""
        logger = ThemeShockLogger()
        logger._dedupe_window_sec = 60.0
        
        match = ThemeMatch(
            theme_id="ai_compute",
            matched_text="anthropic",
            symbols_tradable=["NVDA"],
            symbols_watch_only=[],
            actions_shadow=[ShadowAction.HARD_SKIP_ENTRIES],
        )
        
        event1 = logger.log_theme_event(
            item=sample_scored_item,
            match=match,
            action=ShadowAction.HARD_SKIP_ENTRIES,
            symbol="NVDA",
        )
        event2 = logger.log_theme_event(
            item=sample_scored_item,
            match=match,
            action=ShadowAction.HARD_SKIP_ENTRIES,
            symbol="NVDA",
        )
        
        assert event1 is not None
        assert event2 is None


class TestProcessNewsItemForEntry:
    """Test entry path processing."""

    def test_returns_event_for_matching_entry(self, reset_singleton, sample_scored_item):
        """Should return ThemeEvent when entry symbol matches theme basket."""
        logger = ThemeShockLogger()
        
        event = logger.process_news_item_for_entry(
            item=sample_scored_item,
            candidate_symbol="NVDA",
        )
        
        assert event is not None
        assert event.action_shadow == ShadowAction.HARD_SKIP_ENTRIES
        assert "entry_candidate" in event.managed_flags

    def test_returns_none_for_non_matching_symbol(self, reset_singleton, sample_scored_item):
        """Should return None when symbol is not in theme basket."""
        logger = ThemeShockLogger()
        
        event = logger.process_news_item_for_entry(
            item=sample_scored_item,
            candidate_symbol="AAPL",
        )
        
        assert event is None


class TestProcessNewsItemForOpenPosition:
    """Test open position processing."""

    def test_returns_event_for_managed_open(self, reset_singleton, sample_scored_item):
        """Should return ThemeEvent for managed open position in basket."""
        logger = ThemeShockLogger()
        
        event = logger.process_news_item_for_open_position(
            item=sample_scored_item,
            position_symbol="NVDA",
            position_side="long",
            is_managed_by_bot=True,
        )
        
        assert event is not None
        assert event.action_shadow == ShadowAction.THESIS_EXIT
        assert event.managed_flags.get("open_position") is True

    def test_returns_none_for_non_managed(self, reset_singleton, sample_scored_item):
        """Should return None for non-managed positions."""
        logger = ThemeShockLogger()
        
        event = logger.process_news_item_for_open_position(
            item=sample_scored_item,
            position_symbol="NVDA",
            position_side="long",
            is_managed_by_bot=False,
        )
        
        assert event is None


class TestFlagOffNoOp:
    """Test that flag off = no-op behavior."""

    def test_disabled_returns_no_matches(self, monkeypatch, reset_singleton):
        """When disabled, tag_news_item returns empty list."""
        monkeypatch.setenv("ENABLE_THEME_SHOCK_LOGGER", "0")
        reset_theme_shock_logger()
        
        logger = get_theme_shock_logger()
        
        @dataclass
        class MockItem:
            headline: str = "Anthropic CEO warns"
            summary: str = ""
        
        matches = logger.tag_news_item(MockItem())
        assert matches == []

    def test_disabled_returns_none_for_entry(self, monkeypatch, reset_singleton, sample_scored_item):
        """When disabled, process_news_item_for_entry returns None."""
        monkeypatch.setenv("ENABLE_THEME_SHOCK_LOGGER", "0")
        reset_theme_shock_logger()
        
        logger = get_theme_shock_logger()
        
        event = logger.process_news_item_for_entry(
            item=sample_scored_item,
            candidate_symbol="NVDA",
        )
        
        assert event is None


class TestHandsOffDenylist:
    """Test that HANDS_OFF_DENYLIST symbols never get action beyond alert_only."""

    def test_mu_never_hard_skip(self, reset_singleton, sample_scored_item):
        """MU should never get hard_skip_entries action."""
        logger = ThemeShockLogger()
        
        event = logger.process_news_item_for_entry(
            item=sample_scored_item,
            candidate_symbol="MU",
        )
        
        if event:
            assert event.action_shadow == ShadowAction.ALERT_ONLY

    def test_hqge_never_thesis_exit(self, reset_singleton, sample_scored_item):
        """HQGE should never get thesis_exit action."""
        sample_scored_item.symbol = "HQGE"
        logger = ThemeShockLogger()
        
        event = logger.process_news_item_for_open_position(
            item=sample_scored_item,
            position_symbol="HQGE",
            position_side="long",
            is_managed_by_bot=True,
        )
        
        if event:
            assert event.action_shadow == ShadowAction.ALERT_ONLY

    def test_spcx_never_size_down(self, reset_singleton, sample_scored_item):
        """SPCX should never get size_down_open action."""
        logger = ThemeShockLogger()
        
        match = ThemeMatch(
            theme_id="test",
            matched_text="test",
            symbols_tradable=["SPCX"],
            symbols_watch_only=[],
            actions_shadow=[ShadowAction.SIZE_DOWN_OPEN],
        )
        
        action = logger._determine_action_for_symbol("SPCX", match, is_managed_open=True)
        assert action == ShadowAction.ALERT_ONLY


class TestGetThemesForSymbol:
    """Test getting themes that affect a symbol."""

    def test_nvda_in_multiple_themes(self, reset_singleton):
        """NVDA should be in ai_compute and earnings_chip themes."""
        logger = ThemeShockLogger()
        
        themes = logger.get_themes_for_symbol("NVDA")
        theme_ids = [t.theme_id for t in themes]
        
        assert "ai_compute" in theme_ids
        assert "earnings_chip" in theme_ids

    def test_mu_in_watch_only(self, reset_singleton):
        """MU should be in watch_only for ai_compute theme."""
        logger = ThemeShockLogger()
        
        themes = logger.get_themes_for_symbol("MU")
        
        ai_compute = next((t for t in themes if t.theme_id == "ai_compute"), None)
        assert ai_compute is not None
        assert "MU" in ai_compute.watch_only


class TestAlpacaNewsBusPublisher:
    """Test Alpaca news publisher integration."""

    @pytest.mark.asyncio
    async def test_publisher_disabled_without_keys(self, monkeypatch):
        """Publisher should be disabled without API keys."""
        monkeypatch.delenv("ALPACA_API_KEY", raising=False)
        monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
        
        from core.loops.news_loop import AlpacaNewsBusPublisher
        from core.news_bus import NewsBus
        
        mock_engine = MagicMock()
        mock_engine.is_running = False
        
        publisher = AlpacaNewsBusPublisher(mock_engine, bus=NewsBus())
        
        assert publisher.is_enabled() is False

    @pytest.mark.asyncio
    async def test_publisher_converts_articles(self, monkeypatch):
        """Publisher should convert Alpaca articles to ScoredNewsItem."""
        monkeypatch.setenv("ALPACA_API_KEY", "test_key")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "test_secret")
        
        from core.loops.news_loop import AlpacaNewsBusPublisher
        from core.news_bus import NewsBus
        
        mock_engine = MagicMock()
        mock_engine.is_running = False
        
        publisher = AlpacaNewsBusPublisher(mock_engine, bus=NewsBus())
        
        articles = [
            {
                "id": "123",
                "headline": "Test headline",
                "summary": "Test summary",
                "symbols": ["NVDA", "AMD"],
                "created_at": "2026-09-14T12:00:00Z",
                "url": "https://example.com/test",
            }
        ]
        
        items = publisher._convert_to_scored_items(articles)
        
        assert len(items) == 2
        assert items[0].symbol == "NVDA"
        assert items[1].symbol == "AMD"
        assert items[0].source == "Alpaca News"
        assert items[0].source_tier == 1


class TestSetNewsBusWired:
    """Test that set_news_bus is properly wired in engine."""

    def test_free_news_strategy_has_set_news_bus(self):
        """FreeNewsSignalStrategy should have set_news_bus method."""
        from strategies.news_strategy import FreeNewsSignalStrategy
        
        assert hasattr(FreeNewsSignalStrategy, 'set_news_bus')
        
        mock_commentary = MagicMock()
        strategy = FreeNewsSignalStrategy(mock_commentary)
        
        mock_bus = MagicMock()
        strategy.set_news_bus(mock_bus)
        
        assert strategy._news_bus is mock_bus


class TestForwardRetStubs:
    """Test forward return stubs."""

    def test_forward_returns_are_none_with_todo(self, reset_singleton, sample_scored_item):
        """Forward returns should be None (stub) until implemented."""
        logger = ThemeShockLogger()
        
        match = ThemeMatch(
            theme_id="ai_compute",
            matched_text="anthropic",
            symbols_tradable=["NVDA"],
            symbols_watch_only=[],
            actions_shadow=[ShadowAction.HARD_SKIP_ENTRIES],
        )
        
        event = logger.log_theme_event(
            item=sample_scored_item,
            match=match,
            action=ShadowAction.HARD_SKIP_ENTRIES,
            symbol="NVDA",
        )
        
        assert event.forward_ret_1m is None
        assert event.forward_ret_5m is None
        assert event.forward_ret_15m is None
        assert event.forward_ret_60m is None
        assert event.half_move_before_ingest is None
