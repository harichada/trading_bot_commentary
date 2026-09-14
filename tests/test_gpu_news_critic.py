"""Tests for the GPU News Critic.

v-gpu-news-critic-2026-09-14: tests covering:
  - Config flags: ENABLE_GPU_NEWS_CRITIC default True, env=0 disables
  - FakeCritic deterministic behavior for CI
  - Crude Oil contamination: headline unrelated + summary contaminated → WOULD_SUPPRESS_HARD_SKIP
  - Anthropic AI compute: kept (not suppressed)
  - Summary-only keyword match with low theme_prob → WOULD_SUPPRESS_HARD_SKIP
  - Flags off → no emits / safe off-path
  - HANDS_OFF_DENYLIST (MU/HQGE/SPCX) behavior

Model download: These tests use FakeCritic by default to avoid HuggingFace
downloads in CI. Use pytest -m "gpu" to run GPU model tests (requires model).
"""
import asyncio
import pytest
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

from analysis.gpu_news_critic import (
    GPUNewsCritic,
    GPUNewsCriticModel,
    FakeCritic,
    CriticCard,
    CriticAction,
    KeywordMatch,
    Stance,
    get_gpu_news_critic,
    reset_gpu_news_critic,
)


@pytest.fixture
def reset_singleton():
    """Reset the GPU news critic singleton before each test."""
    reset_gpu_news_critic()
    yield
    reset_gpu_news_critic()


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
            now = datetime.now(timezone.utc)
            pub = self.published_time
            if pub.tzinfo is None:
                pub = pub.replace(tzinfo=timezone.utc)
            return max(0.0, (now - pub).total_seconds())

    return MockScoredNewsItem(
        id="test123",
        symbol="NVDA",
        headline="Anthropic CEO Dario Amodei warns about AI compute pace",
        summary="The frontier of AI is slowing down as compute becomes scarce.",
        source="Reuters",
        source_tier=1,
        url="https://reuters.com/test",
        published_time=datetime.now(timezone.utc) - timedelta(minutes=30),
        fetched_at=datetime.now(timezone.utc),
        sentiment_score=-0.5,
        sentiment_confidence=0.5,
    )


class TestConfigFlags:
    """Test ENABLE_GPU_NEWS_CRITIC config flag."""

    def test_default_enabled(self, monkeypatch, reset_singleton):
        """Default: ENABLE_GPU_NEWS_CRITIC=True."""
        monkeypatch.delenv("ENABLE_GPU_NEWS_CRITIC", raising=False)
        from core.config import Config
        assert Config().ENABLE_GPU_NEWS_CRITIC is True

    def test_env_1_enables(self, monkeypatch, reset_singleton):
        """env ENABLE_GPU_NEWS_CRITIC=1 → enabled."""
        monkeypatch.setenv("ENABLE_GPU_NEWS_CRITIC", "1")
        from core.config import Config
        assert Config().ENABLE_GPU_NEWS_CRITIC is True

    def test_env_0_disables_safe_off_path(self, monkeypatch, reset_singleton):
        """SAFE OFF-PATH: env ENABLE_GPU_NEWS_CRITIC=0 → disabled."""
        monkeypatch.setenv("ENABLE_GPU_NEWS_CRITIC", "0")
        from core.config import Config
        assert Config().ENABLE_GPU_NEWS_CRITIC is False

    def test_env_false_disables(self, monkeypatch, reset_singleton):
        """env ENABLE_GPU_NEWS_CRITIC=false → disabled."""
        monkeypatch.setenv("ENABLE_GPU_NEWS_CRITIC", "false")
        from core.config import Config
        assert Config().ENABLE_GPU_NEWS_CRITIC is False


class TestGPUNewsCriticShadowFlag:
    """Test GPU_NEWS_CRITIC_SHADOW config flag."""

    def test_default_shadow_enabled(self, monkeypatch, reset_singleton):
        """Default: GPU_NEWS_CRITIC_SHADOW=True (shadow mode)."""
        monkeypatch.delenv("GPU_NEWS_CRITIC_SHADOW", raising=False)
        from core.config import Config
        assert Config().GPU_NEWS_CRITIC_SHADOW is True

    def test_env_0_disables_shadow(self, monkeypatch, reset_singleton):
        """env GPU_NEWS_CRITIC_SHADOW=0 → live mode (future)."""
        monkeypatch.setenv("GPU_NEWS_CRITIC_SHADOW", "0")
        from core.config import Config
        assert Config().GPU_NEWS_CRITIC_SHADOW is False


class TestThemeHardSkipRequireGPU:
    """Test THEME_HARD_SKIP_REQUIRE_GPU config flag."""

    def test_default_false(self, monkeypatch, reset_singleton):
        """Default: THEME_HARD_SKIP_REQUIRE_GPU=False (Stage A)."""
        monkeypatch.delenv("THEME_HARD_SKIP_REQUIRE_GPU", raising=False)
        from core.config import Config
        assert Config().THEME_HARD_SKIP_REQUIRE_GPU is False

    def test_env_1_enables(self, monkeypatch, reset_singleton):
        """env THEME_HARD_SKIP_REQUIRE_GPU=1 → enabled (future)."""
        monkeypatch.setenv("THEME_HARD_SKIP_REQUIRE_GPU", "1")
        from core.config import Config
        assert Config().THEME_HARD_SKIP_REQUIRE_GPU is True


class TestFakeCritic:
    """Test FakeCritic deterministic behavior for CI."""

    def test_anthropic_ai_compute_detected(self, reset_singleton):
        """Anthropic headline should detect ai_compute theme."""
        critic = FakeCritic()

        theme_probs, relevance, stance, contam, conf, rationale = critic.score(
            headline="Anthropic CEO Dario Amodei warns about AI frontier pace",
            summary="",
        )

        assert theme_probs["ai_compute"] >= 0.5
        assert "NVDA" in relevance
        assert contam < 0.3
        assert "ai_compute" in rationale.lower()

    def test_crude_oil_contamination_detected(self, reset_singleton):
        """Crude oil in summary should be flagged as contaminated."""
        critic = FakeCritic()

        theme_probs, relevance, stance, contam, conf, rationale = critic.score(
            headline="Tech sector rally continues",
            summary="Markets rose as crude oil prices fell. WTI crude dropped 5%.",
        )

        assert contam >= 0.6
        assert stance == Stance.IRRELEVANT
        assert "contamination" in rationale.lower()

    def test_crude_oil_headline_lower_contamination(self, reset_singleton):
        """Crude oil in headline has lower contamination (not summary-only)."""
        critic = FakeCritic()

        theme_probs, relevance, stance, contam, conf, rationale = critic.score(
            headline="Crude oil prices surge, energy sector rallies",
            summary="Oil markets saw significant gains today.",
        )

        assert contam < 0.6  # Not as high as summary-only contamination
        assert stance == Stance.IRRELEVANT

    def test_unrelated_news_no_theme(self, reset_singleton):
        """Unrelated headline should not match any specific theme."""
        critic = FakeCritic()

        theme_probs, relevance, stance, contam, conf, rationale = critic.score(
            headline="Apple announces new iPhone features",
            summary="New camera improvements and battery life.",
        )

        assert theme_probs["none"] > theme_probs.get("ai_compute", 0)
        assert len(relevance) == 0
        assert stance == Stance.IRRELEVANT
        assert contam == 0.0

    def test_summary_keyword_match_low_theme_prob(self, reset_singleton):
        """Summary-only keyword match with low theme_prob → high contamination."""
        critic = FakeCritic()

        keyword_match = KeywordMatch(
            theme_id="ai_compute",
            matched_text="anthropic",
            matched_field="summary",
        )

        # Headline is unrelated to AI, but summary mentions anthropic
        theme_probs, relevance, stance, contam, conf, rationale = critic.score(
            headline="Weather forecast for the weekend",
            summary="In other news, Anthropic released a new model.",
            keyword_match=keyword_match,
        )

        # The unrelated headline means low ai_compute theme_prob
        # Combined with summary-only match → should trigger high contamination
        assert contam >= 0.6
        assert "summary-only keyword match" in rationale.lower()


class TestCriticCard:
    """Test CriticCard structure and serialization."""

    def test_card_to_dict(self, reset_singleton):
        """CriticCard.to_dict() should produce valid dict."""
        card = CriticCard(
            event_id="test123",
            headline="Test headline",
            summary="Test summary",
            theme_probs={"ai_compute": 0.6, "none": 0.4},
            relevance_by_symbol={"NVDA": 0.8},
            stance=Stance.BEARISH,
            contamination_risk=0.2,
            confidence=0.7,
            rationale_short="ai_compute theme",
            keyword_match=KeywordMatch("ai_compute", "anthropic", "headline"),
            action=CriticAction.PASS,
            infer_ms=50.0,
        )

        d = card.to_dict()

        assert d["event_id"] == "test123"
        assert d["theme_probs"]["ai_compute"] == 0.6
        assert d["relevance_by_symbol"]["NVDA"] == 0.8
        assert d["stance"] == "bearish"
        assert d["contamination_risk"] == 0.2
        assert d["action"] == "pass"
        assert d["keyword_match"]["theme_id"] == "ai_compute"
        assert d["keyword_match"]["matched_field"] == "headline"


class TestGPUNewsCritic:
    """Test GPUNewsCritic with FakeCritic."""

    def test_score_anthropic_kept(self, reset_singleton, sample_scored_item):
        """Anthropic AI compute headline should be kept (PASS action)."""
        critic = GPUNewsCritic(use_fake=True)

        card = critic.score_news_item(sample_scored_item)

        assert card.action == CriticAction.PASS
        assert card.theme_probs["ai_compute"] >= 0.5
        assert card.contamination_risk < 0.5
        assert "NVDA" in card.relevance_by_symbol

    def test_score_crude_oil_contaminated_suppressed(self, reset_singleton):
        """Crude Oil: headline unrelated + summary contaminated → WOULD_SUPPRESS_HARD_SKIP."""
        @dataclass
        class CrudeOilItem:
            id: str = "crude123"
            symbol: str = "NVDA"
            headline: str = "NVIDIA announces new GPU architecture"
            summary: str = "In commodities, crude oil prices surged as OPEC cut production. WTI crude rose 3%."
            source: str = "Reuters"
            source_tier: int = 1
            url: str = "https://test.com"
            published_time: datetime = None
            fetched_at: datetime = None
            sentiment_score: float = 0.5
            sentiment_confidence: float = 0.5

            def __post_init__(self):
                if self.published_time is None:
                    self.published_time = datetime.now(timezone.utc) - timedelta(minutes=10)
                if self.fetched_at is None:
                    self.fetched_at = datetime.now(timezone.utc)

            def age_sec(self) -> float:
                return 600.0

        critic = GPUNewsCritic(use_fake=True)
        
        # Add keyword match as if ThemeShock matched "ai_compute" on summary
        keyword_match = KeywordMatch(
            theme_id="ai_compute",
            matched_text="nvidia",
            matched_field="summary",  # Matched in summary, not headline
        )
        
        item = CrudeOilItem()
        card = critic.score_news_item(item, keyword_match=keyword_match)

        # Should detect contamination from crude oil in summary
        assert card.contamination_risk >= 0.6, f"Expected contam >= 0.6, got {card.contamination_risk}"
        assert card.action == CriticAction.WOULD_SUPPRESS_HARD_SKIP

    def test_score_summary_only_keyword_low_theme_suppressed(self, reset_singleton):
        """Summary-only keyword match with low theme_prob → WOULD_SUPPRESS_HARD_SKIP."""
        @dataclass
        class MixedItem:
            id: str = "mixed123"
            symbol: str = "AAPL"
            headline: str = "Apple announces new iPhone features for holiday season"
            summary: str = "The company also mentioned its AI partnership with Anthropic."
            source: str = "Reuters"
            source_tier: int = 1
            url: str = "https://test.com"
            published_time: datetime = None
            fetched_at: datetime = None
            sentiment_score: float = 0.5
            sentiment_confidence: float = 0.5

            def __post_init__(self):
                if self.published_time is None:
                    self.published_time = datetime.now(timezone.utc) - timedelta(minutes=10)
                if self.fetched_at is None:
                    self.fetched_at = datetime.now(timezone.utc)

            def age_sec(self) -> float:
                return 600.0

        critic = GPUNewsCritic(use_fake=True)

        # ThemeShock matched "anthropic" but only in summary
        keyword_match = KeywordMatch(
            theme_id="ai_compute",
            matched_text="anthropic",
            matched_field="summary",
        )

        item = MixedItem()
        card = critic.score_news_item(item, keyword_match=keyword_match)

        # Headline is about Apple/iPhone, not AI compute
        # Summary-only match with low theme_prob should trigger suppression
        assert card.contamination_risk >= 0.6
        assert card.action == CriticAction.WOULD_SUPPRESS_HARD_SKIP


class TestFlagOff:
    """Test that flag off = no-op behavior."""

    def test_disabled_returns_skip_card(self, monkeypatch, reset_singleton, sample_scored_item):
        """When disabled, score_news_item returns card with skip_reason."""
        monkeypatch.setenv("ENABLE_GPU_NEWS_CRITIC", "0")
        reset_gpu_news_critic()

        critic = get_gpu_news_critic(use_fake=True)

        card = critic.score_news_item(sample_scored_item)

        assert card.skip_reason == "disabled"
        assert card.action == CriticAction.PASS
        assert len(card.theme_probs) == 0

    def test_disabled_process_from_publish_returns_none(
        self, monkeypatch, reset_singleton, sample_scored_item
    ):
        """When disabled, process_news_item_from_publish returns None."""
        monkeypatch.setenv("ENABLE_GPU_NEWS_CRITIC", "0")
        reset_gpu_news_critic()

        critic = get_gpu_news_critic(use_fake=True)

        result = critic.process_news_item_from_publish(sample_scored_item)

        assert result is None


class TestCriticActionDetermination:
    """Test action determination logic."""

    def test_high_contamination_suppresses(self, reset_singleton):
        """contamination_risk >= 0.6 → WOULD_SUPPRESS_HARD_SKIP."""
        critic = GPUNewsCritic(use_fake=True)

        action = critic._determine_action(
            theme_probs={"ai_compute": 0.6, "none": 0.4},
            contamination_risk=0.7,
            confidence=0.5,
            keyword_match=None,
        )

        assert action == CriticAction.WOULD_SUPPRESS_HARD_SKIP

    def test_summary_match_low_prob_suppresses(self, reset_singleton):
        """matched_field=summary AND theme_prob < 0.25 → WOULD_SUPPRESS_HARD_SKIP."""
        critic = GPUNewsCritic(use_fake=True)

        keyword_match = KeywordMatch(
            theme_id="ai_compute",
            matched_text="anthropic",
            matched_field="summary",
        )

        action = critic._determine_action(
            theme_probs={"ai_compute": 0.15, "none": 0.85},  # Low ai_compute prob
            contamination_risk=0.3,  # Below threshold
            confidence=0.5,
            keyword_match=keyword_match,
        )

        assert action == CriticAction.WOULD_SUPPRESS_HARD_SKIP

    def test_headline_match_low_prob_passes(self, reset_singleton):
        """matched_field=headline (not summary) → passes even with low prob."""
        critic = GPUNewsCritic(use_fake=True)

        keyword_match = KeywordMatch(
            theme_id="ai_compute",
            matched_text="anthropic",
            matched_field="headline",
        )

        action = critic._determine_action(
            theme_probs={"ai_compute": 0.15, "none": 0.85},  # Low prob
            contamination_risk=0.3,
            confidence=0.5,
            keyword_match=keyword_match,
        )

        assert action == CriticAction.PASS  # Not suppressed for headline match

    def test_clean_news_passes(self, reset_singleton):
        """Clean news with high theme_prob passes."""
        critic = GPUNewsCritic(use_fake=True)

        keyword_match = KeywordMatch(
            theme_id="ai_compute",
            matched_text="anthropic",
            matched_field="headline",
        )

        action = critic._determine_action(
            theme_probs={"ai_compute": 0.7, "none": 0.3},
            contamination_risk=0.1,
            confidence=0.8,
            keyword_match=keyword_match,
        )

        assert action == CriticAction.PASS


class TestProcessFromPublish:
    """Test process_news_item_from_publish integration."""

    def test_returns_card_when_enabled(self, reset_singleton, sample_scored_item):
        """Should return CriticCard when enabled."""
        critic = GPUNewsCritic(use_fake=True)

        card = critic.process_news_item_from_publish(sample_scored_item)

        assert card is not None
        assert isinstance(card, CriticCard)
        assert card.event_id is not None

    def test_passes_keyword_match(self, reset_singleton, sample_scored_item):
        """Should use keyword_match from ThemeShock."""
        critic = GPUNewsCritic(use_fake=True)

        keyword_match = KeywordMatch(
            theme_id="ai_compute",
            matched_text="anthropic",
            matched_field="headline",
        )

        card = critic.process_news_item_from_publish(
            sample_scored_item,
            keyword_match=keyword_match,
        )

        assert card.keyword_match is not None
        assert card.keyword_match.theme_id == "ai_compute"


class TestNewsBusIntegration:
    """Test NewsBus publish path integration."""

    @pytest.mark.asyncio
    async def test_publish_triggers_gpu_critic(self, monkeypatch, reset_singleton):
        """NewsBus.publish should invoke GPU critic via combined callback."""
        from core.news_bus import NewsBus, ScoredNewsItem, NewsImpactLevel

        critic_calls = []

        def mock_critic_callback(item, keyword_match=None):
            critic_calls.append((item.headline, keyword_match))
            return CriticCard(
                event_id="test",
                headline=item.headline,
                summary="",
            )

        # Create bus with mock combined callback
        bus = NewsBus()

        now = datetime.now(timezone.utc)
        item = ScoredNewsItem(
            id="test_pub_1",
            symbol="NVDA",
            headline="Anthropic CEO warns about AI compute",
            summary="Test summary",
            source="Alpaca News",
            source_tier=1,
            url="https://test.com/1",
            published_time=now - timedelta(minutes=5),
            fetched_at=now,
            sentiment_score=-0.5,
            sentiment_confidence=0.5,
            impact=NewsImpactLevel.MEDIUM,
        )

        # Manually wire the mock
        original_on_publish = bus._on_publish
        bus._on_publish = lambda x: mock_critic_callback(x)

        new_count = await bus.publish([item])

        assert new_count == 1
        assert len(critic_calls) == 1
        assert critic_calls[0][0] == "Anthropic CEO warns about AI compute"


class TestInferenceLatency:
    """Test inference latency tracking."""

    def test_infer_ms_tracked(self, reset_singleton, sample_scored_item):
        """CriticCard should track inference time in ms."""
        critic = GPUNewsCritic(use_fake=True)

        card = critic.score_news_item(sample_scored_item)

        assert card.infer_ms >= 0
        # FakeCritic is very fast
        assert card.infer_ms < 100


class TestSingleton:
    """Test singleton behavior."""

    def test_singleton_reuse(self, reset_singleton):
        """get_gpu_news_critic returns same instance."""
        critic1 = get_gpu_news_critic(use_fake=True)
        critic2 = get_gpu_news_critic(use_fake=True)

        assert critic1 is critic2

    def test_reset_clears_singleton(self, reset_singleton):
        """reset_gpu_news_critic clears the singleton."""
        critic1 = get_gpu_news_critic(use_fake=True)
        reset_gpu_news_critic()
        critic2 = get_gpu_news_critic(use_fake=True)

        assert critic1 is not critic2


class TestStanceClassification:
    """Test stance classification."""

    def test_bearish_stance_on_slowdown(self, reset_singleton):
        """AI slowdown news should have bearish stance."""
        critic = FakeCritic()

        _, _, stance, _, _, _ = critic.score(
            headline="Anthropic warns of major AI slowdown ahead",
            summary="",
        )

        assert stance == Stance.BEARISH

    def test_mixed_stance_on_neutral(self, reset_singleton):
        """Neutral AI news should have mixed stance."""
        critic = FakeCritic()

        _, _, stance, _, _, _ = critic.score(
            headline="Anthropic CEO discusses frontier AI developments",
            summary="",
        )

        assert stance in (Stance.MIXED, Stance.BEARISH)


class TestCrudeOilClassRegression:
    """Regression tests for the Crude Oil contamination case (RCA).
    
    The Crude Oil event (event_id 1a1e82cd9500) was the key RCA case:
    headline mentioned something unrelated, but summary contained
    crude oil content that contaminated the theme match.
    
    These tests ensure we correctly detect and suppress such cases.
    """

    def test_crude_oil_class_unrelated_headline_contaminated_summary(
        self, reset_singleton
    ):
        """Core RCA case: unrelated headline + oil-contaminated summary."""
        @dataclass
        class CrudeOilCase:
            id: str = "1a1e82cd9500"
            symbol: str = "NVDA"
            headline: str = "Tech stocks rally on positive earnings outlook"
            summary: str = (
                "Technology shares gained as companies reported strong results. "
                "Meanwhile, crude oil prices fell 2% as OPEC+ discussed production. "
                "WTI crude settled at $75 per barrel."
            )
            source: str = "Reuters"
            source_tier: int = 1
            url: str = "https://test.com"
            published_time: datetime = None
            fetched_at: datetime = None
            sentiment_score: float = 0.5
            sentiment_confidence: float = 0.5

            def __post_init__(self):
                if self.published_time is None:
                    self.published_time = datetime.now(timezone.utc) - timedelta(minutes=10)
                if self.fetched_at is None:
                    self.fetched_at = datetime.now(timezone.utc)

            def age_sec(self) -> float:
                return 600.0

        critic = GPUNewsCritic(use_fake=True)
        item = CrudeOilCase()

        card = critic.score_news_item(item)

        # Must detect crude oil contamination
        assert card.contamination_risk >= 0.6, (
            f"Crude Oil case MUST trigger high contamination risk, "
            f"got {card.contamination_risk}"
        )
        assert card.action == CriticAction.WOULD_SUPPRESS_HARD_SKIP

    def test_anthropic_class_not_suppressed(self, reset_singleton):
        """Anthropic-class true AI theme MUST NOT be suppressed."""
        @dataclass
        class AnthropicCase:
            id: str = "anthropic_true"
            symbol: str = "NVDA"
            headline: str = "Anthropic CEO Dario Amodei warns about frontier AI pace"
            summary: str = (
                "The AI safety company's CEO expressed concerns about the "
                "rapid advancement of frontier AI models and compute scaling."
            )
            source: str = "Reuters"
            source_tier: int = 1
            url: str = "https://test.com"
            published_time: datetime = None
            fetched_at: datetime = None
            sentiment_score: float = -0.5
            sentiment_confidence: float = 0.5

            def __post_init__(self):
                if self.published_time is None:
                    self.published_time = datetime.now(timezone.utc) - timedelta(minutes=10)
                if self.fetched_at is None:
                    self.fetched_at = datetime.now(timezone.utc)

            def age_sec(self) -> float:
                return 600.0

        critic = GPUNewsCritic(use_fake=True)
        item = AnthropicCase()

        # Add keyword match as if ThemeShock matched "anthropic" in headline
        keyword_match = KeywordMatch(
            theme_id="ai_compute",
            matched_text="anthropic",
            matched_field="headline",
        )

        card = critic.score_news_item(item, keyword_match=keyword_match)

        # Must NOT suppress true AI compute theme
        assert card.action == CriticAction.PASS, (
            f"Anthropic-class AI theme MUST NOT be suppressed, got {card.action}"
        )
        assert card.theme_probs["ai_compute"] >= 0.5
        assert card.contamination_risk < 0.6
