"""v-gpu-news-critic-2026-09-14: Shadow GPU News Critic.

GPU-accelerated news scoring for theme classification and contamination
detection. Implements Research Stage A (docs/research/2026-09-14-gpu-sense-stage-a.md):
- Scores each ScoredNewsItem with theme probabilities, relevance, stance,
  contamination risk, and confidence
- Shadow-only logging: does NOT execute broker calls or mutate orders
- Logs WOULD_SUPPRESS_HARD_SKIP when contamination detected or low theme_prob

Model choice: sentence-transformers/all-MiniLM-L6-v2
- ~22M parameters, ~90MB VRAM
- p95 inference <100ms on RTX 3090 (well under 800ms target)
- Zero-shot classification via embedding similarity
- Leaves 23GB+ headroom for trading FFN and other models
- Widely used, well-tested, no license restrictions

SAFE OFF-PATH: Set ENABLE_GPU_NEWS_CRITIC=0 to disable entirely.
When disabled, no model loading, no GPU inference, no shadow logs.

Graceful degradation:
- If CUDA unavailable: logs once, falls back to CPU inference
- If model download fails: logs error, returns no-op CriticCard with skip_reason
- If inference fails: catches exception, logs, returns no-op card

Respects HANDS_OFF_DENYLIST (MU, HQGE, SPCX) — no actions for these symbols.
"""
from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from core.news_bus import ScoredNewsItem

logger = logging.getLogger("TradingBot")

_CUDA_WARNING_LOGGED = False
_MODEL_ERROR_LOGGED = False
_DB_PERSIST_ERROR_LOGGED = False


class Stance(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    MIXED = "mixed"
    IRRELEVANT = "irrelevant"


class CriticAction(str, Enum):
    PASS = "pass"
    WOULD_SUPPRESS_HARD_SKIP = "would_suppress_hard_skip"
    ALERT_ONLY = "alert_only"


@dataclass
class KeywordMatch:
    """Keyword match info from ThemeShock tagger."""
    theme_id: str
    matched_text: str
    matched_field: str  # "headline" or "summary"


@dataclass
class CriticCard:
    """Structured scoring card from GPU News Critic.
    
    Schema aligned with Research Stage A (docs/research/2026-09-14-gpu-sense-stage-a.md):
    - theme_probs: probability per theme
    - relevance_by_symbol: relevance score 0-1 per symbol
    - stance: bullish/bearish/mixed/irrelevant
    - contamination_risk: 0-1, multi-story/off-topic contamination
    - confidence: 0-1, model confidence
    - rationale_short: brief explanation
    - keyword_match: from ThemeShock tagger (theme_id, matched_text, matched_field)
    
    Logged to db_logger.log_strategy_decision with extra_data=to_dict() so
    Monday rollup can score precision/FP/recall/latency alongside ThemeShock.
    """
    event_id: str
    headline: str
    summary: str
    theme_probs: Dict[str, float] = field(default_factory=dict)
    relevance_by_symbol: Dict[str, float] = field(default_factory=dict)
    stance: Stance = Stance.IRRELEVANT
    contamination_risk: float = 0.0
    confidence: float = 0.0
    rationale_short: str = ""
    keyword_match: Optional[KeywordMatch] = None
    action: CriticAction = CriticAction.PASS
    infer_ms: float = 0.0
    skip_reason: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    news_age_sec: float = 0.0
    source: str = ""
    symbol: str = ""

    def to_dict(self) -> dict:
        """Serialize for structured logging.
        
        Returns all Stage A fields for Research Monday rollup:
        theme_probs, relevance_by_symbol, stance, contamination_risk,
        confidence, rationale_short, keyword_match (with matched_text+matched_field).
        """
        return {
            "event_id": self.event_id,
            "headline": self.headline[:100] if self.headline else "",
            "summary": self.summary[:200] if self.summary else "",
            "theme_probs": self.theme_probs,
            "relevance_by_symbol": self.relevance_by_symbol,
            "stance": self.stance.value,
            "contamination_risk": self.contamination_risk,
            "confidence": self.confidence,
            "rationale_short": self.rationale_short,
            "keyword_match": (
                {
                    "theme_id": self.keyword_match.theme_id,
                    "matched_text": self.keyword_match.matched_text,
                    "matched_field": self.keyword_match.matched_field,
                }
                if self.keyword_match else None
            ),
            "action": self.action.value,
            "infer_ms": self.infer_ms,
            "skip_reason": self.skip_reason,
            "created_at": self.created_at.isoformat(),
            "news_age_sec": self.news_age_sec,
            "source": self.source,
            "symbol": self.symbol,
        }


THEME_DESCRIPTIONS = {
    "ai_compute": "AI compute infrastructure, GPU demand, AI chip supply, frontier AI models, data center capacity",
    "ai_mega_cap": "Large technology companies AI investments, OpenAI, ChatGPT, Microsoft Azure AI, Google AI, Meta AI",
    "fed_risk_off": "Federal Reserve interest rate decisions, FOMC meetings, inflation, monetary policy, bond yields",
    "earnings_chip": "Semiconductor earnings reports, chip company guidance, foundry results, GPU revenue forecasts",
    "memory_hbm": "Memory chips, HBM high bandwidth memory, DRAM, NAND, memory supply and demand",
    "other": "Other market moving news not fitting specific themes",
    "none": "Not relevant to trading themes, general news, unrelated topics",
}

THEME_BASKET = {
    "ai_compute": ["NVDA", "AMD", "AVGO", "SMCI", "ARM"],
    "ai_mega_cap": ["MSFT", "AMZN", "GOOGL", "META"],
    "fed_risk_off": ["QQQ", "SPY", "IWM"],
    "earnings_chip": ["NVDA", "AMD", "INTC", "TSM", "ASML", "AVGO", "QCOM", "MRVL"],
    "memory_hbm": [],  # watch_only: MU
}

STANCE_DESCRIPTORS = {
    Stance.BULLISH: "positive outlook, growth, opportunity, upgrade, beat expectations",
    Stance.BEARISH: "negative outlook, decline, risk, downgrade, miss expectations, slowdown",
    Stance.MIXED: "uncertain, mixed signals, both positive and negative aspects",
    Stance.IRRELEVANT: "not relevant to market trading decisions",
}


class GPUNewsCriticModel:
    """Lazy-loaded GPU model for news scoring.
    
    Uses sentence-transformers for embedding-based zero-shot classification.
    Loads model on first inference, caches embeddings for theme descriptions.
    
    VRAM usage: ~200MB (model) + ~50MB (embeddings cache) = ~250MB total
    Well under 24GB RTX 3090 with >23GB headroom for trading FFN.
    """

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    DEVICE_PREFERENCE = "cuda:0"

    def __init__(self):
        self._model = None
        self._device = None
        self._theme_embeddings: Optional[Dict[str, Any]] = None
        self._stance_embeddings: Optional[Dict[Stance, Any]] = None
        self._loaded = False
        self._load_error: Optional[str] = None

    def _lazy_load(self) -> bool:
        """Lazy-load model on first use. Returns True if ready."""
        global _CUDA_WARNING_LOGGED, _MODEL_ERROR_LOGGED

        if self._loaded:
            return self._load_error is None

        if self._load_error is not None:
            return False

        try:
            import torch
            from sentence_transformers import SentenceTransformer

            if torch.cuda.is_available():
                self._device = self.DEVICE_PREFERENCE
                logger.info(
                    "gpu_news_critic loading model=%s device=%s",
                    self.MODEL_NAME, self._device,
                )
            else:
                self._device = "cpu"
                if not _CUDA_WARNING_LOGGED:
                    logger.warning(
                        "gpu_news_critic CUDA unavailable, falling back to CPU "
                        "(inference will be slower but functional)"
                    )
                    _CUDA_WARNING_LOGGED = True

            self._model = SentenceTransformer(self.MODEL_NAME, device=self._device)

            self._theme_embeddings = {}
            for theme_id, desc in THEME_DESCRIPTIONS.items():
                self._theme_embeddings[theme_id] = self._model.encode(
                    desc, convert_to_tensor=True, normalize_embeddings=True
                )

            self._stance_embeddings = {}
            for stance, desc in STANCE_DESCRIPTORS.items():
                self._stance_embeddings[stance] = self._model.encode(
                    desc, convert_to_tensor=True, normalize_embeddings=True
                )

            self._loaded = True
            logger.info(
                "gpu_news_critic model loaded: %s on %s, "
                "theme_embeddings=%d, stance_embeddings=%d",
                self.MODEL_NAME, self._device,
                len(self._theme_embeddings), len(self._stance_embeddings),
            )
            return True

        except ImportError as e:
            self._load_error = f"missing dependency: {e}"
            if not _MODEL_ERROR_LOGGED:
                logger.error(
                    "gpu_news_critic model load failed (missing deps): %s "
                    "(install: pip install sentence-transformers torch)",
                    e,
                )
                _MODEL_ERROR_LOGGED = True
            return False

        except Exception as e:
            self._load_error = str(e)
            if not _MODEL_ERROR_LOGGED:
                logger.error(
                    "gpu_news_critic model load failed: %s "
                    "(will return no-op cards with skip_reason)",
                    e,
                )
                _MODEL_ERROR_LOGGED = True
            return False

    def score(
        self,
        headline: str,
        summary: str,
        keyword_match: Optional[KeywordMatch] = None,
    ) -> Tuple[Dict[str, float], Dict[str, float], Stance, float, float, str]:
        """Score a news item.
        
        Returns:
            theme_probs: dict of theme_id -> probability
            relevance_by_symbol: dict of symbol -> relevance 0-1
            stance: Stance enum
            contamination_risk: 0-1
            confidence: 0-1
            rationale_short: brief explanation
        """
        import torch

        text = f"{headline} {summary}".strip()
        if not text:
            return {}, {}, Stance.IRRELEVANT, 0.0, 0.0, "empty_text"

        text_embedding = self._model.encode(
            text, convert_to_tensor=True, normalize_embeddings=True
        )

        theme_probs = {}
        for theme_id, theme_emb in self._theme_embeddings.items():
            similarity = torch.nn.functional.cosine_similarity(
                text_embedding.unsqueeze(0), theme_emb.unsqueeze(0)
            ).item()
            theme_probs[theme_id] = max(0.0, (similarity + 1) / 2)

        total = sum(theme_probs.values())
        if total > 0:
            theme_probs = {k: v / total for k, v in theme_probs.items()}

        stance_scores = {}
        for stance, stance_emb in self._stance_embeddings.items():
            similarity = torch.nn.functional.cosine_similarity(
                text_embedding.unsqueeze(0), stance_emb.unsqueeze(0)
            ).item()
            stance_scores[stance] = similarity

        best_stance = max(stance_scores, key=stance_scores.get)
        stance_confidence = (stance_scores[best_stance] + 1) / 2

        relevance_by_symbol = {}
        top_theme = max(theme_probs, key=theme_probs.get)
        top_theme_prob = theme_probs[top_theme]

        if top_theme in THEME_BASKET:
            for symbol in THEME_BASKET[top_theme]:
                relevance_by_symbol[symbol] = top_theme_prob * 0.8

        contamination_risk = self._compute_contamination_risk(
            headline, summary, theme_probs, keyword_match
        )

        confidence = (top_theme_prob + stance_confidence) / 2
        if contamination_risk > 0.5:
            confidence *= (1 - contamination_risk * 0.5)

        rationale_parts = []
        if top_theme != "none":
            rationale_parts.append(f"theme={top_theme}({top_theme_prob:.2f})")
        rationale_parts.append(f"stance={best_stance.value}")
        if contamination_risk > 0.3:
            rationale_parts.append(f"contam={contamination_risk:.2f}")
        if keyword_match:
            rationale_parts.append(
                f"kw_match={keyword_match.theme_id}:{keyword_match.matched_field}"
            )
        rationale_short = "; ".join(rationale_parts)

        return (
            theme_probs,
            relevance_by_symbol,
            best_stance,
            contamination_risk,
            confidence,
            rationale_short,
        )

    def _compute_contamination_risk(
        self,
        headline: str,
        summary: str,
        theme_probs: Dict[str, float],
        keyword_match: Optional[KeywordMatch],
    ) -> float:
        """Compute contamination risk score.
        
        Contamination indicators:
        1. Keyword match in summary but low headline theme_prob
        2. Multiple unrelated topics detected (high "other" + specific theme)
        3. Summary significantly longer than headline (aggregation signal)
        4. Keyword matched theme disagrees with model's top theme
        """
        risk = 0.0

        if keyword_match and keyword_match.matched_field == "summary":
            headline_lower = headline.lower()
            if keyword_match.matched_text.lower() not in headline_lower:
                matched_theme = keyword_match.theme_id
                if matched_theme in theme_probs:
                    theme_prob = theme_probs[matched_theme]
                    if theme_prob < 0.25:
                        risk += 0.5
                    elif theme_prob < 0.35:
                        risk += 0.3

        other_prob = theme_probs.get("other", 0)
        none_prob = theme_probs.get("none", 0)
        specific_themes = [
            v for k, v in theme_probs.items()
            if k not in ("other", "none") and v > 0.15
        ]
        if other_prob > 0.2 and len(specific_themes) >= 1:
            risk += 0.2

        if len(summary) > len(headline) * 5 and len(summary) > 200:
            risk += 0.15

        if keyword_match and keyword_match.theme_id in theme_probs:
            top_theme = max(
                {k: v for k, v in theme_probs.items() if k not in ("other", "none")},
                key=lambda k: theme_probs[k],
                default=None,
            )
            if top_theme and top_theme != keyword_match.theme_id:
                if theme_probs[top_theme] > theme_probs[keyword_match.theme_id] + 0.15:
                    risk += 0.25

        return min(1.0, risk)


class FakeCritic:
    """Deterministic test double for GPU News Critic.
    
    For CI environments where model download is blocked or slow.
    Provides predictable outputs based on headline/summary content.
    
    Key behavior for contamination detection:
    - AI trigger in headline → ai_compute theme, low contamination
    - AI trigger ONLY in summary (not headline) → low theme_prob, high contamination
    - Crude oil/commodities in summary → high contamination
    """

    CONTAMINATION_TRIGGERS = [
        "crude oil",
        "petroleum",
        "opec",
        "oil prices",
        "wti crude",
        "brent crude",
    ]

    AI_COMPUTE_TRIGGERS = [
        "anthropic",
        "amodei",
        "openai",
        "frontier ai",
        "ai slowdown",
        "gpu demand",
        "data center",
    ]

    def score(
        self,
        headline: str,
        summary: str,
        keyword_match: Optional[KeywordMatch] = None,
    ) -> Tuple[Dict[str, float], Dict[str, float], Stance, float, float, str]:
        """Deterministic scoring for tests."""
        headline_lower = headline.lower()
        summary_lower = summary.lower()
        text = f"{headline} {summary}".lower()

        is_contaminated = any(t in text for t in self.CONTAMINATION_TRIGGERS)
        
        headline_has_ai_trigger = any(t in headline_lower for t in self.AI_COMPUTE_TRIGGERS)
        summary_has_ai_trigger = any(t in summary_lower for t in self.AI_COMPUTE_TRIGGERS)
        is_ai_compute = headline_has_ai_trigger or summary_has_ai_trigger

        summary_has_contam_trigger = any(
            t in summary_lower for t in self.CONTAMINATION_TRIGGERS
        )
        headline_has_contam_trigger = any(
            t in headline_lower for t in self.CONTAMINATION_TRIGGERS
        )

        ai_trigger_only_in_summary = summary_has_ai_trigger and not headline_has_ai_trigger

        if headline_has_ai_trigger and not is_contaminated:
            theme_probs = {
                "ai_compute": 0.65,
                "ai_mega_cap": 0.15,
                "fed_risk_off": 0.05,
                "earnings_chip": 0.05,
                "memory_hbm": 0.02,
                "other": 0.05,
                "none": 0.03,
            }
            relevance = {"NVDA": 0.8, "AMD": 0.7, "AVGO": 0.5}
            stance = Stance.BEARISH if "slowdown" in text else Stance.MIXED
            contamination_risk = 0.1
            confidence = 0.75
            rationale = "ai_compute theme detected, relevant to chip basket"

        elif ai_trigger_only_in_summary and not is_contaminated:
            theme_probs = {
                "ai_compute": 0.15,
                "ai_mega_cap": 0.10,
                "fed_risk_off": 0.05,
                "earnings_chip": 0.05,
                "memory_hbm": 0.02,
                "other": 0.30,
                "none": 0.33,
            }
            relevance = {}
            stance = Stance.IRRELEVANT
            contamination_risk = 0.65
            confidence = 0.35
            rationale = "ai trigger only in summary, headline unrelated; summary-only keyword match with low theme_prob"

        elif is_contaminated:
            contamination_risk = 0.7 if summary_has_contam_trigger and not headline_has_contam_trigger else 0.4
            theme_probs = {
                "ai_compute": 0.15 if is_ai_compute else 0.05,
                "ai_mega_cap": 0.05,
                "fed_risk_off": 0.10,
                "earnings_chip": 0.05,
                "memory_hbm": 0.02,
                "other": 0.40,
                "none": 0.23,
            }
            relevance = {}
            stance = Stance.IRRELEVANT
            confidence = 0.3
            rationale = f"contamination detected ({contamination_risk:.2f}), crude oil unrelated"

        else:
            theme_probs = {
                "ai_compute": 0.08,
                "ai_mega_cap": 0.08,
                "fed_risk_off": 0.08,
                "earnings_chip": 0.08,
                "memory_hbm": 0.05,
                "other": 0.25,
                "none": 0.38,
            }
            relevance = {}
            stance = Stance.IRRELEVANT
            contamination_risk = 0.0
            confidence = 0.5
            rationale = "no specific theme match"

        if keyword_match and keyword_match.matched_field == "summary":
            if keyword_match.theme_id in theme_probs:
                if theme_probs[keyword_match.theme_id] < 0.25:
                    contamination_risk = max(contamination_risk, 0.65)
                    if "summary-only keyword match" not in rationale:
                        rationale += "; summary-only keyword match with low theme_prob"

        return (
            theme_probs,
            relevance,
            stance,
            contamination_risk,
            confidence,
            rationale,
        )


class GPUNewsCritic:
    """Shadow GPU News Critic for theme classification and contamination detection.
    
    Scores each ScoredNewsItem and emits structured shadow logs.
    Does NOT execute broker calls or mutate orders.
    
    Config flags:
    - ENABLE_GPU_NEWS_CRITIC: master switch (default True)
    - GPU_NEWS_CRITIC_SHADOW: shadow mode only (default True)
    - THEME_HARD_SKIP_REQUIRE_GPU: require GPU score for hard_skip (default False)
    
    SAFE OFF-PATH: Set ENABLE_GPU_NEWS_CRITIC=0 to disable entirely.
    """

    def __init__(self, engine=None, use_fake: bool = False):
        self._engine = engine
        self._use_fake = use_fake
        self._model: Optional[GPUNewsCriticModel] = None
        self._fake_critic: Optional[FakeCritic] = None

        if use_fake:
            self._fake_critic = FakeCritic()

    def is_enabled(self) -> bool:
        """Check if GPU news critic is enabled."""
        from core.config import Config
        return Config().ENABLE_GPU_NEWS_CRITIC

    def is_shadow_mode(self) -> bool:
        """Check if running in shadow mode (no live actions)."""
        from core.config import Config
        return Config().GPU_NEWS_CRITIC_SHADOW

    def get_hands_off_denylist(self) -> frozenset:
        """Get the HANDS_OFF_DENYLIST from config."""
        from core.config import Config
        return Config().HANDS_OFF_DENYLIST

    def _get_model(self) -> Optional[GPUNewsCriticModel]:
        """Get or create the model instance."""
        if self._use_fake:
            return None

        if self._model is None:
            self._model = GPUNewsCriticModel()

        return self._model

    def _make_event_id(self, headline: str) -> str:
        """Generate a stable event ID."""
        content = f"gpu_critic:{headline[:50]}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def score_news_item(
        self,
        item: "ScoredNewsItem",
        keyword_match: Optional[KeywordMatch] = None,
    ) -> CriticCard:
        """Score a news item and return a CriticCard.
        
        Args:
            item: The ScoredNewsItem from NewsBus.
            keyword_match: Optional keyword match from ThemeShock tagger.
        
        Returns:
            CriticCard with scoring results and action.
        """
        if not self.is_enabled():
            return CriticCard(
                event_id=self._make_event_id(item.headline),
                headline=item.headline,
                summary=getattr(item, "summary", ""),
                skip_reason="disabled",
            )

        headline = item.headline
        summary = getattr(item, "summary", "")
        event_id = self._make_event_id(headline)

        start_ms = time.time() * 1000

        if self._use_fake:
            theme_probs, relevance, stance, contam_risk, confidence, rationale = (
                self._fake_critic.score(headline, summary, keyword_match)
            )
        else:
            model = self._get_model()
            if model is None or not model._lazy_load():
                return CriticCard(
                    event_id=event_id,
                    headline=headline,
                    summary=summary,
                    skip_reason=model._load_error if model else "model_unavailable",
                )

            try:
                theme_probs, relevance, stance, contam_risk, confidence, rationale = (
                    model.score(headline, summary, keyword_match)
                )
            except Exception as e:
                logger.debug("gpu_news_critic score error: %s", e)
                return CriticCard(
                    event_id=event_id,
                    headline=headline,
                    summary=summary,
                    skip_reason=f"inference_error: {e}",
                )

        infer_ms = time.time() * 1000 - start_ms

        action = self._determine_action(
            theme_probs, contam_risk, confidence, keyword_match
        )

        news_age_sec = item.age_sec() if hasattr(item, "age_sec") else 0.0
        source = getattr(item, "source", "")
        symbol = getattr(item, "symbol", "UNKNOWN")

        card = CriticCard(
            event_id=event_id,
            headline=headline,
            summary=summary,
            theme_probs=theme_probs,
            relevance_by_symbol=relevance,
            stance=stance,
            contamination_risk=contam_risk,
            confidence=confidence,
            rationale_short=rationale,
            keyword_match=keyword_match,
            action=action,
            infer_ms=infer_ms,
            news_age_sec=news_age_sec,
            source=source,
            symbol=symbol,
        )

        self._emit_shadow_log(card, item)

        return card

    def _determine_action(
        self,
        theme_probs: Dict[str, float],
        contamination_risk: float,
        confidence: float,
        keyword_match: Optional[KeywordMatch],
    ) -> CriticAction:
        """Determine the critic action based on scores.
        
        WOULD_SUPPRESS_HARD_SKIP when:
        1. contamination_risk >= 0.6, OR
        2. matched_field == "summary" AND low theme_prob for matched theme
        """
        if contamination_risk >= 0.6:
            return CriticAction.WOULD_SUPPRESS_HARD_SKIP

        if keyword_match and keyword_match.matched_field == "summary":
            matched_theme = keyword_match.theme_id
            if matched_theme in theme_probs:
                theme_prob = theme_probs[matched_theme]
                if theme_prob < 0.25:
                    return CriticAction.WOULD_SUPPRESS_HARD_SKIP

        return CriticAction.PASS

    def _emit_shadow_log(self, card: CriticCard, item: "ScoredNewsItem") -> None:
        """Emit structured shadow log for the critic card.
        
        Logs ALL scored items (not just WOULD_SUPPRESS) to the SAME db_logger
        path ThemeShock uses, so Monday rollup can score precision/FP/recall/latency.
        
        Structured log line includes all Stage A fields:
        - theme_probs (top theme + prob shown in line, full dict in extra_data)
        - relevance_by_symbol (in extra_data)
        - stance
        - contamination_risk
        - confidence
        - rationale_short (as reason)
        - keyword_match with matched_text + matched_field
        - infer_ms for latency measurement
        - news_age_sec for lead-time analysis
        """
        if not self.is_shadow_mode():
            return

        if card.skip_reason:
            logger.debug(
                "gpu_news_critic skip event_id=%s skip_reason=%s headline=%s",
                card.event_id, card.skip_reason, card.headline[:50],
            )
            return

        top_theme = max(card.theme_probs, key=card.theme_probs.get) if card.theme_probs else "none"
        top_prob = card.theme_probs.get(top_theme, 0) if card.theme_probs else 0

        kw_info = ""
        if card.keyword_match:
            kw_info = f" kw={card.keyword_match.theme_id}:{card.keyword_match.matched_field}:{card.keyword_match.matched_text[:20]}"

        logger.info(
            "gpu_news_critic "
            "action=%s event_id=%s symbol=%s "
            "theme=%s(%.2f) stance=%s contam=%.2f conf=%.2f "
            "news_age_sec=%.1f infer_ms=%.1f source=%s%s "
            "headline=%s",
            card.action.value,
            card.event_id,
            card.symbol,
            top_theme, top_prob,
            card.stance.value,
            card.contamination_risk,
            card.confidence,
            card.news_age_sec,
            card.infer_ms,
            card.source,
            kw_info,
            card.headline[:60],
        )

        if self._engine and hasattr(self._engine, "db_logger") and self._engine.db_logger:
            global _DB_PERSIST_ERROR_LOGGED
            try:
                self._engine.db_logger.log_strategy_decision(
                    strategy="gpu_news_critic",
                    symbol=card.symbol,
                    action=f"shadow_{card.action.value}",
                    reason=card.rationale_short,
                    extra_data=card.to_dict(),
                )
            except AttributeError as exc:
                if not _DB_PERSIST_ERROR_LOGGED:
                    logger.warning(
                        "gpu_news_critic db_persist_error symbol=%s err=%s "
                        "(log_strategy_decision missing? suppressing future warnings)",
                        card.symbol, exc,
                    )
                    _DB_PERSIST_ERROR_LOGGED = True
            except Exception as exc:
                if not _DB_PERSIST_ERROR_LOGGED:
                    logger.warning(
                        "gpu_news_critic db_persist_error symbol=%s err=%s "
                        "(suppressing future warnings)",
                        card.symbol, exc,
                    )
                    _DB_PERSIST_ERROR_LOGGED = True

    def process_news_item_from_publish(
        self,
        item: "ScoredNewsItem",
        keyword_match: Optional[KeywordMatch] = None,
    ) -> Optional[CriticCard]:
        """Process a news item from NewsBus publish path.
        
        Called after ThemeShock processes the item. Accepts the keyword_match
        from ThemeShock tagger when available.
        
        Returns:
            CriticCard if processed, None if disabled or skipped.
        """
        if not self.is_enabled():
            return None

        return self.score_news_item(item, keyword_match)


_critic_singleton: Optional[GPUNewsCritic] = None


def get_gpu_news_critic(engine=None, use_fake: bool = False) -> GPUNewsCritic:
    """Get or create the singleton GPUNewsCritic instance."""
    global _critic_singleton
    if _critic_singleton is None:
        _critic_singleton = GPUNewsCritic(engine=engine, use_fake=use_fake)
    elif engine is not None and _critic_singleton._engine is None:
        _critic_singleton._engine = engine
    return _critic_singleton


def reset_gpu_news_critic() -> None:
    """Reset the singleton (for testing)."""
    global _critic_singleton
    _critic_singleton = None
