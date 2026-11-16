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
import re
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
    # v-themeshock-ailab-reroute-2026-11-16: populated ONLY when the matching
    # flag is on (None otherwise -> to_dict() is byte-identical to before).
    reroute_diag: Optional[Dict[str, Any]] = None
    contam_diag: Optional[Dict[str, Any]] = None

    def to_dict(self) -> dict:
        """Serialize for structured logging.
        
        Returns all Stage A fields for Research Monday rollup:
        theme_probs, relevance_by_symbol, stance, contamination_risk,
        confidence, rationale_short, keyword_match (with matched_text+matched_field).
        """
        d = {
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
        if self.reroute_diag is not None:
            d.update(self.reroute_diag)
        if self.contam_diag is not None:
            d.update(self.contam_diag)
        return d


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


# ---------------------------------------------------------------------------
# v-themeshock-ailab-reroute-2026-11-16 (Research 11/16 weekly rollup)
#
# Pure, config-free helpers so the golden-set replay can call exactly the
# same logic the critic runs. All are gated by default-False flags at the
# call sites; with every flag off the critic output is unchanged.
# ---------------------------------------------------------------------------

AILAB_RE = re.compile(
    r"\b(anthropic|openai|open ai|xai|x\.ai|mistral(?: ai)?|cohere|perplexity|"
    r"deepseek|safe superintelligence|thinking machines(?: lab)?|scale ai|"
    r"inflection ai|character\.ai)\b",
    re.IGNORECASE,
)
ANTHROPIC_RE = re.compile(r"\b(anthropic|amodei|claude)\b", re.IGNORECASE)

AILAB_FINANCING_RE = re.compile(
    r"\b(ipos?|s-1|f-1|prospectus|go(?:es|ing)? public|files? (?:confidentially )?"
    r"(?:to go public|for (?:an? )?ipo)|public (?:offering|debut|listing)|"
    r"raise[sd]?|raising|fundrais\w*|funding(?: rounds?)?|financing|"
    r"series [a-h]|valuations?|valued at|valuing|stakes?|invests?|invested|investing|"
    r"investments?|investors?|tender offers?|share sale|secondary sale|loans?|lends?|lent)\b",
    re.IGNORECASE,
)

EARNINGS_TERM_RE = re.compile(
    r"\b(eps|earnings|quarterly (?:results|report|revenue|sales|loss|profit)|"
    r"q[1-4] (?:results|earnings|revenue|sales|financials|net (?:loss|income)|loss|profit)|"
    r"reports? q[1-4]|fiscal (?:q[1-4]|quarter)|"
    r"(?:revenue|sales|earnings|profit) (?:beats?|miss(?:es)?|tops?)|"
    r"beats? (?:on|estimates|expectations|forecasts?)|miss(?:es)? (?:on|estimates|expectations|forecasts?)|"
    r"guidance|net (?:loss|income) of)\b",
    re.IGNORECASE,
)

OPTICAL_PRODUCT_RE = re.compile(
    r"\b(dci|data[- ]?center interconnects?|interconnects?|open line systems?|"
    r"transceivers?|optical (?:modules?|transceivers?|networking|network|interconnects?|"
    r"systems?|engines?|platforms?|components?|links?|innovations?)|coherent optics|"
    r"co-packaged optics|pluggables?|silicon photonics|photonics|data optics)\b",
    re.IGNORECASE,
)
# Word-boundary vendor names (avoids 'cien' in 'efficiency'/'scientific').
OPTICAL_NAME_RE = re.compile(
    r"\b(cohr|coherent corp|coherent|cien|ciena|aaoi|applied optoelectronics|"
    r"lumentum|infinera|corning|glw)\b",
    re.IGNORECASE,
)
# Plain oil, word-boundary: oil/oils/oilfield(s). NOT toil/boil/soil/spoil.
PLAIN_OIL_RE = re.compile(r"\b(oil|oils|oilfield|oilfields)\b", re.IGNORECASE)

REROUTE_FROM_THEME = "earnings_chip"
REROUTE_TO_THEME = "ai_compute"


def _top_theme(theme_probs: Dict[str, float]) -> str:
    return max(theme_probs, key=theme_probs.get) if theme_probs else "none"


def ailab_financing_reroute(
    headline: str,
    theme_probs: Dict[str, float],
    enabled: bool,
    second_choice_min_prob: float = 0.0,
) -> Tuple[Dict[str, float], Optional[Dict[str, Any]]]:
    """Re-route earnings_chip -> ai_compute for AI-lab financing headlines.

    Returns (theme_probs, diag). When ``enabled`` is False returns the input
    dict object unchanged and diag=None (off-path is identical).

    When enabled, diag is ALWAYS returned (every row) with:
      original_theme, rerouted_theme, reroute_reason, rerouted_sub_theme,
      reroute_ailab, reroute_financing_term, reroute_earnings_term,
      theme_probs_original (only when rerouted).

    Re-route = swap the earnings_chip and ai_compute probabilities so
    ai_compute becomes top while the distribution is otherwise preserved.
    """
    if not enabled:
        return theme_probs, None
    original = _top_theme(theme_probs)
    hl = headline or ""
    lab = AILAB_RE.search(hl)
    fin = AILAB_FINANCING_RE.search(hl)
    earn = EARNINGS_TERM_RE.search(hl)
    diag: Dict[str, Any] = {
        "original_theme": original,
        "rerouted_theme": original,
        "reroute_reason": None,
        "rerouted_sub_theme": None,
        "reroute_ailab": lab.group(0).lower() if lab else None,
        "reroute_financing_term": fin.group(0).lower() if fin else None,
        "reroute_earnings_term": earn.group(0).lower() if earn else None,
    }
    if original != REROUTE_FROM_THEME or REROUTE_TO_THEME not in theme_probs:
        return theme_probs, diag
    if earn:
        diag["reroute_reason"] = "blocked_earnings_term"
        return theme_probs, diag

    reason = None
    if lab and fin:
        reason = "ailab_financing"
    elif second_choice_min_prob and second_choice_min_prob > 0:
        ranked = sorted(theme_probs, key=theme_probs.get, reverse=True)
        if (
            len(ranked) >= 2
            and ranked[1] == REROUTE_TO_THEME
            and theme_probs[REROUTE_TO_THEME] >= second_choice_min_prob
        ):
            reason = "second_choice_ai_compute"
    if reason is None:
        return theme_probs, diag

    new_probs = dict(theme_probs)
    new_probs[REROUTE_TO_THEME], new_probs[REROUTE_FROM_THEME] = (
        theme_probs[REROUTE_FROM_THEME], theme_probs[REROUTE_TO_THEME],
    )
    diag["rerouted_theme"] = REROUTE_TO_THEME
    diag["reroute_reason"] = reason
    diag["rerouted_sub_theme"] = "anthropic" if ANTHROPIC_RE.search(hl) else None
    diag["theme_probs_original"] = dict(theme_probs)
    return new_probs, diag


def optical_interconnect_guard_hit(headline: str, summary: str = "") -> Optional[str]:
    """Return a reason string when the item is optical-vendor / optical
    interconnect product news (policy: optical -> FP), else None."""
    hl = headline or ""
    m = OPTICAL_NAME_RE.search(hl)
    if m:
        return f"optical_name_headline:{m.group(0).lower()}"
    p = OPTICAL_PRODUCT_RE.search(hl)
    if p:
        n = OPTICAL_NAME_RE.search(f"{hl} {summary or ''}")
        if n:
            return f"optical_product:{p.group(0).lower()}+name:{n.group(0).lower()}"
    return None


def contam_lexicon_decision(
    headline: str,
    summary: str,
    current_risk: float,
    lexicon: List[str],
    ai_triggers: List[str],
    optical_guard: bool = False,
    plain_oil: bool = False,
) -> Tuple[float, Optional[str]]:
    """Lexicon contamination bump (v-book-b-contam-lexicon-2026-09-17) with the
    two optional 2026-11-16 extensions. With both extensions False this is
    exactly the legacy _apply_contam_lexicon logic. Returns (risk, reason)."""
    headline_lower = (headline or "").lower()
    summary_lower = (summary or "").lower()
    text_lower = f"{headline_lower} {summary_lower}"

    reason = None
    has_lexicon_hit = any(token.lower() in text_lower for token in lexicon)
    if has_lexicon_hit:
        reason = "lexicon"
    if not has_lexicon_hit and plain_oil and PLAIN_OIL_RE.search(text_lower):
        has_lexicon_hit, reason = True, "plain_oil"
    guard_reason = optical_interconnect_guard_hit(headline, summary) if optical_guard else None
    if not has_lexicon_hit and guard_reason:
        has_lexicon_hit, reason = True, "optical_guard_name"
    if not has_lexicon_hit:
        return current_risk, None

    headline_has_ai_trigger = any(t in headline_lower for t in ai_triggers)
    if headline_has_ai_trigger:
        if guard_reason:
            return max(current_risk, 0.65), f"optical_guard:{guard_reason}"
        return current_risk, "ai_trigger_exempt"

    return max(current_risk, 0.65), reason


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
        diag: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, float], Dict[str, float], Stance, float, float, str]:
        """Score a news item.

        ``diag`` (optional dict) receives reroute/contam diagnostics when the
        2026-11-16 flags are on; untouched when they are off.
        
        Returns:
            theme_probs: dict of theme_id -> probability
            relevance_by_symbol: dict of symbol -> relevance 0-1
            stance: Stance enum
            contamination_risk: 0-1
            confidence: 0-1
            rationale_short: brief explanation
        """
        text = f"{headline} {summary}".strip()
        if not text:
            return {}, {}, Stance.IRRELEVANT, 0.0, 0.0, "empty_text"

        theme_probs, best_stance, stance_confidence = self._embed_scores(text)
        return self._score_from_probs(
            headline, summary, keyword_match,
            theme_probs, best_stance, stance_confidence, diag,
        )

    def _embed_scores(self, text: str) -> Tuple[Dict[str, float], Stance, float]:
        """Embedding similarity -> (theme_probs, best_stance, stance_confidence)."""
        import torch

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
        return theme_probs, best_stance, stance_confidence

    def _score_from_probs(
        self,
        headline: str,
        summary: str,
        keyword_match: Optional[KeywordMatch],
        theme_probs: Dict[str, float],
        best_stance: Stance,
        stance_confidence: float,
        diag: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, float], Dict[str, float], Stance, float, float, str]:
        """Pure post-embedding scoring (no torch). Unchanged when flags off."""
        from core.config import Config
        cfg = Config()
        if cfg.THEME_CRITIC_AILAB_FINANCING_REROUTE:
            theme_probs, rdiag = ailab_financing_reroute(
                headline, theme_probs, True,
                cfg.THEME_CRITIC_AILAB_SECOND_CHOICE_MIN_PROB,
            )
            if diag is not None and rdiag is not None:
                diag["reroute"] = rdiag

        relevance_by_symbol = {}
        top_theme = max(theme_probs, key=theme_probs.get)
        top_theme_prob = theme_probs[top_theme]

        if top_theme in THEME_BASKET:
            for symbol in THEME_BASKET[top_theme]:
                relevance_by_symbol[symbol] = top_theme_prob * 0.8

        contamination_risk = self._compute_contamination_risk(
            headline, summary, theme_probs, keyword_match, diag=diag
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
        rd = (diag or {}).get("reroute")
        if rd and rd.get("rerouted_theme") != rd.get("original_theme"):
            rationale_parts.append(
                f"reroute={rd['original_theme']}->{rd['rerouted_theme']}({rd['reroute_reason']})"
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

    AI_COMPUTE_TRIGGERS = [
        "anthropic",
        "amodei",
        "openai",
        "frontier ai",
        "ai slowdown",
        "gpu demand",
        "data center",
        "data-center",
        "datacenter",
        "ai compute",
        "ai infrastructure",
    ]

    def _compute_contamination_risk(
        self,
        headline: str,
        summary: str,
        theme_probs: Dict[str, float],
        keyword_match: Optional[KeywordMatch],
        diag: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Compute contamination risk score.
        
        Contamination indicators:
        1. Keyword match in summary but low headline theme_prob
        2. Multiple unrelated topics detected (high "other" + specific theme)
        3. Summary significantly longer than headline (aggregation signal)
        4. Keyword matched theme disagrees with model's top theme
        5. (v-book-b-contam-lexicon-2026-09-17) Lexicon hit on headline/summary
           for Corning/GLW/oil wraps UNLESS AI_COMPUTE_TRIGGERS also present
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

        risk = self._apply_contam_lexicon(headline, summary, risk, diag=diag)

        return min(1.0, risk)

    def _apply_contam_lexicon(
        self,
        headline: str,
        summary: str,
        current_risk: float,
        diag: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Apply lexicon-based contamination detection.

        v-book-b-contam-lexicon-2026-09-17: bumps contamination_risk to >=0.65
        when lexicon tokens found in headline/summary text UNLESS headline
        also has AI_COMPUTE_TRIGGERS (keeps TP bf0c664141dc Nasdaq/AI Leaders).

        Matches on headline/summary text only (NOT symbols_seen dump).
        """
        from core.config import Config
        cfg = Config()

        if not cfg.GPU_CRITIC_CONTAM_LEXICON_ENABLE:
            return current_risk

        lexicon = cfg.GPU_CRITIC_CONTAM_LEXICON
        if not lexicon:
            return current_risk

        optical_guard = cfg.THEME_CRITIC_OPTICAL_INTERCONNECT_GUARD
        plain_oil = cfg.THEME_CRITIC_CONTAM_PLAIN_OIL
        if optical_guard or plain_oil:
            # v-themeshock-ailab-reroute-2026-11-16: log pre- and post-guard.
            pre, pre_reason = contam_lexicon_decision(
                headline, summary, current_risk, lexicon, self.AI_COMPUTE_TRIGGERS,
            )
            post, post_reason = contam_lexicon_decision(
                headline, summary, current_risk, lexicon, self.AI_COMPUTE_TRIGGERS,
                optical_guard=optical_guard, plain_oil=plain_oil,
            )
            if diag is not None:
                diag["contam"] = {
                    "contam_risk_pre_guard": min(1.0, pre),
                    "contam_risk_post_guard": min(1.0, post),
                    "contam_guard_reason": post_reason if post != pre else None,
                    "contam_lexicon_reason_pre": pre_reason,
                    "contam_lexicon_reason_post": post_reason,
                    "contam_optical_guard_on": bool(optical_guard),
                    "contam_plain_oil_on": bool(plain_oil),
                }
            return post

        headline_lower = headline.lower()
        summary_lower = summary.lower()
        text_lower = f"{headline_lower} {summary_lower}"

        has_lexicon_hit = any(token.lower() in text_lower for token in lexicon)
        if not has_lexicon_hit:
            return current_risk

        headline_has_ai_trigger = any(
            t in headline_lower for t in self.AI_COMPUTE_TRIGGERS
        )
        if headline_has_ai_trigger:
            return current_risk

        return max(current_risk, 0.65)


class FakeCritic:
    """Deterministic test double for GPU News Critic.
    
    For CI environments where model download is blocked or slow.
    Provides predictable outputs based on headline/summary content.
    
    Key behavior for contamination detection:
    - AI trigger in headline → ai_compute theme, low contamination
    - AI trigger ONLY in summary (not headline) → low theme_prob, high contamination
    - Crude oil/commodities in summary → high contamination
    - (v-book-b-contam-lexicon-2026-09-17) Lexicon hit on Corning/GLW/oil wraps
      → high contamination UNLESS headline has AI_COMPUTE_TRIGGERS
    """

    CONTAMINATION_TRIGGERS = [
        "crude oil",
        "petroleum",
        "opec",
        "oil prices",
        "wti crude",
        "brent crude",
        "corning",
        "glw",
        "nyse:glw",
        "motor oil",
        "oil crisis",
        "wti",
        "brent",
        "cohr",
        "cien",
        "aaoi",
    ]

    AI_COMPUTE_TRIGGERS = [
        "anthropic",
        "amodei",
        "openai",
        "frontier ai",
        "ai slowdown",
        "gpu demand",
        "data center",
        "data-center",
        "datacenter",
        "ai compute",
        "ai infrastructure",
    ]

    def score(
        self,
        headline: str,
        summary: str,
        keyword_match: Optional[KeywordMatch] = None,
        diag: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, float], Dict[str, float], Stance, float, float, str]:
        """Deterministic scoring for tests."""
        headline_lower = headline.lower()
        summary_lower = summary.lower()
        text = f"{headline} {summary}".lower()

        headline_has_ai_trigger = any(t in headline_lower for t in self.AI_COMPUTE_TRIGGERS)
        summary_has_ai_trigger = any(t in summary_lower for t in self.AI_COMPUTE_TRIGGERS)
        is_ai_compute = headline_has_ai_trigger or summary_has_ai_trigger

        has_contam_trigger = any(t in text for t in self.CONTAMINATION_TRIGGERS)
        is_contaminated = has_contam_trigger and not headline_has_ai_trigger
        
        summary_has_contam_trigger = any(
            t in summary_lower for t in self.CONTAMINATION_TRIGGERS
        )
        headline_has_contam_trigger = any(
            t in headline_lower for t in self.CONTAMINATION_TRIGGERS
        )

        ai_trigger_only_in_summary = summary_has_ai_trigger and not headline_has_ai_trigger

        if headline_has_ai_trigger:
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
            contamination_risk = 0.7 if summary_has_contam_trigger and not headline_has_contam_trigger else 0.65
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
            rationale = f"contamination detected ({contamination_risk:.2f}), lexicon hit"

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

        if diag is not None:
            from core.config import Config
            cfg = Config()
            if cfg.THEME_CRITIC_AILAB_FINANCING_REROUTE:
                theme_probs, rdiag = ailab_financing_reroute(
                    headline, theme_probs, True,
                    cfg.THEME_CRITIC_AILAB_SECOND_CHOICE_MIN_PROB,
                )
                diag["reroute"] = rdiag

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
        diag: Dict[str, Any] = {}

        if self._use_fake:
            theme_probs, relevance, stance, contam_risk, confidence, rationale = (
                self._fake_critic.score(headline, summary, keyword_match, diag=diag)
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
                    model.score(headline, summary, keyword_match, diag=diag)
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
            reroute_diag=diag.get("reroute"),
            contam_diag=diag.get("contam"),
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
        if card.reroute_diag is not None:
            rd = card.reroute_diag
            kw_info += (
                f" original_theme={rd.get('original_theme')}"
                f" rerouted_theme={rd.get('rerouted_theme')}"
                f" reroute_reason={rd.get('reroute_reason')}"
            )
        if card.contam_diag is not None:
            cd = card.contam_diag
            kw_info += (
                f" contam_pre={cd.get('contam_risk_pre_guard', 0.0):.2f}"
                f" contam_post={cd.get('contam_risk_post_guard', 0.0):.2f}"
                f" contam_guard={cd.get('contam_guard_reason')}"
            )

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
