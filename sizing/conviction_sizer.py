"""Conviction-weighted sizing — confluence score → size multiplier.

v-conviction-sizer-shadow-2026-06-11. Operator directive: "when the
trade has real solid confirmation take more risk."

Evidence (research join of 89 historical trades to their time-matched
meta-model probabilities, 2026-06-11):

    meta_proba < 0.60   n=58  WR 46.6%  PF 0.44  avg −0.21%
    0.60 – 0.70         n=22  WR 54.5%  PF 3.01  avg +0.65%
    >= 0.70             n=9   WR 66.7%  PF 5.02  avg +1.47%

The bot's own meta-model already separates its good trades from its
bleed — nobody was using it for size. SHADOW ONLY in this version:
the engine logs the would-be multiplier per routed signal to
conviction_sizer_shadow.ndjson. Promotion gate (roadmap): live shadow
must reproduce the gradient over n >= 20 fresh trades, then wire at
an initial cap of 1.25x before earning 1.5x.

The multiplier NEVER overrides the risk manager's absolute caps —
it scales within them.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("TradingBot")

DEFAULT_LEDGER = "conviction_sizer_shadow.ndjson"

# Multiplier bounds. The floor matters as much as the cap: the PF-0.44
# bucket is where the historical bleed lives.
MULT_MIN = 0.5
MULT_MAX = 1.5


@dataclass(frozen=True)
class ConvictionRead:
    score: float          # 0..1 composite confluence
    multiplier: float     # MULT_MIN..MULT_MAX
    components: Dict[str, float]
    reason: str


def conviction_score(
    meta_proba: Optional[float] = None,
    ml_agrees: Optional[bool] = None,
    allocator_allows: Optional[bool] = None,
    direction: Optional[float] = None,
    volume_ratio: Optional[float] = None,
) -> ConvictionRead:
    """Compose independent confirmations into one conviction read.

    Pure function; every input is optional and a missing input scores
    neutral (0.5 weight contribution) — absence of evidence is not
    evidence of weakness, it just doesn't add conviction.

    Weights: meta_proba dominates (0.5) because it is the only input
    with measured predictive power (PF 0.44 → 5.02 gradient). The
    other four are 0.125 each pending their own evidence.
    """
    comps: Dict[str, float] = {}

    comps["meta"] = (
        max(0.0, min(1.0, float(meta_proba))) if meta_proba is not None
        else 0.5
    )
    comps["ml"] = 1.0 if ml_agrees else (0.5 if ml_agrees is None else 0.0)
    comps["allocator"] = (
        1.0 if allocator_allows else (0.5 if allocator_allows is None else 0.0)
    )
    if direction is None:
        comps["direction"] = 0.5
    else:
        # |direction| 0..10 → 0..1; sign handled upstream by the
        # signal side itself.
        comps["direction"] = max(0.0, min(1.0, abs(float(direction)) / 10.0))
    if volume_ratio is None:
        comps["volume"] = 0.5
    else:
        # 1.0x = neutral 0.5; 2.5x+ = full confirmation.
        comps["volume"] = max(0.0, min(1.0, (float(volume_ratio) - 0.5) / 2.0))

    score = (
        0.5 * comps["meta"]
        + 0.125 * comps["ml"]
        + 0.125 * comps["allocator"]
        + 0.125 * comps["direction"]
        + 0.125 * comps["volume"]
    )
    score = max(0.0, min(1.0, score))

    # Score → multiplier: linear map of [0.3, 0.8] onto
    # [MULT_MIN, MULT_MAX]; clamped outside.
    t = (score - 0.3) / 0.5
    multiplier = MULT_MIN + max(0.0, min(1.0, t)) * (MULT_MAX - MULT_MIN)

    reason = (
        f"conviction {score:.2f} → size x{multiplier:.2f} "
        f"(meta={comps['meta']:.2f}, ml={comps['ml']:.1f}, "
        f"alloc={comps['allocator']:.1f}, dir={comps['direction']:.2f}, "
        f"vol={comps['volume']:.2f})"
    )
    return ConvictionRead(
        score=round(score, 4),
        multiplier=round(multiplier, 3),
        components={k: round(v, 4) for k, v in comps.items()},
        reason=reason,
    )


class ConvictionSizerShadow:
    """Append-only shadow ledger of would-be sizing decisions."""

    def __init__(self, ledger_path: Path | str = DEFAULT_LEDGER) -> None:
        self._ledger_path = Path(ledger_path)

    def evaluate(
        self, strategy: str, symbol: str, **inputs: Any
    ) -> Optional[ConvictionRead]:
        """Score a routed signal and log the would-be multiplier.
        Never raises into the signal path."""
        try:
            read = conviction_score(**inputs)
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "strategy": strategy,
                "symbol": symbol,
                "score": read.score,
                "would_be_multiplier": read.multiplier,
                **{f"in_{k}": v for k, v in read.components.items()},
            }
            try:
                with open(self._ledger_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")
            except OSError as exc:
                logger.warning("conviction_sizer: ledger write failed: %s",
                               exc)
            return read
        except Exception as exc:
            logger.debug("conviction_sizer shadow error: %s", exc)
            return None


# ─────────────────────────────────────────────────────────────────────────────
# v-feature-snapshot-emit-2026-09-09: news gate multiplier shadow comparison
# ─────────────────────────────────────────────────────────────────────────────
GATE_MULT_SHADOW_LEDGER = "news_gate_multiplier_shadow.ndjson"


@dataclass(frozen=True)
class GateMultiplierShadowEntry:
    """Shadow entry comparing news gate multiplier vs placeholder model score."""
    timestamp: str
    symbol: str
    strategy: str
    news_gate_multiplier: float
    corroboration_n: int
    news_age_sec: Optional[float]
    source_tier_min: Optional[int]
    model_score_placeholder: float  # Placeholder for future inference model
    delta: float  # gate_mult - model_score (for measuring gate vs model divergence)


class NewsGateMultiplierShadow:
    """v-feature-snapshot-emit-2026-09-09: shadow log comparing news gate
    multiplier vs a future model score.
    
    This enables offline analysis of whether the deterministic news gate
    (corroboration + freshness + source tier) agrees with what a trained
    model would produce. When inference is enabled, this comparison helps
    identify miscalibration between gates and learned signals.
    
    No trade effect — pure observability for gate/model alignment research.
    """

    def __init__(self, ledger_path: Path | str = GATE_MULT_SHADOW_LEDGER) -> None:
        self._ledger_path = Path(ledger_path)

    def log_comparison(
        self,
        symbol: str,
        strategy: str,
        news_gate_multiplier: float,
        corroboration_n: int,
        news_age_sec: Optional[float] = None,
        source_tier_min: Optional[int] = None,
        model_score_placeholder: Optional[float] = None,
    ) -> Optional[GateMultiplierShadowEntry]:
        """Log a comparison between news gate multiplier and model score.
        
        Args:
            symbol: Ticker symbol.
            strategy: Strategy name (e.g., 'free_news_sentiment').
            news_gate_multiplier: Deterministic gate output (0.0, 0.5, or 1.0).
            corroboration_n: Number of distinct sources.
            news_age_sec: Age of freshest news.
            source_tier_min: Best source tier (1=primary).
            model_score_placeholder: Future model inference output (0-1).
                Default 0.5 (neutral) until inference is enabled.
        
        Returns:
            GateMultiplierShadowEntry if logged successfully, None on error.
        """
        if model_score_placeholder is None:
            model_score_placeholder = 0.5  # Neutral placeholder
        
        delta = news_gate_multiplier - model_score_placeholder
        
        entry = GateMultiplierShadowEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            symbol=symbol,
            strategy=strategy,
            news_gate_multiplier=news_gate_multiplier,
            corroboration_n=corroboration_n,
            news_age_sec=news_age_sec,
            source_tier_min=source_tier_min,
            model_score_placeholder=model_score_placeholder,
            delta=round(delta, 4),
        )
        
        try:
            entry_dict = {
                "timestamp": entry.timestamp,
                "symbol": entry.symbol,
                "strategy": entry.strategy,
                "news_gate_multiplier": entry.news_gate_multiplier,
                "corroboration_n": entry.corroboration_n,
                "news_age_sec": entry.news_age_sec,
                "source_tier_min": entry.source_tier_min,
                "model_score_placeholder": entry.model_score_placeholder,
                "delta": entry.delta,
            }
            with open(self._ledger_path, "a") as f:
                f.write(json.dumps(entry_dict) + "\n")
            return entry
        except OSError as exc:
            logger.warning("news_gate_mult_shadow: ledger write failed: %s", exc)
            return None
        except Exception as exc:
            logger.debug("news_gate_mult_shadow error: %s", exc)
            return None


# Singleton for easy access from risk manager
_gate_mult_shadow: Optional[NewsGateMultiplierShadow] = None


def get_gate_mult_shadow() -> NewsGateMultiplierShadow:
    """Get or create the singleton NewsGateMultiplierShadow."""
    global _gate_mult_shadow
    if _gate_mult_shadow is None:
        _gate_mult_shadow = NewsGateMultiplierShadow()
    return _gate_mult_shadow
