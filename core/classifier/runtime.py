"""Runtime bridge between the live bot and the side classifier.

v-classifier-runtime-2026-05-13. Translates the engine's runtime state
(MarketData + indicators + cached news) into ``SymbolFeatures`` and
manages the trained-predictor lifecycle (lazy checkpoint load, atomic
fail-open on missing model).

Imported by ``core/engine.py`` only when ``Config.USE_SIDE_CLASSIFIER``
is True. The isolation meta-test
(``TestClassifierStaysIsolated``) is updated to allow the import in
this gated form, while still failing CI on any unconditional import.

Fail-open posture: missing checkpoint or torch unavailable → predictor
absent, composer returns rule-only decision, bot continues unaffected.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from core.classifier import classify
from core.classifier.types import SymbolFeatures, SymbolSideDecision

logger = logging.getLogger("TradingBot")


def build_runtime_features(
    market_data: Any,
    news_sentiment_7d: float = 0.0,
    news_fresh_count_today: int = 0,
    news_recency_min: Optional[float] = None,
) -> SymbolFeatures:
    """Construct ``SymbolFeatures`` from the engine's MarketData and
    optional cached news signals.

    The bot does NOT have daily-bar history at signal time (it streams
    5-min bars), so SMA-50, multi-day RS, and event-calendar fields
    that need broader context stay at their neutral defaults. The
    classifier's hard gates that depend on those (e.g.,
    earnings_within_2d) won't fire under shadow mode v1 — they'd
    require a separate daily fetcher (planned phase 2).
    """
    indicators = getattr(market_data, "indicators", {}) or {}

    # Pull what we have. The strategies' indicator dict uses keys like
    # 'rsi', 'sma_20', 'sma_50', 'atr', 'volume_ratio'. None of these
    # are guaranteed to exist — handle absence gracefully.
    rsi = indicators.get("rsi")
    sma_20 = indicators.get("sma_20")
    sma_50 = indicators.get("sma_50")
    atr_raw = indicators.get("atr")
    vol_ratio = indicators.get("volume_ratio", 1.0) or 1.0

    close = float(market_data.close)
    daily_atr_pct = None
    if atr_raw is not None and close > 0:
        try:
            daily_atr_pct = float(atr_raw) / close
        except Exception:
            daily_atr_pct = None

    return SymbolFeatures(
        symbol=market_data.symbol,
        close=close,
        sma_20_daily=float(sma_20) if sma_20 is not None else None,
        sma_50_daily=float(sma_50) if sma_50 is not None else None,
        rsi_14=float(rsi) if rsi is not None else None,
        daily_atr_pct=daily_atr_pct,
        vol_ratio_today=float(vol_ratio),
        news_sentiment_7d=float(news_sentiment_7d),
        news_fresh_count_today=int(news_fresh_count_today),
        news_recency_min=(float(news_recency_min)
                          if news_recency_min is not None else None),
        # All other fields default to neutral.
    )


class ClassifierRuntime:
    """Engine-facing wrapper.

    Lifecycle:
      - Constructed once at engine __init__ if config flag is True.
      - ``shadow_evaluate(market_data, **news)`` is called for every
        accepted signal (or every signal, depending on integration
        point). Returns a SymbolSideDecision or None.
      - Trained predictor is lazily loaded on first use. Failure to
        load is logged ONCE and the runtime continues without the
        trained layer (rule-only decisions).
    """

    def __init__(
        self,
        enabled: bool = False,
        model_path: Optional[str] = None,
        long_threshold: float = 0.4,
        short_threshold: float = 0.4,
    ) -> None:
        self.enabled = enabled
        self.model_path = model_path
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self._predictor = None
        self._predictor_load_attempted = False
        self._predictor_load_failed = False

    def _try_load_predictor(self) -> None:
        """One-shot lazy load. Sets ``_predictor`` or sets the
        failed-load latch. Logs WARNING exactly once on failure."""
        if self._predictor_load_attempted:
            return
        self._predictor_load_attempted = True
        if not self.model_path:
            return
        try:
            # Lazy import — torch / classifier.trained shouldn't be
            # required for the rule layer to work in a torch-less env.
            from core.classifier.trained.ffn_predictor import FFNPredictor
            self._predictor = FFNPredictor.from_checkpoint(
                self.model_path,
                long_threshold=self.long_threshold,
                short_threshold=self.short_threshold,
            )
            logger.info(
                "ClassifierRuntime: loaded predictor from %s",
                self.model_path,
            )
        except Exception as exc:
            self._predictor_load_failed = True
            logger.warning(
                "ClassifierRuntime: predictor load failed (%s); "
                "falling back to rule-only decisions for this session.",
                exc,
            )

    def shadow_evaluate(
        self,
        market_data: Any,
        news_sentiment_7d: float = 0.0,
        news_fresh_count_today: int = 0,
        news_recency_min: Optional[float] = None,
    ) -> Optional[SymbolSideDecision]:
        """Build features, consult composer, return the decision.

        Returns None iff ``enabled=False`` (caller can skip the audit).
        Otherwise always returns a SymbolSideDecision — rule-only when
        the trained predictor is absent.
        """
        if not self.enabled:
            return None

        self._try_load_predictor()
        features = build_runtime_features(
            market_data,
            news_sentiment_7d=news_sentiment_7d,
            news_fresh_count_today=news_fresh_count_today,
            news_recency_min=news_recency_min,
        )
        return classify(
            features,
            trained_predictor=self._predictor,
            long_threshold=self.long_threshold,
            short_threshold=self.short_threshold,
        )
