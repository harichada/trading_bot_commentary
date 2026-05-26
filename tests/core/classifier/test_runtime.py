"""v-classifier-runtime-2026-05-13: tests for the bot-runtime →
SymbolFeatures bridge + lifecycle wrapper.

The bot's signal evaluation has access to:
  - MarketData (close, open, high, low, volume + indicators dict)
  - Current price-book entries
  - Cached news sentiment (per-symbol)

It does NOT have access to:
  - Daily bars (SMA50 needs 50 days of history; the bot streams 5-min)
  - Multi-day relative strength
  - Earnings/halt calendars

The runtime bridge populates whatever IS available at signal time and
leaves the rest at SymbolFeatures defaults (which are neutral, so the
classifier degrades to a no-information posture rather than a wrong
one).

Lifecycle: ``ClassifierRuntime`` lazily loads the checkpoint on first
use. Failure to load triggers fail-open (returns None decisions, logs
once at WARNING). The composer's existing fail-open layer then keeps
the bot rule-only.
"""
from __future__ import annotations

import pytest


class _StubMarketData:
    """Minimal MarketData-shaped object for tests. The real engine
    MarketData has more fields; we duck-type on what the bridge reads."""

    def __init__(self, symbol, close, open=None, high=None, low=None,
                 volume=None, indicators=None):
        self.symbol = symbol
        self.close = close
        self.open = open if open is not None else close
        self.high = high if high is not None else close
        self.low = low if low is not None else close
        self.volume = volume if volume is not None else 1_000_000
        self.indicators = indicators or {}


class TestBuildRuntimeFeatures:
    def test_minimal_market_data(self):
        from core.classifier.runtime import build_runtime_features

        md = _StubMarketData("NVDA", close=110.0, open=109.0)
        f = build_runtime_features(md)
        assert f.symbol == "NVDA"
        assert f.close == 110.0
        # Fields not derivable at runtime stay None / neutral default.
        assert f.sma_50_daily is None or f.sma_50_daily == 0  # neutral

    def test_indicators_populate_what_they_can(self):
        from core.classifier.runtime import build_runtime_features

        md = _StubMarketData("NVDA", close=110.0,
                             indicators={
                                 "rsi": 68.5,
                                 "sma_20": 108.0,
                                 "sma_50": 105.0,
                                 "atr": 2.5,
                                 "volume_ratio": 1.3,
                             })
        f = build_runtime_features(md)
        # RSI from indicators surfaces on SymbolFeatures.rsi_14
        assert f.rsi_14 == 68.5
        # SMAs propagated
        assert f.sma_20_daily == 108.0
        assert f.sma_50_daily == 105.0
        # ATR percent computed
        assert f.daily_atr_pct == pytest.approx(2.5 / 110.0)
        # Volume ratio passes through
        assert f.vol_ratio_today == 1.3

    def test_news_sentiment_kwarg(self):
        from core.classifier.runtime import build_runtime_features

        md = _StubMarketData("F", close=12.0)
        f = build_runtime_features(md, news_sentiment_7d=0.4,
                                       news_fresh_count_today=3)
        assert f.news_sentiment_7d == 0.4
        assert f.news_fresh_count_today == 3


class TestClassifierRuntimeLifecycle:
    def test_disabled_returns_none(self):
        """When ``enabled=False``, the runtime should never load a
        checkpoint or run inference. ``shadow_evaluate`` returns None."""
        from core.classifier.runtime import ClassifierRuntime

        runtime = ClassifierRuntime(enabled=False)
        md = _StubMarketData("NVDA", close=110.0)
        result = runtime.shadow_evaluate(md)
        assert result is None

    def test_enabled_without_checkpoint_fails_open(self):
        """Enabled but checkpoint missing → rule-only decision +
        WARNING logged once."""
        from core.classifier.runtime import ClassifierRuntime

        runtime = ClassifierRuntime(
            enabled=True,
            model_path="/nonexistent/model.pt",
        )
        md = _StubMarketData("NVDA", close=110.0,
                             indicators={"rsi": 35.0, "sma_20": 108.0, "sma_50": 105.0})
        result = runtime.shadow_evaluate(md)
        # Should be the rule-only decision — not None — since the rule
        # layer always works without the checkpoint.
        assert result is not None
        assert result.symbol == "NVDA"

    def test_enabled_with_real_checkpoint(self):
        """If a checkpoint exists, ``shadow_evaluate`` returns a
        composed decision."""
        torch = pytest.importorskip("torch", reason="torch not installed")
        import tempfile
        from pathlib import Path
        from core.classifier.runtime import ClassifierRuntime
        from core.classifier.trained.model import SideClassifierFFN

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "ffn.pt"
            torch.save(SideClassifierFFN().state_dict(), ckpt)
            runtime = ClassifierRuntime(enabled=True, model_path=str(ckpt))

            md = _StubMarketData("NVDA", close=110.0,
                                 indicators={"rsi": 35.0,
                                             "sma_20": 108.0,
                                             "sma_50": 105.0})
            result = runtime.shadow_evaluate(md)
            assert result is not None
            assert result.symbol == "NVDA"
