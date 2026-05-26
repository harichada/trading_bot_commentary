"""v-classifier-features-2026-05-13: tests for the research feature
builder. Pinned because the rule-based classifier consumes its output
shape and the trained model will too."""
from __future__ import annotations

import pytest


def _daily_bars_uptrend():
    """60 bars in a clean uptrend with realistic noise."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    n = 60
    base = np.linspace(100.0, 130.0, n)
    noise = rng.normal(0, 0.5, n)
    closes = base + noise
    df = pd.DataFrame({
        "open": closes - 0.3,
        "high": closes + 0.7,
        "low": closes - 0.7,
        "close": closes,
        "volume": rng.integers(800_000, 1_200_000, n),
    }, index=pd.date_range("2024-01-01", periods=n, freq="B"))
    return df


def _daily_bars_downtrend():
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(43)
    n = 60
    base = np.linspace(130.0, 100.0, n)
    noise = rng.normal(0, 0.5, n)
    closes = base + noise
    df = pd.DataFrame({
        "open": closes + 0.3,
        "high": closes + 0.7,
        "low": closes - 0.7,
        "close": closes,
        "volume": rng.integers(800_000, 1_200_000, n),
    }, index=pd.date_range("2024-01-01", periods=n, freq="B"))
    return df


class TestBuildFeatures:
    def test_uptrend_features_argue_long(self):
        from research.features import build_features
        from core.classifier.rule_based import classify_rule_based
        from core.classifier.types import Side

        df = _daily_bars_uptrend()
        f = build_features("NVDA", df)
        assert f.sma_20_daily is not None
        assert f.sma_50_daily is not None
        # In a clean uptrend, SMA20 > SMA50 and close > SMA20
        assert f.sma_20_daily > f.sma_50_daily
        assert f.close > f.sma_20_daily

        d = classify_rule_based(f)
        # Should at least lean LONG
        assert d.long_score >= 0.5
        # Soft scores are mirror images by construction
        assert d.long_score > d.short_score

    def test_downtrend_features_argue_short(self):
        from research.features import build_features
        from core.classifier.rule_based import classify_rule_based

        df = _daily_bars_downtrend()
        f = build_features("XYZ", df)
        assert f.sma_20_daily < f.sma_50_daily
        d = classify_rule_based(f)
        assert d.short_score > d.long_score

    def test_short_history_returns_partial_features(self):
        """With fewer than 50 bars, SMA-50 is None but SMA-20 may be set."""
        from research.features import build_features
        import pandas as pd

        rows = [(100 + i * 0.5,) for i in range(25)]
        df = pd.DataFrame(rows, columns=["close"])
        df["open"] = df["close"]
        df["high"] = df["close"] + 0.5
        df["low"] = df["close"] - 0.5
        df["volume"] = 1_000_000
        df.index = pd.date_range("2024-01-01", periods=len(df), freq="B")

        f = build_features("ABC", df)
        assert f.sma_50_daily is None
        assert f.sma_20_daily is not None

    def test_atr_pct_is_fractional(self):
        from research.features import build_features

        df = _daily_bars_uptrend()
        f = build_features("NVDA", df)
        assert f.daily_atr_pct is not None
        assert 0 < f.daily_atr_pct < 1.0   # fraction, not absolute price

    def test_news_signals_pass_through(self):
        from research.features import build_features

        df = _daily_bars_uptrend()
        f = build_features(
            "NVDA", df,
            news_sentiment_7d=0.42,
            news_fresh_count_today=3,
            news_recency_min=5.0,
            earnings_within_2d=True,
        )
        assert f.news_sentiment_7d == 0.42
        assert f.news_fresh_count_today == 3
        assert f.earnings_within_2d is True
