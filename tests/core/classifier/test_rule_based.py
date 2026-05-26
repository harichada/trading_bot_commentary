"""v-rule-based-classifier-2026-05-13: tests for the hard gates and
soft scoring of the rule-based side classifier.

Each test fixes ONE feature (or a minimal set) and asserts the
decision the classifier should produce. Defaults on SymbolFeatures
are neutral, so a test that omits a field is asserting "the
classifier's behavior should not depend on that field for this case."
"""
from __future__ import annotations

import pytest


def _features(**overrides):
    """Build SymbolFeatures with sane defaults, override per test."""
    from core.classifier.types import SymbolFeatures

    base = dict(symbol="TEST", close=100.0)
    base.update(overrides)
    return SymbolFeatures(**base)


# ── Hard gates (any one fires ⇒ ∅) ────────────────────────────────────


class TestHardGates:
    def test_earnings_within_2d_forbids_both_sides(self):
        from core.classifier.rule_based import classify_rule_based

        f = _features(earnings_within_2d=True)
        d = classify_rule_based(f)
        assert d.is_forbidden()
        assert "earnings" in d.primary_reason.lower()

    def test_halt_today_forbids_both_sides(self):
        from core.classifier.rule_based import classify_rule_based

        f = _features(halt_today=True)
        d = classify_rule_based(f)
        assert d.is_forbidden()
        assert "halt" in d.primary_reason.lower()

    def test_parabolic_volatility_forbids_both_sides(self):
        """daily_atr_pct > 8% is the parabolic / un-systematic regime."""
        from core.classifier.rule_based import classify_rule_based

        f = _features(daily_atr_pct=0.10, close=10.0)
        d = classify_rule_based(f)
        assert d.is_forbidden()
        assert "atr" in d.primary_reason.lower() or "volatil" in d.primary_reason.lower()

    def test_illiquid_volume_forbids_both_sides(self):
        from core.classifier.rule_based import classify_rule_based

        f = _features(avg_daily_volume_20d=200_000)
        d = classify_rule_based(f)
        assert d.is_forbidden()
        assert "volume" in d.primary_reason.lower() or "liquidity" in d.primary_reason.lower()


# ── Side-only hard gates ──────────────────────────────────────────────


class TestSideOnlyHardGates:
    def test_hard_to_borrow_removes_short(self):
        from core.classifier.rule_based import classify_rule_based
        from core.classifier.types import Side

        # Otherwise neutral features — without HTB, both sides plausible.
        f = _features(hard_to_borrow=True)
        d = classify_rule_based(f)
        assert Side.SHORT not in d.allowed_sides
        # LONG might or might not pass soft scoring; what we assert is
        # that SHORT is explicitly removed regardless of score.

    def test_3plus_lower_lows_removes_long(self):
        """3 consecutive lower-low daily bars = confirmed downtrend;
        no buy-the-dip entries from the rule layer."""
        from core.classifier.rule_based import classify_rule_based
        from core.classifier.types import Side

        f = _features(lower_lows=3)
        d = classify_rule_based(f)
        assert Side.LONG not in d.allowed_sides

    def test_3plus_higher_highs_removes_short(self):
        from core.classifier.rule_based import classify_rule_based
        from core.classifier.types import Side

        f = _features(higher_highs=3)
        d = classify_rule_based(f)
        assert Side.SHORT not in d.allowed_sides

    def test_strong_positive_news_removes_short(self):
        from core.classifier.rule_based import classify_rule_based
        from core.classifier.types import Side

        f = _features(news_sentiment_7d=0.75)
        d = classify_rule_based(f)
        assert Side.SHORT not in d.allowed_sides

    def test_strong_negative_news_removes_long(self):
        from core.classifier.rule_based import classify_rule_based
        from core.classifier.types import Side

        f = _features(news_sentiment_7d=-0.75)
        d = classify_rule_based(f)
        assert Side.LONG not in d.allowed_sides


# ── Soft scoring & end-to-end allowed_sides ───────────────────────────


class TestSoftScoring:
    def test_clean_uptrend_allows_long_only(self):
        """All trend indicators agree: above SMA20 and SMA50, SMA20 > SMA50,
        positive RS, accumulation volume, 2 higher highs (not yet 3 → not
        hard-gating SHORT, but the soft score should still drop SHORT)."""
        from core.classifier.rule_based import classify_rule_based
        from core.classifier.types import Side

        f = _features(
            close=110.0,
            sma_20_daily=105.0,
            sma_50_daily=100.0,
            higher_highs=2,
            up_vs_down_volume_5d=1.8,
            rs_vs_spy_5d=0.03,
            intraday_trend_slope=0.4,
            news_sentiment_7d=0.2,
        )
        d = classify_rule_based(f)
        assert Side.LONG in d.allowed_sides
        assert Side.SHORT not in d.allowed_sides
        assert d.long_score > d.short_score
        assert d.long_score >= 0.55

    def test_clean_downtrend_allows_short_only(self):
        from core.classifier.rule_based import classify_rule_based
        from core.classifier.types import Side

        f = _features(
            close=90.0,
            sma_20_daily=95.0,
            sma_50_daily=100.0,
            lower_lows=2,
            up_vs_down_volume_5d=0.55,
            rs_vs_spy_5d=-0.03,
            intraday_trend_slope=-0.4,
            news_sentiment_7d=-0.2,
        )
        d = classify_rule_based(f)
        assert Side.SHORT in d.allowed_sides
        assert Side.LONG not in d.allowed_sides
        assert d.short_score > d.long_score
        assert d.short_score >= 0.55

    def test_range_bound_allows_both_sides(self):
        """When all trend / RS / volume signals are flat, both sides
        score modestly and both should be allowed."""
        from core.classifier.rule_based import classify_rule_based
        from core.classifier.types import Side

        f = _features(
            close=100.0,
            sma_20_daily=100.0,
            sma_50_daily=100.0,
            higher_highs=0,
            lower_lows=0,
            vol_ratio_today=1.0,
            up_vs_down_volume_5d=1.0,
            rs_vs_spy_5d=0.0,
            intraday_trend_slope=0.0,
            news_sentiment_7d=0.0,
        )
        d = classify_rule_based(f)
        # Either both pass or neither passes — test we don't get LONG-only
        # or SHORT-only out of pure-neutral inputs.
        assert d.allows_long() == d.allows_short()

    def test_scores_in_unit_interval(self):
        from core.classifier.rule_based import classify_rule_based

        f = _features(
            sma_20_daily=110.0, sma_50_daily=100.0,
            up_vs_down_volume_5d=3.0,
        )
        d = classify_rule_based(f)
        assert 0.0 <= d.long_score <= 1.0
        assert 0.0 <= d.short_score <= 1.0


class TestPrimaryReason:
    def test_primary_reason_is_one_line(self):
        from core.classifier.rule_based import classify_rule_based

        f = _features(close=100.0, sma_20_daily=105.0, sma_50_daily=110.0,
                      lower_lows=2)
        d = classify_rule_based(f)
        assert "\n" not in d.primary_reason
        assert len(d.primary_reason) <= 120

    def test_feature_dump_includes_key_inputs(self):
        from core.classifier.rule_based import classify_rule_based

        f = _features(
            sma_20_daily=100.0, sma_50_daily=95.0,
            higher_highs=1, lower_lows=0,
            news_sentiment_7d=0.1,
        )
        d = classify_rule_based(f)
        # Some subset of decisive features must be in the dump for audit.
        assert "long_score" in d.feature_dump or "sma_alignment" in d.feature_dump
