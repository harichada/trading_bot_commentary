"""v-feature-encoder-2026-05-13: tests for the SymbolFeatures → 1D
vector encoding.

The schema is frozen at training time and the checkpoint silently
misaligns if it changes. These tests pin the dimension, the order
of fields, and the neutral-default behavior on missing inputs.
"""
from __future__ import annotations

import math

import numpy as np
import pytest


def _features(**overrides):
    from core.classifier.types import SymbolFeatures

    base = dict(symbol="TEST", close=100.0)
    base.update(overrides)
    return SymbolFeatures(**base)


class TestEncodeShape:
    def test_dim_matches_feature_names(self):
        from core.classifier.trained.feature_encoder import (
            FEATURE_DIM, FEATURE_NAMES,
        )
        assert FEATURE_DIM == len(FEATURE_NAMES)

    def test_dim_is_18(self):
        """Schema lock — adding/removing a feature requires a deliberate
        bump here AND retraining. Don't change without intent."""
        from core.classifier.trained.feature_encoder import FEATURE_DIM
        assert FEATURE_DIM == 18

    def test_output_shape(self):
        from core.classifier.trained.feature_encoder import FEATURE_DIM, encode
        f = _features()
        v = encode(f)
        assert v.shape == (FEATURE_DIM,)
        assert v.dtype == np.float32


class TestEncodeNeutralDefaults:
    def test_missing_sma_yields_zero_relative(self):
        from core.classifier.trained.feature_encoder import encode, FEATURE_NAMES
        f = _features()   # sma_20_daily and sma_50_daily are None
        v = encode(f)
        i_20 = FEATURE_NAMES.index("close_vs_sma20")
        i_50 = FEATURE_NAMES.index("close_vs_sma50")
        i_alignment = FEATURE_NAMES.index("sma20_vs_sma50")
        assert v[i_20] == 0.0
        assert v[i_50] == 0.0
        assert v[i_alignment] == 0.0

    def test_missing_rsi_yields_half(self):
        """RSI defaults to 50/100 = 0.5 — true neutral."""
        from core.classifier.trained.feature_encoder import encode, FEATURE_NAMES
        f = _features(rsi_14=None)
        v = encode(f)
        i_rsi = FEATURE_NAMES.index("rsi_14_norm")
        assert v[i_rsi] == 0.5

    def test_missing_atr_yields_zero(self):
        from core.classifier.trained.feature_encoder import encode, FEATURE_NAMES
        f = _features(daily_atr_pct=None)
        v = encode(f)
        i_atr = FEATURE_NAMES.index("daily_atr_pct")
        assert v[i_atr] == 0.0

    def test_zero_volume_ratio_safe(self):
        """log(0) would blow up; encoder must clamp."""
        from core.classifier.trained.feature_encoder import encode
        f = _features(vol_ratio_today=0.0, up_vs_down_volume_5d=0.0)
        v = encode(f)
        assert np.all(np.isfinite(v))


class TestEncodeKnownValues:
    def test_uptrend_features_are_positive(self):
        """Close above SMAs, SMA20 above SMA50, accumulating volume,
        positive RS → trend-aligned features should be positive."""
        from core.classifier.trained.feature_encoder import encode, FEATURE_NAMES
        f = _features(
            close=110.0,
            sma_20_daily=105.0,
            sma_50_daily=100.0,
            up_vs_down_volume_5d=2.0,
            rs_vs_spy_5d=0.04,
            news_sentiment_7d=0.3,
        )
        v = encode(f)
        for fname in ("close_vs_sma20", "close_vs_sma50", "sma20_vs_sma50",
                      "log_up_vs_down_volume_5d", "rs_vs_spy_5d",
                      "news_sentiment_7d"):
            i = FEATURE_NAMES.index(fname)
            assert v[i] > 0, f"{fname} should be positive in uptrend, got {v[i]}"

    def test_boolean_flags_are_zero_or_one(self):
        from core.classifier.trained.feature_encoder import encode, FEATURE_NAMES
        f_no_event = _features()
        f_event = _features(earnings_within_2d=True, halt_today=True,
                            hard_to_borrow=True, ex_dividend_within_1d=True)
        v_no = encode(f_no_event)
        v_yes = encode(f_event)
        for fname in ("earnings_within_2d", "halt_today",
                      "hard_to_borrow", "ex_dividend_within_1d"):
            i = FEATURE_NAMES.index(fname)
            assert v_no[i] == 0.0
            assert v_yes[i] == 1.0

    def test_log_volume_ratio_is_signed_log(self):
        from core.classifier.trained.feature_encoder import encode, FEATURE_NAMES
        f = _features(vol_ratio_today=math.e)   # log(e) = 1.0
        v = encode(f)
        i = FEATURE_NAMES.index("log_vol_ratio_today")
        assert v[i] == pytest.approx(1.0, abs=1e-5)

        f2 = _features(vol_ratio_today=1.0)   # log(1) = 0.0
        v2 = encode(f2)
        assert v2[i] == pytest.approx(0.0, abs=1e-5)
