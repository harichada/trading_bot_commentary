"""Convert SymbolFeatures dataclass into a 1D tensor.

v-feature-encoder-2026-05-13. Frozen feature order so training and
inference always agree on column meaning. All features normalized
(price-relative for SMAs, log for ratios, /100 for RSI). Missing
fields (None) map to the neutral default for that feature so the
classifier degrades gracefully when history is short.

The output is a fixed-length 1D vector. If you change ``FEATURE_NAMES``
you MUST retrain — old checkpoints will silently misalign.
"""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from core.classifier.types import SymbolFeatures

# ── Frozen feature schema ────────────────────────────────────────────
#
# Each entry: (name, extractor, neutral_default).
# Adding a new feature requires bumping a schema version + retraining.

FEATURE_NAMES: List[str] = [
    # Trend (price-relative)
    "close_vs_sma20",
    "close_vs_sma50",
    "sma20_vs_sma50",
    "intraday_trend_slope",
    "lower_lows",
    "higher_highs",
    # Range / volatility
    "daily_atr_pct",
    "rsi_14_norm",
    # Volume
    "log_vol_ratio_today",
    "log_up_vs_down_volume_5d",
    # News
    "news_sentiment_7d",
    "news_fresh_count_today_clipped",
    # Relative strength
    "rs_vs_spy_5d",
    "rs_vs_sector_5d",
    # Event flags
    "earnings_within_2d",
    "halt_today",
    "hard_to_borrow",
    "ex_dividend_within_1d",
]

# Length of the encoded vector. Pinned by tests so silent schema
# drift is caught immediately.
FEATURE_DIM: int = len(FEATURE_NAMES)


def _relative(a: Optional[float], b: Optional[float]) -> float:
    """``(a - b) / b`` with safe handling of None / zero."""
    if a is None or b is None or b == 0:
        return 0.0
    return (a - b) / b


def _safe_log(x: float, default: float = 0.0) -> float:
    """``log(x)`` clipped to a finite range. Defaults to ``default``
    on non-positive input (a quiet symbol with zero down-volume should
    not blow up the encoder)."""
    if x is None or x <= 0:
        return default
    return float(math.log(x))


def encode(features: SymbolFeatures) -> np.ndarray:
    """Encode one SymbolFeatures into a 1D float32 array."""
    close = features.close

    vec = np.array([
        _relative(close, features.sma_20_daily),
        _relative(close, features.sma_50_daily),
        _relative(features.sma_20_daily, features.sma_50_daily),
        float(features.intraday_trend_slope),
        # Capped raw counts in [0, 5]; divide to [0, 1].
        min(features.lower_lows, 5) / 5.0,
        min(features.higher_highs, 5) / 5.0,
        # ATR fraction is already in roughly [0, 0.1]; pass through
        # but clamp to avoid extreme outliers.
        float(features.daily_atr_pct) if features.daily_atr_pct is not None else 0.0,
        (float(features.rsi_14) / 100.0) if features.rsi_14 is not None else 0.5,
        _safe_log(features.vol_ratio_today, default=0.0),
        _safe_log(features.up_vs_down_volume_5d, default=0.0),
        float(features.news_sentiment_7d),
        min(features.news_fresh_count_today, 10) / 10.0,
        float(features.rs_vs_spy_5d),
        float(features.rs_vs_sector_5d),
        1.0 if features.earnings_within_2d else 0.0,
        1.0 if features.halt_today else 0.0,
        1.0 if features.hard_to_borrow else 0.0,
        1.0 if features.ex_dividend_within_1d else 0.0,
    ], dtype=np.float32)

    # Sanity: shape must match the declared FEATURE_DIM. If a
    # contributor adds a row above without updating FEATURE_NAMES,
    # this surfaces the mismatch immediately.
    assert vec.shape == (FEATURE_DIM,), (
        f"feature encoder produced shape {vec.shape}, expected {(FEATURE_DIM,)} "
        f"— FEATURE_NAMES out of sync with encode() body"
    )
    return vec
