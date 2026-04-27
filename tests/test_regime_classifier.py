"""Unit tests for ml/regime.py — 7-state classifier with median-split cohorts.

States: trend_up_low_vol, trend_up_high_vol, trend_dn_low_vol,
        trend_dn_high_vol, range_tight, range_wide, chop.

Methodology:
- ADX (Wilder 1978) >25 trending, ≤25 ranging.
- Choppiness Index (Dreiss) >61.8 chop (Fibonacci).
- ema_diff = EMA(50) − EMA(200) — sign indicates trend direction.
- Realized vol = EWMA20 of squared log-returns.
- Trend cohort split: median of realized vol within trending bars over
  the rolling lookback. Ranging cohort: same logic. Ties default to
  the conservative bucket (low_vol / tight).
- No look-ahead: classifier at bar t uses only data ≤ t-1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Synthetic-series builders
# ---------------------------------------------------------------------------

def _ts(n: int, start: str = "2025-01-01", freq: str = "5min") -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=n, freq=freq)


def _ohlc_from_close(close: np.ndarray, vol_pct: float = 0.001
                     ) -> pd.DataFrame:
    """Build OHLC from a close series with small intra-bar wiggle."""
    n = len(close)
    rng = np.random.default_rng(0)
    wiggle = rng.normal(scale=vol_pct, size=n) * close
    high = close + np.abs(wiggle)
    low = close - np.abs(wiggle)
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": np.full(n, 1_000_000.0)},
        index=_ts(n),
    )


def _trend_up_series(n: int, drift: float = 0.0008,
                     noise_scale: float = 0.001, seed: int = 1) -> np.ndarray:
    """Monotone-up close with controlled noise."""
    rng = np.random.default_rng(seed)
    log_ret = drift + rng.normal(scale=noise_scale, size=n)
    return 100.0 * np.exp(np.cumsum(log_ret))


def _trend_dn_series(n: int, drift: float = -0.0008,
                     noise_scale: float = 0.001, seed: int = 2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    log_ret = drift + rng.normal(scale=noise_scale, size=n)
    return 100.0 * np.exp(np.cumsum(log_ret))


def _range_series(n: int, half_width: float = 0.5,
                  noise_scale: float = 0.001, seed: int = 3) -> np.ndarray:
    """Mean-reverting close oscillating around 100."""
    rng = np.random.default_rng(seed)
    centre = 100.0
    levels = np.zeros(n)
    levels[0] = centre
    for i in range(1, n):
        # Strong mean-reversion toward centre.
        pull = -0.05 * (levels[i - 1] - centre)
        levels[i] = levels[i - 1] + pull + rng.normal(scale=noise_scale * centre)
    # Bound oscillation amplitude
    return np.clip(levels, centre - half_width, centre + half_width)


def _chop_series(n: int, step: float = 0.4, band: float = 0.6,
                 seed: int = 4) -> np.ndarray:
    """High-CI chop: large bar-to-bar moves bounded in a tight band.

    CI > 61.8 requires sum(TR) >> H-L range. We achieve this with
    alternating-sign moves of magnitude ``step`` (large TR) bounded
    inside ``±band`` around 100 (small H-L range). The signs are
    randomly perturbed so it isn't purely periodic.
    """
    rng = np.random.default_rng(seed)
    levels = np.zeros(n)
    levels[0] = 100.0
    direction = 1.0
    for i in range(1, n):
        # Flip direction unless near a boundary (to enforce the band).
        if levels[i - 1] > 100.0 + band:
            direction = -1.0
        elif levels[i - 1] < 100.0 - band:
            direction = 1.0
        else:
            # Mostly flip, occasionally hold — produces choppy zig-zag.
            if rng.uniform() < 0.85:
                direction *= -1.0
        jitter = rng.normal(scale=0.05)
        levels[i] = levels[i - 1] + direction * step + jitter
        # Hard clamp.
        levels[i] = np.clip(levels[i], 100.0 - band, 100.0 + band)
    return levels


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestComputeRegimeFeatures:
    def test_features_strict_no_lookahead(self) -> None:
        """compute_regime_features at bar t must use only data <= t-1."""
        from ml.regime import compute_regime_features
        n = 600
        close_full = _trend_up_series(n)
        df_full = _ohlc_from_close(close_full)

        feats_full = compute_regime_features(df_full)

        # Recompute on a truncated window ending at index t.
        for t in [300, 400, 500]:
            df_trunc = df_full.iloc[: t + 1]
            feats_trunc = compute_regime_features(df_trunc)
            # The features at the last bar of the truncated frame must equal
            # those at the same bar in the full frame (within float tolerance).
            for col in ["adx", "ema_diff", "ci", "rv"]:
                v_full = feats_full[col].iloc[t]
                v_trunc = feats_trunc[col].iloc[t]
                if pd.isna(v_full) and pd.isna(v_trunc):
                    continue
                assert v_full == pytest.approx(v_trunc, rel=1e-9, abs=1e-9), (
                    f"col={col} t={t} full={v_full} trunc={v_trunc}"
                )


class TestClassifyRegime:
    """Tests for classify_regime over synthetic single-state series."""

    def _make_features_for(self, close: np.ndarray) -> pd.DataFrame:
        from ml.regime import compute_regime_features
        df = _ohlc_from_close(close)
        return compute_regime_features(df)

    def test_classify_pure_uptrend_yields_trend_up(self) -> None:
        from ml.regime import classify_regime
        feats = self._make_features_for(_trend_up_series(600))
        # Skip the warmup region; classifier on the last bar should be up-trend.
        label = classify_regime(feats.iloc[-1])
        assert label.startswith("trend_up_"), f"got {label}"

    def test_classify_pure_downtrend_yields_trend_dn(self) -> None:
        from ml.regime import classify_regime
        feats = self._make_features_for(_trend_dn_series(600))
        label = classify_regime(feats.iloc[-1])
        assert label.startswith("trend_dn_"), f"got {label}"

    def test_classify_tight_range_yields_range(self) -> None:
        from ml.regime import classify_regime
        feats = self._make_features_for(_range_series(600))
        label = classify_regime(feats.iloc[-1])
        assert label.startswith("range_") or label == "chop", f"got {label}"

    def test_classify_chop_yields_chop(self) -> None:
        from ml.regime import classify_regime
        feats = self._make_features_for(_chop_series(600))
        label = classify_regime(feats.iloc[-1])
        # Random walk produces high CI → chop.
        assert label == "chop", f"got {label}"

    def test_classify_insufficient_history_yields_chop(self) -> None:
        """compute_regime_features at the head of a short series produces NaNs;
        classify_regime on a NaN row returns chop."""
        from ml.regime import classify_regime, compute_regime_features
        n = 30
        df = _ohlc_from_close(_trend_up_series(n))
        feats = compute_regime_features(df)
        # First few rows have NaN ADX/EMA → chop.
        label_early = classify_regime(feats.iloc[5])
        assert label_early == "chop", f"got {label_early}"

    def test_classify_chop_priority_over_trend(self) -> None:
        """If CI > 61.8 AND ADX > 25, classifier returns chop."""
        from ml.regime import classify_regime
        # Synthetic feature row.
        row = pd.Series({
            "adx": 30.0,         # trending threshold
            "ema_diff": 1.0,     # up
            "ci": 70.0,          # > 61.8 → chop
            "rv": 0.001,
            "rv_trend_median": 0.001,
            "rv_range_median": 0.001,
        })
        assert classify_regime(row) == "chop"

    def test_classify_adx_boundary_24_99_vs_25_01(self) -> None:
        """ADX = 25.01 → trending; ADX = 24.99 → range."""
        from ml.regime import classify_regime
        base = {"ema_diff": 1.0, "ci": 50.0, "rv": 0.001,
                "rv_trend_median": 0.001, "rv_range_median": 0.001}
        trend_row = pd.Series({**base, "adx": 25.01})
        range_row = pd.Series({**base, "adx": 24.99})
        assert classify_regime(trend_row).startswith("trend_")
        assert classify_regime(range_row).startswith("range_")

    def test_classify_deterministic(self) -> None:
        from ml.regime import classify_regime
        row = pd.Series({
            "adx": 30.0, "ema_diff": 1.0, "ci": 50.0,
            "rv": 0.002, "rv_trend_median": 0.001, "rv_range_median": 0.001,
        })
        labels = [classify_regime(row) for _ in range(50)]
        assert all(l == labels[0] for l in labels)


class TestMedianSplitCohorts:
    """The split between low_vol/high_vol (trend) and tight/wide (range)
    is the rolling median of realized vol within each cohort."""

    def test_trend_high_vol_above_median(self) -> None:
        """rv > rv_trend_median → high_vol bucket."""
        from ml.regime import classify_regime
        row = pd.Series({
            "adx": 30.0, "ema_diff": 1.0, "ci": 50.0,
            "rv": 0.005, "rv_trend_median": 0.001,
            "rv_range_median": 0.0015,
        })
        assert classify_regime(row) == "trend_up_high_vol"

    def test_trend_low_vol_below_median(self) -> None:
        """rv < rv_trend_median → low_vol bucket."""
        from ml.regime import classify_regime
        row = pd.Series({
            "adx": 30.0, "ema_diff": 1.0, "ci": 50.0,
            "rv": 0.0005, "rv_trend_median": 0.001,
            "rv_range_median": 0.0015,
        })
        assert classify_regime(row) == "trend_up_low_vol"

    def test_trend_at_median_defaults_to_low_vol(self) -> None:
        """Tie at exactly rv_trend_median → low_vol (conservative)."""
        from ml.regime import classify_regime
        row = pd.Series({
            "adx": 30.0, "ema_diff": 1.0, "ci": 50.0,
            "rv": 0.001, "rv_trend_median": 0.001,
            "rv_range_median": 0.0015,
        })
        assert classify_regime(row) == "trend_up_low_vol"

    def test_trend_dn_high_vol(self) -> None:
        from ml.regime import classify_regime
        row = pd.Series({
            "adx": 30.0, "ema_diff": -1.0, "ci": 50.0,
            "rv": 0.005, "rv_trend_median": 0.001,
            "rv_range_median": 0.0015,
        })
        assert classify_regime(row) == "trend_dn_high_vol"

    def test_trend_dn_low_vol(self) -> None:
        from ml.regime import classify_regime
        row = pd.Series({
            "adx": 30.0, "ema_diff": -1.0, "ci": 50.0,
            "rv": 0.0005, "rv_trend_median": 0.001,
            "rv_range_median": 0.0015,
        })
        assert classify_regime(row) == "trend_dn_low_vol"

    def test_range_wide_above_median(self) -> None:
        from ml.regime import classify_regime
        row = pd.Series({
            "adx": 15.0, "ema_diff": 0.1, "ci": 50.0,
            "rv": 0.005, "rv_trend_median": 0.001,
            "rv_range_median": 0.001,
        })
        assert classify_regime(row) == "range_wide"

    def test_range_tight_below_median(self) -> None:
        from ml.regime import classify_regime
        row = pd.Series({
            "adx": 15.0, "ema_diff": -0.1, "ci": 50.0,
            "rv": 0.0005, "rv_trend_median": 0.001,
            "rv_range_median": 0.001,
        })
        assert classify_regime(row) == "range_tight"

    def test_range_at_median_defaults_to_tight(self) -> None:
        from ml.regime import classify_regime
        row = pd.Series({
            "adx": 15.0, "ema_diff": 0.1, "ci": 50.0,
            "rv": 0.001, "rv_trend_median": 0.001,
            "rv_range_median": 0.001,
        })
        assert classify_regime(row) == "range_tight"


class TestSevenStateUniverse:
    """Confirm classify_regime never produces a label outside the 7-state set."""
    EXPECTED_STATES = {
        "trend_up_low_vol", "trend_up_high_vol",
        "trend_dn_low_vol", "trend_dn_high_vol",
        "range_tight", "range_wide", "chop",
    }

    def test_all_outputs_in_universe(self) -> None:
        from ml.regime import classify_regime, compute_regime_features
        # Mix of all kinds of series concatenated into one frame.
        close = np.concatenate([
            _trend_up_series(300, seed=11),
            _trend_dn_series(300, seed=12),
            _range_series(300, seed=13),
            _chop_series(300, seed=14),
        ])
        feats = compute_regime_features(_ohlc_from_close(close))
        labels = feats.iloc[300:].apply(classify_regime, axis=1)
        assert set(labels.unique()) <= self.EXPECTED_STATES, (
            f"unexpected labels: {set(labels.unique()) - self.EXPECTED_STATES}"
        )
