"""Tests for ``ml/bars.py`` (López de Prado AFML §2.3).

Time bars are statistically awful for ML — autocorrelated, non-IID returns,
heteroskedastic intraday seasonality. Information-driven bars (dollar /
volume / imbalance) sample at constant *information* rather than constant
*time*, producing better statistical properties.

Today the data pipeline emits 5-min OHLCV; we cannot synthesize true dollar
bars from that. The module exposes the API surface so downstream callers
can switch when tick data lands, but raises ``NotImplementedError`` with a
clear message instead of silently faking dollar bars from coarse data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.bars import dollar_bars, imbalance_bars, time_bars, volume_bars


def _ohlcv(n: int, freq: str = "5min") -> pd.DataFrame:
    idx = pd.date_range("2024-01-02 09:30", periods=n, freq=freq)
    rng = np.random.default_rng(0)
    close = 100 + rng.normal(0, 0.5, size=n).cumsum()
    high = close + rng.uniform(0.05, 0.2, size=n)
    low = close - rng.uniform(0.05, 0.2, size=n)
    op = close + rng.normal(0, 0.05, size=n)
    vol = rng.uniform(1000, 5000, size=n)
    return pd.DataFrame(
        {"Open": op, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=idx,
    )


class TestTimeBars:
    def test_passthrough_when_frequency_matches(self):
        df = _ohlcv(20, freq="5min")
        out = time_bars(df, frequency_min=5)

        assert len(out) == len(df)
        assert list(out.columns) == list(df.columns)


class TestDeferredBarTypes:
    def test_dollar_bars_raises_not_implemented(self):
        df = _ohlcv(20)
        with pytest.raises(NotImplementedError, match="tick"):
            dollar_bars(df, threshold=1e6)

    def test_volume_bars_raises_not_implemented(self):
        df = _ohlcv(20)
        with pytest.raises(NotImplementedError, match="tick"):
            volume_bars(df, threshold=1e4)

    def test_imbalance_bars_raises_not_implemented(self):
        df = _ohlcv(20)
        with pytest.raises(NotImplementedError, match="tick"):
            imbalance_bars(df, theta_threshold=1.0)
