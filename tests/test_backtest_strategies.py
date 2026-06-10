"""Unit tests for backtest_strategies.run_backtest.

Tests use a synthetic in-process bars_loader so they don't touch Postgres.
Strategy-rule correctness is covered separately by test_strategy_replay.py;
these tests focus on the run_backtest contract: shape of the report,
progress_cb behaviour, validation, and dataclass immutability.
"""
from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from backtest_strategies import (
    ProgressEvent,
    SymbolMetrics,
    run_backtest,
    summarize_per_symbol,
    Trade,
)


def _flat_bars(n: int = 200, price: float = 100.0) -> pd.DataFrame:
    """OHLCV with flat prices — no strategy will fire signals.

    Used for shape tests where we don't want trades cluttering the report.
    """
    idx = pd.date_range("2024-06-03 09:30", periods=n, freq="5min")
    return pd.DataFrame({
        "Open": [price] * n,
        "High": [price] * n,
        "Low":  [price] * n,
        "Close": [price] * n,
        "Volume": [10_000.0] * n,
    }, index=idx)


def _make_loader(symbol_to_df: dict[str, pd.DataFrame]):
    """Return a bars_loader callable backed by an in-memory dict."""
    def _load(symbol: str, days: int, frequency: int) -> pd.DataFrame:
        return symbol_to_df.get(symbol, pd.DataFrame())
    return _load


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_requires_exactly_one_of_symbols_or_top_n(self):
        with pytest.raises(ValueError, match="exactly one"):
            asyncio.run(run_backtest(days=30, frequency=5))
        with pytest.raises(ValueError, match="exactly one"):
            asyncio.run(run_backtest(symbols=["AAA"], top_n=5, days=30, frequency=5))

    def test_rejects_bad_days(self):
        with pytest.raises(ValueError, match="days"):
            asyncio.run(run_backtest(symbols=["AAA"], days=0, frequency=5,
                                     bars_loader=_make_loader({})))
        with pytest.raises(ValueError, match="days"):
            asyncio.run(run_backtest(symbols=["AAA"], days=400, frequency=5,
                                     bars_loader=_make_loader({})))

    def test_rejects_bad_frequency(self):
        with pytest.raises(ValueError, match="frequency"):
            asyncio.run(run_backtest(symbols=["AAA"], days=30, frequency=7,
                                     bars_loader=_make_loader({})))


# ---------------------------------------------------------------------------
# Report shape
# ---------------------------------------------------------------------------

class TestRunBacktestShape:
    def test_returns_expected_top_level_keys(self):
        loader = _make_loader({"AAA": _flat_bars(), "BBB": _flat_bars()})
        report = asyncio.run(run_backtest(
            symbols=["AAA", "BBB"], days=30, frequency=5,
            bars_loader=loader, run_id="bt_test_1",
        ))

        for key in ("run_id", "timestamp", "symbols", "symbol_count",
                    "days", "frequency_minutes", "total_trades",
                    "elapsed_seconds", "per_strategy"):
            assert key in report, f"missing key: {key}"

        assert report["run_id"] == "bt_test_1"
        assert report["symbol_count"] == 2
        assert report["symbols"] == ["AAA", "BBB"]
        assert report["days"] == 30
        assert report["frequency_minutes"] == 5

    def test_per_strategy_has_three_known_strategies(self):
        loader = _make_loader({"AAA": _flat_bars()})
        report = asyncio.run(run_backtest(
            symbols=["AAA"], days=30, frequency=5, bars_loader=loader,
        ))
        assert set(report["per_strategy"].keys()) == {
            "breakout", "mean_reversion", "momentum",
        }

    def test_per_strategy_has_all_long_short_breakdown(self):
        loader = _make_loader({"AAA": _flat_bars()})
        report = asyncio.run(run_backtest(
            symbols=["AAA"], days=30, frequency=5, bars_loader=loader,
        ))
        for sides in report["per_strategy"].values():
            assert set(sides.keys()) == {"all", "long", "short"}

    def test_skips_symbols_with_too_few_bars(self):
        # 50 bars < 100 threshold → loader call succeeds but symbol is skipped.
        # symbol_count still reflects what was requested.
        loader = _make_loader({"AAA": _flat_bars(50)})
        report = asyncio.run(run_backtest(
            symbols=["AAA"], days=30, frequency=5, bars_loader=loader,
        ))
        assert report["total_trades"] == 0
        # symbol_count is "what we tried" not "what we processed"
        assert report["symbol_count"] == 1


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------

class TestProgressCallback:
    def test_progress_cb_called_with_monotonic_symbols_done(self):
        loader = _make_loader({s: _flat_bars() for s in ["AAA", "BBB", "CCC"]})
        events: list[ProgressEvent] = []

        def cb(event: ProgressEvent) -> None:
            events.append(event)

        asyncio.run(run_backtest(
            symbols=["AAA", "BBB", "CCC"], days=30, frequency=5,
            bars_loader=loader, progress_cb=cb,
        ))

        # symbols_done is monotonically non-decreasing
        progressions = [e.symbols_done for e in events]
        assert progressions == sorted(progressions)
        # final beat reaches the total
        assert any(e.symbols_done == 3 for e in events)
        # both stages observed
        stages = {e.stage for e in events}
        assert "loading_data" in stages
        assert "replaying" in stages

    def test_async_progress_cb_is_awaited(self):
        loader = _make_loader({"AAA": _flat_bars()})
        seen: list[str] = []

        async def cb(event: ProgressEvent) -> None:
            await asyncio.sleep(0)  # forces coroutine path
            seen.append(event.current_symbol)

        asyncio.run(run_backtest(
            symbols=["AAA"], days=30, frequency=5,
            bars_loader=loader, progress_cb=cb,
        ))
        assert "AAA" in seen


# ---------------------------------------------------------------------------
# Per-symbol drill-down + immutability
# ---------------------------------------------------------------------------

class TestPerSymbol:
    def test_summarize_per_symbol_groups_correctly(self):
        # Build trades by hand — bypass the strategy code path.
        ts = pd.Timestamp("2024-06-03 09:30")
        trades = [
            Trade("breakout", "AAA", "long",  ts, 100.0, ts, 102.0, "target", 5),
            Trade("breakout", "AAA", "long",  ts, 100.0, ts, 99.0,  "stop",   3),
            Trade("breakout", "BBB", "short", ts, 50.0,  ts, 49.0,  "target", 4),
        ]
        by_symbol = summarize_per_symbol(trades)
        assert set(by_symbol.keys()) == {"AAA", "BBB"}
        assert by_symbol["AAA"]["n_trades"] == 2
        assert by_symbol["BBB"]["n_trades"] == 1
        # AAA: one +2% win, one -1% loss → win_rate 50%
        assert by_symbol["AAA"]["win_rate_pct"] == 50.0

    def test_summarize_per_symbol_empty(self):
        assert summarize_per_symbol([]) == {}

    def test_symbol_metrics_is_frozen(self):
        m = SymbolMetrics(n_trades=1, win_rate_pct=100.0,
                          sum_return_pct=2.0, avg_return_pct=2.0)
        with pytest.raises(FrozenInstanceError):
            m.n_trades = 999  # type: ignore[misc]


# ── v-bt-direction-features-2026-06-10 ───────────────────────────────

class TestDirectionReaderFeaturesInReplay:
    """The 2026-05-29 direction gate (breakout + mean-rev) reads five
    features that live analysis/technical.py computes but the backtest
    compute_indicators never provided: ema_20, macd_histogram,
    ema_20_slope_pct, close_vs_sma50_pct, obv_slope_pct. With all five
    defaulting to 0.0, read_direction can never reach the +3.0 the
    breakout gate requires — the 2026-06-09 walk-forward showed the
    gate rejecting 1016/1016 candidates that had passed every other
    gate. Replay must provide the same features live provides."""

    def _frame(self, n=120):
        import numpy as np
        import pandas as pd
        idx = pd.date_range("2026-01-05 09:30", periods=n, freq="5min")
        base = np.linspace(100.0, 110.0, n)  # steady uptrend
        return pd.DataFrame({
            "Open": base, "High": base + 0.5, "Low": base - 0.5,
            "Close": base + 0.1,
            "Volume": np.full(n, 10_000.0),
        }, index=idx)

    def test_direction_features_present(self):
        from backtest.engine import compute_indicators
        ind = compute_indicators(self._frame())
        for col in ("ema_20", "macd_histogram", "ema_20_slope_pct",
                    "close_vs_sma50_pct", "obv_slope_pct"):
            assert col in ind.columns, f"replay missing {col}"

    def test_uptrend_yields_positive_direction(self):
        """In a clean uptrend the direction reader must be able to
        clear the breakout gate's +3.0 — proves the features carry
        real signal in replay, not zeros."""
        from backtest.engine import compute_indicators
        from core.direction_reader import read_direction
        df = self._frame()
        ind = compute_indicators(df)
        row = ind.iloc[-1].to_dict()
        dr = read_direction(float(df["Close"].iloc[-1]), row)
        assert dr.direction >= 3.0, (
            f"clean uptrend read direction={dr.direction} (<3.0) — "
            f"components={dr.components}"
        )

    def test_slope_matches_live_formula(self):
        """ema_20_slope_pct must equal the live 5-bar formula from
        analysis/technical.py: (ema[-1]-ema[-6])/ema[-6]*100."""
        import ta as _ta
        from backtest.engine import compute_indicators
        df = self._frame()
        ind = compute_indicators(df)
        ema = _ta.trend.ema_indicator(df["Close"], window=20)
        expected = (ema.iloc[-1] - ema.iloc[-6]) / ema.iloc[-6] * 100
        got = ind["ema_20_slope_pct"].iloc[-1]
        assert abs(got - expected) < 1e-9, f"{got} != {expected}"
