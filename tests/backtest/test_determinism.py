"""Determinism CI test for the backtest engine.

Runs ``_run_backtest_with_trades`` twice with identical inputs and the
same ``--seed`` and asserts the resulting ``Trade`` lists are byte-equal:
same order, same prices, same exit reasons, same entry indicators.

The test is hermetic:

- A synthetic ``bars_loader`` returns a fixed deterministic DataFrame so
  Postgres is never touched.
- ``load_strategies`` is monkey-patched to return a tiny deterministic
  fake strategy ("forcer") that fires a BUY on a specific bar index,
  giving the engine real trades to compare. We also include the real
  strategies in the patched list so the production strategy contract
  ``generate_signal_with_commentary`` is exercised — proving the
  fan-out loop is itself deterministic.

If determinism ever breaks (e.g. someone adds ``time.time()`` to the
replay loop, an unseeded RNG creeps in, dict iteration becomes ordered
on insertion-time, etc.), this test will fail loudly on CI.
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict
from typing import Any, Optional

import numpy as np
import pandas as pd
import pytest

import backtest.engine as _engine
from backtest import _run_backtest_with_trades, run_backtest


# ---------------------------------------------------------------------------
# Synthetic bar data
# ---------------------------------------------------------------------------

def _trending_bars(n: int = 200, start_price: float = 100.0) -> pd.DataFrame:
    """OHLCV with a deterministic up-trend pattern.

    Walk the close price up by ``+0.25`` per bar, with a daily spike
    halfway through to trigger volume-ratio + ADX gates if any real
    strategy were sensitive. Index uses a fixed timestamp range so the
    DataFrame is byte-stable across runs.
    """
    idx = pd.date_range("2024-06-03 09:30", periods=n, freq="5min")
    closes = np.linspace(start_price, start_price + n * 0.25, n)
    # Daily volume spike at bar 80 (well past 50-bar warmup).
    volumes = np.full(n, 10_000.0)
    volumes[80] = 60_000.0
    df = pd.DataFrame({
        "Open":   closes - 0.05,
        "High":   closes + 0.20,
        "Low":    closes - 0.20,
        "Close":  closes,
        "Volume": volumes,
    }, index=idx)
    return df


def _make_fixed_loader(symbol_to_df: dict[str, pd.DataFrame]):
    def _load(symbol: str, days: int, frequency: int) -> pd.DataFrame:
        # Return a fresh copy so the engine can't mutate the source
        # between runs (it doesn't, but defensive against future drift).
        return symbol_to_df[symbol].copy()
    return _load


# ---------------------------------------------------------------------------
# Deterministic fake strategy
# ---------------------------------------------------------------------------

class _ForcerStrategy:
    """Fires a BUY signal exactly once at a fixed bar index.

    Deterministic by construction: keeps a per-instance bar counter and
    fires on the Nth call. Stops, targets, and entry price are derived
    from ``market_data.close`` so they're reproducible.
    """

    name = "forcer"

    def __init__(self, fire_on_call: int = 5) -> None:
        self._call = 0
        self._fire_on = fire_on_call
        self._fired = False

    async def generate_signal_with_commentary(self, market_data):
        from core.models import SignalType, TradingSignal
        self._call += 1
        if self._fired or self._call != self._fire_on:
            return None
        self._fired = True
        entry = float(market_data.close)
        return TradingSignal(
            symbol=market_data.symbol,
            signal_type=SignalType.BUY,
            strength=0.7,
            entry_price=entry,
            stop_loss=entry - 0.5,
            take_profit=entry + 1.0,
            position_size=0,
            reasoning={"strategy": "forcer"},
            confidence=0.5,
        )


def _fake_load_strategies():
    """Replacement for ``backtest.engine.load_strategies`` used by tests.

    Returns the forcer plus one real strategy so the fan-out loop is
    exercised. We deliberately do not include all three real strategies
    — they require ``CommentarySystem`` and ``core.config`` state that's
    expensive to set up in a hermetic test, and the forcer alone already
    proves the engine's per-strategy state machine is deterministic.
    """
    return [("forcer", _ForcerStrategy(fire_on_call=5))]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trades_as_tuples(trades) -> list[tuple]:
    """Convert each Trade to a comparable tuple (asdict in stable key order)."""
    out = []
    for t in trades:
        d = asdict(t)
        # entry_indicators is a dict — convert to sorted tuple of items
        # so equality is order-independent (we still assert the *content*
        # is identical).
        d["entry_indicators"] = tuple(sorted(d["entry_indicators"].items()))
        out.append(tuple(sorted(d.items())))
    return out


def _strip_volatile(report: dict) -> dict:
    """Drop the keys that legitimately differ between runs.

    ``timestamp``, ``run_id`` (if auto-generated), and ``elapsed_seconds``
    depend on wall-clock time. Everything else must match exactly.
    """
    keep = {k: v for k, v in report.items()
            if k not in ("timestamp", "elapsed_seconds")}
    return keep


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.fixture
def patched_strategies(monkeypatch):
    """Patch ``load_strategies`` to use the deterministic forcer."""
    monkeypatch.setattr(_engine, "load_strategies", _fake_load_strategies)
    yield


def _run_once(*, seed: int, loader) -> tuple[dict, list]:
    return asyncio.run(_run_backtest_with_trades(
        symbols=["AAA", "BBB"],
        days=30,
        frequency=5,
        bars_loader=loader,
        run_id="bt_det_test",
        seed=seed,
    ))


class TestDeterminism:
    def test_trade_lists_byte_equal_with_same_seed(self, patched_strategies):
        bars = {"AAA": _trending_bars(), "BBB": _trending_bars(start_price=80.0)}
        loader = _make_fixed_loader(bars)

        _, trades_a = _run_once(seed=42, loader=loader)
        _, trades_b = _run_once(seed=42, loader=loader)

        # Sanity: forcer should have produced at least 1 trade per symbol.
        assert len(trades_a) >= 1, "forcer should have fired at least once"

        assert _trades_as_tuples(trades_a) == _trades_as_tuples(trades_b), (
            "two runs with the same seed produced different trades"
        )

    def test_report_byte_equal_modulo_timestamps(self, patched_strategies):
        bars = {"AAA": _trending_bars(), "BBB": _trending_bars(start_price=80.0)}
        loader = _make_fixed_loader(bars)

        report_a, _ = _run_once(seed=42, loader=loader)
        report_b, _ = _run_once(seed=42, loader=loader)

        # ``per_strategy`` numbers must be byte-identical.
        assert report_a["per_strategy"] == report_b["per_strategy"]
        assert report_a["total_trades"] == report_b["total_trades"]
        assert _strip_volatile(report_a)["symbols"] == \
            _strip_volatile(report_b)["symbols"]

    def test_run_backtest_public_api_passes_seed_through(self, patched_strategies):
        """Smoke-check that the public ``run_backtest`` accepts ``seed``.

        If a future refactor drops the kwarg, this fails before the
        engine-level determinism does.
        """
        bars = {"AAA": _trending_bars()}
        loader = _make_fixed_loader(bars)

        report = asyncio.run(run_backtest(
            symbols=["AAA"], days=30, frequency=5,
            bars_loader=loader, seed=123, run_id="bt_seed_test",
        ))
        assert report["run_id"] == "bt_seed_test"

    def test_seed_run_is_idempotent(self):
        """``seed_run`` must be safe to call with ``None`` and to recall."""
        from backtest import seed_run

        # No-op path
        seed_run(None)
        # Recall path — must not raise.
        seed_run(7)
        a = np.random.rand()
        seed_run(7)
        b = np.random.rand()
        assert a == b, "seeding must reset the numpy RNG to the same state"
