"""Tests for allocators/regime_allocator.py (P1, shadow mode).

Evidence basis: research/regime_allocated_report_2026-06-10.json —
ER-allocated walk-forward PF 1.11 vs baseline 1.04, +188% vs +106%,
stable across ER cuts 0.30-0.40. Shadow-first per the roadmap's
architecture law: this module logs, it does not gate.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
ENGINE_PATH = REPO_ROOT / "core" / "engine.py"
CONFIG_PATH = REPO_ROOT / "core" / "config.py"


class TestEfficiencyRatio:
    def test_straight_trend_is_one(self):
        from allocators.regime_allocator import efficiency_ratio
        closes = [100 + i for i in range(30)]  # perfectly directional
        er = efficiency_ratio(closes, lookback=20)
        assert er is not None and abs(er - 1.0) < 1e-9

    def test_pure_oscillation_is_zero(self):
        from allocators.regime_allocator import efficiency_ratio
        closes = [100 + (i % 2) for i in range(30)]  # 100,101,100,101…
        er = efficiency_ratio(closes, lookback=20)
        assert er is not None and er < 0.06  # net ≈ 0|1 over 20 |1| moves

    def test_insufficient_history_returns_none(self):
        from allocators.regime_allocator import efficiency_ratio
        assert efficiency_ratio([100.0] * 10, lookback=20) is None

    def test_flat_series_is_zero_not_nan(self):
        from allocators.regime_allocator import efficiency_ratio
        er = efficiency_ratio([100.0] * 30, lookback=20)
        assert er == 0.0 and not math.isnan(er)


class TestAllocate:
    def test_trending_tape_allows_breakout_blocks_mean_rev(self):
        from allocators.regime_allocator import allocate
        a = allocate(0.45, threshold=0.30)
        assert a.tape == "trending"
        assert a.breakout_allowed and not a.mean_rev_allowed

    def test_choppy_tape_allows_mean_rev_blocks_breakout(self):
        from allocators.regime_allocator import allocate
        a = allocate(0.12, threshold=0.30)
        assert a.tape == "choppy"
        assert a.mean_rev_allowed and not a.breakout_allowed

    def test_unknown_er_fails_open(self):
        """No ER (insufficient data / fetch failure) must allow both —
        the shadow must never look like it would halt all trading on a
        data hiccup."""
        from allocators.regime_allocator import allocate
        a = allocate(None, threshold=0.30)
        assert a.tape == "unknown"
        assert a.breakout_allowed and a.mean_rev_allowed

    def test_allocation_is_frozen(self):
        from allocators.regime_allocator import allocate
        a = allocate(0.5, threshold=0.30)
        with pytest.raises(Exception):
            a.tape = "x"  # type: ignore[misc]


class TestShadowLedger:
    """evaluate() is async (the SPY fetch runs in a thread executor
    with a timeout so a hung Schwab API can never block the engine's
    event loop — review finding 2026-06-10). Tests drive it with
    asyncio.run."""

    def test_evaluate_writes_ndjson(self, tmp_path):
        import asyncio
        from allocators.regime_allocator import RegimeAllocatorShadow
        ledger = tmp_path / "ledger.ndjson"
        shadow = RegimeAllocatorShadow(
            fetch_daily_closes=lambda: [100 + i for i in range(30)],
            threshold=0.30,
            ledger_path=ledger,
        )
        a = asyncio.run(shadow.evaluate(strategy="breakout", symbol="NVDA",
                                        signal_side="buy"))
        assert a is not None and a.tape == "trending"
        lines = [json.loads(l) for l in ledger.read_text().splitlines()]
        assert len(lines) == 1
        row = lines[0]
        assert row["strategy"] == "breakout"
        assert row["symbol"] == "NVDA"
        assert row["would_allow"] is True
        assert "er" in row and "timestamp" in row

    def test_fetch_failure_is_silent_and_fails_open(self, tmp_path):
        import asyncio
        from allocators.regime_allocator import RegimeAllocatorShadow

        def _boom():
            raise RuntimeError("schwab down")

        shadow = RegimeAllocatorShadow(
            fetch_daily_closes=_boom, threshold=0.30,
            ledger_path=tmp_path / "ledger.ndjson",
        )
        a = asyncio.run(shadow.evaluate(strategy="mean_reversion",
                                        symbol="AMD", signal_side="buy"))
        assert a is not None and a.tape == "unknown"
        assert a.mean_rev_allowed  # fail-open

    def test_er_is_cached_between_evaluations(self, tmp_path):
        import asyncio
        from allocators.regime_allocator import RegimeAllocatorShadow
        calls = {"n": 0}

        def _fetch():
            calls["n"] += 1
            return [100 + i for i in range(30)]

        shadow = RegimeAllocatorShadow(
            fetch_daily_closes=_fetch, threshold=0.30,
            ledger_path=tmp_path / "ledger.ndjson",
        )

        async def _two():
            await shadow.evaluate(strategy="breakout", symbol="A",
                                  signal_side="buy")
            await shadow.evaluate(strategy="breakout", symbol="B",
                                  signal_side="buy")

        asyncio.run(_two())
        assert calls["n"] == 1, "daily closes must be cached, not re-fetched"

    def test_fetch_runs_in_executor_with_timeout(self):
        """Source marker: the fetch must go through run_in_executor +
        wait_for, never directly on the event loop (MRVL 2026-04-29
        class of bug)."""
        src = (REPO_ROOT / "allocators" / "regime_allocator.py").read_text()
        assert "run_in_executor" in src and "wait_for" in src


class TestEngineWiring:
    """Static source-marker assertions, test_recent_fixes.py style."""

    def test_config_flags_exist(self):
        src = CONFIG_PATH.read_text()
        assert "ENABLE_REGIME_ALLOCATOR_SHADOW" in src
        assert "REGIME_ALLOCATOR_ER_THRESHOLD" in src

    def test_engine_builds_shadow_allocator(self):
        src = ENGINE_PATH.read_text()
        assert "_regime_allocator" in src, (
            "engine must hold a regime allocator shadow instance"
        )
        assert "ENABLE_REGIME_ALLOCATOR_SHADOW" in src

    def test_signal_router_has_shadow_hook(self):
        """The hook must sit beside the side-classifier shadow hook and
        be pure-logging (audit), never a gate."""
        src = ENGINE_PATH.read_text()
        anchor = src.find("side_classifier shadow evaluate error")
        assert anchor != -1, "classifier hook anchor missing"
        window = src[anchor: anchor + 3000]
        assert "regime_allocator" in window, (
            "regime allocator shadow hook must follow the classifier "
            "hook in the signal router"
        )
