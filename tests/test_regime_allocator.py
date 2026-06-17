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


# ── v-regime-gate-live-meanrev-2026-06-16 ────────────────────────────

class TestLiveMeanRevGate:
    """Promotes the allocator from shadow to a LIVE veto on mean-rev
    signals in trending tape (ER >= threshold). Evidence:
    research/meanrev_gated_report_2026-06-15.json — gating mean-rev to
    choppy tape removed the -104.7% w2 disaster (-> +1.4%), lifted
    pooled PF 1.06 -> 1.12 and return +127% -> +180% with 28% fewer
    trades. Breakout/news untouched; fail-open on unknown ER."""

    def test_gate_config_flag_default_on(self):
        from core.config import Config
        assert Config().REGIME_GATE_LIVE_MEANREV is True

    def test_allows_semantics_for_gate(self):
        """The gate vetoes exactly when allows() is False — mean-rev is
        blocked in trending tape, permitted in choppy, permitted on
        unknown ER (fail-open)."""
        from allocators.regime_allocator import allocate
        trending = allocate(0.45, threshold=0.30)
        choppy = allocate(0.10, threshold=0.30)
        unknown = allocate(None, threshold=0.30)
        assert trending.allows("mean_reversion") is False   # VETOED
        assert choppy.allows("mean_reversion") is True       # allowed
        assert unknown.allows("mean_reversion") is True      # fail-open
        # breakout/news are never the gate's target
        assert trending.allows("breakout") is True
        assert trending.allows("news") is True

    def test_oversold_v2_also_gated(self):
        """The v2 mean-rev swap must be gated identically."""
        from allocators.regime_allocator import allocate
        trending = allocate(0.45, threshold=0.30)
        assert trending.allows("oversold_v2") is False

    def test_engine_has_live_veto_wired(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent
               / "core" / "engine.py").read_text()
        anchor = src.find("v-regime-gate-live-meanrev-2026-06-16")
        assert anchor != -1, "live gate block missing"
        block = src[anchor: anchor + 1400]
        assert "REGIME_GATE_LIVE_MEANREV" in block
        assert "mean_reversion" in block
        assert "return" in block, "veto must abort the signal"
        assert "regime_gate_meanrev" in block, "veto must be audited"

    def test_veto_only_targets_meanrev_family(self):
        """The veto condition must restrict to the mean-rev family so
        breakout/news signals are never blocked by it."""
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent
               / "core" / "engine.py").read_text()
        anchor = src.find("v-regime-gate-live-meanrev-2026-06-16")
        block = src[anchor: anchor + 1400]
        assert ('"mean_reversion", "oversold_v2"' in block
                or "'mean_reversion', 'oversold_v2'" in block), (
            "gate must scope to the mean-rev family explicitly")


# ── v-conviction-floor-meanrev-2026-06-17 ────────────────────────────

class TestConvictionFloorMeanRev:
    """Blocks mean-rev signals whose meta_proba is below a floor.
    Evidence: research/conviction_validation_2026-06-11.json — meta<0.60
    bucket ran PF 0.44 (losing) across 89 trades. 2026-06-16 live: the
    bot's all-medium-conviction basket netted -$268; meta<0.60 would
    have blocked ARM+ELF (saved $150, killed no wins). Scope: mean-rev
    family ONLY. Fail-open: a signal with no meta_proba is never
    blocked (can't floor what wasn't measured)."""

    def test_config_flags(self):
        from core.config import Config
        c = Config()
        assert c.ENABLE_CONVICTION_FLOOR_MEANREV is True
        assert abs(c.CONVICTION_FLOOR_META - 0.60) < 1e-9

    def test_engine_has_floor_wired(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent / "core" / "engine.py").read_text()
        anchor = src.find("v-conviction-floor-meanrev-2026-06-17")
        assert anchor != -1, "conviction floor block missing"
        block = src[anchor: anchor + 1900]
        assert "ENABLE_CONVICTION_FLOOR_MEANREV" in block
        assert "CONVICTION_FLOOR_META" in block
        assert "meta_proba" in block
        assert "return" in block, "floor must abort the signal"
        assert "conviction_floor" in block, "must audit the block"

    def test_floor_scope_meanrev_only(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent / "core" / "engine.py").read_text()
        anchor = src.find("v-conviction-floor-meanrev-2026-06-17")
        block = src[anchor: anchor + 1900]
        assert ('"mean_reversion", "oversold_v2"' in block
                or "'mean_reversion', 'oversold_v2'" in block), (
            "floor must scope to the mean-rev family")

    def test_floor_fails_open_on_missing_meta(self):
        """Source marker: the block must require meta_proba is not None
        so a signal without a meta score is never floored."""
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent / "core" / "engine.py").read_text()
        anchor = src.find("v-conviction-floor-meanrev-2026-06-17")
        block = src[anchor: anchor + 1900]
        assert "is not None" in block, "floor must fail-open on missing meta_proba"
