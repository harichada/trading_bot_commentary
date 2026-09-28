"""Source-level wiring checks (engine is not constructible in tests) plus an import smoke test."""
import re
from pathlib import Path


def test_engine_wires_fit_gate_after_regime_gate():
    src = Path("core/engine.py").read_text()
    assert "from core.strategy_fit_gate import evaluate_fit_gate, family_for" in src
    assert "self._strategy_fit = StrategyFit(" in src
    regime = src.index('"regime_gate", signal.symbol, "blocked"')
    gate = src.index('self._audit("strategy_fit", signal.symbol, _fd.action, _fd.reason')
    conviction = src.index("v-conviction-floor-meanrev-2026-06-17")
    assert regime < gate < conviction
    assert "await asyncio.to_thread(self._strategy_fit.reload_if_changed)" in src


def test_engine_module_imports():
    import importlib
    importlib.import_module("core.strategy_fit_gate")
    importlib.import_module("core.engine")


def test_engine_registers_fit_lanes_and_bars_source():
    src = Path("core/engine.py").read_text()
    assert "from strategies.fit_lane import FitLaneStrategy" in src
    assert "for _fam in Config().FIT_LANES_ENABLED:" in src and "def _fit_bars_source(self, symbol" in src
    assert "self._fit_bars_cache = {}" in src


def test_engine_merges_research_history_into_fit_bars():
    """v-strategy-fit-review2-2026-09-25 (Important 3): the provider's 10-day
    minute window is too short for rvol (20 sessions) and or*_range_avg20
    (10), so the nightly research Parquet history is spliced in underneath,
    under the configured root."""
    src = Path("core/engine.py").read_text()
    assert "from core.strategy_fit_bars import (" in src
    assert "load_fit_history(Config().STRATEGY_HUNT_DATA_ROOT, symbol)" in src
    assert "merge_fit_history(_hist5, bars5)" in src and "merge_fit_daily(_hist_daily, daily)" in src
    # The hardcoded research root is gone from the swing-exit call site too.
    assert '"data/strategy_hunt"' not in src


def test_engine_contains_swing_exit_block_and_threads_earnings_io():
    """v-strategy-fit-review2-2026-09-25 (Important 6): the swing block sits
    above the hard SL/TP checks, so it must never raise past itself; and the
    blocking Parquet read runs off the event loop."""
    src = Path("core/engine.py").read_text()
    swing = src.index('self._audit("swing_exit", position.symbol, "exit", why')
    guard = src.index("swing_exit check failed for %s")
    stop_hit = src.index("stop_loss_hit = current_price >= position.stop_loss")
    assert swing < guard < stop_hit, "the except/fall-through must sit between the block and the SL/TP checks"
    assert "await asyncio.to_thread(\n                        load_earnings_dates, Config().STRATEGY_HUNT_DATA_ROOT" in src


def test_engine_wires_swing_exits_and_overnight_gate():
    src = Path("core/engine.py").read_text()
    assert 'self._audit("swing_exit", position.symbol, "exit", why' in src
    assert 'self._audit("swing_overnight_gate", signal.symbol, "skip", "overnight_not_allowed"' in src
    assert src.index('return True, "day_trade_flatten_hour"') < src.index('self._audit("swing_exit"')


def test_engine_passes_the_live_regime_into_the_fit_gate():
    """v-playbook-desk-b1-2026-09-25: the gate consults the regime-conditioned
    row when the market-context regime is known and the flag is on."""
    src = Path("core/engine.py").read_text()
    assert 'Config().STRATEGY_FIT_REGIME_ROWS' in src
    assert 'signal.reasoning.get("market_context_regime")' in src
    gate = src.index("_fd = evaluate_fit_gate(")
    call = src[gate:gate + 400]
    assert "regime=_fit_regime" in call
    assert src.index("_fit_regime =") < gate


def test_engine_wires_the_allowlist_and_context_slice_into_the_fit_lanes():
    """v-go-live-5k-2026-09-28 (Task 2): the lanes read FIT_LANE_LIVE_ALLOWLIST for admission and
    the nightly universe-context slice for the `ctx_*` families, both through callables so the
    flag is re-read live and the slice is cached per export."""
    src = Path("core/engine.py").read_text()
    assert "allowlist=lambda: Config().FIT_LANE_LIVE_ALLOWLIST" in src
    assert "context_source=self._fit_context_source" in src
    assert "def _fit_context_source(self, symbol" in src
    assert "load_fit_context_slice(root, symbol)" in src
    ctor = src.index("_lane = FitLaneStrategy(")
    assert src.index("allowlist=lambda: Config().FIT_LANE_LIVE_ALLOWLIST") > ctor
