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


def test_engine_wires_swing_exits_and_overnight_gate():
    src = Path("core/engine.py").read_text()
    assert 'self._audit("swing_exit", position.symbol, "exit", why' in src
    assert 'self._audit("swing_overnight_gate", signal.symbol, "skip", "overnight_not_allowed"' in src
    assert src.index('return True, "day_trade_flatten_hour"') < src.index('self._audit("swing_exit"')
