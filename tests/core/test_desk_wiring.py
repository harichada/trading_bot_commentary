"""v-playbook-desk-b2-2026-09-25 (Task 5): source-level wiring checks for the desk.

The engine is not constructible in tests (tests/core/test_strategy_fit_wiring.py
explains why), so the desk's three touch points in live code are asserted
against the source text:

  1. `core/engine.py` — the desk-limit hook sits *before* the strategy-fit gate
     and the playbook hook *after* it; both are wrapped in `if
     Config().ENABLE_DESK:` and in a try/except, and the supervisor registers
     `desk_loop` only under the flag.
  2. `core/loops/screener_loop.py` — the screener stops overwriting
     `dynamic_watchlist` once the desk owns it.
  3. `risk/manager.py` — the playbook size multiplier is applied after the
     per-strategy multiplier and only when `ENABLE_PLAYBOOK_GATE` is on.
"""
from __future__ import annotations

import importlib
from pathlib import Path

ENGINE = Path("core/engine.py").read_text()
SCREENER = Path("core/loops/screener_loop.py").read_text()
RISK = Path("risk/manager.py").read_text()
LOOP = Path("core/loops/desk_loop.py").read_text()


# ── engine: hook order ──────────────────────────────────────────────

def test_engine_imports_the_desk_decision_functions():
    assert "from core.desk.daily_plan import evaluate_desk_gate" in ENGINE
    assert "from core.desk.playbook import evaluate_playbook_gate" in ENGINE


def test_desk_limit_audit_precedes_the_fit_gate_and_playbook_follows_it():
    desk = ENGINE.index('self._audit("desk_limit", signal.symbol, _dd.action, _dd.reason')
    fit = ENGINE.index('self._audit("strategy_fit", signal.symbol, _fd.action, _fd.reason')
    playbook = ENGINE.index('self._audit("playbook", signal.symbol, _pd.action, _pd.reason')
    assert desk < fit < playbook


def test_both_hooks_are_guarded_by_enable_desk():
    desk = ENGINE.index('self._audit("desk_limit"')
    playbook = ENGINE.index('self._audit("playbook"')
    # Each hook opens with its own ENABLE_DESK check a few lines above the audit.
    assert "if Config().ENABLE_DESK:" in ENGINE[max(0, desk - 1500):desk]
    assert "if Config().ENABLE_DESK:" in ENGINE[max(0, playbook - 1500):playbook]


def test_both_hooks_catch_their_own_exceptions():
    """A desk failure must never block a signal — each hook ends in a guard."""
    desk = ENGINE.index('self._audit("desk_limit"')
    playbook = ENGINE.index('self._audit("playbook"')
    assert "self._desk_hook_warn(\"desk_limit\"" in ENGINE[desk:desk + 1500]
    assert "self._desk_hook_warn(\"playbook\"" in ENGINE[playbook:playbook + 1800]
    assert ENGINE.count("def _desk_hook_warn") == 1


def test_the_size_multiplier_is_written_only_when_the_gate_is_enabled():
    playbook = ENGINE.index('self._audit("playbook"')
    block = ENGINE[max(0, playbook - 900):playbook + 1800]
    assert "_pb_enabled = Config().ENABLE_PLAYBOOK_GATE" in block
    write = block.index('signal.reasoning["strategy_size_multiplier"] = ')
    guard = block.rindex("if _pb_enabled and ", 0, write)
    assert block.index("_pb_enabled = Config().ENABLE_PLAYBOOK_GATE") < guard < write


def test_desk_limits_flag_drives_only_the_blocking_decision():
    desk = ENGINE.index('self._audit("desk_limit"')
    block = ENGINE[max(0, desk - 900):desk + 600]
    assert "Config().ENABLE_DESK_LIMITS" in block
    assert 'if _dd.action == "skip":' in block


def test_supervisor_registers_the_desk_loop_under_the_flag():
    reg = ENGINE.index('sup.register("desk_loop"')
    assert "if Config().ENABLE_DESK:" in ENGINE[max(0, reg - 700):reg]
    assert "from core.loops.desk_loop import DeskLoop" in ENGINE
    assert "self._desk_loop = DeskLoop(self)" in ENGINE


def test_trades_today_comes_from_a_cached_engine_count():
    assert "async def _desk_trades_today" in ENGINE
    assert "get_todays_bot_entries" in ENGINE
    assert "_DESK_CACHE_SEC" in ENGINE


# ── screener guard ──────────────────────────────────────────────────

def test_screener_does_not_overwrite_a_desk_owned_watchlist():
    guard = SCREENER.index("_desk_owns_watchlist")
    assign = SCREENER.index("self.engine.dynamic_watchlist = final_watchlist")
    assert guard < assign
    assert "ENABLE_DESK_WATCHLIST" in SCREENER
    assert "plan.in_play" in SCREENER


# ── risk sizing step ────────────────────────────────────────────────

def test_risk_applies_the_playbook_multiplier_after_the_strategy_multiplier():
    strategy = RISK.index("strategy_size_multiplier symbol=%s strategy=%s")
    playbook = RISK.index("playbook_size_multiplier symbol=%s")
    context = RISK.index("market_context_conviction symbol=%s")
    assert strategy < playbook < context
    assert "Config().ENABLE_PLAYBOOK_GATE" in RISK


# ── loop shape ──────────────────────────────────────────────────────

def test_the_loop_bounds_its_scan_concurrency_and_uses_threads():
    assert "asyncio.Semaphore(" in LOOP
    assert "asyncio.to_thread" in LOOP
    from core.loops import desk_steps
    assert desk_steps.SCAN_CONCURRENCY == 4


def test_desk_modules_import():
    importlib.import_module("core.loops.desk_loop")
    importlib.import_module("core.loops.desk_steps")
    importlib.import_module("core.engine")


def test_core_desk_never_imports_research():
    """`core` reads the research *data* tree but never imports the package.

    Fix round 1, Minor 4: the desk's loop-side modules are covered too, not just
    `core/desk/*` — they are the ones that read the nightly Parquet files.
    """
    paths = list(Path("core/desk").glob("*.py")) + [
        Path("core/loops/desk_loop.py"),
        Path("core/loops/desk_steps.py"),
        Path("core/loops/desk_trade_stats.py"),
    ]
    for path in paths:
        text = path.read_text()
        assert "import research" not in text, path
        assert "from research" not in text, path


# ────────────────────────────────────────────────────────────────────
# fix round 1 (review r1): hook short-circuit, DB timeout, watchlist cap
# ────────────────────────────────────────────────────────────────────

def test_the_desk_hook_short_circuits_before_the_db_read():
    """Minor 2: no plan loaded means nothing to govern — the hook must not run a
    per-signal `bot_trades` query to reach a governed=False decision."""
    desk = ENGINE.index('self._audit("desk_limit"')
    block = ENGINE[max(0, desk - 1200):desk]
    plan_read = block.index("self._desk_plan_today()")
    guard = block.index("if _desk_plan is not None:")
    trades_read = block.index("self._desk_trades_today()")
    assert plan_read < guard < trades_read


def test_the_trade_count_read_is_time_boxed():
    """Minor 3: a slow Postgres must not hold a signal — the await is bounded and
    a timeout degrades to an untrustworthy (ungoverned) count."""
    assert "_DESK_DB_TIMEOUT_SEC" in ENGINE
    helper = ENGINE.index("async def _desk_trades_today")
    block = ENGINE[helper:helper + 2200]
    assert "asyncio.wait_for(" in block
    assert "asyncio.TimeoutError" in block
    assert 'self._desk_hook_warn("desk_trades_today"' in block


def test_the_desk_watchlist_handover_is_capped():
    """Minor 5: the desk may not hand the engine a watchlist longer than the
    screener's own cap, and pinned symbols come first."""
    assert "WATCHLIST_SIZE" in LOOP
    assert "_watchlist_from(" in LOOP


def test_the_playbook_size_step_has_no_one_share_floor_for_zero():
    """Important 1: the sizing chain's playbook step yields 0 shares, not 1."""
    assert "def playbook_size_step(" in RISK
    assert "playbook_size_zero" in RISK
    step = RISK.index("def playbook_size_step(")
    body = RISK[step:step + 1600]
    zero = body.index("if mult <= PLAYBOOK_SIZE_MULT_MIN:")
    floor = body.index("return max(1, int(position_size * mult)), mult")
    assert zero < floor, "the zero case must be decided before the 1-share floor"
