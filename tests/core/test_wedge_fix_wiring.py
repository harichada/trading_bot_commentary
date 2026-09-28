"""v-wedge-fix-2026-09-28 (Task 4, ruling 5): source-level wiring for the
anti-wedge fix.

The engine is not constructible in tests (see tests/core/test_strategy_fit_wiring.py
for why), so the wiring is asserted against the source text — the same style
tests/core/test_desk_wiring.py uses. What matters here is an absence: NO bare
synchronous provider call may remain anywhere in `core/engine.py`, because one
of them is what froze the live process for a whole session on 2026-09-28.
"""
from __future__ import annotations

import re
from pathlib import Path

ENGINE_PATH = Path("core/engine.py")
ENGINE = ENGINE_PATH.read_text()
API = Path("api/routes.py").read_text()
SCREENER = Path("analysis/screener.py").read_text()
NEWS = Path("strategies/news_strategy.py").read_text()
VERIFIER = Path("analysis/news_verifier.py").read_text()
REALTIME = Path("data_providers/realtime.py").read_text()
CONFIG = Path("core/config.py").read_text()
WATCHDOG_PATH = Path("scripts/bot_watchdog.sh")
WATCHDOG = WATCHDOG_PATH.read_text()


def _code_lines(source: str) -> list[tuple[int, str]]:
    """Line numbers + the code part of each line (comments stripped).

    Crude but sufficient: the patterns below never appear inside a string
    literal in these files, and stripping `#` keeps the prose that explains
    the fix from failing the test that enforces it.
    """
    out = []
    for lineno, raw in enumerate(source.split("\n"), start=1):
        code = raw.split("#", 1)[0]
        if code.strip():
            out.append((lineno, code))
    return out


def _hits(source: str, needle: str) -> list[int]:
    return [lineno for lineno, code in _code_lines(source) if needle in code]


# ── ruling 1: no bare provider call left in the engine ────────────────

def test_no_bare_get_quote_call_remains_in_the_engine():
    assert _hits(ENGINE, "self.data_provider.get_quote(") == []


def test_no_bare_get_market_data_call_remains_in_the_engine():
    assert _hits(ENGINE, "self.data_provider.get_market_data(") == []


def test_every_provider_call_goes_through_the_helper():
    """Each converted site passes the bound method as a VALUE to the helper
    (`get_quote,` — note the comma, not a paren) and is preceded by one of
    the two helpers."""
    for lineno, code in _code_lines(ENGINE):
        if "self.data_provider.get_quote," in code or "self.data_provider.get_market_data," in code:
            window = "\n".join(
                c for n, c in _code_lines(ENGINE) if lineno - 6 <= n <= lineno
            )
            assert (
                "self._provider_call(" in window
                or "self._provider_call_sync(" in window
                # The quote streamer keeps its own semaphore + wait_for
                # (core/engine.py `_fetch_one_quote`), which is the pattern
                # this fix generalised from.
                or "run_in_executor(" in window
            ), f"unhelped provider call at core/engine.py:{lineno}"


def test_the_helpers_are_defined_once_and_delegate_to_the_module():
    assert ENGINE.count("async def _provider_call(") == 1
    assert ENGINE.count("def _provider_call_sync(") == 1
    assert "from core.provider_calls import (" in ENGINE
    assert "provider_call," in ENGINE and "provider_call_sync," in ENGINE


def test_the_position_loop_never_blocks_on_a_quote():
    """Ruling 1's hard requirement: on a quote timeout the position loop
    uses the last known price, and if there is none it skips the position
    for the cycle with an audit line."""
    idx = ENGINE.index('site="position_manager_quote"')
    block = ENGINE[idx:idx + 1200]
    assert "await self._last_known_price(symbol)" in block
    assert '"position_manager", symbol, "skip",' in block
    assert '"quote_timeout"' in block
    assert "continue" in block
    assert ENGINE.count("async def _last_known_price(") == 1


def test_last_known_price_reads_the_price_book_then_the_quote_cache():
    idx = ENGINE.index("async def _last_known_price(")
    block = ENGINE[idx:idx + 1600]
    assert "self.price_book.get_mark(symbol)" in block
    assert "await self.quote_cache.get(symbol)" in block
    # entry_price must NOT be a fallback here — managing an exit against
    # the entry price reads as flat P&L and disables every price-based rule.
    assert "does NOT fall back" in block
    assert "return float(self.entry_price" not in block


def test_history_shaped_calls_use_the_history_timeout():
    for site in (
        "fit_bars_5m", "fit_bars_daily", "spy_daily_closes",
        "multi_timeframe_daily", "position_atr_bars", "exit_conditions_bars",
    ):
        idx = ENGINE.index(f'site="{site}"')
        assert "provider_history_timeout()" in ENGINE[max(0, idx - 400):idx], site


# ── ruling 2: yfinance only ever runs in the isolated pool ────────────

def test_no_module_uses_yfinance_in_process_any_more():
    for name, source in (
        ("analysis/screener.py", SCREENER),
        ("strategies/news_strategy.py", NEWS),
        ("analysis/news_verifier.py", VERIFIER),
        ("data_providers/realtime.py", REALTIME),
    ):
        assert _hits(source, "import yfinance") == [], name
        assert _hits(source, "yf.") == [], name


def test_each_yfinance_caller_goes_through_run_isolated():
    assert "from core.isolated_calls import run_isolated" in SCREENER
    assert "run_isolated(\n            yf_download_daily" in SCREENER
    assert "from core.isolated_calls import run_isolated" in NEWS
    assert "run_isolated(\n                yf_ticker_news" in NEWS
    assert "from core.isolated_calls import run_isolated" in VERIFIER
    assert "run_isolated(\n            yf_ticker_news" in VERIFIER
    assert "from core.isolated_calls import run_isolated_sync" in REALTIME


def test_every_public_realtime_method_is_isolated():
    """The fallback provider's four public methods each delegate to the
    subprocess pool through the one `_isolated` helper."""
    for method in (
        "def get_market_data(", "def get_quote(",
        "def get_multiple_quotes(", "def calculate_market_breadth(",
    ):
        idx = REALTIME.index(method)
        body = REALTIME[idx:idx + 1800]
        assert "self._isolated(" in body, method
    assert REALTIME.count("def _isolated(") == 1


def test_isolated_pool_is_spawn_with_two_workers_and_is_rebuilt_on_timeout():
    source = Path("core/isolated_calls.py").read_text()
    assert "POOL_WORKERS: int = 2" in source
    assert 'multiprocessing.get_context("spawn")' in source
    assert "SIGKILL" in source
    idx = source.index("except asyncio.TimeoutError:")
    assert "shutdown_pool(" in source[idx:idx + 300]


# ── ruling 3: heartbeat on both loops ─────────────────────────────────

def test_engine_registers_the_heartbeat_task_and_starts_the_watchdog():
    assert 'sup.register("loop_heartbeat", self._loop_heartbeat_loop' in ENGINE
    assert "get_heartbeat().start_watchdog()" in ENGINE
    assert ENGINE.count("async def _loop_heartbeat_loop(") == 1


def test_api_loop_pulses_too():
    assert "from core.loop_heartbeat import get_heartbeat" in API
    assert '_heartbeat.run("api"' in API
    assert "_heartbeat.start_watchdog()" in API


# ── config ────────────────────────────────────────────────────────────

def test_the_new_settings_live_in_their_own_section():
    section = CONFIG.index("# v-wedge-fix-2026-09-28")
    tail = CONFIG[section:]
    for name in (
        "PROVIDER_EXECUTOR_WORKERS", "PROVIDER_HISTORY_TIMEOUT_SEC",
        "LOOP_LAG_WARN_SEC", "ISOLATED_YF_ENABLED",
    ):
        assert f"def {name}(" in tail, name


def test_child_processes_do_not_own_the_rotating_log_file():
    """A spawn child re-imports the main module and lands in setup_logging();
    two processes rotating trading_bot.log corrupts it."""
    assert "_mp.parent_process() is not None" in CONFIG
    assert "if not _is_child:" in CONFIG


def test_config_values_are_sane():
    from core.config import Config

    cfg = Config()
    assert cfg.PROVIDER_EXECUTOR_WORKERS >= 1
    assert cfg.PROVIDER_HISTORY_TIMEOUT_SEC > cfg.QUOTE_FETCH_TIMEOUT_SEC
    assert cfg.LOOP_LAG_WARN_SEC > 0
    assert cfg.ISOLATED_YF_ENABLED is True


# ── ruling 4: the external restarter ──────────────────────────────────

def test_watchdog_script_exists_and_is_executable():
    assert WATCHDOG_PATH.exists()
    assert WATCHDOG_PATH.stat().st_mode & 0o111


def test_watchdog_relaunch_sequence_matches_the_operations_manual():
    assert 'nohup "$PYTHON_BIN" "$BOT_SCRIPT"' in WATCHDOG
    start = WATCHDOG.index("post_api /api/start")
    toggle = WATCHDOG.index("""post_api /api/toggle-mode '{"mode":"live"}'""")
    assert start < toggle, "start must precede the live toggle"


def test_watchdog_policy_numbers():
    assert "-m \"$CURL_TIMEOUT\"" in WATCHDOG
    assert "WATCHDOG_CURL_TIMEOUT:-30" in WATCHDOG
    assert "WATCHDOG_MAX_RESTARTS:-2" in WATCHDOG
    assert "WATCHDOG_FAILS_BEFORE_RESTART:-2" in WATCHDOG
    assert "WATCHDOG_TERM_GRACE:-20" in WATCHDOG
    assert "kill -TERM" in WATCHDOG and "kill -KILL" in WATCHDOG
    assert "--dry-run" in WATCHDOG


def test_watchdog_never_prints_the_api_key():
    """The key may be read (curl needs it) but must never be echoed, logged
    or placed on a command line visible in `ps`."""
    for lineno, code in _code_lines(WATCHDOG):
        if "TRADING_API_KEY" not in code:
            continue
        assert not re.search(r"\b(echo|printf|log)\b.*TRADING_API_KEY", code) or (
            "> \"$HEADER_FILE\"" in code
        ), f"scripts/bot_watchdog.sh:{lineno} may leak the key"
    assert '-H "@$HEADER_FILE"' in WATCHDOG


# ══════════════════════════════════════════════════════════════════════
# Fix round 1
# ══════════════════════════════════════════════════════════════════════

MARKET_CONTEXT_PATH = Path("core/market_context.py")
MARKET_CONTEXT = MARKET_CONTEXT_PATH.read_text()

#: Modules that receive a provider object and still call it synchronously
#: without a bound. Reviewed 2026-09-28 (fix round 1): none of them is on the
#: engine's position/order path, and they are outside this round's file list —
#: `api/routes.py` blocks the FastAPI loop (the `/api/status` silence itself),
#: `core/loops/desk_steps.py` and `analysis/active_open_desk.py` block their
#: own supervised loops, `data_providers/file_bars.py` is a bars wrapper used
#: off the trading path. This list must SHRINK, never grow: a new entry means
#: someone reintroduced an unbounded provider call.
KNOWN_UNBOUNDED_PROVIDER_CALLERS = {
    "api/routes.py",
    "core/loops/desk_steps.py",
    "analysis/active_open_desk.py",
    "data_providers/file_bars.py",
}

PROVIDER_CALL_RE = re.compile(
    r"\b\w*provider\.get_(?:quote|quotes|market_data)\s*\("
)

#: The helper modules describe these calls in their prose (that is their whole
#: subject) and never hold a provider object themselves.
SCAN_SKIP = {"core/provider_calls.py", "core/isolated_calls.py"}


def _provider_call_sites(path: Path) -> list[int]:
    return [
        lineno for lineno, code in _code_lines(path.read_text())
        if PROVIDER_CALL_RE.search(code)
    ]


# ── Important 1: the strategy alias is bounded ────────────────────────

def test_no_bare_provider_call_remains_in_market_context():
    """`schwab_provider` is `engine.data_provider` under another name: these
    calls run on the engine loop via strategies' generate_signal_with_commentary."""
    assert _provider_call_sites(MARKET_CONTEXT_PATH) == []


def test_market_context_routes_every_provider_call_through_the_helper():
    assert "from core import provider_calls" in MARKET_CONTEXT
    # Five call sites: day change, SPY quote, SPY VWAP bars, symbol RTH open,
    # in-move bars.
    assert MARKET_CONTEXT.count("provider_calls.provider_call_sync(") == 5
    for site in (
        "market_context_day_change", "market_context_spy_quote",
        "market_context_spy_vwap_bars", "market_context_symbol_rth_open",
        "market_context_inmove_bars",
    ):
        assert f'site="{site}"' in MARKET_CONTEXT, site
    # Bar fetches get the history budget, quotes the quote budget.
    assert MARKET_CONTEXT.count("timeout=provider_calls.history_timeout()") == 2


def test_the_provider_alias_scan_covers_the_whole_tree():
    """Every module that receives a provider object is scanned, not just the
    ones the engine owns. The allowlist is the outstanding work."""
    offenders = set()
    scanned = 0
    for folder in ("core", "api", "strategies", "analysis", "data_providers"):
        for path in sorted(Path(folder).rglob("*.py")):
            rel = path.as_posix()
            if rel in SCAN_SKIP:
                continue
            scanned += 1
            if _provider_call_sites(path):
                offenders.add(rel)
    assert scanned > 50, f"the scan found only {scanned} modules — glob broke"
    assert offenders == KNOWN_UNBOUNDED_PROVIDER_CALLERS, (
        f"unexpected unbounded provider callers: "
        f"{offenders - KNOWN_UNBOUNDED_PROVIDER_CALLERS}; "
        f"fixed (remove from the list): "
        f"{KNOWN_UNBOUNDED_PROVIDER_CALLERS - offenders}"
    )


# ── Important 2: a failed probe rebuilds, loudly ──────────────────────

def test_probe_failure_tears_down_and_warns():
    source = Path("core/isolated_calls.py").read_text()
    assert "PROBE_RETRY_COOLDOWN_SEC" in source
    idx = source.index("if exc is not None:")
    block = source[idx:idx + 900]
    assert "isolated_pool_unavailable" in block
    assert 'shutdown_pool, args=("probe_failed",)' in block
    # A superseded pool's probe must not tear down its successor.
    assert "if generation != _generation:" in source


# ── Important 3: the watchdog reads the engine's state ────────────────

def test_watchdog_parses_bot_running_and_mode():
    assert '"bot_running"[[:space:]]*:[[:space:]]*true' in WATCHDOG
    assert "bot_not_running" in WATCHDOG
    assert "mode_not_live" in WATCHDOG
    assert "require_live_now" in WATCHDOG


# ── Important 4: the screener fails closed ────────────────────────────

def test_screener_quality_gate_fails_closed():
    assert "def _quality_gate_fail_closed(" in SCREENER
    assert SCREENER.count("_quality_gate_fail_closed(") == 4   # def + 3 sites
    assert "no_quality_data" in SCREENER and "short_history" in SCREENER
    assert "return True  # not enough history; pass" not in SCREENER


# ── Minor 5: the live OCO validator awaits its quote ──────────────────

def test_the_effective_oco_validator_is_async_and_awaited():
    definitions = [
        m.start() for m in re.finditer(r"def _validate_oco_prices\(", ENGINE)
    ]
    assert len(definitions) == 2, "the duplicate is expected; see the DEAD CODE note"
    effective = definitions[-1]
    assert ENGINE[effective - 6:effective] == "async ", "the LAST definition wins"
    block = ENGINE[effective:effective + 700]
    assert "await self._provider_call(" in block
    assert ENGINE.count("await self._validate_oco_prices(") == 2
    assert "if not self._validate_oco_prices(" not in ENGINE
    # The shadowed copy is labelled so nobody edits the wrong one.
    assert "DEAD CODE" in ENGINE[max(0, definitions[0] - 500):definitions[0]]


def test_the_news_veto_lookup_keeps_the_blocking_helper_with_a_reason():
    idx = ENGINE.index('site="news_veto_price"')
    block = ENGINE[max(0, idx - 700):idx]
    assert "_provider_call_sync(" in block
    assert "bounded" in block


# ── Minor 6: heartbeat forgets stopped loops, waits out start-up ──────

def test_heartbeat_forgets_and_has_a_startup_grace():
    source = Path("core/loop_heartbeat.py").read_text()
    assert "DEFAULT_STARTUP_GRACE_SEC: float = 30.0" in source
    assert "def forget(self, name: str)" in source
    idx = source.index("async def run(")
    assert "finally:" in source[idx:idx + 1400]
    assert "self.forget(name)" in source[idx:idx + 1400]


# ── Minor 8: nested budgets ───────────────────────────────────────────

def test_realtime_quote_uses_the_quote_budget():
    idx = REALTIME.index("yf_fast_quote, symbol")
    assert "Config().QUOTE_FETCH_TIMEOUT_SEC" in REALTIME[idx:idx + 200]
