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
