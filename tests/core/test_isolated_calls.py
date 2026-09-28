"""v-wedge-fix-2026-09-28 (Task 4, ruling 2): yfinance runs in a subprocess.

A curl_cffi transfer stalled in a TLS write holds the GIL, which freezes
*every* thread in the process — that is how 251 threads ended up parked in
futex_wait with the main thread in poll. A thread timeout cannot recover
from that; only another process can. These tests pin:

  - a child that sleeps past the timeout is killed and the pool recreated
  - the pool is still usable for the next call (no permanent poisoning)
  - the pool is spawn-context with at most 2 workers
  - while the pool is still booting, calls fail fast with None instead of
    burning their timeout waiting for a cold child
  - the flag off path still bounds the call (thread + timeout)

Offline: every payload is a pure-Python function, no network.
"""
from __future__ import annotations

import asyncio
import os
import time

import pytest

from core import isolated_calls


def _child_pid() -> int:
    return os.getpid()


def _child_sleep(seconds: float) -> str:
    time.sleep(seconds)
    return "finished"


def _child_double(value: int) -> int:
    return value * 2


@pytest.fixture(autouse=True)
def _teardown_pool():
    from core import provider_calls

    isolated_calls.shutdown_pool("test_setup")
    provider_calls.reset_state()          # deterministic rate-limiter state
    yield
    isolated_calls.shutdown_pool("test_teardown")
    provider_calls.reset_state()


@pytest.fixture
def ready_pool():
    """Bring the pool up (spawn boots a fresh interpreter — slow) so the
    tests below measure call behaviour, not child start-up.

    One retry: spawning two interpreters can genuinely fail on a loaded box
    (observed once in a full-suite run), and the production path tolerates that
    by rebuilding. A second consecutive failure is a real defect.
    """
    if not isolated_calls.wait_ready(timeout=180.0):
        isolated_calls.shutdown_pool("fixture_retry")
        assert isolated_calls.wait_ready(timeout=180.0), (
            "isolated pool never became ready (two attempts)"
        )
    return isolated_calls.get_pool()


def test_pool_is_spawn_context_with_two_workers(ready_pool):
    assert ready_pool._max_workers == 2
    assert ready_pool._mp_context.get_start_method() == "spawn"


def test_calls_before_the_pool_is_ready_fail_fast_with_none():
    isolated_calls.shutdown_pool("explicit")
    started = time.monotonic()
    out = asyncio.run(
        isolated_calls.run_isolated(_child_double, 4, timeout=30.0, site="warming")
    )
    assert out is None
    assert time.monotonic() - started < 5.0     # did not wait on the cold child


def test_timeout_kills_the_child_and_the_next_call_still_works(ready_pool):
    first_pid = asyncio.run(
        isolated_calls.run_isolated(_child_pid, timeout=30.0, site="pid")
    )
    assert isinstance(first_pid, int)

    started = time.monotonic()
    hung = asyncio.run(
        isolated_calls.run_isolated(_child_sleep, 60.0, timeout=1.0, site="hang")
    )
    assert hung is None
    assert time.monotonic() - started < 3.0

    # The hung child is gone, not merely abandoned.
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if not _pid_alive(first_pid):
            break
        time.sleep(0.2)
    assert not _pid_alive(first_pid), "hung child survived the timeout"

    # And the pool recovers for the next caller.
    assert isolated_calls.wait_ready(timeout=180.0)
    assert asyncio.run(
        isolated_calls.run_isolated(_child_double, 21, timeout=30.0, site="after")
    ) == 42


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def test_sync_variant_returns_values_and_times_out(ready_pool):
    assert isolated_calls.run_isolated_sync(
        _child_double, 5, timeout=30.0, site="sync_ok"
    ) == 10
    started = time.monotonic()
    assert isolated_calls.run_isolated_sync(
        _child_sleep, 60.0, timeout=1.0, site="sync_hang"
    ) is None
    assert time.monotonic() - started < 3.0


def test_flag_off_still_bounds_the_call_in_a_thread(monkeypatch):
    monkeypatch.setenv("ISOLATED_YF_ENABLED", "0")
    assert asyncio.run(
        isolated_calls.run_isolated(_child_double, 8, timeout=5.0, site="off")
    ) == 16
    started = time.monotonic()
    assert asyncio.run(
        isolated_calls.run_isolated(_child_sleep, 30.0, timeout=0.2, site="off_hang")
    ) is None
    assert time.monotonic() - started < 1.0
    assert isolated_calls.get_pool_if_created() is None    # no subprocesses at all


def test_yf_entry_points_are_module_level_and_picklable():
    """The child unpickles by qualified name: bound methods and closures
    cannot cross a spawn boundary, so the yfinance entry points must be
    plain module-level functions in a module that does not drag `core`
    (and its log-file handler) into the child."""
    import pickle

    import yf_isolated

    for name in (
        "yf_download_daily", "yf_ticker_news", "yf_history",
        "yf_fast_quote", "yf_multi_quotes", "yf_market_breadth",
        "child_init",
    ):
        fn = getattr(yf_isolated, name)
        assert pickle.loads(pickle.dumps(fn)) is fn
    assert "core" not in {m.split(".")[0] for m in _module_imports("yf_isolated.py")}


def _module_imports(path: str) -> set[str]:
    """Module-level imports only — the child pays for those on every spawn."""
    import ast
    from pathlib import Path

    tree = ast.parse(Path(path).read_text())
    mods: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            mods.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module)
    return mods


# ── fix round 1 (Important 2): a failed probe must not disable the pool ──

def _probe_boom() -> str:
    raise RuntimeError("probe failed in child")


def test_probe_failure_warns_and_tears_the_pool_down(monkeypatch, caplog):
    """Before the fix a failed probe left the pool alive-but-never-ready, so
    every call for the rest of the session returned None at INFO. Now the
    failure is a rate-limited WARNING and the pool is torn down."""
    import logging

    import yf_isolated

    monkeypatch.setattr(yf_isolated, "probe", _probe_boom)
    with caplog.at_level(logging.WARNING, logger="TradingBot"):
        assert isolated_calls.wait_ready(timeout=120.0) is False
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline and isolated_calls.get_pool_if_created() is not None:
            time.sleep(0.2)
    assert isolated_calls.get_pool_if_created() is None, "pool was not torn down"
    assert any(
        "isolated_pool_unavailable" in r.getMessage() for r in caplog.records
    ), [r.getMessage() for r in caplog.records]


def test_a_call_during_the_probe_cooldown_does_not_respawn_the_pool(monkeypatch):
    import yf_isolated

    monkeypatch.setattr(yf_isolated, "probe", _probe_boom)
    monkeypatch.setattr(isolated_calls, "PROBE_RETRY_COOLDOWN_SEC", 600.0)
    assert isolated_calls.wait_ready(timeout=120.0) is False
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline and isolated_calls.get_pool_if_created() is not None:
        time.sleep(0.2)
    out = asyncio.run(
        isolated_calls.run_isolated(_child_double, 2, timeout=5.0, site="cooldown")
    )
    assert out is None
    assert isolated_calls.get_pool_if_created() is None


def test_the_pool_is_rebuilt_once_the_probe_works_again(monkeypatch):
    import yf_isolated

    monkeypatch.setattr(yf_isolated, "probe", _probe_boom)
    monkeypatch.setattr(isolated_calls, "PROBE_RETRY_COOLDOWN_SEC", 0.0)
    assert isolated_calls.wait_ready(timeout=120.0) is False
    monkeypatch.undo()
    assert isolated_calls.wait_ready(timeout=180.0) is True
    assert asyncio.run(
        isolated_calls.run_isolated(_child_double, 21, timeout=30.0, site="rebuilt")
    ) == 42


# ── fix round 2 (Minor 5): a teardown invalidates the generation ───────

def test_shutdown_pool_bumps_the_generation():
    """The generation guard is what stops a killed pool's probe from tearing
    down its successor, so the teardown itself must invalidate it — not only
    the next creation."""
    isolated_calls.shutdown_pool("test_reset")
    pool = isolated_calls.get_pool()
    gen_before = isolated_calls.current_generation()
    isolated_calls.shutdown_pool("explicit")
    gen_after = isolated_calls.current_generation()
    assert gen_after > gen_before
    assert isolated_calls.get_pool_if_created() is None
    # A late probe callback from the dead pool is now a no-op.
    isolated_calls._on_probe_done(_DoneFuture(RuntimeError("late")), generation=gen_before)
    assert isolated_calls.probe_failed() is False


class _DoneFuture:
    def __init__(self, exc=None):
        self._exc = exc

    def cancelled(self):
        return False

    def exception(self):
        return self._exc
