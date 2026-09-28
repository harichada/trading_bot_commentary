"""v-wedge-fix-2026-09-28 (Task 4, ruling 4/5): the external restarter runs.

Two checks, both offline: the script parses (`bash -n`), and a `--dry-run`
against a dead port reports what it would do without signalling any process,
writing any state, or printing the API key.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = "scripts/bot_watchdog.sh"
SENTINEL_KEY = "sentinel-not-a-real-key-0000"
#: Nothing listens here, so /api/status is unreachable — the failure path.
DEAD_PORT = "1"


def _run(args, tmp_path, extra_env=None, timeout=120):
    env = dict(os.environ)
    env.update({
        "TRADING_API_KEY": SENTINEL_KEY,
        "TRADING_BOT_PORT": DEAD_PORT,
        "WATCHDOG_HOST": "127.0.0.1",
        "WATCHDOG_CURL_TIMEOUT": "2",
        "WATCHDOG_LOG": str(tmp_path / "wd.log"),
        "WATCHDOG_STATE": str(tmp_path / "wd.state"),
    })
    env.update(extra_env or {})
    return subprocess.run(
        ["bash", SCRIPT, *args],
        capture_output=True, text=True, timeout=timeout, env=env,
    )


def test_script_parses():
    done = subprocess.run(
        ["bash", "-n", SCRIPT], capture_output=True, text=True, timeout=60,
    )
    assert done.returncode == 0, done.stderr


def test_dry_run_against_a_dead_port_reports_and_changes_nothing(tmp_path):
    done = _run(["--dry-run"], tmp_path)
    out = done.stdout + done.stderr
    assert "health FAILED http=000" in out
    assert "fail_count=1/2" in out
    assert "waiting for a second consecutive failure" in out
    assert done.returncode == 1
    # Dry run writes neither the log nor the state file.
    assert not (tmp_path / "wd.log").exists()
    assert not (tmp_path / "wd.state").exists()


def test_dry_run_shows_the_full_restart_sequence_without_doing_it(tmp_path):
    done = _run(
        ["--dry-run"], tmp_path,
        extra_env={"WATCHDOG_FAILS_BEFORE_RESTART": "1"},
    )
    out = done.stdout + done.stderr
    assert "dry-run: would restart now" in out
    assert "would launch, then POST /api/start, then POST /api/toggle-mode" in out
    assert '{"mode":"live"}' in out
    assert not (tmp_path / "wd.state").exists()


def test_dry_run_never_prints_the_api_key(tmp_path):
    done = _run(
        ["--dry-run"], tmp_path,
        extra_env={"WATCHDOG_FAILS_BEFORE_RESTART": "1"},
    )
    assert SENTINEL_KEY not in done.stdout
    assert SENTINEL_KEY not in done.stderr


def test_restart_budget_is_respected(tmp_path):
    state = tmp_path / "wd.state"
    state.write_text(
        "FAIL_COUNT=1\n"
        f"RESTART_DATE={__import__('datetime').date.today().isoformat()}\n"
        "RESTART_COUNT=2\n"
    )
    done = _run(["--dry-run"], tmp_path)
    out = done.stdout + done.stderr
    assert "restart budget exhausted" in out
    assert "OPERATOR ACTION REQUIRED" in out
    assert done.returncode == 1


def test_unknown_argument_is_rejected(tmp_path):
    done = _run(["--wat"], tmp_path)
    assert done.returncode == 2
    assert "unknown argument" in done.stderr
