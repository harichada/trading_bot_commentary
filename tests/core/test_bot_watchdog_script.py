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
    assert "health FAILED reason=unreachable_or_error" in out
    assert "http=000" in out
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


# ── fix round 1 (Important 3): /api/status is parsed, not just counted ──

import json
import subprocess as _sp
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer


class _StubStatus(BaseHTTPRequestHandler):
    payload: dict = {}
    code: int = 200

    def do_GET(self):          # noqa: N802 - stdlib naming
        body = json.dumps(self.payload).encode()
        self.send_response(self.code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):         # noqa: N802
        self.do_GET()

    def log_message(self, *args):   # keep pytest output clean
        return


def _serve(payload, code=200):
    """A localhost stub of /api/status. Offline by construction."""
    handler = type("_H", (_StubStatus,), {"payload": payload, "code": code})
    server = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _run_against(payload, tmp_path, code=200, extra_env=None):
    server = _serve(payload, code)
    try:
        port = server.server_address[1]
        env = {"TRADING_BOT_PORT": str(port)}
        env.update(extra_env or {})
        return _run(["--dry-run"], tmp_path, extra_env=env)
    finally:
        server.shutdown()
        server.server_close()


LIVE_OK = {
    "status": "success", "bot_running": True, "mode": "live",
    "positions_count": 2, "schwab_connected": True,
}


def test_a_live_running_engine_is_healthy(tmp_path):
    done = _run_against(LIVE_OK, tmp_path, extra_env={"WATCHDOG_REQUIRE_LIVE": "1"})
    out = done.stdout + done.stderr
    assert "health ok" in out
    assert done.returncode == 0


def test_http_200_with_a_stopped_engine_is_unhealthy(tmp_path):
    payload = dict(LIVE_OK, bot_running=False)
    done = _run_against(payload, tmp_path, extra_env={"WATCHDOG_REQUIRE_LIVE": "1"})
    out = done.stdout + done.stderr
    assert "health FAILED" in out
    assert "reason=bot_not_running" in out
    assert done.returncode == 1


def test_simulation_mode_is_unhealthy_while_live_is_required(tmp_path):
    payload = dict(LIVE_OK, mode="simulation_commentary")
    done = _run_against(payload, tmp_path, extra_env={"WATCHDOG_REQUIRE_LIVE": "1"})
    out = done.stdout + done.stderr
    assert "reason=mode_not_live" in out
    assert "mode=simulation_commentary" in out
    assert done.returncode == 1


def test_simulation_mode_is_fine_outside_the_live_window(tmp_path):
    payload = dict(LIVE_OK, mode="simulation_commentary")
    done = _run_against(payload, tmp_path, extra_env={"WATCHDOG_REQUIRE_LIVE": "0"})
    out = done.stdout + done.stderr
    assert "health ok" in out
    assert done.returncode == 0


def test_a_500_is_unhealthy(tmp_path):
    done = _run_against(LIVE_OK, tmp_path, code=500,
                        extra_env={"WATCHDOG_REQUIRE_LIVE": "1"})
    out = done.stdout + done.stderr
    assert "health FAILED" in out and "http=500" in out
    assert done.returncode == 1


def test_the_live_window_defaults_to_rth_on_weekdays(tmp_path):
    """The default must be `auto`, resolved from ET clock + weekday — not a
    hardcoded yes/no."""
    script = Path(SCRIPT).read_text()
    assert 'WATCHDOG_REQUIRE_LIVE:-auto' in script
    assert "WATCHDOG_LIVE_FROM:-0925" in script
    assert "WATCHDOG_LIVE_TO:-1605" in script
    assert "TZ=America/New_York date +%H%M" in script
    assert "TZ=America/New_York date +%u" in script


# ── fix round 1 (Minor 7): only a python process running the bot is killed ──

def test_a_decoy_process_holding_the_filename_is_never_matched(tmp_path):
    """An editor or grep with the script name in its command line must never
    appear in the kill list."""
    decoy = _sp.Popen(
        ["bash", "-c",
         'exec -a "vim trading_bot_commentary_updated.py" sleep 25'],
    )
    try:
        done = _run_against(
            dict(LIVE_OK, bot_running=False), tmp_path,
            extra_env={"WATCHDOG_REQUIRE_LIVE": "1",
                       "WATCHDOG_FAILS_BEFORE_RESTART": "1"},
        )
        out = done.stdout + done.stderr
        assert "dry-run: would restart now" in out
        assert str(decoy.pid) not in out
    finally:
        decoy.kill()
        decoy.wait(timeout=10)


def test_a_failed_start_or_toggle_escalates(tmp_path):
    """Source-level: a restart that cannot bring the engine back to live must
    say so loudly and exit non-zero (it cannot be exercised without actually
    restarting the bot)."""
    script = Path(SCRIPT).read_text()
    start = script.index("post_api /api/start")
    block = script[start:start + 1400]
    assert "OPERATOR ACTION REQUIRED" in block
    assert script.count("OPERATOR ACTION REQUIRED") >= 2
    assert "2[0-9][0-9]" in block or "is_2xx" in block


def test_watchdog_second_instance_exits_quietly(tmp_path):
    """Two crons (two WSL distros over one disk) must not run the watchdog twice."""
    import os, subprocess, textwrap
    script = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "bot_watchdog.sh")
    lock = tmp_path / "wd.lock"
    holder = subprocess.Popen(["bash", "-c", f"exec 9>'{lock}'; flock 9; sleep 5"])
    try:
        import time; time.sleep(0.5)
        env = dict(os.environ, WATCHDOG_LOCK=str(lock), WATCHDOG_LOG=str(tmp_path / "wd.log"),
                   WATCHDOG_STATE=str(tmp_path / "wd.state"))
        r = subprocess.run(["bash", script, "--dry-run"], env=env, cwd=str(tmp_path),
                           capture_output=True, text=True, timeout=20)
        assert r.returncode == 0
        assert r.stdout.strip() == "" and "health" not in r.stdout
    finally:
        holder.kill()
