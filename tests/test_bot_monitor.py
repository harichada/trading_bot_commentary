"""Unit tests for tools/bot_monitor.py.

The monitor is a pure observer — it never writes to bot state. These
tests verify the parsers correctly extract structure from realistic
log lines so a future change to the monitor doesn't silently lose
visibility into bot activity.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools import bot_monitor as bm


# ────────────────────────────────────────────────────────────────────
# parse_signal_activity
# ────────────────────────────────────────────────────────────────────

class TestParseSignalActivity:
    def test_counts_signals_by_strategy(self):
        entries = [
            {"message": "strategy_decision strategy=mean_reversion symbol=TSLA action=signal_buy"},
            {"message": "strategy_decision strategy=mean_reversion symbol=QBTS action=signal_buy"},
            {"message": "strategy_decision strategy=breakout symbol=NVDA action=signal_buy"},
            {"message": "strategy_decision strategy=news symbol=AMZN action=signal_sell"},
        ]
        a = bm.parse_signal_activity(entries)
        assert a["fired_by_strategy"]["mean_reversion"] == 2
        assert a["fired_by_strategy"]["breakout"] == 1
        assert a["fired_by_strategy"]["news"] == 1

    def test_counts_accepted_by_router(self):
        entries = [
            {"message": "engine_decision component=signal_router symbol=TSLA action=accepted reason=mean_reversion mode=live side=BUY qty=2"},
            {"message": "engine_decision component=signal_router symbol=NVDA action=accepted reason=breakout mode=live side=BUY qty=10"},
        ]
        a = bm.parse_signal_activity(entries)
        # Total accepted across strategies = 2
        assert sum(a["accepted_by_strategy"].values()) == 2

    def test_counts_block_reasons(self):
        entries = [
            {"message": "engine_decision component=signal_router symbol=X action=skip reason=anti_pyramid_recent_attempt"},
            {"message": "engine_decision component=signal_router symbol=Y action=skip reason=anti_pyramid_recent_attempt"},
            {"message": "engine_decision component=signal_router symbol=Z action=skip reason=insufficient_buying_power"},
        ]
        a = bm.parse_signal_activity(entries)
        assert a["block_reasons"]["anti_pyramid_recent_attempt"] == 2
        assert a["block_reasons"]["insufficient_buying_power"] == 1

    def test_captures_rr_audit_warnings(self):
        entries = [
            {
                "timestamp": "2026-06-01 11:30:00,000",
                "level": "WARNING",
                "message": (
                    "rr_audit_low symbol=TSLA strategy=mean_reversion "
                    "entry=441.94 stop=430.89 target=464.04 "
                    "stop_dist=11.05 target_dist=22.10 actual_rr=1.42 "
                    "floor=1.80 — strategy may be capping target"
                ),
            },
        ]
        a = bm.parse_signal_activity(entries)
        assert len(a["rr_audit_low"]) == 1
        assert a["rr_audit_low"][0]["symbol"] == "TSLA"
        assert a["rr_audit_low"][0]["rr"] == 1.42

    def test_captures_direction_gate_skips(self):
        entries = [
            {
                "timestamp": "2026-06-01 11:35:00,000",
                "message": (
                    "engine_decision component=signal_router symbol=NVDA "
                    "action=skip reason=direction_gate_rejected_breakout "
                    "direction=+1.5 phase=ranging ema_stack=mixed "
                    "adx=22.0 high_20=145.2 reason_text=ranging"
                ),
            },
        ]
        a = bm.parse_signal_activity(entries)
        assert len(a["direction_gate_skips"]) == 1
        s = a["direction_gate_skips"][0]
        assert s["symbol"] == "NVDA"
        assert s["phase"] == "ranging"
        assert s["direction"] == 1.5

    def test_captures_orders_placed(self):
        entries = [
            {"timestamp": "2026-06-01 09:37:27,040",
             "message": "Order placed: 1 31 QBTS"},
            {"timestamp": "2026-06-01 12:47:24,487",
             "message": "Order placed: 1 141 BBWI"},
        ]
        a = bm.parse_signal_activity(entries)
        assert len(a["fills"]) == 2

    def test_captures_errors(self):
        entries = [
            {"timestamp": "x", "level": "ERROR", "message": "stream connection lost"},
            {"timestamp": "y", "level": "CRITICAL", "message": "out of memory"},
            {"timestamp": "z", "level": "INFO", "message": "all good"},
        ]
        a = bm.parse_signal_activity(entries)
        assert len(a["errors"]) == 2

    def test_handles_empty_entries(self):
        a = bm.parse_signal_activity([])
        assert a["fired_by_strategy"] == {}
        assert a["block_reasons"] == {}
        assert a["fills"] == []
        assert a["errors"] == []


# ────────────────────────────────────────────────────────────────────
# parse_pnl_progression
# ────────────────────────────────────────────────────────────────────

class TestParsePnlProgression:
    def test_extracts_peak_and_trough(self):
        entries = [
            {"timestamp": "2026-06-01 10:00:00,000",
             "message": "Risk Manager synced with Schwab: P&L=$100.00, Buying Power=$0, Balance=$33000.00"},
            {"timestamp": "2026-06-01 11:00:00,000",
             "message": "Risk Manager synced with Schwab: P&L=$250.00, Buying Power=$0, Balance=$33150.00"},
            {"timestamp": "2026-06-01 12:00:00,000",
             "message": "Risk Manager synced with Schwab: P&L=$-50.00, Buying Power=$0, Balance=$32950.00"},
            {"timestamp": "2026-06-01 13:00:00,000",
             "message": "Risk Manager synced with Schwab: P&L=$80.00, Buying Power=$0, Balance=$33080.00"},
        ]
        p = bm.parse_pnl_progression(entries)
        assert p["samples"] == 4
        assert p["first"][1] == 100.0
        assert p["last"][1] == 80.0
        assert p["peak_pnl"][1] == 250.0
        assert p["trough_pnl"][1] == -50.0


# ────────────────────────────────────────────────────────────────────
# parse_stream_health
# ────────────────────────────────────────────────────────────────────

class TestParseStreamHealth:
    def test_latest_heartbeat_extracted(self):
        entries = [
            {"timestamp": "t1",
             "message": "schwab_stream.watchdog: heartbeat age=0s threshold=120s rth=True heartbeat_count=5"},
            {"timestamp": "t2",
             "message": "schwab_stream.watchdog: heartbeat age=1s threshold=120s rth=True heartbeat_count=6"},
            {"timestamp": "t3",
             "message": "schwab_stream: 12500 ticks total. Top per-symbol: AAPL:100"},
        ]
        s = bm.parse_stream_health(entries)
        assert s["latest_heartbeat"] is not None
        _, age, rth = s["latest_heartbeat"]
        assert age == 1
        assert rth is True
        assert s["tick_count"] == 12500

    def test_no_heartbeats_returns_none(self):
        s = bm.parse_stream_health([{"message": "nothing relevant"}])
        assert s["latest_heartbeat"] is None
        assert s["tick_count"] is None


# ────────────────────────────────────────────────────────────────────
# Formatting helpers
# ────────────────────────────────────────────────────────────────────

class TestFormatters:
    def test_fmt_duration(self):
        assert bm._fmt_duration(30) == "30s"
        assert bm._fmt_duration(90) == "1m 30s"
        assert bm._fmt_duration(3600 + 120) == "1h 2m"

    def test_fmt_money_positive(self):
        assert bm._fmt_money(123.45) == "$123.45"

    def test_fmt_money_negative(self):
        assert bm._fmt_money(-67.89) == "-$67.89"

    def test_fmt_money_none(self):
        assert bm._fmt_money(None) == "—"

    def test_fmt_money_thousands(self):
        assert bm._fmt_money(12345.67) == "$12,345.67"


# ────────────────────────────────────────────────────────────────────
# Smoke — build_report doesn't crash on minimum data
# ────────────────────────────────────────────────────────────────────

class TestBuildReport:
    def test_build_report_runs_without_crash(self):
        """The monitor must be robust to missing bot, missing token,
        empty log — pure observer never raises."""
        # We can't fully mock the env in this test (it'd take more
        # plumbing than the value warrants), but we can at least
        # call build_report and assert it returns a string.
        result = bm.build_report(window_seconds=60)
        assert isinstance(result, str)
        assert "Bot Monitor" in result
        assert "Process health" in result


# ── v-bot-monitor-tail-fix-2026-06-01 ────────────────────────────────

class TestTailLogSplitsLinesCorrectly:
    """Regression test: tail_log used to call `data.split('\\n', 1)[1:]`
    which produced a list of ONE string containing everything after the
    first newline, then iterated once and tried json.loads on that whole
    multi-line blob. Net effect: monitor returned 0 entries despite a
    busy log.

    Fix: use `data.splitlines()[1:]` which properly returns one entry
    per line.

    This test writes a small JSONL log fixture, monkeypatches the
    module's LOG_FILE constant, and asserts tail_log returns the
    expected number of entries.
    """

    def test_tail_log_returns_multiple_entries(self, tmp_path, monkeypatch):
        from datetime import datetime, timedelta
        # Build a fixture with 5 lines all from the last 10 seconds
        # (so they fall inside the default window).
        now = datetime.now()
        lines = []
        for i in range(5):
            ts = (now - timedelta(seconds=i)).strftime("%Y-%m-%d %H:%M:%S,000")
            lines.append(
                '{"timestamp": "' + ts + '", "level": "INFO", '
                '"message": "entry_' + str(i) + '", "module": "x", '
                '"function": "y", "line": 1}'
            )
        # Prepend a partial-looking first line that the parser is
        # expected to skip (mimics seeking mid-file).
        fixture = "...partial first line cut off\n" + "\n".join(lines) + "\n"
        log_path = tmp_path / "test_trading_bot.log"
        log_path.write_text(fixture)
        monkeypatch.setattr(bm, "LOG_FILE", log_path)

        entries = bm.tail_log(seconds_back=60)
        # 5 entries should be returned (partial line skipped).
        assert len(entries) == 5, (
            f"expected 5 entries from 5 JSON lines, got {len(entries)} — "
            "splitlines may have regressed back to split('\\n', 1)"
        )
        messages = [e["message"] for e in entries]
        assert "entry_0" in messages
        assert "entry_4" in messages

    def test_tail_log_handles_single_line(self, tmp_path, monkeypatch):
        """One-line log should still produce zero or one entry without
        crashing. The 'skip first line' rule means this returns 0."""
        from datetime import datetime
        now = datetime.now()
        ts = now.strftime("%Y-%m-%d %H:%M:%S,000")
        line = (
            '{"timestamp": "' + ts + '", "level": "INFO", '
            '"message": "only", "module": "x", "function": "y", "line": 1}\n'
        )
        log_path = tmp_path / "test_one_line.log"
        log_path.write_text(line)
        monkeypatch.setattr(bm, "LOG_FILE", log_path)

        entries = bm.tail_log(seconds_back=60)
        # First line is treated as partial -> skipped. 0 entries expected.
        # The point is no crash.
        assert isinstance(entries, list)

    def test_tail_log_filters_by_time_window(self, tmp_path, monkeypatch):
        """An entry older than the window must NOT appear in results.
        Catches a regression where time filtering breaks while line
        parsing works."""
        from datetime import datetime, timedelta
        now = datetime.now()
        old_ts = (now - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S,000")
        new_ts = (now - timedelta(seconds=5)).strftime("%Y-%m-%d %H:%M:%S,000")
        lines = [
            '{"timestamp": "' + old_ts + '", "level": "INFO", "message": "old"}',
            '{"timestamp": "' + new_ts + '", "level": "INFO", "message": "new"}',
        ]
        fixture = "...partial\n" + "\n".join(lines) + "\n"
        log_path = tmp_path / "test_window.log"
        log_path.write_text(fixture)
        monkeypatch.setattr(bm, "LOG_FILE", log_path)

        entries = bm.tail_log(seconds_back=60)
        messages = [e.get("message") for e in entries]
        assert "new" in messages
        assert "old" not in messages
