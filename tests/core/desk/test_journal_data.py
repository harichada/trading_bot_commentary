"""v-playbook-desk-b2-2026-09-25 (Task 4): journal data adapters and file store.

The adapters in `journal_data.py` are exercised against a fake `db_logger`
(no Postgres): success paths, missing-method paths, and exception paths.

(Split from `test_journal.py` in review round 1 — see
`.superpowers/sdd/2026-09-25-playbook-desk-b2/task-4-findings-r1.md` item 2.
Rendering tests moved to `test_journal_render.py`.)

Review round 1, Important finding 1: `decisions_for` now reads every
`bot_decisions` row for the day via the new `DbLogger.get_bot_decisions_between`
(all components, all modes) instead of the narrow `get_todays_bot_decision_entries`
getter, so `strategy_fit` / `playbook` / `desk_*` shadow-and-skip audits reach
the journal. The old getter is never called (it read the very same table for
action values ("entry", "open_position", "signal_generated") that no current
write path — `core.engine.EngineWithCommentary._audit`, the table's only
writer — ever emits, so keeping it would only double count rows already
covered by `get_bot_decisions_between`).
"""
from __future__ import annotations

import asyncio
import os

from core.desk.journal_data import (
    decisions_for,
    journal_dir,
    journal_path,
    trades_for,
    write_journal,
)


class _FakeDbLogger:
    """Implements only the methods `journal_data` adapters call. `old_entries`
    and `old_getter_called` exist to prove `decisions_for` never falls back to
    the narrow `get_todays_bot_decision_entries` getter (double-count check)."""

    def __init__(self, bot_decisions=None, trades=None, old_entries=None, raise_on=None):
        self._bot_decisions = bot_decisions if bot_decisions is not None else []
        self._trades = trades if trades is not None else []
        self._old_entries = old_entries or {}
        self._raise_on = raise_on or ()
        self.decisions_between_calls = []
        self.old_getter_called = False

    async def get_bot_decisions_between(self, start, end):
        self.decisions_between_calls.append((start, end))
        if "decisions" in self._raise_on:
            raise RuntimeError("boom")
        return self._bot_decisions

    async def get_todays_bot_decision_entries(self, symbols=None):
        self.old_getter_called = True
        return self._old_entries

    async def get_bot_trades_between(self, start, end):
        if "trades" in self._raise_on:
            raise RuntimeError("boom")
        return self._trades


# ────────────────────────────────────────────────────────────────────
# decisions_for / DbLogger.get_bot_decisions_between
# ────────────────────────────────────────────────────────────────────

def test_decisions_for_shapes_rows_from_get_bot_decisions_between():
    fake = _FakeDbLogger(bot_decisions=[
        {"ts": "2026-09-25T10:02:00", "component": "strategy_fit", "symbol": "AAPL",
         "action": "skip", "reason": "fit_fail", "mode": "live", "details": {"family": "orb"}},
    ])
    out = asyncio.run(decisions_for("2026-09-25", fake))
    assert out == [{
        "symbol": "AAPL", "component": "strategy_fit", "action": "skip", "reason": "fit_fail",
        "mode": "live", "ts": "2026-09-25T10:02:00", "details": {"family": "orb"},
    }]


def test_decisions_for_includes_every_component_and_mode_unfiltered():
    fake = _FakeDbLogger(bot_decisions=[
        {"ts": "t1", "component": "desk_limits", "symbol": "MSFT", "action": "shadow",
         "reason": "would_exceed_max_trades", "mode": "sim", "details": {}},
        {"ts": "t2", "component": "playbook", "symbol": "TSLA", "action": "skip",
         "reason": "playbook_suspended", "mode": "live", "details": {}},
        {"ts": "t3", "component": "signal_router", "symbol": None, "action": "received",
         "reason": None, "mode": "live", "details": {}},
    ])
    out = asyncio.run(decisions_for("2026-09-25", fake))
    components = {row["component"] for row in out}
    assert components == {"desk_limits", "playbook", "signal_router"}
    assert len(out) == 3


def test_decisions_for_calls_get_bot_decisions_between_with_date_as_both_bounds():
    fake = _FakeDbLogger(bot_decisions=[])
    asyncio.run(decisions_for("2026-09-25", fake))
    assert fake.decisions_between_calls == [("2026-09-25", "2026-09-25")]


def test_decisions_for_never_calls_old_getter_no_double_counting():
    fake = _FakeDbLogger(
        bot_decisions=[{"ts": "t1", "component": "strategy_fit", "symbol": "AAPL",
                        "action": "skip", "reason": "fit_fail", "mode": "live", "details": {}}],
        old_entries={"AAPL": {"side": "long", "ts": "t0", "strategy": "mean_reversion",
                              "confidence": 0.8}},
    )
    out = asyncio.run(decisions_for("2026-09-25", fake))
    assert fake.old_getter_called is False
    assert len(out) == 1  # only the bot_decisions row, not the old getter's entry too


def test_decisions_for_returns_empty_on_exception():
    fake = _FakeDbLogger(raise_on=("decisions",))
    assert asyncio.run(decisions_for("2026-09-25", fake)) == []


def test_decisions_for_returns_empty_when_method_missing():
    class _NoMethod:
        pass
    assert asyncio.run(decisions_for("2026-09-25", _NoMethod())) == []


def test_decisions_for_malformed_row_skipped_not_raised():
    fake = _FakeDbLogger(bot_decisions=["not-a-dict", {"component": "strategy_fit"}])
    out = asyncio.run(decisions_for("2026-09-25", fake))
    assert len(out) == 1
    assert out[0]["component"] == "strategy_fit"
    assert out[0]["action"] == "unknown"  # missing action defaults, doesn't raise


# ────────────────────────────────────────────────────────────────────
# trades_for
# ────────────────────────────────────────────────────────────────────

def test_trades_for_shapes_rows_with_family_from_lane_map():
    fake = _FakeDbLogger(trades=[
        {"symbol": "AAPL", "side": "long", "strategy": "mean_reversion",
         "entry_time": "2026-09-25T09:35:00", "exit_time": "2026-09-25T10:00:00",
         "pnl": 120.0, "pnl_r": 0.8, "exit_reason": "target", "setup_type": "mean_rev"},
    ])
    out = asyncio.run(trades_for(("2026-09-24", "2026-09-25"), fake))
    assert len(out) == 1
    assert out[0]["family"] == "mean_rev_intraday"
    assert out[0]["r"] == 0.8


def test_trades_for_unmapped_strategy_gets_unknown_family():
    fake = _FakeDbLogger(trades=[
        {"symbol": "XYZ", "side": "short", "strategy": "some_unmapped_strategy",
         "entry_time": None, "exit_time": None, "pnl": None, "pnl_r": None,
         "exit_reason": None, "setup_type": None},
    ])
    out = asyncio.run(trades_for(("2026-09-24", "2026-09-25"), fake))
    assert out[0]["family"] == "unknown"


def test_trades_for_returns_empty_on_exception():
    fake = _FakeDbLogger(raise_on=("trades",))
    assert asyncio.run(trades_for(("2026-09-24", "2026-09-25"), fake)) == []


def test_trades_for_returns_empty_on_bad_date_range():
    fake = _FakeDbLogger(trades=[{"symbol": "AAPL"}])
    assert asyncio.run(trades_for("not-a-range", fake)) == []


# ────────────────────────────────────────────────────────────────────
# journal_path / write_journal (atomic file store)
# ────────────────────────────────────────────────────────────────────

def test_journal_path_layout(tmp_path):
    desk_dir = str(tmp_path / "desk")
    path = journal_path(desk_dir, "2026-09-25")
    assert path == os.path.join(desk_dir, "journal", "2026-09-25.md")
    assert journal_dir(desk_dir) == os.path.join(desk_dir, "journal")


def test_write_journal_round_trip_and_atomicity(tmp_path):
    desk_dir = str(tmp_path / "desk")
    text = "# journal\nsome content\n"
    written = write_journal(text, desk_dir, "2026-09-25")
    assert written == journal_path(desk_dir, "2026-09-25")
    with open(written) as fh:
        assert fh.read() == text
    leftovers = [f for f in os.listdir(journal_dir(desk_dir)) if f.endswith(".tmp")]
    assert leftovers == []


def test_write_journal_returns_none_on_failure(tmp_path, monkeypatch):
    desk_dir = str(tmp_path / "desk")

    def _boom(*a, **kw):
        raise OSError("disk full")

    import core.desk.journal_data as journal_data_module
    monkeypatch.setattr(journal_data_module.tempfile, "mkstemp", _boom)
    assert write_journal("text", desk_dir, "2026-09-25") is None
