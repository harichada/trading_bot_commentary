"""v-playbook-desk-b2-2026-09-25 (Task 3b, fix round 1): `DbLogger.get_bot_trades_between`
and `DbLogger.get_bot_decisions_between` must bind `datetime.date` objects, not strings,
for their `DATE(...) BETWEEN :start AND :end` parameters.

Controller finding: both bound `{"start": str(start), "end": str(end)}`; asyncpg raises
`DataError: invalid input for query argument $1: '2026-09-15' ('str' object has no
attribute 'toordinal')`, which each method's broad `except Exception` turned into a
silent `[]` — so the desk journal's Trades and Decisions sections were always empty.

Offline: a fake `AsyncEngine`/connection records the exact parameter dict passed to
`conn.execute(...)` so these tests can assert on bound *types*, never touching Postgres.
"""
from __future__ import annotations

import asyncio
from datetime import date, datetime

import pytest

from data_providers.db_logger import DbLogger, _ensure_date


# ────────────────────────────────────────────────────────────────────
# _ensure_date — the pure coercion helper
# ────────────────────────────────────────────────────────────────────

def test_ensure_date_from_iso_string():
    assert _ensure_date("2026-09-15") == date(2026, 9, 15)


def test_ensure_date_from_full_timestamp_string():
    assert _ensure_date("2026-09-15T10:30:00") == date(2026, 9, 15)


def test_ensure_date_from_date_object_is_returned_as_is():
    d = date(2026, 9, 15)
    assert _ensure_date(d) is d


def test_ensure_date_from_datetime_is_truncated():
    assert _ensure_date(datetime(2026, 9, 15, 23, 59)) == date(2026, 9, 15)


def test_ensure_date_from_none_is_none():
    assert _ensure_date(None) is None


def test_ensure_date_from_unparseable_string_is_none(caplog):
    with caplog.at_level("WARNING"):
        assert _ensure_date("not-a-date") is None
    assert any("_ensure_date" in r.message for r in caplog.records)


# ────────────────────────────────────────────────────────────────────
# Fake engine/connection harness (no Postgres)
# ────────────────────────────────────────────────────────────────────

class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def mappings(self):
        return self

    def all(self):
        return self._rows


class _FakeConn:
    def __init__(self, rows, calls):
        self._rows = rows
        self._calls = calls

    async def execute(self, query, params):
        self._calls.append(dict(params))
        return _FakeResult(self._rows)


class _FakeBeginCtx:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeEngine:
    def __init__(self, rows=()):
        self.calls: list[dict] = []
        self._conn = _FakeConn(rows, self.calls)

    def begin(self):
        return _FakeBeginCtx(self._conn)


def _fake_logger(rows=()):
    """A `DbLogger` whose `_engine` is a fake that never touches Postgres."""
    db_logger = DbLogger(dsn="postgresql+asyncpg://user:pass@localhost/testdb")
    engine = _FakeEngine(rows=rows)
    db_logger._engine = engine
    return db_logger, engine


_INPUTS = [
    pytest.param("2026-09-15", "2026-09-25", id="iso-strings"),
    pytest.param(date(2026, 9, 15), date(2026, 9, 25), id="date-objects"),
    pytest.param(datetime(2026, 9, 15, 10, 0), datetime(2026, 9, 25, 16, 0), id="datetimes"),
]


@pytest.mark.parametrize("start,end", _INPUTS)
def test_get_bot_trades_between_binds_date_objects(start, end):
    db_logger, engine = _fake_logger(rows=[])
    asyncio.run(db_logger.get_bot_trades_between(start, end))

    assert len(engine.calls) == 1
    bound = engine.calls[0]
    for key, expected in (("start", date(2026, 9, 15)), ("end", date(2026, 9, 25))):
        assert isinstance(bound[key], date)
        assert not isinstance(bound[key], datetime)  # a plain date, not a datetime
        assert bound[key] == expected


@pytest.mark.parametrize("start,end", _INPUTS)
def test_get_bot_decisions_between_binds_date_objects(start, end):
    db_logger, engine = _fake_logger(rows=[])
    asyncio.run(db_logger.get_bot_decisions_between(start, end))

    assert len(engine.calls) == 1
    bound = engine.calls[0]
    for key, expected in (("start", date(2026, 9, 15)), ("end", date(2026, 9, 25))):
        assert isinstance(bound[key], date)
        assert not isinstance(bound[key], datetime)
        assert bound[key] == expected


def test_get_bot_trades_between_round_trips_a_row_on_the_un_migrated_schema():
    """The fix must not just bind correctly — the read must still succeed end to end
    on the un-migrated schema (no `pnl_r` column, R derived from `stop_loss`)."""
    row = {
        "symbol": "aaa", "side": "long", "strategy": "orb_contraction_rvol",
        "entry_time": "2026-09-20T10:00:00", "exit_time": "2026-09-20T11:00:00",
        "pnl": 40.0, "entry_price": 100.0, "stop_loss": 98.0, "quantity": 10.0,
        "exit_reason": "target", "mode": "live",
    }
    db_logger, _ = _fake_logger(rows=[row])
    out = asyncio.run(db_logger.get_bot_trades_between("2026-09-15", "2026-09-25"))

    assert len(out) == 1
    assert out[0]["symbol"] == "AAA"
    assert out[0]["pnl_r"] == 2.0  # 40 / (abs(100-98) * 10)
    assert out[0]["setup_type"] is None  # column absent on the un-migrated schema


def test_get_bot_decisions_between_round_trips_a_row():
    row = {
        "ts": "2026-09-20T10:00:00", "component": "playbook", "symbol": "aaa",
        "action": "skip", "reason": "playbook_suspended", "mode": "live",
        "details_json": {"family": "orb"},
    }
    db_logger, _ = _fake_logger(rows=[row])
    out = asyncio.run(db_logger.get_bot_decisions_between("2026-09-15", "2026-09-25"))

    assert len(out) == 1
    assert out[0]["symbol"] == "AAA"
    assert out[0]["details"] == {"family": "orb"}
