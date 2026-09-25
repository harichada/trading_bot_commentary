"""v-playbook-desk-b2-2026-09-25 (Task 3): the `playbook` subcommand body.

`research/strategy_hunt/playbook_cli.py` is pure plumbing — engine lifecycle, rendering and
exit codes — and none of it needs a database.
"""
import argparse

import pytest

from core.desk.playbook import Playbook, PlaybookEntry
from research.strategy_hunt import playbook_cli as PC
from research.strategy_hunt import playbook_update as PU


def _args(tmp_path, **kw):
    base = dict(root=tmp_path, desk_dir=tmp_path / "desk", postgres=False, families=None)
    base.update(kw)
    return argparse.Namespace(**base)


def _entry(**kw):
    base = dict(family="orb", side="long", regime="risk_on", status="suspended", size_mult=0.0,
                exp_r_recent=-0.4, t_stat_recent=-2.0, n_recent=20, updated_at="now",
                reason="suspended_negative_expectancy")
    base.update(kw)
    return PlaybookEntry(**base)


class _FakeEngine:
    """Records dispose(); `run_playbook_command` must always release the pool it opened."""

    def __init__(self) -> None:
        self.disposed = 0

    def dispose(self) -> None:
        self.disposed += 1


@pytest.fixture
def fake_engine(monkeypatch):
    engine = _FakeEngine()
    monkeypatch.setenv("POSTGRES_DSN", "postgresql://user:pw@localhost/db")
    monkeypatch.setattr("sqlalchemy.create_engine", lambda dsn, **kw: engine)
    return engine


def test_the_engine_is_handed_to_the_updater_and_always_disposed(tmp_path, monkeypatch, fake_engine):
    """Fix r2 nit: the DSN test must actually prove dispose() ran, not just that rc == 0."""
    seen = {}

    def _fake(root, snapshot, now, **kw):
        seen["engine"] = kw["engine"]
        return Playbook(entries={}, updated_at=now)

    monkeypatch.setattr(PU, "update_playbook", _fake)

    assert PC.run_playbook_command(_args(tmp_path, postgres=True), ["toy"]) == 0
    assert seen["engine"] is fake_engine
    assert fake_engine.disposed == 1


def test_the_engine_is_disposed_even_when_the_update_fails(tmp_path, monkeypatch, fake_engine, capsys):
    monkeypatch.setattr(PU, "update_playbook",
                        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("write failed")))

    assert PC.run_playbook_command(_args(tmp_path, postgres=True), ["toy"]) == 1
    assert fake_engine.disposed == 1
    out = capsys.readouterr().out
    assert "playbook ERROR" in out and "write failed" in out and "Traceback" not in out


def test_no_engine_is_built_without_the_postgres_flag(tmp_path, monkeypatch):
    monkeypatch.setattr("sqlalchemy.create_engine",
                        lambda *a, **kw: pytest.fail("no engine may be built without --postgres"))
    seen = {}

    def _fake(root, snapshot, now, **kw):
        seen["engine"] = kw["engine"]
        return Playbook(entries={}, updated_at=now)

    monkeypatch.setattr(PU, "update_playbook", _fake)

    assert PC.run_playbook_command(_args(tmp_path), ["toy"]) == 0
    assert seen["engine"] is None


def test_render_playbook_lists_every_key_and_a_summary(tmp_path):
    pb = Playbook(entries={}, updated_at="now").with_entry(_entry()).with_entry(
        _entry(family="dt_momentum", status="enabled", size_mult=1.25, exp_r_recent=0.2,
               t_stat_recent=3.0, reason="enabled_upsize_fit_confirmed"))

    text = PC.render_playbook(pb, tmp_path / "playbook.json")

    assert "orb" in text and "dt_momentum" in text
    assert "x0.00" in text and "x1.25" in text
    assert "2 key(s)" in text and "'suspended': 1" in text and "'enabled': 1" in text
    # Worst status first: 'enabled' < 'probation' < 'suspended' alphabetically is the tie-break.
    assert text.index("dt_momentum") < text.index("orb")


def test_status_counts_is_empty_for_an_empty_playbook():
    assert PC.status_counts(Playbook()) == {}
