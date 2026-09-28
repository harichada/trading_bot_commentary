import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo
from types import SimpleNamespace
from core.loops import desk_loop as dl

ET = ZoneInfo("America/New_York")


class _Engine:
    def __init__(self):
        self.audits = []

    def _audit(self, component, symbol, action, reason, /, **details):
        self.audits.append((component, action, reason, details))


def _cfg(tmp_path):
    return SimpleNamespace(DESK_DATA_DIR=str(tmp_path), SETTINGS_PROMOTE_TIME_ET="09:25",
                           DESK_PLAN_TIME_ET="09:00", DESK_REFRESH_TIME_ET="09:35", DESK_JOURNAL_TIME_ET="16:05")


def test_settings_step_scheduled_first(tmp_path):
    loop = dl.DeskLoop(_Engine(), config_fn=lambda: _cfg(tmp_path))
    names = [n for n, _, _ in loop._schedule(_cfg(tmp_path))]
    assert names[0] == "settings"


def test_settings_step_promotes_and_audits(tmp_path, monkeypatch):
    calls = []

    def fake_promote(manager, desk_dir, *, now):
        calls.append((desk_dir, now))
        return [{"key": "risk_per_trade_pct", "old": 0.01, "new": 0.02, "effective_at": "x"}]

    monkeypatch.setattr(dl, "promote_pending", fake_promote)
    eng = _Engine()
    loop = dl.DeskLoop(eng, config_fn=lambda: _cfg(tmp_path))
    now = datetime(2026, 9, 29, 9, 26, tzinfo=ET)
    asyncio.run(loop._step_settings(_cfg(tmp_path), now, "2026-09-29"))
    assert calls and calls[0][0] == str(tmp_path)
    assert any(a[0] == "settings_console" and a[1] == "promoted" for a in eng.audits)
