"""v-playbook-desk-b2-2026-09-25: desk config flags — defaults and env overrides.

Mirrors tests/core/test_strategy_fit_config.py: the desk ships fully off, and
every flag is reachable from the environment so a live bot can be steered
without editing yaml.
"""
from __future__ import annotations

DESK_ENV_KEYS = (
    "ENABLE_DESK", "ENABLE_DESK_WATCHLIST", "ENABLE_DESK_LIMITS", "ENABLE_PLAYBOOK_GATE",
    "DESK_IN_PLAY_N", "DESK_SCAN_MAX_SYMBOLS", "DESK_DATA_DIR", "DESK_PLAN_TIME_ET",
    "DESK_REFRESH_TIME_ET", "DESK_JOURNAL_TIME_ET", "DESK_CALENDAR_PATH",
)


def _cfg(monkeypatch, **env):
    for k in DESK_ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    # See tests/core/test_strategy_fit_config.py for why the class is imported
    # directly instead of `import core.config as cc`.
    from core.config import Config
    return Config()


def test_desk_defaults(monkeypatch):
    c = _cfg(monkeypatch)
    assert c.ENABLE_DESK is False
    assert c.ENABLE_DESK_WATCHLIST is False
    assert c.ENABLE_DESK_LIMITS is False
    assert c.ENABLE_PLAYBOOK_GATE is False
    assert c.DESK_IN_PLAY_N == 20
    assert c.DESK_SCAN_MAX_SYMBOLS == 60
    assert c.DESK_DATA_DIR == "data/desk"
    assert c.DESK_PLAN_TIME_ET == "09:00"
    assert c.DESK_REFRESH_TIME_ET == "09:35"
    assert c.DESK_JOURNAL_TIME_ET == "16:05"
    assert c.DESK_CALENDAR_PATH == "data/desk/calendar.json"


def test_desk_env_overrides(monkeypatch):
    c = _cfg(
        monkeypatch,
        ENABLE_DESK="1", ENABLE_DESK_WATCHLIST="true", ENABLE_DESK_LIMITS="yes",
        ENABLE_PLAYBOOK_GATE="on", DESK_IN_PLAY_N="12", DESK_SCAN_MAX_SYMBOLS="150",
        DESK_DATA_DIR="/srv/desk", DESK_PLAN_TIME_ET="08:45",
        DESK_REFRESH_TIME_ET="09:40", DESK_JOURNAL_TIME_ET="16:30",
        DESK_CALENDAR_PATH="/srv/desk/cal.json",
    )
    assert c.ENABLE_DESK is True
    assert c.ENABLE_DESK_WATCHLIST is True
    assert c.ENABLE_DESK_LIMITS is True
    assert c.ENABLE_PLAYBOOK_GATE is True
    assert c.DESK_IN_PLAY_N == 12
    assert c.DESK_SCAN_MAX_SYMBOLS == 150
    assert c.DESK_DATA_DIR == "/srv/desk"
    assert c.DESK_PLAN_TIME_ET == "08:45"
    assert c.DESK_REFRESH_TIME_ET == "09:40"
    assert c.DESK_JOURNAL_TIME_ET == "16:30"
    assert c.DESK_CALENDAR_PATH == "/srv/desk/cal.json"


def test_desk_bool_flags_off_for_falsey_env(monkeypatch):
    c = _cfg(monkeypatch, ENABLE_DESK="0", ENABLE_DESK_WATCHLIST="false",
             ENABLE_DESK_LIMITS="off", ENABLE_PLAYBOOK_GATE="no")
    assert (c.ENABLE_DESK, c.ENABLE_DESK_WATCHLIST,
            c.ENABLE_DESK_LIMITS, c.ENABLE_PLAYBOOK_GATE) == (False, False, False, False)


def test_desk_yaml_fallback_is_read(monkeypatch):
    """With no env var set, the yaml key `trading.<lower_name>` governs."""
    c = _cfg(monkeypatch)
    overrides = {
        "trading.enable_desk": True,
        "trading.enable_desk_watchlist": True,
        "trading.enable_desk_limits": True,
        "trading.enable_playbook_gate": True,
        "trading.desk_in_play_n": 7,
        "trading.desk_scan_max_symbols": 11,
        "trading.desk_data_dir": "yaml/desk",
        "trading.desk_plan_time_et": "09:05",
        "trading.desk_refresh_time_et": "09:45",
        "trading.desk_journal_time_et": "16:10",
        "trading.desk_calendar_path": "yaml/desk/cal.json",
    }
    real_get = c.manager.get
    monkeypatch.setattr(c.manager, "get",
                        lambda key, default=None: overrides.get(key, real_get(key, default)))
    assert c.ENABLE_DESK is True and c.ENABLE_DESK_WATCHLIST is True
    assert c.ENABLE_DESK_LIMITS is True and c.ENABLE_PLAYBOOK_GATE is True
    assert c.DESK_IN_PLAY_N == 7 and c.DESK_SCAN_MAX_SYMBOLS == 11
    assert c.DESK_DATA_DIR == "yaml/desk" and c.DESK_PLAN_TIME_ET == "09:05"
    assert c.DESK_REFRESH_TIME_ET == "09:45" and c.DESK_JOURNAL_TIME_ET == "16:10"
    assert c.DESK_CALENDAR_PATH == "yaml/desk/cal.json"
