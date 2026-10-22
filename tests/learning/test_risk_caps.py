"""v-learning-loop-2026-10-22: 1R clamp, stop/R validation, realized-loss halt."""
from datetime import date, datetime

from core.learning.risk_caps import clamp_shares_to_risk, daily_halt, realized_bot_pnl_today, validate_stop


def test_clamp():
    assert clamp_shares_to_risk(100, 2.0, 150) == 75
    assert clamp_shares_to_risk(50, 2.0, 150) == 50          # never increases
    assert clamp_shares_to_risk(10, 200.0, 150) == 0          # one share > cap -> no trade
    assert clamp_shares_to_risk(10, 0, 150) == 10             # bad input -> unchanged
    assert clamp_shares_to_risk(10, "x", 150) == 10
    assert clamp_shares_to_risk(3, 50.0, 150) == 3            # exactly 150


def test_validate_stop():
    assert validate_stop("long", 100, 98) == (True, "stop_ok")
    assert validate_stop("long", 100, 101)[0] is False
    assert validate_stop("short", 100, 101) == (True, "stop_ok")
    assert validate_stop("short", 100, 99)[0] is False
    assert validate_stop("long", 100, None) == (False, "stop_missing")
    assert validate_stop("long", 100, 0) == (False, "stop_missing")
    assert validate_stop("long", 100, float("nan")) == (False, "stop_missing")


def test_realized_today_excludes_denylist_external_and_other_days():
    today = date(2026, 10, 22)
    hist = [
        {"symbol": "NVDA", "exit_time": datetime(2026, 10, 22, 10), "pnl": -300},
        {"symbol": "HQGE", "exit_time": datetime(2026, 10, 22, 10), "pnl": -900},
        {"symbol": "spcx", "exit_time": "2026-10-22T11:00:00", "pnl": -900},
        {"symbol": "PLTR", "exit_time": datetime(2026, 10, 22, 10), "pnl": -900, "managed_by_bot": False},
        {"symbol": "MU", "exit_time": datetime(2026, 10, 22, 10), "pnl": -900, "is_external": True},
        {"symbol": "AMD", "exit_time": datetime(2026, 10, 21, 10), "pnl": -900},
        {"symbol": "AMD", "exit_time": "2026-10-22T12:00:00", "pnl": -250},
        {"symbol": "OPEN", "pnl": -999},                      # still open -> no exit_time
        "garbage",
    ]
    assert realized_bot_pnl_today(hist, today, frozenset({"HQGE", "SPCX"})) == -550


def test_daily_halt():
    assert daily_halt(-500, 500)[0] is True
    assert daily_halt(-499.99, 500)[0] is False
    assert daily_halt(200, 500)[0] is False
    assert daily_halt("bad", 500)[0] is False
