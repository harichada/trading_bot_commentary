from datetime import date, datetime, time
import pandas as pd


def test_sessions_held_counts_weekdays_inclusive():
    from core.swing_exits import sessions_held
    assert sessions_held(datetime(2025, 9, 22, 10), datetime(2025, 9, 22, 15)) == 1           # Mon same day
    assert sessions_held(datetime(2025, 9, 22, 10), datetime(2025, 9, 26, 15)) == 5           # Mon..Fri
    assert sessions_held(datetime(2025, 9, 26, 10), datetime(2025, 9, 29, 15)) == 2           # Fri..Mon skips weekend


def test_time_stop_only_multiday_and_after_1550():
    from core.swing_exits import should_time_stop
    r = {"hold_kind": "multiday", "max_hold_sessions": 3}
    assert should_time_stop(r, datetime(2025, 9, 22, 10), datetime(2025, 9, 24, 15, 40), 10) == (False, "")
    assert should_time_stop(r, datetime(2025, 9, 22, 10), datetime(2025, 9, 24, 15, 50), 10) == (True, "swing_time_stop")
    assert should_time_stop({"hold_kind": "multiday"}, datetime(2025, 9, 22, 10), datetime(2025, 9, 24, 15, 55), 2) == (True, "swing_time_stop")
    assert should_time_stop({"is_day_trade": True}, datetime(2025, 9, 22, 10), datetime(2025, 9, 30, 15, 55), 2) == (False, "")


def test_earnings_eve_flatten(tmp_path):
    from core.swing_exits import should_earnings_eve_flatten, load_earnings_dates, next_weekday
    assert next_weekday(date(2025, 9, 26)) == date(2025, 9, 29)
    e = {date(2025, 9, 29)}
    assert should_earnings_eve_flatten("X", datetime(2025, 9, 26, 15, 50), e) == (True, "earnings_eve_flatten")
    assert should_earnings_eve_flatten("X", datetime(2025, 9, 26, 12, 0), e) == (False, "")
    assert should_earnings_eve_flatten("X", datetime(2025, 9, 25, 15, 50), e) == (False, "")
    (tmp_path / "earnings").mkdir()
    pd.DataFrame({"date": pd.to_datetime(["2025-09-29"])}).to_parquet(tmp_path / "earnings" / "X.parquet", index=False)
    assert load_earnings_dates(str(tmp_path), "X") == {date(2025, 9, 29)} and load_earnings_dates(str(tmp_path), "Y") == set()


def test_load_earnings_dates_malformed_column_returns_empty_set(tmp_path):
    from core.swing_exits import load_earnings_dates
    (tmp_path / "earnings").mkdir()
    pd.DataFrame({"date": ["not-a-date", "??"]}).to_parquet(tmp_path / "earnings" / "Z.parquet", index=False)
    assert load_earnings_dates(str(tmp_path), "Z") == set()


def test_time_stop_coerces_non_numeric_max_hold_sessions():
    """v-strategy-fit-review2-2026-09-25 (Important 6): `max_hold_sessions`
    comes out of a persisted `reasoning` dict, so it can be a string or any
    other junk a past writer left in trading_state.json. It must fall back to
    the default rather than raising into _evaluate_exit_conditions, where a
    ValueError would have skipped the hard SL/TP checks below it."""
    from core.swing_exits import should_time_stop

    entry, now = datetime(2025, 9, 22, 10), datetime(2025, 9, 24, 15, 55)   # 3 sessions held

    # Non-numeric values fall back to default_max.
    for junk in ("many", "3.5.1", [], {}, object()):
        r = {"hold_kind": "multiday", "max_hold_sessions": junk}
        assert should_time_stop(r, entry, now, 2) == (True, "swing_time_stop")     # default 2 <= 3 held
        assert should_time_stop(r, entry, now, 10) == (False, "")                  # default 10 > 3 held

    # A numeric string still works (int("3") == 3), and so does a float.
    assert should_time_stop({"hold_kind": "multiday", "max_hold_sessions": "3"}, entry, now, 10) == (True, "swing_time_stop")
    assert should_time_stop({"hold_kind": "multiday", "max_hold_sessions": 3.0}, entry, now, 10) == (True, "swing_time_stop")
    # None/0 already fell back to the default before this fix; keep it that way.
    assert should_time_stop({"hold_kind": "multiday", "max_hold_sessions": None}, entry, now, 2) == (True, "swing_time_stop")


def test_overnight_hold_kind_counts_as_multiday():
    """v-batch1-task1-r1 (Important 1): an "overnight" position (entered at 15:55, carried into the
    next session) is held past the close, so both engine call sites that key off `is_multiday` must
    see it -- the SWING_ALLOW_OVERNIGHT entry gate and the swing time-stop / earnings-eve exit
    check. Leaving it out meant an overnight lane could enter while overnight holding was disabled
    and would then never be time-stopped or flattened before a print."""
    from core.swing_exits import is_multiday
    assert is_multiday({"hold_kind": "overnight"}) is True
    assert is_multiday({"hold_kind": "multiday"}) is True
    assert is_multiday({"hold_kind": "intraday"}) is False
    assert is_multiday({}) is False and is_multiday(None) is False


def test_overnight_time_stop_waits_for_the_session_it_is_carried_into():
    """An overnight family's Entry carries time_stop=1 (the simulator exits it at the next session's
    open, so the number is unused there), and that reaches live as max_hold_sessions=1. Taken
    literally, the time stop would fire at 15:5x on the entry session itself -- flattening the
    position minutes after it was opened, a zero-hold round trip that pays both legs' costs. An
    overnight hold is therefore floored at OVERNIGHT_MIN_SESSIONS (2): the entry session plus the
    one it is carried into."""
    from core.swing_exits import OVERNIGHT_MIN_SESSIONS, should_time_stop
    assert OVERNIGHT_MIN_SESSIONS == 2
    r = {"hold_kind": "overnight", "max_hold_sessions": 1}
    entry = datetime(2025, 9, 25, 15, 55)                                    # Thu 15:55
    assert should_time_stop(r, entry, datetime(2025, 9, 25, 15, 56), 10) == (False, "")
    assert should_time_stop(r, entry, datetime(2025, 9, 26, 15, 50), 10) == (True, "swing_time_stop")
    # The floor only raises a too-short hold; a longer explicit one is still honoured.
    assert should_time_stop({"hold_kind": "overnight", "max_hold_sessions": 5},
                            datetime(2025, 9, 22, 10), datetime(2025, 9, 24, 15, 55), 10) == (False, "")
