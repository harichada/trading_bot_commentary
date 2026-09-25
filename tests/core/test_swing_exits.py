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
