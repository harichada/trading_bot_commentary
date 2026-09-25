import numpy as np
import pandas as pd


def _daily(n=40, gap_day=30):
    idx = pd.bdate_range("2025-01-02", periods=n, name="date")
    close = np.full(n, 100.0)
    open_ = close.copy()
    vol = np.full(n, 1_000_000)
    open_[gap_day] = 112.0          # +12% gap
    close[gap_day:] = 112.0
    vol[gap_day] = 5_000_000        # 5x volume
    return pd.DataFrame({"open": open_, "high": close + 1, "low": open_ - 1, "close": close, "volume": vol}, index=idx)


def test_infer_event_days_finds_gap_with_volume():
    from research.strategy_hunt.data.earnings import infer_event_days
    d = _daily()
    days = infer_event_days(d)
    assert days == {d.index[30]}


def test_infer_event_days_ignores_gap_without_volume():
    from research.strategy_hunt.data.earnings import infer_event_days
    d = _daily()
    d.loc[d.index[30], "volume"] = 1_000_000
    assert infer_event_days(d) == set()


def test_update_and_load_earnings_roundtrip(tmp_path):
    from research.strategy_hunt.data.earnings import (
        update_earnings,
        load_earnings,
        load_earnings_table,
        earnings_days,
    )
    fetched = pd.DatetimeIndex(["2025-02-20 16:30", "2025-05-21 08:00"])
    n = update_earnings(tmp_path, "NVDA", lambda s: fetched)
    assert n == 2
    assert load_earnings(tmp_path, "NVDA") == {pd.Timestamp("2025-02-20"), pd.Timestamp("2025-05-21")}
    d = _daily()
    assert earnings_days(tmp_path, "NVDA", d) == {pd.Timestamp("2025-02-20"), pd.Timestamp("2025-05-21"), d.index[30]}
    # legacy DatetimeIndex fetcher still works: surprise columns are present but NaN
    table = load_earnings_table(tmp_path, "NVDA")
    assert list(table.columns) == ["eps_estimate", "reported_eps", "surprise_pct"]
    assert table["eps_estimate"].isna().all()
    assert table["reported_eps"].isna().all()
    assert table["surprise_pct"].isna().all()
    assert set(table.index) == {pd.Timestamp("2025-02-20"), pd.Timestamp("2025-05-21")}


def test_load_missing_is_empty(tmp_path):
    from research.strategy_hunt.data.earnings import load_earnings
    assert load_earnings(tmp_path, "ZZZZ") == set()


def test_load_earnings_table_missing_is_empty(tmp_path):
    from research.strategy_hunt.data.earnings import load_earnings_table
    table = load_earnings_table(tmp_path, "ZZZZ")
    assert list(table.columns) == ["eps_estimate", "reported_eps", "surprise_pct"]
    assert table.empty


def test_update_and_load_earnings_table_with_surprise(tmp_path):
    from research.strategy_hunt.data.earnings import update_earnings, load_earnings, load_earnings_table
    fetched = pd.DataFrame(
        {
            "EPS Estimate": [1.10, 0.95],
            "Reported EPS": [1.15, np.nan],
            "Surprise(%)": [5.5, np.nan],
        },
        index=pd.DatetimeIndex(["2025-02-20 16:30", "2025-05-21 08:00"], tz="America/New_York"),
    )
    n = update_earnings(tmp_path, "NVDA", lambda s: fetched)
    assert n == 2
    assert load_earnings(tmp_path, "NVDA") == {pd.Timestamp("2025-02-20"), pd.Timestamp("2025-05-21")}

    table = load_earnings_table(tmp_path, "NVDA")
    assert list(table.columns) == ["eps_estimate", "reported_eps", "surprise_pct"]
    assert set(table.index) == {pd.Timestamp("2025-02-20"), pd.Timestamp("2025-05-21")}
    row = table.loc[pd.Timestamp("2025-02-20")]
    assert row["eps_estimate"] == 1.10
    assert row["reported_eps"] == 1.15
    assert row["surprise_pct"] == 5.5
    nan_row = table.loc[pd.Timestamp("2025-05-21")]
    assert np.isnan(nan_row["reported_eps"])
    assert np.isnan(nan_row["surprise_pct"])


def test_update_earnings_dataframe_missing_columns_defaults_to_nan(tmp_path):
    """A DataFrame lacking the surprise columns entirely still round-trips (all-NaN table)."""
    from research.strategy_hunt.data.earnings import update_earnings, load_earnings_table
    fetched = pd.DataFrame(index=pd.DatetimeIndex(["2025-03-01"]))
    n = update_earnings(tmp_path, "AAPL", lambda s: fetched)
    assert n == 1
    table = load_earnings_table(tmp_path, "AAPL")
    assert table.loc[pd.Timestamp("2025-03-01")].isna().all()
