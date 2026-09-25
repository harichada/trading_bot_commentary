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
    from research.strategy_hunt.data.earnings import update_earnings, load_earnings, earnings_days
    fetched = pd.DatetimeIndex(["2025-02-20 16:30", "2025-05-21 08:00"])
    n = update_earnings(tmp_path, "NVDA", lambda s: fetched)
    assert n == 2
    assert load_earnings(tmp_path, "NVDA") == {pd.Timestamp("2025-02-20"), pd.Timestamp("2025-05-21")}
    d = _daily()
    assert earnings_days(tmp_path, "NVDA", d) == {pd.Timestamp("2025-02-20"), pd.Timestamp("2025-05-21"), d.index[30]}


def test_load_missing_is_empty(tmp_path):
    from research.strategy_hunt.data.earnings import load_earnings
    assert load_earnings(tmp_path, "ZZZZ") == set()
