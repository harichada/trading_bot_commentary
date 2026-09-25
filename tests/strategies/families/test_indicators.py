import numpy as np
import pandas as pd
import pytest


def _session(day, base=100.0, n=78, drift=0.0):
    idx = pd.date_range(f"{day} 09:30", periods=n, freq="5min", name="ts")
    close = base + drift * np.arange(n)
    return pd.DataFrame({"open": close - 0.1, "high": close + 0.4, "low": close - 0.4, "close": close,
                         "volume": 1000 + 10 * np.arange(n)}, index=idx)


def _bars(days, **kw):
    return pd.concat([_session(d, **kw) for d in days])


def _daily(bars5):
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    out = bars5.groupby(bars5.index.normalize()).agg(agg)
    out.index.name = "date"
    return out


DAYS = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2024-09-02", periods=130)]


def test_frame_aligned_and_session_columns():
    from strategies.families.indicators import indicator_frame
    b = _bars(DAYS[:3])
    ind = indicator_frame(b, _daily(b))
    assert ind.index.equals(b.index)
    day2 = ind.loc["2024-09-03"]
    assert (day2["bar_of_day"].to_numpy() == np.arange(78)).all()
    assert day2["session_open"].iloc[0] == b.loc["2024-09-03"].iloc[0]["open"]
    assert day2["prev_close"].iloc[0] == b.loc["2024-09-02"].iloc[-1]["close"]
    assert day2["gap_pct"].iloc[5] == pytest.approx((day2["session_open"].iloc[0] / day2["prev_close"].iloc[0] - 1) * 100)


def test_vwap_and_opening_range():
    from strategies.families.indicators import indicator_frame
    b = _bars(DAYS[:2])
    ind = indicator_frame(b, _daily(b))
    d = ind.loc["2024-09-02"]
    bars = b.loc["2024-09-02"]
    tp = (bars["high"] + bars["low"] + bars["close"]) / 3
    vwap_expected = (tp * bars["volume"]).cumsum() / bars["volume"].cumsum()
    assert np.allclose(d["vwap"].to_numpy(), vwap_expected.to_numpy())
    assert np.isnan(d["or15_high"].iloc[1]) and d["or15_high"].iloc[3] == bars["high"].iloc[:3].max()
    assert np.isnan(d["or30_low"].iloc[4]) and d["or30_low"].iloc[6] == bars["low"].iloc[:6].min()
    assert (d["or15_high"].iloc[3:] == bars["high"].iloc[:3].max()).all()


def test_rvol_compares_cumulative_volume_to_prior_sessions():
    from strategies.families.indicators import indicator_frame
    days = DAYS[:25]
    b = _bars(days)
    last = b.loc[days[-1]].copy()
    b.loc[last.index, "volume"] = last["volume"] * 3
    ind = indicator_frame(b, _daily(b))
    assert ind.loc[days[-1]]["rvol"].iloc[10] == pytest.approx(3.0, rel=1e-6)
    assert np.isnan(ind.loc[days[0]]["rvol"].iloc[10])


def test_daily_features_are_lagged_one_session():
    from strategies.families.indicators import daily_features, indicator_frame
    b = _bars(DAYS, drift=0.01)
    daily = _daily(b)
    df = daily_features(daily)
    d_last = daily.index[-1]
    assert df.loc[d_last, "d_close"] == daily["close"].iloc[-2]
    assert df.loc[d_last, "d_high20"] == daily["high"].iloc[-22:-2].max()
    assert np.isnan(df.loc[daily.index[0], "d_close"])
    ind = indicator_frame(b, daily)
    assert ind.loc[DAYS[-1]]["d_close"].iloc[0] == daily["close"].iloc[-2]
    assert ind["d_sma50"].notna().sum() > 0


def test_rs_vs_spy_same_basis():
    from strategies.families.indicators import indicator_frame
    b = _bars(DAYS[:2], drift=0.1)          # stock rises 0.1/bar from 100
    spy = _bars(DAYS[:2], drift=0.0)        # flat
    ind = indicator_frame(b, _daily(b), spy5=spy)
    row = ind.loc["2024-09-02"].iloc[10]
    expected = (b.loc["2024-09-02"].iloc[10]["close"] / b.loc["2024-09-02"].iloc[0]["open"] - 1) * 100
    assert row["rs_vs_spy_pct"] == pytest.approx(expected - (100.0 / 99.9 - 1) * 100)


def test_closed_bars_drops_in_progress_bar():
    import pandas as pd
    from strategies.families.indicators import closed_bars
    idx = pd.date_range("2025-01-02 09:30", periods=5, freq="5min", name="ts")
    b = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1}, index=idx)
    assert len(closed_bars(b, pd.Timestamp("2025-01-02 09:52"))) == 4      # 09:50 bar still forming
    assert len(closed_bars(b, pd.Timestamp("2025-01-02 09:55"))) == 5


def test_safe_atr_boundary_len_equals_window_is_valid_len_below_is_nan():
    # Exercise _safe_atr directly (not via indicator_frame/daily_features): ta's ADXIndicator has its
    # own, separate out-of-bounds bug on very short 5-min frames, which would make an indicator_frame-level
    # boundary test fail for an unrelated reason. This isolates the exact fix under review.
    from strategies.families.indicators import _safe_atr
    c14 = pd.Series(range(100, 114), dtype=float)
    h14, l14 = c14 + 0.5, c14 - 0.5
    r14 = _safe_atr(h14, l14, c14, window=14)
    assert len(r14) == 14 and not np.isnan(r14.iloc[-1])   # len(c) == window: must compute, not NaN

    c13 = pd.Series(range(100, 113), dtype=float)
    h13, l13 = c13 + 0.5, c13 - 0.5
    r13 = _safe_atr(h13, l13, c13, window=14)
    assert len(r13) == 13 and np.isnan(r13.iloc[-1])   # len(c) < window: NaN, no raise
