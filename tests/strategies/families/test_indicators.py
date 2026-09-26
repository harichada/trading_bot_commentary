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


def test_indicator_frame_without_context_has_no_ctx_columns():
    from strategies.families.indicators import indicator_frame
    b = _bars(DAYS[:2])
    ind = indicator_frame(b, _daily(b))
    assert not any(c.startswith("ctx_") for c in ind.columns)


def test_indicator_frame_joins_ctx_columns_per_session():
    from strategies.families.indicators import indicator_frame
    b = _bars(DAYS[:2])
    context = pd.DataFrame(
        {"rvol_rank5": [1, 5], "adv_bucket": ["mega", "small"]},
        index=pd.DatetimeIndex([DAYS[0], DAYS[1]], name="date"),
    )
    ind = indicator_frame(b, _daily(b), context=context)
    # bar 0 is masked (v-playbook-desk-b1-6 Minor 6: rvol_rank5 ranks rvol_open5, which needs the
    # first 5-minute bar to have closed); every later bar carries the session's value.
    assert (ind.loc[DAYS[0], "ctx_rvol_rank5"].iloc[1:] == 1).all()
    assert (ind.loc[DAYS[1], "ctx_rvol_rank5"].iloc[1:] == 5).all()
    assert ind.loc[DAYS[0], "ctx_rvol_rank5"].iloc[:1].isna().all()
    assert (ind.loc[DAYS[0], "ctx_adv_bucket"] == "mega").all()
    assert (ind.loc[DAYS[1], "ctx_adv_bucket"] == "small").all()
    # unrelated columns are unaffected by the join
    assert ind.loc[DAYS[0], "bar_of_day"].iloc[0] == 0
    assert ind.loc[DAYS[1]]["session_open"].iloc[0] == b.loc[DAYS[1]].iloc[0]["open"]


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


# ---- v-playbook-desk-b1-6 (final review, Minor 6) — known-at-open ctx guard ---------------------

def _ctx_frame(days):
    return pd.DataFrame({"rvol_open5": 1.5, "rvol_open15": 2.5, "rvol_rank5": 3.0, "gap_pct": 1.0,
                         "atr14_pct": 2.0},
                        index=pd.DatetimeIndex([pd.Timestamp(d) for d in days], name="date"))


def test_ctx_opening_columns_are_nan_before_their_bars_have_closed():
    """`rvol_open5` needs the first 5-minute bar to have closed and `rvol_open15` the first
    three; the universe-context frame is per *session*, so the join would otherwise hand a
    bar-0 caller a number built from bars that had not happened yet. The guard makes the
    known-at-open contract mechanical rather than a comment in context.py."""
    from strategies.families.indicators import indicator_frame
    b = _bars(DAYS[:2], n=20)
    ind = indicator_frame(b, _daily(b), context=_ctx_frame(DAYS[:2]))
    for day in DAYS[:2]:
        d = ind.loc[day]
        assert np.isnan(d["ctx_rvol_open5"].iloc[0]) and np.isnan(d["ctx_rvol_rank5"].iloc[0])
        assert (d["ctx_rvol_open5"].iloc[1:] == 1.5).all()
        assert (d["ctx_rvol_rank5"].iloc[1:] == 3.0).all()
        assert d["ctx_rvol_open15"].iloc[:3].isna().all()
        assert (d["ctx_rvol_open15"].iloc[3:] == 2.5).all()
        # columns that are genuinely known at the open are untouched
        assert (d["ctx_gap_pct"] == 1.0).all() and (d["ctx_atr14_pct"] == 2.0).all()


def test_ctx_guard_is_a_no_op_when_those_columns_are_absent():
    from strategies.families.indicators import indicator_frame
    b = _bars(DAYS[:2], n=20)
    ctx = pd.DataFrame({"gap_pct": 1.0}, index=pd.DatetimeIndex([pd.Timestamp(d) for d in DAYS[:2]],
                                                                name="date"))
    ind = indicator_frame(b, _daily(b), context=ctx)
    assert (ind["ctx_gap_pct"] == 1.0).all() and "ctx_rvol_open5" not in ind.columns


# ---- Batch-1 Task 1: slower-family daily features ---------------------------------------------

def _mk_daily(closes, start="2025-01-06"):
    idx = pd.bdate_range(start, periods=len(closes), name="date")
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame({"open": c, "high": c + 0.5, "low": c - 0.5, "close": c, "volume": 1000.0}, index=idx)


def test_new_daily_features_inherit_the_one_session_lag():
    import ta
    from strategies.families.indicators import daily_features
    daily = _mk_daily([100.0 + 0.7 * i - (3.0 if i % 5 == 0 else 0.0) for i in range(210)])
    c = daily["close"]
    df = daily_features(daily)
    assert df["d_sma200"].iloc[-1] == pytest.approx(c.rolling(200).mean().iloc[-2])
    assert df["d_sma5"].iloc[-1] == pytest.approx(c.rolling(5).mean().iloc[-2])
    assert df["d_rsi2"].iloc[-1] == pytest.approx(ta.momentum.rsi(c, window=2).iloc[-2])
    assert np.isnan(df["d_sma200"].iloc[0]) and np.isnan(df["d_rsi2"].iloc[0])


def test_down_streak_counts_consecutive_down_closes_through_the_prior_session():
    from strategies.families.indicators import daily_features
    df = daily_features(_mk_daily([100.0, 99.0, 98.0, 97.0, 98.0, 97.0]))
    assert np.isnan(df["d_down_streak"].iloc[0])
    assert list(df["d_down_streak"].iloc[1:]) == [0.0, 1.0, 2.0, 3.0, 0.0]


def test_week_start_is_the_current_sessions_own_property_not_lagged():
    from strategies.families.indicators import daily_features
    daily = _mk_daily([100.0 + i for i in range(10)])          # Mon 2025-01-06 .. Fri 2025-01-17
    df = daily_features(daily)
    assert list(df["d_week_start"]) == [0.0] * 5 + [1.0] + [0.0] * 4
    assert df["d_close"].iloc[5] == pytest.approx(daily["close"].iloc[4])   # the d_* lag still holds
    # a Tuesday after a Monday holiday still starts the week (weekday 1 <= previous weekday 4)
    holiday = daily.drop(index=[pd.Timestamp("2025-01-13")])
    assert list(daily_features(holiday)["d_week_start"])[5] == 1.0


def test_new_daily_features_reach_the_indicator_frame():
    from strategies.families.indicators import indicator_frame
    b = _bars(DAYS[:3])
    ind = indicator_frame(b, _daily(b))
    for name in ("d_rsi2", "d_sma200", "d_sma5", "d_down_streak", "d_week_start"):
        assert name in ind.columns
    assert ind["d_week_start"].notna().all()


# ---- Batch-2 Task C: d_month_start --------------------------------------------------------------

def test_month_start_flags_exactly_the_first_session_of_each_month():
    from strategies.families.indicators import daily_features
    daily = _mk_daily([100.0 + i for i in range(65)], start="2025-01-06")   # Mon 2025-01-06 ..
    df = daily_features(daily)
    months = df.index.to_series().dt.to_period("M")
    first_of_month = ~months.duplicated()
    # Mirrors _week_start's own documented edge case: the very first row in the frame has no
    # predecessor, so it can't be flagged even though it is trivially the first session of its
    # month.
    expected = first_of_month.copy()
    expected.iloc[0] = False
    assert list(df["d_month_start"]) == list(expected.astype(float))
    assert df["d_close"].iloc[22] == pytest.approx(daily["close"].iloc[21])   # the d_* lag still holds


def test_month_start_is_the_current_sessions_own_property_not_lagged():
    from strategies.families.indicators import daily_features
    daily = _mk_daily([100.0 + i for i in range(45)], start="2025-01-06")
    df = daily_features(daily)
    # 2025-02-03 is the first trading session of February -> flagged; the day before is not.
    assert df.loc["2025-02-03", "d_month_start"] == 1.0
    assert df.loc["2025-01-31", "d_month_start"] == 0.0


def test_new_daily_features_reach_the_indicator_frame_including_month_start():
    from strategies.families.indicators import indicator_frame
    b = _bars(DAYS[:3])
    ind = indicator_frame(b, _daily(b))
    assert "d_month_start" in ind.columns
    assert ind["d_month_start"].notna().all()
