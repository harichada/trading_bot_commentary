"""Per-symbol universe-context features: RVOL, gap, relative strength, catalysts, ADV buckets.

The on-disk side (`build_universe_context` / `load_context_slice`) lives in
`test_context_builder_io.py` -- split out in v-playbook-desk-b1-6 (final review Minor 3) to keep
both modules inside the test line budget.
"""
import numpy as np
import pandas as pd
import pytest

N = 40
DATES = pd.bdate_range("2025-01-02", periods=N, name="date")
# DATES[1] = Fri 2025-01-03, DATES[2] = Mon 2025-01-06: a weekend gap for the "earnings date
# strictly between two sessions" case.
SPIKE_DAY = 20         # session index with the RVOL spike
GAP_DAY = 25           # session index with the 2% opening gap (symbol A only)
NEWS_DAY = 20
EARNINGS_EQ_DAY = 10   # earnings date == this session; earnings_prev fires on EARNINGS_EQ_DAY + 1


def _bars5(bar0_volumes: dict[int, float], base_bar0: float = 1000.0) -> pd.DataFrame:
    """3 bars/session at 09:30/09:35/09:40. Only the first-bar volume is ever read by
    symbol_daily_features; bar1/bar2 volumes are a flat 500 and OHLC values are placeholders."""
    rows = []
    idx = []
    for i, d in enumerate(DATES):
        bar0 = bar0_volumes.get(i, base_bar0)
        for j, vol in enumerate((bar0, 500.0, 500.0)):
            idx.append(d + pd.Timedelta(minutes=5 * j))
            rows.append({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": vol})
    df = pd.DataFrame(rows, index=pd.DatetimeIndex(idx, name="ts"))
    return df


def _daily(close: np.ndarray, open_: np.ndarray, volume: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {"open": open_, "high": close + 1, "low": close - 1, "close": close, "volume": volume},
        index=DATES,
    )


def _rising_daily(daily_volume: float) -> pd.DataFrame:
    """close[t] = 100+t (monotonic), open[t] = prior close (no gap), flat daily volume."""
    close = 100.0 + np.arange(N)
    open_ = close.copy()
    open_[1:] = close[:-1]
    open_[0] = 100.0
    return _daily(close, open_, np.full(N, daily_volume))


def _spy_daily(flat_close: float = 200.0) -> pd.DataFrame:
    close = np.full(N, flat_close)
    return _daily(close, close.copy(), np.full(N, 1_000_000.0))


def _empty_earnings() -> pd.DataFrame:
    empty = pd.DataFrame(columns=["eps_estimate", "reported_eps", "surprise_pct"])
    empty.index = pd.DatetimeIndex([], name="date")
    return empty


def _empty_news() -> pd.DataFrame:
    return pd.DataFrame(columns=["created_at", "headline", "source", "sentiment"])


# --- rvol_open5 / rvol_rank5 / gap_pct / rs_20 / pct_from_52w_high --------------------------


def test_rvol_open5_hand_computed_on_spike_day():
    from research.strategy_hunt.context import symbol_daily_features

    # baseline first-bar volume is 1000 every day except the spike day -> the prior-14-session
    # mean (sessions 6..19) is exactly 1000, so rvol_open5[SPIKE_DAY] = spike / 1000.
    bars5 = _bars5({SPIKE_DAY: 3000.0})
    daily = _rising_daily(daily_volume=20_000_000.0)
    spy = _spy_daily()
    feat = symbol_daily_features("A", bars5, daily, spy, _empty_earnings(), _empty_news())
    assert feat.loc[DATES[SPIKE_DAY], "rvol_open5"] == pytest.approx(3.0)
    # fewer than 5 prior sessions -> NaN
    assert pd.isna(feat.loc[DATES[3], "rvol_open5"])


def test_rvol_rank5_orders_symbols_on_spike_day():
    from research.strategy_hunt.context import rank_cross_section, symbol_daily_features

    spy = _spy_daily()
    specs = {"A": 3000.0, "B": 2000.0, "C": 1000.0}  # C has no spike (rvol == 1.0)
    frames = {}
    for sym, spike_vol in specs.items():
        bars5 = _bars5({SPIKE_DAY: spike_vol})
        daily = _rising_daily(daily_volume=1_000_000.0)
        frames[sym] = symbol_daily_features(sym, bars5, daily, spy, _empty_earnings(), _empty_news())

    combined = rank_cross_section(frames)
    on_day = combined[combined["date"] == DATES[SPIKE_DAY]].set_index("symbol")
    assert on_day.loc["A", "rvol_rank5"] == 1
    assert on_day.loc["B", "rvol_rank5"] == 2
    assert on_day.loc["C", "rvol_rank5"] == 3


def test_gap_pct_hand_computed():
    from research.strategy_hunt.context import symbol_daily_features

    daily = _rising_daily(daily_volume=1_000_000.0)
    daily.loc[DATES[GAP_DAY], "open"] = daily.loc[DATES[GAP_DAY - 1], "close"] * 1.02
    bars5 = _bars5({})
    spy = _spy_daily()
    feat = symbol_daily_features("A", bars5, daily, spy, _empty_earnings(), _empty_news())
    assert feat.loc[DATES[GAP_DAY], "gap_pct"] == pytest.approx(2.0)
    assert pd.isna(feat.loc[DATES[0], "gap_pct"])  # no prior close


def test_rs_20_equals_symbol_return_minus_flat_spy():
    from research.strategy_hunt.context import symbol_daily_features

    daily = _rising_daily(daily_volume=1_000_000.0)  # close[t] = 100+t
    bars5 = _bars5({})
    spy = _spy_daily()  # flat -> spy_ret == 0 for any window
    feat = symbol_daily_features("A", bars5, daily, spy, _empty_earnings(), _empty_news())
    close24, close4 = 100.0 + 24, 100.0 + 4  # close[GAP_DAY-1], close[GAP_DAY-1-20]
    expected = (close24 / close4 - 1) * 100
    assert feat.loc[DATES[GAP_DAY], "rs_20"] == pytest.approx(expected)
    # too little history for rs_60/rs_120 in a 40-session frame
    assert feat["rs_60"].isna().all() and feat["rs_120"].isna().all()


def test_rs_20_hand_computed_with_non_flat_spy():
    """A flat SPY can't detect a bug that drops the SPY leg entirely (rs_N would still equal
    sym_ret by coincidence). Use a declining SPY so the two legs clearly differ."""
    from research.strategy_hunt.context import symbol_daily_features

    daily = _rising_daily(daily_volume=1_000_000.0)  # close[t] = 100+t
    bars5 = _bars5({})
    spy_close = 300.0 - 0.5 * np.arange(N)  # spy declines -> spy_ret is negative
    spy = _daily(spy_close, spy_close.copy(), np.full(N, 1_000_000.0))
    feat = symbol_daily_features("A", bars5, daily, spy, _empty_earnings(), _empty_news())

    close24, close4 = 100.0 + 24, 100.0 + 4
    spy24, spy4 = spy_close[24], spy_close[4]
    expected = ((close24 / close4 - 1) - (spy24 / spy4 - 1)) * 100
    assert feat.loc[DATES[GAP_DAY], "rs_20"] == pytest.approx(expected)
    # sanity: this must differ from the flat-SPY case, proving the SPY leg is actually subtracted
    flat_expected = (close24 / close4 - 1) * 100
    assert feat.loc[DATES[GAP_DAY], "rs_20"] != pytest.approx(flat_expected)


def test_rs_reindex_nan_when_spy_calendar_missing_a_date():
    """Exercises the `.reindex(sym_ret.index)` alignment: a session present in the symbol's
    calendar but entirely absent from SPY's must produce NaN (not a crash, not a stale/mislabeled
    value borrowed from a neighboring row)."""
    from research.strategy_hunt.context import symbol_daily_features

    daily = _rising_daily(daily_volume=1_000_000.0)  # close[t] = 100+t
    bars5 = _bars5({})
    spy_close = 300.0 - 0.5 * np.arange(N)
    spy_full = _daily(spy_close, spy_close.copy(), np.full(N, 1_000_000.0))
    dropped_day = 35  # after GAP_DAY's rs_20 window [DATES[4]..DATES[24]] -> that window is unaffected
    spy_gap = spy_full.drop(index=DATES[dropped_day])

    feat = symbol_daily_features("A", bars5, daily, spy_gap, _empty_earnings(), _empty_news())

    assert pd.isna(feat.loc[DATES[dropped_day], "rs_20"])
    close24, close4 = 100.0 + 24, 100.0 + 4
    spy24, spy4 = spy_close[24], spy_close[4]
    expected = ((close24 / close4 - 1) - (spy24 / spy4 - 1)) * 100
    assert feat.loc[DATES[GAP_DAY], "rs_20"] == pytest.approx(expected)


def test_pct_from_52w_high_all_nan_with_short_history():
    """min_periods=HIGH_MIN_PERIODS (126) means a 40-session fixture never accumulates enough
    history to produce a value at all -- see test_pct_from_52w_high_nan_during_warmup_then_finite
    below for the warmup -> finite transition on a longer fixture."""
    from research.strategy_hunt.context import symbol_daily_features

    daily = _rising_daily(daily_volume=1_000_000.0)  # strictly increasing close
    bars5 = _bars5({})
    spy = _spy_daily()
    feat = symbol_daily_features("A", bars5, daily, spy, _empty_earnings(), _empty_news())
    assert feat["pct_from_52w_high"].isna().all()


# --- warmup (min_periods) and one-session-lag pinning for pct_from_52w_high / adv_dollars_20 /
# --- atr14_pct -----------------------------------------------------------------------------


def _long_rising_frame(n: int, daily_volume: float = 1_000_000.0):
    """Same shape as _rising_daily/_spy_daily/_bars5 but for an arbitrary session count, with an
    empty bars5 (rvol_* isn't under test here)."""
    dates = pd.bdate_range("2020-01-02", periods=n, name="date")
    close = 100.0 + np.arange(n)
    open_ = close.copy()
    open_[1:] = close[:-1]
    open_[0] = 100.0
    daily = pd.DataFrame(
        {"open": open_, "high": close + 1, "low": close - 1, "close": close, "volume": np.full(n, daily_volume)},
        index=dates,
    )
    spy_close = np.full(n, 200.0)
    spy = pd.DataFrame(
        {"open": spy_close, "high": spy_close + 1, "low": spy_close - 1, "close": spy_close,
         "volume": np.full(n, 1_000_000.0)},
        index=dates,
    )
    bars5 = pd.DataFrame(columns=["open", "high", "low", "close", "volume"], index=pd.DatetimeIndex([], name="ts"))
    return dates, daily, spy, bars5


def test_pct_from_52w_high_nan_during_warmup_then_finite():
    from research.strategy_hunt.context import HIGH_MIN_PERIODS, symbol_daily_features

    dates, daily, spy, bars5 = _long_rising_frame(HIGH_MIN_PERIODS + 14)
    feat = symbol_daily_features("A", bars5, daily, spy, _empty_earnings(), _empty_news())
    # fewer than HIGH_MIN_PERIODS (126) prior sessions -> NaN
    assert feat.iloc[:HIGH_MIN_PERIODS]["pct_from_52w_high"].isna().all()
    # exactly HIGH_MIN_PERIODS prior sessions -> first finite value
    assert pd.notna(feat.loc[dates[HIGH_MIN_PERIODS], "pct_from_52w_high"])
    assert feat.loc[dates[HIGH_MIN_PERIODS], "pct_from_52w_high"] == pytest.approx(0.0, abs=1e-9)


def test_adv_dollars_20_nan_during_warmup_then_finite():
    from research.strategy_hunt.context import ADV_MIN_PERIODS, symbol_daily_features

    dates, daily, spy, bars5 = _long_rising_frame(30)
    feat = symbol_daily_features("A", bars5, daily, spy, _empty_earnings(), _empty_news())
    assert feat.iloc[:ADV_MIN_PERIODS]["adv_dollars_20"].isna().all()
    assert pd.notna(feat.loc[dates[ADV_MIN_PERIODS], "adv_dollars_20"])


def test_adv_dollars_20_uses_only_prior_sessions():
    """One-session lag, pinned by hand: today's own dollar volume must not leak into today's
    own adv_dollars_20, but it must show up in tomorrow's."""
    from research.strategy_hunt.context import symbol_daily_features

    dates, daily, spy, bars5 = _long_rising_frame(15)
    perturbed = daily.copy()
    perturbed.loc[dates[12], "volume"] = 999_000_000.0  # huge volume, day 12 only

    base = symbol_daily_features("A", bars5, daily, spy, _empty_earnings(), _empty_news())
    pert = symbol_daily_features("A", bars5, perturbed, spy, _empty_earnings(), _empty_news())

    assert pd.notna(base.loc[dates[12], "adv_dollars_20"])
    assert base.loc[dates[12], "adv_dollars_20"] == pytest.approx(pert.loc[dates[12], "adv_dollars_20"])
    assert base.loc[dates[13], "adv_dollars_20"] != pytest.approx(pert.loc[dates[13], "adv_dollars_20"])


def test_atr14_pct_uses_only_prior_sessions():
    """One-session lag, pinned by hand: today's own high/low range must not leak into today's
    own atr14_pct, but it must show up in tomorrow's."""
    from research.strategy_hunt.context import symbol_daily_features

    dates, daily, spy, bars5 = _long_rising_frame(20)
    perturbed = daily.copy()
    perturbed.loc[dates[15], "high"] = perturbed.loc[dates[15], "close"] + 50.0  # huge range, day 15 only
    perturbed.loc[dates[15], "low"] = perturbed.loc[dates[15], "close"] - 50.0

    base = symbol_daily_features("A", bars5, daily, spy, _empty_earnings(), _empty_news())
    pert = symbol_daily_features("A", bars5, perturbed, spy, _empty_earnings(), _empty_news())

    assert pd.notna(base.loc[dates[15], "atr14_pct"])
    assert base.loc[dates[15], "atr14_pct"] == pytest.approx(pert.loc[dates[15], "atr14_pct"])
    assert base.loc[dates[16], "atr14_pct"] != pytest.approx(pert.loc[dates[16], "atr14_pct"])


# --- earnings_prev / eps_surprise_pct --------------------------------------------------------


def _earnings_table() -> pd.DataFrame:
    return pd.DataFrame(
        {"eps_estimate": [1.0, 0.5], "reported_eps": [1.2, 0.4], "surprise_pct": [20.0, -20.0]},
        index=pd.DatetimeIndex([DATES[EARNINGS_EQ_DAY], pd.Timestamp("2025-01-04")], name="date"),
    )


def test_earnings_prev_equal_to_prior_session():
    from research.strategy_hunt.context import symbol_daily_features

    daily = _rising_daily(daily_volume=1_000_000.0)
    bars5 = _bars5({})
    spy = _spy_daily()
    feat = symbol_daily_features("A", bars5, daily, spy, _earnings_table(), _empty_news())
    assert bool(feat.loc[DATES[EARNINGS_EQ_DAY], "earnings_today"]) is True
    assert bool(feat.loc[DATES[EARNINGS_EQ_DAY + 1], "earnings_prev"]) is True
    assert feat.loc[DATES[EARNINGS_EQ_DAY + 1], "eps_surprise_pct"] == pytest.approx(20.0)
    assert bool(feat.loc[DATES[EARNINGS_EQ_DAY + 1], "earnings_today"]) is False
    assert bool(feat.loc[DATES[EARNINGS_EQ_DAY], "earnings_prev"]) is False


def test_earnings_prev_strictly_between_sessions():
    from research.strategy_hunt.context import symbol_daily_features

    assert (DATES[2] - DATES[1]).days == 3  # confirms the Fri -> Mon weekend gap
    daily = _rising_daily(daily_volume=1_000_000.0)
    bars5 = _bars5({})
    spy = _spy_daily()
    feat = symbol_daily_features("A", bars5, daily, spy, _earnings_table(), _empty_news())
    assert bool(feat.loc[DATES[2], "earnings_prev"]) is True
    assert bool(feat.loc[DATES[2], "earnings_today"]) is False
    assert feat.loc[DATES[2], "eps_surprise_pct"] == pytest.approx(-20.0)


# --- news_count_24h / news_sent_24h -----------------------------------------------------------


def test_news_count_24h_counts_only_the_prior_24h():
    from research.strategy_hunt.context import symbol_daily_features

    open_ts = DATES[NEWS_DAY] + pd.Timedelta(hours=9, minutes=30)
    news = pd.DataFrame(
        {
            "created_at": [
                open_ts - pd.Timedelta(hours=1),
                open_ts - pd.Timedelta(hours=5),
                open_ts - pd.Timedelta(hours=23),
                open_ts - pd.Timedelta(hours=25),  # outside the 24h window
                open_ts + pd.Timedelta(hours=1),   # after the open
            ],
            "headline": ["a", "b", "c", "d", "e"],
            "source": ["x"] * 5,
            "sentiment": [0.5, -0.2, 0.3, 0.9, -0.9],
        }
    )
    daily = _rising_daily(daily_volume=1_000_000.0)
    bars5 = _bars5({})
    spy = _spy_daily()
    feat = symbol_daily_features("A", bars5, daily, spy, _empty_earnings(), news)
    assert feat.loc[DATES[NEWS_DAY], "news_count_24h"] == pytest.approx(3.0)
    assert feat.loc[DATES[NEWS_DAY], "news_sent_24h"] == pytest.approx((0.5 - 0.2 + 0.3) / 3)


# --- rvol_open15 ------------------------------------------------------------------------------


def _bars5_custom(dates: pd.DatetimeIndex, sessions: dict[int, list[float]],
                   default: tuple[float, ...] = (1000.0, 500.0, 500.0)) -> pd.DataFrame:
    """`sessions` maps session index -> explicit bar volumes (1-3 of them) for that session;
    sessions not listed get the 3-bar `default`."""
    idx, rows = [], []
    for i, d in enumerate(dates):
        for j, vol in enumerate(sessions.get(i, list(default))):
            idx.append(d + pd.Timedelta(minutes=5 * j))
            rows.append({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": vol})
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx, name="ts"))


def test_rvol_open15_hand_computed_and_nan_with_partial_session():
    from research.strategy_hunt.context import symbol_daily_features

    dates = pd.bdate_range("2025-04-01", periods=25, name="date")
    spike_day, partial_day = 20, 21
    # baseline first-three-bar volume is 1000+500+500=2000 every day; prior 14 sessions (6..19)
    # of the spike/partial day are all baseline -> mean 2000.
    sessions = {spike_day: [1000.0, 1000.0, 1000.0], partial_day: [1000.0, 500.0]}  # only 2 bars
    bars5 = _bars5_custom(dates, sessions)
    flat = np.full(25, 100.0)
    daily = pd.DataFrame({"open": flat, "high": flat + 1, "low": flat - 1, "close": flat,
                          "volume": np.full(25, 1_000_000.0)}, index=dates)
    spy_flat = np.full(25, 200.0)
    spy = pd.DataFrame({"open": spy_flat, "high": spy_flat + 1, "low": spy_flat - 1, "close": spy_flat,
                        "volume": np.full(25, 1_000_000.0)}, index=dates)

    feat = symbol_daily_features("A", bars5, daily, spy, _empty_earnings(), _empty_news())
    assert feat.loc[dates[spike_day], "rvol_open15"] == pytest.approx(1.5)  # 3000/2000
    assert pd.isna(feat.loc[dates[partial_day], "rvol_open15"])  # only 2 bars that session


# --- adv_bucket thresholds ---------------------------------------------------------------------


def test_adv_bucket_thresholds():
    from research.strategy_hunt.context import rank_cross_section, symbol_daily_features

    spy = _spy_daily()
    # close is ~100-139 (see _rising_daily); volumes chosen with enough margin that close*volume
    # stays clearly inside the intended bucket across the whole range.
    levels = {"MEGA": 20_000_000.0, "LARGE": 3_000_000.0, "MID": 300_000.0, "SMALL": 50_000.0}
    frames = {}
    for sym, vol in levels.items():
        bars5 = _bars5({})
        daily = _rising_daily(daily_volume=vol)
        frames[sym] = symbol_daily_features(sym, bars5, daily, spy, _empty_earnings(), _empty_news())

    combined = rank_cross_section(frames)
    on_day = combined[combined["date"] == DATES[GAP_DAY]].set_index("symbol")
    assert on_day.loc["MEGA", "adv_bucket"] == "mega"
    assert on_day.loc["LARGE", "adv_bucket"] == "large"
    assert on_day.loc["MID", "adv_bucket"] == "mid"
    assert on_day.loc["SMALL", "adv_bucket"] == "small"


def test_adv_bucket_nan_is_small():
    from research.strategy_hunt.context import _adv_bucket

    assert _adv_bucket(float("nan")) == "small"
    assert _adv_bucket(1e9) == "mega"
    assert _adv_bucket(2e8) == "large"
    assert _adv_bucket(2e7) == "mid"
    assert _adv_bucket(1.0) == "small"


# --- mom_12_1 / mom_rank_12_1 (Batch-2 Task C) -------------------------------------------------


def test_mom_12_1_hand_computed_with_flat_spy():
    """close[t] = 100+t; flat SPY -> the SPY leg is 0 for any window, so mom_12_1 is exactly the
    symbol's own (close.shift(21)/close.shift(251) - 1) * 100."""
    from research.strategy_hunt.context import symbol_daily_features

    dates, daily, spy, bars5 = _long_rising_frame(280)
    feat = symbol_daily_features("A", bars5, daily, spy, _empty_earnings(), _empty_news())
    close = daily["close"]
    i = 260
    expected = (close.iloc[i - 21] / close.iloc[i - 251] - 1) * 100
    assert feat.loc[dates[i], "mom_12_1"] == pytest.approx(expected)
    # fewer than 1+251 prior sessions -> NaN (the shift(251) leg is undefined)
    assert feat["mom_12_1"].iloc[:251].isna().all()
    assert pd.notna(feat["mom_12_1"].iloc[251])


def test_mom_12_1_hand_computed_with_non_flat_spy():
    """A flat SPY can't detect a bug that drops the SPY leg entirely (mirrors
    test_rs_20_hand_computed_with_non_flat_spy). Use a declining SPY so the two legs differ."""
    from research.strategy_hunt.context import symbol_daily_features

    dates, daily, spy, bars5 = _long_rising_frame(280)  # close[t] = 100+t
    spy_close = 300.0 - 0.5 * np.arange(280)
    spy = pd.DataFrame(
        {"open": spy_close, "high": spy_close + 1, "low": spy_close - 1, "close": spy_close,
         "volume": np.full(280, 1_000_000.0)}, index=dates,
    )
    feat = symbol_daily_features("A", bars5, daily, spy, _empty_earnings(), _empty_news())

    close = daily["close"]
    i = 260
    sym_leg = close.iloc[i - 21] / close.iloc[i - 251] - 1
    spy_leg = spy_close[i - 21] / spy_close[i - 251] - 1
    expected = (sym_leg - spy_leg) * 100
    assert feat.loc[dates[i], "mom_12_1"] == pytest.approx(expected)
    flat_expected = sym_leg * 100
    assert feat.loc[dates[i], "mom_12_1"] != pytest.approx(flat_expected)


def test_mom_12_1_lag_mirrors_relative_strengths_shift_pattern():
    """One-session lag, pinned by hand: perturbing a single day's close must not move that same
    day's own mom_12_1, but must show up exactly MOM_SKIP_SESSIONS and MOM_LOOKBACK_SESSIONS
    sessions later (the two legs the value is built from)."""
    from research.strategy_hunt.context import symbol_daily_features

    dates, daily, spy, bars5 = _long_rising_frame(560)
    perturbed = daily.copy()
    perturbed.loc[dates[300], "close"] *= 1.5  # huge one-day perturbation, day 300 only (>= 251
    # so that day's own mom_12_1 -- which needs close.shift(251) -- is already finite)

    base = symbol_daily_features("A", bars5, daily, spy, _empty_earnings(), _empty_news())
    pert = symbol_daily_features("A", bars5, perturbed, spy, _empty_earnings(), _empty_news())

    assert pd.notna(base.loc[dates[300], "mom_12_1"])
    assert base.loc[dates[300], "mom_12_1"] == pytest.approx(pert.loc[dates[300], "mom_12_1"])
    # 21 sessions later: shift(21) now reads the perturbed close at day 300
    assert base.loc[dates[321], "mom_12_1"] != pytest.approx(pert.loc[dates[321], "mom_12_1"])
    # 251 sessions later: shift(251) now reads the perturbed close at day 300
    assert base.loc[dates[551], "mom_12_1"] != pytest.approx(pert.loc[dates[551], "mom_12_1"])


def test_mom_rank_12_1_orders_symbols_by_momentum():
    from research.strategy_hunt.context import rank_cross_section, symbol_daily_features

    n = 280
    dates = pd.bdate_range("2020-01-02", periods=n, name="date")
    spy_close = np.full(n, 200.0)
    spy = pd.DataFrame({"open": spy_close, "high": spy_close + 1, "low": spy_close - 1,
                        "close": spy_close, "volume": np.full(n, 1_000_000.0)}, index=dates)
    empty_bars5 = pd.DataFrame(columns=["open", "high", "low", "close", "volume"],
                               index=pd.DatetimeIndex([], name="ts"))
    frames = {}
    for sym, slope in {"SLOW": 0.5, "MED": 1.0, "FAST": 2.0}.items():
        close = 100.0 + slope * np.arange(n)
        open_ = close.copy()
        open_[1:] = close[:-1]
        open_[0] = 100.0
        daily = pd.DataFrame({"open": open_, "high": close + 1, "low": close - 1, "close": close,
                             "volume": np.full(n, 1_000_000.0)}, index=dates)
        frames[sym] = symbol_daily_features(sym, empty_bars5, daily, spy, _empty_earnings(), _empty_news())

    combined = rank_cross_section(frames)
    on_day = combined[combined["date"] == dates[260]].set_index("symbol")
    assert on_day.loc["SLOW", "mom_rank_12_1"] < on_day.loc["MED", "mom_rank_12_1"] < on_day.loc["FAST", "mom_rank_12_1"]


# --- vol_rank_60 (Batch-2 Task C) --------------------------------------------------------------


def test_vol_60_uses_only_log_returns_through_the_prior_session():
    """One-session lag, pinned by hand: today's own return must not leak into today's own vol_60,
    but it must show up in tomorrow's (mirrors test_atr14_pct_uses_only_prior_sessions)."""
    from research.strategy_hunt.context import symbol_daily_features

    dates, daily, spy, bars5 = _long_rising_frame(90)
    perturbed = daily.copy()
    perturbed.loc[dates[70], "close"] *= 1.5  # one huge jump, day 70 only

    base = symbol_daily_features("A", bars5, daily, spy, _empty_earnings(), _empty_news())
    pert = symbol_daily_features("A", bars5, perturbed, spy, _empty_earnings(), _empty_news())

    assert pd.notna(base.loc[dates[70], "vol_60"])
    assert base.loc[dates[70], "vol_60"] == pytest.approx(pert.loc[dates[70], "vol_60"])
    # tomorrow's window includes today's now-perturbed log return -> must differ
    assert base.loc[dates[71], "vol_60"] != pytest.approx(pert.loc[dates[71], "vol_60"])


def test_vol_60_is_internal_only_not_a_ctx_column():
    from strategies.families.context import CTX_COLUMNS

    assert "vol_60" not in CTX_COLUMNS


def test_vol_rank_60_orders_symbols_by_realized_volatility():
    from research.strategy_hunt.context import rank_cross_section, symbol_daily_features

    n = 100
    dates = pd.bdate_range("2020-01-02", periods=n, name="date")
    spy_close = np.full(n, 200.0)
    spy = pd.DataFrame({"open": spy_close, "high": spy_close + 1, "low": spy_close - 1,
                        "close": spy_close, "volume": np.full(n, 1_000_000.0)}, index=dates)
    empty_bars5 = pd.DataFrame(columns=["open", "high", "low", "close", "volume"],
                               index=pd.DatetimeIndex([], name="ts"))
    sign = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    frames = {}
    for sym, amp in {"FLAT": 0.0, "MED": 0.01, "HIGH": 0.05}.items():
        close = 100.0 * (1 + amp * sign)
        daily = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close,
                             "volume": np.full(n, 1_000_000.0)}, index=dates)
        frames[sym] = symbol_daily_features(sym, empty_bars5, daily, spy, _empty_earnings(), _empty_news())

    combined = rank_cross_section(frames)
    on_day = combined[combined["date"] == dates[80]].set_index("symbol")
    assert on_day.loc["FLAT", "vol_rank_60"] < on_day.loc["MED", "vol_rank_60"] < on_day.loc["HIGH", "vol_rank_60"]

