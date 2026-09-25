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


def test_pct_from_52w_high_never_positive_for_monotonic_rise():
    from research.strategy_hunt.context import symbol_daily_features

    daily = _rising_daily(daily_volume=1_000_000.0)  # strictly increasing close
    bars5 = _bars5({})
    spy = _spy_daily()
    feat = symbol_daily_features("A", bars5, daily, spy, _empty_earnings(), _empty_news())
    vals = feat["pct_from_52w_high"].dropna()
    assert (vals <= 0).all()
    # prior close IS the rolling high when the series is monotonic increasing -> exactly 0
    assert (vals.abs() < 1e-9).all()


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


# --- build_universe_context / load_context_slice ------------------------------------------------


def _write_symbol(tmp_path, symbol: str, n: int = 15, daily_volume: float = 1_000_000.0) -> None:
    from research.strategy_hunt.data.bars import _write, bars5_path, daily_path

    dates = pd.bdate_range("2025-02-03", periods=n, name="date")
    close = np.full(n, 50.0)
    daily = pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": np.full(n, daily_volume)},
        index=dates,
    )
    idx, rows = [], []
    for d in dates:
        for j in range(3):
            idx.append(d + pd.Timedelta(minutes=5 * j))
            rows.append({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1000.0})
    bars5 = pd.DataFrame(rows, index=pd.DatetimeIndex(idx, name="ts"))
    _write(bars5_path(tmp_path, symbol), bars5, "ts")
    _write(daily_path(tmp_path, symbol), daily, "date")


def test_build_universe_context_writes_file_and_load_context_slice_reads_it(tmp_path):
    from research.strategy_hunt.context import build_universe_context, load_context_slice
    from strategies.families.context import CTX_COLUMNS

    _write_symbol(tmp_path, "SPY", daily_volume=1_000_000.0)
    _write_symbol(tmp_path, "VIX", daily_volume=1_000_000.0)
    _write_symbol(tmp_path, "TEST1", daily_volume=5_000_000.0)
    _write_symbol(tmp_path, "TEST2", daily_volume=1_000_000.0)

    out_path = build_universe_context(tmp_path)
    assert out_path == tmp_path / "context" / "universe_daily.parquet"
    assert out_path.exists()

    written = pd.read_parquet(out_path)
    assert set(written["symbol"].unique()) == {"TEST1", "TEST2"}  # SPY/VIX excluded
    assert list(written.columns) == ["date", "symbol", *CTX_COLUMNS]

    slice1 = load_context_slice(tmp_path, "TEST1")
    assert not slice1.empty
    assert slice1.index.name == "date"
    assert list(slice1.columns) == list(CTX_COLUMNS)

    empty = load_context_slice(tmp_path, "NOSUCHSYMBOL")
    assert empty.empty and list(empty.columns) == list(CTX_COLUMNS)


def test_load_context_slice_missing_root_is_empty(tmp_path):
    from research.strategy_hunt.context import load_context_slice
    from strategies.families.context import CTX_COLUMNS

    out = load_context_slice(tmp_path, "AAPL")
    assert out.empty
    assert list(out.columns) == list(CTX_COLUMNS)
    assert isinstance(out.index, pd.DatetimeIndex) and out.index.name == "date"


def test_build_universe_context_skips_symbol_on_failure(tmp_path, monkeypatch):
    from research.strategy_hunt import context as C

    _write_symbol(tmp_path, "SPY")
    _write_symbol(tmp_path, "TEST1")

    def boom(symbol, bars5, daily, spy_daily, earnings_table, news):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(C, "symbol_daily_features", boom)
    out_path = C.build_universe_context(tmp_path)
    written = pd.read_parquet(out_path)
    assert written.empty  # the only non-SPY symbol failed and was skipped, not raised
