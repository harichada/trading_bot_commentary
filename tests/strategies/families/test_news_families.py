import numpy as np
import pandas as pd
import pytest


def _session(day, n=78):
    idx = pd.date_range(f"{day} 09:30", periods=n, freq="5min", name="ts")
    return pd.DataFrame({"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1000}, index=idx)


def _ind_for(bars):
    ind = pd.DataFrame(index=bars.index)
    ind["bar_of_day"] = bars.groupby(bars.index.normalize()).cumcount()
    for col, val in {"rvol": 1.0, "vwap": 100.0, "atr": 1.0, "d_atr14": 2.0}.items():
        ind[col] = val
    return ind


def test_news_catalyst_long_after_confirmation_inside_window():
    from strategies.families.news import news_catalyst_entries
    bars = _session("2025-01-02")
    ind = _ind_for(bars)
    news = pd.DataFrame({"created_at": [pd.Timestamp("2025-01-02 10:02")], "headline": ["beat"], "source": ["b"], "sentiment": [0.6]})
    i0 = 7                                                      # 10:05 bar is first >= 10:02
    bars.loc[bars.index[i0 + 2], ["open", "close"]] = [100.0, 100.4]   # green, above vwap
    ind.loc[ind.index[i0 + 2], "rvol"] = 2.5
    p = {"window_minutes": 30, "min_abs_sentiment": 0.5, "target_r": 2.0}
    es = news_catalyst_entries(bars, ind, pd.DataFrame(), news, p)
    assert len(es) == 1 and es[0].side == "long" and es[0].entry_ts == bars.index[i0 + 3]
    assert es[0].stop_px == pytest.approx(100.0 - 1.5) and es[0].target_px == pytest.approx(100.0 + 3.0)
    assert news_catalyst_entries(bars, ind, pd.DataFrame(), news, {**p, "window_minutes": 15})[0].side == "long"
    ind.loc[ind.index[i0 + 2], "rvol"] = 1.0
    assert news_catalyst_entries(bars, ind, pd.DataFrame(), news, p) == []


def test_news_catalyst_short_and_sentiment_floor_and_window_end():
    from strategies.families.news import news_catalyst_entries
    bars = _session("2025-01-02")
    ind = _ind_for(bars)
    news = pd.DataFrame({"created_at": [pd.Timestamp("2025-01-02 11:00")], "headline": ["miss"], "source": ["b"], "sentiment": [-0.4]})
    i0 = 18
    bars.loc[bars.index[i0 + 5], ["open", "close"]] = [100.0, 99.6]
    ind.loc[ind.index[i0 + 5], "rvol"] = 3.0
    p = {"window_minutes": 60, "min_abs_sentiment": 0.3, "target_r": 1.5}
    es = news_catalyst_entries(bars, ind, pd.DataFrame(), news, p)
    assert len(es) == 1 and es[0].side == "short" and es[0].tag == "news_short"
    assert news_catalyst_entries(bars, ind, pd.DataFrame(), news, {**p, "min_abs_sentiment": 0.5}) == []
    assert news_catalyst_entries(bars, ind, pd.DataFrame(), news, {**p, "window_minutes": 15}) == []


def test_post_news_drift_enters_next_open_in_move_direction():
    from strategies.families.news import post_news_drift_entries
    days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2025-01-02", periods=25)]
    bars = pd.concat([_session(d) for d in days])
    ind = _ind_for(bars)
    daily = pd.DataFrame({"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1000},
                         index=pd.DatetimeIndex(pd.to_datetime(days), name="date"))
    d_event = daily.index[22]
    daily.loc[d_event, ["close", "volume"]] = [105.0, 4000]        # +5%, 4x volume
    daily.loc[daily.index[23:], "close"] = 105.0
    news = pd.DataFrame({"created_at": [d_event + pd.Timedelta(hours=8)], "headline": ["x"], "source": ["b"], "sentiment": [0.1]})
    p = {"vol_mult": 3.0, "min_ret_pct": 4.0, "hold_days": 5}
    es = post_news_drift_entries(bars, ind, daily, news, p)
    assert len(es) == 1 and es[0].side == "long" and es[0].tag == "drift_long"
    assert es[0].entry_ts == pd.Timestamp(days[23] + " 09:30") and es[0].time_stop == 5 and es[0].hold_kind == "multiday"
    assert es[0].stop_px == pytest.approx(100.0 - 3.0) and es[0].target_px == pytest.approx(100.0 + 9.0)
    assert post_news_drift_entries(bars, ind, daily, news, {**p, "min_ret_pct": 6.0}) == []
    assert post_news_drift_entries(bars, ind, daily, news.iloc[0:0], p) == []      # no news that day


def test_family_descriptors():
    from strategies.families.news import FAMILIES_NEWS
    from strategies.families.common import grid_combos
    assert {f.name for f in FAMILIES_NEWS} == {"news_catalyst", "post_news_drift"}
    assert all(len(grid_combos(f.grid)) <= 27 for f in FAMILIES_NEWS)
