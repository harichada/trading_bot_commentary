"""v-playbook-desk-b2-2026-09-25: opening features for the 09:35 in-play scan.

Session-bar conventions match `strategies/families/indicators.py`: 5-minute
bars, sessions grouped by `index.normalize()`, bar_of_day 0 at 09:30, so the
"opening 15 minutes" is the session's first three bars.

Nothing here touches the network or the real research tree.
"""
from __future__ import annotations

import math
from datetime import datetime

import pandas as pd
import pytest

from core.desk.inplay_features import load_prior_context, opening_features
from core.desk.models import InPlayRow

TODAY = "2026-09-25"                       # a Friday
NOW = datetime(2026, 9, 25, 9, 45)
PRIOR_SESSIONS = [d.strftime("%Y-%m-%d")
                  for d in pd.bdate_range(end="2026-09-24", periods=14)]


def _session(day: str, volumes, opens=None, closes=None) -> pd.DataFrame:
    """One session of 5-minute bars starting at 09:30."""
    n = len(volumes)
    opens = list(opens) if opens is not None else [100.0] * n
    closes = list(closes) if closes is not None else [100.0] * n
    idx = pd.DatetimeIndex(
        [pd.Timestamp(f"{day} 09:30") + pd.Timedelta(minutes=5 * i) for i in range(n)],
        name="ts",
    )
    return pd.DataFrame(
        {"open": opens, "high": [max(o, c) for o, c in zip(opens, closes)],
         "low": [min(o, c) for o, c in zip(opens, closes)],
         "close": closes, "volume": [float(v) for v in volumes]},
        index=idx,
    )


def _history(days=PRIOR_SESSIONS, first3=1000.0, close=100.0) -> pd.DataFrame:
    """Prior sessions whose first three bars total `first3` and which close at `close`."""
    frames = [_session(d, [first3 * 0.4, first3 * 0.3, first3 * 0.3, 9999.0],
                       closes=[close] * 4) for d in days]
    return pd.concat(frames)


def _today(volumes=(1000.0, 500.0, 500.0), open_px=103.0) -> pd.DataFrame:
    opens = [open_px] + [101.0] * (len(volumes) - 1)
    return _session(TODAY, list(volumes), opens=opens)


def _ctx(**kw) -> dict:
    base = {"earnings_today": False, "earnings_prev": False,
            "news_count_24h": 0, "rs_20": 4.0}
    base.update(kw)
    return base


# ────────────────────────────────────────────────────────────────────
# the happy path: rvol exactly 2.0, gap exactly +3%
# ────────────────────────────────────────────────────────────────────

def test_opening_features_on_a_two_week_frame():
    row = opening_features("aapl", _today(), _history(), _ctx(earnings_today=True), NOW)
    assert isinstance(row, InPlayRow)
    assert row.symbol == "AAPL"
    assert row.rvol_open15 == pytest.approx(2.0)
    assert row.gap_pct == pytest.approx(3.0)
    assert row.catalyst == 1.0
    assert row.rs_20 == pytest.approx(4.0)
    assert row.reasons == ()
    assert math.isnan(row.score)          # the ranker fills score in, not the features


def test_gap_is_negative_when_the_open_is_below_the_prior_close():
    row = opening_features("AAPL", _today(open_px=97.0), _history(), _ctx(), NOW)
    assert row.gap_pct == pytest.approx(-3.0)


def test_median_uses_only_the_last_fourteen_prior_sessions():
    old = [d.strftime("%Y-%m-%d") for d in pd.bdate_range(end="2026-09-04", periods=6)]
    frame = pd.concat([_history(days=old, first3=100000.0), _history()])
    row = opening_features("AAPL", _today(), frame, _ctx(), NOW)
    assert row.rvol_open15 == pytest.approx(2.0)


def test_prior_sessions_with_fewer_than_three_bars_are_skipped():
    short = _session("2026-09-24", [10.0, 10.0])
    frame = pd.concat([_history(days=PRIOR_SESSIONS[:13]), short])
    row = opening_features("AAPL", _today(), frame, _ctx(), NOW)
    assert row.rvol_open15 == pytest.approx(2.0)


def test_history_rows_dated_today_are_ignored():
    frame = pd.concat([_history(), _session(TODAY, [1.0, 1.0, 1.0], closes=[50.0] * 3)])
    row = opening_features("AAPL", _today(), frame, _ctx(), NOW)
    assert row.rvol_open15 == pytest.approx(2.0)
    assert row.gap_pct == pytest.approx(3.0)


def test_timezone_aware_frames_are_converted_to_naive_et():
    today = _today().tz_localize("America/New_York")
    hist = _history().tz_localize("America/New_York")
    row = opening_features("AAPL", today, hist, _ctx(), NOW)
    assert row.rvol_open15 == pytest.approx(2.0)
    assert row.gap_pct == pytest.approx(3.0)


# ────────────────────────────────────────────────────────────────────
# insufficient data -> NaN fields with a reason, never an exception
# ────────────────────────────────────────────────────────────────────

def test_fewer_than_five_prior_sessions_leaves_rvol_nan():
    row = opening_features("AAPL", _today(), _history(days=PRIOR_SESSIONS[-4:]), _ctx(), NOW)
    assert math.isnan(row.rvol_open15)
    assert "missing:rvol_open15" in row.reasons
    assert row.gap_pct == pytest.approx(3.0)      # the gap only needs the prior close


def test_exactly_five_prior_sessions_is_enough():
    row = opening_features("AAPL", _today(), _history(days=PRIOR_SESSIONS[-5:]), _ctx(), NOW)
    assert row.rvol_open15 == pytest.approx(2.0)


def test_fewer_than_three_bars_today_leaves_rvol_nan_but_keeps_the_gap():
    row = opening_features("AAPL", _today(volumes=(1000.0, 500.0)), _history(), _ctx(), NOW)
    assert math.isnan(row.rvol_open15)
    assert "missing:rvol_open15" in row.reasons
    assert row.gap_pct == pytest.approx(3.0)


def test_no_bars_for_today_leaves_both_intraday_features_nan():
    row = opening_features("AAPL", _session("2026-09-24", [1.0, 2.0, 3.0]), _history(),
                           _ctx(earnings_today=True), NOW)
    assert math.isnan(row.rvol_open15) and math.isnan(row.gap_pct)
    assert "missing:today_bars" in row.reasons
    assert row.catalyst == 1.0 and row.rs_20 == pytest.approx(4.0)


def test_empty_frames_produce_an_all_nan_row_with_reasons():
    row = opening_features("AAPL", pd.DataFrame(), pd.DataFrame(), None, NOW)
    assert math.isnan(row.rvol_open15) and math.isnan(row.gap_pct) and math.isnan(row.rs_20)
    assert row.catalyst == 0.0
    assert "missing:today_bars" in row.reasons
    assert "missing:prev_close" in row.reasons
    assert "missing:prior_ctx" in row.reasons


def test_no_prior_session_leaves_the_gap_nan():
    row = opening_features("AAPL", _today(), None, _ctx(), NOW)
    assert math.isnan(row.gap_pct) and math.isnan(row.rvol_open15)
    assert "missing:prev_close" in row.reasons


def test_zero_median_open_volume_leaves_rvol_nan():
    row = opening_features("AAPL", _today(), _history(first3=0.0), _ctx(), NOW)
    assert math.isnan(row.rvol_open15)
    assert "missing:rvol_open15" in row.reasons


def test_garbage_frames_never_raise():
    row = opening_features("AAPL", "not a frame", 17, _ctx(), NOW)   # type: ignore[arg-type]
    assert row.symbol == "AAPL"
    assert math.isnan(row.rvol_open15) and math.isnan(row.gap_pct)
    assert row.reasons                     # some reason is always given


def test_missing_volume_column_never_raises():
    today = _today().drop(columns=["volume"])
    row = opening_features("AAPL", today, _history(), _ctx(), NOW)
    assert math.isnan(row.rvol_open15)
    assert row.gap_pct == pytest.approx(3.0)


# ────────────────────────────────────────────────────────────────────
# catalyst and rs_20, from the nightly context row
# ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("ctx_kw,expected", [
    ({"earnings_today": True}, 1.0),
    ({"earnings_prev": True}, 1.0),
    ({"news_count_24h": 3}, 1.0),
    ({"news_count_24h": 9}, 1.0),
    ({"news_count_24h": 2}, 0.0),
    ({}, 0.0),
])
def test_catalyst_rules(ctx_kw, expected):
    row = opening_features("AAPL", _today(), _history(), _ctx(**ctx_kw), NOW)
    assert row.catalyst == expected


def test_catalyst_tolerates_numpy_style_and_string_values():
    row = opening_features("AAPL", _today(), _history(),
                           {"earnings_today": 1, "news_count_24h": "x"}, NOW)
    assert row.catalyst == 1.0


def test_prior_ctx_none_means_no_catalyst_and_no_rs():
    row = opening_features("AAPL", _today(), _history(), None, NOW)
    assert row.catalyst == 0.0 and math.isnan(row.rs_20)
    assert "missing:prior_ctx" in row.reasons


def test_unusable_rs_20_is_reported_missing():
    row = opening_features("AAPL", _today(), _history(), _ctx(rs_20="n/a"), NOW)
    assert math.isnan(row.rs_20)
    assert "missing:rs_20" in row.reasons


# ────────────────────────────────────────────────────────────────────
# load_prior_context — last row of the nightly universe context parquet
# ────────────────────────────────────────────────────────────────────

def _write_context(root, rows) -> None:
    ctx_dir = root / "context"
    ctx_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(ctx_dir / "universe_daily.parquet", index=False)


def test_load_prior_context_returns_the_last_row_for_the_symbol(tmp_path):
    _write_context(tmp_path, [
        {"date": pd.Timestamp("2026-09-22"), "symbol": "AAPL", "rs_20": 1.0,
         "news_count_24h": 0, "earnings_today": False},
        {"date": pd.Timestamp("2026-09-24"), "symbol": "AAPL", "rs_20": 4.5,
         "news_count_24h": 7, "earnings_today": True},
        {"date": pd.Timestamp("2026-09-23"), "symbol": "MSFT", "rs_20": -2.0,
         "news_count_24h": 1, "earnings_today": False},
    ])
    out = load_prior_context(tmp_path, "aapl")
    assert out is not None
    assert out["rs_20"] == pytest.approx(4.5)
    assert int(out["news_count_24h"]) == 7
    assert bool(out["earnings_today"]) is True
    assert out["date"] == "2026-09-24"
    assert out["symbol"] == "AAPL"


def test_load_prior_context_feeds_opening_features(tmp_path):
    _write_context(tmp_path, [{"date": pd.Timestamp("2026-09-24"), "symbol": "AAPL",
                               "rs_20": -12.0, "news_count_24h": 4,
                               "earnings_today": False, "earnings_prev": False}])
    row = opening_features("AAPL", _today(), _history(),
                           load_prior_context(tmp_path, "AAPL"), NOW)
    assert row.rs_20 == pytest.approx(-12.0)
    assert row.catalyst == 1.0


def test_load_prior_context_unknown_symbol_is_none(tmp_path):
    _write_context(tmp_path, [{"date": pd.Timestamp("2026-09-24"), "symbol": "AAPL",
                               "rs_20": 1.0}])
    assert load_prior_context(tmp_path, "NVDA") is None


def test_load_prior_context_missing_file_is_none(tmp_path):
    assert load_prior_context(tmp_path, "AAPL") is None
    assert load_prior_context(tmp_path / "nope", "AAPL") is None


def test_load_prior_context_corrupt_file_is_none(tmp_path):
    ctx_dir = tmp_path / "context"
    ctx_dir.mkdir(parents=True)
    (ctx_dir / "universe_daily.parquet").write_bytes(b"not parquet at all")
    assert load_prior_context(tmp_path, "AAPL") is None


def test_load_prior_context_bad_root_or_symbol_is_none(tmp_path):
    assert load_prior_context(None, "AAPL") is None            # type: ignore[arg-type]
    assert load_prior_context(tmp_path, "") is None


# ────────────────────────────────────────────────────────────────────
# fix round 1: a broken clock must not rank a stale session, and the
# computation must never raise whatever the frame looks like
# ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("clock", [None, pd.NaT, "garbage", object()])
def test_unusable_clock_never_ranks_a_stale_session(clock):
    """Without a usable clock there is no "today", so no intraday feature is
    reported — ranking yesterday's session with empty reasons would be worse
    than ranking nothing."""
    row = opening_features("AAPL", _today(), _history(), _ctx(earnings_today=True), clock)
    assert math.isnan(row.rvol_open15) and math.isnan(row.gap_pct)
    assert "missing:today_bars" in row.reasons
    assert row.catalyst == 1.0                    # context features still stand
    assert row.rs_20 == pytest.approx(4.0)


def test_duplicated_volume_column_never_raises():
    """A frame whose `volume` label appears twice makes column access return a
    DataFrame; the row degrades instead of the 09:35 scan dying."""
    today = _today()
    broken = pd.concat([today, today[["volume"]]], axis=1)
    assert list(broken.columns).count("volume") == 2
    row = opening_features("AAPL", broken, _history(), _ctx(), NOW)
    assert row.symbol == "AAPL"
    assert math.isnan(row.rvol_open15) and math.isnan(row.gap_pct)
    assert "missing:today_bars" in row.reasons


def test_load_prior_context_ties_on_date_keep_the_last_row(tmp_path):
    """Rows repeating the newest date (a rebuild appending) resolve to the last
    one written, not to whichever the sort happened to land on: the date sort
    has to be stable, which an unstable one is not once the tie group is
    scattered through the file."""
    rows = [{"date": pd.Timestamp("2026-09-24" if i % 3 == 0
                                  else f"2026-09-{1 + (i % 20):02d}"),
             "symbol": "AAPL", "rs_20": float(i)} for i in range(20)]
    _write_context(tmp_path, rows)
    out = load_prior_context(tmp_path, "AAPL")
    assert out is not None
    assert out["date"] == "2026-09-24"
    assert out["rs_20"] == pytest.approx(18.0)      # the last 2026-09-24 row


def test_module_docstring_warns_about_the_median_baseline_and_rth_bars():
    import core.desk.inplay_features as mod
    doc = (mod.__doc__ or "").lower()
    assert "median" in doc and "mean" in doc     # live baseline vs the context column
    assert "regular" in doc                      # regular-hours bars only
