"""v-playbook-desk-b2-2026-09-25 (Task 5): the desk loop's pure steps.

Everything the loop does that is *not* I/O lives in `core/loops/desk_steps.py`
as a total function, so the loop body stays a thin schedule and these rules can
be tested without an engine: the ET clock crossings, the breadth reading, the
losing-day streak, the candidate list, and the journal's trailing summary.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from core.desk.models import DailyPlan
from core.loops import desk_steps as steps


def _et(y, m, d, hh, mm) -> datetime:
    return datetime(y, m, d, hh, mm, tzinfo=steps.ET_TZ)


# ── clock ───────────────────────────────────────────────────────────

def test_session_date_str_is_the_et_calendar_date():
    assert steps.session_date_str(_et(2026, 9, 25, 9, 35)) == "2026-09-25"


def test_weekends_are_not_session_days():
    assert steps.is_session_day(_et(2026, 9, 25, 9, 0)) is True      # Friday
    assert steps.is_session_day(_et(2026, 9, 26, 9, 0)) is False     # Saturday
    assert steps.is_session_day(_et(2026, 9, 27, 9, 0)) is False     # Sunday
    assert steps.is_session_day(_et(2026, 9, 28, 9, 0)) is True      # Monday


def test_parse_et_time_falls_back_on_garbage():
    assert steps.parse_et_time("09:35", "09:00") == (9, 35)
    assert steps.parse_et_time("", "09:00") == (9, 0)
    assert steps.parse_et_time("nonsense", "09:00") == (9, 0)
    assert steps.parse_et_time("25:99", "16:05") == (16, 5)


def test_crossed_is_true_from_the_target_onward():
    target = (9, 35)
    assert steps.crossed(_et(2026, 9, 25, 9, 34), target) is False
    assert steps.crossed(_et(2026, 9, 25, 9, 35), target) is True
    assert steps.crossed(_et(2026, 9, 25, 15, 0), target) is True


# ── breadth ─────────────────────────────────────────────────────────

def test_breadth_score_prefers_advancing_percent():
    assert steps.breadth_score({"advancing_percent": 70.0, "advances": 1, "declines": 9}) == 0.7


def test_breadth_score_from_advance_decline_counts():
    assert steps.breadth_score({"advances": 6, "declines": 4}) == 0.6


def test_breadth_score_from_the_ratio():
    assert steps.breadth_score({"advance_decline_ratio": 3.0}) == 0.75


def test_breadth_score_from_new_highs_and_lows():
    assert steps.breadth_score({"new_highs": 30, "new_lows": 10}) == 0.75


def test_breadth_score_from_the_signed_advance_decline():
    assert steps.breadth_score({"advance_decline": 0.0}) == 0.5
    assert steps.breadth_score({"advance_decline": 20}) == 1.0
    assert steps.breadth_score({"advance_decline": -20}) == 0.0


def test_breadth_score_is_none_when_there_is_no_reading():
    assert steps.breadth_score(None) is None
    assert steps.breadth_score({}) is None
    assert steps.breadth_score({"vix": 18.0}) is None
    assert steps.breadth_score("nonsense") is None
    assert steps.breadth_score({"advances": 0, "declines": 0}) is None


# ── daily loss budget ───────────────────────────────────────────────

def test_daily_loss_pct_converts_a_fraction_to_percent():
    assert steps.daily_loss_pct(0.05) == 5.0
    assert steps.daily_loss_pct(1.0) == 100.0


def test_daily_loss_pct_passes_a_percent_through():
    assert steps.daily_loss_pct(5.0) == 5.0


def test_daily_loss_pct_is_zero_for_garbage():
    assert steps.daily_loss_pct(None) == 0.0
    assert steps.daily_loss_pct("x") == 0.0
    assert steps.daily_loss_pct(float("nan")) == 0.0
    assert steps.daily_loss_pct(-3) == 0.0


# ── losing-day streak and prior-day R ───────────────────────────────

def _trade(date: str, r: float) -> dict:
    return {"exit_time": f"{date} 15:45:00", "pnl_r": r, "strategy": "mean_reversion"}


def test_daily_r_stats_reads_the_prior_session():
    trades = [_trade("2026-09-24", 1.5), _trade("2026-09-24", -0.5),
              _trade("2026-09-23", -2.0)]
    streak, prev_r = steps.daily_r_stats(trades, "2026-09-25")
    assert prev_r == 1.0
    assert streak == 0


def test_daily_r_stats_counts_consecutive_losing_days():
    trades = [_trade("2026-09-24", -1.0), _trade("2026-09-23", -0.5),
              _trade("2026-09-22", 2.0)]
    streak, prev_r = steps.daily_r_stats(trades, "2026-09-25")
    assert (streak, prev_r) == (2, -1.0)


def test_daily_r_stats_ignores_today_and_the_future():
    trades = [_trade("2026-09-25", -5.0), _trade("2026-09-26", -5.0),
              _trade("2026-09-24", 0.5)]
    streak, prev_r = steps.daily_r_stats(trades, "2026-09-25")
    assert (streak, prev_r) == (0, 0.5)


def test_daily_r_stats_is_flat_with_no_usable_rows():
    assert steps.daily_r_stats([], "2026-09-25") == (0, 0.0)
    assert steps.daily_r_stats([{"pnl_r": None}], "2026-09-25") == (0, 0.0)
    assert steps.daily_r_stats("nonsense", "2026-09-25") == (0, 0.0)


# ── candidate list ──────────────────────────────────────────────────

def test_scan_symbols_is_pinned_then_watchlist_then_universe():
    out = steps.scan_symbols(["spy", "QQQ"], ["QQQ", "NVDA"], ["TSLA", "AMD"], 10)
    assert out == ["SPY", "QQQ", "NVDA", "TSLA", "AMD"]


def test_scan_symbols_caps_only_the_universe_slice():
    out = steps.scan_symbols(["SPY"], ["NVDA"], ["A", "B", "C"], 2)
    assert out == ["SPY", "NVDA", "A", "B"]


def test_scan_symbols_survives_garbage():
    assert steps.scan_symbols(None, None, None, 5) == []
    assert steps.scan_symbols(["", None, 7], ["nvda"], [], 5) == ["NVDA"]


def test_load_scan_universe_ranks_by_rvol_then_adv(tmp_path):
    frame = pd.DataFrame([
        {"date": "2026-09-24", "symbol": "AAA", "rvol_rank5": 3.0, "adv_dollars_20": 1e9},
        {"date": "2026-09-24", "symbol": "BBB", "rvol_rank5": 1.0, "adv_dollars_20": 1e8},
        {"date": "2026-09-24", "symbol": "CCC", "rvol_rank5": 2.0, "adv_dollars_20": 5e8},
        {"date": "2026-09-23", "symbol": "OLD", "rvol_rank5": 1.0, "adv_dollars_20": 9e9},
    ])
    ctx = tmp_path / "context"
    ctx.mkdir()
    frame.to_parquet(ctx / "universe_daily.parquet", index=False)
    assert steps.load_scan_universe(str(tmp_path), 10) == ["BBB", "CCC", "AAA"]
    assert steps.load_scan_universe(str(tmp_path), 2) == ["BBB", "CCC"]


def test_load_scan_universe_never_raises(tmp_path):
    assert steps.load_scan_universe(str(tmp_path / "nope"), 5) == []
    assert steps.load_scan_universe(None, 5) == []


# ── provider bars ───────────────────────────────────────────────────

class _Provider:
    def __init__(self, frame=None, boom=False):
        self.frame = frame
        self.boom = boom
        self.calls = []

    def get_market_data(self, symbol, **kw):
        self.calls.append((symbol, kw))
        if self.boom:
            raise RuntimeError("provider down")
        return self.frame


def _utc_session(day: str, rows: int = 4) -> pd.DataFrame:
    # 13:30 UTC == 09:30 ET during daylight saving.
    idx = pd.date_range(f"{day} 13:30", periods=rows, freq="5min", tz="UTC")
    return pd.DataFrame({"Open": 10.0, "High": 10.5, "Low": 9.5, "Close": 10.2,
                         "Volume": 1000.0}, index=idx)


def test_fetch_today_bars5_normalizes_to_naive_et_regular_hours():
    frame = pd.concat([_utc_session("2026-09-24"), _utc_session("2026-09-25")])
    out = steps.fetch_today_bars5(_Provider(frame), "NVDA")
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert out.index.tz is None
    assert str(out.index[0]) == "2026-09-24 09:30:00"


def test_fetch_today_bars5_drops_extended_hours():
    """A naive index is UTC (what `data_providers/schwab.py` builds from epoch ms):
    12:00 UTC is 08:00 ET (pre-market), 20:30 UTC is 16:30 ET (post-market)."""
    idx = pd.DatetimeIndex(["2026-09-25 12:00", "2026-09-25 13:30", "2026-09-25 20:30"])
    frame = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
                          "volume": 1.0}, index=idx)
    out = steps.fetch_today_bars5(_Provider(frame), "NVDA")
    assert len(out) == 1 and str(out.index[0]) == "2026-09-25 09:30:00"


def test_fetch_today_bars5_reads_a_naive_index_as_utc_like_fit_bars_source():
    """v-playbook-desk-b2-final-2026-09-25 (I2): the Schwab provider hands back a
    *naive UTC* index, and `engine._fit_bars_source` localizes UTC before converting
    to ET. Reading those stamps as ET instead kept 09:30-15:55 **UTC** — 05:30-11:55
    ET — so the in-play features saw pre-market bars and missed the afternoon.

    A full 09:30-15:55 ET session is 13:30-19:55 UTC: 78 five-minute bars."""
    idx = pd.date_range("2026-09-25 13:30", periods=96, freq="5min")   # naive UTC
    frame = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
                          "volume": 1.0}, index=idx)
    out = steps.fetch_today_bars5(_Provider(frame), "NVDA")
    assert len(out) == 78
    assert str(out.index[0]) == "2026-09-25 09:30:00"
    assert str(out.index[-1]) == "2026-09-25 15:55:00"


def test_fetch_today_bars5_matches_fit_bars_source_on_the_same_frame():
    """Both readers must normalize identically — the desk's in-play features and the
    fit lane's 5-minute features are computed from the same provider frame."""
    idx = pd.date_range("2026-09-25 13:30", periods=96, freq="5min")   # naive UTC
    frame = pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0,
                          "Volume": 1.0}, index=idx)
    out = steps.fetch_today_bars5(_Provider(frame), "NVDA")

    mirror = frame.rename(columns={c: c.lower() for c in frame.columns})
    mirror.index = (mirror.index.tz_localize("UTC")
                    .tz_convert("America/New_York").tz_localize(None))
    mirror = mirror.between_time("09:30", "15:55")
    assert list(out.index) == list(mirror.index)


def test_fetch_today_bars5_returns_none_on_failure():
    assert steps.fetch_today_bars5(_Provider(boom=True), "NVDA") is None
    assert steps.fetch_today_bars5(_Provider(frame=pd.DataFrame()), "NVDA") is None
    assert steps.fetch_today_bars5(None, "NVDA") is None


# ── journal trailing summary ────────────────────────────────────────

def test_trailing_summary_groups_expectancy_by_family_and_regime():
    trades = [
        {"exit_time": "2026-09-24 10:00", "pnl_r": 1.0, "family": "mean_rev_intraday",
         "regime": "risk_on"},
        {"exit_time": "2026-09-24 11:00", "pnl_r": -0.5, "family": "mean_rev_intraday",
         "regime": "risk_on"},
        {"exit_time": "2026-09-23 11:00", "pnl_r": -1.0, "family": "orb"},
    ]
    out = steps.trailing_summary(trades, {})
    rows = {(r["family"], r["regime"]): r for r in out["expectancy"]}
    mr = rows[("mean_rev_intraday", "risk_on")]
    assert mr["n"] == 2 and round(mr["exp_r"], 4) == 0.25 and round(mr["pf"], 2) == 2.0
    orb = rows[("orb", "unknown")]
    assert orb["n"] == 1 and orb["exp_r"] == -1.0 and orb["pf"] == 0.0


def test_trailing_summary_flags_sit_out_days_and_max_trades_days():
    plans = {
        "2026-09-23": DailyPlan(date="2026-09-23", sit_out=True, sit_out_reason="vix_spike",
                                max_trades=2),
        "2026-09-24": DailyPlan(date="2026-09-24", max_trades=2),
    }
    trades = [
        {"exit_time": "2026-09-23 10:00", "pnl_r": 0.8, "family": "orb"},
        {"exit_time": "2026-09-24 10:00", "pnl_r": 0.1, "family": "orb"},
        {"exit_time": "2026-09-24 11:00", "pnl_r": 0.2, "family": "orb"},
    ]
    out = steps.trailing_summary(trades, plans)
    assert out["sit_out_days"] == [{"date": "2026-09-23", "reason": "vix_spike",
                                   "would_have_r": 0.8}]
    assert out["max_trades_days"] == [{"date": "2026-09-24", "max_trades": 2,
                                       "trades_taken": 2}]


def test_trailing_summary_survives_garbage():
    out = steps.trailing_summary(None, None)
    assert out == {"expectancy": [], "sit_out_days": [], "max_trades_days": []}


# ── playbook changes ────────────────────────────────────────────────

def test_playbook_changes_today_lists_entries_stamped_today():
    from core.desk.playbook import Playbook, PlaybookEntry
    today = PlaybookEntry(family="orb", side="long", regime="mixed", status="suspended",
                          size_mult=0.0, exp_r_recent=-0.3, t_stat_recent=-2.0, n_recent=20,
                          updated_at="2026-09-25T18:00:00", reason="bad")
    old = PlaybookEntry(family="orb", side="short", regime="mixed", status="enabled",
                        size_mult=1.0, exp_r_recent=0.2, t_stat_recent=1.0, n_recent=50,
                        updated_at="2026-09-01T18:00:00", reason="fine")
    book = Playbook(entries={today.key: today, old.key: old})
    changes = steps.playbook_changes_today(book, "2026-09-25")
    assert changes == [{"family": "orb", "side": "long", "regime": "mixed",
                        "old_status": "", "new_status": "suspended", "reason": "bad"}]


def test_playbook_changes_today_survives_garbage():
    assert steps.playbook_changes_today(None, "2026-09-25") == []
