import asyncio
import json
from datetime import datetime

import pandas as pd
import pytest


class _Commentary:
    def __init__(self): self.cards = []
    def add_commentary(self, c): self.cards.append(c)


class _Fit:
    def __init__(self, rows, params): self.rows, self.params = rows, params
    def lookup(self, symbol, strategy, side): return self.rows.get((symbol, strategy, side))
    def params_for(self, symbol, strategy, side="long"): return self.params.get((symbol, strategy, side), {})


class _Row:
    def __init__(self, verdict, score, source="walkforward"):
        self.verdict, self.score, self.source = verdict, score, source


def _bars(n_sessions=3, last_close=100.0):
    frames = []
    for d in pd.bdate_range("2025-09-22", periods=n_sessions):
        idx = pd.date_range(f"{d.date()} 09:30", periods=78, freq="5min", name="ts")
        frames.append(pd.DataFrame({"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1000}, index=idx))
    b = pd.concat(frames)
    return b.iloc[:-30]        # today's session partially complete (ends 13:25)


def _full_bars(n_sessions=3):
    """Same shape as _bars() but without truncating the last session, so
    the last bar lands at the real RTH close (15:55)."""
    frames = []
    for d in pd.bdate_range("2025-09-22", periods=n_sessions):
        idx = pd.date_range(f"{d.date()} 09:30", periods=78, freq="5min", name="ts")
        frames.append(pd.DataFrame({"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1000}, index=idx))
    return pd.concat(frames)


def _md(symbol="NVDA", close=100.0):
    from core.models import MarketData
    return MarketData(symbol=symbol, timestamp=datetime(2025, 9, 24, 13, 32), open=close, high=close, low=close, close=close,
                      volume=1, timeframe="5min", indicators={})


def _toy_family(monkeypatch):
    from strategies.families import registry
    from strategies.families.common import Family, make_entry
    def fn(bars5, ind, daily, news, params):
        i = len(bars5) - 2                              # signal on last closed bar -> entry at synthetic bar
        e = make_entry(bars5, i, "long", 99.0, 102.0, 10, params.get("hold_kind", "intraday"), "toy")
        return [e] if e else []
    monkeypatch.setitem(registry.FAMILIES, "toy", Family("toy", fn, {"hold_kind": ["intraday"]}, "intraday", ("long",)))


def _lane(monkeypatch, tmp_path, live=False, rows=None):
    _toy_family(monkeypatch)
    from strategies.fit_lane import FitLaneStrategy
    rows = rows if rows is not None else {("NVDA", "toy", "long"): _Row("PASS", 0.4)}
    fit = _Fit(rows, {("NVDA", "toy", "long"): {"hold_kind": "intraday"}})
    src = lambda sym: (_bars(), pd.DataFrame())
    return FitLaneStrategy(_Commentary(), "toy", fit, src, lambda: ("toy",) if live else (), ledger_dir=tmp_path,
                           clock=lambda: datetime(2025, 9, 24, 13, 32))


def test_shadow_mode_writes_ledger_and_returns_none(monkeypatch, tmp_path):
    lane = _lane(monkeypatch, tmp_path, live=False)
    assert asyncio.run(lane.generate_signal_with_commentary(_md())) is None
    lines = (tmp_path / "shadow_fit_toy.ndjson").read_text().strip().splitlines()
    rec = json.loads(lines[-1])
    assert rec["symbol"] == "NVDA" and rec["side"] == "long" and rec["entry_px"] == 100.0 and rec["stop_px"] == 99.0 and rec["verdict"] == "PASS"


def test_live_mode_returns_signal_with_fit_reasoning(monkeypatch, tmp_path):
    from core.models import SignalType
    lane = _lane(monkeypatch, tmp_path, live=True)
    sig = asyncio.run(lane.generate_signal_with_commentary(_md()))
    assert sig is not None and sig.signal_type == SignalType.BUY and sig.stop_loss == 99.0 and sig.take_profit == 102.0
    r = sig.reasoning
    assert r["strategy"] == "fit:toy" and r["fit_family"] == "toy" and r["is_day_trade"] is True and r["hold_kind"] == "intraday"
    assert r["fit_params"] == {"hold_kind": "intraday"} and r["fit_verdict"] == "PASS" and 0.2 <= sig.strength <= 1.0


def test_no_row_and_insufficient_bars(monkeypatch, tmp_path):
    lane = _lane(monkeypatch, tmp_path, live=True, rows={})
    assert asyncio.run(lane.generate_signal_with_commentary(_md())) is None
    lane2 = _lane(monkeypatch, tmp_path, live=True)
    lane2._bars_source = lambda sym: (_bars().iloc[:20], pd.DataFrame())
    assert asyncio.run(lane2.generate_signal_with_commentary(_md())) is None


def test_synthetic_next_bar_and_stale_entries_ignored(monkeypatch, tmp_path, caplog):
    from strategies.fit_lane import with_synthetic_next_bar
    b = _bars()
    nb = with_synthetic_next_bar(b, 101.0)
    assert nb.index[-1] == b.index[-1] + pd.Timedelta(minutes=5) and nb["open"].iloc[-1] == 101.0 and nb["volume"].iloc[-1] == 0

    # NOTE (Task 31 review): registered under its own family name ("toy_stale")
    # and constructed directly (not via _lane()), because _lane() always calls
    # _toy_family(monkeypatch) first, which would clobber a "toy"-named stale
    # registration made before it. FitLaneStrategy also now resolves the family
    # eagerly in __init__ (review fix #2), so the family must be registered
    # before construction, not after.
    from strategies.families import registry
    from strategies.families.common import Family, make_entry
    from strategies.fit_lane import FitLaneStrategy

    def stale_fn(bars5, ind, daily, news, params):
        e = make_entry(bars5, 5, "long", 99.0, 102.0, 10, "intraday", "old")
        return [e] if e else []

    monkeypatch.setitem(registry.FAMILIES, "toy_stale", Family("toy_stale", stale_fn, {"x": [1]}, "intraday", ("long",)))
    fit = _Fit({("NVDA", "toy_stale", "long"): _Row("PASS", 0.4)}, {})
    src = lambda sym: (_bars(), pd.DataFrame())
    lane = FitLaneStrategy(_Commentary(), "toy_stale", fit, src, lambda: ("toy_stale",), ledger_dir=tmp_path,
                           clock=lambda: datetime(2025, 9, 24, 13, 32))

    with caplog.at_level("INFO", logger="TradingBot"):
        result = asyncio.run(lane.generate_signal_with_commentary(_md()))
    assert result is None
    assert not (tmp_path / "shadow_fit_toy_stale.ndjson").exists()
    assert "reason=no_entry_now" in caplog.text


def test_signal_carries_symbol_from_market_data(monkeypatch, tmp_path):
    """Review fix #1 (Critical): make_entry() never receives an explicit
    symbol from the toy family, so entry.symbol falls back to "" — the
    adapter must still emit a signal carrying the real market-data symbol."""
    lane = _lane(monkeypatch, tmp_path, live=True)
    sig = asyncio.run(lane.generate_signal_with_commentary(_md(symbol="NVDA")))
    assert sig is not None and sig.symbol == "NVDA"


def test_family_fn_exception_is_contained(monkeypatch, tmp_path):
    """Review fix #2 (Important): a family fn that raises must not escape
    generate_signal_with_commentary."""
    from strategies.families import registry
    from strategies.families.common import Family
    from strategies.fit_lane import FitLaneStrategy

    def boom_fn(bars5, ind, daily, news, params):
        raise RuntimeError("boom")

    monkeypatch.setitem(registry.FAMILIES, "boom", Family("boom", boom_fn, {"x": [1]}, "intraday", ("long",)))
    fit = _Fit({("NVDA", "boom", "long"): _Row("PASS", 0.4)}, {})
    src = lambda sym: (_bars(), pd.DataFrame())
    lane = FitLaneStrategy(_Commentary(), "boom", fit, src, lambda: ("boom",), ledger_dir=tmp_path,
                           clock=lambda: datetime(2025, 9, 24, 13, 32))
    assert asyncio.run(lane.generate_signal_with_commentary(_md())) is None


def test_ledger_write_failure_does_not_raise(monkeypatch, tmp_path):
    """Review fix #2 (Important): an unwritable ledger dir must not stop a
    live decision from being returned."""
    lane = _lane(monkeypatch, tmp_path, live=True)
    bad_dir = tmp_path / "not_a_dir"
    bad_dir.write_text("i am a file, not a directory")
    lane._ledger_dir = bad_dir
    sig = asyncio.run(lane.generate_signal_with_commentary(_md()))
    assert sig is not None


def test_unknown_family_raises_at_construction(monkeypatch, tmp_path):
    """Review fix #2 (Important): a typo'd family name fails fast at
    construction (caught by the engine's registration loop), not on
    every analysis tick."""
    from strategies.fit_lane import FitLaneStrategy
    fit = _Fit({}, {})
    src = lambda sym: (_bars(), pd.DataFrame())
    with pytest.raises(KeyError):
        FitLaneStrategy(_Commentary(), "no_such_family_xyz", fit, src, lambda: (), ledger_dir=tmp_path)


def test_stale_bars_after_close_skipped(monkeypatch, tmp_path, caplog):
    """Review fix #3 (Important): clock well after the close with bars
    ending at 15:55 -> the synthetic bar would land at 16:00, past RTH
    close, so the adapter skips instead of polluting the ledger."""
    lane = _lane(monkeypatch, tmp_path, live=True)
    lane._bars_source = lambda sym: (_full_bars(3), pd.DataFrame())
    lane._clock = lambda: datetime(2025, 9, 24, 16, 7)
    with caplog.at_level("INFO", logger="TradingBot"):
        result = asyncio.run(lane.generate_signal_with_commentary(_md()))
    assert result is None
    assert not (tmp_path / "shadow_fit_toy.ndjson").exists()
    assert "reason=stale_bars" in caplog.text


def test_stale_bars_before_synthetic_bar_prints(monkeypatch, tmp_path, caplog):
    """Review fix #3 (Important): clock before the synthetic bar's own
    timestamp -> nothing to evaluate yet, so the snapshot is rejected.

    v-strategy-fit-review2-2026-09-25 (Important 4): the last closed bar is
    yesterday's 15:55, so the synthetic bar is today 09:30; a 09:20 clock is
    still ahead of it."""
    lane = _lane(monkeypatch, tmp_path, live=True)
    lane._bars_source = lambda sym: (_full_bars(2), pd.DataFrame())
    lane._clock = lambda: datetime(2025, 9, 24, 9, 20)
    with caplog.at_level("INFO", logger="TradingBot"):
        result = asyncio.run(lane.generate_signal_with_commentary(_md()))
    assert result is None
    assert not (tmp_path / "shadow_fit_toy.ndjson").exists()
    assert "reason=stale_bars" in caplog.text


def test_stale_bars_past_opening_window(monkeypatch, tmp_path, caplog):
    """v-strategy-fit-review2-2026-09-25 (Important 4): the opening window is
    09:30-09:45 (synthetic bar + 15 min). A 09:50 clock is past it, so the
    session-open snapshot is stale even though the market is open."""
    lane = _lane(monkeypatch, tmp_path, live=True)
    lane._bars_source = lambda sym: (_full_bars(2), pd.DataFrame())
    lane._clock = lambda: datetime(2025, 9, 24, 9, 50)
    with caplog.at_level("INFO", logger="TradingBot"):
        result = asyncio.run(lane.generate_signal_with_commentary(_md()))
    assert result is None
    assert not (tmp_path / "shadow_fit_toy.ndjson").exists()
    assert "reason=stale_bars" in caplog.text


def test_insufficient_closed_bars(monkeypatch, tmp_path, caplog):
    """Review fix #4 (Important): the <60 bar floor applies to *closed*
    bars, not raw bars -- 70 raw bars of which only 50 are closed must
    still skip with reason insufficient_bars."""
    idx = pd.date_range("2025-09-24 09:30", periods=70, freq="5min", name="ts")
    bars = pd.DataFrame({"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1000}, index=idx)
    lane = _lane(monkeypatch, tmp_path, live=True)
    lane._bars_source = lambda sym: (bars, pd.DataFrame())
    lane._clock = lambda: datetime(2025, 9, 24, 13, 44)     # cutoff 13:39 -> keeps bars 09:30..13:25 (50 bars)
    with caplog.at_level("INFO", logger="TradingBot"):
        result = asyncio.run(lane.generate_signal_with_commentary(_md()))
    assert result is None
    assert "reason=insufficient_bars" in caplog.text


def test_empty_bars_skip_insufficient_bars_no_error(monkeypatch, tmp_path, caplog):
    """Regression (Task 31 re-review): the engine's _fit_bars_source degrades
    to two empty DataFrames on any fetch failure. closed_bars() raises
    TypeError on an empty frame's default RangeIndex, so bars5.empty must be
    checked *before* calling closed_bars() -- otherwise every fetch failure
    is misreported as a caught exception (action=error reason=fit_lane_error)
    instead of the expected clean insufficient_bars skip, and a spurious
    warning is logged on every tick for every symbol/family."""
    lane = _lane(monkeypatch, tmp_path, live=True)
    lane._bars_source = lambda sym: (pd.DataFrame(), pd.DataFrame())
    with caplog.at_level("DEBUG", logger="TradingBot"):
        result = asyncio.run(lane.generate_signal_with_commentary(_md()))
    assert result is None
    assert "reason=insufficient_bars" in caplog.text
    assert "reason=fit_lane_error" not in caplog.text
    assert not any(r.levelname == "WARNING" for r in caplog.records)


# ──────────────────────────────────────────────────────────────────────
# v-strategy-fit-review2-2026-09-25 — Critical 1: live entries require a
# walk-forward PASS row for the entry's own side.
# ──────────────────────────────────────────────────────────────────────

def _ledger_lines(tmp_path, family="toy"):
    p = tmp_path / f"shadow_fit_{family}.ndjson"
    return [json.loads(l) for l in p.read_text().strip().splitlines()] if p.exists() else []


def test_watch_row_in_live_family_stays_shadow_with_ledger_line(monkeypatch, tmp_path, caplog):
    """Critical 1: a WATCH row must not emit a live signal just because the
    family is in FIT_LANE_LIVE_ENTRIES_ENABLED. The entry is still recorded
    in the shadow ledger so the evidence keeps accumulating."""
    lane = _lane(monkeypatch, tmp_path, live=True,
                 rows={("NVDA", "toy", "long"): _Row("WATCH", 0.2, "walkforward")})
    with caplog.at_level("INFO", logger="TradingBot"):
        result = asyncio.run(lane.generate_signal_with_commentary(_md()))
    assert result is None
    assert "reason=fit_lane_shadow_verdict" in caplog.text
    recs = _ledger_lines(tmp_path)
    assert len(recs) == 1 and recs[0]["verdict"] == "WATCH" and recs[0]["symbol"] == "NVDA"


def test_pass_walkforward_row_in_live_family_emits_signal(monkeypatch, tmp_path):
    """Critical 1: the one combination that may go live — PASS from the
    walk-forward pipeline, family enabled for live entries."""
    from core.models import SignalType
    lane = _lane(monkeypatch, tmp_path, live=True,
                 rows={("NVDA", "toy", "long"): _Row("PASS", 0.4, "walkforward")})
    sig = asyncio.run(lane.generate_signal_with_commentary(_md()))
    assert sig is not None and sig.signal_type == SignalType.BUY
    assert sig.reasoning["fit_verdict"] == "PASS"


def test_pass_coldstart_row_in_live_family_stays_shadow(monkeypatch, tmp_path, caplog):
    """Critical 1: PASS alone isn't enough — a cold-start row has no
    walk-forward evidence behind it, so it stays shadow."""
    lane = _lane(monkeypatch, tmp_path, live=True,
                 rows={("NVDA", "toy", "long"): _Row("PASS", 0.4, "coldstart")})
    with caplog.at_level("INFO", logger="TradingBot"):
        result = asyncio.run(lane.generate_signal_with_commentary(_md()))
    assert result is None
    assert "reason=fit_lane_shadow_verdict" in caplog.text
    assert len(_ledger_lines(tmp_path)) == 1


def test_live_eligible_predicate():
    from strategies.fit_lane import live_eligible
    assert live_eligible(_Row("PASS", 0.4, "walkforward")) is True
    assert live_eligible(_Row("PASS", 0.4, "coldstart")) is False
    assert live_eligible(_Row("WATCH", 0.4, "walkforward")) is False
    assert live_eligible(_Row("FAIL", 0.4, "walkforward")) is False
    assert live_eligible(None) is False


def test_clamp_guards_nan_score(monkeypatch, tmp_path):
    """Minor: a NaN score must clamp to the 0.2 floor, not become strength
    1.0 (or NaN) on the emitted signal."""
    from strategies.fit_lane import _clamp
    assert _clamp(float("nan"), 0.2, 1.0) == 0.2
    assert _clamp(float("inf"), 0.2, 1.0) == 0.2
    assert _clamp(None, 0.2, 1.0) == 0.2
    assert _clamp(0.6, 0.2, 1.0) == 0.6

    lane = _lane(monkeypatch, tmp_path, live=True,
                 rows={("NVDA", "toy", "long"): _Row("PASS", float("nan"), "walkforward")})
    sig = asyncio.run(lane.generate_signal_with_commentary(_md()))
    assert sig is not None and sig.strength == 0.2 and sig.confidence == 0.2


# ──────────────────────────────────────────────────────────────────────
# v-strategy-fit-review2-2026-09-25 — Important 4: session-open mode.
# ──────────────────────────────────────────────────────────────────────

def test_next_rth_bar_maps_1555_to_next_weekday_open():
    from strategies.fit_lane import next_rth_bar
    assert next_rth_bar(pd.Timestamp("2025-09-23 15:55")) == pd.Timestamp("2025-09-24 09:30")
    assert next_rth_bar(pd.Timestamp("2025-09-26 15:55")) == pd.Timestamp("2025-09-29 09:30")   # Fri -> Mon
    assert next_rth_bar(pd.Timestamp("2025-09-24 13:25")) == pd.Timestamp("2025-09-24 13:30")


def _session_open_family(monkeypatch, name="toy_open"):
    """Toy family shaped like the real session-open families: it signals at
    bar_of_day == 0 via make_entry(bars5, j - 1, ...) with a multiday hold."""
    from strategies.families import registry
    from strategies.families.common import Family, make_entry
    import numpy as np

    def fn(bars5, ind, daily, news, params):
        out = []
        for j in [int(i) for i in np.flatnonzero(ind["bar_of_day"].to_numpy() == 0) if i > 0]:
            px = float(bars5["open"].iloc[j])
            e = make_entry(bars5, j - 1, "long", px - 1.0, px + 2.0, 10, "multiday", "toy_open")
            if e:
                out.append(e)
        return out

    monkeypatch.setitem(registry.FAMILIES, name, Family(name, fn, {"x": [1]}, "multiday", ("long",)))
    return name


def _daily_for(bars5):
    """Daily frame covering every session present in `bars5` (not the new one)."""
    g = bars5.groupby(bars5.index.normalize())
    d = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                      "close": g["close"].last(), "volume": g["volume"].sum()})
    d.index.name = "date"
    return d


def _open_lane(monkeypatch, tmp_path, clock, bars5, rows=None):
    from strategies.fit_lane import FitLaneStrategy
    name = _session_open_family(monkeypatch)
    rows = rows if rows is not None else {("NVDA", name, "long"): _Row("PASS", 0.4, "walkforward")}
    fit = _Fit(rows, {})
    src = lambda sym: (bars5, _daily_for(bars5))
    return FitLaneStrategy(_Commentary(), name, fit, src, lambda: (name,), ledger_dir=tmp_path,
                           clock=lambda: clock)


def test_session_open_family_fires_in_opening_window(monkeypatch, tmp_path):
    """Important 4: last closed bar is yesterday's 15:55 and the clock is
    09:33 today -> the synthetic bar is today's 09:30 open, so a
    bar_of_day == 0 family is reachable live."""
    bars5 = _full_bars(3)                       # 09/22..09/24, last bar 2025-09-24 15:55
    lane = _open_lane(monkeypatch, tmp_path, datetime(2025, 9, 25, 9, 33), bars5)
    sig = asyncio.run(lane.generate_signal_with_commentary(_md()))
    assert sig is not None
    recs = _ledger_lines(tmp_path, "toy_open")
    assert recs and recs[-1]["ts"] == "2025-09-25T09:30:00" and recs[-1]["hold_kind"] == "multiday"


def test_session_open_family_stale_past_0945(monkeypatch, tmp_path, caplog):
    """Important 4: the same snapshot at 09:50 is past the opening window."""
    bars5 = _full_bars(3)
    lane = _open_lane(monkeypatch, tmp_path, datetime(2025, 9, 25, 9, 50), bars5)
    with caplog.at_level("INFO", logger="TradingBot"):
        result = asyncio.run(lane.generate_signal_with_commentary(_md()))
    assert result is None
    assert "reason=stale_bars" in caplog.text


def test_session_open_friday_close_maps_to_monday(monkeypatch, tmp_path):
    """Important 4: a Friday 15:55 last close with a Monday 09:33 clock
    yields Monday's 09:30 synthetic bar (the weekend is skipped)."""
    frames = []
    for d in pd.bdate_range("2025-09-24", periods=3):        # Wed, Thu, Fri 2025-09-26
        idx = pd.date_range(f"{d.date()} 09:30", periods=78, freq="5min", name="ts")
        frames.append(pd.DataFrame({"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0,
                                    "volume": 1000}, index=idx))
    bars5 = pd.concat(frames)
    assert bars5.index[-1] == pd.Timestamp("2025-09-26 15:55")
    lane = _open_lane(monkeypatch, tmp_path, datetime(2025, 9, 29, 9, 33), bars5)
    sig = asyncio.run(lane.generate_signal_with_commentary(_md()))
    assert sig is not None
    assert _ledger_lines(tmp_path, "toy_open")[-1]["ts"] == "2025-09-29T09:30:00"


def test_with_daily_placeholder_adds_new_session_date():
    from strategies.fit_lane import with_daily_placeholder
    d = _daily_for(_full_bars(2))
    out = with_daily_placeholder(d, pd.Timestamp("2025-09-24 09:30"))
    assert pd.Timestamp("2025-09-24") in out.index and len(out) == len(d) + 1
    assert out["volume"].iloc[-1] == 0 and pd.isna(out["close"].iloc[-1])
    # Idempotent for a date already present, and safe on an empty frame.
    assert with_daily_placeholder(out, pd.Timestamp("2025-09-24 09:30")).equals(out)
    assert with_daily_placeholder(pd.DataFrame(), pd.Timestamp("2025-09-24")).empty


def test_fit_lane_frames_module_is_pure_and_reexported():
    """v-strategy-fit-review2-2026-09-25: the pure helpers were split into
    strategies/fit_lane_frames.py for the 400-line file budget. That module
    must stay pure (no research/core/api imports) and every name must remain
    reachable from strategies.fit_lane so existing call sites don't break."""
    import pathlib
    import re

    import strategies.fit_lane as F

    src = pathlib.Path("strategies/fit_lane_frames.py").read_text()
    assert not re.search(r"^\s*(from|import)\s+(research|core|api)\b", src, re.M)

    for name in ("with_synthetic_next_bar", "next_rth_bar", "with_daily_placeholder",
                 "live_eligible", "news_frame_from_items", "ledger_path", "entry_to_signal",
                 "FitLaneStrategy"):
        assert hasattr(F, name), name

    # Both halves stay inside the project's 400-line file budget.
    for path in ("strategies/fit_lane.py", "strategies/fit_lane_frames.py"):
        assert len(pathlib.Path(path).read_text().splitlines()) < 400, path
