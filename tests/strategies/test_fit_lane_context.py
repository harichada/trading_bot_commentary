"""v-go-live-5k-2026-09-28 (Task 2): the live fit lane's universe-context slice.

`mom_12_1` (and every other family reading a `ctx_*` column) could never fire live: the lane
called `indicator_frame(bars, daily, spy5)` with no `context=`, so every `ctx_*` column was
absent and `ctx()` returned all-NaN. The nightly research export writes one row per session per
symbol; at a session's open its newest row is the *previous* session's, which is exactly the
known-at-open contract — so the slice is carried forward onto the new session, with the
session-local columns (gap, opening rvol, today's earnings) masked on a carried row because
those describe a session that has not happened yet.

Offline: hand-built frames, an injected context loader, no provider, no parquet.
"""
import asyncio
from datetime import datetime

import numpy as np
import pandas as pd
import pytest


# ──────────────────────────────────────────────────────────────────────
# align_context (pure)
# ──────────────────────────────────────────────────────────────────────

def _ctx_frame(dates, **cols):
    idx = pd.DatetimeIndex(pd.to_datetime(list(dates)), name="date")
    return pd.DataFrame({k: [v] * len(idx) for k, v in cols.items()}, index=idx)


def _bar_index(days):
    return pd.DatetimeIndex(
        [ts for d in days for ts in pd.date_range(f"{d} 09:30", periods=3, freq="5min")], name="ts")


def test_align_context_matches_exact_session_dates():
    from strategies.fit_lane_frames import align_context
    ctx = _ctx_frame(["2025-11-27", "2025-11-28"], mom_rank_12_1=0.9, gap_pct=5.0)
    out = align_context(ctx, _bar_index(["2025-11-27", "2025-11-28"]))
    assert list(out.index) == [pd.Timestamp("2025-11-27"), pd.Timestamp("2025-11-28")]
    assert out["gap_pct"].tolist() == [5.0, 5.0]


def test_align_context_carries_the_prior_session_forward():
    """The one-session lag: at 12-01's open the newest row is 11-28's."""
    from strategies.fit_lane_frames import align_context
    ctx = _ctx_frame(["2025-11-27", "2025-11-28"], mom_rank_12_1=0.9, gap_pct=5.0,
                     rvol_open5=3.0, earnings_today=True)
    out = align_context(ctx, _bar_index(["2025-11-28", "2025-12-01"]))
    carried = out.loc[pd.Timestamp("2025-12-01")]
    assert carried["mom_rank_12_1"] == 0.9                      # slow factor: carried
    for session_local in ("gap_pct", "rvol_open5", "earnings_today"):
        assert pd.isna(carried[session_local]), session_local    # describes a session not yet run
    assert out.loc[pd.Timestamp("2025-11-28"), "gap_pct"] == 5.0   # exact rows keep everything


def test_align_context_stops_carrying_a_badly_stale_export():
    """If the nightly refresh has been broken for days the lane must stand down rather than
    trade on a stale cross-section."""
    from strategies.fit_lane_frames import CTX_MAX_CARRY_SESSIONS, align_context
    days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2025-11-03", periods=CTX_MAX_CARRY_SESSIONS + 2)]
    ctx = _ctx_frame([days[0]], mom_rank_12_1=0.9)
    out = align_context(ctx, _bar_index(days))
    assert out["mom_rank_12_1"].iloc[1] == 0.9                   # within the carry window
    assert pd.isna(out["mom_rank_12_1"].iloc[-1])                # past it


def test_align_context_handles_missing_and_empty_inputs():
    from strategies.fit_lane_frames import align_context
    assert align_context(None, _bar_index(["2025-11-28"])) is None
    assert align_context(pd.DataFrame(), _bar_index(["2025-11-28"])) is None
    ctx = _ctx_frame(["2025-11-28"], mom_rank_12_1=0.9)
    assert align_context(ctx, pd.DatetimeIndex([])) is None
    # The caller's frame is never mutated.
    align_context(ctx, _bar_index(["2025-12-01"]))
    assert ctx["mom_rank_12_1"].tolist() == [0.9]


# ──────────────────────────────────────────────────────────────────────
# a real mom_12_1 entry through the live lane
# ──────────────────────────────────────────────────────────────────────

SESSION = "2025-12-01"          # first business day of the month
PRIOR = "2025-11-28"


class _Commentary:
    def __init__(self): self.cards = []
    def add_commentary(self, c): self.cards.append(c)


class _Fit:
    def __init__(self, rows, params): self.rows, self.params = rows, params
    def lookup(self, symbol, strategy, side): return self.rows.get((symbol, strategy, side))
    def params_for(self, symbol, strategy, side="long"): return self.params.get((symbol, strategy, side), {})


def _session_bars():
    """Two full prior sessions plus the new session's 09:30 bar (the only closed bar of the day
    at 09:38, which is when a bar_of_day == 0 family's next-bar fill is 'now')."""
    frames = []
    for d in pd.bdate_range(end=PRIOR, periods=2):
        idx = pd.date_range(f"{d.date()} 09:30", periods=78, freq="5min", name="ts")
        frames.append(pd.DataFrame({"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0,
                                    "volume": 1000}, index=idx))
    frames.append(pd.DataFrame({"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1000},
                               index=pd.DatetimeIndex([pd.Timestamp(f"{SESSION} 09:30")], name="ts")))
    return pd.concat(frames)


def _session_daily():
    """210 rising sessions through the prior close: d_close > d_sma200 and a finite d_atr14."""
    idx = pd.bdate_range(end=PRIOR, periods=210)
    close = pd.Series(np.linspace(60.0, 100.0, len(idx)), index=idx)
    d = pd.DataFrame({"open": close, "high": close + 1.0, "low": close - 1.0, "close": close,
                      "volume": 1_000_000.0}, index=idx)
    d.index.name = "date"
    return d


def _md(symbol="NVDA", close=100.0):
    from core.models import MarketData
    return MarketData(symbol=symbol, timestamp=datetime(2025, 12, 1, 9, 38), open=close, high=close,
                      low=close, close=close, volume=1, timeframe="5min", indicators={})


def _mom_lane(tmp_path, context, rows=None, allowlist="mom_12_1:long"):
    from strategies.fit_lane import FitLaneStrategy
    fit = _Fit(rows or {}, {("NVDA", "mom_12_1", "long"): {"rank_min": 0.8, "entry_style": "market"}})
    return FitLaneStrategy(
        _Commentary(), "mom_12_1", fit, lambda sym: (_session_bars(), _session_daily()),
        lambda: ("mom_12_1",), ledger_dir=tmp_path,
        clock=lambda: datetime(2025, 12, 1, 9, 38),
        allowlist=lambda: allowlist,
        regime_source=lambda sym: "risk_on",
        context_source=lambda sym: context,
    )


def test_mom_12_1_fires_live_on_a_carried_forward_context_slice(tmp_path):
    from core.models import SignalType
    ctx = _ctx_frame([d.strftime("%Y-%m-%d") for d in pd.bdate_range(end=PRIOR, periods=5)],
                     mom_rank_12_1=0.95)
    sig = asyncio.run(_mom_lane(tmp_path, ctx).generate_signal_with_commentary(_md()))
    assert sig is not None and sig.signal_type == SignalType.BUY and sig.symbol == "NVDA"
    assert sig.reasoning["fit_family"] == "mom_12_1" and sig.reasoning["hold_kind"] == "multiday"
    assert sig.reasoning["entry_pattern"] == "mom_12_1_long"
    assert sig.reasoning["fit_lane_admission"] == "allowlist"
    assert sig.reasoning["market_context_regime"] == "risk_on"
    assert sig.entry_price == pytest.approx(100.0) and sig.stop_loss < 100.0 < sig.take_profit


def test_mom_12_1_does_not_fire_without_the_context_slice(tmp_path, caplog):
    """The pre-Task-2 state: no ctx_* columns -> mom_rank_12_1 is NaN -> no entry, no raise."""
    lane = _mom_lane(tmp_path, None)
    with caplog.at_level("DEBUG", logger="TradingBot"):
        assert asyncio.run(lane.generate_signal_with_commentary(_md())) is None
    assert "reason=no_entry_now" in caplog.text
    assert "reason=fit_lane_error" not in caplog.text
    assert not any(r.levelname == "WARNING" for r in caplog.records)


def test_mom_12_1_does_not_fire_on_an_empty_context_slice(tmp_path, caplog):
    lane = _mom_lane(tmp_path, pd.DataFrame())
    with caplog.at_level("DEBUG", logger="TradingBot"):
        assert asyncio.run(lane.generate_signal_with_commentary(_md())) is None
    assert "reason=no_entry_now" in caplog.text
    assert "reason=fit_lane_error" not in caplog.text


def test_context_loader_failure_does_not_break_the_lane(tmp_path, caplog):
    def boom(symbol):
        raise RuntimeError("unreadable parquet")

    lane = _mom_lane(tmp_path, None)
    lane._context_source = boom
    with caplog.at_level("DEBUG", logger="TradingBot"):
        assert asyncio.run(lane.generate_signal_with_commentary(_md())) is None
    assert "reason=fit_lane_error" not in caplog.text


def test_mom_12_1_context_slice_out_of_carry_range_does_not_fire(tmp_path, caplog):
    """A context export that stopped a fortnight ago must not authorise a live factor entry."""
    ctx = _ctx_frame(["2025-11-03"], mom_rank_12_1=0.95)
    lane = _mom_lane(tmp_path, ctx)
    with caplog.at_level("INFO", logger="TradingBot"):
        assert asyncio.run(lane.generate_signal_with_commentary(_md())) is None
    assert "reason=no_entry_now" in caplog.text


def test_default_context_source_reads_the_configured_root(monkeypatch, tmp_path):
    """With no injected loader the lane reads `<STRATEGY_HUNT_DATA_ROOT>/context/...` through
    core.strategy_fit_bars.load_context_slice."""
    import strategies.fit_lane as F
    seen = {}

    def fake_loader(root, symbol):
        seen["args"] = (root, symbol)
        return _ctx_frame([d.strftime("%Y-%m-%d") for d in pd.bdate_range(end=PRIOR, periods=5)],
                          mom_rank_12_1=0.95)

    monkeypatch.setattr(F, "load_context_slice", fake_loader)
    monkeypatch.setenv("STRATEGY_HUNT_DATA_ROOT", str(tmp_path / "hunt"))
    fit = _Fit({}, {("NVDA", "mom_12_1", "long"): {"rank_min": 0.8, "entry_style": "market"}})
    lane = F.FitLaneStrategy(_Commentary(), "mom_12_1", fit,
                             lambda sym: (_session_bars(), _session_daily()),
                             lambda: ("mom_12_1",), ledger_dir=tmp_path,
                             clock=lambda: datetime(2025, 12, 1, 9, 38),
                             allowlist=lambda: "mom_12_1:long",
                             regime_source=lambda sym: "risk_on")
    sig = asyncio.run(lane.generate_signal_with_commentary(_md()))
    assert seen["args"] == (str(tmp_path / "hunt"), "NVDA")
    assert sig is not None and sig.reasoning["fit_family"] == "mom_12_1"
