import asyncio
import json
from datetime import datetime

import numpy as np
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
    def __init__(self, verdict, score): self.verdict, self.score = verdict, score


def _bars(n_sessions=3, last_close=100.0):
    frames = []
    for d in pd.bdate_range("2025-09-22", periods=n_sessions):
        idx = pd.date_range(f"{d.date()} 09:30", periods=78, freq="5min", name="ts")
        frames.append(pd.DataFrame({"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1000}, index=idx))
    b = pd.concat(frames)
    return b.iloc[:-30]        # today's session partially complete (ends 13:25)


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


def test_synthetic_next_bar_and_stale_entries_ignored(monkeypatch, tmp_path):
    from strategies.fit_lane import with_synthetic_next_bar
    b = _bars()
    nb = with_synthetic_next_bar(b, 101.0)
    assert nb.index[-1] == b.index[-1] + pd.Timedelta(minutes=5) and nb["open"].iloc[-1] == 101.0 and nb["volume"].iloc[-1] == 0
    from strategies.families import registry
    from strategies.families.common import Family, make_entry
    def stale_fn(bars5, ind, daily, news, params):
        e = make_entry(bars5, 5, "long", 99.0, 102.0, 10, "intraday", "old")
        return [e] if e else []
    # NOTE (Task 31): _lane() calls _toy_family(monkeypatch) as its first
    # statement, which re-registers the normal (non-stale) "toy" family.
    # If the stale registration below ran *before* _lane(), it would be
    # clobbered and this test would spuriously exercise the normal family
    # instead of the stale one. Register the stale family after
    # constructing the lane so it's the one in effect when
    # generate_signal_with_commentary() does its (call-time) get_family lookup.
    lane = _lane(monkeypatch, tmp_path, live=True)
    monkeypatch.setitem(registry.FAMILIES, "toy", Family("toy", stale_fn, {"x": [1]}, "intraday", ("long",)))
    assert asyncio.run(lane.generate_signal_with_commentary(_md())) is None
