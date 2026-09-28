"""v-go-live-5k-2026-09-28 (Task 2): the live fit lane under FIT_LANE_LIVE_ALLOWLIST.

Lane-level counterpart to tests/strategies/test_fit_lane_admission.py (which tests the pure
rule): the allowlist has to survive the whole adapter path — the no-row early return, the
ledger, the best-of pick and the emitted signal's reasoning — and the live market-context
regime has to reach both the admission decision and `reasoning["market_context_regime"]` so
the downstream fit gate and playbook gate see the same regime the lane admitted on.

Offline: a toy family, hand-built frames, no provider, no network.
"""
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
    """n/exp_r default to a thin (unmeasurable) sample, which never blocks — fix round 1."""

    def __init__(self, verdict, score=0.4, source="walkforward", n=10, exp_r=0.0):
        self.verdict, self.score, self.source = verdict, score, source
        self.n, self.exp_r = n, exp_r


def _bars(n_sessions=3):
    frames = []
    for d in pd.bdate_range("2025-09-22", periods=n_sessions):
        idx = pd.date_range(f"{d.date()} 09:30", periods=78, freq="5min", name="ts")
        frames.append(pd.DataFrame({"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0,
                                    "volume": 1000}, index=idx))
    return pd.concat(frames).iloc[:-30]          # today's session ends 13:25


def _md(symbol="NVDA", close=100.0):
    from core.models import MarketData
    return MarketData(symbol=symbol, timestamp=datetime(2025, 9, 24, 13, 32), open=close, high=close,
                      low=close, close=close, volume=1, timeframe="5min", indicators={})


def _toy_family(monkeypatch, name="toy"):
    from strategies.families import registry
    from strategies.families.common import Family, make_entry

    def fn(bars5, ind, daily, news, params):
        e = make_entry(bars5, len(bars5) - 2, "long", 99.0, 102.0, 10, "intraday", "toy")
        return [e] if e else []

    monkeypatch.setitem(registry.FAMILIES, name, Family(name, fn, {"hold_kind": ["intraday"]},
                                                        "intraday", ("long",)))
    return name


def _lane(monkeypatch, tmp_path, rows, allowlist="", regime="mixed", live=True):
    name = _toy_family(monkeypatch)
    from strategies.fit_lane import FitLaneStrategy
    fit = _Fit(rows, {("NVDA", name, "long"): {"hold_kind": "intraday"}})
    return FitLaneStrategy(
        _Commentary(), name, fit, lambda sym: (_bars(), pd.DataFrame()),
        lambda: (name,) if live else (), ledger_dir=tmp_path,
        clock=lambda: datetime(2025, 9, 24, 13, 32),
        allowlist=lambda: allowlist,
        regime_source=lambda sym: regime,
    )


def _ledger(tmp_path, family="toy"):
    p = tmp_path / f"shadow_fit_{family}.ndjson"
    return [json.loads(l) for l in p.read_text().strip().splitlines()] if p.exists() else []


# ──────────────────────────────────────────────────────────────────────
# default (flag "") — today's behaviour, byte for byte
# ──────────────────────────────────────────────────────────────────────

def test_default_empty_allowlist_keeps_watch_rows_in_shadow(monkeypatch, tmp_path, caplog):
    lane = _lane(monkeypatch, tmp_path, {("NVDA", "toy", "long"): _Row("WATCH")}, allowlist="")
    with caplog.at_level("INFO", logger="TradingBot"):
        assert asyncio.run(lane.generate_signal_with_commentary(_md())) is None
    assert "reason=fit_lane_shadow_verdict" in caplog.text
    assert len(_ledger(tmp_path)) == 1


def test_walkforward_pass_row_is_admitted_as_pass(monkeypatch, tmp_path):
    lane = _lane(monkeypatch, tmp_path, {("NVDA", "toy", "long"): _Row("PASS")}, allowlist="")
    sig = asyncio.run(lane.generate_signal_with_commentary(_md()))
    assert sig is not None and sig.reasoning["fit_lane_admission"] == "pass"
    assert sig.reasoning["market_context_regime"] == "mixed"


# ──────────────────────────────────────────────────────────────────────
# the allowlist admission matrix, through the lane
# ──────────────────────────────────────────────────────────────────────

def test_allowlisted_watch_row_goes_live(monkeypatch, tmp_path, caplog):
    from core.models import SignalType
    lane = _lane(monkeypatch, tmp_path, {("NVDA", "toy", "long"): _Row("WATCH", 0.3)},
                 allowlist="toy:long")
    with caplog.at_level("INFO", logger="TradingBot"):
        sig = asyncio.run(lane.generate_signal_with_commentary(_md()))
    assert sig is not None and sig.signal_type == SignalType.BUY
    assert sig.reasoning["fit_lane_admission"] == "allowlist"
    assert sig.reasoning["fit_verdict"] == "WATCH"
    assert "reason=fit_lane_live" in caplog.text
    assert len(_ledger(tmp_path)) == 1          # the shadow evidence keeps accumulating


def test_allowlisted_measured_loser_stays_shadow(monkeypatch, tmp_path, caplog):
    """Fix round 1: only a *measured* loser (n >= 60, exp_r < 0) blocks an allowlisted lane."""
    lane = _lane(monkeypatch, tmp_path,
                 {("NVDA", "toy", "long"): _Row("FAIL", n=150, exp_r=-0.08)},
                 allowlist="toy:long")
    with caplog.at_level("INFO", logger="TradingBot"):
        assert asyncio.run(lane.generate_signal_with_commentary(_md())) is None
    assert "reason=fit_lane_shadow_verdict" in caplog.text
    assert "fit_lane_symbol_check=blocked_negative" in caplog.text


def test_allowlisted_thin_fail_row_goes_live(monkeypatch, tmp_path):
    """The sleeve's actual situation: every mr_20d/mom_12_1 row is FAIL on the sample floor
    (n < 60), so the lane must trade it on the pooled family evidence."""
    lane = _lane(monkeypatch, tmp_path,
                 {("NVDA", "toy", "long"): _Row("FAIL", n=12, exp_r=-0.4)},
                 allowlist="toy:long")
    sig = asyncio.run(lane.generate_signal_with_commentary(_md()))
    assert sig is not None and sig.reasoning["fit_lane_admission"] == "allowlist"
    assert sig.reasoning["fit_lane_symbol_check"] == "thin"


def test_admitted_signal_records_the_symbol_check_and_neutral_sizing(monkeypatch, tmp_path):
    lane = _lane(monkeypatch, tmp_path, {}, allowlist="toy:long")
    sig = asyncio.run(lane.generate_signal_with_commentary(_md()))
    assert sig is not None
    assert sig.reasoning["fit_lane_symbol_check"] == "absent"
    assert sig.reasoning["sizing_basis"] == "risk_per_trade"
    assert sig.strength == 0.6 and sig.confidence == 0.6


def test_pass_row_keeps_its_score_based_sizing(monkeypatch, tmp_path):
    lane = _lane(monkeypatch, tmp_path,
                 {("NVDA", "toy", "long"): _Row("PASS", 0.4, n=200, exp_r=0.3)}, allowlist="")
    sig = asyncio.run(lane.generate_signal_with_commentary(_md()))
    assert sig is not None and sig.reasoning["sizing_basis"] == "fit_score"
    assert sig.strength == pytest.approx(0.8)
    assert sig.reasoning["fit_lane_symbol_check"] == "positive"


def test_allowlisted_wrong_regime_stays_shadow(monkeypatch, tmp_path, caplog):
    lane = _lane(monkeypatch, tmp_path, {("NVDA", "toy", "long"): _Row("WATCH")},
                 allowlist="toy:long:risk_on", regime="risk_off")
    with caplog.at_level("INFO", logger="TradingBot"):
        assert asyncio.run(lane.generate_signal_with_commentary(_md())) is None
    assert "reason=fit_lane_shadow_verdict" in caplog.text


def test_allowlisted_unknown_regime_counts_as_mixed(monkeypatch, tmp_path):
    lane = _lane(monkeypatch, tmp_path, {("NVDA", "toy", "long"): _Row("WATCH")},
                 allowlist="toy:long:mixed", regime="unknown")
    sig = asyncio.run(lane.generate_signal_with_commentary(_md()))
    assert sig is not None and sig.reasoning["fit_lane_admission"] == "allowlist"
    assert sig.reasoning["market_context_regime"] == "unknown"    # audited as read, not as mapped


def test_allowlisted_family_not_in_live_entries_stays_shadow(monkeypatch, tmp_path, caplog):
    """The allowlist widens *which rows* a live family may take; it never makes a family live.
    FIT_LANE_LIVE_ENTRIES_ENABLED is still the master switch."""
    lane = _lane(monkeypatch, tmp_path, {("NVDA", "toy", "long"): _Row("WATCH")},
                 allowlist="toy:long", live=False)
    with caplog.at_level("INFO", logger="TradingBot"):
        assert asyncio.run(lane.generate_signal_with_commentary(_md())) is None
    assert "reason=fit_lane_shadow" in caplog.text


def test_allowlisted_side_with_no_row_at_all_goes_live(monkeypatch, tmp_path):
    """Most sleeve symbols have no measured row for the lane; with the family allowlisted the
    lane must still reach admission (the pre-existing `fit_no_row` early return would have
    skipped the symbol before the entry was ever computed) and the ledger line must survive a
    missing row."""
    lane = _lane(monkeypatch, tmp_path, {}, allowlist="toy:long")
    sig = asyncio.run(lane.generate_signal_with_commentary(_md()))
    assert sig is not None and sig.reasoning["fit_lane_admission"] == "allowlist"
    assert sig.reasoning["fit_verdict"] == ""
    rec = _ledger(tmp_path)[-1]
    assert rec["verdict"] == "" and rec["score"] is None and rec["symbol"] == "NVDA"


def test_no_row_and_no_allowlist_still_skips_early(monkeypatch, tmp_path, caplog):
    lane = _lane(monkeypatch, tmp_path, {}, allowlist="")
    with caplog.at_level("INFO", logger="TradingBot"):
        assert asyncio.run(lane.generate_signal_with_commentary(_md())) is None
    assert "reason=fit_no_row" in caplog.text
    assert _ledger(tmp_path) == []


def test_regime_source_failure_degrades_to_unknown(monkeypatch, tmp_path):
    """The regime read must never take the lane down: a raising reader means `unknown`, which
    the admission rule treats as `mixed`."""
    def boom(symbol):
        raise RuntimeError("cache unreachable")

    lane = _lane(monkeypatch, tmp_path, {("NVDA", "toy", "long"): _Row("WATCH")},
                 allowlist="toy:long:mixed")
    lane._regime_source = boom
    sig = asyncio.run(lane.generate_signal_with_commentary(_md()))
    assert sig is not None and sig.reasoning["market_context_regime"] == "unknown"


def test_default_regime_source_reads_market_context(monkeypatch, tmp_path):
    """Wiring check: with no injected reader the lane reads core.market_context."""
    import strategies.fit_lane as F
    name = _toy_family(monkeypatch)
    fit = _Fit({("NVDA", name, "long"): _Row("WATCH")}, {})
    lane = F.FitLaneStrategy(_Commentary(), name, fit, lambda sym: (_bars(), pd.DataFrame()),
                             lambda: (name,), ledger_dir=tmp_path,
                             clock=lambda: datetime(2025, 9, 24, 13, 32),
                             allowlist=lambda: "toy:long:risk_on")
    monkeypatch.setattr(F, "read_live_regime", lambda symbol: "risk_on")
    sig = asyncio.run(lane.generate_signal_with_commentary(_md()))
    assert sig is not None and sig.reasoning["market_context_regime"] == "risk_on"


def test_read_live_regime_never_raises():
    from strategies.fit_lane import read_live_regime
    assert isinstance(read_live_regime("NVDA"), str)


def test_fit_lane_files_stay_inside_the_line_budget():
    import pathlib
    for path in ("strategies/fit_lane.py", "strategies/fit_lane_frames.py",
                 "strategies/fit_lane_admission.py"):
        assert len(pathlib.Path(path).read_text().splitlines()) < 400, path


# Task 2 review, Important 1: with two admitted sides the first entry keeps the seat unless a later
# one has a strictly better score; an unmeasured (row=None) side must not be displaced by a worse row.
def test_best_of_is_not_order_dependent_when_an_admitted_side_has_no_row():
    from types import SimpleNamespace
    import pandas as pd
    from strategies.fit_lane import FitLaneStrategy
    from strategies.families.models import Entry

    ts = pd.Timestamp("2026-01-05 09:35")
    long_e = Entry("AAA", "long", ts, 100.0, 97.0, 106.0, 5, "multiday", "t")
    short_e = Entry("AAA", "short", ts, 100.0, 103.0, 94.0, 5, "multiday", "t")
    bad = SimpleNamespace(score=-5.0)
    best_entry, best_row = FitLaneStrategy._best_of([long_e, short_e], {"long": None, "short": bad})
    assert best_entry is long_e and best_row is None
    best_entry, best_row = FitLaneStrategy._best_of([short_e, long_e], {"long": None, "short": bad})
    assert best_entry is short_e and best_row is bad
