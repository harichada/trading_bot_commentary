"""Live adapter for rsi_turn (v-turn-engine-2026-10-05): micro snapshot →
mf_* columns on the last bar, verdict side channel → cache + ledger, swing
exit thesis_failed, engine hook."""
from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd

from core.micro_book import MicroBook
from core.turn_verdicts import TurnVerdict, TurnVerdicts
from strategies.fit_lane_turn import MICRO_FIELDS, attach_micro, record_turn_verdict, verdict_from_attrs


def _snap(age=5.0, with_quote=True):
    return SimpleNamespace(age_s=age, buy_vol_15=300.0, sell_vol_15=700.0, vol_15=1000.0, cvd_15=-400.0, cvd_60=-900.0,
                           vol_60=4000.0, sell_share_15=0.7, large_buy_15=50.0, large_sell_15=10.0, large_vol_15=60.0,
                           n_updates_15=40, imb_ema=0.4, spread_ema=0.0005,
                           bid=100.0 if with_quote else None, ask=100.05 if with_quote else None,
                           bid_size=500.0 if with_quote else 0.0, ask_size=200.0 if with_quote else 0.0)


def _ind(n=5):
    idx = pd.date_range("2026-10-06 09:30", periods=n, freq="5min", name="ts")
    return pd.DataFrame({"d_rsi14": 20.0}, index=idx)


def test_attach_micro_fills_only_the_last_bar():
    out = attach_micro(_ind(), _snap())
    assert set(MICRO_FIELDS) <= set(out.columns)
    assert np.isnan(out["mf_cvd_15"].iloc[0]) and out["mf_cvd_15"].iloc[-1] == -400.0
    assert out["mf_imb_ema"].iloc[-1] == 0.4 and out["mf_aggressor_rule"].iloc[-1] == "quote_conflated"
    assert out["mf_spread_baseline"].iloc[-1] == out["mf_spread_ema"].iloc[-1]       # uncalibrated: z = 0


def test_attach_micro_ignores_stale_or_missing_snapshots_and_quoteless_books():
    out = attach_micro(_ind(), _snap(age=1000.0))
    assert np.isnan(out["mf_cvd_15"].iloc[-1]) and (out["mf_aggressor_rule"] == "").all()
    out = attach_micro(_ind(), None)
    assert np.isnan(out["mf_vol_15"].iloc[-1])
    out = attach_micro(_ind(), _snap(with_quote=False))
    assert out["mf_cvd_15"].iloc[-1] == -400.0 and np.isnan(out["mf_imb_ema"].iloc[-1])    # tape yes, book no


def test_verdict_side_channel_and_cache():
    bars = pd.DataFrame({"close": [1.0, 2.0]}, index=pd.date_range("2026-10-06 09:30", periods=2, freq="5min"))
    assert verdict_from_attrs(bars, bars.index[-1]) is None
    bars.attrs["rsi_turn_verdicts"] = {bars.index[-1]: {"score": np.float64(42.0), "theta": 60, "vetoes": [], "cvd_60": -5.0, "Z": 0.5}}
    v = verdict_from_attrs(bars, bars.index[-1])
    assert v["score"] == 42.0 and isinstance(v["score"], float) and v["Z"] == 0.5
    TurnVerdicts.instance().clear()
    record_turn_verdict("mu", bars.index[-1], v)
    latest = TurnVerdicts.instance().latest("MU")
    assert latest is not None and latest.score == 42.0 and latest.theta == 60.0 and latest.cvd_60 == -5.0


def test_should_thesis_fail_rules():
    from core.swing_exits import should_thesis_fail
    now = datetime(2026, 10, 7, 11, 0)
    entry = datetime(2026, 10, 6, 10, 0)
    reasoning = {"fit_family": "rsi_turn", "fit_params": {"theta": 60}, "hold_kind": "multiday"}
    bad = TurnVerdict("MU", now - timedelta(minutes=3), score=30.0, theta=60.0, cvd_60=-100.0, vetoes=())
    hit, why = should_thesis_fail(reasoning, "MU", entry, now, bad)
    assert hit and why.startswith("thesis_failed")
    ok = TurnVerdict("MU", now - timedelta(minutes=3), score=55.0, theta=60.0, cvd_60=-100.0, vetoes=())
    assert should_thesis_fail(reasoning, "MU", entry, now, ok) == (False, "")
    stale = TurnVerdict("MU", now - timedelta(hours=2), score=10.0, theta=60.0, cvd_60=-100.0, vetoes=())
    assert should_thesis_fail(reasoning, "MU", entry, now, stale) == (False, "")
    assert should_thesis_fail({"fit_family": "mr_20d"}, "MU", entry, now, bad) == (False, "")
    assert should_thesis_fail(reasoning, "MU", entry, now, None) == (False, "")
    old_entry = datetime(2026, 9, 25, 10, 0)
    assert should_thesis_fail(reasoning, "MU", old_entry, now, bad) == (False, "")


def test_engine_and_fit_lane_wiring_present():
    import inspect
    import core.engine as eng
    import strategies.fit_lane as fl
    src = inspect.getsource(eng)
    assert "should_thesis_fail(" in src and "TurnVerdicts.instance().latest(position.symbol)" in src
    lane_src = inspect.getsource(fl.FitLaneStrategy._generate)
    assert 'if family == "rsi_turn":' in lane_src and "attach_micro(ind, self._micro_provider(symbol))" in lane_src
    assert "record_turn_verdict(symbol, now_ts, thesis)" in lane_src


def test_thesis_fail_needs_a_verdict_after_the_fill_and_a_minimum_hold():
    from core.swing_exits import should_thesis_fail
    reasoning = {"fit_family": "rsi_turn", "fit_params": {"theta": 60}, "hold_kind": "multiday"}
    entry = datetime(2026, 10, 6, 10, 0)
    before_fill = TurnVerdict("MU", datetime(2026, 10, 6, 9, 55), 20.0, 60.0, -1.0, ())
    assert should_thesis_fail(reasoning, "MU", entry, datetime(2026, 10, 6, 10, 5), before_fill) == (False, "")
    fresh = TurnVerdict("MU", datetime(2026, 10, 6, 10, 10), 20.0, 60.0, -1.0, ())
    assert should_thesis_fail(reasoning, "MU", entry, datetime(2026, 10, 6, 10, 12), fresh) == (False, "")   # < 15 min held
    assert should_thesis_fail(reasoning, "MU", entry, datetime(2026, 10, 6, 10, 20), fresh)[0] is True


def test_family_keeps_only_the_current_bars_verdict():
    import pandas as pd
    from strategies.families.rsi_turn import rsi_turn_entries
    from tests.strategies.families.test_rsi_turn import _bars5, _daily, _ind
    daily = _daily()
    days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range(daily.index[-1] + pd.Timedelta(days=1), periods=2)]
    bars = _bars5(days)
    rsi_turn_entries(bars, _ind(bars), daily, None, {"theta": 30, "stop_atr": 1.5, "time_stop_days": 5})
    v = bars.attrs.get("rsi_turn_verdicts", {})
    assert len(v) == 1 and "has_quotes" in next(iter(v.values()))


def test_shadow_norow_family_is_evaluated_and_ledgered(monkeypatch, tmp_path, caplog):
    """A shadow-only family with no fit row (a brand-new family) must still be
    evaluated and write to its ledger; it must never produce a live signal."""
    import asyncio, json
    import pandas as pd
    from tests.strategies.test_fit_lane_live_admission import _Commentary, _Fit, _bars, _md, _toy_family
    from strategies.fit_lane import FitLaneStrategy
    name = _toy_family(monkeypatch)
    lane = FitLaneStrategy(_Commentary(), name, _Fit({}, {}), lambda sym: (_bars(), pd.DataFrame()),
                           lambda: (), ledger_dir=tmp_path, clock=lambda: datetime(2025, 9, 24, 13, 32),
                           allowlist=lambda: "", regime_source=lambda sym: "mixed",
                           shadow_norow=lambda: (name,))
    with caplog.at_level("INFO", logger="TradingBot"):
        assert asyncio.run(lane.generate_signal_with_commentary(_md())) is None
    assert "fit_no_row" not in caplog.text
    rows = [json.loads(l) for l in (tmp_path / f"shadow_fit_{name}.ndjson").read_text().splitlines()]
    assert rows and rows[0]["symbol"] == "NVDA"
    # the same lane without the shadow allowance keeps today's skip
    lane2 = FitLaneStrategy(_Commentary(), name, _Fit({}, {}), lambda sym: (_bars(), pd.DataFrame()),
                            lambda: (), ledger_dir=tmp_path / "b", clock=lambda: datetime(2025, 9, 24, 13, 32),
                            allowlist=lambda: "", regime_source=lambda sym: "mixed", shadow_norow=lambda: ())
    caplog.clear()
    with caplog.at_level("INFO", logger="TradingBot"):
        asyncio.run(lane2.generate_signal_with_commentary(_md()))
    assert "fit_no_row" in caplog.text


def test_config_shadow_norow_default_and_env(monkeypatch):
    from core.config import Config, config_manager
    monkeypatch.delenv("FIT_LANE_SHADOW_NOROW_FAMILIES", raising=False)
    monkeypatch.delitem(config_manager.config.setdefault("trading", {}), "fit_lane_shadow_norow_families", raising=False)
    assert Config().FIT_LANE_SHADOW_NOROW_FAMILIES == ("rsi_turn",)
    monkeypatch.setenv("FIT_LANE_SHADOW_NOROW_FAMILIES", "rsi_turn, rsi_dip")
    assert Config().FIT_LANE_SHADOW_NOROW_FAMILIES == ("rsi_turn", "rsi_dip")
