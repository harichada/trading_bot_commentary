"""Entry-time regime labels: thresholds, the known-at-entry contract, and the lagged VIX proxy."""
import numpy as np
import pandas as pd
import pytest

SESSION_BARS = 78


def _spy_session(day: str, closes: list[float], open_px: float = 100.0) -> pd.DataFrame:
    """A 5-minute SPY session whose bar closes are `closes` (bar 0 opens at `open_px`)."""
    idx = pd.date_range(f"{day} 09:30", periods=len(closes), freq="5min", name="ts")
    c = np.asarray(closes, dtype=float)
    o = np.r_[open_px, c[:-1]]
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) + 0.1, "low": np.minimum(o, c) - 0.1,
                         "close": c, "volume": 1000.0}, index=idx)


def _ramp_then_fade(day: str) -> pd.DataFrame:
    """+0.5% by 10:00 (bar 5's close), −1.0% at the session close."""
    up = np.linspace(100.0, 100.5, 6)                       # bars 0..5 -> 09:55 closes at 10:00
    down = np.linspace(100.5, 99.0, SESSION_BARS - 6)       # bars 6..77
    return _spy_session(day, list(up) + list(down))


def _vix(dates: list[str], closes: list[float]) -> pd.DataFrame:
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in dates], name="date")
    return pd.DataFrame({"close": closes}, index=idx)


# ──────────────────────
# thresholds
# ──────────────────────

def test_classify_regime_matches_live_thresholds():
    from research.strategy_hunt.data.regime import classify_regime
    assert classify_regime(0.5, 20.0, -2.0) == "risk_on"
    assert classify_regime(0.5, 15.0, 1.0) == "risk_on"        # VIX low
    assert classify_regime(-0.8, 25.0, 6.0) == "risk_off"
    assert classify_regime(0.1, 25.0, 12.0) == "risk_off"      # VIX spike alone
    assert classify_regime(0.5, 20.0, 4.0) == "mixed"          # nervous rally
    assert classify_regime(0.0, 20.0, 0.0) == "mixed"


def test_classify_regime_is_parity_with_core_market_context():
    """The research label and the live gate's label must be the same function of the same
    three inputs; only the *inputs* differ (research reconstructs them from history)."""
    from core.market_context import _classify_regime
    from research.strategy_hunt.data.regime import classify_regime
    for spy in (-2.0, -0.6, -0.5, -0.31, -0.3, 0.0, 0.3, 0.31, 1.5):
        for vix in (11.0, 15.9, 16.0, 22.0, 40.0):
            for chg in (-12.0, -0.1, 0.0, 2.9, 3.0, 5.0, 5.1, 10.0, 10.1, 30.0):
                assert classify_regime(spy, vix, chg) == _classify_regime(spy, vix, chg), (spy, vix, chg)


# ──────────────────────
# known at entry
# ──────────────────────

def test_label_uses_only_spy_bars_closed_before_the_entry_bar():
    """The day SPY was +0.5% at 10:00 and closed −1.0% is risk_on for a 10:00 entry.

    Under the old daily-close label this trade was stamped with the *close*'s regime,
    which the live gate could not have known when the order went in."""
    from research.strategy_hunt.data.regime import build_regime_lookup
    spy5 = _ramp_then_fade("2025-03-04")
    vix = _vix(["2025-02-28", "2025-03-03", "2025-03-04"], [20.4, 20.0, 31.0])
    lookup = build_regime_lookup(spy5, vix)
    label = lookup(pd.Timestamp("2025-03-04 10:00"))
    assert label != "risk_off"
    assert label == "risk_on"


def test_label_never_changes_when_bars_after_the_entry_change():
    from research.strategy_hunt.data.regime import build_regime_lookup
    vix = _vix(["2025-02-28", "2025-03-03", "2025-03-04"], [20.4, 20.0, 31.0])
    ts = pd.Timestamp("2025-03-04 10:00")
    base = _ramp_then_fade("2025-03-04")
    before = build_regime_lookup(base, vix)(ts)
    for factor in (0.5, 2.0, 10.0):
        crashed = base.copy()
        after_entry = crashed.index >= ts
        for c in ("open", "high", "low", "close"):
            crashed.loc[after_entry, c] = crashed.loc[after_entry, c] * factor
        assert build_regime_lookup(crashed, vix)(ts) == before


def test_first_bar_of_session_has_no_day_to_date_change():
    """Nothing has closed yet at the opening bar, so day-to-date SPY change is 0.0 —
    never the close's move."""
    from research.strategy_hunt.data.regime import build_regime_lookup
    spy5 = _ramp_then_fade("2025-03-04")
    vix = _vix(["2025-02-28", "2025-03-03", "2025-03-04"], [20.4, 20.0, 31.0])
    lookup = build_regime_lookup(spy5, vix)
    # spy_change 0.0, prior-session VIX 20.0 / −1.96% -> mixed (0.0 is not > 0.3)
    assert lookup(pd.Timestamp("2025-03-04 09:30")) == "mixed"


def test_vix_inputs_are_the_prior_sessions_close_and_change():
    """VIX is a *lagged* proxy: intraday VIX history is unavailable, so the label uses the
    prior session's close and the prior session's close-to-close change."""
    from research.strategy_hunt.data.regime import build_regime_lookup
    spy5 = _ramp_then_fade("2025-03-04")
    # VIX explodes +55% on the entry day itself; the prior session was +1%.
    vix = _vix(["2025-02-28", "2025-03-03", "2025-03-04"], [20.0, 20.2, 31.0])
    assert build_regime_lookup(spy5, vix)(pd.Timestamp("2025-03-04 10:00")) == "mixed"
    # the same spike one session earlier *is* visible at entry
    vix_prior = _vix(["2025-02-28", "2025-03-03", "2025-03-04"], [20.0, 31.0, 20.2])
    assert build_regime_lookup(spy5, vix_prior)(pd.Timestamp("2025-03-04 10:00")) == "risk_off"


def test_missing_spy_session_is_unknown():
    from research.strategy_hunt.data.regime import build_regime_lookup
    spy5 = _ramp_then_fade("2025-03-04")
    vix = _vix(["2025-02-28", "2025-03-03", "2025-03-04", "2025-03-05"], [20.4, 20.0, 31.0, 30.0])
    lookup = build_regime_lookup(spy5, vix)
    assert lookup(pd.Timestamp("2025-03-05 10:00")) == "unknown"      # no SPY bars that session
    assert lookup(pd.Timestamp("2030-01-02 10:00")) == "unknown"


def test_missing_prior_vix_is_unknown():
    from research.strategy_hunt.data.regime import build_regime_lookup
    spy5 = _ramp_then_fade("2025-03-04")
    # only one prior VIX close: no close-to-close change is computable for it
    assert build_regime_lookup(spy5, _vix(["2025-03-03"], [20.0]))(pd.Timestamp("2025-03-04 10:00")) == "unknown"
    assert build_regime_lookup(spy5, _vix([], []))(pd.Timestamp("2025-03-04 10:00")) == "unknown"


def test_empty_spy_lookup_is_all_unknown():
    from research.strategy_hunt.data.regime import build_regime_lookup
    empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"],
                         index=pd.DatetimeIndex([], name="ts"))
    lookup = build_regime_lookup(empty, _vix(["2025-03-03", "2025-03-04"], [20.0, 20.2]))
    assert lookup(pd.Timestamp("2025-03-04 10:00")) == "unknown"


def test_labels_are_per_session_and_use_each_sessions_own_open():
    from research.strategy_hunt.data.regime import build_regime_lookup
    d1 = _ramp_then_fade("2025-03-04")
    # next session opens at 99.0 and rallies 1% by 10:00
    up = np.linspace(99.0, 99.99, 6)
    d2 = _spy_session("2025-03-05", list(up) + list(np.linspace(99.99, 99.0, SESSION_BARS - 6)), open_px=99.0)
    spy5 = pd.concat([d1, d2])
    vix = _vix(["2025-02-28", "2025-03-03", "2025-03-04", "2025-03-05"], [20.4, 20.0, 19.0, 18.0])
    lookup = build_regime_lookup(spy5, vix)
    assert lookup(pd.Timestamp("2025-03-04 10:00")) == "risk_on"
    assert lookup(pd.Timestamp("2025-03-05 10:00")) == "risk_on"     # +1% off *its own* open


def test_lookup_is_callable_and_cached_stable():
    from research.strategy_hunt.data.regime import build_regime_lookup
    spy5 = _ramp_then_fade("2025-03-04")
    vix = _vix(["2025-02-28", "2025-03-03", "2025-03-04"], [20.4, 20.0, 31.0])
    lookup = build_regime_lookup(spy5, vix)
    ts = pd.Timestamp("2025-03-04 10:00")
    assert lookup(ts) == lookup(ts) == lookup(ts.to_pydatetime())


def test_daily_close_regime_helpers_are_gone():
    """The look-ahead label must not survive as an importable alternative."""
    import research.strategy_hunt.data.regime as mod
    assert not hasattr(mod, "regime_series")
    assert not hasattr(mod, "regime_at")


@pytest.mark.parametrize("bad", [None, "not-a-frame"])
def test_build_regime_lookup_rejects_non_frames(bad):
    from research.strategy_hunt.data.regime import build_regime_lookup
    with pytest.raises((TypeError, AttributeError)):
        build_regime_lookup(bad, bad)
