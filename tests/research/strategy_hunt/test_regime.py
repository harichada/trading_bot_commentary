import pandas as pd


def test_classify_regime_matches_live_thresholds():
    from research.strategy_hunt.data.regime import classify_regime
    assert classify_regime(0.5, 20.0, -2.0) == "risk_on"
    assert classify_regime(0.5, 15.0, 1.0) == "risk_on"        # VIX low
    assert classify_regime(-0.8, 25.0, 6.0) == "risk_off"
    assert classify_regime(0.1, 25.0, 12.0) == "risk_off"      # VIX spike alone
    assert classify_regime(0.5, 20.0, 4.0) == "mixed"          # nervous rally
    assert classify_regime(0.0, 20.0, 0.0) == "mixed"


def test_regime_series_and_lookup():
    from research.strategy_hunt.data.regime import regime_series, regime_at
    idx = pd.bdate_range("2025-01-02", periods=3, name="date")
    spy = pd.DataFrame({"close": [100.0, 101.0, 100.0]}, index=idx)
    vix = pd.DataFrame({"close": [20.0, 19.0, 22.0]}, index=idx)
    s = regime_series(spy, vix)
    assert s.iloc[0] == "unknown"                 # no prior close
    assert s.iloc[1] == "risk_on"                 # +1% SPY, VIX -5%
    assert s.iloc[2] == "risk_off"                # -0.99% SPY, VIX +15.8%
    assert regime_at(s, pd.Timestamp("2025-01-03 10:15")) == "risk_on"
    assert regime_at(s, pd.Timestamp("2030-01-01")) == "unknown"
