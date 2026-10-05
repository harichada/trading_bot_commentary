"""rsi_turn family: zone gate from the symbol's own RSI history, score from the
four components, entry levels, vetoes, graceful empties."""
import numpy as np
import pandas as pd

from strategies.families.rsi_turn import bars_15m, rsi_turn_entries, score_bar
from core import turn_score as ts


def _daily(n=300, crash_days=12):
    """A year of drifting closes ending in a straight-down run so the lagged
    daily RSI sits far below the trailing 5th percentile on the last day."""
    rng = np.random.default_rng(1)
    closes = 100 + np.cumsum(rng.normal(0.05, 0.8, n))
    closes[-crash_days:] = closes[-crash_days - 1] * np.linspace(1.0, 0.80, crash_days)
    idx = pd.bdate_range("2025-01-02", periods=n, name="date")
    return pd.DataFrame({"open": closes, "high": closes + 0.5, "low": closes - 0.5, "close": closes, "volume": 1e6}, index=idx)


def _bars5(days, open_px=80.0):
    frames = []
    for d in days:
        idx = pd.date_range(f"{d} 09:30", periods=78, freq="5min", name="ts")
        # a V inside the day: down for 30 bars, then up — the 15-min RSI turns up
        path = np.concatenate([np.linspace(0, -3, 30), np.linspace(-3, 1, 48)])
        c = open_px + path
        frames.append(pd.DataFrame({"open": c, "high": c + 0.1, "low": c - 0.1, "close": c, "volume": 1000}, index=idx))
    return pd.concat(frames)


def _ind(bars, absorbing=True):
    ind = pd.DataFrame(index=bars.index)
    ind["d_rsi14"] = 12.0          # lagged daily RSI, deep in the zone
    ind["d_atr14"] = 2.0
    ind["d_bb_mid"] = 95.0         # 20-day mean well above price
    ind["mf_buy_vol_15"] = 300.0 if absorbing else 800.0
    ind["mf_sell_vol_15"] = 700.0 if absorbing else 200.0
    ind["mf_vol_15"] = 1000.0
    ind["mf_cvd_15"] = 50.0 if absorbing else -600.0
    ind["mf_cvd_60"] = -900.0
    ind["mf_large_buy_15"] = 100.0
    ind["mf_large_sell_15"] = 20.0
    ind["mf_large_vol_15"] = 120.0
    ind["mf_aggressor_rule"] = "tick"
    return ind


def test_bars_15m_builds_rsi_and_atr():
    b15 = bars_15m(_bars5(["2026-01-05"]))
    assert len(b15) == 26 and b15["rsi"].notna().sum() > 5 and (b15["atr_frac"].dropna() > 0).all()


def test_score_bar_components_and_zone_gate():
    zone = ts.ZoneStats(q05=24.0, q25=40.0, q50=50.0, q75=60.0, n=252)
    rsi15 = np.array([30, 28, 26, 24, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 13, 15, 17, 19, 21, 23, 25], float)
    lows = np.linspace(90, 80, len(rsi15))
    lows[-6:] = [79, 79.2, 79.4, 79.6, 79.8, 80]
    row = _ind(_bars5(["2026-01-05"])).iloc[0].to_dict()
    in_zone, score, detail, vetoes = score_bar(row, zone, rsi15, lows, 4.0, 0.0, 0.01, None)
    assert in_zone and vetoes == () and 0 < score <= 100
    assert detail["Z"] > 0.3 and detail["M"] > 0 and detail["A"] > 0 and detail["B"] == 0.0   # archive: no quotes
    out_of_zone = dict(row, d_rsi14=40.0)
    assert score_bar(out_of_zone, zone, rsi15, lows, 4.0, 0.0, 0.01, None)[0] is False


def test_family_enters_on_a_turn_and_respects_theta_and_1r():
    daily = _daily()
    days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range(daily.index[-1] + pd.Timedelta(days=1), periods=3)]
    bars = _bars5(days)
    ind = _ind(bars)
    params = {"theta": 30, "stop_atr": 1.5, "time_stop_days": 5}
    out = rsi_turn_entries(bars, ind, daily, None, params)
    assert out, "deep zone + turning 15-min RSI + absorbing tape must produce an entry"
    e = out[0]
    assert e.side == "long" and e.tag == "rsi_turn" and e.target_px == 95.0 and e.stop_px == e.entry_px - 3.0 - (e.entry_px - bars["close"].iloc[0]) or e.stop_px < e.entry_px
    assert len(out) <= len(days)            # spaced by the hold
    # an impossible threshold → nothing
    assert rsi_turn_entries(bars, ind, daily, None, {**params, "theta": 101}) == []
    # mean too close for 1R → nothing
    ind2 = ind.copy(); ind2["d_bb_mid"] = 79.0
    assert rsi_turn_entries(bars, ind2, daily, None, params) == []


def test_family_handles_missing_inputs():
    bars = _bars5(["2026-01-05"])
    assert rsi_turn_entries(bars, _ind(bars), None, None, {"theta": 60}) == []
    assert rsi_turn_entries(bars.iloc[:0], _ind(bars).iloc[:0], _daily(), None, {"theta": 60}) == []
    short = _daily(n=40)
    assert rsi_turn_entries(bars, _ind(bars), short, None, {"theta": 10}) == []    # < 60 sessions of RSI history
