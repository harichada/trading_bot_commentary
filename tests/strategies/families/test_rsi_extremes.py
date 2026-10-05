"""rsi_dip / rsi_fade (v-rsi-extremes-2026-10-05): session-open evaluation of
yesterday's RSI(14), ATR stop, 20-day-mean target, 1R minimum, spacing."""
import numpy as np
import pandas as pd

from tests.strategies.families.test_swing import _bars, _ind, DAYS


def _ind_rsi(bars, rsi, close=100.0, sma200=90.0, bb_mid=108.0, atr=2.0):
    ind = _ind(bars)
    ind["d_rsi14"] = rsi
    ind["d_close"] = close
    ind["d_sma200"] = sma200
    ind["d_bb_mid"] = bb_mid
    ind["d_atr14"] = atr
    return ind


def test_rsi_dip_enters_below_threshold_with_stop_target_and_spacing():
    from strategies.families.rsi_extremes import rsi_dip_entries
    bars = _bars(DAYS[:12])
    ind = _ind_rsi(bars, rsi=22.0)
    params = {"rsi_low": 25, "trend_filter": "none", "stop_atr": 1.5, "time_stop_days": 5}
    out = rsi_dip_entries(bars, ind, None, None, params)
    assert out, "RSI 22 < 25 must propose a long"
    e = out[0]
    assert e.side == "long" and e.tag == "rsi_dip"
    assert e.stop_px == 100.0 - 3.0 and e.target_px == 108.0      # 1.5 x ATR 2.0; target = 20d mean
    # spaced by the hold: at most one entry per 5 sessions
    assert len(out) <= 12 // 5 + 1
    # RSI above the threshold → nothing
    assert rsi_dip_entries(bars, _ind_rsi(bars, rsi=28.0), None, None, params) == []


def test_rsi_dip_requires_1r_to_the_mean_and_respects_trend_filter():
    from strategies.families.rsi_extremes import rsi_dip_entries
    bars = _bars(DAYS[:6])
    base = {"rsi_low": 25, "stop_atr": 2.5, "time_stop_days": 5}
    # mean only 2.0 above the open, risk is 5.0 → skipped
    assert rsi_dip_entries(bars, _ind_rsi(bars, rsi=20.0, bb_mid=102.0), None, None,
                           {**base, "trend_filter": "none"}) == []
    # trend filter: close below the 200d SMA blocks; above allows
    below = _ind_rsi(bars, rsi=20.0, sma200=120.0, bb_mid=110.0)
    above = _ind_rsi(bars, rsi=20.0, sma200=90.0, bb_mid=110.0)
    assert rsi_dip_entries(bars, below, None, None, {**base, "trend_filter": "sma200"}) == []
    assert rsi_dip_entries(bars, above, None, None, {**base, "trend_filter": "sma200"})


def test_rsi_fade_shorts_above_threshold_with_mirrored_levels():
    from strategies.families.rsi_extremes import rsi_fade_entries
    bars = _bars(DAYS[:6])
    ind = _ind_rsi(bars, rsi=82.0, bb_mid=92.0)
    out = rsi_fade_entries(bars, ind, None, None, {"rsi_high": 80, "stop_atr": 1.5, "time_stop_days": 5})
    assert out and out[0].side == "short" and out[0].tag == "rsi_fade"
    assert out[0].stop_px == 103.0 and out[0].target_px == 92.0
    assert rsi_fade_entries(bars, _ind_rsi(bars, rsi=70.0, bb_mid=92.0), None, None,
                            {"rsi_high": 80, "stop_atr": 1.5, "time_stop_days": 5}) == []


def test_families_registered_within_grid_cap():
    from strategies.families.common import grid_combos
    from strategies.families.registry import get_family, family_names
    for name in ("rsi_dip", "rsi_fade"):
        fam = get_family(name)
        assert name in family_names()
        assert len(grid_combos(fam.grid)) <= 27
    assert get_family("rsi_dip").sides == ("long",) and get_family("rsi_fade").sides == ("short",)
