"""v-turn-engine-2026-10-05: archive minute bins (tick rule) and the 5-minute mf_* features."""
import numpy as np
import pandas as pd

from research.strategy_hunt.data.micro import (MF_COLUMNS, features_5m, join_micro, minute_bins_from_ticks,
                                                tag_aggressor_tick_rule)


def test_tick_rule_vectorised_matches_scalar_rule():
    from core.micro_book import aggressor_tick_rule
    prices = pd.Series([100.0, 100.0, 100.1, 100.1, 100.0, 100.0, 100.2])
    got = tag_aggressor_tick_rule(prices).tolist()
    prev, side, want = None, 0, []
    for p in prices:
        side = aggressor_tick_rule(p, prev, side)
        want.append(side)
        prev = p
    assert got == want == [0, 0, 1, 1, -1, -1, 1]


def _ticks(day="2025-02-03", minutes=70, seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    px = 100.0
    t0 = pd.Timestamp(f"{day} 09:30")
    for i in range(minutes * 20):
        px = round(px + rng.choice([-0.01, 0.0, 0.01]), 2)
        rows.append({"ts": t0 + pd.Timedelta(seconds=3 * i), "price": px, "size": float(rng.choice([1, 50, 100, 900]))})
    return pd.DataFrame(rows)


def test_minute_bins_and_features_shapes_and_identities():
    ticks = _ticks()
    bins = minute_bins_from_ticks(ticks)
    assert len(bins) == 70
    assert (bins["buy_vol"] + bins["sell_vol"] <= bins["vol"] + 1e-9).all()      # side-0 prints count in vol only
    assert (bins["large_vol"] <= bins["vol"]).all() and bins["aggressor_rule"].eq("tick").all()
    assert (bins["vwap"].between(bins["low"], bins["high"])).all()
    feats = features_5m(bins)
    assert list(feats.columns) == list(MF_COLUMNS)
    assert feats.index.min() == pd.Timestamp("2025-02-03 09:30") and (feats.index.minute % 5 == 0).all()
    # trailing 15-minute sum at the 09:45 bar close (09:49) equals the raw sum of minutes 09:35..09:49
    row = feats.loc[pd.Timestamp("2025-02-03 09:45")]
    raw = bins[(bins.t_open >= "2025-02-03 09:35") & (bins.t_open <= "2025-02-03 09:49")]
    assert row["mf_buy_vol_15"] == raw["buy_vol"].sum() and row["mf_vol_15"] == raw["vol"].sum()
    assert row["mf_cvd_15"] == raw["buy_vol"].sum() - raw["sell_vol"].sum()
    assert 0.0 <= row["mf_sell_share_15"] <= 1.0 and row["mf_aggressor_rule"] == "tick"
    late = feats.loc[pd.Timestamp("2025-02-03 10:35")]
    raw60 = bins[(bins.t_open >= "2025-02-03 09:40") & (bins.t_open <= "2025-02-03 10:39")]
    assert late["mf_cvd_60"] == raw60["buy_vol"].sum() - raw60["sell_vol"].sum()


def test_join_micro_fills_missing_bars_and_empty_feats():
    idx = pd.date_range("2025-02-03 09:30", periods=20, freq="5min", name="ts")
    ind = pd.DataFrame({"d_rsi14": 30.0}, index=idx)
    feats = features_5m(minute_bins_from_ticks(_ticks(minutes=30)))
    j = join_micro(ind, feats)
    assert set(MF_COLUMNS) <= set(j.columns) and len(j) == 20
    assert j["mf_aggressor_rule"].iloc[0] == "tick" and j["mf_aggressor_rule"].iloc[-1] == ""
    assert np.isnan(j["mf_cvd_15"].iloc[-1])
    j2 = join_micro(ind, pd.DataFrame())
    assert j2["mf_cvd_15"].isna().all() and (j2["mf_aggressor_rule"] == "").all()


def test_empty_inputs():
    assert minute_bins_from_ticks(pd.DataFrame(columns=["ts", "price", "size"])).empty
    assert features_5m(pd.DataFrame()).empty


def test_archive_wall_clock_undoes_the_collectors_fixed_offset():
    from research.strategy_hunt.data.micro import archive_session_window, archive_wall_clock
    # a print stored as 10:30 EDT on an October day was really the 09:30 open
    s = pd.Series([pd.Timestamp("2024-10-02 10:30:00-04:00"), pd.Timestamp("2025-02-03 09:30:00-05:00")])
    wall = archive_wall_clock(s)
    assert wall.dt.strftime("%H:%M").tolist() == ["09:30", "09:30"]
    d0, d1 = archive_session_window("2024-10-02")
    assert d0.tz_convert("America/New_York").strftime("%H:%M") == "10:30" and (d1 - d0) == pd.Timedelta(hours=6, minutes=30)


def test_load_minute_bins_skips_empty_placeholder_days(tmp_path):
    from research.strategy_hunt.data.micro import load_minute_bins
    d = tmp_path / "MU"; d.mkdir()
    pd.DataFrame().to_parquet(d / "2025-01-01.parquet", index=False)
    minute_bins_from_ticks(_ticks(minutes=5)).to_parquet(d / "2025-02-03.parquet", index=False)
    out = load_minute_bins(tmp_path, "MU")
    assert len(out) == 5 and "t_open" in out.columns
    assert load_minute_bins(tmp_path, "NOPE").empty
