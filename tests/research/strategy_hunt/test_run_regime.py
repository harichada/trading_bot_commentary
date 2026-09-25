"""The run's regime wiring: one entry-time lookup, built from the real SPY tape, for both variants.

v-playbook-desk-b1-6 (final review, Critical 1). Kept out of `test_run.py` so that module
stays inside the test line budget.
"""
import numpy as np
import pandas as pd
import pytest

SESSION_BARS = 78


def _bars(start="2023-01-03", sessions=360, seed=0, drift=0.0):
    rng = np.random.default_rng(seed)
    frames, px = [], 100.0
    for d in pd.bdate_range(start, periods=sessions):
        idx = pd.date_range(f"{d.date()} 09:30", periods=SESSION_BARS, freq="5min", name="ts")
        close = px + rng.normal(drift, 0.05, SESSION_BARS).cumsum()
        frames.append(pd.DataFrame({"open": close - 0.02, "high": close + 0.3, "low": close - 0.3,
                                    "close": close, "volume": 1000.0}, index=idx))
        px = float(close[-1])
    return pd.concat(frames)


@pytest.fixture
def root(tmp_path):
    from research.strategy_hunt.data.bars import _write, bars5_path, daily_path, to_daily
    b = _bars()
    for sym in ("AAA", "SPY"):
        _write(bars5_path(tmp_path, sym), b, "ts")
        _write(daily_path(tmp_path, sym), to_daily(b), "date")
    vix_days = pd.DatetimeIndex(sorted(b.index.normalize().unique()), name="date")
    vix = pd.DataFrame({"open": 20.0, "high": 21.0, "low": 19.0,
                        "close": np.linspace(22.0, 18.0, len(vix_days)), "volume": 0.0}, index=vix_days)
    _write(daily_path(tmp_path, "VIX"), vix, "date")
    return tmp_path


def test_load_context_exposes_an_entry_time_regime_lookup(root):
    from research.strategy_hunt.data.regime import RegimeLookup
    from research.strategy_hunt.run import load_context
    ctx = load_context(root)
    assert isinstance(ctx["regime_lookup"], RegimeLookup)
    assert "regime" not in ctx                     # the daily-close series is gone
    assert not ctx["spy5"].empty
    # a mid-session entry on a session with prior VIX gets a real label
    ts = pd.Timestamp("2023-02-01 11:00")
    assert ctx["regime_lookup"](ts) in ("risk_on", "risk_off", "mixed")


def test_both_variants_share_one_lookup_built_from_the_real_spy_tape(root, monkeypatch):
    """The shuffle control permutes only the symbol's own bars; SPY -- and therefore every
    regime label -- must be identical between the two runs."""
    import research.strategy_hunt.run as run_mod
    from research.strategy_hunt.data.bars import load_bars5
    from strategies.families.common import Family, make_entry, one_per_session
    from strategies.families import registry

    def _toy(bars5, ind, daily, news, params):
        out = []
        for i in np.flatnonzero(ind["bar_of_day"].to_numpy() == 10):
            if i + 1 < len(bars5):
                px = float(bars5["open"].iloc[i + 1])
                e = make_entry(bars5, int(i), "long", px - 1.0, px + 2.0, 30, "intraday", "toy")
                if e:
                    out.append(e)
        return one_per_session(out)

    monkeypatch.setitem(registry.FAMILIES, "toy", Family("toy", _toy, {"w": [1.0]}, "intraday", ("long",)))
    seen: list[dict] = []
    real_run = run_mod.run_symbol_family

    def _spy(*a, **kw):
        seen.append({"lookup": kw.get("regime_lookup"), "spy5": kw.get("spy5")})
        return real_run(*a, **kw)

    monkeypatch.setattr(run_mod, "run_symbol_family", _spy)
    run_mod.process_symbol("AAA", root, ["toy"], force=True)
    run_mod.process_symbol("AAA", root, ["toy"], shuffle_seed=7, force=True, variant="shuffle")

    assert len(seen) == 2
    spy_on_disk = load_bars5(root, "SPY")
    stamps = [pd.Timestamp(f"{d.date()} 11:00") for d in spy_on_disk.index.normalize().unique()]
    assert [seen[0]["lookup"](t) for t in stamps] == [seen[1]["lookup"](t) for t in stamps]
    for rec in seen:
        pd.testing.assert_frame_equal(rec["spy5"], spy_on_disk)


def test_trades_are_stamped_with_the_entry_bars_regime(root):
    """End to end: every stamped regime equals the lookup's label for that trade's entry_ts."""
    import research.strategy_hunt.run as run_mod
    from research.strategy_hunt.paths import trades_path
    from strategies.families.common import Family, make_entry, one_per_session
    from strategies.families import registry

    def _toy(bars5, ind, daily, news, params):
        out = []
        for i in np.flatnonzero(ind["bar_of_day"].to_numpy() == 10):
            if i + 1 < len(bars5):
                px = float(bars5["open"].iloc[i + 1])
                e = make_entry(bars5, int(i), "long", px - 1.0, px + 2.0, 30, "intraday", "toy")
                if e:
                    out.append(e)
        return one_per_session(out)

    registry.FAMILIES["toy"] = Family("toy", _toy, {"w": [1.0]}, "intraday", ("long",))
    try:
        run_mod.process_symbol("AAA", root, ["toy"], force=True)
        df = pd.read_parquet(trades_path(root, "AAA"))
        lookup = run_mod.load_context(root)["regime_lookup"]
        assert not df.empty
        assert [lookup(pd.Timestamp(t)) for t in df["entry_ts"]] == list(df["regime"])
        assert set(df["regime"]) - {"unknown"}          # not every trade is unknown
    finally:
        registry.FAMILIES.pop("toy", None)
