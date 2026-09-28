import os
import pytest
from core import settings_console as sc

def test_registry_keys_unique_and_complete():
    keys = [k.key for k in sc.KNOBS]
    assert len(keys) == len(set(keys))
    for expected in ("bot_capital_allocation", "risk_per_trade_pct", "max_daily_loss", "max_positions",
                     "swing_max_hold_sessions", "orb_live_entries_enabled", "day_trade_live_entries_enabled",
                     "day_trade_short_live_entries_enabled", "mean_rev_live_entries_enabled",
                     "fit_lane_live_entries_enabled", "fit_lane_live_allowlist", "fit_lanes_enabled",
                     "enable_playbook_gate", "enable_desk_limits"):
        assert expected in sc.KNOB_BY_KEY
    assert sc.KNOB_BY_KEY["fit_lanes_enabled"].restart_required is True
    assert "desk_streak_since" not in sc.KNOB_BY_KEY

@pytest.mark.parametrize("raw", ["0", "1.5", "-0.1", "abc", ""])
def test_fraction_rejects_out_of_range(raw):
    with pytest.raises(ValueError):
        sc.coerce(sc.KNOB_BY_KEY["risk_per_trade_pct"], raw)

def test_fraction_accepts_edge():
    assert sc.coerce(sc.KNOB_BY_KEY["risk_per_trade_pct"], "1") == 1.0
    assert sc.coerce(sc.KNOB_BY_KEY["max_daily_loss"], 0.03) == 0.03

def test_money_respects_ceiling():
    knob = sc.KNOB_BY_KEY["bot_capital_allocation"]
    assert sc.coerce(knob, "4500", ceiling=5000) == 4500.0
    with pytest.raises(ValueError):
        sc.coerce(knob, 5001, ceiling=5000)
    with pytest.raises(ValueError):
        sc.coerce(knob, 0, ceiling=5000)

def test_int_bounds():
    assert sc.coerce(sc.KNOB_BY_KEY["max_positions"], "5") == 5
    with pytest.raises(ValueError):
        sc.coerce(sc.KNOB_BY_KEY["max_positions"], 0)
    with pytest.raises(ValueError):
        sc.coerce(sc.KNOB_BY_KEY["max_positions"], "2.5")

@pytest.mark.parametrize("raw,expected", [("1", True), ("true", True), ("on", True), (False, False), ("0", False), ("no", False)])
def test_bool_coercion(raw, expected):
    assert sc.coerce(sc.KNOB_BY_KEY["orb_live_entries_enabled"], raw) is expected

def test_bool_rejects_garbage():
    with pytest.raises(ValueError):
        sc.coerce(sc.KNOB_BY_KEY["orb_live_entries_enabled"], "maybe")

def test_csv_families_known_only():
    knob = sc.KNOB_BY_KEY["fit_lane_live_entries_enabled"]
    assert sc.coerce(knob, "mr_20d, mom_12_1") == ("mr_20d", "mom_12_1")
    assert sc.coerce(knob, ["mr_20d"]) == ("mr_20d",)
    assert sc.coerce(knob, "") == ()
    with pytest.raises(ValueError):
        sc.coerce(knob, "mr_20d,bogus")

def test_allowlist_grammar():
    knob = sc.KNOB_BY_KEY["fit_lane_live_allowlist"]
    assert sc.coerce(knob, "mr_20d:long,mom_12_1:long:risk_on|mixed") == ("mr_20d:long", "mom_12_1:long:risk_on|mixed")
    for bad in ("mr_20d:sideways", "mom_12_1:long:bull", "bogus:long", "mr_20d:long:risk_on:extra"):
        with pytest.raises(ValueError):
            sc.coerce(knob, bad)

def test_classify_numeric():
    knob = sc.KNOB_BY_KEY["risk_per_trade_pct"]
    assert sc.classify(knob, 0.01, 0.005) == "tighten"
    assert sc.classify(knob, 0.01, 0.02) == "loosen"
    assert sc.classify(knob, 0.01, 0.01) == "noop"
    assert sc.classify(knob, None, 0.01) == "tighten"

def test_classify_bool_lane_and_gate():
    lane = sc.KNOB_BY_KEY["orb_live_entries_enabled"]
    assert sc.classify(lane, False, True) == "loosen"
    assert sc.classify(lane, True, False) == "tighten"
    gate = sc.KNOB_BY_KEY["enable_playbook_gate"]
    assert sc.classify(gate, True, False) == "loosen"
    assert sc.classify(gate, False, True) == "tighten"

def test_classify_csv():
    knob = sc.KNOB_BY_KEY["fit_lane_live_entries_enabled"]
    assert sc.classify(knob, ("mr_20d",), ("mr_20d", "mom_12_1")) == "loosen"
    assert sc.classify(knob, ("mr_20d", "mom_12_1"), ("mr_20d",)) == "tighten"
    assert sc.classify(knob, ("mr_20d", "mom_12_1"), ("mom_12_1", "mr_20d")) == "noop"
    assert sc.classify(knob, ("mr_20d",), ("mom_12_1",)) == "loosen"   # swap adds one

def test_env_shadowed(monkeypatch):
    knob = sc.KNOB_BY_KEY["risk_per_trade_pct"]
    monkeypatch.delenv("RISK_PER_TRADE_PCT", raising=False)
    assert sc.env_shadowed(knob) is False
    monkeypatch.setenv("RISK_PER_TRADE_PCT", "0.01")
    assert sc.env_shadowed(knob) is True
    assert sc.env_shadowed(sc.KNOB_BY_KEY["max_positions"]) is False   # no env name
