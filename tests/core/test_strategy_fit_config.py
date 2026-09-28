def _cfg(monkeypatch, **env):
    for k in ("ENABLE_STRATEGY_FIT_GATE", "STRATEGY_FIT_SHADOW", "STRATEGY_FIT_PATH", "STRATEGY_FIT_MAX_AGE_DAYS",
              "FIT_LANES_ENABLED", "FIT_LANE_LIVE_ENTRIES_ENABLED", "SWING_ALLOW_OVERNIGHT", "SWING_MAX_HOLD_SESSIONS",
              "STRATEGY_HUNT_DATA_ROOT", "STRATEGY_FIT_REGIME_ROWS", "FIT_LANE_LIVE_ALLOWLIST",
              "FIT_LANE_ALLOWLIST_STRENGTH"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    # NOTE: `import core.config as cc` is unsafe here -- core/__init__.py
    # does `from core.config import ..., config, ...`, which rebinds the
    # `core.config` *package attribute* to the `config` Config instance
    # (shadowing the submodule of the same name). `import X.Y as Z` binds
    # via attribute access on X, so `cc` would end up being that Config
    # instance rather than the module, and `cc.Config()` would raise
    # AttributeError regardless of this file's own correctness. Importing
    # the class directly sidesteps that pre-existing package quirk.
    from core.config import Config
    return Config()


def test_defaults(monkeypatch):
    c = _cfg(monkeypatch)
    assert c.ENABLE_STRATEGY_FIT_GATE is False and c.STRATEGY_FIT_SHADOW is True
    assert c.STRATEGY_FIT_PATH.endswith("data/strategy_hunt/fit/strategy_fit.json") and c.STRATEGY_FIT_MAX_AGE_DAYS == 7
    assert c.FIT_LANES_ENABLED == () and c.FIT_LANE_LIVE_ENTRIES_ENABLED == ()
    assert c.SWING_ALLOW_OVERNIGHT is False and c.SWING_MAX_HOLD_SESSIONS == 10
    # v-strategy-fit-review2-2026-09-25 (Important 3)
    assert c.STRATEGY_HUNT_DATA_ROOT == "data/strategy_hunt"
    # v-playbook-desk-b1-2026-09-25: regime-conditioned rows are consulted by default;
    # they can only narrow the gate (a missing regime row falls back to the plain row).
    assert c.STRATEGY_FIT_REGIME_ROWS is True


def test_env_overrides(monkeypatch):
    c = _cfg(monkeypatch, ENABLE_STRATEGY_FIT_GATE="1", STRATEGY_FIT_SHADOW="0", STRATEGY_FIT_MAX_AGE_DAYS="3",
             FIT_LANES_ENABLED="orb, vwap_reversion", FIT_LANE_LIVE_ENTRIES_ENABLED="orb", SWING_ALLOW_OVERNIGHT="true",
             SWING_MAX_HOLD_SESSIONS="5", STRATEGY_FIT_PATH="/tmp/x.json",
             STRATEGY_HUNT_DATA_ROOT="/srv/hunt", STRATEGY_FIT_REGIME_ROWS="0")
    assert c.ENABLE_STRATEGY_FIT_GATE is True and c.STRATEGY_FIT_SHADOW is False and c.STRATEGY_FIT_MAX_AGE_DAYS == 3
    assert c.FIT_LANES_ENABLED == ("orb", "vwap_reversion") and c.FIT_LANE_LIVE_ENTRIES_ENABLED == ("orb",)
    assert c.SWING_ALLOW_OVERNIGHT is True and c.SWING_MAX_HOLD_SESSIONS == 5 and c.STRATEGY_FIT_PATH == "/tmp/x.json"
    assert c.STRATEGY_HUNT_DATA_ROOT == "/srv/hunt" and c.STRATEGY_FIT_REGIME_ROWS is False


# ──────────────────────────────────────────────────────────────────────
# v-go-live-5k-2026-09-28 (Task 2): the sleeve live allowlist
# ──────────────────────────────────────────────────────────────────────

def test_fit_lane_live_allowlist_defaults_to_off(monkeypatch):
    """Default "" -> no (family, side) is admitted by the allowlist, so the lane keeps the
    pre-existing walk-forward-PASS-only rule byte for byte."""
    c = _cfg(monkeypatch)
    assert c.FIT_LANE_LIVE_ALLOWLIST == ()


def test_fit_lane_live_allowlist_env_override(monkeypatch):
    c = _cfg(monkeypatch, FIT_LANE_LIVE_ALLOWLIST="mr_20d:long, mom_12_1:long:risk_on|mixed")
    assert c.FIT_LANE_LIVE_ALLOWLIST == ("mr_20d:long", "mom_12_1:long:risk_on|mixed")


def test_fit_lane_live_allowlist_parses_into_admission_entries(monkeypatch):
    """The config property hands raw entries through; `strategies.fit_lane_admission` owns the
    grammar (so `core` never has to know what a regime is)."""
    from strategies.fit_lane_admission import AllowEntry, parse_allowlist
    c = _cfg(monkeypatch, FIT_LANE_LIVE_ALLOWLIST="mr_20d:long,mom_12_1:long:risk_on|mixed")
    assert parse_allowlist(c.FIT_LANE_LIVE_ALLOWLIST) == (
        AllowEntry("mr_20d", "long", ()),
        AllowEntry("mom_12_1", "long", ("risk_on", "mixed")),
    )


def test_fit_lane_allowlist_strength_default_and_overrides(monkeypatch):
    """v-go-live-5k-2026-09-28 (fix round 1): the neutral strength an allowlisted sleeve signal
    carries so it sizes on the risk-per-trade path instead of a score-derived Kelly fraction."""
    monkeypatch.delenv("FIT_LANE_ALLOWLIST_STRENGTH", raising=False)
    c = _cfg(monkeypatch)
    assert c.FIT_LANE_ALLOWLIST_STRENGTH == 0.6
    assert _cfg(monkeypatch, FIT_LANE_ALLOWLIST_STRENGTH="0.75").FIT_LANE_ALLOWLIST_STRENGTH == 0.75
    # Out of the (0, 1] range, or unusable, falls back to the default rather than sizing on a
    # nonsense multiplier.
    for bad in ("0", "-0.5", "1.4", "abc", "nan"):
        assert _cfg(monkeypatch, FIT_LANE_ALLOWLIST_STRENGTH=bad).FIT_LANE_ALLOWLIST_STRENGTH == 0.6
