def _cfg(monkeypatch, **env):
    for k in ("ENABLE_STRATEGY_FIT_GATE", "STRATEGY_FIT_SHADOW", "STRATEGY_FIT_PATH", "STRATEGY_FIT_MAX_AGE_DAYS",
              "FIT_LANES_ENABLED", "FIT_LANE_LIVE_ENTRIES_ENABLED", "SWING_ALLOW_OVERNIGHT", "SWING_MAX_HOLD_SESSIONS",
              "STRATEGY_HUNT_DATA_ROOT"):
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


def test_env_overrides(monkeypatch):
    c = _cfg(monkeypatch, ENABLE_STRATEGY_FIT_GATE="1", STRATEGY_FIT_SHADOW="0", STRATEGY_FIT_MAX_AGE_DAYS="3",
             FIT_LANES_ENABLED="orb, vwap_reversion", FIT_LANE_LIVE_ENTRIES_ENABLED="orb", SWING_ALLOW_OVERNIGHT="true",
             SWING_MAX_HOLD_SESSIONS="5", STRATEGY_FIT_PATH="/tmp/x.json",
             STRATEGY_HUNT_DATA_ROOT="/srv/hunt")
    assert c.ENABLE_STRATEGY_FIT_GATE is True and c.STRATEGY_FIT_SHADOW is False and c.STRATEGY_FIT_MAX_AGE_DAYS == 3
    assert c.FIT_LANES_ENABLED == ("orb", "vwap_reversion") and c.FIT_LANE_LIVE_ENTRIES_ENABLED == ("orb",)
    assert c.SWING_ALLOW_OVERNIGHT is True and c.SWING_MAX_HOLD_SESSIONS == 5 and c.STRATEGY_FIT_PATH == "/tmp/x.json"
    assert c.STRATEGY_HUNT_DATA_ROOT == "/srv/hunt"
