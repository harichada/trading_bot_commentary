"""v-learning-loop-2026-10-22: every learning/risk flag defaults OFF; env overrides."""
import pytest

FLAGS = ("ENABLE_LEARNING_SCORECARD", "LEARNING_AUTODEMOTE_SHADOW", "LEARNING_AUTODEMOTE_ENFORCE",
         "LEARNING_EXIT_SHADOW", "ENABLE_LOSS_ATTRIBUTION_REPORT", "REQUIRE_STOP_R_SHADOW",
         "REQUIRE_STOP_R_ENFORCE", "ENABLE_RISK_CAP_PER_TRADE", "ENABLE_DAILY_REALIZED_LOSS_HALT")


class _M:
    def __init__(self, d=None):
        self.d = d or {}

    def get(self, k, default=None):
        return self.d.get(k, default)


def _cfg(d=None):
    from core.config import Config
    c = Config()
    c.manager = _M(d)   # instance attribute; the real yaml is never read
    return c


@pytest.mark.parametrize("flag", FLAGS)
def test_default_off(monkeypatch, flag):
    monkeypatch.delenv(flag, raising=False)
    assert getattr(_cfg(), flag) is False


@pytest.mark.parametrize("flag", FLAGS)
def test_env_on(monkeypatch, flag):
    monkeypatch.setenv(flag, "1")
    assert getattr(_cfg(), flag) is True


def test_yaml_on(monkeypatch):
    monkeypatch.delenv("LEARNING_AUTODEMOTE_ENFORCE", raising=False)
    assert _cfg({"trading.learning_autodemote_enforce": True}).LEARNING_AUTODEMOTE_ENFORCE is True


def test_caps_defaults_and_bad_values(monkeypatch):
    for k in ("RISK_CAP_PER_TRADE_USD", "DAILY_REALIZED_LOSS_HALT_USD"):
        monkeypatch.delenv(k, raising=False)
    c = _cfg()
    assert c.RISK_CAP_PER_TRADE_USD == 150.0 and c.DAILY_REALIZED_LOSS_HALT_USD == 500.0
    monkeypatch.setenv("RISK_CAP_PER_TRADE_USD", "-5")
    assert _cfg().RISK_CAP_PER_TRADE_USD == 150.0
    monkeypatch.setenv("RISK_CAP_PER_TRADE_USD", "100")
    assert _cfg().RISK_CAP_PER_TRADE_USD == 100.0
