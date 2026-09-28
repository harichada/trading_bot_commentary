import json
import threading
from datetime import datetime
from zoneinfo import ZoneInfo
import pytest
from core import settings_console_store as store

ET = ZoneInfo("America/New_York")

class FakeManager:
    def __init__(self, data=None):
        self.data = dict(data or {})
        self.updates = []
    def get(self, key, default=None):
        return self.data.get(key, default)
    def update(self, key, value):
        self.data[key] = value
        self.updates.append((key, value))

class FakeCfg:
    RISK_PER_TRADE_PCT = 0.01
    BOT_CAPITAL_ALLOCATION = 5000.0
    ORB_LIVE_ENTRIES_ENABLED = True
    FIT_LANE_LIVE_ENTRIES_ENABLED = ("mr_20d", "mom_12_1")
    MAX_POSITIONS = 5
    ENABLE_PLAYBOOK_GATE = True

def _mon(h=10, m=0):  # Monday 2026-09-28
    return datetime(2026, 9, 28, h, m, tzinfo=ET)

@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    for name in ("RISK_PER_TRADE_PCT", "BOT_CAPITAL_ALLOCATION", "ORB_LIVE_ENTRIES_ENABLED",
                 "FIT_LANE_LIVE_ENTRIES_ENABLED", "ENABLE_PLAYBOOK_GATE", "FIT_LANES_ENABLED"):
        monkeypatch.delenv(name, raising=False)

def test_next_promotion_same_day_before_925():
    assert store.next_promotion(_mon(8, 0)) == _mon(9, 25)

def test_next_promotion_after_925_rolls_to_next_session_day():
    assert store.next_promotion(_mon(10, 0)) == datetime(2026, 9, 29, 9, 25, tzinfo=ET)
    fri = datetime(2026, 10, 2, 15, 0, tzinfo=ET)
    assert store.next_promotion(fri) == datetime(2026, 10, 5, 9, 25, tzinfo=ET)

def test_tighten_applies_immediately_and_audits(tmp_path):
    m = FakeManager()
    r = store.submit_change(m, FakeCfg(), str(tmp_path), "risk_per_trade_pct", "0.005",
                            now=_mon(), actor="test", reason="cut risk")
    assert r.status == "applied" and r.direction == "tighten"
    assert m.data["trading.risk_per_trade_pct"] == 0.005
    rows = store.read_audit(str(tmp_path))
    assert rows[-1]["event"] == "applied" and rows[-1]["key"] == "risk_per_trade_pct"

def test_loosen_queues_until_next_open(tmp_path):
    m = FakeManager()
    r = store.submit_change(m, FakeCfg(), str(tmp_path), "risk_per_trade_pct", 0.02,
                            now=_mon(), actor="test", reason="more risk")
    assert r.status == "pending" and r.direction == "loosen"
    assert "trading.risk_per_trade_pct" not in m.data
    pend = store.list_pending(str(tmp_path))
    assert len(pend) == 1 and pend[0]["key"] == "risk_per_trade_pct" and pend[0]["id"] == r.pending_id
    assert pend[0]["effective_at"].startswith("2026-09-29T09:25")

def test_second_loosen_replaces_pending_for_same_key(tmp_path):
    m = FakeManager()
    store.submit_change(m, FakeCfg(), str(tmp_path), "risk_per_trade_pct", 0.02, now=_mon(), actor="t", reason="")
    store.submit_change(m, FakeCfg(), str(tmp_path), "risk_per_trade_pct", 0.03, now=_mon(), actor="t", reason="")
    pend = store.list_pending(str(tmp_path))
    assert len(pend) == 1 and pend[0]["new"] == 0.03

def test_promote_applies_due_rows_only(tmp_path):
    m = FakeManager()
    store.submit_change(m, FakeCfg(), str(tmp_path), "risk_per_trade_pct", 0.02, now=_mon(), actor="t", reason="")
    assert store.promote_pending(m, str(tmp_path), now=datetime(2026, 9, 29, 9, 0, tzinfo=ET)) == []
    applied = store.promote_pending(m, str(tmp_path), now=datetime(2026, 9, 29, 9, 26, tzinfo=ET))
    assert [a["key"] for a in applied] == ["risk_per_trade_pct"]
    assert m.data["trading.risk_per_trade_pct"] == 0.02
    assert store.list_pending(str(tmp_path)) == []
    assert store.read_audit(str(tmp_path))[-1]["event"] == "promoted"

def test_cancel_pending(tmp_path):
    m = FakeManager()
    r = store.submit_change(m, FakeCfg(), str(tmp_path), "orb_live_entries_enabled", True, now=_mon(), actor="t", reason="")
    # ORB already True in FakeCfg → noop; use a real loosen instead
    r = store.submit_change(m, FakeCfg(), str(tmp_path), "max_positions", 8, now=_mon(), actor="t", reason="")
    assert r.status == "pending"
    assert store.cancel_pending(str(tmp_path), r.pending_id, now=_mon(), actor="t") is True
    assert store.list_pending(str(tmp_path)) == []
    assert store.cancel_pending(str(tmp_path), "nope", now=_mon(), actor="t") is False

def test_csv_stored_as_list_and_removal_is_tighten(tmp_path):
    m = FakeManager()
    r = store.submit_change(m, FakeCfg(), str(tmp_path), "fit_lane_live_entries_enabled", "mr_20d",
                            now=_mon(), actor="t", reason="drop momentum")
    assert r.status == "applied" and m.data["trading.fit_lane_live_entries_enabled"] == ["mr_20d"]

def test_rejections(tmp_path, monkeypatch):
    m = FakeManager()
    assert store.submit_change(m, FakeCfg(), str(tmp_path), "nope", 1, now=_mon(), actor="t", reason="").status == "rejected"
    assert store.submit_change(m, FakeCfg(), str(tmp_path), "fit_lanes_enabled", "mr_20d", now=_mon(), actor="t", reason="").reason == "restart_required"
    monkeypatch.setenv("RISK_PER_TRADE_PCT", "0.01")
    r = store.submit_change(m, FakeCfg(), str(tmp_path), "risk_per_trade_pct", 0.005, now=_mon(), actor="t", reason="")
    assert r.status == "rejected" and r.reason.startswith("env_shadowed")
    monkeypatch.delenv("RISK_PER_TRADE_PCT")
    r = store.submit_change(m, FakeCfg(), str(tmp_path), "bot_capital_allocation", 9000, now=_mon(), actor="t", reason="")
    assert r.status == "rejected"
    assert store.read_audit(str(tmp_path))[-1]["event"] == "rejected"

def test_block_and_unblock(tmp_path):
    m = FakeManager()
    r = store.set_block_new_entries(m, str(tmp_path), True, now=_mon(), actor="t", reason="glitch")
    assert r.status == "applied" and m.data["trading.block_new_entries"] is True
    r = store.set_block_new_entries(m, str(tmp_path), False, now=_mon(), actor="t", reason="")
    assert r.status == "rejected" and m.data["trading.block_new_entries"] is True
    r = store.set_block_new_entries(m, str(tmp_path), False, now=_mon(), actor="t", reason="ok", confirm="YES")
    assert r.status == "applied" and m.data["trading.block_new_entries"] is False
    events = [row["event"] for row in store.read_audit(str(tmp_path))]
    assert events[-2:] == ["rejected", "unblock_entries"] or events[-1] == "unblock_entries"

def test_pending_file_corruption_degrades_to_empty(tmp_path):
    (tmp_path / store.PENDING_FILE).write_text("{not json")
    assert store.list_pending(str(tmp_path)) == []


# ── Fix round 1 regression tests ─────────────────────────────────────

def test_tighten_cancels_stale_pending_loosen_for_same_key(tmp_path):
    m = FakeManager()
    pending = store.submit_change(m, FakeCfg(), str(tmp_path), "risk_per_trade_pct", 0.02,
                                   now=_mon(), actor="t", reason="more risk")
    assert pending.status == "pending"
    tightened = store.submit_change(m, FakeCfg(), str(tmp_path), "risk_per_trade_pct", 0.005,
                                     now=_mon(), actor="t", reason="cut risk")
    assert tightened.status == "applied" and tightened.direction == "tighten"
    assert store.list_pending(str(tmp_path)) == []
    events = [row["event"] for row in store.read_audit(str(tmp_path))]
    assert events[-2:] == ["cancelled", "applied"]
    cancelled_row = [r for r in store.read_audit(str(tmp_path)) if r["event"] == "cancelled"][-1]
    assert cancelled_row["reason"] == "superseded_by_tighten"
    assert cancelled_row["key"] == "risk_per_trade_pct"


def test_concurrent_loosen_submissions_do_not_clobber_each_other(tmp_path):
    m = FakeManager()
    desk_dir = str(tmp_path)
    errors: list[Exception] = []

    def worker(key, raw):
        try:
            for _ in range(50):
                store.submit_change(m, FakeCfg(), desk_dir, key, raw, now=_mon(), actor="t", reason="")
        except Exception as exc:  # pragma: no cover - surfaced via assertion below
            errors.append(exc)

    t1 = threading.Thread(target=worker, args=("risk_per_trade_pct", 0.02))
    t2 = threading.Thread(target=worker, args=("max_positions", 8))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert not errors
    keys = {row["key"] for row in store.list_pending(desk_dir)}
    assert keys == {"risk_per_trade_pct", "max_positions"}


def test_read_audit_survives_invalid_byte_sequence(tmp_path):
    (tmp_path / store.AUDIT_FILE).write_bytes(b'{"ts": "ok"}\n\xff\xfe not valid utf-8\n')
    assert store.read_audit(str(tmp_path)) == []


def test_read_audit_limit_zero_returns_empty(tmp_path):
    m = FakeManager()
    store.submit_change(m, FakeCfg(), str(tmp_path), "risk_per_trade_pct", "0.005",
                        now=_mon(), actor="t", reason="cut risk")
    assert store.read_audit(str(tmp_path), limit=0) == []


class ValidatingManager(FakeManager):
    """Mimics ConfigManager: exposes `config` and `validate_candidate`."""
    def __init__(self, data=None):
        super().__init__(data)
        self.config = {"schwab": {}, "trading": {}, "commentary": {}, "technical_analysis": {}}
    def update(self, key, value):
        super().update(key, value)
        node = self.config
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    @staticmethod
    def validate_candidate(config):
        v = config.get("trading", {}).get("max_positions")
        if v is not None and not (1 <= v <= 10):
            raise ValueError("Max positions must be 1-10 (test band)")


def test_submit_rejects_value_the_config_validator_refuses(tmp_path):
    m = ValidatingManager()
    r = store.submit_change(m, FakeCfg(), str(tmp_path), "max_positions", 3, now=_mon(), actor="t", reason="")
    assert r.status == "applied" and m.config["trading"]["max_positions"] == 3
    # 20 passes the registry (1..50) but not the manager's band → rejected, nothing written
    r = store.submit_change(m, FakeCfg(), str(tmp_path), "max_positions", 20, now=_mon(), actor="t", reason="")
    assert r.status == "rejected" and r.reason.startswith("config_validation")
    assert m.config["trading"]["max_positions"] == 3
    assert store.list_pending(str(tmp_path)) == []


def test_promotion_revalidates_against_todays_ceiling(tmp_path):
    m = FakeManager({"trading.settings_console.allocation_ceiling": 5000})
    cfg = FakeCfg(); cfg.BOT_CAPITAL_ALLOCATION = 3000.0
    r = store.submit_change(m, cfg, str(tmp_path), "bot_capital_allocation", 4500, now=_mon(), actor="t", reason="")
    assert r.status == "pending"
    m.data["trading.settings_console.allocation_ceiling"] = 4000   # lowered overnight
    applied = store.promote_pending(m, str(tmp_path), now=datetime(2026, 9, 29, 9, 30, tzinfo=ET))
    assert applied == []
    assert "trading.bot_capital_allocation" not in m.data
    assert store.list_pending(str(tmp_path)) == []          # dropped, not retried forever
    assert store.read_audit(str(tmp_path))[-1]["event"] == "promotion_rejected"


def test_promote_time_comes_from_cfg(tmp_path):
    m = FakeManager()
    cfg = FakeCfg(); cfg.SETTINGS_PROMOTE_TIME_ET = "09:31"
    r = store.submit_change(m, cfg, str(tmp_path), "max_positions", 8, now=_mon(8, 0), actor="t", reason="")
    assert r.status == "pending" and r.effective_at.startswith("2026-09-28T09:31")
    assert store.promote_time_of(FakeCfg()) == (9, 25)      # attribute absent → default
    cfg.SETTINGS_PROMOTE_TIME_ET = "garbage"
    assert store.promote_time_of(cfg) == (9, 25)
