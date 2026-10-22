"""v-learning-loop-2026-10-22: exit replay rules, loss attribution, nightly flag gating."""
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from research.learning.data import normalize
from research.learning.exit_replay import replay_one, summarize
from research.learning import loss_attribution, nightly


def _row(**over):
    r = dict(id=1, symbol="NVDA", side="long", strategy="day_trade_momentum", setup_type="",
             entry_et="2026-09-22 09:42:30-04", exit_et="2026-09-22 10:00:00-04", entry_price=100, exit_price=99,
             quantity=10, pnl=-10, pnl_r="", initial_stop=98, initial_tp=104, initial_risk_per_share="",
             exit_reason="proactive_macd_flipped_bearish", mode="live", source="bot", is_external="f",
             is_hands_off="f", sig_entry=100, sig_stop=98, sig_target=104)
    r.update(over)
    return r


def _bars(rows):
    return pd.DataFrame([dict(ts=pd.Timestamp(t), o=o, h=h, l=l, c=c) for t, o, h, l, c in rows])


def test_normalize_derives_r_and_excludes_hands_off():
    ts = normalize([_row(), _row(id=2, symbol="HQGE"), _row(id=3, is_external="t"), _row(id=4, mode="simulation"),
                    _row(id=5, source="external")], ["HQGE", "SPCX"])
    assert [t.id for t in ts] == [1]
    assert ts[0].rps == 2 and ts[0].r == -0.5 and ts[0].r_source == "derived:initial_stop"
    assert ts[0].entry_ts == pd.Timestamp("2026-09-22 09:42:30")


def test_replay_stop_target_flatten_and_ties():
    t = normalize([_row()])[0]
    # entry-minute bar ignored even though it hits the stop
    b = _bars([("2026-09-22 09:42", 100, 100, 97, 99), ("2026-09-22 09:43", 100, 104.5, 99.5, 104)])
    r = replay_one(t, b)
    assert r.status == "ok" and r.cf_exit_kind == "target" and r.cf_pnl == 40 and r.cf_r == 2.0
    tie = replay_one(t, _bars([("2026-09-22 09:43", 100, 105, 97, 100)]))
    assert tie.cf_exit_kind == "stop" and tie.cf_pnl == -20
    gap = replay_one(t, _bars([("2026-09-22 09:43", 97, 97.5, 96, 97)]))
    assert gap.cf_exit_kind == "stop_gap" and gap.cf_pnl == -30
    fl = replay_one(t, _bars([("2026-09-22 09:43", 100, 101, 99, 100), ("2026-09-22 15:00", 101.5, 102, 101, 101)]))
    assert fl.cf_exit_kind == "time_flatten" and fl.cf_pnl == 15
    inc = replay_one(t, _bars([("2026-09-22 09:43", 100, 101, 99, 100)]))
    assert inc.status == "incomplete" and inc.cf_pnl is None
    assert replay_one(t, None).status == "no_bars"


def test_replay_short_mirror():
    t = normalize([_row(side="short", initial_stop=102, initial_tp=96, pnl=-10)])[0]
    r = replay_one(t, _bars([("2026-09-22 09:43", 100, 100.5, 95.5, 96)]))
    assert r.cf_exit_kind == "target" and r.cf_pnl == 40


def test_summary_marks_unreplayed():
    t = normalize([_row()])[0]
    s = summarize([replay_one(t, None)])
    assert s["n_no_bars"] == 1 and s["n_replayed"] == 0 and s["unreplayed_actual_pnl"] == -10


def test_attribution_buckets():
    t1 = normalize([_row()])[0]
    t2 = normalize([_row(id=2, symbol="AMD", entry_et="2026-09-22 09:44:00-04", sig_entry=99.9)])[0]
    bars = {"1": _bars([("2026-09-22 09:42", 100, 100.2, 99, 99.5), ("2026-09-22 09:47", 99.5, 99.6, 99, 99.2)])}
    res = loss_attribution.attribute([t1, t2], bars)
    s = res["summary"]
    assert s["n"] == 2 and s["pnl"] == -20
    assert s["cluster"]["n_clusters"] == 1 and s["cluster"]["clusters"][0]["symbols"] == ["NVDA", "AMD"]
    assert s["cost"]["slip_usd_total"] == 1.0          # AMD filled 0.1 worse x 10
    assert s["exit"]["early_losing_exits"] == 2
    assert "## Cluster" in loss_attribution.render(res, "a", "b")


def test_nightly_all_flags_off_is_noop(tmp_path):
    class C:
        pass
    res = nightly.run(C(), reader=None, now=datetime(2026, 10, 22, 22, 55), out_dir=tmp_path / "l",
                      desk_dir=tmp_path / "d", docs_dir=tmp_path / "docs")
    assert res["skipped"] and not (tmp_path / "l").exists()


def _reader(rows):
    def read(q, params):
        if "FROM bot_trades" in q:
            return pd.DataFrame(rows)
        return pd.DataFrame(columns=["ts", "o", "h", "l", "c"])
    return read


def test_nightly_shadow_logs_but_does_not_touch_playbook(tmp_path):
    class C:
        LEARNING_AUTODEMOTE_SHADOW = True
        ENABLE_LEARNING_SCORECARD = True
        HANDS_OFF_DENYLIST = frozenset({"HQGE", "SPCX"})
    rows = [_row(id=k, side="short", initial_stop=101, initial_tp=95, pnl=-10, exit_reason="stop_loss",
                 entry_et=f"2026-10-15 09:5{k}:00-04", exit_et=f"2026-10-15 10:0{k}:00-04") for k in range(3)]
    res = nightly.run(C(), _reader(rows), datetime(2026, 10, 15, 22, 55), tmp_path / "l", tmp_path / "d", tmp_path / "docs")
    assert {d["trigger"] for d in res["demotions"]} == {"day_loss_gt_2r", "three_stops_in_session"}
    assert not (tmp_path / "d" / "playbook.json").exists()
    assert any("scorecard-2026-10-15.json" in w for w in res["wrote"])


def test_nightly_enforce_suspends_in_playbook(tmp_path):
    from core.desk.playbook import load_playbook
    class C:
        LEARNING_AUTODEMOTE_ENFORCE = True
        HANDS_OFF_DENYLIST = frozenset()
    rows = [_row(id=k, side="short", initial_stop=101, initial_tp=95, pnl=-10, exit_reason="stop_loss",
                 entry_et=f"2026-10-15 09:5{k}:00-04", exit_et=f"2026-10-15 10:0{k}:00-04") for k in range(3)]
    nightly.run(C(), _reader(rows), datetime(2026, 10, 15, 22, 55), tmp_path / "l", tmp_path / "d", tmp_path / "docs")
    book = load_playbook(tmp_path / "d" / "playbook.json")
    assert book.get("dt_momentum", "short", "mixed").status == "suspended"
    assert book.get("dt_momentum", "long", "mixed") is None


def test_migration_is_nullable_and_idempotent():
    import importlib.util
    spec = importlib.util.spec_from_file_location("m", Path(__file__).resolve().parents[2] / "scripts" / "migrate_fee_columns.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    for s in m.statements():
        assert "ADD COLUMN IF NOT EXISTS" in s and "NOT NULL" not in s and "DEFAULT" not in s
    assert m.main([]) == 0   # dry run
