"""v-playbook-desk-b2-2026-09-25 (Task 3): snapshot fit verdicts and the nightly rewrite.

Split out of `test_playbook_update.py` (fix r2) which keeps the converter/reader tests; shared
builders live in `playbook_helpers.py`.
"""
import json

import pandas as pd
import pytest

from research.strategy_hunt.playbook_update import (
    RECENT_DAYS,
    best_verdict,
    resolved_trades_by_key,
    update_playbook,
)

from tests.research.strategy_hunt.playbook_helpers import (  # noqa: F401 - `root` is a fixture
    NOW,
    _ledger,
    _live_frames,
    _live_row,
    _prev_entry,
    _reader,
    _seed,
    _snapshot,
    root,
)


def test_best_verdict_prefers_pass_over_watch_over_fail():
    snap = _snapshot(
        pooled={"orb:long:risk_on": {"verdict": "WATCH"}},
        regime={"AAA": {"orb:long:risk_on": {"verdict": "FAIL"}},
                "BBB": {"orb:long:risk_on": {"verdict": "PASS"}}},
    )
    assert best_verdict(snap, "orb", "long", "risk_on") == "PASS"


def test_best_verdict_uses_the_pooled_row_when_it_is_the_only_one():
    snap = _snapshot(pooled={"orb:long:risk_off": {"verdict": "WATCH"}})
    assert best_verdict(snap, "orb", "long", "risk_off") == "WATCH"
    assert best_verdict(snap, "orb", "long", "risk_on") is None
    assert best_verdict(snap, "dt_momentum", "long", "risk_off") is None


def test_best_verdict_falls_back_to_unconditional_rows_only_for_unknown_regime():
    snap = _snapshot(rows={"AAA": {"orb:long": {"verdict": "PASS"}}})
    assert best_verdict(snap, "orb", "long", "unknown") == "PASS"
    assert best_verdict(snap, "orb", "long", "risk_on") is None


def test_best_verdict_tolerates_malformed_sections():
    assert best_verdict({"pooled_rows": ["nope"], "regime_rows": "bad", "rows": None},
                        "orb", "long", "unknown") is None
    assert best_verdict({}, "orb", "long", "risk_on") is None


# ────────────────────────── update_playbook ──────────────────────────

def test_update_playbook_writes_the_file_and_applies_the_rules(root, tmp_path):
    from core.desk.playbook import load_playbook, playbook_path

    desk = tmp_path / "desk"
    ledger_dir = tmp_path / "ledgers"
    _ledger(ledger_dir)
    losers = [("AAA", "long", "orb_contraction_rvol", f"2026-09-{d:02d}T10:00:00",
               f"2026-09-{d:02d}T11:00:00", -1.0 if d % 2 else -0.8) for d in range(1, 13)]
    frames = _live_frames(losers)

    pb = update_playbook(root, _snapshot(), NOW, desk_dir=desk, families=["toy"],
                         reader=_reader(frames), ledger_dir=ledger_dir)

    assert pb.updated_at == NOW
    entry = pb.get("orb", "long", "unknown")
    assert entry is not None and entry.status == "suspended" and entry.size_mult == 0.0
    assert pb.get("toy", "long", "unknown").status == "probation"

    on_disk = load_playbook(playbook_path(desk))
    assert on_disk.entries == pb.entries


def test_update_playbook_slices_the_last_recent_window_trades_and_can_upsize(root, tmp_path):
    """r1 finding 2: the window is the last 60 trades, and 60 >= the 40-trade upsize floor, so
    a full window of winners with a PASS row reaches 1.25."""
    from core.desk.playbook import RECENT_WINDOW

    desk = tmp_path / "desk"
    # 80 trades inside the 90-day bound: the oldest 20 are heavy losers the slice must drop.
    rows = [_live_row(88 - i, -5.0 if i < 20 else 0.3, hour=10 + (i % 5)) for i in range(80)]
    snap = _snapshot(pooled={"orb:long:unknown": {"verdict": "PASS"}})

    pb = update_playbook(root, snap, NOW, desk_dir=desk, families=[],
                         reader=_reader(_live_frames(rows)))

    entry = pb.get("orb", "long", "unknown")
    assert entry.n_recent == RECENT_WINDOW == 60
    assert entry.exp_r_recent == pytest.approx(0.3)
    assert entry.status == "enabled" and entry.size_mult == 1.25
    assert entry.reason == "enabled_upsize_fit_confirmed"


def test_update_playbook_window_is_time_bounded(root, tmp_path):
    """r1 finding 3: trades older than RECENT_DAYS are out of the sample, so a stale losing
    streak cannot re-suspend a key every night forever."""
    desk = tmp_path / "desk"
    rows = [_live_row(RECENT_DAYS + 5 + i, -3.0) for i in range(20)]        # too old to count
    rows += [_live_row(3 + i, 0.4, hour=11) for i in range(12)]             # this is the sample

    pb = update_playbook(root, _snapshot(), NOW, desk_dir=desk, families=[],
                         reader=_reader(_live_frames(rows)))

    entry = pb.get("orb", "long", "unknown")
    assert RECENT_DAYS == 90
    assert entry.n_recent == 12 and entry.exp_r_recent == pytest.approx(0.4)
    assert entry.status == "enabled"


def test_update_playbook_lapses_a_suspension_when_the_sample_is_all_stale(root, tmp_path):
    """A key whose only trades are 31-90 days old still gets measured, but it may not stay
    suspended: no resolved trade in 30 days means the suspension lapses to probation."""
    desk = tmp_path / "desk"
    rows = [_live_row(40 + i, -1.0 if i % 2 else -0.8, hour=10 + (i % 4)) for i in range(12)]

    pb = update_playbook(root, _snapshot(), NOW, desk_dir=desk, families=[],
                         reader=_reader(_live_frames(rows)))

    entry = pb.get("orb", "long", "unknown")
    assert entry.n_recent == 12 and entry.exp_r_recent < 0
    assert entry.status == "probation" and entry.size_mult == 0.5
    assert entry.reason == "suspension_lapsed_no_recent_trades"


def test_update_playbook_keeps_a_suspension_backed_by_fresh_trades(root, tmp_path):
    desk = tmp_path / "desk"
    rows = [_live_row(2 + i, -1.0 if i % 2 else -0.8, hour=10 + (i % 4)) for i in range(12)]

    pb = update_playbook(root, _snapshot(), NOW, desk_dir=desk, families=[],
                         reader=_reader(_live_frames(rows)))

    entry = pb.get("orb", "long", "unknown")
    assert entry.status == "suspended" and entry.reason == "suspended_negative_expectancy"


def test_update_playbook_lapses_a_carried_forward_suspension(root, tmp_path):
    """r1 finding 3: a suspended key places no live orders, so silence is not evidence — after
    30 days with no resolved trade the suspension lapses to probation."""
    desk = tmp_path / "desk"
    _seed(desk, _prev_entry())

    pb = update_playbook(root, _snapshot(), NOW, desk_dir=desk, families=[], reader=None)

    kept = pb.get("news_catalyst", "short", "mixed")
    assert kept.status == "probation" and kept.size_mult == 0.5
    assert kept.reason == "suspension_lapsed_no_recent_trades" and kept.updated_at == NOW
    assert (kept.exp_r_recent, kept.n_recent) == (-0.4, 20)     # the evidence is kept
    assert pb.updated_at == NOW


def test_update_playbook_carries_a_non_suspended_key_forward_verbatim(root, tmp_path):
    desk = tmp_path / "desk"
    prev = _prev_entry(status="enabled", size_mult=1.0, exp_r_recent=0.2, t_stat_recent=2.0,
                       reason="enabled_normal")
    _seed(desk, prev)

    pb = update_playbook(root, _snapshot(), NOW, desk_dir=desk, families=[], reader=None)

    assert pb.get("news_catalyst", "short", "mixed") == prev


def test_shadow_and_live_trades_land_in_one_regime_key_and_both_count(root, tmp_path):
    """r1 finding 10 + r2 finding 1a: a resolved ledger trade and a live `fit:toy` bot_trades row
    describe the same setup, so with the shadow tape now labelled by its entry-time regime they
    share a *real* (family, side, regime) key — not the `unknown` bucket the gate can only reach
    through a fallback — and both count toward one sample."""
    from tests.research.strategy_hunt.playbook_helpers import write_regime_context

    write_regime_context(root)                               # SPY +0.5% by 09:55, VIX 15
    desk = tmp_path / "desk"
    ledger_dir = tmp_path / "ledgers"
    _ledger(ledger_dir)                                      # two shadow trades, family "toy"
    rows = [_live_row(4, 0.9, strategy="fit:toy")]            # one live trade, same family/side
    # The audit row must sit inside REGIME_MATCH_WINDOW before the 09:00 entry _live_row builds.
    entry_day = (pd.Timestamp(NOW) - pd.Timedelta(days=4)).normalize()
    audit = [("AAA", (entry_day + pd.Timedelta(hours=8, minutes=45)).isoformat(),
              json.dumps({"regime": "risk_on"}))]

    frames = _live_frames(rows, decisions=audit)
    pb = update_playbook(root, _snapshot(), NOW, desk_dir=desk, families=["toy"],
                         reader=_reader(frames), ledger_dir=ledger_dir)

    grouped = resolved_trades_by_key(root, ["toy"], reader=_reader(frames),
                                     ledger_dir=ledger_dir, now=NOW)
    key = ("toy", "long", "risk_on")
    assert sorted(t.source for t in grouped[key]) == ["live", "shadow", "shadow"]
    assert pb.get(*key).n_recent == 3
    assert pb.get("toy", "long", "unknown") is None


def test_update_playbook_never_writes_outside_the_desk_dir(root, tmp_path):
    desk = tmp_path / "nested" / "desk"
    pb = update_playbook(root, _snapshot(), NOW, desk_dir=desk, families=[], reader=None)

    assert (desk / "playbook.json").exists()
    assert json.loads((desk / "playbook.json").read_text())["updated_at"] == NOW
    assert pb.entries == {}

