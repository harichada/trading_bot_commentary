"""v-learning-loop-2026-10-22: scorecard metrics + demoter triggers; never promotes."""
from datetime import date

from core.desk.playbook import (Playbook, PlaybookEntry, STATUS_ENABLED, STATUS_PROBATION,
                                STATUS_SUSPENDED)
from core.learning import demoter
from core.learning.scorecard import LearnTrade, metrics, rolling_scorecard, write_scorecard


def T(r, sess="2026-10-15", i=0, reason="proactive_macd", fam="dt_momentum", side="long", regime="mixed", src="live"):
    return LearnTrade(fam, side, regime, r, sess, f"{sess}T10:{i:02d}:00", reason, "X", r * 50, src)


def test_metrics_basic():
    m = metrics([T(1.0, i=1), T(-0.5, i=2), T(-0.5, i=3, sess="2026-10-16")])
    assert m["n"] == 3 and m["sessions"] == 2
    assert m["profit_factor"] == 1.0 and m["win_rate"] == round(1 / 3, 4)
    assert m["expectancy_r"] == 0.0 and m["max_dd_r"] == 1.0
    assert m["stage_a"] == "FAIL"


def test_rolling_scorecard_keys_and_pooled(tmp_path):
    card = rolling_scorecard([T(1, regime="mixed", i=1), T(-1, regime="risk_on", i=2)])
    assert set(card) == {"dt_momentum|long|mixed", "dt_momentum|long|risk_on", "dt_momentum|long|*"}
    assert card["dt_momentum|long|*"]["n"] == 2
    p = write_scorecard(tmp_path, card, "2026-10-22")
    assert p.exists() and "floors" in p.read_text()


def test_day_loss_trigger():
    d = demoter.evaluate([T(-0.8, i=1), T(-0.8, i=2), T(-0.6, i=3)], date(2026, 10, 15))
    assert [x.trigger for x in d] == ["day_loss_gt_2r"]


def test_day_loss_exactly_2r_does_not_trigger():
    assert demoter.evaluate([T(-1.0, i=1), T(-1.0, i=2)], date(2026, 10, 15)) == []


def test_three_consecutive_stops_trigger():
    ts = [T(-1.0, i=k, reason="stop_loss", side="short") for k in range(3)]
    trig = {x.trigger for x in demoter.evaluate(ts, date(2026, 10, 15))}
    assert trig == {"day_loss_gt_2r", "three_stops_in_session"}
    ts2 = [T(-1.0, i=0, reason="stop_loss"), T(0.2, i=1), T(-1.0, i=2, reason="stop_loss"), T(-1.0, i=3, reason="stop_loss")]
    assert "three_stops_in_session" not in {x.trigger for x in demoter.evaluate(ts2, date(2026, 10, 15))}


def test_rolling30_trigger_needs_min_n_and_neg_exp():
    small = [T(-0.1, sess=f"2026-10-{d:02d}", i=1) for d in range(10, 19)]   # 9 trades
    assert demoter.evaluate(small, date(2026, 10, 20)) == []
    big = small + [T(-0.1, sess="2026-10-19", i=1)]
    assert [x.trigger for x in demoter.evaluate(big, date(2026, 10, 20))] == ["rolling30_pf_lt1_neg"]
    win = [T(0.3 if k % 2 else -0.2, sess=f"2026-10-{10 + k:02d}", i=1) for k in range(10)]
    assert demoter.evaluate(win, date(2026, 10, 20)) == []


def test_old_sessions_outside_lookback_do_not_trigger():
    ts = [T(-1.0, sess="2026-09-01", i=k, reason="stop_loss") for k in range(3)]
    assert demoter.evaluate(ts, date(2026, 10, 22)) == []


def test_shadow_trades_never_demote():
    ts = [T(-1.0, i=k, reason="stop_loss", src="shadow") for k in range(3)]
    assert demoter.evaluate(ts, date(2026, 10, 15)) == []


def _entry(status, mult, regime="mixed"):
    return PlaybookEntry("dt_momentum", "long", regime, status, mult, 0.5, 2.0, 50, "x", "enabled_normal")


def test_apply_demotions_only_suspends_and_never_promotes():
    book = Playbook(entries={("dt_momentum", "long", "mixed"): _entry(STATUS_ENABLED, 1.25),
                             ("dt_momentum", "long", "risk_on"): _entry(STATUS_PROBATION, 0.5, "risk_on"),
                             ("gap_play", "long", "mixed"): PlaybookEntry("gap_play", "long", "mixed", STATUS_ENABLED, 1.0, 0.4, 1.2, 14, "x", "e")})
    d = [demoter.Demotion("dt_momentum", "long", "day_loss_gt_2r", "")]
    out = demoter.apply_demotions(book, d, "2026-10-22T22:55:00")
    for regime in demoter.REGIMES:
        e = out.entries[("dt_momentum", "long", regime)]
        assert e.status == STATUS_SUSPENDED and e.size_mult == 0.0
        assert e.reason == "learning_demote:day_loss_gt_2r"
    assert out.entries[("gap_play", "long", "mixed")] == book.entries[("gap_play", "long", "mixed")]
    assert demoter.apply_demotions(book, [], "t") is book


def test_apply_demotions_is_monotone_over_any_input():
    """No input can raise a status or a size multiplier."""
    rank = {STATUS_SUSPENDED: 0, STATUS_PROBATION: 1, STATUS_ENABLED: 2}
    book = Playbook(entries={("f", "long", r): PlaybookEntry("f", "long", r, s, m, 0, 0, 0, "", "")
                             for r, s, m in (("mixed", STATUS_SUSPENDED, 0.0), ("risk_on", STATUS_ENABLED, 1.5),
                                             ("risk_off", STATUS_PROBATION, 0.5))})
    out = demoter.apply_demotions(book, [demoter.Demotion("f", "long", "t", ""), demoter.Demotion("g", "short", "t", "")], "t")
    for k, e in out.entries.items():
        if k in book.entries:
            assert rank[e.status] <= rank[book.entries[k].status] and e.size_mult <= book.entries[k].size_mult
        else:
            assert e.status == STATUS_SUSPENDED


def test_suspended_entry_is_skipped_by_existing_gate_only_when_enabled():
    from core.desk.playbook import evaluate_playbook_gate
    out = demoter.apply_demotions(Playbook(), [demoter.Demotion("dt_momentum", "short", "t", "")], "t")
    assert evaluate_playbook_gate(out, "dt_momentum", "short", "mixed", True).action == "skip"
    assert evaluate_playbook_gate(out, "dt_momentum", "short", "mixed", False).action == "shadow"
    assert evaluate_playbook_gate(out, "dt_momentum", "long", "mixed", True).action == "allow"
