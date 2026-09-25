"""v-playbook-desk-b2-2026-09-25 (Task 4): journal rendering.

`render_journal` is pure — every section is exercised on hand-built
fixtures, including the three "what to change" derivation rules and the
"missing data renders 'none'" contract.

(Split from `test_journal.py` in review round 1 — see
`.superpowers/sdd/2026-09-25-playbook-desk-b2/task-4-findings-r1.md` item 2.
Adapter/data-store tests moved to `test_journal_data.py`.)
"""
from __future__ import annotations

from core.desk.journal import render_journal
from core.desk.models import DailyPlan, InPlayRow


# ────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────

def _plan(**kw) -> DailyPlan:
    base = dict(
        date="2026-09-25", regime="risk_on", spy_change_pct=0.6, vix=15.0,
        vix_change_pct=-2.0, breadth_score=0.7, calendar_flags=(),
        long_ok=True, short_ok=False, risk_budget_pct=1.2, max_trades=4,
        sit_out=False, sit_out_reason="", notes=(), built_at="2026-09-25T09:00:00",
        refreshed_at="2026-09-25T09:35:00",
        in_play=tuple(
            InPlayRow(symbol=f"SYM{i}", score=1.0 - i * 0.05, rvol_open15=2.5,
                      gap_pct=1.5, catalyst=1.0, rs_20=3.0, reasons=("rvol",))
            for i in range(12)
        ),
    )
    base.update(kw)
    return DailyPlan(**base)


def _decision(**kw) -> dict:
    base = dict(symbol="AAPL", component="strategy_fit", action="proceed", reason="")
    base.update(kw)
    return base


def _trade(**kw) -> dict:
    base = dict(symbol="AAPL", side="long", family="orb", r=0.8, exit_reason="target",
                entry_time="2026-09-25T09:35:00", exit_time="2026-09-25T10:12:00")
    base.update(kw)
    return base


def _playbook_change(**kw) -> dict:
    base = dict(family="orb", side="long", regime="risk_on", old_status="enabled",
                new_status="probation", reason="exp_r_negative")
    base.update(kw)
    return base


# ────────────────────────────────────────────────────────────────────
# Plan section
# ────────────────────────────────────────────────────────────────────

def test_plan_section_renders_regime_bias_budget_max_trades_sit_out_and_top10():
    out = render_journal(_plan(), [], [], [], {})
    assert "## Plan" in out
    assert "regime: risk_on" in out
    assert "bias: long only" in out
    assert "risk budget: 1.20%" in out
    assert "max trades: 4" in out
    assert "sit-out: no" in out
    # top 10 only, not all 12
    assert "SYM0" in out and "SYM9" in out
    assert "SYM10" not in out and "SYM11" not in out


def test_plan_section_bias_labels():
    assert "bias: neutral (no bias)" in render_journal(
        _plan(long_ok=False, short_ok=False), [], [], [], {})
    assert "bias: short only" in render_journal(
        _plan(long_ok=False, short_ok=True), [], [], [], {})
    assert "bias: long+short" in render_journal(
        _plan(long_ok=True, short_ok=True), [], [], [], {})


def test_plan_section_sit_out_shows_reason():
    out = render_journal(_plan(sit_out=True, sit_out_reason="vix_spike"), [], [], [], {})
    assert "sit-out: yes (vix_spike)" in out


def test_plan_section_missing_plan_renders_none():
    out = render_journal(None, [], [], [], {})
    assert "## Plan" in out
    assert "none" in out.split("## Decisions")[0]


def test_plan_section_empty_in_play_renders_none():
    out = render_journal(_plan(in_play=()), [], [], [], {})
    plan_block = out.split("## Decisions")[0]
    assert "in-play (top 10):" in plan_block
    assert "none" in plan_block


# ────────────────────────────────────────────────────────────────────
# Decisions section
# ────────────────────────────────────────────────────────────────────

def test_decisions_counts_by_component_and_action():
    decisions = [
        _decision(component="strategy_fit", action="proceed"),
        _decision(component="strategy_fit", action="proceed"),
        _decision(component="strategy_fit", action="skip", reason="fit_fail"),
        _decision(component="playbook", action="shadow", reason="playbook_suspended"),
    ]
    out = render_journal(_plan(), decisions, [], [], {})
    assert "strategy_fit/proceed: 2" in out
    assert "strategy_fit/skip: 1" in out
    assert "playbook/shadow: 1" in out


def test_decisions_every_skip_listed_with_reason():
    decisions = [
        _decision(symbol="AAPL", component="strategy_fit", action="skip", reason="fit_fail"),
        _decision(symbol="MSFT", component="strategy_fit", action="skip", reason="fit_stale"),
        _decision(symbol="NVDA", component="strategy_fit", action="proceed"),
    ]
    out = render_journal(_plan(), decisions, [], [], {})
    assert "AAPL strategy_fit: fit_fail" in out
    assert "MSFT strategy_fit: fit_stale" in out


def test_decisions_shadow_audits_that_would_have_blocked():
    decisions = [
        _decision(symbol="TSLA", component="playbook", action="shadow", reason="playbook_suspended"),
        _decision(symbol="AMD", component="strategy_fit", action="proceed"),
    ]
    out = render_journal(_plan(), decisions, [], [], {})
    assert "TSLA playbook: playbook_suspended" in out


def _blocked_section(out: str) -> str:
    """Just the "shadow audits that would have blocked" bullets (up to the next
    section heading) — the rest of the journal has "none" lines of its own."""
    return out.split("shadow audits that would have blocked:")[1].split("\n##")[0]


def test_shadow_section_lists_only_the_governed_components():
    """v-playbook-desk-b2-final-2026-09-25 (I5): "would have blocked" is a claim only
    the desk's own shadow gates can make. Other components write `action="shadow"` for
    their own reasons, and listing them there mislabelled them as blocked trades — they
    belong in the per-component summary only."""
    decisions = [
        _decision(symbol="TSLA", component="playbook", action="shadow",
                  reason="playbook_suspended"),
        _decision(symbol="NVDA", component="desk_limit", action="shadow",
                  reason="desk_max_trades"),
        _decision(symbol="AMD", component="strategy_fit", action="shadow",
                  reason="fit_fail"),
        _decision(symbol="SPY", component="regime_allocator", action="shadow",
                  reason="tape_negative"),
    ]
    out = render_journal(_plan(), decisions, [], [], {})
    blocked = _blocked_section(out)

    assert "TSLA playbook: playbook_suspended" in blocked
    assert "NVDA desk_limit: desk_max_trades" in blocked
    assert "AMD strategy_fit: fit_fail" in blocked
    assert "regime_allocator" not in blocked


def test_an_ungoverned_shadow_row_is_still_counted_in_the_summary():
    decisions = [_decision(symbol="SPY", component="regime_allocator", action="shadow",
                           reason="tape_negative")]
    out = render_journal(_plan(), decisions, [], [], {})
    assert "regime_allocator/shadow: 1" in out
    assert _blocked_section(out).strip() == "- none"


def test_decisions_empty_renders_none():
    out = render_journal(_plan(), [], [], [], {})
    decisions_block = out.split("## Decisions")[1].split("## Trades")[0]
    assert "none" in decisions_block


# ────────────────────────────────────────────────────────────────────
# Trades section
# ────────────────────────────────────────────────────────────────────

def test_trades_section_renders_symbol_side_family_r_exit_hold():
    out = render_journal(_plan(), [], [_trade()], [], {})
    trades_block = out.split("## Trades")[1].split("## Expectancy")[0]
    assert "AAPL long family=orb R=0.80 exit=target hold=37m" in trades_block


def test_trades_section_hold_falls_back_to_explicit_field():
    t = _trade(hold="2h5m", entry_time=None, exit_time=None)
    out = render_journal(_plan(), [], [t], [], {})
    assert "hold=2h5m" in out


def test_trades_section_missing_hold_data_renders_none():
    t = {"symbol": "AAPL", "side": "long", "family": "orb", "r": 0.5, "exit_reason": "target"}
    out = render_journal(_plan(), [], [t], [], {})
    assert "hold=none" in out


def test_trades_section_empty_renders_none():
    out = render_journal(_plan(), [], [], [], {})
    trades_block = out.split("## Trades")[1].split("## Expectancy")[0]
    assert "none" in trades_block


# ────────────────────────────────────────────────────────────────────
# Expectancy by setup x regime
# ────────────────────────────────────────────────────────────────────

def test_expectancy_today_computed_from_trades():
    trades = [
        _trade(family="orb", regime="risk_on", r=1.0),
        _trade(family="orb", regime="risk_on", r=-0.5),
        _trade(family="mean_rev_intraday", regime="mixed", r=0.3),
    ]
    out = render_journal(_plan(), [], trades, [], {})
    expectancy_block = out.split("## Expectancy")[1].split("## Playbook")[0]
    assert "orb/risk_on: n=2 exp_R=0.25 PF=2.00" in expectancy_block
    assert "mean_rev_intraday/mixed: n=1 exp_R=0.30 PF=inf" in expectancy_block


def test_expectancy_trailing_rows_rendered():
    trailing = {"expectancy": [
        {"family": "orb", "regime": "risk_on", "n": 25, "exp_r": 0.15, "pf": 1.4},
    ]}
    out = render_journal(_plan(), [], [], [], trailing)
    assert "orb/risk_on: n=25 exp_R=0.15 PF=1.40" in out


def test_expectancy_missing_data_renders_none_both_subsections():
    out = render_journal(_plan(), [], [], [], {})
    expectancy_block = out.split("## Expectancy")[1].split("## Playbook")[0]
    assert expectancy_block.count("none") == 2


# ────────────────────────────────────────────────────────────────────
# Playbook changes
# ────────────────────────────────────────────────────────────────────

def test_playbook_changes_rendered():
    out = render_journal(_plan(), [], [], [_playbook_change()], {})
    assert "orb/long/risk_on: enabled -> probation (exp_r_negative)" in out


def test_playbook_changes_empty_renders_none():
    out = render_journal(_plan(), [], [], [], {})
    block = out.split("## Playbook Changes")[1].split("## What to Change")[0]
    assert "none" in block


# ────────────────────────────────────────────────────────────────────
# What to change — the three derivation rules
# ────────────────────────────────────────────────────────────────────

def test_what_to_change_suspend_candidate_rule():
    trailing = {"expectancy": [
        {"family": "orb", "regime": "risk_on", "n": 20, "exp_r": -0.1, "pf": 0.6},
        {"family": "dt_momentum", "regime": "mixed", "n": 19, "exp_r": -0.2, "pf": 0.5},  # n<20: no
        {"family": "news_catalyst", "regime": "risk_on", "n": 30, "exp_r": 0.1, "pf": 1.2},  # positive: no
    ]}
    out = render_journal(_plan(), [], [], [], trailing)
    change_block = out.split("## What to Change")[1]
    assert "orb/risk_on: suspend candidate (n=20, exp_R=-0.10)" in change_block
    assert "dt_momentum" not in change_block
    assert "news_catalyst" not in change_block


def test_what_to_change_sit_out_review_rule():
    trailing = {"sit_out_days": [
        {"date": "2026-09-18", "reason": "vix_spike", "would_have_r": 1.2},
        {"date": "2026-09-19", "reason": "two_losing_days", "would_have_r": -0.4},  # negative: no
    ]}
    out = render_journal(_plan(), [], [], [], trailing)
    change_block = out.split("## What to Change")[1]
    assert "2026-09-18: review sit-out rule (would-have R=1.20, reason=vix_spike)" in change_block
    assert "2026-09-19" not in change_block


def test_what_to_change_max_trades_rule():
    trailing = {"max_trades_days": [
        {"date": "2026-09-20", "max_trades": 4, "trades_taken": 4},
        {"date": "2026-09-21", "max_trades": 4, "trades_taken": 2},  # under: no
    ]}
    out = render_journal(_plan(), [], [], [], trailing)
    change_block = out.split("## What to Change")[1]
    assert "2026-09-20: raise/lower max_trades (hit 4/4)" in change_block
    assert "2026-09-21" not in change_block


def test_what_to_change_all_three_rules_together():
    trailing = {
        "expectancy": [{"family": "orb", "regime": "risk_on", "n": 22, "exp_r": -0.3, "pf": 0.4}],
        "sit_out_days": [{"date": "2026-09-18", "reason": "vix_spike", "would_have_r": 0.9}],
        "max_trades_days": [{"date": "2026-09-20", "max_trades": 4, "trades_taken": 5}],
    }
    out = render_journal(_plan(), [], [], [], trailing)
    change_block = out.split("## What to Change")[1]
    assert "suspend candidate" in change_block
    assert "review sit-out rule" in change_block
    assert "raise/lower max_trades" in change_block


def test_what_to_change_empty_renders_none():
    out = render_journal(_plan(), [], [], [], {})
    change_block = out.split("## What to Change")[1]
    assert "none" in change_block


# ────────────────────────────────────────────────────────────────────
# Missing data end-to-end: never raises, renders "none" everywhere
# ────────────────────────────────────────────────────────────────────

def test_render_journal_with_all_missing_data_never_raises_and_renders_none():
    out = render_journal(None, None, None, None, None)
    for header in ("## Plan", "## Decisions", "## Trades", "## Expectancy",
                   "## Playbook Changes", "## What to Change"):
        assert header in out
    assert out.count("none") >= 6


def test_render_journal_tolerates_malformed_rows():
    # not dicts, missing keys, wrong types — must not raise
    out = render_journal(
        _plan(),
        decisions=["not-a-dict", {}, {"action": "skip"}],
        trades=[{}, {"symbol": "AAPL"}, "garbage"],
        playbook_changes=[{}],
        trailing={"expectancy": "not-a-list", "sit_out_days": None, "max_trades_days": [{}]},
    )
    assert "## Plan" in out
