"""v-go-live-5k-2026-09-28 (Task 2): the sleeve live-admission rule.

A fit-lane entry goes live when its family is in FIT_LANE_LIVE_ENTRIES_ENABLED **and** either
the pre-existing rule holds (a walk-forward PASS row for that side) or the (family, side) is
allowlisted by FIT_LANE_LIVE_ALLOWLIST for the live market-context regime and the symbol's own
unconditional row is not a *measured loser*. Everything here is pure: rows in, an audit reason
out.

Fix round 1: the blocking test is no longer "verdict == FAIL". Per-symbol rows for the sleeve
families are FAIL by the pipeline's sample floors (n >= 150 etc.), not by negative edge, so
verdict-based blocking admitted only the symbols the pipeline had never measured. The pooled
family evidence is the basis; the symbol row only excludes a measured loser — n >= 60 AND
exp_r < 0.
"""
import pytest


class _Row:
    """A fit row as the admission rule reads it. `n`/`exp_r` default to the shape of a row the
    pipeline wrote but could not measure (thin sample), which must not block."""

    def __init__(self, verdict, score=0.3, source="walkforward", n=10, exp_r=0.0):
        self.verdict, self.score, self.source = verdict, score, source
        self.n, self.exp_r = n, exp_r


def _loser(verdict="FAIL"):
    """A row the pipeline measured and rejected on edge, not on sample size."""
    return _Row(verdict, n=150, exp_r=-0.08)


def _winner(verdict="WATCH"):
    return _Row(verdict, n=150, exp_r=0.05)


# ──────────────────────────────────────────────────────────────────────
# parse_allowlist
# ──────────────────────────────────────────────────────────────────────

def test_parse_allowlist_empty_is_off():
    from strategies.fit_lane_admission import parse_allowlist
    assert parse_allowlist("") == ()
    assert parse_allowlist(None) == ()
    assert parse_allowlist(()) == ()
    assert parse_allowlist("   ") == ()


def test_parse_allowlist_family_side_entries():
    from strategies.fit_lane_admission import AllowEntry, parse_allowlist
    out = parse_allowlist("mr_20d:long,mom_12_1:long")
    assert out == (AllowEntry("mr_20d", "long", ()), AllowEntry("mom_12_1", "long", ()))
    # A tuple (what Config() hands back for a yaml list) parses the same way, and case and
    # whitespace are normalized.
    assert parse_allowlist((" MR_20D : Long ",)) == (AllowEntry("mr_20d", "long", ()),)


def test_parse_allowlist_regime_qualified_entries():
    from strategies.fit_lane_admission import parse_allowlist
    out = parse_allowlist("mom_12_1:long:risk_on|mixed, mr_20d:short:risk_off")
    assert [e.regimes for e in out] == [("risk_on", "mixed"), ("risk_off",)]


@pytest.mark.parametrize("raw", ["garbage", "mr_20d", "mr_20d:sideways", ":long",
                                 "mr_20d:long:bogus", "mr_20d:long:", "a:b:c:d"])
def test_parse_allowlist_skips_malformed_entries(raw, caplog):
    """A typo must never silently widen live trading, and must never crash the lane: the entry
    is dropped with one warning and the rest of the list still parses."""
    from strategies.fit_lane_admission import parse_allowlist
    import strategies.fit_lane_admission as A
    A._WARNED.clear()
    with caplog.at_level("WARNING", logger="TradingBot"):
        out = parse_allowlist(f"{raw},mr_20d:long")
    assert [(e.family, e.side) for e in out] == [("mr_20d", "long")]
    assert "allowlist" in caplog.text


def test_parse_allowlist_warns_once_per_bad_entry(caplog):
    from strategies.fit_lane_admission import parse_allowlist
    import strategies.fit_lane_admission as A
    A._WARNED.clear()
    with caplog.at_level("WARNING", logger="TradingBot"):
        for _ in range(5):
            parse_allowlist("nonsense_entry")
    assert sum("nonsense_entry" in r.getMessage() for r in caplog.records) == 1


# ──────────────────────────────────────────────────────────────────────
# regime handling
# ──────────────────────────────────────────────────────────────────────

def test_normalize_regime_treats_unknown_as_mixed():
    from strategies.fit_lane_admission import normalize_regime
    assert normalize_regime("unknown") == "mixed"
    assert normalize_regime(None) == "mixed"
    assert normalize_regime("") == "mixed"
    assert normalize_regime("RISK_ON") == "risk_on"
    assert normalize_regime("risk_off") == "risk_off"


def test_entry_without_regime_part_permits_every_regime():
    from strategies.fit_lane_admission import allowlist_permits, parse_allowlist
    al = parse_allowlist("mr_20d:long")
    for regime in ("risk_on", "risk_off", "mixed", "unknown", None):
        assert allowlist_permits(al, "mr_20d", "long", regime) is True
    assert allowlist_permits(al, "mr_20d", "short", "risk_on") is False
    assert allowlist_permits(al, "mom_12_1", "long", "risk_on") is False


def test_entry_with_regime_part_permits_only_those_regimes():
    from strategies.fit_lane_admission import allowlist_permits, parse_allowlist
    al = parse_allowlist("mom_12_1:long:risk_on|mixed")
    assert allowlist_permits(al, "mom_12_1", "long", "risk_on") is True
    assert allowlist_permits(al, "mom_12_1", "long", "mixed") is True
    assert allowlist_permits(al, "mom_12_1", "long", "unknown") is True      # unknown == mixed
    assert allowlist_permits(al, "mom_12_1", "long", "risk_off") is False


def test_allowlisted_sides_for_family():
    from strategies.fit_lane_admission import allowlisted_sides, parse_allowlist
    al = parse_allowlist("mom_12_1:long:risk_on,mom_12_1:short,mr_20d:long")
    assert allowlisted_sides(al, "mom_12_1") == ("long", "short")
    assert allowlisted_sides(al, "mr_20d") == ("long",)
    assert allowlisted_sides(al, "orb") == ()


# ──────────────────────────────────────────────────────────────────────
# the admission matrix
# ──────────────────────────────────────────────────────────────────────

def test_admit_walkforward_pass_row_needs_no_allowlist():
    from strategies.fit_lane_admission import ADMIT_PASS, admit
    assert admit(_Row("PASS"), "mr_20d", "long", "risk_off", ()) == ADMIT_PASS


def test_admit_allowlisted_watch_row_is_admitted():
    from strategies.fit_lane_admission import ADMIT_ALLOWLIST, admit, parse_allowlist
    al = parse_allowlist("mr_20d:long")
    assert admit(_Row("WATCH"), "mr_20d", "long", "mixed", al) == ADMIT_ALLOWLIST


def test_admit_allowlisted_absent_row_is_admitted():
    """The pipeline has no measurement for this (symbol, family, side) at all — the allowlist is
    the sleeve's own gate, so an unmeasured symbol is admitted (nothing rejected it)."""
    from strategies.fit_lane_admission import ADMIT_ALLOWLIST, admit, parse_allowlist
    assert admit(None, "mom_12_1", "long", "risk_on", parse_allowlist("mom_12_1:long")) == ADMIT_ALLOWLIST


def test_admit_allowlisted_measured_loser_is_refused():
    """n >= 60 and exp_r < 0: the pipeline measured this symbol's lane and it lost money. The
    allowlist is a statement about a strategy and may not overrule that."""
    from strategies.fit_lane_admission import admit, parse_allowlist
    assert admit(_loser(), "mr_20d", "long", "mixed", parse_allowlist("mr_20d:long")) is None


def test_admit_allowlisted_fail_on_sample_floor_is_admitted():
    """Fix round 1: a FAIL that comes from the pipeline's sample floors (n < 60), not from a
    negative edge, does not block — otherwise the sleeve could only trade unmeasured symbols
    (43 of 539 for mr_20d:long)."""
    from strategies.fit_lane_admission import ADMIT_ALLOWLIST, admit, parse_allowlist
    thin_fail = _Row("FAIL", n=12, exp_r=-0.4)
    assert admit(thin_fail, "mr_20d", "long", "mixed", parse_allowlist("mr_20d:long")) == ADMIT_ALLOWLIST


def test_admit_allowlisted_measured_winner_is_admitted():
    from strategies.fit_lane_admission import ADMIT_ALLOWLIST, admit, parse_allowlist
    assert admit(_winner(), "mr_20d", "long", "mixed",
                 parse_allowlist("mr_20d:long")) == ADMIT_ALLOWLIST


# ──────────────────────────────────────────────────────────────────────
# symbol_check: the per-symbol audit label
# ──────────────────────────────────────────────────────────────────────

def test_symbol_check_labels():
    from strategies.fit_lane_admission import (SYMBOL_ABSENT, SYMBOL_BLOCKED, SYMBOL_MIN_N,
                                               SYMBOL_POSITIVE, SYMBOL_THIN, symbol_check)
    assert symbol_check(None) == SYMBOL_ABSENT
    assert symbol_check(_Row("FAIL", n=SYMBOL_MIN_N - 1, exp_r=-0.9)) == SYMBOL_THIN
    assert symbol_check(_Row("FAIL", n=SYMBOL_MIN_N, exp_r=-0.01)) == SYMBOL_BLOCKED
    assert symbol_check(_Row("WATCH", n=SYMBOL_MIN_N, exp_r=0.0)) == SYMBOL_POSITIVE
    assert symbol_check(_Row("PASS", n=500, exp_r=0.2)) == SYMBOL_POSITIVE


def test_symbol_check_unmeasurable_edge_counts_as_thin():
    """A row with a big n but a NaN/missing exp_r was not measured usefully either — it must not
    block, and it must not be reported as a positive."""
    from strategies.fit_lane_admission import SYMBOL_THIN, symbol_check
    assert symbol_check(_Row("FAIL", n=500, exp_r=float("nan"))) == SYMBOL_THIN

    class _NoFields:
        verdict, score, source = "FAIL", 0.1, "walkforward"

    assert symbol_check(_NoFields()) == SYMBOL_THIN


def test_admit_allowlisted_wrong_regime_is_refused():
    from strategies.fit_lane_admission import admit, parse_allowlist
    al = parse_allowlist("mom_12_1:long:risk_on")
    assert admit(_Row("WATCH"), "mom_12_1", "long", "risk_off", al) is None


def test_admit_not_allowlisted_watch_row_is_refused():
    from strategies.fit_lane_admission import admit
    assert admit(_Row("WATCH"), "mr_20d", "long", "mixed", ()) is None


def test_admit_coldstart_pass_row_needs_the_allowlist():
    """A cold-start PASS is not walk-forward evidence, so it is not `pass`; it is not a measured
    loser either, so an allowlisted lane may still take it (as `allowlist`)."""
    from strategies.fit_lane_admission import ADMIT_ALLOWLIST, admit, parse_allowlist
    row = _Row("PASS", source="coldstart")
    assert admit(row, "mr_20d", "long", "mixed", ()) is None
    assert admit(row, "mr_20d", "long", "mixed", parse_allowlist("mr_20d:long")) == ADMIT_ALLOWLIST


def test_admit_wrong_side_of_the_allowlist_is_refused():
    from strategies.fit_lane_admission import admit, parse_allowlist
    al = parse_allowlist("mr_20d:long")
    assert admit(_Row("WATCH"), "mr_20d", "short", "mixed", al) is None


def test_admission_module_does_not_import_research():
    import pathlib
    import re
    src = pathlib.Path("strategies/fit_lane_admission.py").read_text()
    assert not re.search(r"^\s*(from|import)\s+research\b", src, re.M)


# ──────────────────────────────────────────────────────────────────────
# fix round 1: neutral sizing for an allowlisted signal
# ──────────────────────────────────────────────────────────────────────

def _entry(side="long"):
    import pandas as pd
    from strategies.families.models import Entry
    return Entry("", side, pd.Timestamp("2025-12-01 09:35"), 100.0, 97.0, 160.0, 20, "multiday",
                 "mom_12_1_long")


def test_allowlisted_signal_gets_the_neutral_strength_and_sizing_basis(monkeypatch):
    """An allowlisted entry has no score to derive a Kelly fraction from (an absent row's score
    is NaN, which the clamp floors at 0.2 and the risk manager then caps at Kelly 0.20). It must
    size on the risk-per-trade path instead, at a neutral strength."""
    from strategies.fit_lane_admission import (ADMIT_ALLOWLIST, DEFAULT_ALLOWLIST_STRENGTH,
                                               entry_to_signal)
    sig = entry_to_signal(_entry(), "mom_12_1", {"rank_min": 0.8}, "", float("nan"), "NVDA",
                          admission=ADMIT_ALLOWLIST, regime="risk_on")
    assert sig.strength == DEFAULT_ALLOWLIST_STRENGTH == 0.6
    assert sig.confidence == DEFAULT_ALLOWLIST_STRENGTH
    assert sig.reasoning["sizing_basis"] == "risk_per_trade"


def test_allowlist_strength_is_configurable(monkeypatch):
    import strategies.fit_lane_admission as A
    monkeypatch.setattr(A, "allowlist_strength", lambda: 0.75)
    sig = A.entry_to_signal(_entry(), "mom_12_1", {}, "WATCH", 0.1, "NVDA",
                            admission=A.ADMIT_ALLOWLIST)
    assert sig.strength == 0.75 and sig.confidence == 0.75


def test_allowlist_strength_reads_config_and_falls_back(monkeypatch):
    from strategies.fit_lane_admission import DEFAULT_ALLOWLIST_STRENGTH, allowlist_strength
    monkeypatch.delenv("FIT_LANE_ALLOWLIST_STRENGTH", raising=False)
    assert allowlist_strength() == DEFAULT_ALLOWLIST_STRENGTH
    monkeypatch.setenv("FIT_LANE_ALLOWLIST_STRENGTH", "0.8")
    assert allowlist_strength() == 0.8
    monkeypatch.setenv("FIT_LANE_ALLOWLIST_STRENGTH", "nonsense")
    assert allowlist_strength() == DEFAULT_ALLOWLIST_STRENGTH


def test_pass_admission_still_sizes_off_the_fit_score():
    """The walk-forward path is unchanged: a real score still drives strength, and the basis is
    labelled as such."""
    from strategies.fit_lane_admission import ADMIT_PASS, entry_to_signal
    sig = entry_to_signal(_entry(), "mr_20d", {}, "PASS", 0.45, "NVDA", admission=ADMIT_PASS)
    assert sig.strength == pytest.approx(0.9) and sig.confidence == pytest.approx(0.9)
    assert sig.reasoning["sizing_basis"] == "fit_score"
