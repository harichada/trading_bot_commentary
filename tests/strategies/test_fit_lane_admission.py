"""v-go-live-5k-2026-09-28 (Task 2): the sleeve live-admission rule.

A fit-lane entry goes live when its family is in FIT_LANE_LIVE_ENTRIES_ENABLED **and** either
the pre-existing rule holds (a walk-forward PASS row for that side) or the (family, side) is
allowlisted by FIT_LANE_LIVE_ALLOWLIST for the live market-context regime and the symbol's own
unconditional row does not FAIL. Everything here is pure: rows in, an audit reason out.
"""
import pytest


class _Row:
    def __init__(self, verdict, score=0.3, source="walkforward"):
        self.verdict, self.score, self.source = verdict, score, source


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


def test_admit_allowlisted_fail_row_is_refused():
    from strategies.fit_lane_admission import admit, parse_allowlist
    assert admit(_Row("FAIL"), "mr_20d", "long", "mixed", parse_allowlist("mr_20d:long")) is None


def test_admit_allowlisted_wrong_regime_is_refused():
    from strategies.fit_lane_admission import admit, parse_allowlist
    al = parse_allowlist("mom_12_1:long:risk_on")
    assert admit(_Row("WATCH"), "mom_12_1", "long", "risk_off", al) is None


def test_admit_not_allowlisted_watch_row_is_refused():
    from strategies.fit_lane_admission import admit
    assert admit(_Row("WATCH"), "mr_20d", "long", "mixed", ()) is None


def test_admit_coldstart_pass_row_needs_the_allowlist():
    """A cold-start PASS is not walk-forward evidence, so it is not `pass`; it is not a FAIL
    either, so an allowlisted lane may still take it (as `allowlist`)."""
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
