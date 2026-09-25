"""v-playbook-desk-b2-2026-09-25: in-play scoring.

score = 0.4*min(rvol_open15/3, 1) + 0.2*min(|gap_pct|/5, 1) + 0.2*catalyst
        + 0.2*min(|rs_20|/10, 1)

Every component is capped at 1, a NaN input scores 0 and leaves a
`missing:<field>` reason, and pinned symbols are always in the output.
"""
from __future__ import annotations

import math

from core.desk.inplay_ranker import rank_in_play, score_row
from core.desk.models import InPlayRow

NAN = float("nan")


def _row(symbol: str, rvol=NAN, gap=NAN, catalyst=0.0, rs=NAN, reasons=()) -> InPlayRow:
    return InPlayRow(symbol=symbol, rvol_open15=rvol, gap_pct=gap,
                     catalyst=catalyst, rs_20=rs, reasons=tuple(reasons))


# ────────────────────────────────────────────────────────────────────
# score_row — the formula, by hand
# ────────────────────────────────────────────────────────────────────

def test_score_formula_by_hand_three_rows():
    # A: every component saturated -> 0.4 + 0.2 + 0.2 + 0.2
    a = score_row(_row("AAA", rvol=6.0, gap=10.0, catalyst=1.0, rs=-20.0))
    # B: every component at half -> 0.2 + 0.1 + 0.0 + 0.1
    b = score_row(_row("BBB", rvol=1.5, gap=2.5, catalyst=0.0, rs=5.0))
    # C: rvol exactly at the cap, no gap reading, catalyst on, flat rs
    c = score_row(_row("CCC", rvol=3.0, gap=NAN, catalyst=1.0, rs=0.0))
    assert a.score == 1.0
    assert b.score == 0.4
    assert c.score == 0.6


def test_score_components_are_capped_not_extrapolated():
    high = score_row(_row("AAA", rvol=30.0, gap=500.0, catalyst=1.0, rs=99.0))
    cap = score_row(_row("AAA", rvol=3.0, gap=5.0, catalyst=1.0, rs=10.0))
    assert high.score == cap.score == 1.0


def test_score_uses_absolute_gap_and_rs():
    up = score_row(_row("AAA", rvol=0.0, gap=2.5, catalyst=0.0, rs=5.0))
    down = score_row(_row("AAA", rvol=0.0, gap=-2.5, catalyst=0.0, rs=-5.0))
    assert up.score == down.score == 0.2


def test_score_nan_inputs_count_as_zero_with_reasons():
    out = score_row(InPlayRow(symbol="AAA", catalyst=NAN))
    assert out.score == 0.0
    assert out.reasons == ("missing:rvol_open15", "missing:gap_pct",
                           "missing:catalyst", "missing:rs_20")


def test_score_keeps_existing_reasons_and_does_not_duplicate():
    row = _row("AAA", rvol=3.0, reasons=("earnings", "missing:gap_pct"))
    out = score_row(row)
    assert out.reasons == ("earnings", "missing:gap_pct", "missing:rs_20")


def test_score_returns_a_new_row_and_never_mutates():
    row = _row("AAA", rvol=3.0, gap=1.0, catalyst=1.0, rs=1.0)
    out = score_row(row)
    assert out is not row
    assert math.isnan(row.score)
    assert (out.rvol_open15, out.gap_pct, out.catalyst, out.rs_20) == (3.0, 1.0, 1.0, 1.0)


def test_score_non_finite_counts_as_missing_like_nan():
    # inf serializes as null in InPlayRow.to_dict; it must not saturate a component.
    out = score_row(_row("AAA", rvol=float("inf"), gap=float("-inf")))
    assert out.score == 0.0
    assert "missing:rvol_open15" in out.reasons and "missing:gap_pct" in out.reasons


def test_score_clamps_out_of_range_components():
    assert score_row(_row("AAA", rvol=-5.0)).score == 0.0        # negative rvol -> 0
    assert score_row(_row("AAA", catalyst=7.0)).score == 0.2     # catalyst is 0/1
    assert score_row(_row("AAA", catalyst=-1.0)).score == 0.0


# ────────────────────────────────────────────────────────────────────
# rank_in_play — ordering, top_n, pinned
# ────────────────────────────────────────────────────────────────────

def test_rank_orders_by_score_desc_then_symbol():
    rows = [_row("BBB", rvol=1.5, gap=2.5), _row("AAA", rvol=1.5, gap=2.5),
            _row("CCC", rvol=6.0, gap=10.0, catalyst=1.0, rs=20.0)]
    out = rank_in_play(rows, pinned=[], top_n=10)
    assert [r.symbol for r in out] == ["CCC", "AAA", "BBB"]
    assert [r.score for r in out] == [1.0, 0.3, 0.3]


def test_rank_truncates_to_top_n():
    rows = [_row(s, rvol=v) for s, v in (("AAA", 3.0), ("BBB", 2.0), ("CCC", 1.0))]
    out = rank_in_play(rows, pinned=[], top_n=2)
    assert [r.symbol for r in out] == ["AAA", "BBB"]


def test_rank_top_n_zero_or_negative_keeps_only_pinned():
    rows = [_row("AAA", rvol=3.0)]
    assert rank_in_play(rows, pinned=[], top_n=0) == ()
    out = rank_in_play(rows, pinned=["ZZZ"], top_n=-3)
    assert [r.symbol for r in out] == ["ZZZ"]


def test_pinned_symbol_absent_from_candidates_is_appended_and_flagged():
    rows = [_row("AAA", rvol=3.0), _row("BBB", rvol=2.0)]
    out = rank_in_play(rows, pinned=["zzz"], top_n=1)
    assert [r.symbol for r in out] == ["AAA", "ZZZ"]
    pinned = out[-1]
    assert pinned.reasons == ("missing:rvol_open15", "missing:gap_pct",
                              "missing:rs_20", "pinned")
    assert pinned.score == 0.0


def test_pinned_candidate_below_the_cut_is_appended_with_its_features():
    rows = [_row("AAA", rvol=3.0), _row("BBB", rvol=1.5, gap=2.5)]
    out = rank_in_play(rows, pinned=["BBB"], top_n=1)
    assert [r.symbol for r in out] == ["AAA", "BBB"]
    assert out[-1].rvol_open15 == 1.5
    assert out[-1].score == 0.3
    assert "pinned" in out[-1].reasons


def test_pinned_candidate_already_ranked_keeps_its_place_and_is_flagged_once():
    rows = [_row("AAA", rvol=3.0), _row("BBB", rvol=1.0)]
    out = rank_in_play(rows, pinned=["AAA", "AAA"], top_n=5)
    assert [r.symbol for r in out] == ["AAA", "BBB"]
    assert out[0].reasons.count("pinned") == 1


def test_pinned_order_follows_the_pinned_list():
    out = rank_in_play([], pinned=["MSFT", "AAPL"], top_n=5)
    assert [r.symbol for r in out] == ["MSFT", "AAPL"]


def test_rank_ignores_blank_symbols_and_non_rows():
    rows = [_row("", rvol=9.0), _row("AAA", rvol=3.0)]
    out = rank_in_play(rows, pinned=["", "  "], top_n=5)
    assert [r.symbol for r in out] == ["AAA"]


def test_rank_dedupes_duplicate_candidate_symbols_keeping_the_best():
    rows = [_row("AAA", rvol=1.0), _row("aaa", rvol=3.0)]
    out = rank_in_play(rows, pinned=[], top_n=5)
    assert [r.symbol for r in out] == ["AAA"]
    assert out[0].score == 0.4


def test_rank_is_deterministic_and_does_not_mutate_its_input():
    rows = [_row("BBB", rvol=1.0), _row("AAA", rvol=1.0)]
    first = rank_in_play(rows, pinned=["CCC"], top_n=5)
    second = rank_in_play(rows, pinned=["CCC"], top_n=5)
    assert first == second
    assert [r.symbol for r in rows] == ["BBB", "AAA"]
    assert all(math.isnan(r.score) for r in rows)


def test_rank_returns_a_tuple_of_rows():
    out = rank_in_play([_row("AAA", rvol=3.0)], pinned=[], top_n=5)
    assert isinstance(out, tuple) and all(isinstance(r, InPlayRow) for r in out)
