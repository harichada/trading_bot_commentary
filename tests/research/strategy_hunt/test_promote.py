"""Tests for research/strategy_hunt/promote.py: the pure promotion-table builder and its
markdown/`.env`-lines renderer. See task-33-brief.md for the interface contract."""
import argparse


def _row(symbol, strategy, side, verdict, source="walkforward"):
    return {"symbol": symbol, "strategy": strategy, "side": side, "verdict": verdict, "source": source}


def _snapshot(rows):
    nested: dict = {}
    for r in rows:
        nested.setdefault(r["symbol"], {})[f"{r['strategy']}:{r['side']}"] = r
    return {"computed_at": "2026-09-25T00:00:00", "floors": {}, "rows": nested}


def _pass_rows(strategy, n, side="long"):
    return [_row(f"SYM{i}", strategy, side, "PASS") for i in range(n)]


def test_promotion_table_promote_live_when_clean_and_short_family_needs_more_evidence():
    from research.strategy_hunt.promote import promotion_table

    rows = _pass_rows("orb", 12) + _pass_rows("mr_20d", 3)
    snap = _snapshot(rows)
    scorecards = {"orb": {"verdict": "GO"}}
    table = promotion_table(snap, scorecards, lanes_enabled=("orb",), live_enabled=(), swing_allowed=False)

    by_family = {r["family"]: r for r in table}
    orb = by_family["orb"]
    assert orb["pass_symbols"] == 12
    assert orb["hold_kind"] == "intraday"
    assert orb["registered"] is True
    assert orb["live"] is False
    assert orb["scorecard_verdict"] == "GO"
    assert orb["blocker"] == ""
    assert orb["recommendation"] == "promote_live"

    mr = by_family["mr_20d"]
    assert mr["pass_symbols"] == 3
    assert mr["blocker"] == "no_pass_symbols"
    assert mr["recommendation"] == "collect_shadow"


def test_promotion_table_swing_overnight_blocker_for_multiday_family():
    from research.strategy_hunt.promote import promotion_table

    rows = _pass_rows("mr_20d", 12)
    snap = _snapshot(rows)
    scorecards = {"mr_20d": {"verdict": "GO"}}
    table = promotion_table(snap, scorecards, lanes_enabled=("mr_20d",), live_enabled=(), swing_allowed=False)

    mr = next(r for r in table if r["family"] == "mr_20d")
    assert mr["hold_kind"] == "multiday"
    assert mr["blocker"] == "swing_overnight_disabled"
    assert mr["recommendation"] == "hold"


def test_promotion_table_swing_allowed_lets_multiday_family_through():
    from research.strategy_hunt.promote import promotion_table

    rows = _pass_rows("mr_20d", 12)
    snap = _snapshot(rows)
    scorecards = {"mr_20d": {"verdict": "GO"}}
    table = promotion_table(snap, scorecards, lanes_enabled=("mr_20d",), live_enabled=(), swing_allowed=True)

    mr = next(r for r in table if r["family"] == "mr_20d")
    assert mr["blocker"] == ""
    assert mr["recommendation"] == "promote_live"


def test_promotion_table_not_registered_blocks_before_scorecard_checks():
    from research.strategy_hunt.promote import promotion_table

    rows = _pass_rows("orb", 12)
    snap = _snapshot(rows)
    table = promotion_table(snap, {}, lanes_enabled=(), live_enabled=(), swing_allowed=False)

    orb = next(r for r in table if r["family"] == "orb")
    assert orb["blocker"] == "not_registered"
    assert orb["recommendation"] == "register_shadow"


def test_promotion_table_scorecard_missing_vs_no_go():
    from research.strategy_hunt.promote import promotion_table

    rows = _pass_rows("orb", 12)
    snap = _snapshot(rows)

    missing = promotion_table(snap, {}, lanes_enabled=("orb",), live_enabled=(), swing_allowed=False)
    orb_missing = next(r for r in missing if r["family"] == "orb")
    assert orb_missing["scorecard_verdict"] == "none"
    assert orb_missing["blocker"] == "scorecard_missing"
    assert orb_missing["recommendation"] == "collect_shadow"

    no_go = promotion_table(snap, {"orb": {"verdict": "NO_GO"}}, lanes_enabled=("orb",), live_enabled=(),
                             swing_allowed=False)
    orb_no_go = next(r for r in no_go if r["family"] == "orb")
    assert orb_no_go["scorecard_verdict"] == "NO_GO"
    assert orb_no_go["blocker"] == "scorecard_no_go"
    assert orb_no_go["recommendation"] == "hold"


def test_promotion_table_accepts_scorecard_value_of_literal_none_string():
    from research.strategy_hunt.promote import promotion_table

    rows = _pass_rows("orb", 12)
    snap = _snapshot(rows)
    table = promotion_table(snap, {"orb": "none"}, lanes_enabled=("orb",), live_enabled=(), swing_allowed=False)
    orb = next(r for r in table if r["family"] == "orb")
    assert orb["scorecard_verdict"] == "none"
    assert orb["blocker"] == "scorecard_missing"


def test_promotion_table_one_row_per_known_family():
    from research.strategy_hunt.promote import promotion_table
    from strategies.families import family_names

    table = promotion_table({"rows": {}}, {}, (), (), False)
    assert {r["family"] for r in table} == set(family_names())
    assert all(r["blocker"] == "no_pass_symbols" for r in table)


def test_promotion_table_already_live_family_recommends_hold_not_promote_live():
    """An already-live family has nothing new to promote: recommendation must be "hold", not
    "promote_live", so it's distinguishable from a family that's newly clear to promote."""
    from research.strategy_hunt.promote import promotion_table

    rows = _pass_rows("orb", 12)
    snap = _snapshot(rows)
    table = promotion_table(snap, {"orb": {"verdict": "GO"}}, lanes_enabled=("orb",), live_enabled=("orb",),
                             swing_allowed=False)
    orb = next(r for r in table if r["family"] == "orb")
    assert orb["live"] is True
    assert orb["blocker"] == ""
    assert orb["recommendation"] == "hold"


def test_render_promotion_includes_markdown_table_and_live_env_line():
    from research.strategy_hunt.promote import promotion_table, render_promotion

    rows = _pass_rows("orb", 12)
    snap = _snapshot(rows)
    table = promotion_table(snap, {"orb": {"verdict": "GO"}}, lanes_enabled=("orb",), live_enabled=(),
                             swing_allowed=False)
    md = render_promotion(table)

    assert "| family " in md
    assert "orb" in md
    assert "FIT_LANE_LIVE_ENTRIES_ENABLED=orb" in md
    assert "FIT_LANES_ENABLED=orb" in md


def test_render_promotion_swing_note_only_for_blocked_multiday_family():
    from research.strategy_hunt.promote import promotion_table, render_promotion

    rows = _pass_rows("mr_20d", 12)
    snap = _snapshot(rows)
    table = promotion_table(snap, {"mr_20d": {"verdict": "GO"}}, lanes_enabled=("mr_20d",), live_enabled=(),
                             swing_allowed=False)
    md = render_promotion(table)
    assert "SWING_ALLOW_OVERNIGHT=1" in md
    assert "mr_20d" in md.split("SWING_ALLOW_OVERNIGHT=1")[1].splitlines()[0]


def test_render_promotion_no_changes_recommended_message():
    from research.strategy_hunt.promote import promotion_table, render_promotion

    table = promotion_table({"rows": {}}, {}, (), (), False)
    md = render_promotion(table)
    assert "no changes recommended" in md.lower()


def test_render_promotion_omits_live_env_line_for_already_live_family_with_nothing_new():
    """Nothing is newly promotable (orb is already live) -> no FIT_LANE_LIVE_ENTRIES_ENABLED
    line should be printed at all, even though orb's own `live` flag is True."""
    from research.strategy_hunt.promote import promotion_table, render_promotion

    rows = _pass_rows("orb", 12)
    snap = _snapshot(rows)
    table = promotion_table(snap, {"orb": {"verdict": "GO"}}, lanes_enabled=("orb",), live_enabled=("orb",),
                             swing_allowed=False)
    md = render_promotion(table)
    assert "FIT_LANE_LIVE_ENTRIES_ENABLED" not in md


def test_render_promotion_live_line_unions_already_live_with_newly_promoted():
    """When another family *is* newly promotable, the printed FIT_LANE_LIVE_ENTRIES_ENABLED
    line is the union of already-live and newly-promoted families -- that's fine/expected."""
    from research.strategy_hunt.promote import promotion_table, render_promotion

    rows = _pass_rows("orb", 12) + _pass_rows("mr_20d", 12)
    snap = _snapshot(rows)
    scorecards = {"orb": {"verdict": "GO"}, "mr_20d": {"verdict": "GO"}}
    table = promotion_table(snap, scorecards, lanes_enabled=("orb", "mr_20d"), live_enabled=("orb",),
                             swing_allowed=True)
    md = render_promotion(table)
    line = next(l for l in md.splitlines() if l.startswith("FIT_LANE_LIVE_ENTRIES_ENABLED="))
    assert set(line.split("=", 1)[1].split(",")) == {"orb", "mr_20d"}


def test_promotion_table_watch_symbols_filtered_to_walkforward_source():
    from research.strategy_hunt.promote import promotion_table

    rows = _pass_rows("orb", 12) + [_row("COLD1", "orb", "long", "WATCH", source="coldstart"),
                                     _row("SYM99", "orb", "long", "WATCH", source="walkforward")]
    snap = _snapshot(rows)
    table = promotion_table(snap, {}, lanes_enabled=(), live_enabled=(), swing_allowed=False)
    orb = next(r for r in table if r["family"] == "orb")
    assert orb["watch_symbols"] == 1


def test_cmd_promote_isolates_a_single_familys_ledger_error(tmp_path, monkeypatch):
    """One family's shadow ledger blows up in `resolve()`; `cmd_promote` must not abort the
    whole run -- it logs a warning, treats that family's scorecard as "none", and still
    produces a table (and exit code 0) covering every other family."""
    from research.strategy_hunt import shadow_resolver as SR
    from research.strategy_hunt.cli import cmd_promote

    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "shadow_fit_orb.ndjson").write_text(
        '{"symbol": "AAA", "side": "long", "ts": "2024-01-05T10:00:00", "entry_px": 100.0, '
        '"stop_px": 99.0, "target_px": 102.0, "time_stop": 20, "hold_kind": "intraday", "tag": "t", '
        '"params": {}, "verdict": "PASS", "score": 0.5}\n')

    def _boom(records, root, family, cost_mult=1.0):
        if family == "orb":
            raise RuntimeError("boom: incompatible ledger")
        return []

    monkeypatch.setattr(SR, "resolve", _boom)

    a = argparse.Namespace(root=tmp_path / "strategy_hunt_root")
    rc = cmd_promote(a)
    assert rc == 0

    out = tmp_path / "docs" / "research" / "strategy_hunt_promotion.md"
    assert out.exists()
    text = out.read_text()
    assert "| orb " in text
    assert "| mr_20d " in text
    assert "| vwap_reversion " in text
