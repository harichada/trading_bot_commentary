"""v-playbook-desk-b2-2026-09-25 (Task 3): the nightly updater's two tapes.

Offline: the shadow side runs the real `shadow_resolver.resolve` against a tmp root of generated
bars, and the live side gets its `bot_trades` / `bot_decisions` frames from an injected SQL reader
callable (no Postgres). The verdict/rewrite tests live in `test_playbook_nightly.py`; shared
builders in `playbook_helpers.py`.
"""
import json

import pandas as pd
import pytest

from research.strategy_hunt.playbook_update import (
    FAMILY_BY_STRATEGY,
    LIVE_LOOKBACK_DAYS,
    LIVE_REGIME_SQL,
    LIVE_TRADES_SQL,
    group_by_key,
    live_resolved_trades,
    resolved_from_live,
    resolved_from_sim,
    resolved_trades_by_key,
)

from tests.research.strategy_hunt.playbook_helpers import (  # noqa: F401 - `root` is a fixture
    NOW,
    _ledger,
    _live_frames,
    _reader,
    _sim_trade,
    root,
)


# ────────────────────────── the family map ──────────────────────────

def test_family_by_strategy_matches_the_live_lane_map():
    """Research may import strategies, never the reverse — so the map is copied here and
    pinned by this test. (The plan cites `strategies.fit_lane.LANE_MAP`; the live map
    actually lives in `core/strategy_fit_gate.py`, which is what is asserted.)"""
    from core.strategy_fit_gate import LANE_MAP

    assert FAMILY_BY_STRATEGY == LANE_MAP


# ────────────────────────── shadow side ──────────────────────────

def test_resolved_from_sim_carries_family_side_regime_and_r():
    out = resolved_from_sim([_sim_trade(), _sim_trade(side="short", regime="mixed", r_multiple=-1.0)], "orb")

    assert [(t.family, t.side, t.regime, t.r, t.source) for t in out] == [
        ("orb", "long", "risk_on", 2.0, "shadow"),
        ("orb", "short", "mixed", -1.0, "shadow"),
    ]
    assert out[0].exit_ts == "2026-09-01T11:00:00"


def test_resolved_from_sim_defaults_a_missing_regime_to_unknown():
    out = resolved_from_sim([_sim_trade(regime="")], "orb")
    assert out[0].regime == "unknown"


def test_resolved_from_sim_drops_a_non_finite_r():
    out = resolved_from_sim([_sim_trade(r_multiple=float("nan")), _sim_trade()], "orb")
    assert len(out) == 1


# ────────────────────────── live side ──────────────────────────

def test_resolved_from_live_maps_strategies_to_families_and_joins_the_regime():
    frames = _live_frames([
        ("AAA", "long", "orb_contraction_rvol", "2026-09-01T10:00:00", "2026-09-01T11:00:00", 1.5),
        ("BBB", "short", "mean_reversion_short", "2026-09-02T10:00:00", "2026-09-02T12:00:00", -1.0),
    ], decisions=[
        ("AAA", "2026-09-01T09:58:00", json.dumps({"regime": "risk_on", "family": "orb"})),
        ("BBB", "2026-09-02T09:59:00", json.dumps({"regime": "risk_off"})),
    ])
    out = resolved_from_live(frames[LIVE_TRADES_SQL], frames[LIVE_REGIME_SQL])

    assert [(t.family, t.side, t.regime, t.r, t.source) for t in out] == [
        ("orb", "long", "risk_on", 1.5, "live"),
        ("mean_rev_intraday", "short", "risk_off", -1.0, "live"),
    ]


def test_resolved_from_live_drops_null_r_and_unmapped_strategies():
    frames = _live_frames([
        ("AAA", "long", "orb_contraction_rvol", "2026-09-01T10:00:00", "2026-09-01T11:00:00", None),
        ("AAA", "long", "some_unlisted_strategy", "2026-09-01T10:00:00", "2026-09-01T11:00:00", 1.0),
        ("AAA", "long", "orb_contraction_rvol", "2026-09-03T10:00:00", "2026-09-03T11:00:00", 0.7),
    ])
    out = resolved_from_live(frames[LIVE_TRADES_SQL], frames[LIVE_REGIME_SQL])

    assert [(t.family, t.r) for t in out] == [("orb", 0.7)]


def test_resolved_from_live_regime_is_unknown_without_a_matching_decision():
    frames = _live_frames(
        [("AAA", "long", "orb_contraction_rvol", "2026-09-01T10:00:00", "2026-09-01T11:00:00", 1.0)],
        decisions=[("AAA", "2026-08-01T09:58:00", json.dumps({"regime": "risk_on"}))],  # far too old
    )
    out = resolved_from_live(frames[LIVE_TRADES_SQL], frames[LIVE_REGIME_SQL])

    assert [t.regime for t in out] == ["unknown"]


def test_resolved_from_live_ignores_a_garbage_regime_label_or_details_blob():
    frames = _live_frames(
        [("AAA", "long", "orb_contraction_rvol", "2026-09-01T10:00:00", "2026-09-01T11:00:00", 1.0),
         ("BBB", "long", "orb_contraction_rvol", "2026-09-01T10:00:00", "2026-09-01T11:00:00", 1.0)],
        decisions=[("AAA", "2026-09-01T09:58:00", json.dumps({"regime": "moon"})),
                   ("BBB", "2026-09-01T09:58:00", "{not json")],
    )
    out = resolved_from_live(frames[LIVE_TRADES_SQL], frames[LIVE_REGIME_SQL])

    assert [t.regime for t in out] == ["unknown", "unknown"]


@pytest.mark.parametrize("raw,expected", [("BUY", "long"), ("sell", "short"), ("LONG", "long"), ("Short", "short")])
def test_resolved_from_live_normalises_the_side(raw, expected):
    frames = _live_frames([("AAA", raw, "orb_contraction_rvol", "2026-09-01T10:00:00", "2026-09-01T11:00:00", 1.0)])
    out = resolved_from_live(frames[LIVE_TRADES_SQL], frames[LIVE_REGIME_SQL])
    assert [t.side for t in out] == [expected]


def test_resolved_from_live_skips_a_row_with_an_unreadable_exit_time(caplog):
    """r1 finding 11: an unsortable exit_time must be dropped with a warning, never kept as an
    empty string that sorts as the oldest trade in the window."""
    frames = _live_frames([
        ("AAA", "long", "orb_contraction_rvol", "2026-09-01T10:00:00", "not a timestamp", 1.0),
        ("AAA", "long", "orb_contraction_rvol", "2026-09-02T10:00:00", "2026-09-02T11:00:00", 0.5),
    ])
    with caplog.at_level("WARNING"):
        out = resolved_from_live(frames[LIVE_TRADES_SQL], frames[LIVE_REGIME_SQL])

    assert [t.r for t in out] == [0.5]
    assert any("exit_time" in r.message for r in caplog.records)


def test_resolved_from_sim_skips_a_trade_with_an_unreadable_exit_ts(caplog):
    with caplog.at_level("WARNING"):
        out = resolved_from_sim([_sim_trade(exit_ts="nonsense"), _sim_trade()], "orb")

    assert len(out) == 1 and out[0].exit_ts == "2026-09-01T11:00:00"


def test_resolved_from_live_drops_an_unreadable_side():
    frames = _live_frames([("AAA", "sideways", "orb_contraction_rvol", "2026-09-01T10:00:00",
                           "2026-09-01T11:00:00", 1.0)])
    assert resolved_from_live(frames[LIVE_TRADES_SQL], frames[LIVE_REGIME_SQL]) == []


def test_live_resolved_trades_uses_the_injected_reader():
    seen = []
    frames = _live_frames([("AAA", "long", "orb_contraction_rvol", "2026-09-01T10:00:00",
                           "2026-09-01T11:00:00", 1.0)])

    out = live_resolved_trades(_reader(frames, seen), now=NOW)

    assert [q for q, _ in seen] == [LIVE_TRADES_SQL, LIVE_REGIME_SQL]
    assert [t.family for t in out] == ["orb"]


def test_live_reads_are_bounded_to_the_lookback_window():
    """r1 finding 7: both reads are bounded (and parameterized), so a multi-year bot_trades
    table is never dragged through pandas in full."""
    seen = []
    frames = _live_frames([])
    live_resolved_trades(_reader(frames, seen), now=NOW)

    assert LIVE_LOOKBACK_DAYS == 120
    since = pd.Timestamp(NOW) - pd.Timedelta(days=LIVE_LOOKBACK_DAYS)
    for query, params in seen:
        assert ":since" in query
        assert pd.Timestamp(params["since"]) == since
    assert "exit_time >= :since" in LIVE_TRADES_SQL and "ts >= :since" in LIVE_REGIME_SQL


def test_live_resolved_trades_never_raises_on_a_failing_reader(caplog):
    def _boom(query, params=None):
        raise RuntimeError("no database here")

    with caplog.at_level("WARNING"):
        assert live_resolved_trades(_boom) == []
    assert any("playbook" in r.message.lower() or "live" in r.message.lower() for r in caplog.records)


def test_live_resolved_trades_with_no_reader_is_empty():
    assert live_resolved_trades(None) == []


# ────────────────────────── grouping and merge ──────────────────────────

def test_group_by_key_sorts_each_key_oldest_first():
    from core.desk.playbook import ResolvedTrade

    trades = [
        ResolvedTrade("orb", "long", "risk_on", 1.0, "2026-09-03T15:00:00", "live"),
        ResolvedTrade("orb", "long", "risk_on", 2.0, "2026-09-01T15:00:00", "shadow"),
        ResolvedTrade("orb", "short", "mixed", 3.0, "2026-09-02T15:00:00", "shadow"),
    ]
    grouped = group_by_key(trades)

    assert set(grouped) == {("orb", "long", "risk_on"), ("orb", "short", "mixed")}
    assert [t.r for t in grouped[("orb", "long", "risk_on")]] == [2.0, 1.0]


def test_resolved_trades_by_key_merges_the_shadow_ledger_and_live_trades(root, tmp_path):
    ledger_dir = tmp_path / "ledgers"
    _ledger(ledger_dir)
    frames = _live_frames([("AAA", "long", "orb_contraction_rvol", "2026-09-01T10:00:00",
                            "2026-09-01T11:00:00", 1.25)],
                          decisions=[("AAA", "2026-09-01T09:55:00", json.dumps({"regime": "risk_on"}))])

    grouped = resolved_trades_by_key(root, ["toy"], reader=_reader(frames), ledger_dir=ledger_dir,
                                     now=NOW)

    shadow_key = ("toy", "long", "unknown")
    assert shadow_key in grouped and len(grouped[shadow_key]) == 2
    assert all(t.source == "shadow" for t in grouped[shadow_key])
    assert [t.r for t in grouped[("orb", "long", "risk_on")]] == [1.25]


def test_resolved_trades_by_key_survives_a_family_that_cannot_be_resolved(root, tmp_path, caplog):
    """A missing ledger is simply no trades; a broken one must not take the whole run down."""
    ledger_dir = tmp_path / "ledgers"
    bad = ledger_dir / "shadow_fit_broken.ndjson"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text(json.dumps({"symbol": "AAA"}) + "\n")      # no ts/side/prices → resolve raises
    _ledger(ledger_dir)

    with caplog.at_level("WARNING"):
        grouped = resolved_trades_by_key(root, ["broken", "toy", "absent"], ledger_dir=ledger_dir)

    assert ("toy", "long", "unknown") in grouped
    assert not any(k[0] == "broken" for k in grouped)


def test_resolved_from_live_joins_tz_aware_decisions_to_naive_trade_times():
    """Postgres hands back tz-aware timestamps; comparing them to naive ones raises, which
    would silently cost the entire live tape."""
    trades = pd.DataFrame([{"symbol": "AAA", "side": "long", "strategy": "orb_contraction_rvol",
                            "entry_time": pd.Timestamp("2026-09-01 14:00:00"),
                            "exit_time": pd.Timestamp("2026-09-01 15:00:00"), "pnl_r": 1.0}])
    decisions = pd.DataFrame([{"symbol": "AAA", "ts": pd.Timestamp("2026-09-01 13:58:00", tz="UTC"),
                               "details_json": {"regime": "risk_off"}}])

    assert [t.regime for t in resolved_from_live(trades, decisions)] == ["risk_off"]


# ── entry-time regime for the shadow tape (fix r2, finding 1a) ──

def test_shadow_trades_are_stamped_with_the_entry_time_regime(root, tmp_path):
    """A 10:00 entry on a session that is +0.5% at 09:55 and closes -1% must be labelled from
    what had closed by 10:00 (risk_on), never from the session's close (risk_off)."""
    from tests.research.strategy_hunt.playbook_helpers import write_regime_context

    write_regime_context(root)
    ledger_dir = tmp_path / "ledgers"
    _ledger(ledger_dir)

    grouped = resolved_trades_by_key(root, ["toy"], ledger_dir=ledger_dir, now=NOW)

    assert ("toy", "long", "risk_off") not in grouped
    assert [t.regime for t in grouped[("toy", "long", "risk_on")]] == ["risk_on", "risk_on"]


def test_a_session_the_lookup_cannot_label_stays_unknown(root, tmp_path):
    """SPY has no bars for the entry session → `unknown`, never a guess."""
    from tests.research.strategy_hunt.playbook_helpers import spy_bars, write_regime_context

    write_regime_context(root, spy=spy_bars(sessions=("2026-07-06",)))
    ledger_dir = tmp_path / "ledgers"
    _ledger(ledger_dir)                       # entries on 07-06 (labelled) and 07-09 (no SPY)

    grouped = resolved_trades_by_key(root, ["toy"], ledger_dir=ledger_dir, now=NOW)

    assert [t.regime for t in grouped[("toy", "long", "risk_on")]] == ["risk_on"]
    assert [t.regime for t in grouped[("toy", "long", "unknown")]] == ["unknown"]


def test_no_spy_bars_at_all_leaves_every_shadow_trade_unknown(root, tmp_path):
    ledger_dir = tmp_path / "ledgers"
    _ledger(ledger_dir)

    grouped = resolved_trades_by_key(root, ["toy"], ledger_dir=ledger_dir, now=NOW)

    assert set(grouped) == {("toy", "long", "unknown")}


def test_regime_lookup_for_never_raises_and_is_built_once(root, monkeypatch):
    """The lookup is built once per updater run (not per trade) and a broken context tree is a
    warning plus `unknown` labels, never an exception out of the nightly job."""
    from research.strategy_hunt import playbook_update as PU
    from tests.research.strategy_hunt.playbook_helpers import write_regime_context

    write_regime_context(root)
    calls = []
    real = PU.regime_lookup_for
    monkeypatch.setattr(PU, "regime_lookup_for", lambda r: calls.append(r) or real(r))

    resolved_trades_by_key(root, ["toy", "orb", "dt_momentum"], now=NOW)
    assert len(calls) == 1

    monkeypatch.setattr(PU, "load_context", lambda root_: (_ for _ in ()).throw(OSError("nope")))
    assert PU.regime_lookup_for(root) is None


def test_resolved_from_sim_keeps_a_regime_the_resolver_already_stamped(root):
    """A resolver that labels its own trades wins: the lookup only fills in `unknown`."""
    out = resolved_from_sim([_sim_trade(regime="risk_off")], "orb",
                            regime_lookup=lambda ts: "risk_on")
    assert [t.regime for t in out] == ["risk_off"]


def test_resolved_from_sim_uses_the_lookup_for_unknown_trades():
    out = resolved_from_sim([_sim_trade(regime="unknown")], "orb", regime_lookup=lambda ts: "mixed")
    assert [t.regime for t in out] == ["mixed"]


def test_resolved_from_sim_survives_a_raising_lookup(caplog):
    def _boom(ts):
        raise ValueError("bad bars")

    with caplog.at_level("WARNING"):
        out = resolved_from_sim([_sim_trade(regime="unknown")], "orb", regime_lookup=_boom)
    assert [t.regime for t in out] == ["unknown"]
