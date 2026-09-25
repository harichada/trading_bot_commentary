"""v-playbook-desk-b2-2026-09-25 (Task 3): the nightly playbook updater.

Offline: the shadow side runs the real `shadow_resolver.resolve` against a tmp
root of generated bars, and the live side gets its `bot_trades` /
`bot_decisions` frames from an injected SQL reader callable (no Postgres).
"""
import json

import numpy as np
import pandas as pd
import pytest

from research.strategy_hunt.playbook_update import (
    FAMILY_BY_STRATEGY,
    LIVE_LOOKBACK_DAYS,
    LIVE_REGIME_SQL,
    LIVE_TRADES_SQL,
    RECENT_DAYS,
    best_verdict,
    group_by_key,
    live_resolved_trades,
    resolved_from_live,
    resolved_from_sim,
    resolved_trades_by_key,
    update_playbook,
)

NOW = "2026-09-25T20:00:00"


# ────────────────────────── fixtures ──────────────────────────

def _bars(start="2026-07-01", sessions=30, seed=0):
    rng = np.random.default_rng(seed)
    frames, px = [], 100.0
    for d in pd.bdate_range(start, periods=sessions):
        idx = pd.date_range(f"{d.date()} 09:30", periods=78, freq="5min", name="ts")
        close = px + rng.normal(0, 0.05, 78).cumsum()
        frames.append(pd.DataFrame({"open": close - 0.02, "high": close + 0.3, "low": close - 0.3,
                                    "close": close, "volume": 1000}, index=idx))
        px = float(close[-1])
    return pd.concat(frames)


@pytest.fixture
def root(tmp_path):
    from research.strategy_hunt.data.bars import _write, bars5_path, daily_path, to_daily

    b = _bars()
    _write(bars5_path(tmp_path, "AAA"), b, "ts")
    _write(daily_path(tmp_path, "AAA"), to_daily(b), "date")
    return tmp_path


def _ledger(ledger_dir, family="toy", timestamps=("2026-07-06T10:00:00", "2026-07-09T10:00:00")):
    from research.strategy_hunt.shadow_resolver import default_ledger_path

    p = default_ledger_path(family, ledger_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps({
        "ts": ts, "symbol": "AAA", "side": "long", "entry_px": 100.0, "stop_px": 99.0,
        "target_px": 100.5, "time_stop": 20, "hold_kind": "intraday", "tag": "t",
        "params": {"w": 1.0}, "verdict": "PASS", "score": 0.5,
    }) for ts in timestamps) + "\n")
    return p


def _sim_trade(**kw):
    from research.strategy_hunt.models import SimTrade

    base = dict(symbol="AAA", family="orb", side="long", entry_ts=pd.Timestamp("2026-09-01 10:00"),
                exit_ts=pd.Timestamp("2026-09-01 11:00"), entry_px=100.0, stop_px=99.0, target_px=102.0,
                exit_px=102.0, entry_fill=100.0, exit_fill=102.0, r_multiple=2.0, pnl_pct=0.02,
                cost_pct=0.0005, exit_reason="target", hold_bars=12, hold_sessions=1, tag="t",
                regime="risk_on", window_id=-1, params_json="{}")
    base.update(kw)
    return SimTrade(**base)


def _live_frames(rows, decisions=()):
    trades = pd.DataFrame(list(rows), columns=["symbol", "side", "strategy", "entry_time", "exit_time", "pnl_r"])
    dec = pd.DataFrame(list(decisions), columns=["symbol", "ts", "details_json"])
    return {LIVE_TRADES_SQL: trades, LIVE_REGIME_SQL: dec}


def _reader(frames, seen=None):
    """A `reader(query, params)` over pre-built frames (the injected SQL reader contract)."""
    def _read(query, params=None):
        if seen is not None:
            seen.append((query, params))
        return frames[query]
    return _read


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


# ────────────────────────── snapshot verdicts ──────────────────────────

def _snapshot(pooled=None, regime=None, rows=None):
    return {"computed_at": NOW, "floors": {}, "rows": rows or {}, "regime_rows": regime or {},
            "pooled_rows": pooled or {}, "baseline_pooled_rows": {}}


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


def _live_row(day_offset, r, strategy="orb_contraction_rvol", hour=10, symbol="AAA", side="long"):
    """One bot_trades row `day_offset` days before NOW (entry and exit on the same day)."""
    day = (pd.Timestamp(NOW) - pd.Timedelta(days=day_offset)).normalize()
    return (symbol, side, strategy, (day + pd.Timedelta(hours=hour - 1)).isoformat(),
            (day + pd.Timedelta(hours=hour)).isoformat(), r)


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


def _seed(desk, *entries):
    from core.desk.playbook import Playbook, playbook_path, save_playbook

    pb = Playbook(entries={}, updated_at="2026-06-01T00:00:00")
    for e in entries:
        pb = pb.with_entry(e)
    save_playbook(pb, playbook_path(desk))


def _prev_entry(**kw):
    from core.desk.playbook import PlaybookEntry

    base = dict(family="news_catalyst", side="short", regime="mixed", status="suspended",
                size_mult=0.0, exp_r_recent=-0.4, t_stat_recent=-2.5, n_recent=20,
                updated_at="2026-06-01T00:00:00", reason="suspended_negative_expectancy")
    base.update(kw)
    return PlaybookEntry(**base)


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


def test_shadow_and_live_trades_land_in_one_key_and_both_count(root, tmp_path):
    """r1 finding 10: a resolved ledger trade and a live `fit:toy` bot_trades row describe the
    same setup, so they must share a (family, side, regime) key and one sample."""
    desk = tmp_path / "desk"
    ledger_dir = tmp_path / "ledgers"
    _ledger(ledger_dir)                                      # two shadow trades, family "toy"
    rows = [_live_row(4, 0.9, strategy="fit:toy")]            # one live trade, same family/side

    pb = update_playbook(root, _snapshot(), NOW, desk_dir=desk, families=["toy"],
                         reader=_reader(_live_frames(rows)), ledger_dir=ledger_dir)

    grouped = resolved_trades_by_key(root, ["toy"], reader=_reader(_live_frames(rows)),
                                     ledger_dir=ledger_dir, now=NOW)
    key = ("toy", "long", "unknown")
    assert sorted(t.source for t in grouped[key]) == ["live", "shadow", "shadow"]
    assert pb.get(*key).n_recent == 3


def test_update_playbook_never_writes_outside_the_desk_dir(root, tmp_path):
    desk = tmp_path / "nested" / "desk"
    pb = update_playbook(root, _snapshot(), NOW, desk_dir=desk, families=[], reader=None)

    assert (desk / "playbook.json").exists()
    assert json.loads((desk / "playbook.json").read_text())["updated_at"] == NOW
    assert pb.entries == {}


def test_resolved_from_live_joins_tz_aware_decisions_to_naive_trade_times():
    """Postgres hands back tz-aware timestamps; comparing them to naive ones raises, which
    would silently cost the entire live tape."""
    trades = pd.DataFrame([{"symbol": "AAA", "side": "long", "strategy": "orb_contraction_rvol",
                            "entry_time": pd.Timestamp("2026-09-01 14:00:00"),
                            "exit_time": pd.Timestamp("2026-09-01 15:00:00"), "pnl_r": 1.0}])
    decisions = pd.DataFrame([{"symbol": "AAA", "ts": pd.Timestamp("2026-09-01 13:58:00", tz="UTC"),
                               "details_json": {"regime": "risk_off"}}])

    assert [t.regime for t in resolved_from_live(trades, decisions)] == ["risk_off"]
