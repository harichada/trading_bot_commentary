"""Batch-2 Task A: the selection baseline replays the real trade schedule on random peers.

The path-shuffle control keeps every session's open-to-close return, so for a multi-day hold it
only ever tests intraday *timing*. The selection control keeps the real *schedule* -- same entry
timestamp, same side, same relative stop/target, same hold -- and reassigns each trade to a seeded
random peer symbol, which is the control that tests stock *selection*.
"""
import json

import numpy as np
import pandas as pd
import pytest

from research.strategy_hunt.models import TRADE_COLUMNS

SESSION_BARS = 12
BASE_PX = {"AAA": 100.0, "BBB": 50.0, "CCC": 200.0}
SESSIONS = 8
TOY_MULTI = "toy_multi"


def _bars(base: float, days: pd.DatetimeIndex) -> pd.DataFrame:
    frames = []
    for k, d in enumerate(days):
        idx = pd.date_range(f"{d.date()} 09:30", periods=SESSION_BARS, freq="5min", name="ts")
        close = base * (1.0 + 0.001 * (np.arange(SESSION_BARS) + SESSION_BARS * k))
        frames.append(pd.DataFrame({"open": close * 0.999, "high": close * 1.02, "low": close * 0.98,
                                    "close": close, "volume": 10_000}, index=idx))
    return pd.concat(frames)


def _trade(symbol: str, entry_ts: pd.Timestamp, side: str = "long", *, family: str = TOY_MULTI,
           rel_stop: float = 0.01, rel_target: float = 0.03, entry_px: float = 100.0,
           params: dict | None = None, entry_style: str | None = None) -> dict:
    sgn = 1.0 if side == "long" else -1.0
    row = {"symbol": symbol, "family": family, "side": side, "entry_ts": entry_ts,
           "exit_ts": entry_ts + pd.Timedelta(days=1), "entry_px": entry_px,
           "stop_px": entry_px * (1 - sgn * rel_stop), "target_px": entry_px * (1 + sgn * rel_target),
           "exit_px": entry_px, "entry_fill": entry_px, "exit_fill": entry_px, "r_multiple": 0.5,
           "pnl_pct": 0.5, "cost_pct": 0.05, "exit_reason": "time_stop", "hold_bars": 20,
           "hold_sessions": 2, "tag": "toy", "regime": "mixed", "window_id": 3,
           "params_json": json.dumps(params if params is not None else {"time_stop_days": 2})}
    if entry_style is not None:
        row["entry_style"] = entry_style
    return row


def _write_trades(root, symbol: str, rows: list[dict], variant: str | None = None) -> None:
    from research.strategy_hunt.paths import trades_path

    tp = trades_path(root, symbol, variant)
    tp.parent.mkdir(parents=True, exist_ok=True)
    cols = list(dict.fromkeys([*TRADE_COLUMNS, *(k for r in rows for k in r)]))
    pd.DataFrame(rows).reindex(columns=cols).to_parquet(tp, index=False)


@pytest.fixture
def days() -> pd.DatetimeIndex:
    return pd.bdate_range("2024-01-02", periods=SESSIONS)


@pytest.fixture
def toy_family(monkeypatch):
    from strategies.families import registry
    from strategies.families.common import Family

    fam = Family(TOY_MULTI, lambda *a, **kw: [], {"time_stop_days": [2]}, "multiday", ("long", "short"))
    monkeypatch.setitem(registry.FAMILIES, TOY_MULTI, fam)
    return fam


@pytest.fixture
def root(tmp_path, days, toy_family):
    """3-symbol root with a real trades/ tape: two AAA longs, one BBB long, one CCC short."""
    from research.strategy_hunt.data.bars import _write, bars5_path, daily_path, to_daily

    for sym, base in BASE_PX.items():
        b = _bars(base, days)
        _write(bars5_path(tmp_path, sym), b, "ts")
        _write(daily_path(tmp_path, sym), to_daily(b), "date")
    _write_trades(tmp_path, "AAA", [_trade("AAA", pd.Timestamp(f"{days[2].date()} 09:45")),
                                    _trade("AAA", pd.Timestamp(f"{days[4].date()} 10:05"))])
    _write_trades(tmp_path, "BBB", [_trade("BBB", pd.Timestamp(f"{days[3].date()} 10:00"), entry_px=50.0)])
    _write_trades(tmp_path, "CCC", [_trade("CCC", pd.Timestamp(f"{days[5].date()} 10:10"), side="short",
                                            entry_px=200.0, rel_stop=0.02, rel_target=0.04)])
    return tmp_path


def _plans(root):
    from research.strategy_hunt.selection_baseline import build_plans, real_trades, selection_universe

    frame = real_trades(root)
    universe = selection_universe(root)
    return build_plans(frame, root, universe, seed=7)


# ──────────────────────────────  the draw  ──────────────────────────────


def test_every_trade_is_replayed_on_a_peer_that_is_not_its_own_symbol(root):
    plans, dropped = _plans(root)

    assert len(plans) == 4 and dropped == {}
    assert all(p.peer != p.source_symbol for p in plans)
    assert all(p.peer in BASE_PX for p in plans)
    # the schedule is kept: the replayed entry timestamps are exactly the real ones
    assert sorted(p.entry_ts for p in plans) == sorted(pd.Timestamp(t) for t in real_entry_stamps(root))


def real_entry_stamps(root):
    from research.strategy_hunt.selection_baseline import real_trades

    return list(pd.to_datetime(real_trades(root)["entry_ts"]))


def test_the_draw_is_seeded_by_seed_symbol_and_timestamp(root):
    from research.strategy_hunt.selection_baseline import draw_peer

    peers = ("BBB", "CCC")
    ts = pd.Timestamp("2024-01-04 09:45")
    always = {p: True for p in peers}
    first = draw_peer(7, "AAA", ts, peers, always.__getitem__)
    assert first == draw_peer(7, "AAA", ts, peers, always.__getitem__)
    # a different seed / symbol / timestamp is a different draw stream
    assert {draw_peer(s, "AAA", ts, peers, always.__getitem__) for s in range(40)} == set(peers)
    assert {draw_peer(7, "AAA", ts + pd.Timedelta(minutes=5 * k), peers, always.__getitem__)
            for k in range(40)} == set(peers)


def test_a_trade_whose_peers_all_lack_the_bar_is_dropped_after_three_redraws(root, days):
    from research.strategy_hunt.selection_baseline import draw_peer

    assert draw_peer(7, "AAA", pd.Timestamp("2024-01-04 09:45"), ("BBB", "CCC"), lambda p: False) is None


def test_trades_at_a_timestamp_no_peer_has_are_dropped_and_counted(root, days):
    """AAA gets an extra session BBB and CCC do not have; a trade in it has no possible peer."""
    from research.strategy_hunt.data.bars import _write, bars5_path, load_bars5
    from research.strategy_hunt.selection_baseline import replay_selection

    extra_day = pd.Timestamp("2024-01-15")
    b = pd.concat([load_bars5(root, "AAA"), _bars(BASE_PX["AAA"], pd.DatetimeIndex([extra_day]))])
    _write(bars5_path(root, "AAA"), b, "ts")
    lonely = pd.Timestamp(f"{extra_day.date()} 09:45")
    _write_trades(root, "AAA", [_trade("AAA", pd.Timestamp(f"{days[2].date()} 09:45")),
                                _trade("AAA", lonely)])

    summary = replay_selection(root, seed=7)

    assert summary["trades"] == 4
    assert summary["selection_dropped"] == 1
    assert summary["replayed"] == 3
    assert summary["dropped"]["no_peer_bar"] == 1


# ──────────────────────────────  the replayed entry  ──────────────────────────────


def test_the_relative_stop_and_target_are_preserved_to_1e_9(root):
    from research.strategy_hunt.data.splits import load_adjusted
    from research.strategy_hunt.selection_baseline import plan_entry

    plans, _ = _plans(root)
    assert plans
    for p in plans:
        bars5, _daily = load_adjusted(root, p.peer)
        entry = plan_entry(p, bars5)
        sgn = 1.0 if p.side == "long" else -1.0
        fill = float(bars5["open"].loc[p.entry_ts])
        assert entry is not None
        assert entry.symbol == p.peer and entry.entry_ts == p.entry_ts
        assert entry.entry_px == pytest.approx(fill, abs=1e-12)
        assert (fill - entry.stop_px) * sgn / fill == pytest.approx(p.rel_stop, abs=1e-9)
        assert (entry.target_px - fill) * sgn / fill == pytest.approx(p.rel_target, abs=1e-9)
        assert entry.tag == f"sel:{p.family}"
        assert entry.hold_kind == "multiday" and entry.time_stop == 2


def test_the_signal_bar_is_one_bar_before_so_the_fill_is_the_peers_bar_at_entry_ts(root):
    from research.strategy_hunt.data.splits import load_adjusted
    from research.strategy_hunt.selection_baseline import plan_entry

    plans, _ = _plans(root)
    p = plans[0]
    bars5, _daily = load_adjusted(root, p.peer)
    entry = plan_entry(p, bars5)
    assert entry.entry_ts == p.entry_ts
    assert entry.entry_px == pytest.approx(float(bars5["open"].loc[p.entry_ts]))


def test_entry_style_defaults_to_market_and_is_copied_when_present(root, days):
    from research.strategy_hunt.selection_baseline import build_plans, real_trades, selection_universe

    plans, _ = _plans(root)
    assert {p.entry_style for p in plans} == {"market"}       # column absent -> market

    _write_trades(root, "AAA", [_trade("AAA", pd.Timestamp(f"{days[2].date()} 09:45"),
                                       entry_style="limit_open")])
    frame = real_trades(root)
    styled, _ = build_plans(frame, root, selection_universe(root), seed=7)
    assert {p.entry_style for p in styled if p.source_symbol == "AAA"} == {"limit_open"}


def test_the_time_stop_comes_from_params_json_for_a_multiday_family(root, days):
    from research.strategy_hunt.selection_baseline import build_plans, real_trades, selection_universe

    _write_trades(root, "AAA", [_trade("AAA", pd.Timestamp(f"{days[2].date()} 09:45"),
                                       params={"time_stop_days": 5})])
    plans, _ = build_plans(real_trades(root), root, selection_universe(root), seed=7)
    assert {p.time_stop for p in plans if p.source_symbol == "AAA"} == {5}


def test_the_time_stop_falls_back_to_the_realized_hold_when_params_carry_none(root, days):
    from research.strategy_hunt.selection_baseline import build_plans, real_trades, selection_universe

    _write_trades(root, "AAA", [_trade("AAA", pd.Timestamp(f"{days[2].date()} 09:45"), params={})])
    plans, _ = build_plans(real_trades(root), root, selection_universe(root), seed=7)
    # the real trade's hold_sessions (2) is the honest horizon left on the record
    assert {p.time_stop for p in plans if p.source_symbol == "AAA"} == {2}


# ──────────────────────────────  the written control tape  ──────────────────────────────


def test_replay_writes_trades_selection_with_the_trade_columns_and_the_peer_as_symbol(root):
    from research.strategy_hunt.paths import trades_dir
    from research.strategy_hunt.selection_baseline import replay_selection

    summary = replay_selection(root, seed=7)

    files = sorted(trades_dir(root, "selection").glob("*.parquet"))
    assert files, "the selection control wrote no trades"
    frames = [pd.read_parquet(f) for f in files]
    out = pd.concat(frames, ignore_index=True)
    assert list(out.columns) == list(TRADE_COLUMNS)
    assert len(out) == summary["replayed"] == 4
    for f, df in zip(files, frames):
        assert set(df["symbol"]) == {f.stem}
    assert set(out["family"]) == {TOY_MULTI}
    assert set(out["tag"]) == {f"sel:{TOY_MULTI}"}
    # the relative risk of each source trade survived onto its peer
    rel = ((out["entry_px"] - out["stop_px"]).abs() / out["entry_px"]).round(9)
    assert set(rel) <= {0.01, 0.02}


def test_replay_is_deterministic_for_a_seed(root):
    from research.strategy_hunt.paths import trades_dir
    from research.strategy_hunt.selection_baseline import replay_selection

    def _tape() -> dict:
        return {f.stem: pd.read_parquet(f) for f in sorted(trades_dir(root, "selection").glob("*.parquet"))}

    first_summary = replay_selection(root, seed=7)
    first = _tape()
    again_summary = replay_selection(root, seed=7)
    again = _tape()

    assert first_summary == again_summary
    assert set(first) == set(again)
    for sym, df in first.items():
        pd.testing.assert_frame_equal(df, again[sym])


def test_a_stale_peer_file_from_an_earlier_seed_is_not_left_behind(root):
    from research.strategy_hunt.paths import trades_dir, trades_path
    from research.strategy_hunt.selection_baseline import replay_selection

    ghost = trades_path(root, "ZZZ", "selection")
    ghost.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=list(TRADE_COLUMNS)).to_parquet(ghost, index=False)

    replay_selection(root, seed=7)

    assert not ghost.exists()
    assert {f.stem for f in trades_dir(root, "selection").glob("*.parquet")} <= set(BASE_PX)


def test_replay_with_no_real_trades_is_an_empty_summary_not_an_error(tmp_path, days, toy_family):
    from research.strategy_hunt.data.bars import _write, bars5_path
    from research.strategy_hunt.selection_baseline import replay_selection

    _write(bars5_path(tmp_path, "AAA"), _bars(100.0, days), "ts")

    summary = replay_selection(tmp_path, seed=7)

    assert summary["trades"] == 0 and summary["replayed"] == 0 and summary["selection_dropped"] == 0
    assert summary["errors"] == []


def test_a_single_symbol_universe_has_no_peers_and_drops_everything(tmp_path, days, toy_family):
    from research.strategy_hunt.data.bars import _write, bars5_path, daily_path, to_daily
    from research.strategy_hunt.selection_baseline import replay_selection

    b = _bars(100.0, days)
    _write(bars5_path(tmp_path, "AAA"), b, "ts")
    _write(daily_path(tmp_path, "AAA"), to_daily(b), "date")
    _write_trades(tmp_path, "AAA", [_trade("AAA", pd.Timestamp(f"{days[2].date()} 09:45"))])

    summary = replay_selection(tmp_path, seed=7)

    assert summary["trades"] == 1 and summary["replayed"] == 0
    assert summary["selection_dropped"] == 1


def test_an_unusable_trade_row_is_counted_not_fatal(root, days):
    """A corrupt row (here a NaN `window_id`) is dropped and counted; the rest of the tape replays."""
    from research.strategy_hunt.selection_baseline import DROP_BAD_RECORD, replay_selection

    good = _trade("AAA", pd.Timestamp(f"{days[2].date()} 09:45"))
    _write_trades(root, "AAA", [good, {**good, "window_id": float("nan")}])

    summary = replay_selection(root, seed=7)

    assert summary["dropped"][DROP_BAD_RECORD] == 1
    assert summary["replayed"] == 3 and summary["trades"] == 4


def test_a_trade_with_no_stop_distance_is_counted_as_bad_geometry(root, days):
    from research.strategy_hunt.selection_baseline import DROP_BAD_GEOMETRY, replay_selection

    flat = _trade("AAA", pd.Timestamp(f"{days[2].date()} 09:45"))
    flat["stop_px"] = flat["entry_px"]
    _write_trades(root, "AAA", [flat])

    summary = replay_selection(root, seed=7)

    assert summary["dropped"][DROP_BAD_GEOMETRY] == 1


# ─────────────────────────  Task F item 4: the horizon comes off the record  ─────────────────────────


def test_the_time_stop_is_read_from_the_trade_when_the_record_carries_it(root, days):
    """Batch-2 Task F item 4. Recovering the horizon from `params_json` could only ever get it
    right when the family exposed it as a grid axis, and the realized-hold fallback shortened the
    control's horizon for every trade that exited early -- which is most of them. The trade now
    records `Entry.time_stop` itself, so the replay reproduces the schedule it is controlling for
    even when the source trade stopped out on its second bar."""
    from research.strategy_hunt.selection_baseline import build_plans, real_trades, selection_universe

    early = _trade("AAA", pd.Timestamp(f"{days[2].date()} 09:45"), params={"time_stop_days": 2})
    early.update({"time_stop": 9, "hold_sessions": 1, "hold_bars": 2, "exit_reason": "stop"})
    _write_trades(root, "AAA", [early])
    plans, _ = build_plans(real_trades(root), root, selection_universe(root), seed=7)
    assert {p.time_stop for p in plans if p.source_symbol == "AAA"} == {9}


def test_the_time_stop_still_falls_back_for_a_tape_without_the_column(root, days):
    """An older `trades/` tape has no `time_stop` column at all (NaN after the reindex), so the
    params/realized-hold recovery has to stay as the fallback -- a pre-Task-F root must remain
    replayable."""
    from research.strategy_hunt.selection_baseline import build_plans, real_trades, selection_universe

    _write_trades(root, "AAA", [_trade("AAA", pd.Timestamp(f"{days[2].date()} 09:45"),
                                       params={"time_stop_days": 5})])
    plans, _ = build_plans(real_trades(root), root, selection_universe(root), seed=7)
    assert {p.time_stop for p in plans if p.source_symbol == "AAA"} == {5}


def test_time_stop_for_prefers_the_trades_own_horizon_over_every_recovery():
    from research.strategy_hunt.selection_baseline import time_stop_for

    assert time_stop_for("multiday", {"time_stop_days": 2}, 20, 1, trade_time_stop=9) == 9
    assert time_stop_for("intraday", {"time_stop_bars": 30}, 4, 1, trade_time_stop=66) == 66
    # 0 / NaN / a nonsense value means "the record does not know", not "zero bars"
    assert time_stop_for("multiday", {"time_stop_days": 2}, 20, 1, trade_time_stop=0) == 2
    assert time_stop_for("multiday", {}, 20, 3, trade_time_stop=float("nan")) == 3


# ──────────────────────────────  fix round 1  ──────────────────────────────


def test_the_replayed_entry_keeps_a_limit_open_style(root, days):
    """Task B's `entry_style` reaches the peer's `Entry`, not just the plan."""
    from research.strategy_hunt.data.splits import load_adjusted
    from research.strategy_hunt.selection_baseline import plan_entry

    _write_trades(root, "AAA", [_trade("AAA", pd.Timestamp(f"{days[2].date()} 09:45"),
                                       entry_style="limit_open")])
    plans, _ = _plans(root)
    styled = [p for p in plans if p.source_symbol == "AAA"]
    assert styled
    for p in styled:
        entry = plan_entry(p, load_adjusted(root, p.peer)[0])
        assert entry is not None and entry.entry_style == "limit_open"
    for p in (p for p in plans if p.source_symbol != "AAA"):
        assert plan_entry(p, load_adjusted(root, p.peer)[0]).entry_style == "market"


def test_plan_entry_is_none_on_the_peers_first_bar(root, days):
    """There is no signal bar before the frame's first bar, so the fill cannot be built."""
    import dataclasses

    from research.strategy_hunt.data.splits import load_adjusted
    from research.strategy_hunt.selection_baseline import plan_entry

    plans, _ = _plans(root)
    bars5, _daily = load_adjusted(root, plans[0].peer)
    assert plan_entry(dataclasses.replace(plans[0], entry_ts=bars5.index[0]), bars5) is None
    # ... and a timestamp the peer never traded is None too
    assert plan_entry(dataclasses.replace(plans[0], entry_ts=pd.Timestamp("1999-01-04 09:35")), bars5) is None


def test_plan_entry_is_none_for_an_intraday_plan_on_a_session_open(root, days):
    """`make_entry` refuses an intraday entry whose signal bar is in the previous session."""
    import dataclasses

    from research.strategy_hunt.data.splits import load_adjusted
    from research.strategy_hunt.selection_baseline import plan_entry

    plans, _ = _plans(root)
    bars5, _daily = load_adjusted(root, plans[0].peer)
    session_open = pd.Timestamp(f"{days[3].date()} 09:30")
    assert session_open in bars5.index
    intraday = dataclasses.replace(plans[0], entry_ts=session_open, hold_kind="intraday", time_stop=6)
    assert plan_entry(intraday, bars5) is None
    # the same plan one bar later is fine: its signal bar is in the same session
    assert plan_entry(dataclasses.replace(intraday, entry_ts=bars5.index[bars5.index.get_loc(session_open) + 1]),
                      bars5) is not None


def test_a_tz_aware_entry_ts_column_is_refused_loudly_and_counted(root, days, caplog):
    """Bars are tz-naive; a tz-aware schedule can match no bar, so say so once instead of
    dropping every trade as an unusable record."""
    import logging

    from research.strategy_hunt.selection_baseline import DROP_TZ_AWARE, replay_selection

    for sym in ("AAA", "BBB", "CCC"):
        tp = root / "trades" / f"{sym}.parquet"
        df = pd.read_parquet(tp)
        df["entry_ts"] = pd.to_datetime(df["entry_ts"]).dt.tz_localize("America/New_York")
        df.to_parquet(tp, index=False)

    with caplog.at_level(logging.WARNING, logger="strategy_hunt.selection_baseline"):
        summary = replay_selection(root, seed=7)

    assert summary["replayed"] == 0
    assert summary["dropped"][DROP_TZ_AWARE] == summary["trades"] == 4
    assert any("tz-aware" in m for m in caplog.messages)


def test_a_peer_with_no_replayable_entry_writes_no_file(root, days):
    """A zero-row Parquet is not a control tape: it would only make the scorer's coverage
    numbers look better than the run was."""
    from research.strategy_hunt.data.splits import load_adjusted
    from research.strategy_hunt.paths import trades_path
    from research.strategy_hunt.run import load_context
    from research.strategy_hunt.selection_baseline import replay_peer
    import dataclasses

    plans, _ = _plans(root)
    peer = plans[0].peer
    bars5, _daily = load_adjusted(root, peer)
    doomed = [dataclasses.replace(plans[0], peer=peer, entry_ts=bars5.index[0])]

    rec = replay_peer(peer, doomed, root, load_context(root), 6000.0, 1.0, None)

    assert rec["replayed"] == 0 and rec["dropped"] == 1
    assert not trades_path(root, peer, "selection").exists()


def test_replay_reports_per_group_pairing_counts(root):
    """`selection_groups` is what makes a drop visible: the control's n per pooled group next to
    the real tape's, keyed (family, side, regime) exactly as `pooled_rows` groups."""
    from research.strategy_hunt.selection_baseline import replay_selection

    summary = replay_selection(root, seed=7)

    groups = summary["selection_groups"]
    assert set(groups) == {"toy_multi|long|mixed", "toy_multi|short|mixed"}
    assert groups["toy_multi|long|mixed"] == {"real": 3, "replayed": 3, "dropped": 0, "replayed_frac": 1.0}
    assert groups["toy_multi|short|mixed"]["real"] == 1
    assert summary["underreplayed_groups"] == []


def test_a_dropped_trade_shows_up_as_an_underreplayed_group(root, days):
    from research.strategy_hunt.data.bars import _write, bars5_path, load_bars5
    from research.strategy_hunt.selection_baseline import replay_selection

    extra_day = pd.Timestamp("2024-01-15")
    b = pd.concat([load_bars5(root, "AAA"), _bars(BASE_PX["AAA"], pd.DatetimeIndex([extra_day]))])
    _write(bars5_path(root, "AAA"), b, "ts")
    # five of the six long trades can be replayed; the sixth has no peer with that bar
    _write_trades(root, "AAA", [_trade("AAA", pd.Timestamp(f"{days[k].date()} 09:45")) for k in (1, 2, 3, 4)]
                  + [_trade("AAA", pd.Timestamp(f"{extra_day.date()} 09:45"))])

    summary = replay_selection(root, seed=7)

    long_group = summary["selection_groups"]["toy_multi|long|mixed"]
    assert long_group["real"] == 6 and long_group["replayed"] == 5
    assert long_group["replayed_frac"] == pytest.approx(5 / 6)
    assert summary["underreplayed_groups"] == ["toy_multi|long|mixed"]
