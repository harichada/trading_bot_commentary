import json

import numpy as np
import pandas as pd
import pytest


def _bars(start="2024-01-02", sessions=30, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, periods=sessions)
    frames, px = [], 100.0
    for d in days:
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
    return tmp_path, b


def _ledger_record(ts, side="long", entry_px=100.0, stop_px=99.0, target_px=102.0, symbol="AAA"):
    return {"ts": ts, "symbol": symbol, "side": side, "entry_px": entry_px, "stop_px": stop_px,
            "target_px": target_px, "time_stop": 20, "hold_kind": "intraday", "tag": "test",
            "params": {"w": 1.0}, "verdict": "PASS", "score": 0.5}


def test_read_ledger_skips_malformed_line_and_counts_it(tmp_path, caplog):
    from research.strategy_hunt.shadow_resolver import read_ledger

    p = tmp_path / "shadow_fit_toy.ndjson"
    lines = [
        json.dumps(_ledger_record("2024-01-05T10:00:00")),
        "{this is not valid json",
        json.dumps(_ledger_record("2024-01-05T10:02:00")),
    ]
    p.write_text("\n".join(lines) + "\n")

    with caplog.at_level("WARNING"):
        records = read_ledger(p)

    assert len(records) == 2
    assert any("1" in r.message and "skip" in r.message.lower() for r in caplog.records)


def test_ledger_entries_builds_entry_objects():
    from research.strategy_hunt.shadow_resolver import ledger_entries

    records = [_ledger_record("2024-01-05T10:00:00")]
    entries = ledger_entries(records)

    assert len(entries) == 1
    e = entries[0]
    assert e.symbol == "AAA" and e.side == "long" and e.entry_px == 100.0
    assert e.entry_ts == pd.Timestamp("2024-01-05T10:00:00")


def test_resolve_snaps_off_grid_entry_and_keeps_positive_hold_bars(root):
    from research.strategy_hunt.shadow_resolver import resolve

    tmp_path, bars = root
    on_grid_ts = pd.Timestamp("2024-01-05 10:00:00")
    off_grid_ts = pd.Timestamp("2024-01-05 10:02:00")
    next_bar_ts = pd.Timestamp("2024-01-05 10:05:00")
    assert on_grid_ts in bars.index
    assert off_grid_ts not in bars.index
    assert next_bar_ts in bars.index

    records = [_ledger_record(on_grid_ts.isoformat()), _ledger_record(off_grid_ts.isoformat())]
    trades = resolve(records, tmp_path, "toy", cost_mult=1.0)

    assert len(trades) == 2
    assert all(t.hold_bars > 0 for t in trades)
    snapped = [t for t in trades if t.entry_ts == next_bar_ts]
    assert len(snapped) == 1


def test_resolve_drops_entries_with_no_bar_at_or_after_ts(root):
    from research.strategy_hunt.shadow_resolver import resolve

    tmp_path, bars = root
    far_future_ts = bars.index[-1] + pd.Timedelta(days=30)
    records = [_ledger_record(far_future_ts.isoformat())]
    trades = resolve(records, tmp_path, "toy", cost_mult=1.0)
    assert trades == []


def test_stage_a_scorecard_returns_dict_with_verdict_and_side_metrics(root):
    from research.strategy_hunt.shadow_resolver import resolve, stage_a_scorecard

    tmp_path, _bars5 = root
    records = [_ledger_record("2024-01-05T10:00:00"), _ledger_record("2024-01-08T10:00:00")]
    trades = resolve(records, tmp_path, "toy")
    card = stage_a_scorecard(trades, "toy")

    assert card["family"] == "toy"
    assert card["verdict"] in ("GO", "NO_GO")
    assert set(card["long"]) >= {"n", "pf", "wr", "exp_r", "max_dd_pct", "verdict"}
    assert card["long"]["n"] == 2
    assert card["short"]["n"] == 0


def test_write_scorecard_creates_file(tmp_path, monkeypatch):
    from research.strategy_hunt import shadow_resolver as SR

    monkeypatch.setattr(SR, "REPORTS_DIR", tmp_path / "reports")
    card = {
        "family": "toy", "verdict": "NO_GO",
        "long": {"n": 0, "sessions": 0, "pf": 0.0, "wr": 0.0, "exp_r": 0.0, "max_dd_pct": 0.0,
                 "max_losing_day_r": 0.0, "verdict": "FAIL"},
        "short": {"n": 0, "sessions": 0, "pf": 0.0, "wr": 0.0, "exp_r": 0.0, "max_dd_pct": 0.0,
                  "max_losing_day_r": 0.0, "verdict": "FAIL"},
    }
    p = SR.write_scorecard(tmp_path, "toy", card, "2026-09-25T00:00:00")
    assert p.exists() and p.name == "scorecard-toy-2026-09-25.md"
    assert "NO_GO" in p.read_text()


# ──────────────────────────────────────────────────────────────────────────────
# Batch-2 Task F item 3 — snapping must not silently drop new `Entry` fields
# ──────────────────────────────────────────────────────────────────────────────

def test_snap_entry_preserves_every_entry_field_it_does_not_reprice(root):
    """`_snap_entry` rebuilds the `Entry` to move it onto the 5-min grid. Listing the fields by
    hand meant each new field (Batch-2 Task B's `entry_style`) was silently reset to its default,
    so a limit-entry shadow record resolved as a market fill. `dataclasses.replace` copies
    everything and names only what actually changes."""
    import dataclasses

    from research.strategy_hunt.shadow_resolver import _snap_entry
    from strategies.families.models import Entry

    _tmp, bars = root
    on_grid = bars.index[10]
    off_grid = on_grid - pd.Timedelta(minutes=2)
    e = Entry("AAA", "long", off_grid, 100.0, 99.0, 102.0, 20, "intraday", "test", "limit_open")

    snapped = _snap_entry(e, bars)
    assert snapped is not None
    assert snapped.entry_ts == on_grid
    assert snapped.entry_px == float(bars["open"].iloc[10])
    delta = snapped.entry_px - e.entry_px
    assert snapped.stop_px == pytest.approx(e.stop_px + delta)
    assert snapped.target_px == pytest.approx(e.target_px + delta)
    # every field other than the three the snap is allowed to move is carried over verbatim
    moved = {"entry_ts", "entry_px", "stop_px", "target_px"}
    for f in dataclasses.fields(Entry):
        if f.name not in moved:
            assert getattr(snapped, f.name) == getattr(e, f.name), f.name
    assert snapped.entry_style == "limit_open"


def test_snap_entry_returns_an_on_grid_entry_unchanged_and_drops_one_past_the_tape(root):
    from research.strategy_hunt.shadow_resolver import _snap_entry
    from strategies.families.models import Entry

    _tmp, bars = root
    e = Entry("AAA", "long", bars.index[5], 100.0, 99.0, 102.0, 20, "intraday", "test", "limit_open")
    assert _snap_entry(e, bars) is e
    late = Entry("AAA", "long", bars.index[-1] + pd.Timedelta(days=5), 100.0, 99.0, 102.0, 20,
                 "intraday", "test")
    assert _snap_entry(late, bars) is None
