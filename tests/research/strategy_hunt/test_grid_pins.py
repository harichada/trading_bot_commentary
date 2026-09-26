"""Batch-2 Task F item 1: `run --grid-pin key=value` pins one grid axis for a pinned A/B lane.

The B2 question "is the cost model the whole gap?" is an A/B on one axis (`entry_style`), and a
grid search answers it only by accident: the walk-forward selector is free to pick whichever style
won in-sample, so a run cannot be attributed to either arm. A pin collapses that axis to a single
value for every family that *has* it, leaving every other family untouched -- so two runs, one per
arm, are the same search with one variable changed.

The pin is part of `_config_key` (a checkpoint from the other arm must not be reused), of the run
manifest, and of the report's search accounting (grid points are counted after pinning -- the point
of that section is the multiple-testing burden actually paid).
"""
import json

import numpy as np
import pandas as pd
import pytest

from strategies.families.common import Family, make_entry, one_per_session

# ──────────────────────────────  parsing the flag  ──────────────────────────────


def test_parse_grid_pins_reads_repeated_key_value_tokens():
    from research.strategy_hunt.cli_args import parse_grid_pins

    assert parse_grid_pins(["entry_style=limit_open", "time_stop_days=5"]) == {
        "entry_style": "limit_open", "time_stop_days": "5"}
    assert parse_grid_pins(None) == {}
    assert parse_grid_pins([]) == {}
    # a repeated key is the last one given, not both (argparse append cannot dedupe)
    assert parse_grid_pins(["w=1", "w=2"]) == {"w": "2"}
    # a value may itself contain "=" -- only the first separator splits
    assert parse_grid_pins(["tag=a=b"]) == {"tag": "a=b"}


@pytest.mark.parametrize("token", ["entry_style", "=limit_open", "  =1", ""])
def test_parse_grid_pins_rejects_a_malformed_token(token):
    from research.strategy_hunt.cli_args import parse_grid_pins

    with pytest.raises(ValueError):
        parse_grid_pins([token])


# ──────────────────────────────  applying the pin to a grid  ──────────────────────────────


def _fam(grid: dict) -> Family:
    return Family("toy", lambda *a, **kw: [], grid, "intraday", ("long",))


def test_pin_applies_only_where_the_key_exists_and_leaves_the_family_alone_otherwise():
    from research.strategy_hunt.walkforward.runner import pinned_family

    styled = _fam({"w": [0.5, 1.0], "entry_style": ["market", "limit_open"]})
    plain = _fam({"w": [0.5, 1.0]})
    pins = {"entry_style": "limit_open"}

    pinned = pinned_family(styled, pins)
    assert pinned.grid == {"w": [0.5, 1.0], "entry_style": ["limit_open"]}
    assert styled.grid["entry_style"] == ["market", "limit_open"]      # immutable: source untouched
    assert pinned_family(plain, pins) is plain                         # no such axis -> same object
    assert pinned_family(styled, {}) is styled                         # no pins -> same object


def test_pin_values_are_typed_like_the_grid_they_replace():
    from research.strategy_hunt.walkforward.runner import pinned_family

    ints = pinned_family(_fam({"time_stop_days": [2, 5]}), {"time_stop_days": "3"}).grid
    assert ints["time_stop_days"] == [3] and isinstance(ints["time_stop_days"][0], int)
    floats = pinned_family(_fam({"rank_min": [0.8, 0.9]}), {"rank_min": "0.85"}).grid
    assert floats["rank_min"] == [0.85] and isinstance(floats["rank_min"][0], float)
    strs = pinned_family(_fam({"entry_style": ["market", "limit_open"]}), {"entry_style": "limit_open"}).grid
    assert strs["entry_style"] == ["limit_open"] and isinstance(strs["entry_style"][0], str)
    # a mixed axis (int and float values) is widened to float, never silently truncated
    mixed = pinned_family(_fam({"k": [1, 2.5]}), {"k": "1.5"}).grid
    assert mixed["k"] == [1.5]


def test_a_pin_that_cannot_be_typed_fails_loudly():
    from research.strategy_hunt.walkforward.runner import pinned_family

    with pytest.raises(ValueError):
        pinned_family(_fam({"time_stop_days": [2, 5]}), {"time_stop_days": "soon"})


def test_pinning_collapses_the_grid_combo_count():
    from strategies.families.common import grid_combos
    from research.strategy_hunt.walkforward.runner import pinned_family

    fam = _fam({"w": [0.5, 1.0], "entry_style": ["market", "limit_open"]})
    assert len(grid_combos(fam.grid)) == 4
    assert len(grid_combos(pinned_family(fam, {"entry_style": "limit_open"}).grid)) == 2


# ──────────────────────────────  the checkpoint config key  ──────────────────────────────


def test_config_key_carries_the_pins_and_only_when_there_are_any():
    """A pinned arm must not reuse the other arm's checkpoints -- but an *unpinned* run's key has
    to stay byte-identical to what earlier runs wrote, or every existing checkpoint invalidates."""
    from research.strategy_hunt.run import _config_key

    base = _config_key(["toy"], 1.0, None, 6000.0, None)
    assert "grid_pins" not in base
    assert _config_key(["toy"], 1.0, None, 6000.0, None, grid_pins={}) == base
    pinned = _config_key(["toy"], 1.0, None, 6000.0, None, grid_pins={"entry_style": "limit_open"})
    assert pinned != base and pinned["grid_pins"] == {"entry_style": "limit_open"}
    other = _config_key(["toy"], 1.0, None, 6000.0, None, grid_pins={"entry_style": "market"})
    assert other != pinned


# ──────────────────────────────  threading it through the run  ──────────────────────────────


def _bars(start="2023-01-03", sessions=330, seed=0):
    rng = np.random.default_rng(seed)
    frames, px = [], 100.0
    for d in pd.bdate_range(start, periods=sessions):
        idx = pd.date_range(f"{d.date()} 09:30", periods=78, freq="5min", name="ts")
        close = px + rng.normal(0, 0.05, 78).cumsum()
        frames.append(pd.DataFrame({"open": close - 0.02, "high": close + 0.3, "low": close - 0.3,
                                    "close": close, "volume": 1000}, index=idx))
        px = float(close[-1])
    return pd.concat(frames)


def _toy_entries(bars5, ind, daily, news, params):
    out = []
    for i in np.flatnonzero(ind["bar_of_day"].to_numpy() == 10):
        if i + 1 < len(bars5):
            px = float(bars5["open"].iloc[i + 1])
            e = make_entry(bars5, int(i), "long", px - params["w"], px + 2 * params["w"], 30, "intraday", "toy")
            if e:
                out.append(e)
    return one_per_session(out)


PINNABLE = Family("toy", _toy_entries, {"w": [0.5, 1.0]}, "intraday", ("long",))


@pytest.fixture
def root(tmp_path, monkeypatch):
    from research.strategy_hunt.data.bars import _write, bars5_path, daily_path, to_daily
    from strategies.families import registry

    b = _bars()
    for sym in ("AAA", "SPY"):
        _write(bars5_path(tmp_path, sym), b, "ts")
        _write(daily_path(tmp_path, sym), to_daily(b), "date")
    monkeypatch.setitem(registry.FAMILIES, "toy", PINNABLE)
    return tmp_path


def test_process_symbol_pins_the_grid_before_the_walkforward_runner(root, monkeypatch):
    from research.strategy_hunt import run as R

    seen: list[dict] = []

    def _capture(family, symbol, *a, **kw):
        seen.append(dict(family.grid))
        return [], []

    monkeypatch.setattr(R, "run_symbol_family", _capture)
    assert R.process_symbol("AAA", root, ["toy"], grid_pins={"w": "1.0"})["status"] == "done"
    assert seen == [{"w": [1.0]}]
    assert PINNABLE.grid == {"w": [0.5, 1.0]}          # the registry's family is untouched


def test_process_symbol_checkpoint_invalidates_when_the_pin_changes(root):
    from research.strategy_hunt.run import checkpoint_path, process_symbol

    assert process_symbol("AAA", root, ["toy"], grid_pins={"w": "1.0"})["status"] == "done"
    assert process_symbol("AAA", root, ["toy"], grid_pins={"w": "1.0"})["status"] == "skipped"
    assert process_symbol("AAA", root, ["toy"], grid_pins={"w": "0.5"})["status"] == "done"
    cfg = json.loads(checkpoint_path(root, "AAA").read_text())["config"]
    assert cfg["grid_pins"] == {"w": "0.5"}


def test_process_symbol_choices_record_only_the_pinned_value(root):
    from research.strategy_hunt.run import choices_path, process_symbol

    assert process_symbol("AAA", root, ["toy"], grid_pins={"w": "1.0"})["status"] == "done"
    choices = json.loads(choices_path(root, "AAA").read_text())["toy"]
    assert choices and {c["params"]["w"] for c in choices} == {1.0}


# ──────────────────────────────  the CLI flag and the manifest  ──────────────────────────────


def test_grid_pin_flag_is_repeatable_and_defaults_to_none():
    from research.strategy_hunt.cli import build_parser

    assert build_parser().parse_args(["run"]).grid_pin is None
    a = build_parser().parse_args(["run", "--grid-pin", "entry_style=limit_open", "--grid-pin", "w=1"])
    assert a.grid_pin == ["entry_style=limit_open", "w=1"]
    assert build_parser().parse_args(["all", "--grid-pin", "entry_style=market"]).grid_pin == \
        ["entry_style=market"]


def test_cmd_run_threads_the_pins_and_records_them_in_the_manifest(monkeypatch, tmp_path):
    from research.strategy_hunt import cli
    from research.strategy_hunt.paths import reports_dir

    seen: dict = {}

    def _fake_run_symbols(symbols, root, fams, **kw):
        seen.update(kw)
        return [{"symbol": "AAA", "status": "done", "n_trades": 3}]

    monkeypatch.setattr(cli, "run_symbols", _fake_run_symbols)
    monkeypatch.setattr(cli, "_symbols_arg", lambda v, r: ["AAA"])

    assert cli.main(["run", "--root", str(tmp_path), "--families", "toy",
                     "--grid-pin", "entry_style=limit_open"]) == 0
    assert seen["grid_pins"] == {"entry_style": "limit_open"}
    rec = json.loads(next(reports_dir(tmp_path).glob("*.json")).read_text())
    assert rec["run"]["grid_pins"] == {"entry_style": "limit_open"}


def test_a_malformed_pin_is_a_clean_failure_not_a_traceback(monkeypatch, tmp_path, capsys):
    from research.strategy_hunt import cli

    monkeypatch.setattr(cli, "run_symbols", lambda *a, **kw: pytest.fail("must not run"))
    assert cli.main(["run", "--root", str(tmp_path), "--grid-pin", "entry_style"]) == 1
    assert "--grid-pin" in capsys.readouterr().out


def test_both_control_namespaces_inherit_the_pins():
    """A control has to search the same collapsed grid as the arm it is the control for."""
    from research.strategy_hunt import cli

    a = cli.build_parser().parse_args(["all", "--grid-pin", "entry_style=limit_open"])
    for ns in (cli._baseline_namespace(a), cli._selection_namespace(a)):
        assert ns.grid_pin == ["entry_style=limit_open"]


# ──────────────────────────────  the report  ──────────────────────────────


def _row(symbol, strategy, side, verdict, exp_r, n=40, source="walkforward"):
    return {"symbol": symbol, "strategy": strategy, "side": side, "verdict": verdict, "n": n,
            "exp_r": exp_r, "exp_r_lb": 0.0, "t_stat": 1.0, "pf": 1.2, "wr": 0.5, "max_dd_r": 2.0,
            "stability": 0.5, "decay": 0.9, "score": exp_r, "source": source,
            "regime_exp_json": "{}", "params_json": "{}"}


def test_report_header_states_the_grid_pins():
    from research.strategy_hunt.report import render_report

    rows = [_row("NVDA", "orb", "long", "PASS", 0.2)]
    md = render_report(rows, "2025-09-25T00:00:00",
                       run_manifest={"families": ["orb"], "grid_pins": {"entry_style": "limit_open"}})
    assert "Grid pins: entry_style=limit_open" in md
    assert "Grid pins" not in render_report(rows, "2025-09-25T00:00:00",
                                           run_manifest={"families": ["orb"]})
    assert "Grid pins" not in render_report(rows, "2025-09-25T00:00:00")


def test_search_accounting_counts_grid_points_after_pinning():
    """`mr_20d` is band_sigma(2) x rsi3_extreme(3) x time_stop_days(2) x entry_style(2) = 24 combos
    x 2 sides = 48 unpinned; pinning `entry_style` halves the grid the run actually searched, and
    the multiple-testing count has to say so."""
    from research.strategy_hunt.report import render_report

    rows = [_row("AMD", "mr_20d", "short", "WATCH", 0.05)]
    md = render_report(rows, "2025-09-25T00:00:00",
                       run_manifest={"families": ["mr_20d"], "grid_pins": {"entry_style": "limit_open"}})
    section = md.split("## Search accounting")[1].split("## Controls")[0]
    assert "| mr_20d | 12 × 2 = 24 | 1 | 0 | 1 |" in section
    assert "| **total** | 24 | 1 | 0 | 1 |" in section


def test_search_accounting_ignores_a_malformed_grid_pins_section():
    """Every optional manifest section degrades rather than raising (the report is evidence, and
    half a report is worse than a slightly less precise one)."""
    from research.strategy_hunt.report import render_report

    rows = [_row("AMD", "mr_20d", "short", "WATCH", 0.05)]
    md = render_report(rows, "2025-09-25T00:00:00",
                       run_manifest={"families": ["mr_20d"], "grid_pins": ["entry_style=limit_open"]})
    section = md.split("## Search accounting")[1].split("## Controls")[0]
    assert "| mr_20d | 24 × 2 = 48 | 1 | 0 | 1 |" in section


# Task F review, Important 1: a pin key that no family has must not become a silent no-op.
def test_unknown_grid_pin_keys_are_reported_and_known_ones_are_not():
    from research.strategy_hunt.cli_args import unknown_grid_pin_keys

    assert unknown_grid_pin_keys({"entry_stlye": "limit_open"}, ["mr_20d", "overnight_hold"]) == ["entry_stlye"]
    assert unknown_grid_pin_keys({"entry_style": "limit_open"}, ["mr_20d", "overnight_hold"]) == []
    # known on one family, absent on the other: still fine (pins apply where the axis exists)
    assert unknown_grid_pin_keys({"band_sigma": "2.0"}, ["mr_20d", "overnight_hold"]) == []


def test_cmd_run_rejects_a_pin_key_that_no_run_family_has(tmp_path, monkeypatch, capsys):
    import argparse
    from research.strategy_hunt import cli

    called = {}
    monkeypatch.setattr(cli, "run_symbols", lambda *a, **k: called.setdefault("ran", True) or [])
    monkeypatch.setattr(cli, "_symbols_arg", lambda s, r: ["AAA"])
    a = argparse.Namespace(root=tmp_path, symbols=None, families="mr_20d", min_days=250, workers=1, force=True,
                           cost_mult=1.0, shuffle_seed=None, notional=6000.0, no_coldstart=False, postgres=False,
                           holdout_months=0, grid_pin=["entry_stlye=limit_open"], shuffle_mode="path")
    rc = cli.cmd_run(a)
    out = capsys.readouterr().out
    assert rc == 1 and "entry_stlye" in out and "ran" not in called
