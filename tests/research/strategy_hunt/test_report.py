import json

import pandas as pd
import pytest

from research.strategy_hunt.models import TRADE_COLUMNS


def _row(sym, fam, side, verdict, exp_r, n=200, source="walkforward", regime=None):
    return {"symbol": sym, "strategy": fam, "side": side, "computed_at": "2025-09-25T00:00:00", "window_end": "2025-09-24",
            "n": n, "sessions": 40, "exp_r": exp_r, "exp_r_lb": 0.02, "t_stat": 2.5, "pf": 1.4, "wr": 0.5, "max_dd_r": 5.0,
            "stability": 0.7, "decay": 0.8, "score": max(exp_r, 0) * 0.7 ** 0.5, "verdict": verdict, "failed_floors": "[]",
            "params_json": "{}",
            "regime_exp_json": json.dumps(regime or {"risk_on": {"n": n // 2, "exp_r": exp_r + 0.1}, "mixed": {"n": n // 2, "exp_r": exp_r - 0.1}}),
            "source": source}


def _trade_row(sym, fam, side, entry_ts="2025-09-02 10:00", exit_ts="2025-09-02 11:00"):
    entry, exit_ = pd.Timestamp(entry_ts), pd.Timestamp(exit_ts)
    d = {"symbol": sym, "family": fam, "side": side, "entry_ts": entry, "exit_ts": exit_, "entry_px": 100.0,
         "stop_px": 98.0, "target_px": 104.0, "exit_px": 102.0, "entry_fill": 100.0, "exit_fill": 102.0,
         "r_multiple": 1.0, "pnl_pct": 2.0, "cost_pct": 0.0, "exit_reason": "target", "hold_bars": 5,
         "hold_sessions": 1, "tag": "", "regime": "mixed", "window_id": 0, "params_json": "{}"}
    # Batch-2 Task B: `entry_style` is left out on purpose, so this fixture also covers a trades
    # frame written before that column existed.
    return {k: d[k] for k in TRADE_COLUMNS if k in d}


def test_render_report_sections():
    from research.strategy_hunt.report import render_report
    rows = [_row("NVDA", "orb", "long", "PASS", 0.2), _row("TSLA", "orb", "long", "FAIL", -0.1),
            _row("AMD", "mr_20d", "short", "WATCH", 0.05), _row("NEW", "orb", "long", "WATCH", float("nan"), n=0, source="coldstart")]
    trades = pd.DataFrame({"family": ["orb", "orb", "mr_20d"], "side": ["long", "long", "short"], "r_multiple": [1.0, -0.5, 0.3],
                           "regime": ["risk_on", "mixed", "mixed"]})
    controls = {"cost_0.5": [_row("X", "orb", "long", "PASS", 0.3), _row("Y", "orb", "long", "FAIL", -0.2)],
                "cost_2.0": [_row("NVDA", "orb", "long", "WATCH", 0.05)]}
    md = render_report(rows, "2025-09-25T00:00:00", trades, controls)
    for section in ("## Verdicts by family", "## Top lanes by score", "## Pooled out-of-sample by family",
                    "## Per-regime expectancy by family", "## Pooled rows vs shuffled baseline",
                    "## Search accounting", "## Controls", "## Symbols with no passing lane", "## Caveats",
                    "## Portfolio replay of PASS lanes", "## Cost calibration"):
        assert section in md
    assert "| orb | long | 1 | 0 | 1 | 1 |" in md            # PASS, WATCH(walkforward), FAIL, coldstart WATCH
    assert "NVDA" in md and "cost_0.5" in md and "cost_2.0" in md
    # v-playbook-desk-b1-5 task 10 fix r1: the separate-root shuffle control (and its
    # false-discovery line) is gone -- the in-root baseline supersedes it.
    assert "false-discovery" not in md
    assert "TSLA" in md.split("## Symbols with no passing lane")[1]
    # portfolio and calibration were not passed -> not available
    assert "_not available_" in md.split("## Portfolio replay of PASS lanes")[1].split("## Cost calibration")[0]
    assert "_not available_" in md.split("## Cost calibration")[1]


def test_top_lanes_table_has_t_stat_and_exp_r_lb_columns():
    from research.strategy_hunt.report import render_report
    rows = [_row("NVDA", "orb", "long", "PASS", 0.2)]
    md = render_report(rows, "2025-09-25T00:00:00")
    top = md.split("## Top lanes by score")[1].split("## Pooled out-of-sample by family")[0]
    header = top.strip().splitlines()[0]
    assert "t_stat" in header and "exp_r_lb" in header
    assert "2.500" in top and "0.020" in top


def test_render_report_with_portfolio_and_calibration():
    from research.strategy_hunt.report import render_report
    rows = [_row("NVDA", "orb", "long", "PASS", 0.2)]
    portfolio = {"taken": 12, "skipped_capacity": 3, "skipped_symbol": 1, "skipped_size": 0, "final_equity": 45123.45,
                 "total_return_pct": 3.73, "max_drawdown_pct": 4.1, "daily_sharpe": 1.25,
                 "equity_curve": pd.Series([43500.0, 44000.0, 45123.45],
                                            index=pd.to_datetime(["2025-08-01", "2025-08-15", "2025-09-01"])),
                 "by_family": {"orb": {"taken": 12, "pnl": 1623.45}}}
    calibration = {"n": 87, "median_frac": 0.0006, "p75_frac": 0.0012,
                   "by_bucket": {"mega": {"n": 40, "median_frac": 0.0003}}, "usable": True}
    md = render_report(rows, "2025-09-25T00:00:00", portfolio=portfolio, calibration=calibration)
    section = md.split("## Portfolio replay of PASS lanes")[1].split("## Cost calibration")[0]
    assert "12" in section and "45123.45" in section and "3.73" in section and "4.1" in section and "1.25" in section
    assert "orb" in section and "1623.45" in section
    cal_section = md.split("## Cost calibration")[1]
    assert "87" in cal_section and "0.0006" in cal_section and "0.0012" in cal_section and "mega" in cal_section


def test_write_report_creates_file(tmp_path, monkeypatch):
    from research.strategy_hunt import report as R
    monkeypatch.setattr(R, "REPORTS_DIR", tmp_path / "reports")
    (tmp_path / "trades").mkdir()
    trade = _trade_row("NVDA", "orb", "long")
    pd.DataFrame([trade]).to_parquet(tmp_path / "trades" / "NVDA.parquet")
    rows = [_row("NVDA", "orb", "long", "PASS", 0.2)]
    p = R.write_report(tmp_path, rows, "2025-09-25T00:00:00")
    # v-playbook-desk-b1-5 task 10: run-stamped, so two runs on one day keep both reports
    assert p.exists() and p.name == "2025-09-25T0000-strategy-hunt.md"
    md = p.read_text()
    assert "orb" in md and "## Portfolio replay of PASS lanes" in md
    csv_path = tmp_path / "reports" / "2025-09-25T0000-equity-curve.csv"
    assert csv_path.exists()
    csv_text = csv_path.read_text()
    assert csv_text.splitlines()[0] == "date,equity"


def test_write_report_with_empty_trades_frame_renders_zero_portfolio(tmp_path, monkeypatch):
    from research.strategy_hunt import report as R
    monkeypatch.setattr(R, "REPORTS_DIR", tmp_path / "reports")
    (tmp_path / "trades").mkdir()
    pd.DataFrame([], columns=list(TRADE_COLUMNS)).to_parquet(tmp_path / "trades" / "NVDA.parquet")
    rows = [_row("NVDA", "orb", "long", "PASS", 0.2)]
    p = R.write_report(tmp_path, rows, "2025-09-25T00:00:00")
    md = p.read_text()
    section = md.split("## Portfolio replay of PASS lanes")[1].split("## Cost calibration")[0]
    assert "- taken: 0" in section
    csv_path = tmp_path / "reports" / "2025-09-25T0000-equity-curve.csv"
    assert csv_path.exists()
    assert csv_path.read_text().splitlines() == ["date,equity"]


# ──────────────────────
# v-playbook-desk-b1-5 task 10 — pooled-vs-baseline section, pooled control counts, run-stamped files
# ──────────────────────

def _pooled_row(fam, side, regime, exp_r, baseline_exp_r, verdict="WATCH", failed=("min_excess_r",), n=1200):
    excess = exp_r - baseline_exp_r if baseline_exp_r == baseline_exp_r else float("nan")
    return {"symbol": "*", "strategy": fam, "side": side, "regime": regime, "computed_at": "2025-09-25T00:00:00",
            "window_end": "2025-09-24", "n": n, "sessions": 80, "exp_r": exp_r, "exp_r_lb": 0.01, "t_stat": 4.0,
            "pf": 1.3, "wr": 0.5, "max_dd_r": 90.0, "stability": 0.8, "decay": 0.9, "score": 0.1,
            "verdict": verdict, "failed_floors": json.dumps(list(failed)), "params_json": "{}",
            "regime_exp_json": "{}", "source": "pooled", "t_stat_session": 3.4, "recovery": 4.2,
            "baseline_exp_r": baseline_exp_r, "baseline_n": 0 if baseline_exp_r != baseline_exp_r else 900,
            "excess_r": excess}


def test_pooled_vs_baseline_section_renders_the_excess_column_sorted_by_excess():
    from research.strategy_hunt.report import render_report
    pooled = [_pooled_row("orb", "long", "risk_on", 0.12, 0.10),
              _pooled_row("orb", "long", "mixed", 0.30, 0.10, verdict="PASS", failed=()),
              _pooled_row("mr_20d", "short", "risk_off", 0.20, float("nan"), failed=("no_baseline",))]
    md = render_report([_row("NVDA", "orb", "long", "PASS", 0.2)], "2025-09-25T00:00:00", pooled=pooled)
    assert "## Pooled rows vs shuffled baseline" in md
    section = md.split("## Pooled rows vs shuffled baseline")[1].split("## Controls")[0]
    header = section.strip().splitlines()[0]
    for col in ("family", "side", "regime", "exp_R", "baseline exp_R", "excess", "t_session", "recovery",
                "verdict", "failed floors"):
        assert col in header
    body = [ln for ln in section.strip().splitlines()[2:] if ln.startswith("|")]
    assert "0.200" in body[0] and "mixed" in body[0]          # biggest excess first
    assert "0.020" in body[1]
    assert "no_baseline" in body[2]                            # NaN excess sorts last


def test_pooled_vs_baseline_section_without_pooled_rows():
    from research.strategy_hunt.report import render_report
    md = render_report([_row("NVDA", "orb", "long", "PASS", 0.2)], "2025-09-25T00:00:00")
    section = md.split("## Pooled rows vs shuffled baseline")[1].split("## Controls")[0]
    assert "_no pooled rows_" in section


def test_controls_section_reports_pooled_pass_and_watch_counts():
    from research.strategy_hunt.report import render_report
    controls = {"cost_2.0": [_row("NVDA", "orb", "long", "PASS", 0.2),
                             _pooled_row("orb", "long", "risk_on", 0.30, 0.10, verdict="PASS", failed=()),
                             _pooled_row("orb", "short", "mixed", 0.12, 0.10)]}
    md = render_report([_row("NVDA", "orb", "long", "PASS", 0.2)], "2025-09-25T00:00:00", controls=controls)
    section = md.split("## Controls")[1]
    assert "pooled PASS 1" in section and "pooled WATCH 1" in section


def test_load_controls_includes_the_control_pooled_rows(tmp_path):
    from research.strategy_hunt.report import _load_controls
    fit = tmp_path / "fit"
    fit.mkdir()
    (fit / "strategy_fit_cost_2.0.json").write_text(json.dumps(
        {"computed_at": "x", "rows": {"NVDA": {"orb:long": _row("NVDA", "orb", "long", "PASS", 0.2)}},
         "pooled_rows": {"orb:long:risk_on": _pooled_row("orb", "long", "risk_on", 0.3, 0.1,
                                                          verdict="PASS", failed=())}}))
    controls = _load_controls(tmp_path)
    assert set(controls) == {"cost_2.0"}
    sources = [r.get("source") for r in controls["cost_2.0"]]
    assert sources.count("walkforward") == 1 and sources.count("pooled") == 1


def test_write_report_is_run_stamped_so_two_runs_keep_both_files(tmp_path, monkeypatch):
    from research.strategy_hunt import report as R
    monkeypatch.setattr(R, "REPORTS_DIR", tmp_path / "reports")
    rows = [_row("NVDA", "orb", "long", "PASS", 0.2)]
    first = R.write_report(tmp_path, rows, "2025-09-25T01:05:00")
    second = R.write_report(tmp_path, rows, "2025-09-25T02:30:00")
    assert first != second and first.exists() and second.exists()
    assert first.name == "2025-09-25T0105-strategy-hunt.md"
    assert second.name == "2025-09-25T0230-strategy-hunt.md"
    assert sorted(p.name for p in (tmp_path / "reports").glob("*-strategy-hunt.md")) == [first.name, second.name]


def test_write_report_writes_a_manifest_json_with_the_same_stem(tmp_path, monkeypatch):
    from research.strategy_hunt import report as R
    monkeypatch.setattr(R, "REPORTS_DIR", tmp_path / "reports")
    pooled = [_pooled_row("orb", "long", "risk_on", 0.30, 0.10, verdict="PASS", failed=())]
    rows = [_row("NVDA", "orb", "long", "PASS", 0.2), _row("TSLA", "orb", "long", "FAIL", -0.1)]
    p = R.write_report(tmp_path, rows, "2025-09-25T01:05:00", pooled=pooled)
    manifest = p.with_suffix(".json")
    assert manifest.exists() and manifest.parent == p.parent
    rec = json.loads(manifest.read_text())
    assert rec["computed_at"] == "2025-09-25T01:05:00" and rec["report"] == p.name
    assert rec["verdicts"] == {"PASS": 1, "FAIL": 1} and rec["pooled_verdicts"] == {"PASS": 1}
    assert rec["n_rows"] == 2 and rec["n_symbols"] == 2 and rec["pooled_with_baseline"] == 1


def test_report_stem_from_computed_at():
    from research.strategy_hunt.report import report_stem
    assert report_stem("2025-09-25T01:05:00") == "2025-09-25T0105"
    assert report_stem("2025-09-25") == "2025-09-25T0000"
    with pytest.raises(ValueError):
        report_stem("nope")


def test_control_files_no_longer_include_the_separate_root_shuffle_control():
    """v-playbook-desk-b1-5 task 10 fix r1: the in-root shuffled baseline supersedes the
    separate-root shuffle run, and no documented command produces strategy_fit_shuffle.json."""
    from research.strategy_hunt.report import CONTROL_FILES
    assert set(CONTROL_FILES) == {"cost_0.5", "cost_2.0"}
    assert all("shuffle" not in f for f in CONTROL_FILES.values())


def test_report_manifest_tolerates_rows_missing_keys():
    """Finding 5: the manifest reads rows with `.get` defaults like the rest of the module, so a
    hand-written or partial snapshot row cannot raise on the way to disk."""
    from research.strategy_hunt.report import _report_manifest
    rows = [{"source": "walkforward"}, {"symbol": "NVDA", "source": "walkforward", "verdict": "PASS"}]
    pooled = [{"baseline_n": None}, {"verdict": "WATCH", "baseline_n": 3}]
    rec = _report_manifest(rows, pooled, "2025-09-25T01:05:00", "r.md")
    assert rec["verdicts"] == {"": 1, "PASS": 1} and rec["pooled_verdicts"] == {"": 1, "WATCH": 1}
    assert rec["n_rows"] == 2 and rec["n_symbols"] == 2 and rec["pooled_with_baseline"] == 1
    assert rec["computed_at"] == "2025-09-25T01:05:00" and rec["report"] == "r.md"


# ──────────────────────
# strategy-search-batch1 task 2 — "Search accounting": grid combos x sides vs PASS/WATCH,
# from the registry and the run manifest, so multiple testing is visible
# ──────────────────────

def test_search_accounting_section_reports_grid_points_and_totals():
    """`orb` and `mr_20d` are real registered families (strategies/families/registry.py):
    orb = range_minutes(2) x rvol_min(3) x atr_stop(2) = 12 combos x 2 sides = 24 grid points;
    mr_20d = band_sigma(2) x rsi3_extreme(3) x time_stop_days(2) x entry_style(2) = 24 combos
    x 2 sides = 48 (the `entry_style` axis is Batch-2 Task B -- and the point of this section is
    that adding a grid axis shows up in the multiple-testing count, so the number must move)."""
    from research.strategy_hunt.report import render_report
    rows = [_row("NVDA", "orb", "long", "PASS", 0.2), _row("TSLA", "orb", "long", "FAIL", -0.1),
            _row("AMD", "mr_20d", "short", "WATCH", 0.05)]
    run_manifest = {"families": ["orb", "mr_20d"], "holdout_months": 0}
    md = render_report(rows, "2025-09-25T00:00:00", run_manifest=run_manifest)
    section = md.split("## Search accounting")[1].split("## Controls")[0]
    assert "| orb | 12 × 2 = 24 | 2 | 1 | 0 |" in section
    assert "| mr_20d | 24 × 2 = 48 | 1 | 0 | 1 |" in section
    assert "| **total** | 72 | 3 | 1 | 1 |" in section
    assert "2 families, 72 grid points × sides evaluated, 1 PASS" in section


def test_search_accounting_without_manifest_falls_back_to_rows_families():
    from research.strategy_hunt.report import render_report
    rows = [_row("NVDA", "orb", "long", "PASS", 0.2),
            _row("NEW", "orb", "long", "WATCH", float("nan"), n=0, source="coldstart")]
    md = render_report(rows, "2025-09-25T00:00:00")
    section = md.split("## Search accounting")[1].split("## Controls")[0]
    # the coldstart row must not count toward the unconditional rows scored (source != walkforward)
    assert "| orb | 12 × 2 = 24 | 1 | 1 | 0 |" in section
    assert "1 families, 24 grid points × sides evaluated, 1 PASS" in section


def test_search_accounting_handles_an_unregistered_family_name():
    """A family name in the run manifest that the registry does not know (e.g. a manifest
    written before that family was registered, or a hand-edited one) degrades to zero grid
    points instead of raising -- the same contract every other optional section holds."""
    from research.strategy_hunt.report import render_report
    rows = [_row("NVDA", "made_up_family", "long", "PASS", 0.2)]
    md = render_report(rows, "2025-09-25T00:00:00", run_manifest={"families": ["made_up_family"]})
    section = md.split("## Search accounting")[1].split("## Controls")[0]
    assert "| made_up_family | 0 × 0 = 0 | 1 | 1 | 0 |" in section


def test_search_accounting_with_no_families_at_all():
    from research.strategy_hunt.report import render_report
    md = render_report([], "2025-09-25T00:00:00")
    section = md.split("## Search accounting")[1].split("## Controls")[0]
    assert "_no families run_" in section


def test_report_header_states_the_holdout_period_when_set():
    from research.strategy_hunt.report import render_report
    rows = [_row("NVDA", "orb", "long", "PASS", 0.2)]
    md_held = render_report(rows, "2025-09-25T00:00:00", run_manifest={"families": ["orb"], "holdout_months": 6})
    assert "Holdout" in md_held and "6 months" in md_held
    md_unheld = render_report(rows, "2025-09-25T00:00:00", run_manifest={"families": ["orb"], "holdout_months": 0})
    assert "Holdout" not in md_unheld
    md_no_manifest = render_report(rows, "2025-09-25T00:00:00")
    assert "Holdout" not in md_no_manifest


def test_write_report_reads_the_run_manifest_for_search_accounting_and_holdout_header(tmp_path, monkeypatch):
    from research.strategy_hunt import report as R
    from research.strategy_hunt.run import write_manifest

    monkeypatch.setattr(R, "REPORTS_DIR", tmp_path / "reports")
    write_manifest(tmp_path, {"date": "2025-09-25", "run": {"families": ["orb"], "holdout_months": 6}})
    rows = [_row("NVDA", "orb", "long", "PASS", 0.2)]
    p = R.write_report(tmp_path, rows, "2025-09-25T01:05:00")
    md = p.read_text()
    assert "Holdout" in md and "6 months" in md
    section = md.split("## Search accounting")[1].split("## Controls")[0]
    assert "orb" in section and "24" in section


def test_write_report_with_no_run_manifest_on_disk_degrades_gracefully(tmp_path, monkeypatch):
    """No `<date>.json` on disk (report generated standalone, or from a snapshot with no
    matching run manifest) must not raise -- families fall back to the rows themselves."""
    from research.strategy_hunt import report as R

    monkeypatch.setattr(R, "REPORTS_DIR", tmp_path / "reports")
    rows = [_row("NVDA", "orb", "long", "PASS", 0.2)]
    p = R.write_report(tmp_path, rows, "2025-09-25T01:05:00")
    md = p.read_text()
    assert "Holdout" not in md
    section = md.split("## Search accounting")[1].split("## Controls")[0]
    assert "orb" in section
