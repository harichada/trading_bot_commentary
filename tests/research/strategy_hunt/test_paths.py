"""v-playbook-desk-b1-5 task 12: control roots write their own manifest and reports.

`reports_dir(root)` is the single place that decides where a root's manifest, reports,
and scorecards land: the committed `research/reports/strategy_hunt/` for the real run
(`DEFAULT_ROOT`), or `<root>/reports` for any other root (a `--cost-mult` control run,
a shadow-resolver replay, etc.) so a control run can never overwrite -- or bury its own
evidence inside -- the real run's committed reports.
"""
from pathlib import Path

import pandas as pd

from research.strategy_hunt.models import TRADE_COLUMNS
from research.strategy_hunt.paths import reports_dir
from research.strategy_hunt.run import DEFAULT_ROOT

COMMITTED_DIR = Path("research/reports/strategy_hunt")


def _committed_dir_snapshot() -> dict:
    """mtime per file under the committed reports dir -- used to prove a control-root
    write left it untouched without assuming it starts out empty (real runs already
    populate it)."""
    if not COMMITTED_DIR.exists():
        return {}
    return {p: p.stat().st_mtime_ns for p in COMMITTED_DIR.rglob("*") if p.is_file()}


def test_reports_dir_of_default_root_is_the_committed_dir():
    assert reports_dir(DEFAULT_ROOT) == Path("research/reports/strategy_hunt")


def test_reports_dir_of_default_root_passed_unresolved_is_still_the_committed_dir(tmp_path, monkeypatch):
    # Compares resolved paths, not identity: chdir somewhere else and pass DEFAULT_ROOT's
    # own literal value again -- still the committed dir.
    monkeypatch.chdir(tmp_path)
    assert reports_dir(Path("data/strategy_hunt")) == Path("research/reports/strategy_hunt")


def test_reports_dir_of_a_tmp_root_is_under_the_root(tmp_path):
    control_root = tmp_path / "strategy_hunt_cost2"
    assert reports_dir(control_root) == control_root / "reports"


def test_write_manifest_for_a_tmp_root_does_not_touch_the_committed_dir(tmp_path):
    from research.strategy_hunt.run import write_manifest

    before = _committed_dir_snapshot()
    control_root = tmp_path / "strategy_hunt_cost2"
    p = write_manifest(control_root, {"date": "2026-09-25", "run": {"ok": True}})

    assert p == control_root / "reports" / "2026-09-25.json"
    assert p.exists()
    assert _committed_dir_snapshot() == before


def test_write_report_for_a_tmp_root_writes_md_json_csv_under_root_reports(tmp_path):
    from research.strategy_hunt.report import write_report

    before = _committed_dir_snapshot()
    control_root = tmp_path / "strategy_hunt_cost2"
    (control_root / "trades").mkdir(parents=True)
    trade = {"symbol": "NVDA", "family": "orb", "side": "long",
              "entry_ts": pd.Timestamp("2025-09-02 10:00"), "exit_ts": pd.Timestamp("2025-09-02 11:00"),
              "entry_px": 100.0, "stop_px": 98.0, "target_px": 104.0, "exit_px": 102.0,
              "entry_fill": 100.0, "exit_fill": 102.0, "r_multiple": 1.0, "pnl_pct": 2.0,
              "cost_pct": 0.0, "exit_reason": "target", "hold_bars": 5, "hold_sessions": 1,
              "tag": "", "regime": "mixed", "window_id": 0, "params_json": "{}"}
    # Batch-2 Task B: `entry_style` is absent on purpose -- this fixture doubles as the "a trades
    # Parquet written before the column existed still loads" case the report path has to tolerate.
    trade = {k: trade[k] for k in TRADE_COLUMNS if k in trade}
    pd.DataFrame([trade]).to_parquet(control_root / "trades" / "NVDA.parquet")
    rows = [{"symbol": "NVDA", "strategy": "orb", "side": "long", "computed_at": "2025-09-25T00:00:00",
             "window_end": "2025-09-24", "n": 200, "sessions": 40, "exp_r": 0.2, "exp_r_lb": 0.02,
             "t_stat": 2.5, "pf": 1.4, "wr": 0.5, "max_dd_r": 5.0, "stability": 0.7, "decay": 0.8,
             "score": 0.14, "verdict": "PASS", "failed_floors": "[]", "params_json": "{}",
             "regime_exp_json": "{}", "source": "walkforward"}]

    p = write_report(control_root, rows, "2025-09-25T00:00:00")

    reports = control_root / "reports"
    assert p == reports / "2025-09-25T0000-strategy-hunt.md"
    assert p.exists()
    assert p.with_suffix(".json").exists()
    assert (reports / "2025-09-25T0000-equity-curve.csv").exists()
    assert _committed_dir_snapshot() == before
