import json
import pandas as pd

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
    return {k: d[k] for k in TRADE_COLUMNS}


def test_render_report_sections_and_fdr():
    from research.strategy_hunt.report import render_report
    rows = [_row("NVDA", "orb", "long", "PASS", 0.2), _row("TSLA", "orb", "long", "FAIL", -0.1),
            _row("AMD", "mr_20d", "short", "WATCH", 0.05), _row("NEW", "orb", "long", "WATCH", float("nan"), n=0, source="coldstart")]
    trades = pd.DataFrame({"family": ["orb", "orb", "mr_20d"], "side": ["long", "long", "short"], "r_multiple": [1.0, -0.5, 0.3],
                           "regime": ["risk_on", "mixed", "mixed"]})
    controls = {"shuffle": [_row("X", "orb", "long", "PASS", 0.3), _row("Y", "orb", "long", "FAIL", -0.2)],
                "cost_2.0": [_row("NVDA", "orb", "long", "WATCH", 0.05)]}
    md = render_report(rows, "2025-09-25T00:00:00", trades, controls)
    for section in ("## Verdicts by family", "## Top lanes by score", "## Pooled out-of-sample by family",
                    "## Per-regime expectancy by family", "## Controls", "## Symbols with no passing lane", "## Caveats",
                    "## Portfolio replay of PASS lanes", "## Cost calibration"):
        assert section in md
    assert "| orb | long | 1 | 0 | 1 | 1 |" in md            # PASS, WATCH(walkforward), FAIL, coldstart WATCH
    assert "NVDA" in md and "false-discovery" in md and "1 / 1" in md
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
    assert p.exists() and p.name == "2025-09-25-strategy-hunt.md"
    md = p.read_text()
    assert "orb" in md and "## Portfolio replay of PASS lanes" in md
    csv_path = tmp_path / "reports" / "2025-09-25-equity-curve.csv"
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
    csv_path = tmp_path / "reports" / "2025-09-25-equity-curve.csv"
    assert csv_path.exists()
    assert csv_path.read_text().splitlines() == ["date,equity"]
