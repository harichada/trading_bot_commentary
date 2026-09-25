import json
import math
import pandas as pd
import pytest

from research.strategy_hunt.models import SimTrade


def _t(r, w=0, day="2025-01-02", regime="mixed"):
    ts = pd.Timestamp(f"{day} 10:00")
    return SimTrade("NVDA", "orb", "long", ts, ts + pd.Timedelta(minutes=30), 100, 99.0, 102.0, 100 + r, 100,
                    100 + r, r, r, 0.0, "target" if r > 0 else "stop", 6, 1, "orb_long", regime, w, '{"k": 1}')


def _trades():
    days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2025-01-02", periods=20)]
    out = []
    for i, d in enumerate(days):
        out.append(_t(1.5 if i % 3 else -1.0, w=i // 5, day=d, regime="risk_on" if i % 2 else "mixed"))
    return out


def test_build_row_fields_and_score():
    from research.strategy_hunt.fit.writer import build_row, ROW_KEYS
    row = build_row("NVDA", "orb", "long", _trades(), 4, {"k": 1}, "2025-09-25T00:00:00", "2025-09-24")
    assert tuple(row) == ROW_KEYS
    assert row["n"] == 20 and row["verdict"] in ("PASS", "WATCH", "FAIL") and row["source"] == "walkforward"
    assert row["score"] == pytest.approx(row["exp_r"] * math.sqrt(row["stability"]))
    assert json.loads(row["params_json"]) == {"k": 1}
    assert set(json.loads(row["regime_exp_json"])) == {"risk_on", "mixed"}
    assert isinstance(json.loads(row["failed_floors"]), list)
    assert "t_stat" in row


def test_build_row_inf_pf_and_negative_score():
    from research.strategy_hunt.fit.writer import build_row
    row = build_row("NVDA", "orb", "long", [_t(1.0), _t(2.0)], 1, {}, "2025-09-25", "2025-09-24")
    assert row["pf"] == 999.0
    row2 = build_row("NVDA", "orb", "long", [_t(-1.0), _t(-2.0)], 1, {}, "2025-09-25", "2025-09-24")
    assert row2["score"] == 0.0 and row2["verdict"] == "FAIL"


def test_write_fit_parquet_json_and_sqlite_upsert(tmp_path):
    from sqlalchemy import create_engine, text
    from research.strategy_hunt.fit.writer import build_row, write_fit, load_snapshot, upsert_postgres
    rows = [build_row("NVDA", "orb", "long", _trades(), 4, {"k": 1}, "2025-09-25T01:00:00", "2025-09-24"),
            build_row("TSLA", "orb", "short", _trades(), 4, {"k": 2}, "2025-09-25T01:00:00", "2025-09-24")]
    engine = create_engine("sqlite:///" + str(tmp_path / "t.db"))
    out = write_fit(rows, tmp_path, "2025-09-25T01:00:00", engine=engine)
    assert (tmp_path / "fit" / "fit_2025-09-25.parquet").exists()
    snap = load_snapshot(tmp_path)
    assert snap["computed_at"] == "2025-09-25T01:00:00" and snap["rows"]["NVDA"]["orb:long"]["n"] == 20
    assert snap["floors"]["min_n"] == 150
    with engine.connect() as c:
        assert c.execute(text("SELECT count(*) FROM strategy_fit")).scalar() == 2
    rows2 = [dict(rows[0], n=21)]
    assert upsert_postgres(rows2, engine) == 1
    with engine.connect() as c:
        assert c.execute(text("SELECT n FROM strategy_fit WHERE symbol='NVDA'")).scalar() == 21
        assert c.execute(text("SELECT count(*) FROM strategy_fit")).scalar() == 2
    assert "json" in out and "parquet" in out


def test_load_snapshot_missing(tmp_path):
    from research.strategy_hunt.fit.writer import load_snapshot
    assert load_snapshot(tmp_path) == {"computed_at": None, "floors": {}, "rows": {}}
