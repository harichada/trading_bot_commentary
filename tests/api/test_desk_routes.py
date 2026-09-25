"""Tests for the read-only desk endpoints (api/routes.py):
`GET /api/desk/plan`, `GET /api/desk/playbook`, `GET /api/desk/journal`.

TDD: written BEFORE the routes exist. Follows the client construction
pattern from tests/test_strategy_fit_endpoint.py: build a TestClient over
the real FastAPI app in api.routes with TRADING_API_KEY unset (bearer
middleware allows unauthenticated requests) and `trading_engine` stubbed to
None — these endpoints are file-backed readers only and must not depend on
any engine attribute.

`DESK_DATA_DIR` is pointed at `tmp_path` via monkeypatch.setenv, mirroring
how `core.config.Config().DESK_DATA_DIR` reads the env var fresh on every
call (see core/config.py::Config.DESK_DATA_DIR).
"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

from core.desk.models import DailyPlan, InPlayRow
from core.desk.playbook import Playbook, PlaybookEntry, playbook_path, save_playbook
from core.desk.store import save_plan
from core.desk.journal_data import write_journal


def _make_client(monkeypatch, desk_dir):
    """Import api.routes and return a TestClient with auth disabled and
    DESK_DATA_DIR pointed at `desk_dir`.

    Import first, then delenv: importing api.routes pulls in core.config,
    whose module-level `load_dotenv()` call populates TRADING_API_KEY from
    the repo's .env file the first time it runs in a process. Deleting the
    env var before that import would just have it reloaded from .env;
    deleting it after guarantees the bearer middleware (which reads
    os.getenv on every request) sees it unset.
    """
    import api.routes as routes

    monkeypatch.delenv("TRADING_API_KEY", raising=False)
    monkeypatch.setattr(routes, "trading_engine", None)
    monkeypatch.setenv("DESK_DATA_DIR", str(desk_dir))

    client = TestClient(routes.app)
    return client, routes


def _today_et() -> str:
    return datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")


def _sample_plan(date: str) -> DailyPlan:
    return DailyPlan(
        date=date,
        regime="risk_on",
        spy_change_pct=0.8,
        vix=14.2,
        vix_change_pct=-2.0,
        breadth_score=0.65,
        calendar_flags=("low:08:30:CPI",),
        long_ok=True,
        short_ok=False,
        risk_budget_pct=1.0,
        max_trades=6,
        sit_out=False,
        sit_out_reason="",
        in_play=(
            InPlayRow(symbol="NVDA", score=0.9, rvol_open15=3.0, gap_pct=2.5,
                      catalyst=1.0, rs_20=8.0, reasons=("rvol",)),
        ),
        notes=("regime risk_on",),
        built_at=f"{date}T09:00:00-04:00",
        refreshed_at="",
    )


# ──────────────────────────────────────────────────────────────────────
# GET /api/desk/plan
# ──────────────────────────────────────────────────────────────────────

class TestDeskPlanEndpoint:
    def test_plan_present_returns_ok_true_and_plan_dict(self, tmp_path, monkeypatch):
        date = "2026-09-24"
        plan = _sample_plan(date)
        save_plan(plan, str(tmp_path))

        client, _routes = _make_client(monkeypatch, tmp_path)
        response = client.get("/api/desk/plan", params={"date": date})

        assert response.status_code == 200
        body = response.json()
        assert body["ok"] is True
        assert body["date"] == date
        assert body["plan"]["regime"] == "risk_on"
        assert body["plan"]["max_trades"] == 6
        assert body["plan"]["in_play"][0]["symbol"] == "NVDA"

    def test_plan_missing_returns_ok_false(self, tmp_path, monkeypatch):
        client, _routes = _make_client(monkeypatch, tmp_path)
        response = client.get("/api/desk/plan", params={"date": "2026-09-24"})

        assert response.status_code == 200
        body = response.json()
        assert body["ok"] is False
        assert body["error"] == "no_plan"

    def test_plan_bad_date_returns_ok_false_without_touching_disk(self, tmp_path, monkeypatch):
        client, _routes = _make_client(monkeypatch, tmp_path)
        response = client.get("/api/desk/plan", params={"date": "2026-13-40"})

        assert response.status_code == 200
        body = response.json()
        assert body["ok"] is False
        assert body["error"] == "invalid_date"

    def test_plan_traversal_looking_date_rejected(self, tmp_path, monkeypatch):
        client, _routes = _make_client(monkeypatch, tmp_path)
        response = client.get(
            "/api/desk/plan", params={"date": "../../../../etc/passwd"}
        )

        assert response.status_code == 200
        body = response.json()
        assert body["ok"] is False
        assert body["error"] == "invalid_date"
        # No file was ever created outside DESK_DATA_DIR's plans subdir.
        assert not (tmp_path / "plans").exists() or not any(
            (tmp_path / "plans").iterdir()
        )

    def test_plan_default_date_is_today_et(self, tmp_path, monkeypatch):
        date = _today_et()
        plan = _sample_plan(date)
        save_plan(plan, str(tmp_path))

        client, _routes = _make_client(monkeypatch, tmp_path)
        response = client.get("/api/desk/plan")

        assert response.status_code == 200
        body = response.json()
        assert body["ok"] is True
        assert body["date"] == date


# ──────────────────────────────────────────────────────────────────────
# GET /api/desk/playbook
# ──────────────────────────────────────────────────────────────────────

class TestDeskPlaybookEndpoint:
    def test_playbook_present_returns_ok_true_and_entries(self, tmp_path, monkeypatch):
        entry = PlaybookEntry(
            family="orb", side="long", regime="risk_on", status="enabled",
            size_mult=1.0, exp_r_recent=0.2, t_stat_recent=1.5, n_recent=45,
            updated_at="2026-09-24T21:00:00+00:00", reason="enabled_normal",
        )
        playbook = Playbook().with_entry(entry).with_updated_at("2026-09-24T21:00:00+00:00")
        save_playbook(playbook, playbook_path(str(tmp_path)))

        client, _routes = _make_client(monkeypatch, tmp_path)
        response = client.get("/api/desk/playbook")

        assert response.status_code == 200
        body = response.json()
        assert body["ok"] is True
        assert "orb|long|risk_on" in body["playbook"]["entries"]
        assert body["playbook"]["entries"]["orb|long|risk_on"]["status"] == "enabled"

    def test_playbook_missing_returns_ok_false(self, tmp_path, monkeypatch):
        client, _routes = _make_client(monkeypatch, tmp_path)
        response = client.get("/api/desk/playbook")

        assert response.status_code == 200
        body = response.json()
        assert body["ok"] is False
        assert body["error"] == "no_playbook"


# ──────────────────────────────────────────────────────────────────────
# GET /api/desk/journal
# ──────────────────────────────────────────────────────────────────────

class TestDeskJournalEndpoint:
    def test_journal_present_returns_ok_true_and_markdown(self, tmp_path, monkeypatch):
        date = "2026-09-24"
        write_journal("# Desk Journal 2026-09-24\n\nno trades.", str(tmp_path), date)

        client, _routes = _make_client(monkeypatch, tmp_path)
        response = client.get("/api/desk/journal", params={"date": date})

        assert response.status_code == 200
        body = response.json()
        assert body["ok"] is True
        assert body["date"] == date
        assert "Desk Journal 2026-09-24" in body["markdown"]

    def test_journal_missing_returns_ok_false_no_journal(self, tmp_path, monkeypatch):
        client, _routes = _make_client(monkeypatch, tmp_path)
        response = client.get("/api/desk/journal", params={"date": "2026-09-24"})

        assert response.status_code == 200
        body = response.json()
        assert body["ok"] is False
        assert body["error"] == "no_journal"

    def test_journal_bad_date_returns_ok_false(self, tmp_path, monkeypatch):
        client, _routes = _make_client(monkeypatch, tmp_path)
        response = client.get("/api/desk/journal", params={"date": "not-a-date"})

        assert response.status_code == 200
        body = response.json()
        assert body["ok"] is False
        assert body["error"] == "invalid_date"
