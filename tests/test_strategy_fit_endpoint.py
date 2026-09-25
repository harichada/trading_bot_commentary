"""Tests for GET /api/strategy-fit (api/routes.py).

TDD: written BEFORE the route exists. Follows the client construction
pattern from tests/test_api_auth.py: build a TestClient over the real
FastAPI app in api.routes, with TRADING_API_KEY unset so the bearer
middleware allows unauthenticated requests.
"""
import json
from datetime import datetime

import pytest
from fastapi.testclient import TestClient


def _make_client(monkeypatch):
    """Import api.routes and return a TestClient with auth disabled.

    Import first, then delenv: importing api.routes pulls in
    core.config, whose module-level `load_dotenv()` call populates
    TRADING_API_KEY from the repo's .env file the first time it runs
    in a process. Deleting the env var before that import would just
    have it reloaded from .env; deleting it after guarantees the
    bearer middleware (which reads os.getenv on every request) sees
    it unset.
    """
    import api.routes as routes

    monkeypatch.delenv("TRADING_API_KEY", raising=False)
    monkeypatch.setattr(routes, "trading_engine", None)

    client = TestClient(routes.app)
    return client, routes


def _write_snapshot(tmp_path, rows):
    snapshot = {
        "computed_at": datetime.now().isoformat(),
        "floors": {},
        "rows": rows,
    }
    p = tmp_path / "strategy_fit.json"
    p.write_text(json.dumps(snapshot))
    return p


def test_strategy_fit_endpoint_returns_rows_and_summary(tmp_path, monkeypatch):
    row = {
        "symbol": "NVDA",
        "strategy": "orb",
        "side": "long",
        "verdict": "PASS",
        "score": 0.5,
        "exp_r": 0.5,
        "exp_r_lb": 0.1,
        "t_stat": 2.5,
        "pf": 1.4,
        "wr": 0.5,
        "stability": 0.7,
        "decay": 0.9,
        "n": 200,
        "source": "walkforward",
        "params_json": json.dumps({"k": 1}),
        "computed_at": datetime.now().isoformat(),
    }
    snapshot_path = _write_snapshot(tmp_path, {"NVDA": {"orb:long": row}})
    monkeypatch.setenv("STRATEGY_FIT_PATH", str(snapshot_path))

    client, _routes = _make_client(monkeypatch)

    response = client.get("/api/strategy-fit", params={"symbol": "NVDA"})

    assert response.status_code == 200
    body = response.json()
    assert len(body["rows"]) == 1
    assert body["rows"][0]["symbol"] == "NVDA"
    assert body["rows"][0]["strategy"] == "orb"
    assert body["summary"]["counts"]["PASS"] == 1


def test_strategy_fit_endpoint_missing_path_yields_empty_rows_and_error(tmp_path, monkeypatch):
    missing_path = tmp_path / "does_not_exist.json"
    monkeypatch.setenv("STRATEGY_FIT_PATH", str(missing_path))

    client, _routes = _make_client(monkeypatch)

    response = client.get("/api/strategy-fit", params={"symbol": "NVDA"})

    assert response.status_code == 200
    body = response.json()
    assert body["rows"] == []
    assert body["summary"]["last_error"]


def test_strategy_fit_endpoint_without_symbol_returns_empty_rows(tmp_path, monkeypatch):
    row = {
        "symbol": "NVDA",
        "strategy": "orb",
        "side": "long",
        "verdict": "PASS",
        "score": 0.5,
        "exp_r": 0.5,
        "exp_r_lb": 0.1,
        "t_stat": 2.5,
        "pf": 1.4,
        "wr": 0.5,
        "stability": 0.7,
        "decay": 0.9,
        "n": 200,
        "source": "walkforward",
        "params_json": json.dumps({"k": 1}),
        "computed_at": datetime.now().isoformat(),
    }
    snapshot_path = _write_snapshot(tmp_path, {"NVDA": {"orb:long": row}})
    monkeypatch.setenv("STRATEGY_FIT_PATH", str(snapshot_path))

    client, _routes = _make_client(monkeypatch)

    response = client.get("/api/strategy-fit")

    assert response.status_code == 200
    body = response.json()
    assert body["rows"] == []


# ──────────────────────────────────────────────────────────────────────
# v-strategy-fit-review2-2026-09-25 — Important 5: NaN metrics and symbol
# case.
# ──────────────────────────────────────────────────────────────────────

def test_strategy_fit_endpoint_coldstart_row_with_nan_metrics_returns_200(tmp_path, monkeypatch):
    """A cold-start row carries NaN for every metric the walk-forward pipeline
    never measured. Starlette's JSONResponse serializes with allow_nan=False,
    so bare NaN raised ValueError and the endpoint answered 500. Non-finite
    floats must come back as JSON null instead.
    """
    row = {
        "symbol": "NVDA",
        "strategy": "orb",
        "side": "long",
        "verdict": "WATCH",
        "source": "coldstart",
        "n": 0,
        "params_json": json.dumps({}),
        "computed_at": datetime.now().isoformat(),
        # score / exp_r / exp_r_lb / t_stat / pf / wr / stability / decay all
        # absent -> FitRow.from_dict defaults each to nan.
    }
    snapshot_path = _write_snapshot(tmp_path, {"NVDA": {"orb:long": row}})
    monkeypatch.setenv("STRATEGY_FIT_PATH", str(snapshot_path))

    client, _routes = _make_client(monkeypatch)
    response = client.get("/api/strategy-fit", params={"symbol": "NVDA"})

    assert response.status_code == 200
    body = response.json()
    assert len(body["rows"]) == 1
    r = body["rows"][0]
    for metric in ("score", "exp_r", "exp_r_lb", "t_stat", "pf", "wr", "stability", "decay"):
        assert r[metric] is None, f"{metric} should be null, got {r[metric]!r}"
    # Non-numeric fields survive untouched.
    assert r["verdict"] == "WATCH" and r["source"] == "coldstart" and r["n"] == 0


def test_strategy_fit_endpoint_uppercases_symbol_query_param(tmp_path, monkeypatch):
    """Snapshot keys are upper-case symbols, so a lower-case query param used
    to silently return zero rows."""
    row = {
        "symbol": "NVDA", "strategy": "orb", "side": "long", "verdict": "PASS", "score": 0.5,
        "exp_r": 0.5, "exp_r_lb": 0.1, "t_stat": 2.5, "pf": 1.4, "wr": 0.5, "stability": 0.7,
        "decay": 0.9, "n": 200, "source": "walkforward", "params_json": json.dumps({}),
        "computed_at": datetime.now().isoformat(),
    }
    snapshot_path = _write_snapshot(tmp_path, {"NVDA": {"orb:long": row}})
    monkeypatch.setenv("STRATEGY_FIT_PATH", str(snapshot_path))

    client, _routes = _make_client(monkeypatch)
    for param in ("nvda", "Nvda", "NVDA"):
        response = client.get("/api/strategy-fit", params={"symbol": param})
        assert response.status_code == 200
        assert len(response.json()["rows"]) == 1, f"no rows for symbol={param!r}"


def test_fit_row_to_json_dict_replaces_non_finite_floats():
    from core.strategy_fit import FitRow

    row = FitRow.from_dict({"symbol": "NVDA", "strategy": "orb", "side": "long", "verdict": "WATCH",
                            "source": "coldstart", "n": 3, "pf": float("inf"), "wr": 0.5})
    d = row.to_json_dict()
    assert d["score"] is None and d["exp_r"] is None
    assert d["pf"] is None                      # +inf is non-finite too
    assert d["wr"] == 0.5 and d["n"] == 3 and d["symbol"] == "NVDA"
    json.dumps(d, allow_nan=False)              # must not raise
