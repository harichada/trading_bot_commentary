"""Tests for multitenant_app.py — clean multi-tenant FastAPI application."""

import os
import uuid
from unittest.mock import patch, MagicMock, AsyncMock

os.environ.setdefault("DATABASE_URL", "postgresql://rudra:rudra_dev_2024@localhost:5432/rudra_dev")

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from multitenant_app import app


@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ── Health ──────────────────────────────────────────────────────────────────

class TestHealth:
    @pytest.mark.asyncio
    async def test_health_returns_ok(self, client):
        resp = await client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "uptime_seconds" in data

    @pytest.mark.asyncio
    async def test_health_shows_not_configured_without_auth(self, client):
        resp = await client.get("/api/health")
        data = resp.json()
        # Without auth cookie, trader_status should be not_configured
        assert data["trader_status"] in ("not_configured", "stopped")


# ── Auth Required ───────────────────────────────────────────────────────────

class TestAuthRequired:
    @pytest.mark.asyncio
    async def test_state_requires_auth(self, client):
        resp = await client.get("/api/state")
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_trades_requires_auth(self, client):
        resp = await client.get("/api/trades")
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_config_requires_auth(self, client):
        resp = await client.get("/api/config")
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_start_requires_auth(self, client):
        resp = await client.post("/api/start")
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_scan_requires_auth(self, client):
        resp = await client.post("/api/scan")
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_metrics_requires_auth(self, client):
        resp = await client.get("/api/metrics")
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_tracker_requires_auth(self, client):
        resp = await client.get("/api/tracker/positions")
        assert resp.status_code == 401


# ── Auth Exempt ─────────────────────────────────────────────────────────────

class TestAuthExempt:
    @pytest.mark.asyncio
    async def test_health_no_auth(self, client):
        resp = await client.get("/api/health")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_auth_me_no_auth(self, client):
        resp = await client.get("/api/auth/me")
        # With auth enabled, returns 401 with auth_enabled: true
        # With auth disabled, returns 200 with user: null
        assert resp.status_code in (200, 401)
        data = resp.json()
        assert "auth_enabled" in data

    @pytest.mark.asyncio
    async def test_docs_no_auth(self, client):
        resp = await client.get("/docs")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_root_no_auth(self, client):
        resp = await client.get("/")
        assert resp.status_code == 200


# ── Authenticated but No Credentials ───────────────────────────────────────

def _mock_auth_user():
    """Mock an authenticated user with no broker credentials."""
    return {
        "sub": "testuser@test.com",
        "name": "Test User",
        "role": "admin",
        "user_id": str(uuid.uuid4()),
        "tier": "free",
        "is_admin": True,
    }


class TestNoCredentials:
    """User is authenticated but has not stored broker credentials."""

    @pytest.mark.asyncio
    async def test_state_returns_setup_required(self, client):
        with patch("multitenant_app.get_current_user", return_value=_mock_auth_user()), \
             patch("multitenant_app.auth_enabled", return_value=True):
            resp = await client.get("/api/state")
            assert resp.status_code == 200
            data = resp.json()
            assert data.get("needs_setup") or data.get("setup_required") or data["status"] == "not_configured"

    @pytest.mark.asyncio
    async def test_trades_returns_empty(self, client):
        with patch("multitenant_app.get_current_user", return_value=_mock_auth_user()), \
             patch("multitenant_app.auth_enabled", return_value=True):
            resp = await client.get("/api/trades")
            assert resp.status_code == 200
            data = resp.json()
            assert data["trades"] == []
            assert data.get("needs_setup") is True

    @pytest.mark.asyncio
    async def test_metrics_returns_empty(self, client):
        with patch("multitenant_app.get_current_user", return_value=_mock_auth_user()), \
             patch("multitenant_app.auth_enabled", return_value=True):
            resp = await client.get("/api/metrics")
            assert resp.status_code == 200
            data = resp.json()
            assert data["total_trades"] == 0
            assert data.get("needs_setup") is True

    @pytest.mark.asyncio
    async def test_equity_curve_returns_empty(self, client):
        with patch("multitenant_app.get_current_user", return_value=_mock_auth_user()), \
             patch("multitenant_app.auth_enabled", return_value=True):
            resp = await client.get("/api/equity_curve")
            assert resp.status_code == 200
            data = resp.json()
            assert data["dates"] == []

    @pytest.mark.asyncio
    async def test_start_returns_setup_error(self, client):
        with patch("multitenant_app.get_current_user", return_value=_mock_auth_user()), \
             patch("multitenant_app.auth_enabled", return_value=True):
            resp = await client.post("/api/start")
            assert resp.status_code == 400
            assert "credentials" in resp.json()["error"].lower()

    @pytest.mark.asyncio
    async def test_tracker_returns_empty(self, client):
        with patch("multitenant_app.get_current_user", return_value=_mock_auth_user()), \
             patch("multitenant_app.auth_enabled", return_value=True):
            resp = await client.get("/api/tracker/positions")
            assert resp.status_code == 200
            data = resp.json()
            assert data["positions"] == []
            assert data.get("needs_setup") is True

    @pytest.mark.asyncio
    async def test_config_returns_defaults(self, client):
        with patch("multitenant_app.get_current_user", return_value=_mock_auth_user()), \
             patch("multitenant_app.auth_enabled", return_value=True):
            resp = await client.get("/api/config")
            assert resp.status_code == 200
            data = resp.json()
            assert "config" in data
            assert data.get("needs_setup") is True

    @pytest.mark.asyncio
    async def test_strategies_returns_list(self, client):
        with patch("multitenant_app.get_current_user", return_value=_mock_auth_user()), \
             patch("multitenant_app.auth_enabled", return_value=True):
            resp = await client.get("/api/strategies")
            assert resp.status_code == 200
            data = resp.json()
            assert "strategies" in data
            assert "active" in data


# ── Response Shape Validation ───────────────────────────────────────────────

class TestResponseShapes:
    """Verify API responses match what the frontend expects."""

    @pytest.mark.asyncio
    async def test_health_shape(self, client):
        resp = await client.get("/api/health")
        data = resp.json()
        for key in ["status", "version", "uptime_seconds", "trader_status", "equity", "daily_pnl", "open_positions"]:
            assert key in data, f"Missing key: {key}"

    @pytest.mark.asyncio
    async def test_no_engine_state_shape(self):
        from user_engine import NO_ENGINE_STATE
        assert "status" in NO_ENGINE_STATE
        assert "equity" in NO_ENGINE_STATE
        assert "positions" in NO_ENGINE_STATE
        assert "needs_setup" in NO_ENGINE_STATE or "setup_required" in NO_ENGINE_STATE


# ── User Management Endpoints ───────────────────────────────────────────────

class TestUserEndpoints:
    @pytest.mark.asyncio
    async def test_users_me_requires_auth(self, client):
        resp = await client.get("/api/users/me")
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_admin_users_requires_auth(self, client):
        resp = await client.get("/api/admin/users")
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_admin_tiers_requires_auth(self, client):
        resp = await client.get("/api/admin/tiers")
        assert resp.status_code == 401


# ── SPA Serving ─────────────────────────────────────────────────────────────

class TestSPA:
    @pytest.mark.asyncio
    async def test_root_returns_html(self, client):
        resp = await client.get("/")
        assert resp.status_code == 200
        # Should return index.html or JSON fallback
        content_type = resp.headers.get("content-type", "")
        assert "html" in content_type or "json" in content_type

    @pytest.mark.asyncio
    async def test_unknown_path_returns_spa(self, client):
        resp = await client.get("/scanner")
        assert resp.status_code == 200
        # SPA catch-all should return index.html

    @pytest.mark.asyncio
    async def test_api_unknown_returns_error(self, client):
        resp = await client.get("/api/nonexistent")
        # Either 401 (auth required) or 404
        assert resp.status_code in (401, 404, 405)
