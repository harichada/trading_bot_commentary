"""Tests for API authentication module (api/auth.py).

TDD: These tests are written BEFORE the implementation.
They define the expected contract for verify_api_key and verify_ws_token.
"""
import os
import pytest
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers – build a minimal FastAPI app wired up with the auth dependency
# ---------------------------------------------------------------------------

def _make_app():
    """Create a throwaway FastAPI app that uses verify_api_key as a dependency."""
    from api.auth import verify_api_key

    app = FastAPI()

    @app.get("/protected")
    def protected_route(key=pytest.importorskip("fastapi").Depends(verify_api_key)):  # noqa: B008
        return {"ok": True}

    return app


# ---------------------------------------------------------------------------
# verify_api_key – HTTP / Bearer token tests
# ---------------------------------------------------------------------------

class TestVerifyApiKey:
    """Integration tests for the verify_api_key FastAPI dependency."""

    def test_no_env_var_allows_unauthenticated_request(self, monkeypatch):
        """When TRADING_API_KEY is unset, any request is allowed (dev/graceful degradation)."""
        monkeypatch.delenv("TRADING_API_KEY", raising=False)

        # Re-import after env change so the module sees the new env state.
        import importlib
        import api.auth as auth_mod
        importlib.reload(auth_mod)

        from api.auth import verify_api_key
        from fastapi import FastAPI, Depends
        from fastapi.testclient import TestClient

        app = FastAPI()

        @app.get("/protected")
        def route(key=Depends(verify_api_key)):
            return {"ok": True}

        client = TestClient(app)
        response = client.get("/protected")
        assert response.status_code == 200
        assert response.json() == {"ok": True}

    def test_correct_bearer_token_allows_request(self, monkeypatch):
        """A correct Bearer token passes authentication."""
        monkeypatch.setenv("TRADING_API_KEY", "secret-token-abc")

        import importlib
        import api.auth as auth_mod
        importlib.reload(auth_mod)

        from api.auth import verify_api_key
        from fastapi import FastAPI, Depends
        from fastapi.testclient import TestClient

        app = FastAPI()

        @app.get("/protected")
        def route(key=Depends(verify_api_key)):
            return {"ok": True}

        client = TestClient(app)
        response = client.get("/protected", headers={"Authorization": "Bearer secret-token-abc"})
        assert response.status_code == 200
        assert response.json() == {"ok": True}

    def test_wrong_bearer_token_returns_401(self, monkeypatch):
        """A wrong Bearer token is rejected with HTTP 401."""
        monkeypatch.setenv("TRADING_API_KEY", "secret-token-abc")

        import importlib
        import api.auth as auth_mod
        importlib.reload(auth_mod)

        from api.auth import verify_api_key
        from fastapi import FastAPI, Depends
        from fastapi.testclient import TestClient

        app = FastAPI()

        @app.get("/protected")
        def route(key=Depends(verify_api_key)):
            return {"ok": True}

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/protected", headers={"Authorization": "Bearer wrong-token"})
        assert response.status_code == 401

    def test_missing_auth_header_returns_401_when_key_is_set(self, monkeypatch):
        """When TRADING_API_KEY is set but no Authorization header is sent, return 401."""
        monkeypatch.setenv("TRADING_API_KEY", "secret-token-abc")

        import importlib
        import api.auth as auth_mod
        importlib.reload(auth_mod)

        from api.auth import verify_api_key
        from fastapi import FastAPI, Depends
        from fastapi.testclient import TestClient

        app = FastAPI()

        @app.get("/protected")
        def route(key=Depends(verify_api_key)):
            return {"ok": True}

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/protected")
        assert response.status_code == 401

    def test_empty_bearer_token_returns_401_when_key_is_set(self, monkeypatch):
        """An empty/blank Bearer token is rejected with 401 when auth is enabled."""
        monkeypatch.setenv("TRADING_API_KEY", "secret-token-abc")

        import importlib
        import api.auth as auth_mod
        importlib.reload(auth_mod)

        from api.auth import verify_api_key
        from fastapi import FastAPI, Depends
        from fastapi.testclient import TestClient

        app = FastAPI()

        @app.get("/protected")
        def route(key=Depends(verify_api_key)):
            return {"ok": True}

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/protected", headers={"Authorization": "Bearer "})
        assert response.status_code == 401

    def test_timing_safe_comparison_used(self, monkeypatch):
        """verify_api_key must use secrets.compare_digest, not == operator."""
        import ast
        import inspect
        import api.auth as auth_mod

        source = inspect.getsource(auth_mod.verify_api_key)
        # Ensure the source uses secrets.compare_digest and NOT plain == comparison
        assert "compare_digest" in source, "Must use secrets.compare_digest for timing safety"
        # The raw == should not appear in the function body between token values
        # (comparing credentials.credentials == api_key is forbidden)
        assert "credentials.credentials ==" not in source


# ---------------------------------------------------------------------------
# verify_ws_token – WebSocket token tests
# ---------------------------------------------------------------------------

class TestVerifyWsToken:
    """Unit tests for the verify_ws_token helper function."""

    def test_correct_token_returns_true(self, monkeypatch):
        """verify_ws_token returns True when the token matches TRADING_API_KEY."""
        monkeypatch.setenv("TRADING_API_KEY", "ws-secret-xyz")

        import importlib
        import api.auth as auth_mod
        importlib.reload(auth_mod)

        assert auth_mod.verify_ws_token("ws-secret-xyz") is True

    def test_no_env_var_returns_true(self, monkeypatch):
        """verify_ws_token returns True when TRADING_API_KEY is unset (dev mode)."""
        monkeypatch.delenv("TRADING_API_KEY", raising=False)

        import importlib
        import api.auth as auth_mod
        importlib.reload(auth_mod)

        assert auth_mod.verify_ws_token("any-token") is True
        assert auth_mod.verify_ws_token(None) is True
        assert auth_mod.verify_ws_token("") is True

    def test_wrong_token_returns_false(self, monkeypatch):
        """verify_ws_token returns False when the token does not match."""
        monkeypatch.setenv("TRADING_API_KEY", "ws-secret-xyz")

        import importlib
        import api.auth as auth_mod
        importlib.reload(auth_mod)

        assert auth_mod.verify_ws_token("wrong-token") is False

    def test_none_token_returns_false_when_key_set(self, monkeypatch):
        """verify_ws_token returns False when None is passed and auth is enabled."""
        monkeypatch.setenv("TRADING_API_KEY", "ws-secret-xyz")

        import importlib
        import api.auth as auth_mod
        importlib.reload(auth_mod)

        assert auth_mod.verify_ws_token(None) is False

    def test_empty_token_returns_false_when_key_set(self, monkeypatch):
        """verify_ws_token returns False when empty string is passed and auth is enabled."""
        monkeypatch.setenv("TRADING_API_KEY", "ws-secret-xyz")

        import importlib
        import api.auth as auth_mod
        importlib.reload(auth_mod)

        assert auth_mod.verify_ws_token("") is False


# ---------------------------------------------------------------------------
# Config integration – TRADING_API_KEY property
# ---------------------------------------------------------------------------

class TestConfigTradingApiKey:
    """Tests for the TRADING_API_KEY property on Config."""

    def test_returns_none_when_unset(self, monkeypatch):
        """Config.TRADING_API_KEY returns None when env var is not set."""
        monkeypatch.delenv("TRADING_API_KEY", raising=False)
        # TRADING_API_KEY property reads os.getenv at call time; no module
        # reload needed — monkeypatch already patches the live env.
        from core.config import Config
        c = Config()
        assert c.TRADING_API_KEY is None

    def test_returns_value_when_set(self, monkeypatch):
        """Config.TRADING_API_KEY returns the env var value when set."""
        monkeypatch.setenv("TRADING_API_KEY", "my-api-key")
        from core.config import Config
        c = Config()
        assert c.TRADING_API_KEY == "my-api-key"
