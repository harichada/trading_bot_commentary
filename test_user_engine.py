"""Tests for user_engine.py — per-user engine pool and profit cap."""

import os
import uuid
from unittest.mock import patch, MagicMock, AsyncMock
import pytest

os.environ.setdefault("DATABASE_URL", "postgresql://rudra:rudra_dev_2024@localhost:5432/rudra_dev")


class TestProfitCapChecker:
    """Test profit cap enforcement."""

    def setup_method(self):
        from user_engine import ProfitCapChecker
        self.checker = ProfitCapChecker()

    def test_system_user_always_allowed(self):
        from user_engine import SYSTEM_USER_ID
        ok, reason = self.checker.can_enter(SYSTEM_USER_ID or "")
        assert ok is True
        assert reason == ""

    def test_none_user_allowed(self):
        ok, reason = self.checker.can_enter(None)
        assert ok is True

    def test_empty_user_allowed(self):
        ok, reason = self.checker.can_enter("")
        assert ok is True

    def test_free_tier_under_cap(self):
        """User with $100 monthly P&L on free tier ($200 cap) should be allowed."""
        with patch("user_management.UserRepository.get_monthly_pnl") as mock_pnl:
            mock_pnl.return_value = {"monthly_pnl": 100, "tier": "free"}
            ok, reason = self.checker.can_enter("test-user-id")
            assert ok is True

    def test_free_tier_at_cap(self):
        """User at $200 monthly P&L on free tier should be blocked."""
        with patch("user_management.UserRepository.get_monthly_pnl") as mock_pnl:
            mock_pnl.return_value = {"monthly_pnl": 200, "tier": "free"}
            ok, reason = self.checker.can_enter("test-user-id")
            assert ok is False
            assert "cap reached" in reason

    def test_enterprise_unlimited(self):
        """Enterprise tier has no cap."""
        with patch("user_management.UserRepository.get_monthly_pnl") as mock_pnl:
            mock_pnl.return_value = {"monthly_pnl": 999999, "tier": "enterprise"}
            ok, reason = self.checker.can_enter("test-user-id")
            assert ok is True

    def test_pro_tier_under_cap(self):
        """Pro tier with $4000 P&L (cap $5000) should be allowed."""
        with patch("user_management.UserRepository.get_monthly_pnl") as mock_pnl:
            mock_pnl.return_value = {"monthly_pnl": 4000, "tier": "pro"}
            ok, reason = self.checker.can_enter("test-user-id")
            assert ok is True

    def test_db_error_fails_open(self):
        """If DB check fails, allow entry (fail-open)."""
        with patch("user_management.UserRepository.get_monthly_pnl") as mock_pnl:
            mock_pnl.side_effect = Exception("DB down")
            ok, reason = self.checker.can_enter("test-user-id")
            assert ok is True


class TestResolveUserId:
    """Test user_id resolution from request."""

    def test_auth_disabled_returns_system(self):
        from user_engine import resolve_user_id, SYSTEM_USER_ID
        mock_req = MagicMock()
        with patch("auth.auth_enabled", return_value=False):
            uid = resolve_user_id(mock_req)
            assert uid == (SYSTEM_USER_ID or "")

    def test_auth_enabled_no_jwt_returns_system(self):
        from user_engine import resolve_user_id, SYSTEM_USER_ID
        mock_req = MagicMock()
        with patch("auth.auth_enabled", return_value=True), \
             patch("auth.get_current_user", return_value=None):
            uid = resolve_user_id(mock_req)
            assert uid == (SYSTEM_USER_ID or "")

    def test_auth_enabled_with_jwt_returns_user_id(self):
        from user_engine import resolve_user_id
        mock_req = MagicMock()
        test_uid = str(uuid.uuid4())
        with patch("auth.auth_enabled", return_value=True), \
             patch("auth.get_current_user", return_value={"user_id": test_uid}):
            uid = resolve_user_id(mock_req)
            assert uid == test_uid


class TestUserEngineManager:
    """Test engine pool management."""

    def setup_method(self):
        from user_engine import UserEngineManager
        self.mock_global = MagicMock()
        self.mock_global.status = "running"
        self.mgr = UserEngineManager(self.mock_global)

    @pytest.mark.asyncio
    async def test_system_user_gets_global_trader(self):
        from user_engine import SYSTEM_USER_ID
        trader = await self.mgr.get_trader(SYSTEM_USER_ID or "")
        assert trader is self.mock_global

    @pytest.mark.asyncio
    async def test_none_user_gets_global_trader(self):
        trader = await self.mgr.get_trader(None)
        assert trader is self.mock_global

    @pytest.mark.asyncio
    async def test_empty_user_gets_global_trader(self):
        trader = await self.mgr.get_trader("")
        assert trader is self.mock_global

    @pytest.mark.asyncio
    async def test_unknown_user_no_creds_gets_global(self):
        """User with no credentials falls back to global trader."""
        test_uid = str(uuid.uuid4())
        with patch("user_management.CredentialManager.get_all_for_user", return_value=[]):
            trader = await self.mgr.get_trader(test_uid)
            assert trader is self.mock_global

    @pytest.mark.asyncio
    async def test_engine_count(self):
        assert self.mgr.engine_count == 0

    @pytest.mark.asyncio
    async def test_get_all_active_empty(self):
        assert self.mgr.get_all_active() == {}

    @pytest.mark.asyncio
    async def test_global_trader_property(self):
        assert self.mgr.global_trader is self.mock_global

    @pytest.mark.asyncio
    async def test_remove_nonexistent(self):
        result = await self.mgr.remove_trader("nonexistent")
        assert result is False


class TestMonthlyPnl:
    """Test monthly P&L tracking in UserRepository."""

    def setup_method(self):
        from user_management import UserRepository
        self.repo = UserRepository()
        self.test_email = f"pnl_test_{uuid.uuid4().hex[:8]}@test.local"
        user = self.repo.upsert_on_login(self.test_email, "PnL Test", "", "google")
        self.user_id = str(user["user_id"])

    def teardown_method(self):
        from user_management import _execute
        _execute("DELETE FROM subscriptions WHERE user_id = %s", (self.user_id,), fetch="none")
        _execute("DELETE FROM users WHERE email = %s", (self.test_email,), fetch="none")

    def test_update_increments_pnl(self):
        result = self.repo.update_monthly_pnl(self.user_id, 50.0)
        assert result is not None
        assert float(result["monthly_pnl"]) == 50.0

        result2 = self.repo.update_monthly_pnl(self.user_id, 30.0)
        assert float(result2["monthly_pnl"]) == 80.0

    def test_get_monthly_pnl(self):
        self.repo.update_monthly_pnl(self.user_id, 100.0)
        info = self.repo.get_monthly_pnl(self.user_id)
        assert info is not None
        assert float(info["monthly_pnl"]) == 100.0
        assert info["tier"] == "free"

    def test_negative_pnl(self):
        """Losses should decrease the P&L."""
        self.repo.update_monthly_pnl(self.user_id, 100.0)
        result = self.repo.update_monthly_pnl(self.user_id, -30.0)
        assert float(result["monthly_pnl"]) == 70.0


class TestCredentialGetAll:
    """Test get_all_for_user on CredentialManager."""

    def setup_method(self):
        from cryptography.fernet import Fernet
        self.key = Fernet.generate_key().decode()
        os.environ["CREDENTIAL_ENCRYPTION_KEY"] = self.key
        import user_management
        user_management._cred_mgr = None
        from user_management import CredentialManager, UserRepository
        self.mgr = CredentialManager()
        self.repo = UserRepository()
        self.test_email = f"getall_test_{uuid.uuid4().hex[:8]}@test.local"
        user = self.repo.upsert_on_login(self.test_email, "GetAll Test", "", "google")
        self.user_id = str(user["user_id"])

    def teardown_method(self):
        from user_management import _execute
        _execute("DELETE FROM user_credentials WHERE user_id = %s", (self.user_id,), fetch="none")
        _execute("DELETE FROM users WHERE email = %s", (self.test_email,), fetch="none")
        os.environ.pop("CREDENTIAL_ENCRYPTION_KEY", None)

    def test_get_all_empty(self):
        result = self.mgr.get_all_for_user(self.user_id)
        assert result == []

    def test_get_all_with_credentials(self):
        self.mgr.store(self.user_id, "alpaca", {"api_key": "PK1", "secret_key": "SK1"}, "paper", True)
        self.mgr.store(self.user_id, "oanda", {"api_key": "OA1", "secret_key": "OS1"}, "default", True)
        result = self.mgr.get_all_for_user(self.user_id)
        assert len(result) == 2
        brokers = {c["broker"] for c in result}
        assert brokers == {"alpaca", "oanda"}
        # Credentials should be decrypted
        alpaca = next(c for c in result if c["broker"] == "alpaca")
        assert alpaca["credentials"]["api_key"] == "PK1"


class TestContextVarConfig:
    """Test that context var Alpaca config override works."""

    def test_default_returns_env_config(self):
        """Without context override, _get_alpaca_config reads env vars."""
        from gap_fade_app import _get_alpaca_config, _alpaca_config_ctx
        assert _alpaca_config_ctx.get() is None
        # Should work normally (returns config from env or None)
        cfg = _get_alpaca_config()
        # Don't assert specific values since env may not have Alpaca keys in test

    def test_context_override(self):
        """Context var override should take priority over env vars."""
        from gap_fade_app import _get_alpaca_config, _alpaca_config_ctx
        override = {"api_key": "TEST", "secret_key": "TEST_SECRET",
                     "trade_api_key": "TEST", "trade_secret_key": "TEST_SECRET",
                     "base_url": "https://test", "data_url": "https://test-data"}
        token = _alpaca_config_ctx.set(override)
        try:
            cfg = _get_alpaca_config()
            assert cfg["api_key"] == "TEST"
            assert cfg["base_url"] == "https://test"
        finally:
            _alpaca_config_ctx.reset(token)

        # After reset, should return normal config
        cfg2 = _get_alpaca_config()
        assert cfg2 is None or cfg2.get("api_key") != "TEST"
