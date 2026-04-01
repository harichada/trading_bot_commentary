"""Tests for user_management.py — multi-tenant user management."""

import json
import os
import uuid
from unittest.mock import patch, MagicMock

import pytest

# Ensure DATABASE_URL is set for tests (using the dev DB)
os.environ.setdefault("DATABASE_URL", "postgresql://rudra:rudra_dev_2024@localhost:5432/rudra_dev")


class TestUserRepository:
    """Test UserRepository CRUD operations."""

    def setup_method(self):
        from user_management import UserRepository
        self.repo = UserRepository()
        self.test_email = f"test_{uuid.uuid4().hex[:8]}@test.local"

    def teardown_method(self):
        """Clean up test user."""
        from user_management import _execute
        _execute("DELETE FROM users WHERE email = %s", (self.test_email,), fetch="none")

    def test_upsert_creates_new_user(self):
        result = self.repo.upsert_on_login(self.test_email, "Test User", "", "google")
        assert result is not None
        assert result["email"] == self.test_email
        assert result["name"] == "Test User"
        assert result["tier"] == "free"
        assert result["is_admin"] is False

    def test_upsert_updates_existing_user(self):
        self.repo.upsert_on_login(self.test_email, "First Name", "", "google")
        result = self.repo.upsert_on_login(self.test_email, "Updated Name", "pic.jpg", "github")
        assert result["name"] == "Updated Name"
        assert result["picture"] == "pic.jpg"
        assert result["provider"] == "github"

    def test_get_by_email(self):
        self.repo.upsert_on_login(self.test_email, "Test", "", "google")
        result = self.repo.get_by_email(self.test_email)
        assert result is not None
        assert result["email"] == self.test_email

    def test_get_by_email_not_found(self):
        result = self.repo.get_by_email("nonexistent@nowhere.local")
        assert result is None

    def test_get_by_id(self):
        created = self.repo.upsert_on_login(self.test_email, "Test", "", "google")
        result = self.repo.get_by_id(str(created["user_id"]))
        assert result is not None
        assert result["email"] == self.test_email

    def test_update_profile(self):
        created = self.repo.upsert_on_login(self.test_email, "Old", "", "google")
        result = self.repo.update_profile(
            str(created["user_id"]), "New Name", {"theme": "dark"}
        )
        assert result["name"] == "New Name"
        assert json.loads(result["settings_json"]) == {"theme": "dark"}

    def test_list_all(self):
        self.repo.upsert_on_login(self.test_email, "Test", "", "google")
        users = self.repo.list_all()
        assert isinstance(users, list)
        assert len(users) >= 1
        emails = [u["email"] for u in users]
        assert self.test_email in emails

    def test_update_tier(self):
        created = self.repo.upsert_on_login(self.test_email, "Test", "", "google")
        result = self.repo.update_tier(str(created["user_id"]), "pro")
        assert result["tier"] == "pro"

    def test_update_tier_invalid(self):
        created = self.repo.upsert_on_login(self.test_email, "Test", "", "google")
        result = self.repo.update_tier(str(created["user_id"]), "invalid_tier")
        assert result is None


class TestCredentialManager:
    """Test encrypted credential storage."""

    def setup_method(self):
        from cryptography.fernet import Fernet
        self.key = Fernet.generate_key().decode()
        os.environ["CREDENTIAL_ENCRYPTION_KEY"] = self.key
        # Reset singleton
        import user_management
        user_management._cred_mgr = None
        from user_management import CredentialManager, UserRepository
        self.mgr = CredentialManager()
        self.repo = UserRepository()
        self.test_email = f"cred_test_{uuid.uuid4().hex[:8]}@test.local"
        user = self.repo.upsert_on_login(self.test_email, "Cred Test", "", "google")
        self.user_id = str(user["user_id"])

    def teardown_method(self):
        from user_management import _execute
        _execute("DELETE FROM user_credentials WHERE user_id = %s", (self.user_id,), fetch="none")
        _execute("DELETE FROM users WHERE email = %s", (self.test_email,), fetch="none")
        os.environ.pop("CREDENTIAL_ENCRYPTION_KEY", None)

    def test_store_and_list(self):
        result = self.mgr.store(
            self.user_id, "alpaca",
            {"api_key": "PK123", "secret_key": "SK456"},
            "paper", True
        )
        assert result["broker"] == "alpaca"
        assert result["label"] == "paper"

        creds = self.mgr.list_for_user(self.user_id)
        assert len(creds) == 1
        # Secret should NOT appear in list
        assert "credentials_encrypted" not in creds[0]
        assert "api_key" not in str(creds[0])

    def test_encrypt_decrypt_roundtrip(self):
        original = {"api_key": "TEST_KEY_123", "secret_key": "SECRET_456"}
        stored = self.mgr.store(self.user_id, "alpaca", original, "test", True)
        decrypted = self.mgr.get_decrypted(self.user_id, stored["id"])
        assert decrypted is not None
        assert decrypted["credentials"] == original

    def test_cross_tenant_access_blocked(self):
        stored = self.mgr.store(self.user_id, "alpaca", {"key": "val"}, "default", True)
        # Try accessing with a different user_id
        fake_uid = str(uuid.uuid4())
        result = self.mgr.get_decrypted(fake_uid, stored["id"])
        assert result is None

    def test_delete(self):
        stored = self.mgr.store(self.user_id, "alpaca", {"key": "val"}, "default", True)
        assert self.mgr.delete(self.user_id, stored["id"]) is True
        assert self.mgr.list_for_user(self.user_id) == []

    def test_delete_nonexistent(self):
        assert self.mgr.delete(self.user_id, 99999) is False

    def test_no_encryption_key_raises(self):
        os.environ.pop("CREDENTIAL_ENCRYPTION_KEY", None)
        import user_management
        user_management._cred_mgr = None
        from user_management import CredentialManager
        mgr = CredentialManager()
        with pytest.raises(RuntimeError, match="not configured"):
            mgr.store(self.user_id, "alpaca", {"key": "v"}, "x", True)


class TestConfigManager:
    """Test per-user config management."""

    def setup_method(self):
        from user_management import ConfigManager, UserRepository
        self.mgr = ConfigManager()
        self.repo = UserRepository()
        self.test_email = f"cfg_test_{uuid.uuid4().hex[:8]}@test.local"
        user = self.repo.upsert_on_login(self.test_email, "Cfg Test", "", "google")
        self.user_id = str(user["user_id"])

    def teardown_method(self):
        from user_management import _execute
        _execute("DELETE FROM user_configs WHERE user_id = %s", (self.user_id,), fetch="none")
        _execute("DELETE FROM users WHERE email = %s", (self.test_email,), fetch="none")

    def test_get_default_config(self):
        result = self.mgr.get_config(self.user_id)
        assert result["config"] == {}
        assert result["active_strategy"] == "classic_gap_fade"
        assert result["updated_at"] is None

    def test_update_config_partial_merge(self):
        self.mgr.update_config(self.user_id, {"stop_pct": 0.02, "max_positions": 5})
        result = self.mgr.get_config(self.user_id)
        assert result["config"]["stop_pct"] == 0.02
        assert result["config"]["max_positions"] == 5

        # Partial update preserves old fields
        self.mgr.update_config(self.user_id, {"risk_pct": 0.01})
        result2 = self.mgr.get_config(self.user_id)
        assert result2["config"]["stop_pct"] == 0.02  # preserved
        assert result2["config"]["risk_pct"] == 0.01  # added

    def test_update_config_invalid_field(self):
        with pytest.raises(ValueError, match="Invalid config fields"):
            self.mgr.update_config(self.user_id, {"nonexistent_field": 42})


class TestSubscriptionManager:
    """Test subscription/tier info."""

    def setup_method(self):
        from user_management import SubscriptionManager, UserRepository
        self.mgr = SubscriptionManager()
        self.repo = UserRepository()
        self.test_email = f"sub_test_{uuid.uuid4().hex[:8]}@test.local"
        user = self.repo.upsert_on_login(self.test_email, "Sub Test", "", "google")
        self.user_id = str(user["user_id"])

    def teardown_method(self):
        from user_management import _execute
        _execute("DELETE FROM subscriptions WHERE user_id = %s", (self.user_id,), fetch="none")
        _execute("DELETE FROM users WHERE email = %s", (self.test_email,), fetch="none")

    def test_default_free_subscription(self):
        result = self.mgr.get_subscription(self.user_id)
        assert result["tier"] == "free"
        assert result["status"] == "active"
        assert result["monthly_profit_cap"] == 200

    def test_subscription_after_tier_change(self):
        self.repo.update_tier(self.user_id, "pro")
        result = self.mgr.get_subscription(self.user_id)
        assert result["tier"] == "pro"
        assert result["monthly_profit_cap"] == 5000


class TestTierLimits:
    """Test tier configuration."""

    def test_all_tiers_defined(self):
        from user_management import TIER_LIMITS, VALID_TIERS
        assert VALID_TIERS == {"free", "starter", "pro", "enterprise"}
        for tier in VALID_TIERS:
            assert tier in TIER_LIMITS
            limits = TIER_LIMITS[tier]
            assert "monthly_profit_cap" in limits
            assert "max_positions" in limits
            assert "brokers" in limits
            assert "features" in limits

    def test_free_tier_is_paper_only(self):
        from user_management import TIER_LIMITS
        assert TIER_LIMITS["free"]["paper_only"] is True
        assert TIER_LIMITS["pro"]["paper_only"] is False

    def test_enterprise_unlimited(self):
        from user_management import TIER_LIMITS
        assert TIER_LIMITS["enterprise"]["monthly_profit_cap"] is None


class TestAuthIntegration:
    """Test auth.py DB-backed user upsert integration."""

    def test_upsert_called_on_token_creation(self):
        """Verify _upsert_user_on_login is called during token creation."""
        os.environ["AUTH_JWT_SECRET"] = "test_secret_for_jwt_testing_123456"
        try:
            from auth import _create_token
            with patch("auth._upsert_user_on_login") as mock_upsert:
                mock_upsert.return_value = {
                    "user_id": str(uuid.uuid4()),
                    "tier": "pro",
                    "is_admin": True,
                }
                token = _create_token({
                    "email": "test@test.com",
                    "name": "Test",
                    "picture": "",
                    "provider": "google",
                })
                mock_upsert.assert_called_once()
                assert token  # JWT string was generated
        finally:
            os.environ.pop("AUTH_JWT_SECRET", None)

    def test_token_has_user_id_when_db_available(self):
        """JWT should contain user_id, tier, is_admin from DB."""
        import jwt as pyjwt
        os.environ["AUTH_JWT_SECRET"] = "test_secret_for_jwt_testing_123456"
        try:
            from auth import _create_token
            test_uid = str(uuid.uuid4())
            with patch("auth._upsert_user_on_login") as mock_upsert:
                mock_upsert.return_value = {
                    "user_id": test_uid,
                    "tier": "starter",
                    "is_admin": False,
                }
                token = _create_token({
                    "email": "test@test.com",
                    "name": "Test",
                    "picture": "",
                    "provider": "google",
                })
            payload = pyjwt.decode(token, "test_secret_for_jwt_testing_123456", algorithms=["HS256"])
            assert payload["user_id"] == test_uid
            assert payload["tier"] == "starter"
            assert payload["is_admin"] is False
        finally:
            os.environ.pop("AUTH_JWT_SECRET", None)

    def test_token_works_without_db(self):
        """JWT creation should succeed even when DB is unavailable."""
        import jwt as pyjwt
        os.environ["AUTH_JWT_SECRET"] = "test_secret_for_jwt_testing_123456"
        try:
            from auth import _create_token
            with patch("auth._upsert_user_on_login") as mock_upsert:
                mock_upsert.return_value = None  # DB failure
                token = _create_token({
                    "email": "test@test.com",
                    "name": "Test",
                    "picture": "",
                    "provider": "google",
                })
            payload = pyjwt.decode(token, "test_secret_for_jwt_testing_123456", algorithms=["HS256"])
            assert payload["sub"] == "test@test.com"
            assert "user_id" not in payload  # no DB data
        finally:
            os.environ.pop("AUTH_JWT_SECRET", None)
