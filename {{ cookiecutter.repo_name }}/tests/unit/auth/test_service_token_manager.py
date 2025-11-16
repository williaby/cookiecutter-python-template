"""Unit tests for ServiceTokenManager.

Tests cover:
- Token creation and validation
- Token rotation and revocation
- Emergency revocation scenarios
- Usage analytics and reporting
- Token cleanup operations
- Error handling and edge cases
"""

# ruff: noqa: S105, S106

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.auth.service_token_manager import ServiceTokenManager


class TestServiceTokenManager:
    """Test cases for ServiceTokenManager."""

    @pytest.fixture
    def token_manager(self):
        """Create ServiceTokenManager instance for testing."""
        return ServiceTokenManager()

    @pytest.fixture
    def sample_token_data(self):
        """Sample token data for testing."""
        return {
            "token_name": "test-token",
            "metadata": {"permissions": ["api_read", "system_status"], "environment": "test"},
            "expires_at": datetime.now(UTC) + timedelta(days=30),
            "is_active": True,
        }

    def test_generate_token(self, token_manager):
        """Test token generation produces valid tokens."""
        token = token_manager.generate_token()

        # Check token format
        assert token.startswith("sk_")
        assert len(token) == 67  # 3 chars (sk_) + 64 hex chars

        # Check token is hex after prefix
        token_hex = token[3:]
        int(token_hex, 16)  # Should not raise ValueError

        # Generate another token to ensure uniqueness
        token2 = token_manager.generate_token()
        assert token != token2

    def test_hash_token(self, token_manager):
        """Test token hashing produces consistent results."""
        token = "sk_test123"
        hash1 = token_manager.hash_token(token)
        hash2 = token_manager.hash_token(token)

        # Same token should produce same hash
        assert hash1 == hash2

        # Hash should be SHA-256 format
        assert len(hash1) == 64
        int(hash1, 16)  # Should not raise ValueError

        # Different token should produce different hash
        different_token = "sk_test456"
        hash3 = token_manager.hash_token(different_token)
        assert hash1 != hash3

    @pytest.mark.asyncio
    async def test_create_service_token_success(self, token_manager, sample_token_data):
        """Test successful service token creation."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar.return_value = 0  # Token name doesn't exist
        mock_session.execute.return_value = mock_result

        # Mock the refresh operation to set the token ID
        async def mock_refresh(token):
            token.id = "12345"  # Simulate database setting the ID

        mock_session.refresh = AsyncMock(side_effect=mock_refresh)

        with patch("src.auth.service_token_manager.get_db") as mock_get_db:
            mock_get_db.return_value.__aenter__.return_value = mock_session
            mock_get_db.return_value.__aexit__.return_value = None

            token_value, token_id = await token_manager.create_service_token(**sample_token_data)

            # Verify token format
            assert token_value.startswith("sk_")
            assert len(token_value) == 67
            assert token_id == "12345"

            # Verify database operations
            mock_session.execute.assert_called_once()
            mock_session.add.assert_called_once()
            mock_session.commit.assert_called_once()
            mock_session.refresh.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_service_token_duplicate_name(self, token_manager, sample_token_data):
        """Test creating token with duplicate name raises ValueError."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar.return_value = 1  # Token name already exists
        mock_session.execute.return_value = mock_result

        with patch("src.auth.service_token_manager.get_db") as mock_get_db:
            mock_get_db.return_value.__aenter__.return_value = mock_session
            mock_get_db.return_value.__aexit__.return_value = None

            with pytest.raises(ValueError, match="already exists"):
                await token_manager.create_service_token(**sample_token_data)

    @pytest.mark.asyncio
    async def test_revoke_service_token_success(self, token_manager):
        """Test successful token revocation."""
        mock_session = AsyncMock()

        # Mock finding the token
        mock_result = MagicMock()
        mock_token_record = MagicMock()
        mock_token_record.id = "12345"
        mock_token_record.token_name = "test-token"
        mock_token_record.is_active = True
        mock_result.fetchone.return_value = mock_token_record
        mock_session.execute.return_value = mock_result

        with patch("src.auth.service_token_manager.get_db") as mock_get_db:
            mock_get_db.return_value.__aenter__.return_value = mock_session
            mock_get_db.return_value.__aexit__.return_value = None

            success = await token_manager.revoke_service_token("test-token", "test_revocation")

            assert success is True

            # Verify database operations
            assert mock_session.execute.call_count == 2  # Find token + update token
            mock_session.add.assert_called_once()  # Add audit event
            mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_revoke_service_token_not_found(self, token_manager):
        """Test revoking non-existent token returns False."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.fetchone.return_value = None  # Token not found
        mock_session.execute.return_value = mock_result

        with patch("src.auth.service_token_manager.get_db") as mock_get_db:
            mock_get_db.return_value.__aenter__.return_value = mock_session
            mock_get_db.return_value.__aexit__.return_value = None

            success = await token_manager.revoke_service_token("nonexistent-token")

            assert success is False

    @pytest.mark.asyncio
    async def test_emergency_revoke_all_tokens(self, token_manager):
        """Test emergency revocation of all tokens."""
        mock_session = AsyncMock()

        # Mock finding active tokens
        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 5  # 5 active tokens

        # Mock responses in order
        mock_session.execute.side_effect = [
            mock_count_result,  # Count query
            MagicMock(),  # Update query
        ]

        with patch("src.auth.service_token_manager.get_db") as mock_get_db:
            mock_get_db.return_value.__aenter__.return_value = mock_session
            mock_get_db.return_value.__aexit__.return_value = None

            revoked_count = await token_manager.emergency_revoke_all_tokens("security_incident")

            assert revoked_count == 5

            # Verify database operations
            assert mock_session.execute.call_count == 2  # Count + update
            mock_session.add.assert_called_once()  # Add audit event
            mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_emergency_revoke_no_active_tokens(self, token_manager):
        """Test emergency revocation with no active tokens."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar.return_value = 0  # No active tokens
        mock_session.execute.return_value = mock_result

        with patch("src.auth.service_token_manager.get_db") as mock_get_db:
            mock_get_db.return_value.__aenter__.return_value = mock_session
            mock_get_db.return_value.__aexit__.return_value = None

            revoked_count = await token_manager.emergency_revoke_all_tokens("test_reason")

            assert revoked_count == 0

    @pytest.mark.asyncio
    async def test_rotate_service_token_success(self, token_manager):
        """Test successful token rotation."""
        mock_session = AsyncMock()

        # Mock finding the old token
        mock_find_result = MagicMock()
        mock_old_token = MagicMock()
        mock_old_token.id = "old-id"
        mock_old_token.token_name = "test-token"
        mock_old_token.token_metadata = {"permissions": ["api_read"]}
        mock_old_token.expires_at = None
        mock_find_result.fetchone.return_value = mock_old_token

        # Mock the refresh operation to set the new token ID
        async def mock_refresh(token):
            token.id = "new-id"

        mock_session.execute.return_value = mock_find_result
        mock_session.refresh = AsyncMock(side_effect=mock_refresh)

        with patch("src.auth.service_token_manager.get_db") as mock_get_db:
            mock_get_db.return_value.__aenter__.return_value = mock_session
            mock_get_db.return_value.__aexit__.return_value = None

            result = await token_manager.rotate_service_token("test-token", "scheduled_rotation")

            assert result is not None
            new_token_value, new_token_id = result

            # Verify new token format
            assert new_token_value.startswith("sk_")
            assert len(new_token_value) == 67
            assert new_token_id == "new-id"

            # Verify database operations
            assert mock_session.execute.call_count == 2  # Find + deactivate old token
            mock_session.add.assert_called()  # Add new token and audit event
            mock_session.commit.assert_called_once()
            mock_session.refresh.assert_called_once()

    @pytest.mark.asyncio
    async def test_rotate_service_token_not_found(self, token_manager):
        """Test rotating non-existent token returns None."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.fetchone.return_value = None  # Token not found
        mock_session.execute.return_value = mock_result

        with patch("src.auth.service_token_manager.get_db") as mock_get_db:
            mock_get_db.return_value.__aenter__.return_value = mock_session
            mock_get_db.return_value.__aexit__.return_value = None

            result = await token_manager.rotate_service_token("nonexistent-token")

            assert result is None

    @pytest.mark.asyncio
    async def test_get_token_usage_analytics_single_token(self, token_manager):
        """Test getting analytics for a single token."""
        mock_session = AsyncMock()

        # Mock token data
        mock_token_result = MagicMock()
        mock_token_data = MagicMock()
        mock_token_data.token_name = "test-token"
        mock_token_data.usage_count = 100
        mock_token_data.last_used = datetime.now(UTC)
        mock_token_data.created_at = datetime.now(UTC) - timedelta(days=30)
        mock_token_data.is_active = True
        mock_token_data.is_expired = False
        mock_token_result.fetchone.return_value = mock_token_data

        # Mock events data
        mock_events_result = MagicMock()
        mock_event = MagicMock()
        mock_event.event_type = "service_token_auth"
        mock_event.success = True
        mock_event.created_at = datetime.now(UTC)
        mock_events_result.fetchall.return_value = [mock_event]

        # Configure mock session to return different results for different queries
        mock_session.execute.side_effect = [mock_token_result, mock_events_result]

        with patch("src.auth.service_token_manager.get_db") as mock_get_db:
            mock_get_db.return_value.__aenter__.return_value = mock_session
            mock_get_db.return_value.__aexit__.return_value = None

            analytics = await token_manager.get_token_usage_analytics("test-token", days=30)

            assert analytics["token_name"] == "test-token"
            assert analytics["usage_count"] == 100
            assert analytics["is_active"] is True
            assert analytics["is_expired"] is False
            assert len(analytics["recent_events"]) == 1
            assert analytics["events_count"] == 1

    @pytest.mark.asyncio
    async def test_get_token_usage_analytics_all_tokens(self, token_manager):
        """Test getting analytics for all tokens."""
        mock_session = AsyncMock()

        # Mock summary data
        mock_summary_result = MagicMock()
        mock_summary = MagicMock()
        mock_summary.total_tokens = 10
        mock_summary.active_tokens = 8
        mock_summary.inactive_tokens = 2
        mock_summary.expired_tokens = 1
        mock_summary.total_usage = 500
        mock_summary.avg_usage_per_token = 50.0
        mock_summary_result.fetchone.return_value = mock_summary

        # Mock top tokens data
        mock_top_tokens_result = MagicMock()
        mock_top_token = MagicMock()
        mock_top_token.token_name = "popular-token"
        mock_top_token.usage_count = 200
        mock_top_token.last_used = datetime.now(UTC)
        mock_top_tokens_result.fetchall.return_value = [mock_top_token]

        # Configure mock session
        mock_session.execute.side_effect = [mock_summary_result, mock_top_tokens_result]

        with patch("src.auth.service_token_manager.get_db") as mock_get_db:
            mock_get_db.return_value.__aenter__.return_value = mock_session
            mock_get_db.return_value.__aexit__.return_value = None

            analytics = await token_manager.get_token_usage_analytics(days=30)

            assert "summary" in analytics
            assert analytics["summary"]["total_tokens"] == 10
            assert analytics["summary"]["active_tokens"] == 8
            assert analytics["summary"]["total_usage"] == 500

            assert "top_tokens" in analytics
            assert len(analytics["top_tokens"]) == 1
            assert analytics["top_tokens"][0]["token_name"] == "popular-token"

    @pytest.mark.asyncio
    async def test_cleanup_expired_tokens_deactivate(self, token_manager):
        """Test cleaning up expired tokens by deactivation."""
        mock_session = AsyncMock()

        # Mock expired tokens
        mock_expired_result = MagicMock()
        mock_expired_token = MagicMock()
        mock_expired_token.id = "expired-id"
        mock_expired_token.token_name = "expired-token"
        mock_expired_result.fetchall.return_value = [mock_expired_token]

        mock_session.execute.side_effect = [
            mock_expired_result,  # Find expired tokens
            MagicMock(),  # Update tokens
        ]

        with patch("src.auth.service_token_manager.get_db") as mock_get_db:
            mock_get_db.return_value.__aenter__.return_value = mock_session
            mock_get_db.return_value.__aexit__.return_value = None

            result = await token_manager.cleanup_expired_tokens(deactivate_only=True)

            assert result["expired_tokens_processed"] == 1
            assert result["action"] == "deactivated"
            assert "expired-token" in result["token_names"]

    @pytest.mark.asyncio
    async def test_cleanup_expired_tokens_delete(self, token_manager):
        """Test cleaning up expired tokens by deletion."""
        mock_session = AsyncMock()

        # Mock expired tokens
        mock_expired_result = MagicMock()
        mock_expired_token = MagicMock()
        mock_expired_token.id = "expired-id"
        mock_expired_token.token_name = "expired-token"
        mock_expired_result.fetchall.return_value = [mock_expired_token]

        mock_session.execute.side_effect = [
            mock_expired_result,  # Find expired tokens
            MagicMock(),  # Delete tokens
        ]

        with patch("src.auth.service_token_manager.get_db") as mock_get_db:
            mock_get_db.return_value.__aenter__.return_value = mock_session
            mock_get_db.return_value.__aexit__.return_value = None

            result = await token_manager.cleanup_expired_tokens(deactivate_only=False)

            assert result["expired_tokens_processed"] == 1
            assert result["action"] == "deleted"
            assert "expired-token" in result["token_names"]

    @pytest.mark.asyncio
    async def test_cleanup_no_expired_tokens(self, token_manager):
        """Test cleanup when no expired tokens exist."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []  # No expired tokens
        mock_session.execute.return_value = mock_result

        with patch("src.auth.service_token_manager.get_db") as mock_get_db:
            mock_get_db.return_value.__aenter__.return_value = mock_session
            mock_get_db.return_value.__aexit__.return_value = None

            result = await token_manager.cleanup_expired_tokens()

            assert result["expired_tokens_processed"] == 0
            assert result["action"] == "none_needed"

    @pytest.mark.asyncio
    async def test_token_lifecycle_integration(self, token_manager):
        """Test complete token lifecycle: create -> use -> rotate -> revoke."""
        mock_session = AsyncMock()
        mock_get_db_patcher = patch("src.auth.service_token_manager.get_db")

        with mock_get_db_patcher as mock_get_db:
            mock_get_db.return_value.__aenter__.return_value = mock_session
            mock_get_db.return_value.__aexit__.return_value = None

            # 1. Create token
            mock_result = MagicMock()
            mock_result.scalar.return_value = 0  # Token name doesn't exist
            mock_session.execute.return_value = mock_result

            # Mock refresh for token creation
            async def mock_refresh_create(token):
                if not hasattr(token, "id") or token.id is None:
                    token.id = "token-id-123"

            mock_session.refresh = AsyncMock(side_effect=mock_refresh_create)

            token_value, token_id = await token_manager.create_service_token(
                token_name="lifecycle-test",
                metadata={"permissions": ["api_read"]},
                is_active=True,
            )

            assert token_value.startswith("sk_")
            assert token_id == "token-id-123"

            # 2. Rotate token
            mock_find_result = MagicMock()
            mock_old_token = MagicMock()
            mock_old_token.id = "token-id-123"
            mock_old_token.token_name = "lifecycle-test"
            mock_old_token.token_metadata = {"permissions": ["api_read"]}
            mock_old_token.expires_at = None
            mock_find_result.fetchone.return_value = mock_old_token

            # Update refresh mock for token rotation
            async def mock_refresh_rotate(token):
                if hasattr(token, "token_name") and "rotated" in token.token_name:
                    token.id = "new-token-id-456"
                elif not hasattr(token, "id") or token.id is None:
                    token.id = "token-id-123"

            mock_session.refresh = AsyncMock(side_effect=mock_refresh_rotate)
            mock_session.execute.return_value = mock_find_result

            rotation_result = await token_manager.rotate_service_token("lifecycle-test")

            assert rotation_result is not None
            new_token_value, new_token_id = rotation_result
            assert new_token_value.startswith("sk_")
            assert new_token_id == "new-token-id-456"

            # 3. Revoke new token
            mock_revoke_result = MagicMock()
            mock_revoke_token = MagicMock()
            mock_revoke_token.id = "new-token-id-456"
            mock_revoke_token.token_name = "lifecycle-test_rotated_20240101_120000"
            mock_revoke_result.fetchone.return_value = mock_revoke_token
            mock_session.execute.return_value = mock_revoke_result

            revoke_success = await token_manager.revoke_service_token("new-token-id-456")

            assert revoke_success is True

    def test_token_security_properties(self, token_manager):
        """Test security properties of generated tokens."""
        # Generate multiple tokens
        tokens = [token_manager.generate_token() for _ in range(100)]

        # All tokens should be unique
        assert len(set(tokens)) == 100

        # All tokens should have correct format
        for token in tokens:
            assert token.startswith("sk_")
            assert len(token) == 67

        # Token hashes should be unique
        hashes = [token_manager.hash_token(token) for token in tokens]
        assert len(set(hashes)) == 100

        # Hashes should be SHA-256 format
        for hash_value in hashes:
            assert len(hash_value) == 64
            int(hash_value, 16)  # Should not raise ValueError

    @pytest.mark.asyncio
    async def test_error_handling_database_failure(self, token_manager):
        """Test error handling when database operations fail."""
        with patch("src.auth.service_token_manager.get_db") as mock_get_db:
            # Simulate database connection failure
            mock_get_db.side_effect = Exception("Database connection failed")

            with pytest.raises(Exception, match="Database connection failed"):
                await token_manager.create_service_token("test-token")

            with pytest.raises(Exception, match="Database connection failed"):
                await token_manager.revoke_service_token("test-token")

            with pytest.raises(Exception, match="Database connection failed"):
                await token_manager.rotate_service_token("test-token")

    @pytest.mark.asyncio
    async def test_get_token_analytics_not_found(self, token_manager):
        """Test analytics for non-existent token."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.fetchone.return_value = None  # Token not found
        mock_session.execute.return_value = mock_result

        with patch("src.auth.service_token_manager.get_db") as mock_get_db:
            mock_get_db.return_value.__aenter__.return_value = mock_session
            mock_get_db.return_value.__aexit__.return_value = None

            analytics = await token_manager.get_token_usage_analytics("nonexistent-token")

            assert "error" in analytics
            assert analytics["error"] == "Token not found"
