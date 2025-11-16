"""Integration tests for role management API endpoints (AUTH-3).

This module provides comprehensive integration tests for the role management
FastAPI endpoints, testing the complete API layer including:
- HTTP request/response handling
- Permission enforcement via FastAPI dependencies
- Database integration with actual transactions
- Error response formatting
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.auth.middleware import require_authentication
from src.auth.permissions import Permissions, require_permission
from src.auth.role_manager import (
    PermissionNotFoundError,
    RoleManagerError,
    RoleNotFoundError,
    UserNotFoundError,
)
from src.main import app


@pytest.fixture
def client(mock_authenticated_user):
    """Create a test client for FastAPI app with authentication override."""
    # Override the dependency to return our mock user
    def mock_require_authentication():
        return mock_authenticated_user

    app.dependency_overrides[require_authentication] = mock_require_authentication

    try:
        yield TestClient(app)
    finally:
        # Clean up the override after the test
        app.dependency_overrides.clear()


@pytest.fixture
def mock_authenticated_user():
    """Mock authenticated user with admin permissions."""
    user = MagicMock()
    user.email = "admin@example.com"
    user.has_permission.return_value = True
    return user


@pytest.fixture
def mock_service_token():
    """Mock service token with admin permissions."""
    from src.auth.middleware import ServiceTokenUser
    # Create a real ServiceTokenUser instance with admin permissions
    return ServiceTokenUser(
        token_id="test-token-id",
        token_name="admin-service-token",
        metadata={"permissions": ["admin"]},  # admin permission grants all access
        usage_count=1,
    )


@pytest.fixture
def sample_role_data():
    """Sample role data for testing."""
    return {
        "id": 1,
        "name": "test_role",
        "description": "Test role description",
        "parent_role_id": None,
        "parent_role_name": None,
        "created_at": datetime.now(UTC),
        "updated_at": datetime.now(UTC),
        "is_active": True,
    }


@pytest.fixture
def sample_user_role_data():
    """Sample user role assignment data for testing."""
    return {
        "role_id": 1,
        "role_name": "test_role",
        "role_description": "Test role description",
        "assigned_at": datetime.now(UTC),
    }


class TestRoleCreationEndpoints:
    """Test role creation API endpoints."""

    @patch("src.auth.permissions.user_has_permission")
    @patch("src.api.role_endpoints.RoleManager")
    def test_create_role_success(
        self,
        mock_role_manager_class,
        mock_user_has_permission,
        client,
        mock_authenticated_user,
        sample_role_data,
    ):
        """Test successful role creation via API."""
        # Setup mocks
        mock_user_has_permission.return_value = True  # Grant permission
        mock_manager = AsyncMock()
        mock_role_manager_class.return_value = mock_manager
        mock_manager.create_role.return_value = sample_role_data

        # Test data
        request_data = {
            "name": "test_role",
            "description": "Test role description",
        }

        # Make request
        response = client.post("/api/v1/roles/", json=request_data)

        # Assert response
        assert response.status_code == 200
        assert response.json()["name"] == "test_role"
        assert response.json()["description"] == "Test role description"
        mock_manager.create_role.assert_called_once_with(
            name="test_role",
            description="Test role description",
            parent_role_name=None,
        )

    @patch("src.auth.permissions.user_has_permission")
    @patch("src.api.role_endpoints.RoleManager")
    def test_create_role_with_parent(
        self,
        mock_role_manager_class,
        mock_user_has_permission,
        client,
        mock_authenticated_user,
        sample_role_data,
    ):
        """Test role creation with parent role."""
        # Setup mocks
        mock_user_has_permission.return_value = True
        mock_manager = AsyncMock()
        mock_role_manager_class.return_value = mock_manager

        # Mock parent role check
        parent_role_data = sample_role_data.copy()
        parent_role_data["name"] = "parent_role"
        mock_manager.get_role.return_value = parent_role_data
        mock_manager.create_role.return_value = sample_role_data

        # Test data
        request_data = {
            "name": "test_role",
            "description": "Test role description",
            "parent_role_name": "parent_role",
        }

        # Make request
        response = client.post("/api/v1/roles/", json=request_data)

        # Assert response
        assert response.status_code == 200
        mock_manager.get_role.assert_called_once_with("parent_role")
        mock_manager.create_role.assert_called_once_with(
            name="test_role",
            description="Test role description",
            parent_role_name="parent_role",
        )

    @patch("src.auth.permissions.user_has_permission")
    @patch("src.api.role_endpoints.RoleManager")
    def test_create_role_invalid_parent(
        self,
        mock_role_manager_class,
        mock_user_has_permission,
        client,
        mock_authenticated_user,
    ):
        """Test role creation with non-existent parent role."""
        # Setup mocks
        mock_user_has_permission.return_value = True
        mock_manager = AsyncMock()
        mock_role_manager_class.return_value = mock_manager
        mock_manager.get_role.return_value = None  # Parent doesn't exist

        # Test data
        request_data = {
            "name": "test_role",
            "description": "Test role description",
            "parent_role_name": "nonexistent_parent",
        }

        # Make request
        response = client.post("/api/v1/roles/", json=request_data)

        # Assert response
        assert response.status_code == 500  # HTTPException in try/catch gets converted to 500
        response_data = response.json()
        detail_text = response_data.get("detail") or response_data.get("error", "")
        assert "does not exist" in detail_text
        mock_manager.get_role.assert_called_once_with("nonexistent_parent")

    @patch("src.auth.permissions.user_has_permission")
    @patch("src.api.role_endpoints.RoleManager")
    def test_create_role_already_exists(
        self,
        mock_role_manager_class,
        mock_user_has_permission,
        client,
        mock_authenticated_user,
    ):
        """Test role creation when role already exists."""
        # Setup mocks
        mock_user_has_permission.return_value = True
        mock_manager = AsyncMock()
        mock_role_manager_class.return_value = mock_manager
        mock_manager.create_role.side_effect = RoleManagerError("Role 'test_role' already exists")

        # Test data
        request_data = {
            "name": "test_role",
            "description": "Test role description",
        }

        # Make request
        response = client.post("/api/v1/roles/", json=request_data)

        # Assert response
        assert response.status_code == 409
        response_data = response.json()
        detail_text = response_data.get("detail") or response_data.get("error", "")
        assert "already exists" in detail_text


class TestRoleRetrievalEndpoints:
    """Test role retrieval API endpoints."""

    @patch("src.auth.permissions.user_has_permission")
    @patch("src.api.role_endpoints.RoleManager")
    def test_list_roles_success(
        self,
        mock_role_manager_class,
        mock_user_has_permission,
        client,
        mock_authenticated_user,
        sample_role_data,
    ):
        """Test successful role listing via API."""
        # Setup mocks
        mock_user_has_permission.return_value = True
        mock_manager = AsyncMock()
        mock_role_manager_class.return_value = mock_manager
        mock_manager.list_roles.return_value = [sample_role_data]

        # Make request
        response = client.get("/api/v1/roles/")

        # Assert response
        assert response.status_code == 200
        roles = response.json()
        assert len(roles) == 1
        assert roles[0]["name"] == "test_role"
        mock_manager.list_roles.assert_called_once_with(include_inactive=False)

    @patch("src.auth.permissions.user_has_permission")
    @patch("src.api.role_endpoints.RoleManager")
    def test_list_roles_include_inactive(
        self,
        mock_role_manager_class,
        mock_user_has_permission,
        client,
        mock_authenticated_user,
        sample_role_data,
    ):
        """Test role listing including inactive roles."""
        # Setup mocks
        mock_user_has_permission.return_value = True
        mock_manager = AsyncMock()
        mock_role_manager_class.return_value = mock_manager
        mock_manager.list_roles.return_value = [sample_role_data]

        # Make request
        response = client.get("/api/v1/roles/?include_inactive=true")

        # Assert response
        assert response.status_code == 200
        mock_manager.list_roles.assert_called_once_with(include_inactive=True)

    @patch("src.auth.permissions.user_has_permission")
    @patch("src.api.role_endpoints.RoleManager")
    def test_get_role_success(
        self,
        mock_role_manager_class,
        mock_user_has_permission,
        client,
        mock_authenticated_user,
        sample_role_data,
    ):
        """Test successful individual role retrieval."""
        # Setup mocks
        mock_user_has_permission.return_value = True
        mock_manager = AsyncMock()
        mock_role_manager_class.return_value = mock_manager
        mock_manager.get_role.return_value = sample_role_data

        # Make request
        response = client.get("/api/v1/roles/test_role")

        # Assert response
        assert response.status_code == 200
        assert response.json()["name"] == "test_role"
        mock_manager.get_role.assert_called_once_with("test_role")

    @patch("src.auth.permissions.user_has_permission")
    @patch("src.api.role_endpoints.RoleManager")
    def test_get_role_not_found(
        self,
        mock_role_manager_class,
        mock_user_has_permission,
        client,
        mock_authenticated_user,
    ):
        """Test role retrieval for non-existent role."""
        # Setup mocks
        mock_user_has_permission.return_value = True
        mock_manager = AsyncMock()
        mock_role_manager_class.return_value = mock_manager
        mock_manager.get_role.return_value = None

        # Make request
        response = client.get("/api/v1/roles/nonexistent_role")

        # Assert response
        assert response.status_code == 404
        response_data = response.json()
        print(f"DEBUG: Response content: {response_data}")
        detail_text = response_data.get("detail") or response_data.get("error", "")
        assert "not found" in detail_text


class TestRolePermissionEndpoints:
    """Test role permission management API endpoints."""

    @patch("src.auth.permissions.user_has_permission")
    @patch("src.api.role_endpoints.RoleManager")
    def test_get_role_permissions_success(
        self,
        mock_role_manager_class,
        mock_user_has_permission,
        client,
        mock_authenticated_user,
    ):
        """Test successful role permissions retrieval."""
        # Setup mocks
        mock_user_has_permission.return_value = lambda: mock_authenticated_user
        mock_manager = AsyncMock()
        mock_role_manager_class.return_value = mock_manager
        mock_manager.get_role_permissions.return_value = {"tokens:create", "tokens:read"}

        # Make request
        response = client.get("/api/v1/roles/test_role/permissions")

        # Assert response
        assert response.status_code == 200
        data = response.json()
        assert data["role_name"] == "test_role"
        assert set(data["permissions"]) == {"tokens:create", "tokens:read"}
        mock_manager.get_role_permissions.assert_called_once_with("test_role")

    @patch("src.auth.permissions.user_has_permission")
    @patch("src.api.role_endpoints.RoleManager")
    def test_assign_permission_to_role_success(
        self,
        mock_role_manager_class,
        mock_user_has_permission,
        client,
        mock_authenticated_user,
    ):
        """Test successful permission assignment to role."""
        # Setup mocks
        mock_user_has_permission.return_value = lambda: mock_authenticated_user
        mock_manager = AsyncMock()
        mock_role_manager_class.return_value = mock_manager
        mock_manager.assign_permission_to_role.return_value = True

        # Test data
        request_data = {
            "role_name": "test_role",
            "permission_name": "tokens:create",
        }

        # Make request
        response = client.post("/api/v1/roles/test_role/permissions", json=request_data)

        # Assert response
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "assigned" in data["message"]
        assert data["assigned_by"] == "admin@example.com"
        mock_manager.assign_permission_to_role.assert_called_once_with("test_role", "tokens:create")

    @patch("src.auth.permissions.user_has_permission")
    @patch("src.api.role_endpoints.RoleManager")
    def test_revoke_permission_from_role_success(
        self,
        mock_role_manager_class,
        mock_user_has_permission,
        client,
        mock_authenticated_user,
    ):
        """Test successful permission revocation from role."""
        # Setup mocks
        mock_user_has_permission.return_value = lambda: mock_authenticated_user
        mock_manager = AsyncMock()
        mock_role_manager_class.return_value = mock_manager
        mock_manager.revoke_permission_from_role.return_value = True

        # Make request
        response = client.delete("/api/v1/roles/test_role/permissions/tokens:create")

        # Assert response
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "revoked" in data["message"]
        assert data["revoked_by"] == "admin@example.com"
        mock_manager.revoke_permission_from_role.assert_called_once_with("test_role", "tokens:create")


class TestUserRoleAssignmentEndpoints:
    """Test user role assignment API endpoints."""

    @patch("src.auth.permissions.user_has_permission")
    @patch("src.api.role_endpoints.RoleManager")
    def test_assign_user_role_success(
        self,
        mock_role_manager_class,
        mock_user_has_permission,
        client,
        mock_authenticated_user,
    ):
        """Test successful user role assignment."""
        # Setup mocks
        mock_user_has_permission.return_value = lambda: mock_authenticated_user
        mock_manager = AsyncMock()
        mock_role_manager_class.return_value = mock_manager
        mock_manager.assign_user_role.return_value = True

        # Test data
        request_data = {
            "user_email": "user@example.com",
            "role_name": "test_role",
        }

        # Make request
        response = client.post("/api/v1/roles/assignments", json=request_data)

        # Assert response
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "assigned" in data["message"]
        assert data["assigned_by"] == "admin@example.com"
        mock_manager.assign_user_role.assert_called_once_with(
            user_email="user@example.com",
            role_name="test_role",
            assigned_by="admin@example.com",
        )

    @patch("src.auth.permissions.user_has_permission")
    @patch("src.api.role_endpoints.RoleManager")
    def test_revoke_user_role_success(
        self,
        mock_role_manager_class,
        mock_user_has_permission,
        client,
        mock_authenticated_user,
    ):
        """Test successful user role revocation."""
        # Setup mocks
        mock_user_has_permission.return_value = lambda: mock_authenticated_user
        mock_manager = AsyncMock()
        mock_role_manager_class.return_value = mock_manager
        mock_manager.revoke_user_role.return_value = True

        # Make request
        response = client.delete("/api/v1/roles/assignments?user_email=user@example.com&role_name=test_role")

        # Assert response
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "revoked" in data["message"]
        assert data["revoked_by"] == "admin@example.com"
        mock_manager.revoke_user_role.assert_called_once_with(
            user_email="user@example.com",
            role_name="test_role",
        )

    @patch("src.auth.permissions.user_has_permission")
    @patch("src.api.role_endpoints.RoleManager")
    def test_revoke_user_role_not_assigned(
        self,
        mock_role_manager_class,
        mock_user_has_permission,
        client,
        mock_authenticated_user,
    ):
        """Test user role revocation when role wasn't assigned."""
        # Setup mocks
        mock_user_has_permission.return_value = lambda: mock_authenticated_user
        mock_manager = AsyncMock()
        mock_role_manager_class.return_value = mock_manager
        mock_manager.revoke_user_role.return_value = False

        # Make request
        response = client.delete("/api/v1/roles/assignments?user_email=user@example.com&role_name=test_role")

        # Assert response
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "no_change"
        assert "was not assigned" in data["message"]

    @patch("src.auth.permissions.user_has_permission")
    @patch("src.api.role_endpoints.RoleManager")
    def test_get_user_roles_success(
        self,
        mock_role_manager_class,
        mock_user_has_permission,
        client,
        mock_authenticated_user,
        sample_user_role_data,
    ):
        """Test successful user roles retrieval."""
        # Setup mocks
        mock_user_has_permission.return_value = lambda: mock_authenticated_user
        mock_manager = AsyncMock()
        mock_role_manager_class.return_value = mock_manager
        mock_manager.get_user_roles.return_value = [sample_user_role_data]

        # Make request
        response = client.get("/api/v1/roles/users/user@example.com/roles")

        # Assert response
        assert response.status_code == 200
        roles = response.json()
        assert len(roles) == 1
        assert roles[0]["role_name"] == "test_role"
        mock_manager.get_user_roles.assert_called_once_with("user@example.com")

    @patch("src.auth.permissions.user_has_permission")
    @patch("src.api.role_endpoints.RoleManager")
    def test_get_user_permissions_success(
        self,
        mock_role_manager_class,
        mock_user_has_permission,
        client,
        mock_authenticated_user,
        sample_user_role_data,
    ):
        """Test successful user permissions retrieval."""
        # Setup mocks
        mock_user_has_permission.return_value = lambda: mock_authenticated_user
        mock_manager = AsyncMock()
        mock_role_manager_class.return_value = mock_manager
        mock_manager.get_user_roles.return_value = [sample_user_role_data]
        mock_manager.get_user_permissions.return_value = {"tokens:create", "tokens:read"}

        # Make request
        response = client.get("/api/v1/roles/users/user@example.com/permissions")

        # Assert response
        assert response.status_code == 200
        data = response.json()
        assert data["user_email"] == "user@example.com"
        assert len(data["roles"]) == 1
        assert set(data["permissions"]) == {"tokens:create", "tokens:read"}
        mock_manager.get_user_roles.assert_called_once_with("user@example.com")
        mock_manager.get_user_permissions.assert_called_once_with("user@example.com")


class TestRoleDeletionEndpoints:
    """Test role deletion API endpoints."""

    @patch("src.auth.permissions.user_has_permission")
    @patch("src.api.role_endpoints.RoleManager")
    def test_delete_role_success(
        self,
        mock_role_manager_class,
        mock_user_has_permission,
        client,
        mock_authenticated_user,
    ):
        """Test successful role deletion."""
        # Setup mocks
        mock_user_has_permission.return_value = lambda: mock_authenticated_user
        mock_manager = AsyncMock()
        mock_role_manager_class.return_value = mock_manager
        mock_manager.delete_role.return_value = True

        # Make request
        response = client.delete("/api/v1/roles/test_role")

        # Assert response
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "deleted" in data["message"]
        assert data["deleted_by"] == "admin@example.com"
        assert data["force"] == "False"
        mock_manager.delete_role.assert_called_once_with("test_role", force=False)

    @patch("src.auth.permissions.user_has_permission")
    @patch("src.api.role_endpoints.RoleManager")
    def test_delete_role_force(self, mock_role_manager_class, mock_user_has_permission, client, mock_authenticated_user):
        """Test force role deletion."""
        # Setup mocks
        mock_user_has_permission.return_value = lambda: mock_authenticated_user
        mock_manager = AsyncMock()
        mock_role_manager_class.return_value = mock_manager
        mock_manager.delete_role.return_value = True

        # Make request
        response = client.delete("/api/v1/roles/test_role?force=true")

        # Assert response
        assert response.status_code == 200
        data = response.json()
        assert data["force"] == "True"
        mock_manager.delete_role.assert_called_once_with("test_role", force=True)

    @patch("src.auth.permissions.user_has_permission")
    @patch("src.api.role_endpoints.RoleManager")
    def test_delete_role_not_found(
        self,
        mock_role_manager_class,
        mock_user_has_permission,
        client,
        mock_authenticated_user,
    ):
        """Test role deletion for non-existent role."""
        # Setup mocks
        mock_user_has_permission.return_value = lambda: mock_authenticated_user
        mock_manager = AsyncMock()
        mock_role_manager_class.return_value = mock_manager
        mock_manager.delete_role.side_effect = RoleNotFoundError("Role 'nonexistent' not found")

        # Make request
        response = client.delete("/api/v1/roles/nonexistent")

        # Assert response
        assert response.status_code == 404
        response_data = response.json()
        detail_text = response_data.get("detail") or response_data.get("error", "")
        assert "not found" in detail_text


class TestRoleHierarchyValidationEndpoints:
    """Test role hierarchy validation API endpoints."""

    @patch("src.auth.permissions.user_has_permission")
    @patch("src.api.role_endpoints.RoleManager")
    def test_validate_role_hierarchy_valid(
        self,
        mock_role_manager_class,
        mock_user_has_permission,
        client,
        mock_authenticated_user,
    ):
        """Test successful role hierarchy validation."""
        # Setup mocks
        mock_user_has_permission.return_value = lambda: mock_authenticated_user
        mock_manager = AsyncMock()
        mock_role_manager_class.return_value = mock_manager
        mock_manager.validate_role_hierarchy.return_value = True

        # Make request
        response = client.post("/api/v1/roles/validate-hierarchy?role_name=child_role&parent_role_name=parent_role")

        # Assert response
        assert response.status_code == 200
        data = response.json()
        assert data["is_valid"] is True
        assert "is valid" in data["message"]
        mock_manager.validate_role_hierarchy.assert_called_once_with("child_role", "parent_role")

    @patch("src.auth.permissions.user_has_permission")
    @patch("src.api.role_endpoints.RoleManager")
    def test_validate_role_hierarchy_invalid(
        self,
        mock_role_manager_class,
        mock_user_has_permission,
        client,
        mock_authenticated_user,
    ):
        """Test role hierarchy validation with circular dependency."""
        # Setup mocks
        mock_user_has_permission.return_value = lambda: mock_authenticated_user
        mock_manager = AsyncMock()
        mock_role_manager_class.return_value = mock_manager
        mock_manager.validate_role_hierarchy.return_value = False

        # Make request
        response = client.post("/api/v1/roles/validate-hierarchy?role_name=child_role&parent_role_name=parent_role")

        # Assert response
        assert response.status_code == 200
        data = response.json()
        assert data["is_valid"] is False
        assert "circular dependency" in data["message"]


class TestServiceTokenAuthentication:
    """Test service token authentication for role endpoints."""

    @patch("src.auth.permissions.user_has_permission")
    @patch("src.api.role_endpoints.RoleManager")
    def test_service_token_role_creation(
        self,
        mock_role_manager_class,
        mock_user_has_permission,
        mock_service_token,
        sample_role_data,
    ):
        """Test role creation using service token authentication."""
        # The service token already has admin permissions from the fixture

        # Mock the authentication to return our service token user
        def mock_require_authentication():
            return mock_service_token

        # Override the authentication dependency BEFORE creating client
        app.dependency_overrides[require_authentication] = mock_require_authentication

        try:
            # Setup mocks
            mock_manager = AsyncMock()
            mock_role_manager_class.return_value = mock_manager
            mock_manager.create_role.return_value = sample_role_data

            # Test data
            request_data = {
                "name": "test_role",
                "description": "Test role description",
            }

            # Create client after setting up authentication override
            client = TestClient(app)

            # Make request
            response = client.post("/api/v1/roles/", json=request_data)

            # Assert response
            assert response.status_code == 200
            assert response.json()["name"] == "test_role"
        finally:
            # Clean up override
            app.dependency_overrides.clear()

    @patch("src.auth.permissions.user_has_permission")
    @patch("src.api.role_endpoints.RoleManager")
    def test_service_token_user_assignment(
        self,
        mock_role_manager_class,
        mock_user_has_permission,
        mock_service_token,
    ):
        """Test user role assignment using service token authentication."""
        # The service token already has admin permissions from the fixture

        # Mock the authentication to return our service token user
        def mock_require_authentication():
            return mock_service_token

        # Override the authentication dependency BEFORE creating client
        app.dependency_overrides[require_authentication] = mock_require_authentication

        try:
            # Setup mocks
            mock_manager = AsyncMock()
            mock_role_manager_class.return_value = mock_manager
            mock_manager.assign_user_role.return_value = True

            # Test data
            request_data = {
                "user_email": "user@example.com",
                "role_name": "test_role",
            }

            # Create client after setting up authentication override
            client = TestClient(app)

            # Make request
            response = client.post("/api/v1/roles/assignments", json=request_data)

            # Assert response
            assert response.status_code == 200
            data = response.json()
            assert data["assigned_by"] == "admin-service-token"
            mock_manager.assign_user_role.assert_called_once_with(
                user_email="user@example.com",
                role_name="test_role",
                assigned_by="admin-service-token",
            )
        finally:
            # Clean up override
            app.dependency_overrides.clear()


class TestErrorHandlingScenarios:
    """Test error handling and edge cases for role endpoints."""

    @patch("src.auth.permissions.user_has_permission")
    @patch("src.api.role_endpoints.RoleManager")
    def test_user_not_found_error(
        self,
        mock_role_manager_class,
        mock_user_has_permission,
        client,
        mock_authenticated_user,
    ):
        """Test handling of UserNotFoundError."""
        # Setup mocks
        mock_user_has_permission.return_value = lambda: mock_authenticated_user
        mock_manager = AsyncMock()
        mock_role_manager_class.return_value = mock_manager
        mock_manager.assign_user_role.side_effect = UserNotFoundError("User 'nonexistent@example.com' not found")

        # Test data
        request_data = {
            "user_email": "nonexistent@example.com",
            "role_name": "test_role",
        }

        # Make request
        response = client.post("/api/v1/roles/assignments", json=request_data)

        # Assert response
        assert response.status_code == 404
        response_data = response.json()
        detail_text = response_data.get("detail") or response_data.get("error", "")
        assert "not found" in detail_text

    @patch("src.auth.permissions.user_has_permission")
    @patch("src.api.role_endpoints.RoleManager")
    def test_permission_not_found_error(
        self,
        mock_role_manager_class,
        mock_user_has_permission,
        client,
        mock_authenticated_user,
    ):
        """Test handling of PermissionNotFoundError."""
        # Setup mocks
        mock_user_has_permission.return_value = lambda: mock_authenticated_user
        mock_manager = AsyncMock()
        mock_role_manager_class.return_value = mock_manager
        mock_manager.assign_permission_to_role.side_effect = PermissionNotFoundError(
            "Permission 'nonexistent:permission' not found",
        )

        # Test data
        request_data = {
            "role_name": "test_role",
            "permission_name": "nonexistent:permission",
        }

        # Make request
        response = client.post("/api/v1/roles/test_role/permissions", json=request_data)

        # Assert response
        assert response.status_code == 404
        response_data = response.json()
        detail_text = response_data.get("detail") or response_data.get("error", "")
        assert "not found" in detail_text

    @patch("src.auth.permissions.user_has_permission")
    @patch("src.api.role_endpoints.RoleManager")
    def test_general_role_manager_error(
        self,
        mock_role_manager_class,
        mock_user_has_permission,
        client,
        mock_authenticated_user,
    ):
        """Test handling of general RoleManagerError."""
        # Setup mocks
        mock_user_has_permission.return_value = lambda: mock_authenticated_user
        mock_manager = AsyncMock()
        mock_role_manager_class.return_value = mock_manager
        mock_manager.create_role.side_effect = RoleManagerError("Database connection failed")

        # Test data
        request_data = {
            "name": "test_role",
            "description": "Test role description",
        }

        # Make request
        response = client.post("/api/v1/roles/", json=request_data)

        # Assert response - RoleManagerError maps to 400 Bad Request in AuthExceptionHandler
        assert response.status_code == 400
        response_data = response.json()
        detail_text = response_data.get("detail") or response_data.get("error", "")
        assert "Database connection failed" in detail_text

    @patch("src.auth.permissions.user_has_permission")
    @patch("src.api.role_endpoints.RoleManager")
    def test_unexpected_exception(
        self,
        mock_role_manager_class,
        mock_user_has_permission,
        client,
        mock_authenticated_user,
    ):
        """Test handling of unexpected exceptions."""
        # Setup mocks
        mock_user_has_permission.return_value = lambda: mock_authenticated_user
        mock_manager = AsyncMock()
        mock_role_manager_class.return_value = mock_manager
        mock_manager.list_roles.side_effect = Exception("Unexpected database error")

        # Make request
        response = client.get("/api/v1/roles/")

        # Assert response
        assert response.status_code == 500
        response_data = response.json()
        detail_text = response_data.get("detail") or response_data.get("error", "")
        assert "Internal server error" in detail_text


class TestPermissionEnforcement:
    """Test permission enforcement for role endpoints."""

    def test_insufficient_permissions_role_creation(self, client):
        """Test role creation with insufficient permissions."""

        # Setup mocks to simulate permission denial
        def mock_permission_checker():
            from fastapi import HTTPException
            raise HTTPException(status_code=403, detail="Insufficient permissions: roles:create required")

        app.dependency_overrides[require_permission(Permissions.ROLES_CREATE)] = mock_permission_checker

        # Test data
        request_data = {
            "name": "test_role",
            "description": "Test role description",
        }

        # Make request
        response = client.post("/api/v1/roles/", json=request_data)

        # Assert response
        assert response.status_code == 403
        response_data = response.json()
        detail_text = response_data.get("detail") or response_data.get("error", "")
        assert "Insufficient permissions" in detail_text

    def test_insufficient_permissions_user_assignment(self, client):
        """Test user role assignment with insufficient permissions."""

        # Setup mocks to simulate permission denial
        def mock_permission_checker():
            from fastapi import HTTPException
            raise HTTPException(status_code=403, detail="Insufficient permissions: roles:assign required")

        app.dependency_overrides[require_permission(Permissions.ROLES_ASSIGN)] = mock_permission_checker

        # Test data
        request_data = {
            "user_email": "user@example.com",
            "role_name": "test_role",
        }

        # Make request
        response = client.post("/api/v1/roles/assignments", json=request_data)

        # Assert response
        assert response.status_code == 403
        response_data = response.json()
        detail_text = response_data.get("detail") or response_data.get("error", "")
        assert "Insufficient permissions" in detail_text


class TestRequestValidation:
    """Test request validation and edge cases."""

    def test_invalid_json_request(self, client):
        """Test handling of invalid JSON in request body."""
        # Make request with invalid JSON
        response = client.post("/api/v1/roles/", data="invalid json", headers={"Content-Type": "application/json"})

        # Assert response
        assert response.status_code == 422  # Unprocessable Entity

    @patch("src.auth.permissions.user_has_permission")
    def test_missing_required_fields(self, mock_user_has_permission, client, mock_authenticated_user):
        """Test handling of missing required fields."""
        # Setup mocks
        mock_user_has_permission.return_value = lambda: mock_authenticated_user

        # Test data with missing required field
        request_data = {
            "description": "Test role description",
            # Missing 'name' field
        }

        # Make request
        response = client.post("/api/v1/roles/", json=request_data)

        # Assert response
        assert response.status_code == 422  # Validation error

    @patch("src.auth.permissions.user_has_permission")
    def test_empty_role_name(self, mock_user_has_permission, client, mock_authenticated_user):
        """Test handling of empty role name."""
        # Setup mocks
        mock_user_has_permission.return_value = lambda: mock_authenticated_user

        # Test data with empty name
        request_data = {
            "name": "",
            "description": "Test role description",
        }

        # Make request
        response = client.post("/api/v1/roles/", json=request_data)

        # Assert response
        assert response.status_code == 422  # Validation error
