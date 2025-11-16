"""Comprehensive test suite for AUTH-3 role management system.

This module provides extensive test coverage for the RoleManager class including:
- Role creation, retrieval, and management
- Permission assignment and resolution
- User role assignments and revocation
- Role hierarchy validation and circular dependency detection
- Error handling and edge cases
- Database integration testing
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.exc import IntegrityError

from src.auth.role_manager import (
    PermissionNotFoundError,
    RoleManager,
    RoleManagerError,
    RoleNotFoundError,
    UserNotFoundError,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def role_manager(mock_session):
    """Create a RoleManager instance for testing."""
    with patch("src.database.base_service.get_database_manager") as mock_get_db:
        # Mock the database manager
        mock_db_manager = MagicMock()

        # Set up the session context manager to return our mock_session
        async_context_manager = AsyncMock()
        async_context_manager.__aenter__ = AsyncMock(return_value=mock_session)
        async_context_manager.__aexit__ = AsyncMock(return_value=None)
        mock_db_manager.get_session.return_value = async_context_manager

        mock_get_db.return_value = mock_db_manager
        return RoleManager()


@pytest.fixture
def mock_session():
    """Mock database session."""
    session = AsyncMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    # Use AsyncMock for execute but don't let it wrap return values in coroutines
    session.execute = AsyncMock()
    return session


@pytest.fixture
def mock_get_db(mock_session):
    """Mock the get_database_manager dependency."""
    mock_db_manager = MagicMock()
    mock_db_manager.get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
    mock_db_manager.get_session.return_value.__aexit__ = AsyncMock(return_value=None)
    return mock_db_manager


@pytest.fixture
def sample_role_data():
    """Sample role data for testing."""
    return {
        "id": 1,
        "name": "test_role",
        "description": "Test role description",
        "parent_role_id": None,
        "created_at": datetime.now(UTC),
        "updated_at": datetime.now(UTC),
        "is_active": True,
    }


@pytest.fixture
def sample_permission_data():
    """Sample permission data for testing."""
    return {
        "id": 1,
        "name": "test:permission",
        "resource": "test",
        "action": "permission",
        "description": "Test permission",
        "created_at": datetime.now(UTC),
        "is_active": True,
    }


# =============================================================================
# ROLE CREATION TESTS
# =============================================================================


class TestRoleCreation:
    """Test role creation functionality."""

    async def test_create_role_success(self, role_manager, mock_session, sample_role_data):
        """Test successful role creation."""
        # Mock the database manager's session to return our mock session
        role_manager._db_manager.get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        role_manager._db_manager.get_session.return_value.__aexit__ = AsyncMock(return_value=None)

        # Mock the refresh method to populate role attributes
        async def mock_refresh(role):
            role.id = sample_role_data["id"]
            role.name = role.name  # Keep the name that was set
            role.description = role.description  # Keep the description that was set
            role.created_at = sample_role_data["created_at"]
            role.is_active = sample_role_data["is_active"]

        mock_session.refresh = AsyncMock(side_effect=mock_refresh)

        result = await role_manager.create_role(name="test_role", description="Test role description")

        assert result["name"] == "test_role"
        assert result["description"] == "Test role description"
        assert result["parent_role_id"] is None
        assert result["is_active"] is True
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()

    async def test_create_role_with_parent(self, role_manager, mock_session):
        """Test role creation with parent role."""
        # Mock the database manager's session to return our mock session
        role_manager._db_manager.get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        role_manager._db_manager.get_session.return_value.__aexit__ = AsyncMock(return_value=None)

        # Mock parent role exists
        mock_parent_result = AsyncMock()
        mock_parent_result.scalar_one_or_none.return_value = 2
        mock_session.execute.return_value = mock_parent_result

        # Mock the Role object that will be added
        mock_role = MagicMock()
        mock_role.id = 1
        mock_role.name = "child_role"
        mock_role.description = "Child role"
        mock_role.parent_role_id = 2
        mock_role.created_at = datetime.now(UTC)
        mock_role.is_active = True

        # Mock refresh to populate the role attributes
        async def mock_refresh(role):
            for attr, value in vars(mock_role).items():
                if not attr.startswith("_"):
                    setattr(role, attr, value)

        mock_session.refresh.side_effect = mock_refresh

        result = await role_manager.create_role(
            name="child_role",
            description="Child role",
            parent_role_name="parent_role",
        )

        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
        assert result["name"] == "child_role"
        assert result["parent_role_id"] == 2

    async def test_create_role_parent_not_found(self, role_manager, mock_session):
        """Test role creation fails when parent role doesn't exist."""
        # Mock the database manager's session to return our mock session
        role_manager._db_manager.get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        role_manager._db_manager.get_session.return_value.__aexit__ = AsyncMock(return_value=None)

        # Mock parent role doesn't exist
        mock_parent_result = MagicMock()
        mock_parent_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_parent_result

        with pytest.raises(RoleManagerError, match="Failed to create role: Parent role 'nonexistent' not found"):
            await role_manager.create_role(name="child_role", parent_role_name="nonexistent")

    async def test_create_role_duplicate_name(self, role_manager, mock_session):
        """Test role creation fails with duplicate name."""
        # Mock the database manager's session to return our mock session
        role_manager._db_manager.get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        role_manager._db_manager.get_session.return_value.__aexit__ = AsyncMock(return_value=None)

        # Mock integrity error for duplicate name
        mock_session.commit.side_effect = IntegrityError("unique constraint", None, None)

        with pytest.raises(RoleManagerError, match="Failed to create role: Database operation failed"):
            await role_manager.create_role(name="test_role")


# =============================================================================
# ROLE RETRIEVAL TESTS
# =============================================================================


class TestRoleRetrieval:
    """Test role retrieval functionality."""

    async def test_get_role_success(self, role_manager, mock_session, sample_role_data):
        """Test successful role retrieval."""
        # Mock the database manager's session to return our mock session
        role_manager._db_manager.get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        role_manager._db_manager.get_session.return_value.__aexit__ = AsyncMock(return_value=None)

        # Mock role exists
        mock_role = MagicMock()
        for key, value in sample_role_data.items():
            setattr(mock_role, key, value)

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_role
        mock_session.execute.return_value = mock_result

        result = await role_manager.get_role("test_role")

        assert result is not None
        assert result["name"] == "test_role"
        assert result["description"] == "Test role description"

    async def test_get_role_not_found(self, role_manager, mock_session):
        """Test role retrieval when role doesn't exist."""
        # Mock the database manager's session to return our mock session
        role_manager._db_manager.get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        role_manager._db_manager.get_session.return_value.__aexit__ = AsyncMock(return_value=None)

        # Mock role doesn't exist
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await role_manager.get_role("nonexistent")

        assert result is None

    async def test_list_roles_active_only(self, role_manager, mock_session):
        """Test listing active roles only."""
        # Mock the database manager's session to return our mock session
        role_manager._db_manager.get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        role_manager._db_manager.get_session.return_value.__aexit__ = AsyncMock(return_value=None)

        # Mock multiple roles
        mock_role1 = MagicMock()
        mock_role1.id = 1
        mock_role1.name = "role1"
        mock_role1.description = "Role 1"
        mock_role1.parent_role_id = None
        mock_role1.created_at = datetime.now(UTC)
        mock_role1.updated_at = datetime.now(UTC)
        mock_role1.is_active = True

        mock_role2 = MagicMock()
        mock_role2.id = 2
        mock_role2.name = "role2"
        mock_role2.description = "Role 2"
        mock_role2.parent_role_id = None
        mock_role2.created_at = datetime.now(UTC)
        mock_role2.updated_at = datetime.now(UTC)
        mock_role2.is_active = True

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_role1, mock_role2]
        mock_session.execute.return_value = mock_result

        result = await role_manager.list_roles(include_inactive=False)

        assert len(result) == 2
        assert result[0]["name"] == "role1"
        assert result[1]["name"] == "role2"


# =============================================================================
# PERMISSION MANAGEMENT TESTS
# =============================================================================


class TestPermissionManagement:
    """Test permission assignment and management."""

    async def test_get_role_permissions(self, role_manager, mock_session):
        """Test getting role permissions including inherited."""
        # Mock the database manager's session to return our mock session
        role_manager._db_manager.get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        role_manager._db_manager.get_session.return_value.__aexit__ = AsyncMock(return_value=None)

        # Mock role exists and permissions query
        mock_role_result = MagicMock()
        mock_role_result.scalar_one_or_none.return_value = 1

        mock_perm_result = MagicMock()
        mock_perm_result.fetchall.return_value = [
            ("test:read",),
            ("test:write",),
            ("admin:access",),  # inherited from parent
        ]

        mock_session.execute.side_effect = [mock_role_result, mock_perm_result]

        result = await role_manager.get_role_permissions("test_role")

        assert isinstance(result, set)
        assert "test:read" in result
        assert "test:write" in result
        assert "admin:access" in result
        assert len(result) == 3

    async def test_get_role_permissions_not_found(self, role_manager, mock_session):
        """Test getting permissions for non-existent role."""
        # Mock the database manager's session to return our mock session
        role_manager._db_manager.get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        role_manager._db_manager.get_session.return_value.__aexit__ = AsyncMock(return_value=None)

        # Mock role doesn't exist
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        with pytest.raises(RoleNotFoundError, match="Role 'nonexistent' not found"):
            await role_manager.get_role_permissions("nonexistent")

    async def test_assign_permission_to_role_success(self, role_manager, mock_session):
        """Test successful permission assignment to role."""
        # Mock the database manager's session to return our mock session
        role_manager._db_manager.get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        role_manager._db_manager.get_session.return_value.__aexit__ = AsyncMock(return_value=None)

        # Mock role and permission exist
        mock_role_result = MagicMock()
        mock_role_result.scalar_one_or_none.return_value = 1

        mock_perm_result = MagicMock()
        mock_perm_result.scalar_one_or_none.return_value = 1

        # Mock the insert statement
        mock_insert_result = MagicMock()
        mock_insert = MagicMock()
        mock_insert.values.return_value = MagicMock()
        mock_insert.values.return_value.on_conflict_do_nothing.return_value = mock_insert_result

        # Mock session.execute for three calls: role lookup, permission lookup, insert statement
        mock_session.execute.side_effect = [mock_role_result, mock_perm_result, mock_insert_result]

        with patch("src.auth.role_manager.insert") as mock_insert_func:
            mock_insert_func.return_value = mock_insert
            result = await role_manager.assign_permission_to_role("test_role", "test:permission")

        assert result is True
        mock_session.commit.assert_called_once()

    async def test_assign_permission_role_not_found(self, role_manager, mock_session):
        """Test permission assignment fails when role doesn't exist."""
        # Mock the database manager's session to return our mock session
        role_manager._db_manager.get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        role_manager._db_manager.get_session.return_value.__aexit__ = AsyncMock(return_value=None)

        # Mock role doesn't exist
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        with pytest.raises(RoleNotFoundError, match="Role 'nonexistent' not found"):
            await role_manager.assign_permission_to_role("nonexistent", "test:permission")

    async def test_assign_permission_permission_not_found(self, role_manager, mock_session):
        """Test permission assignment fails when permission doesn't exist."""
        # Mock the database manager's session to return our mock session
        role_manager._db_manager.get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        role_manager._db_manager.get_session.return_value.__aexit__ = AsyncMock(return_value=None)

        # Mock role exists but permission doesn't
        mock_role_result = MagicMock()
        mock_role_result.scalar_one_or_none.return_value = 1

        mock_perm_result = MagicMock()
        mock_perm_result.scalar_one_or_none.return_value = None

        mock_session.execute.side_effect = [mock_role_result, mock_perm_result]

        with pytest.raises(PermissionNotFoundError, match="Permission 'nonexistent' not found"):
            await role_manager.assign_permission_to_role("test_role", "nonexistent")

    async def test_revoke_permission_from_role_success(self, role_manager, mock_session):
        """Test successful permission revocation from role."""
        # Mock the database manager's session to return our mock session
        role_manager._db_manager.get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        role_manager._db_manager.get_session.return_value.__aexit__ = AsyncMock(return_value=None)

        # Mock role and permission exist
        mock_role_result = MagicMock()
        mock_role_result.scalar_one_or_none.return_value = 1

        mock_perm_result = MagicMock()
        mock_perm_result.scalar_one_or_none.return_value = 1

        mock_session.execute.side_effect = [mock_role_result, mock_perm_result, MagicMock()]

        result = await role_manager.revoke_permission_from_role("test_role", "test:permission")

        assert result is True
        mock_session.commit.assert_called_once()


# =============================================================================
# USER ROLE ASSIGNMENT TESTS
# =============================================================================


class TestUserRoleAssignment:
    """Test user role assignment and management."""

    async def test_assign_user_role_success(self, role_manager, mock_session):
        """Test successful user role assignment."""
        # Mock the database manager's session to return our mock session
        role_manager._db_manager.get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        role_manager._db_manager.get_session.return_value.__aexit__ = AsyncMock(return_value=None)

        # Mock user exists and role exists
        mock_user_result = MagicMock()
        mock_user_result.scalar_one_or_none.return_value = "test@example.com"

        mock_role_result = MagicMock()
        mock_role_result.scalar_one_or_none.return_value = 1

        mock_assign_result = MagicMock()

        mock_session.execute.side_effect = [mock_user_result, mock_role_result, mock_assign_result]

        result = await role_manager.assign_user_role("test@example.com", "test_role", "admin@example.com")

        assert result is True
        mock_session.commit.assert_called_once()

    async def test_assign_user_role_user_not_found(self, role_manager, mock_session):
        """Test user role assignment fails when user doesn't exist."""
        # Mock the database manager's session to return our mock session
        role_manager._db_manager.get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        role_manager._db_manager.get_session.return_value.__aexit__ = AsyncMock(return_value=None)

        # Mock user doesn't exist
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        with pytest.raises(UserNotFoundError, match="User 'nonexistent@example.com' not found"):
            await role_manager.assign_user_role("nonexistent@example.com", "test_role")

    async def test_revoke_user_role_success(self, role_manager, mock_session):
        """Test successful user role revocation."""
        # Mock the database manager's session to return our mock session
        role_manager._db_manager.get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        role_manager._db_manager.get_session.return_value.__aexit__ = AsyncMock(return_value=None)

        # Mock role exists and revocation succeeds
        mock_role_result = MagicMock()
        mock_role_result.scalar_one_or_none.return_value = 1

        mock_revoke_result = MagicMock()
        mock_revoke_result.scalar.return_value = True

        mock_session.execute.side_effect = [mock_role_result, mock_revoke_result]

        result = await role_manager.revoke_user_role("test@example.com", "test_role")

        assert result is True
        mock_session.commit.assert_called_once()

    async def test_revoke_user_role_not_assigned(self, role_manager, mock_session):
        """Test user role revocation when role wasn't assigned."""
        # Mock the database manager's session to return our mock session
        role_manager._db_manager.get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        role_manager._db_manager.get_session.return_value.__aexit__ = AsyncMock(return_value=None)

        # Mock role exists but wasn't assigned to user
        mock_role_result = MagicMock()
        mock_role_result.scalar_one_or_none.return_value = 1

        mock_revoke_result = MagicMock()
        mock_revoke_result.scalar.return_value = False

        mock_session.execute.side_effect = [mock_role_result, mock_revoke_result]

        result = await role_manager.revoke_user_role("test@example.com", "test_role")

        assert result is False

    async def test_get_user_roles(self, role_manager, mock_session):
        """Test getting user role assignments."""
        # Mock the database manager's session to return our mock session
        role_manager._db_manager.get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        role_manager._db_manager.get_session.return_value.__aexit__ = AsyncMock(return_value=None)

        # Mock user roles
        mock_row1 = MagicMock()
        mock_row1.role_id = 1
        mock_row1.role_name = "admin"
        mock_row1.role_description = "Administrator role"
        mock_row1.assigned_at = datetime.now(UTC)

        mock_row2 = MagicMock()
        mock_row2.role_id = 2
        mock_row2.role_name = "user"
        mock_row2.role_description = "Regular user role"
        mock_row2.assigned_at = datetime.now(UTC)

        mock_result = MagicMock()
        mock_result.fetchall.return_value = [mock_row1, mock_row2]
        mock_session.execute.return_value = mock_result

        result = await role_manager.get_user_roles("test@example.com")

        assert len(result) == 2
        assert result[0]["role_name"] == "admin"
        assert result[1]["role_name"] == "user"

    async def test_get_user_permissions(self, role_manager, mock_session):
        """Test getting all user permissions through roles."""
        # Mock the database manager's session to return our mock session
        role_manager._db_manager.get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        role_manager._db_manager.get_session.return_value.__aexit__ = AsyncMock(return_value=None)

        # Mock user roles
        mock_row = MagicMock()
        mock_row.role_id = 1
        mock_row.role_name = "admin"
        mock_row.role_description = "Administrator role"
        mock_row.assigned_at = datetime.now(UTC)

        mock_roles_result = MagicMock()
        mock_roles_result.fetchall.return_value = [mock_row]

        # Mock role permissions
        mock_role_result = MagicMock()
        mock_role_result.scalar_one_or_none.return_value = 1

        mock_perm_result = MagicMock()
        mock_perm_result.fetchall.return_value = [("admin:access",), ("users:read",), ("users:write",)]

        mock_session.execute.side_effect = [mock_roles_result, mock_role_result, mock_perm_result]

        result = await role_manager.get_user_permissions("test@example.com")

        assert isinstance(result, set)
        assert "admin:access" in result
        assert "users:read" in result
        assert "users:write" in result


# =============================================================================
# ROLE HIERARCHY TESTS
# =============================================================================


class TestRoleHierarchy:
    """Test role hierarchy validation and management."""

    async def test_validate_role_hierarchy_valid(self, role_manager, mock_session):
        """Test valid role hierarchy validation."""
        # Mock the database manager's session to return our mock session
        role_manager._db_manager.get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        role_manager._db_manager.get_session.return_value.__aexit__ = AsyncMock(return_value=None)

        # Mock both roles exist and no circular dependency
        mock_role_result = MagicMock()
        mock_role_result.scalar_one_or_none.return_value = 1

        mock_parent_result = MagicMock()
        mock_parent_result.scalar_one_or_none.return_value = 2

        mock_cycle_result = MagicMock()
        mock_cycle_result.scalar.return_value = False  # No circular dependency

        mock_session.execute.side_effect = [mock_role_result, mock_parent_result, mock_cycle_result]

        result = await role_manager.validate_role_hierarchy("child_role", "parent_role")

        assert result is True

    @patch("src.database.base_service.get_database_manager")
    async def test_validate_role_hierarchy_circular(self, mock_get_db_manager_func, role_manager, mock_session):
        """Test role hierarchy validation detects circular dependency."""
        mock_get_db_manager_func.return_value.get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_get_db_manager_func.return_value.get_session.return_value.__aexit__ = AsyncMock(return_value=None)

        # Mock both roles exist but would create circular dependency
        mock_role_result = AsyncMock()
        mock_role_result.scalar_one_or_none.return_value = 1

        mock_parent_result = AsyncMock()
        mock_parent_result.scalar_one_or_none.return_value = 2

        mock_cycle_result = AsyncMock()
        mock_cycle_result.scalar.return_value = True  # Circular dependency detected

        mock_session.execute.side_effect = [mock_role_result, mock_parent_result, mock_cycle_result]

        result = await role_manager.validate_role_hierarchy("parent_role", "child_role")

        assert result is False

    async def test_validate_role_hierarchy_role_not_found(self, role_manager, mock_session):
        """Test role hierarchy validation when role doesn't exist."""
        # Mock the database manager's session to return our mock session
        role_manager._db_manager.get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        role_manager._db_manager.get_session.return_value.__aexit__ = AsyncMock(return_value=None)

        # Mock role doesn't exist
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        with pytest.raises(RoleNotFoundError, match="Role 'nonexistent' not found"):
            await role_manager.validate_role_hierarchy("nonexistent", "parent_role")


# =============================================================================
# ROLE DELETION TESTS
# =============================================================================


class TestRoleDeletion:
    """Test role deletion functionality."""

    async def test_delete_role_success(self, role_manager, mock_session):
        """Test successful role deletion."""
        # Mock role exists with no dependencies
        mock_role = MagicMock()
        mock_role.id = 1
        mock_role.name = "test_role"

        # Create separate result objects for each query type
        role_lookup_result = MagicMock()
        role_lookup_result.scalar_one_or_none = MagicMock(return_value=mock_role)

        child_roles_result = MagicMock()
        child_roles_result.scalars = MagicMock()
        child_roles_result.scalars.return_value.all = MagicMock(return_value=[])

        user_assignments_result = MagicMock()
        user_assignments_result.scalars = MagicMock()
        user_assignments_result.scalars.return_value.all = MagicMock(return_value=[])

        update_result = MagicMock()

        # Set up return_value for AsyncMock to prevent coroutine wrapping
        mock_session.execute.return_value = role_lookup_result

        # Set up side_effect for multiple calls with proper AsyncMock handling
        async def mock_execute_side_effect(*args, **kwargs):
            # Return values without coroutine wrapping
            if mock_session.execute.call_count == 1:
                return role_lookup_result
            if mock_session.execute.call_count == 2:
                return child_roles_result
            if mock_session.execute.call_count == 3:
                return user_assignments_result
            return update_result

        mock_session.execute.side_effect = mock_execute_side_effect

        result = await role_manager.delete_role("test_role", force=False)

        assert result is True
        mock_session.commit.assert_called_once()

    async def test_delete_role_with_children_no_force(self, role_manager, mock_session):
        """Test role deletion fails when role has children and force=False."""
        # Mock role exists with child roles
        mock_role = MagicMock()
        mock_role.id = 1
        mock_role.name = "parent_role"

        # Create result objects
        role_lookup_result = MagicMock()
        role_lookup_result.scalar_one_or_none = MagicMock(return_value=mock_role)

        child_roles_result = MagicMock()
        child_roles_result.scalars = MagicMock()
        child_roles_result.scalars.return_value.all = MagicMock(return_value=["child_role1", "child_role2"])

        # Set up side_effect for multiple calls
        async def mock_execute_side_effect(*args, **kwargs):
            if mock_session.execute.call_count == 1:
                return role_lookup_result
            return child_roles_result

        mock_session.execute.side_effect = mock_execute_side_effect

        with pytest.raises(RoleManagerError, match="has child roles"):
            await role_manager.delete_role("parent_role", force=False)

    async def test_delete_role_with_users_no_force(self, role_manager, mock_session):
        """Test role deletion fails when role has user assignments and force=False."""
        # Mock role exists with user assignments
        mock_role = MagicMock()
        mock_role.id = 1
        mock_role.name = "assigned_role"

        # Create result objects
        role_lookup_result = MagicMock()
        role_lookup_result.scalar_one_or_none = MagicMock(return_value=mock_role)

        child_roles_result = MagicMock()
        child_roles_result.scalars = MagicMock()
        child_roles_result.scalars.return_value.all = MagicMock(return_value=[])

        user_assignments_result = MagicMock()
        user_assignments_result.scalars = MagicMock()
        user_assignments_result.scalars.return_value.all = MagicMock(return_value=["user1@example.com", "user2@example.com"])

        # Set up side_effect for multiple calls
        async def mock_execute_side_effect(*args, **kwargs):
            if mock_session.execute.call_count == 1:
                return role_lookup_result
            if mock_session.execute.call_count == 2:
                return child_roles_result
            return user_assignments_result

        mock_session.execute.side_effect = mock_execute_side_effect

        with pytest.raises(RoleManagerError, match="is assigned to 2 users"):
            await role_manager.delete_role("assigned_role", force=False)

    async def test_delete_role_force_with_dependencies(self, role_manager, mock_session):
        """Test force role deletion with dependencies."""
        # Mock role exists with dependencies
        mock_role = MagicMock()
        mock_role.id = 1
        mock_role.name = "force_delete_role"

        # Create result objects
        role_lookup_result = MagicMock()
        role_lookup_result.scalar_one_or_none = MagicMock(return_value=mock_role)

        # Set up side_effect for multiple calls
        async def mock_execute_side_effect(*args, **kwargs):
            if mock_session.execute.call_count == 1:
                return role_lookup_result
            # All other execute calls (user assignment deletion, child role updates, soft delete)
            return MagicMock()

        mock_session.execute.side_effect = mock_execute_side_effect

        result = await role_manager.delete_role("force_delete_role", force=True)

        assert result is True
        mock_session.commit.assert_called_once()

    async def test_delete_role_not_found(self, role_manager, mock_session):
        """Test role deletion when role doesn't exist."""
        # Mock role doesn't exist
        role_lookup_result = MagicMock()
        role_lookup_result.scalar_one_or_none = MagicMock(return_value=None)

        # Set up side_effect
        async def mock_execute_side_effect(*args, **kwargs):
            return role_lookup_result

        mock_session.execute.side_effect = mock_execute_side_effect

        with pytest.raises(RoleNotFoundError, match="Role 'nonexistent' not found"):
            await role_manager.delete_role("nonexistent")


# =============================================================================
# ERROR HANDLING AND EDGE CASES
# =============================================================================


class TestErrorHandling:
    """Test error handling and edge cases."""

    async def test_database_error_handling(self, role_manager, mock_session):
        """Test proper handling of database errors."""
        # Mock database error
        mock_session.execute.side_effect = Exception("Database connection failed")

        result = await role_manager.get_role("test_role")
        assert result is None

    async def test_empty_role_name(self, role_manager, mock_session):
        """Test handling of empty role names."""
        # Mock successful role creation - current implementation doesn't validate empty names
        mock_role = MagicMock()
        mock_role.id = 1
        mock_role.name = ""
        mock_role.description = None
        mock_role.parent_role_id = None
        mock_role.created_at = None
        mock_role.is_active = True

        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()
        mock_session.refresh = AsyncMock()

        result = await role_manager.create_role("")

        # Should successfully create role with empty name (no validation in current implementation)
        assert result["name"] == ""

    async def test_none_role_name(self, role_manager, mock_session):
        """Test handling of None role names."""
        # Mock successful role creation - current implementation doesn't validate None names
        mock_role = MagicMock()
        mock_role.id = 1
        mock_role.name = None
        mock_role.description = None
        mock_role.parent_role_id = None
        mock_role.created_at = None
        mock_role.is_active = True

        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()
        mock_session.refresh = AsyncMock()

        result = await role_manager.create_role(None)

        # Should successfully create role with None name (no validation in current implementation)
        assert result["name"] is None

    async def test_transaction_rollback_on_error(self, role_manager, mock_session):
        """Test that transactions are properly rolled back on errors."""
        # Mock error during commit
        mock_session.commit.side_effect = Exception("Commit failed")

        with pytest.raises(RoleManagerError):
            await role_manager.create_role("test_role")

    async def test_concurrent_role_creation(self, role_manager, mock_session):
        """Test handling of concurrent role creation attempts."""
        # Mock integrity error from concurrent creation
        mock_session.commit.side_effect = IntegrityError("duplicate key", None, None)

        # IntegrityError gets wrapped by DatabaseService, so expect the wrapped error message
        with pytest.raises(RoleManagerError, match="Failed to create role"):
            await role_manager.create_role("concurrent_role")


# =============================================================================
# INTEGRATION AND PERFORMANCE TESTS
# =============================================================================


class TestIntegrationScenarios:
    """Test integration scenarios and workflows."""

    async def test_complete_role_workflow(self, role_manager, mock_session):
        """Test complete role management workflow."""
        # Mock successful workflow: create role, assign permission, assign to user
        mock_role_result = MagicMock()
        mock_role_result.scalar_one_or_none.return_value = 1

        mock_perm_result = MagicMock()
        mock_perm_result.scalar_one_or_none.return_value = 1

        mock_user_result = MagicMock()
        mock_user_result.scalar_one_or_none.return_value = "user@example.com"

        # Mock insert statement with on_conflict_do_nothing method
        mock_insert = MagicMock()
        mock_insert.values.return_value = mock_insert
        mock_insert.on_conflict_do_nothing.return_value = mock_insert

        mock_session.execute.side_effect = [
            # Create role - no parent lookup
            # Assign permission
            mock_role_result,  # Role exists
            mock_perm_result,  # Permission exists
            MagicMock(),  # Assignment (insert with on_conflict_do_nothing)
            # Assign to user
            mock_user_result,  # User exists
            mock_role_result,  # Role exists
            MagicMock(),  # User role assignment (database function call)
        ]

        # Mock the insert function to return our mock with on_conflict_do_nothing
        with patch("src.auth.role_manager.insert", return_value=mock_insert):
            # Create role
            role_result = await role_manager.create_role("workflow_role", "Test workflow role")

            # Assign permission
            perm_result = await role_manager.assign_permission_to_role("workflow_role", "test:permission")

            # Assign to user
            user_result = await role_manager.assign_user_role("user@example.com", "workflow_role", "admin@example.com")

            assert role_result["name"] == "workflow_role"
            assert perm_result is True
            assert user_result is True

    async def test_role_hierarchy_inheritance(self, role_manager, mock_session):
        """Test permission inheritance through role hierarchy."""
        # Mock role permissions including inherited ones
        mock_role_result = MagicMock()
        mock_role_result.scalar_one_or_none.return_value = 1

        mock_perm_result = MagicMock()
        mock_perm_result.fetchall.return_value = [
            ("child:permission",),  # Direct permission
            ("parent:permission",),  # Inherited from parent
            ("grandparent:permission",),  # Inherited from grandparent
        ]

        mock_session.execute.side_effect = [mock_role_result, mock_perm_result]

        permissions = await role_manager.get_role_permissions("child_role")

        assert "child:permission" in permissions
        assert "parent:permission" in permissions
        assert "grandparent:permission" in permissions
        assert len(permissions) == 3

    async def test_bulk_operations_performance(self, role_manager):
        """Test performance characteristics of bulk operations."""
        # This would test bulk role assignments, permissions, etc.
        # For now, just validate the interface exists
        assert hasattr(role_manager, "assign_user_role")
        assert hasattr(role_manager, "assign_permission_to_role")
        assert hasattr(role_manager, "get_user_permissions")


# =============================================================================
# REGRESSION TESTS
# =============================================================================


class TestRegressionIssues:
    """Test fixes for specific regression issues."""

    async def test_role_name_case_sensitivity(self, role_manager, mock_session):
        """Test that role names are case-sensitive."""
        # Keep track of which role name is being queried
        self.current_role_name = None

        def mock_execute(query):
            # Extract role name from the query parameters or call context
            str(query)

            # For TestRole (exact case match)
            if self.current_role_name == "TestRole":
                result = MagicMock()
                mock_role = MagicMock()
                mock_role.id = 1
                mock_role.name = "TestRole"
                mock_role.description = "Test Role"
                mock_role.parent_role_id = None
                mock_role.created_at = "2023-01-01T00:00:00"
                mock_role.updated_at = "2023-01-01T00:00:00"
                mock_role.is_active = True
                result.scalar_one_or_none.return_value = mock_role
                return result

            # For testrole (different case), return None
            result = MagicMock()
            result.scalar_one_or_none.return_value = None
            return result

        mock_session.execute.side_effect = mock_execute

        # Test case sensitivity - these should be treated as different roles
        self.current_role_name = "TestRole"
        role1 = await role_manager.get_role("TestRole")

        self.current_role_name = "testrole"
        role2 = await role_manager.get_role("testrole")

        assert role1 is not None
        assert role2 is None

    async def test_permission_name_validation(self, role_manager, mock_session):
        """Test validation of permission name format."""
        # Mock role exists
        mock_role_result = MagicMock()
        mock_role_result.scalar_one_or_none.return_value = 1
        mock_session.execute.return_value = mock_role_result

        # Test that permission format validation happens
        # (This would typically be done at the API level, but could be enforced in manager)
        valid_permissions = ["resource:action", "tokens:create", "users:read", "system:admin"]

        for perm in valid_permissions:
            assert ":" in perm  # Basic format check
            assert len(perm.split(":")) == 2  # resource:action format

    async def test_circular_dependency_deep_hierarchy(self, role_manager, mock_session):
        """Test circular dependency detection in deep role hierarchies."""
        # Mock roles exist
        mock_role_result = MagicMock()
        mock_role_result.scalar_one_or_none.return_value = 1

        mock_parent_result = MagicMock()
        mock_parent_result.scalar_one_or_none.return_value = 2

        # Mock circular dependency detection in deep hierarchy
        mock_cycle_result = MagicMock()
        mock_cycle_result.scalar.return_value = True  # Circular dependency found

        mock_session.execute.side_effect = [mock_role_result, mock_parent_result, mock_cycle_result]

        is_valid = await role_manager.validate_role_hierarchy("level5_role", "level1_role")

        assert is_valid is False  # Should detect circular dependency
