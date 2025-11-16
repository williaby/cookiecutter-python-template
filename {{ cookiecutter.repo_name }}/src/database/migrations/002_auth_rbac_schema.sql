-- PromptCraft Authorization and Permission System Schema Migration
-- Migration: 002_auth_rbac_schema.sql
-- Description: Role-based access control (RBAC) schema for AUTH-3 Advanced Authorization and Permission System
-- Author: Phase 1 Issue AUTH-3 Implementation
-- Date: 2025-08-12
-- Dependencies: 001_auth_schema.sql (AUTH-1 and AUTH-2)

-- Ensure we have necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Enable row-level security for enhanced security
SET row_security = on;

-- =============================================================================
-- AUTH-3 ROLE-BASED ACCESS CONTROL TABLES
-- =============================================================================

-- Create roles table for hierarchical role-based access control
CREATE TABLE IF NOT EXISTS roles (
    -- Primary key
    id SERIAL PRIMARY KEY,

    -- Role identification
    name VARCHAR(50) NOT NULL UNIQUE,
    description TEXT NULL,

    -- Role hierarchy support
    parent_role_id INTEGER NULL REFERENCES roles(id) ON DELETE SET NULL,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Status
    is_active BOOLEAN NOT NULL DEFAULT TRUE,

    -- Constraints
    CONSTRAINT chk_roles_name_format CHECK (name ~ '^[a-z0-9_]+$'),
    CONSTRAINT chk_roles_no_self_parent CHECK (id != parent_role_id)
);

-- Add table comment
COMMENT ON TABLE roles IS 'Role definitions for hierarchical role-based access control (AUTH-3)';

-- Add column comments
COMMENT ON COLUMN roles.id IS 'Unique role identifier';
COMMENT ON COLUMN roles.name IS 'Unique role name (lowercase, underscore-separated)';
COMMENT ON COLUMN roles.description IS 'Human-readable description of the role';
COMMENT ON COLUMN roles.parent_role_id IS 'Parent role ID for inheritance (NULL for top-level roles)';
COMMENT ON COLUMN roles.created_at IS 'Role creation timestamp';
COMMENT ON COLUMN roles.updated_at IS 'Role last update timestamp';
COMMENT ON COLUMN roles.is_active IS 'Whether the role is active and can be assigned';

-- Create permissions table for fine-grained access control
CREATE TABLE IF NOT EXISTS permissions (
    -- Primary key
    id SERIAL PRIMARY KEY,

    -- Permission identification
    name VARCHAR(100) NOT NULL UNIQUE,
    resource VARCHAR(50) NOT NULL,
    action VARCHAR(50) NOT NULL,
    description TEXT NULL,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Status
    is_active BOOLEAN NOT NULL DEFAULT TRUE,

    -- Constraints
    CONSTRAINT chk_permissions_name_format CHECK (name ~ '^[a-z0-9_:]+$'),
    CONSTRAINT chk_permissions_resource_format CHECK (resource ~ '^[a-z0-9_]+$'),
    CONSTRAINT chk_permissions_action_format CHECK (action ~ '^[a-z0-9_]+$')
);

-- Add table comment
COMMENT ON TABLE permissions IS 'Permission definitions for fine-grained access control (AUTH-3)';

-- Add column comments
COMMENT ON COLUMN permissions.id IS 'Unique permission identifier';
COMMENT ON COLUMN permissions.name IS 'Unique permission name (format: resource:action)';
COMMENT ON COLUMN permissions.resource IS 'Resource type this permission applies to';
COMMENT ON COLUMN permissions.action IS 'Action this permission allows';
COMMENT ON COLUMN permissions.description IS 'Human-readable description of the permission';
COMMENT ON COLUMN permissions.created_at IS 'Permission creation timestamp';
COMMENT ON COLUMN permissions.is_active IS 'Whether the permission is active and can be assigned';

-- Create junction table for role-permission relationships (many-to-many)
CREATE TABLE IF NOT EXISTS role_permissions (
    role_id INTEGER NOT NULL REFERENCES roles(id) ON DELETE CASCADE,
    permission_id INTEGER NOT NULL REFERENCES permissions(id) ON DELETE CASCADE,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Primary key
    PRIMARY KEY (role_id, permission_id)
);

-- Add table comment
COMMENT ON TABLE role_permissions IS 'Junction table for role-permission assignments (AUTH-3)';

-- Create junction table for user-role relationships (many-to-many)
-- Uses email from existing user_sessions table as foreign key
CREATE TABLE IF NOT EXISTS user_roles (
    user_email VARCHAR(255) NOT NULL REFERENCES user_sessions(email) ON DELETE CASCADE,
    role_id INTEGER NOT NULL REFERENCES roles(id) ON DELETE CASCADE,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    assigned_by VARCHAR(255) NULL, -- Email of admin who assigned the role

    -- Primary key
    PRIMARY KEY (user_email, role_id)
);

-- Add table comment
COMMENT ON TABLE user_roles IS 'Junction table for user-role assignments (AUTH-3)';

-- Add column comments
COMMENT ON COLUMN user_roles.user_email IS 'User email from user_sessions table';
COMMENT ON COLUMN user_roles.role_id IS 'Role ID from roles table';
COMMENT ON COLUMN user_roles.created_at IS 'Role assignment timestamp';
COMMENT ON COLUMN user_roles.assigned_by IS 'Email of admin who assigned the role';

-- =============================================================================
-- AUTH-3 INDEXES FOR OPTIMAL PERFORMANCE
-- =============================================================================

-- Roles table indexes
CREATE INDEX IF NOT EXISTS idx_roles_name ON roles(name);
CREATE INDEX IF NOT EXISTS idx_roles_active ON roles(is_active);
CREATE INDEX IF NOT EXISTS idx_roles_parent ON roles(parent_role_id);
CREATE INDEX IF NOT EXISTS idx_roles_created_at ON roles(created_at DESC);

-- Permissions table indexes
CREATE INDEX IF NOT EXISTS idx_permissions_name ON permissions(name);
CREATE INDEX IF NOT EXISTS idx_permissions_resource ON permissions(resource);
CREATE INDEX IF NOT EXISTS idx_permissions_action ON permissions(action);
CREATE INDEX IF NOT EXISTS idx_permissions_active ON permissions(is_active);
CREATE INDEX IF NOT EXISTS idx_permissions_resource_action ON permissions(resource, action);

-- Role-permission junction table indexes
CREATE INDEX IF NOT EXISTS idx_role_permissions_role ON role_permissions(role_id);
CREATE INDEX IF NOT EXISTS idx_role_permissions_permission ON role_permissions(permission_id);

-- User-role junction table indexes
CREATE INDEX IF NOT EXISTS idx_user_roles_email ON user_roles(user_email);
CREATE INDEX IF NOT EXISTS idx_user_roles_role ON user_roles(role_id);
CREATE INDEX IF NOT EXISTS idx_user_roles_assigned_by ON user_roles(assigned_by);

-- Composite indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_roles_active_name ON roles(is_active, name);
CREATE INDEX IF NOT EXISTS idx_permissions_active_name ON permissions(is_active, name);

-- =============================================================================
-- AUTH-3 FUNCTIONS AND PROCEDURES
-- =============================================================================

-- Create function to automatically update updated_at timestamp for roles
CREATE OR REPLACE FUNCTION update_role_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger to automatically update updated_at when roles are modified
CREATE TRIGGER trigger_update_role_updated_at
    BEFORE UPDATE ON roles
    FOR EACH ROW
    EXECUTE FUNCTION update_role_updated_at();

-- Create function to get all permissions for a role (including inherited)
CREATE OR REPLACE FUNCTION get_role_permissions(p_role_id INTEGER)
RETURNS TABLE(
    permission_id INTEGER,
    permission_name VARCHAR(100),
    resource VARCHAR(50),
    action VARCHAR(50)
) AS $$
WITH RECURSIVE role_hierarchy AS (
    -- Base case: start with the specified role
    SELECT id, parent_role_id, name, 0 as depth
    FROM roles
    WHERE id = p_role_id AND is_active = true

    UNION ALL

    -- Recursive case: find parent roles
    SELECT r.id, r.parent_role_id, r.name, rh.depth + 1
    FROM roles r
    INNER JOIN role_hierarchy rh ON r.id = rh.parent_role_id
    WHERE r.is_active = true AND rh.depth < 10 -- Prevent infinite recursion
)
SELECT DISTINCT
    p.id as permission_id,
    p.name as permission_name,
    p.resource,
    p.action
FROM role_hierarchy rh
INNER JOIN role_permissions rp ON rh.id = rp.role_id
INNER JOIN permissions p ON rp.permission_id = p.id
WHERE p.is_active = true
ORDER BY p.resource, p.action;
$$ LANGUAGE sql;

-- Add function comment
COMMENT ON FUNCTION get_role_permissions(INTEGER) IS 'Get all permissions for a role including inherited permissions from parent roles';

-- Create function to check if a user has a specific permission
CREATE OR REPLACE FUNCTION user_has_permission(p_user_email VARCHAR(255), p_permission_name VARCHAR(100))
RETURNS BOOLEAN AS $$
DECLARE
    has_perm BOOLEAN DEFAULT FALSE;
BEGIN
    -- Check if user has the permission through any of their roles
    SELECT EXISTS(
        SELECT 1
        FROM user_roles ur
        INNER JOIN roles r ON ur.role_id = r.id
        INNER JOIN role_permissions rp ON r.id = rp.role_id
        INNER JOIN permissions p ON rp.permission_id = p.id
        WHERE ur.user_email = p_user_email
        AND p.name = p_permission_name
        AND r.is_active = true
        AND p.is_active = true

        UNION

        -- Also check inherited permissions from parent roles
        SELECT 1
        FROM user_roles ur
        INNER JOIN get_role_permissions(ur.role_id) grp ON grp.permission_name = p_permission_name
        WHERE ur.user_email = p_user_email
    ) INTO has_perm;

    RETURN has_perm;
END;
$$ LANGUAGE plpgsql;

-- Add function comment
COMMENT ON FUNCTION user_has_permission(VARCHAR, VARCHAR) IS 'Check if a user has a specific permission through their roles';

-- Create function to get all roles for a user
CREATE OR REPLACE FUNCTION get_user_roles(p_user_email VARCHAR(255))
RETURNS TABLE(
    role_id INTEGER,
    role_name VARCHAR(50),
    role_description TEXT,
    assigned_at TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        r.id as role_id,
        r.name as role_name,
        r.description as role_description,
        ur.created_at as assigned_at
    FROM user_roles ur
    INNER JOIN roles r ON ur.role_id = r.id
    WHERE ur.user_email = p_user_email
    AND r.is_active = true
    ORDER BY ur.created_at DESC;
END;
$$ LANGUAGE plpgsql;

-- Add function comment
COMMENT ON FUNCTION get_user_roles(VARCHAR) IS 'Get all active roles assigned to a user';

-- Create function to assign role to user
CREATE OR REPLACE FUNCTION assign_user_role(
    p_user_email VARCHAR(255),
    p_role_id INTEGER,
    p_assigned_by VARCHAR(255) DEFAULT NULL
)
RETURNS BOOLEAN AS $$
DECLARE
    role_exists BOOLEAN;
    user_exists BOOLEAN;
BEGIN
    -- Check if role exists and is active
    SELECT EXISTS(
        SELECT 1 FROM roles WHERE id = p_role_id AND is_active = true
    ) INTO role_exists;

    IF NOT role_exists THEN
        RAISE EXCEPTION 'Role with ID % does not exist or is inactive', p_role_id;
    END IF;

    -- Check if user exists in user_sessions
    SELECT EXISTS(
        SELECT 1 FROM user_sessions WHERE email = p_user_email
    ) INTO user_exists;

    IF NOT user_exists THEN
        RAISE EXCEPTION 'User with email % does not exist', p_user_email;
    END IF;

    -- Insert role assignment (ON CONFLICT DO NOTHING for idempotency)
    INSERT INTO user_roles (user_email, role_id, assigned_by)
    VALUES (p_user_email, p_role_id, p_assigned_by)
    ON CONFLICT (user_email, role_id) DO NOTHING;

    RETURN TRUE;
EXCEPTION
    WHEN OTHERS THEN
        RAISE;
END;
$$ LANGUAGE plpgsql;

-- Add function comment
COMMENT ON FUNCTION assign_user_role(VARCHAR, INTEGER, VARCHAR) IS 'Assign a role to a user with validation';

-- Create function to revoke role from user
CREATE OR REPLACE FUNCTION revoke_user_role(
    p_user_email VARCHAR(255),
    p_role_id INTEGER
)
RETURNS BOOLEAN AS $$
BEGIN
    DELETE FROM user_roles
    WHERE user_email = p_user_email AND role_id = p_role_id;

    RETURN FOUND;
END;
$$ LANGUAGE plpgsql;

-- Add function comment
COMMENT ON FUNCTION revoke_user_role(VARCHAR, INTEGER) IS 'Revoke a role from a user';

-- =============================================================================
-- AUTH-3 DEFAULT ROLES AND PERMISSIONS
-- =============================================================================

-- Insert default permissions
INSERT INTO permissions (name, resource, action, description) VALUES
    -- Service token management permissions
    ('tokens:create', 'tokens', 'create', 'Create new service tokens'),
    ('tokens:read', 'tokens', 'read', 'View service token information'),
    ('tokens:update', 'tokens', 'update', 'Update service token metadata'),
    ('tokens:delete', 'tokens', 'delete', 'Revoke or delete service tokens'),
    ('tokens:rotate', 'tokens', 'rotate', 'Rotate service token values'),

    -- User management permissions
    ('users:read', 'users', 'read', 'View user information and sessions'),
    ('users:update', 'users', 'update', 'Update user preferences and metadata'),
    ('users:delete', 'users', 'delete', 'Delete user sessions and data'),

    -- Role management permissions
    ('roles:create', 'roles', 'create', 'Create new roles'),
    ('roles:read', 'roles', 'read', 'View role information'),
    ('roles:update', 'roles', 'update', 'Update role definitions'),
    ('roles:delete', 'roles', 'delete', 'Delete roles'),
    ('roles:assign', 'roles', 'assign', 'Assign roles to users'),

    -- Permission management permissions
    ('permissions:create', 'permissions', 'create', 'Create new permissions'),
    ('permissions:read', 'permissions', 'read', 'View permission information'),
    ('permissions:update', 'permissions', 'update', 'Update permission definitions'),
    ('permissions:delete', 'permissions', 'delete', 'Delete permissions'),

    -- System administration permissions
    ('system:admin', 'system', 'admin', 'Full system administration access'),
    ('system:status', 'system', 'status', 'View system status and health'),
    ('system:audit', 'system', 'audit', 'Access audit logs and analytics'),
    ('system:monitor', 'system', 'monitor', 'Access monitoring and metrics'),

    -- API access permissions
    ('api:access', 'api', 'access', 'Basic API access'),
    ('api:admin', 'api', 'admin', 'Administrative API access')
ON CONFLICT (name) DO NOTHING;

-- Insert default roles with hierarchy
INSERT INTO roles (name, description, parent_role_id) VALUES
    ('super_admin', 'Super administrator with full system access', NULL),
    ('admin', 'Administrator with management access', NULL),
    ('user_manager', 'User management specialist', (SELECT id FROM roles WHERE name = 'admin')),
    ('token_manager', 'Service token management specialist', (SELECT id FROM roles WHERE name = 'admin')),
    ('api_user', 'Standard API user', NULL),
    ('readonly_user', 'Read-only access user', NULL)
ON CONFLICT (name) DO NOTHING;

-- Assign permissions to roles
-- Super admin gets all permissions
INSERT INTO role_permissions (role_id, permission_id)
SELECT
    (SELECT id FROM roles WHERE name = 'super_admin'),
    id
FROM permissions
ON CONFLICT DO NOTHING;

-- Admin gets most permissions (excluding super admin only permissions)
INSERT INTO role_permissions (role_id, permission_id)
SELECT
    (SELECT id FROM roles WHERE name = 'admin'),
    id
FROM permissions
WHERE name NOT IN ('system:admin')
ON CONFLICT DO NOTHING;

-- User manager gets user and role management permissions
INSERT INTO role_permissions (role_id, permission_id)
SELECT
    (SELECT id FROM roles WHERE name = 'user_manager'),
    id
FROM permissions
WHERE name IN (
    'users:read', 'users:update', 'users:delete',
    'roles:read', 'roles:assign',
    'system:status', 'api:access'
)
ON CONFLICT DO NOTHING;

-- Token manager gets token management permissions
INSERT INTO role_permissions (role_id, permission_id)
SELECT
    (SELECT id FROM roles WHERE name = 'token_manager'),
    id
FROM permissions
WHERE name IN (
    'tokens:create', 'tokens:read', 'tokens:update', 'tokens:delete', 'tokens:rotate',
    'system:status', 'system:audit', 'api:access'
)
ON CONFLICT DO NOTHING;

-- API user gets basic API access
INSERT INTO role_permissions (role_id, permission_id)
SELECT
    (SELECT id FROM roles WHERE name = 'api_user'),
    id
FROM permissions
WHERE name IN ('api:access', 'system:status')
ON CONFLICT DO NOTHING;

-- Readonly user gets read permissions only
INSERT INTO role_permissions (role_id, permission_id)
SELECT
    (SELECT id FROM roles WHERE name = 'readonly_user'),
    id
FROM permissions
WHERE name IN (
    'tokens:read', 'users:read', 'roles:read', 'permissions:read',
    'system:status', 'api:access'
)
ON CONFLICT DO NOTHING;

-- =============================================================================
-- AUTH-3 VIEWS FOR ANALYTICS AND REPORTING
-- =============================================================================

-- Create view for role hierarchy visualization
CREATE OR REPLACE VIEW role_hierarchy_view AS
WITH RECURSIVE role_tree AS (
    -- Root roles (no parent)
    SELECT
        id,
        name,
        description,
        parent_role_id,
        0 as level,
        name::text as path
    FROM roles
    WHERE parent_role_id IS NULL AND is_active = true

    UNION ALL

    -- Child roles
    SELECT
        r.id,
        r.name,
        r.description,
        r.parent_role_id,
        rt.level + 1,
        (rt.path || ' -> ' || r.name)::text
    FROM roles r
    INNER JOIN role_tree rt ON r.parent_role_id = rt.id
    WHERE r.is_active = true AND rt.level < 10
)
SELECT
    id,
    name,
    description,
    parent_role_id,
    level,
    path
FROM role_tree
ORDER BY path;

-- Add view comment
COMMENT ON VIEW role_hierarchy_view IS 'Hierarchical view of role relationships for visualization';

-- Create view for user permission summary
CREATE OR REPLACE VIEW user_permissions_summary AS
SELECT
    ur.user_email,
    r.name as role_name,
    p.name as permission_name,
    p.resource,
    p.action,
    ur.created_at as role_assigned_at,
    ur.assigned_by
FROM user_roles ur
INNER JOIN roles r ON ur.role_id = r.id
INNER JOIN role_permissions rp ON r.id = rp.role_id
INNER JOIN permissions p ON rp.permission_id = p.id
WHERE r.is_active = true AND p.is_active = true
ORDER BY ur.user_email, r.name, p.resource, p.action;

-- Add view comment
COMMENT ON VIEW user_permissions_summary IS 'Summary of all user permissions through their assigned roles';

-- =============================================================================
-- AUTH-3 PERMISSIONS AND SECURITY
-- =============================================================================

-- Grant appropriate permissions for the application user
GRANT SELECT, INSERT, UPDATE, DELETE ON roles TO promptcraft_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON permissions TO promptcraft_app;
GRANT SELECT, INSERT, DELETE ON role_permissions TO promptcraft_app;
GRANT SELECT, INSERT, DELETE ON user_roles TO promptcraft_app;

-- Grant usage on sequences
GRANT USAGE, SELECT ON SEQUENCE roles_id_seq TO promptcraft_app;
GRANT USAGE, SELECT ON SEQUENCE permissions_id_seq TO promptcraft_app;

-- Grant access to views
GRANT SELECT ON role_hierarchy_view TO promptcraft_app;
GRANT SELECT ON user_permissions_summary TO promptcraft_app;

-- Grant execute permissions on functions
GRANT EXECUTE ON FUNCTION get_role_permissions(INTEGER) TO promptcraft_app;
GRANT EXECUTE ON FUNCTION user_has_permission(VARCHAR, VARCHAR) TO promptcraft_app;
GRANT EXECUTE ON FUNCTION get_user_roles(VARCHAR) TO promptcraft_app;
GRANT EXECUTE ON FUNCTION assign_user_role(VARCHAR, INTEGER, VARCHAR) TO promptcraft_app;
GRANT EXECUTE ON FUNCTION revoke_user_role(VARCHAR, INTEGER) TO promptcraft_app;
GRANT EXECUTE ON FUNCTION update_role_updated_at() TO promptcraft_app;

-- =============================================================================
-- MIGRATION COMPLETION LOG
-- =============================================================================

DO $$
BEGIN
    RAISE NOTICE 'Migration 002_auth_rbac_schema.sql completed successfully';
    RAISE NOTICE 'AUTH-3 Components Created:';
    RAISE NOTICE '  - Tables: roles, permissions, role_permissions, user_roles';
    RAISE NOTICE '  - Functions: get_role_permissions, user_has_permission, get_user_roles';
    RAISE NOTICE '  - Functions: assign_user_role, revoke_user_role, update_role_updated_at';
    RAISE NOTICE '  - Views: role_hierarchy_view, user_permissions_summary';
    RAISE NOTICE '  - Default roles: super_admin, admin, user_manager, token_manager, api_user, readonly_user';
    RAISE NOTICE '  - Default permissions: 23 permissions covering tokens, users, roles, system, and API access';
    RAISE NOTICE '  - Triggers: automatic updated_at timestamp for roles';
    RAISE NOTICE '  - Indexes: optimized for permission checking and role hierarchy queries';
    RAISE NOTICE 'AUTH-3 Role-Based Access Control system is ready for integration';
    RAISE NOTICE 'Integration with AUTH-2 service tokens: extend ServiceTokenUser.has_permission()';
    RAISE NOTICE 'Integration with AUTH-1 JWT users: use user_has_permission() function';
END $$;
