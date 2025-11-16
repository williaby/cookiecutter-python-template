-- PromptCraft Authentication Schema Migration
-- Migration: 001_auth_schema.sql
-- Description: Complete authentication database schema for both AUTH-1 enhanced Cloudflare Access and AUTH-2 service token management
-- Author: Phase 1 Issue AUTH-1 and AUTH-2 Implementation
-- Date: 2025-08-01

-- Create extension for UUID generation if not exists
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Enable row-level security for enhanced security
SET row_security = on;

-- =============================================================================
-- AUTH-2 SERVICE TOKEN MANAGEMENT TABLES
-- =============================================================================

-- Create service_tokens table for AUTH-2 service token management
CREATE TABLE IF NOT EXISTS service_tokens (
    -- Primary key
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Token identification
    token_name VARCHAR(255) NOT NULL UNIQUE,
    token_hash VARCHAR(255) NOT NULL UNIQUE,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_used TIMESTAMPTZ NULL,
    expires_at TIMESTAMPTZ NULL,

    -- Usage tracking
    usage_count INTEGER NOT NULL DEFAULT 0,

    -- Status
    is_active BOOLEAN NOT NULL DEFAULT TRUE,

    -- Flexible metadata storage
    token_metadata JSONB NULL,

    -- Constraints
    CONSTRAINT valid_expiration_date CHECK (expires_at IS NULL OR expires_at > created_at),
    CONSTRAINT non_negative_usage_count CHECK (usage_count >= 0)
);

-- Add table comment
COMMENT ON TABLE service_tokens IS 'Service tokens for API authentication and authorization';

-- Add column comments
COMMENT ON COLUMN service_tokens.id IS 'Unique identifier for the service token';
COMMENT ON COLUMN service_tokens.token_name IS 'Human-readable name for the token';
COMMENT ON COLUMN service_tokens.token_hash IS 'SHA-256 hash of the actual token value';
COMMENT ON COLUMN service_tokens.created_at IS 'Timestamp when token was created';
COMMENT ON COLUMN service_tokens.last_used IS 'Timestamp when token was last used';
COMMENT ON COLUMN service_tokens.expires_at IS 'Token expiration timestamp (null for non-expiring tokens)';
COMMENT ON COLUMN service_tokens.usage_count IS 'Number of times token has been used';
COMMENT ON COLUMN service_tokens.is_active IS 'Whether the token is currently active';
COMMENT ON COLUMN service_tokens.token_metadata IS 'Additional token metadata (permissions, client info, etc.)';

-- Create indexes for optimal query performance
CREATE INDEX IF NOT EXISTS idx_service_tokens_token_hash ON service_tokens(token_hash);
CREATE INDEX IF NOT EXISTS idx_service_tokens_token_name ON service_tokens(token_name);
CREATE INDEX IF NOT EXISTS idx_service_tokens_active ON service_tokens(is_active);
CREATE INDEX IF NOT EXISTS idx_service_tokens_expires_at ON service_tokens(expires_at);
CREATE INDEX IF NOT EXISTS idx_service_tokens_last_used ON service_tokens(last_used);

-- Composite index for active tokens lookup (most common query pattern)
CREATE INDEX IF NOT EXISTS idx_service_tokens_active_hash ON service_tokens(is_active, token_hash);

-- Index on JSONB metadata for efficient JSON queries
CREATE INDEX IF NOT EXISTS idx_service_tokens_metadata_gin ON service_tokens USING GIN(token_metadata);

-- Create a partial index for only active tokens (performance optimization)
CREATE INDEX IF NOT EXISTS idx_service_tokens_active_only ON service_tokens(token_hash, expires_at)
WHERE is_active = true;

-- =============================================================================
-- AUTH-1 USER SESSION AND EVENT LOGGING TABLES
-- =============================================================================

-- User session management table
-- Tracks authenticated user sessions, preferences, and metadata
CREATE TABLE IF NOT EXISTS user_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) NOT NULL,
    cloudflare_sub VARCHAR(255) NOT NULL,
    first_seen TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_seen TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    session_count INTEGER DEFAULT 1,
    preferences JSONB DEFAULT '{}',
    user_metadata JSONB DEFAULT '{}'
);

-- Create indexes for optimal query performance
CREATE INDEX IF NOT EXISTS idx_user_sessions_email ON user_sessions(email);
CREATE INDEX IF NOT EXISTS idx_user_sessions_cloudflare_sub ON user_sessions(cloudflare_sub);
CREATE INDEX IF NOT EXISTS idx_user_sessions_last_seen ON user_sessions(last_seen DESC);
CREATE UNIQUE INDEX IF NOT EXISTS idx_user_sessions_email_unique ON user_sessions(email);

-- Add table comments for documentation
COMMENT ON TABLE user_sessions IS 'User session tracking for authenticated users';
COMMENT ON COLUMN user_sessions.id IS 'Unique session identifier';
COMMENT ON COLUMN user_sessions.email IS 'User email from JWT token';
COMMENT ON COLUMN user_sessions.cloudflare_sub IS 'Cloudflare subject identifier from JWT';
COMMENT ON COLUMN user_sessions.first_seen IS 'First authentication timestamp';
COMMENT ON COLUMN user_sessions.last_seen IS 'Most recent authentication timestamp';
COMMENT ON COLUMN user_sessions.session_count IS 'Total number of authentication sessions';
COMMENT ON COLUMN user_sessions.preferences IS 'User preferences and settings (JSONB)';
COMMENT ON COLUMN user_sessions.user_metadata IS 'Additional user metadata and tracking data (JSONB)';

-- Authentication event logging table
-- Logs all authentication attempts and events for audit trail
CREATE TABLE IF NOT EXISTS authentication_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_email VARCHAR(255) NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    ip_address INET,
    user_agent TEXT,
    cloudflare_ray_id VARCHAR(100),
    success BOOLEAN DEFAULT TRUE,
    error_details JSONB,
    performance_metrics JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for optimal query performance and audit searches
CREATE INDEX IF NOT EXISTS idx_auth_events_user_email ON authentication_events(user_email);
CREATE INDEX IF NOT EXISTS idx_auth_events_event_type ON authentication_events(event_type);
CREATE INDEX IF NOT EXISTS idx_auth_events_success ON authentication_events(success);
CREATE INDEX IF NOT EXISTS idx_auth_events_created_at ON authentication_events(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_auth_events_cloudflare_ray_id ON authentication_events(cloudflare_ray_id)
    WHERE cloudflare_ray_id IS NOT NULL;

-- Composite index for common query patterns
CREATE INDEX IF NOT EXISTS idx_auth_events_user_type_time ON authentication_events(user_email, event_type, created_at DESC);

-- Add table comments for documentation
COMMENT ON TABLE authentication_events IS 'Authentication event logging for audit trail';
COMMENT ON COLUMN authentication_events.id IS 'Unique event identifier';
COMMENT ON COLUMN authentication_events.user_email IS 'User email from authentication attempt';
COMMENT ON COLUMN authentication_events.event_type IS 'Type of authentication event (login, refresh, validation)';
COMMENT ON COLUMN authentication_events.ip_address IS 'Client IP address';
COMMENT ON COLUMN authentication_events.user_agent IS 'Client user agent string';
COMMENT ON COLUMN authentication_events.cloudflare_ray_id IS 'Cloudflare Ray ID for request tracing';
COMMENT ON COLUMN authentication_events.success IS 'Whether authentication was successful';
COMMENT ON COLUMN authentication_events.error_details IS 'Error details for failed authentication attempts (JSONB)';
COMMENT ON COLUMN authentication_events.performance_metrics IS 'Performance timing metrics for the request (JSONB)';
COMMENT ON COLUMN authentication_events.created_at IS 'Event creation timestamp';

-- Create event type check constraint for data validation
ALTER TABLE authentication_events
ADD CONSTRAINT chk_event_type
CHECK (event_type IN ('login', 'refresh', 'validation', 'logout', 'token_expired', 'token_invalid'));

-- =============================================================================
-- AUTH-2 SERVICE TOKEN FUNCTIONS AND TRIGGERS
-- =============================================================================

-- Create function to automatically update last_used timestamp
CREATE OR REPLACE FUNCTION update_service_token_last_used()
RETURNS TRIGGER AS $$
BEGIN
    -- Only update if usage_count is being incremented
    IF NEW.usage_count > OLD.usage_count THEN
        NEW.last_used = NOW();
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger to automatically update last_used when usage_count changes
CREATE TRIGGER trigger_update_service_token_last_used
    BEFORE UPDATE ON service_tokens
    FOR EACH ROW
    EXECUTE FUNCTION update_service_token_last_used();

-- Create function to check token expiration
CREATE OR REPLACE FUNCTION is_service_token_expired(token_expires_at TIMESTAMPTZ)
RETURNS BOOLEAN AS $$
BEGIN
    IF token_expires_at IS NULL THEN
        RETURN FALSE;
    END IF;
    RETURN NOW() > token_expires_at;
END;
$$ LANGUAGE plpgsql;

-- Create function to check token validity (active and not expired)
CREATE OR REPLACE FUNCTION is_service_token_valid(token_is_active BOOLEAN, token_expires_at TIMESTAMPTZ)
RETURNS BOOLEAN AS $$
BEGIN
    RETURN token_is_active AND NOT is_service_token_expired(token_expires_at);
END;
$$ LANGUAGE plpgsql;

-- Create view for valid tokens only (frequently used in queries)
CREATE OR REPLACE VIEW valid_service_tokens AS
SELECT
    id,
    token_name,
    token_hash,
    created_at,
    last_used,
    expires_at,
    usage_count,
    is_active,
    token_metadata,
    is_service_token_expired(expires_at) as is_expired,
    is_service_token_valid(is_active, expires_at) as is_valid
FROM service_tokens
WHERE is_service_token_valid(is_active, expires_at);

-- Add view comment
COMMENT ON VIEW valid_service_tokens IS 'View of all valid (active and non-expired) service tokens';

-- Create stored procedure for token validation with usage tracking
CREATE OR REPLACE FUNCTION validate_service_token(
    p_token_hash VARCHAR(255),
    p_increment_usage BOOLEAN DEFAULT TRUE
)
RETURNS TABLE(
    token_id UUID,
    token_name VARCHAR(255),
    expires_at TIMESTAMPTZ,
    usage_count INTEGER,
    token_metadata JSONB,
    is_valid BOOLEAN
) AS $$
DECLARE
    token_record RECORD;
BEGIN
    -- Find and validate token
    SELECT
        st.id,
        st.token_name,
        st.expires_at,
        st.usage_count,
        st.token_metadata,
        is_service_token_valid(st.is_active, st.expires_at) as valid
    INTO token_record
    FROM service_tokens st
    WHERE st.token_hash = p_token_hash;

    -- Return empty result if token not found
    IF NOT FOUND THEN
        RETURN;
    END IF;

    -- Increment usage count if requested and token is valid
    IF p_increment_usage AND token_record.valid THEN
        UPDATE service_tokens
        SET usage_count = usage_count + 1
        WHERE token_hash = p_token_hash;

        -- Update the usage count in our return record
        token_record.usage_count := token_record.usage_count + 1;
    END IF;

    -- Return token information
    token_id := token_record.id;
    token_name := token_record.token_name;
    expires_at := token_record.expires_at;
    usage_count := token_record.usage_count;
    token_metadata := token_record.token_metadata;
    is_valid := token_record.valid;

    RETURN NEXT;
END;
$$ LANGUAGE plpgsql;

-- Add stored procedure comment
COMMENT ON FUNCTION validate_service_token(VARCHAR, BOOLEAN) IS 'Validate service token and optionally increment usage count';

-- Create cleanup procedure for expired tokens
CREATE OR REPLACE FUNCTION cleanup_expired_tokens(
    p_delete_expired BOOLEAN DEFAULT FALSE,
    p_deactivate_expired BOOLEAN DEFAULT TRUE
)
RETURNS TABLE(
    cleaned_count INTEGER,
    action_taken VARCHAR(50)
) AS $$
DECLARE
    expired_count INTEGER;
BEGIN
    -- Count expired tokens
    SELECT COUNT(*) INTO expired_count
    FROM service_tokens
    WHERE is_service_token_expired(expires_at) AND is_active = true;

    IF p_delete_expired THEN
        -- Delete expired tokens
        DELETE FROM service_tokens
        WHERE is_service_token_expired(expires_at);

        cleaned_count := expired_count;
        action_taken := 'deleted';
    ELSIF p_deactivate_expired THEN
        -- Deactivate expired tokens
        UPDATE service_tokens
        SET is_active = false
        WHERE is_service_token_expired(expires_at) AND is_active = true;

        cleaned_count := expired_count;
        action_taken := 'deactivated';
    ELSE
        cleaned_count := expired_count;
        action_taken := 'counted_only';
    END IF;

    RETURN NEXT;
END;
$$ LANGUAGE plpgsql;

-- Add cleanup procedure comment
COMMENT ON FUNCTION cleanup_expired_tokens(BOOLEAN, BOOLEAN) IS 'Clean up expired service tokens by deletion or deactivation';

-- =============================================================================
-- AUTH-1 USER SESSION AND EVENT LOGGING FUNCTIONS
-- =============================================================================

-- Create function to automatically update last_seen timestamp
CREATE OR REPLACE FUNCTION update_user_session_last_seen()
RETURNS TRIGGER AS $$
BEGIN
    NEW.last_seen = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger to update last_seen on session_count changes
CREATE TRIGGER trigger_update_last_seen
    BEFORE UPDATE OF session_count ON user_sessions
    FOR EACH ROW
    EXECUTE FUNCTION update_user_session_last_seen();

-- Create function for session count increment with upsert
CREATE OR REPLACE FUNCTION upsert_user_session(
    p_email VARCHAR(255),
    p_cloudflare_sub VARCHAR(255),
    p_preferences JSONB DEFAULT '{}',
    p_user_metadata JSONB DEFAULT '{}'
) RETURNS user_sessions AS $$
DECLARE
    result user_sessions;
BEGIN
    -- Try to update existing session
    UPDATE user_sessions
    SET session_count = session_count + 1,
        last_seen = NOW(),
        preferences = COALESCE(p_preferences, preferences),
        user_metadata = COALESCE(p_user_metadata, user_metadata)
    WHERE email = p_email
    RETURNING * INTO result;

    -- If no existing session, insert new one
    IF NOT FOUND THEN
        INSERT INTO user_sessions (email, cloudflare_sub, session_count, preferences, user_metadata)
        VALUES (p_email, p_cloudflare_sub, 1, p_preferences, p_user_metadata)
        RETURNING * INTO result;
    END IF;

    RETURN result;
END;
$$ LANGUAGE plpgsql;

-- Create function to log authentication events with performance metrics
CREATE OR REPLACE FUNCTION log_auth_event(
    p_user_email VARCHAR(255),
    p_event_type VARCHAR(50),
    p_ip_address INET DEFAULT NULL,
    p_user_agent TEXT DEFAULT NULL,
    p_cloudflare_ray_id VARCHAR(100) DEFAULT NULL,
    p_success BOOLEAN DEFAULT TRUE,
    p_error_details JSONB DEFAULT NULL,
    p_performance_metrics JSONB DEFAULT NULL
) RETURNS UUID AS $$
DECLARE
    event_id UUID;
BEGIN
    INSERT INTO authentication_events (
        user_email, event_type, ip_address, user_agent,
        cloudflare_ray_id, success, error_details, performance_metrics
    )
    VALUES (
        p_user_email, p_event_type, p_ip_address, p_user_agent,
        p_cloudflare_ray_id, p_success, p_error_details, p_performance_metrics
    )
    RETURNING id INTO event_id;

    RETURN event_id;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- ADDITIONAL INDEXES FOR PERFORMANCE
-- =============================================================================

-- Create indexes on JSONB columns for better query performance
CREATE INDEX IF NOT EXISTS idx_user_sessions_preferences_gin ON user_sessions USING GIN (preferences);
CREATE INDEX IF NOT EXISTS idx_user_sessions_metadata_gin ON user_sessions USING GIN (user_metadata);
CREATE INDEX IF NOT EXISTS idx_auth_events_error_details_gin ON authentication_events USING GIN (error_details);
CREATE INDEX IF NOT EXISTS idx_auth_events_performance_gin ON authentication_events USING GIN (performance_metrics);

-- =============================================================================
-- ANALYTICS AND REPORTING VIEWS
-- =============================================================================

-- Create view for authentication analytics
CREATE OR REPLACE VIEW auth_analytics AS
SELECT
    DATE_TRUNC('hour', created_at) as hour_bucket,
    event_type,
    success,
    COUNT(*) as event_count,
    COUNT(DISTINCT user_email) as unique_users,
    AVG(CASE
        WHEN performance_metrics->>'total_time_ms' IS NOT NULL
        THEN (performance_metrics->>'total_time_ms')::numeric
        ELSE NULL
    END) as avg_response_time_ms
FROM authentication_events
WHERE created_at >= NOW() - INTERVAL '7 days'
GROUP BY DATE_TRUNC('hour', created_at), event_type, success
ORDER BY hour_bucket DESC;

-- Create view for user session summary
CREATE OR REPLACE VIEW user_session_summary AS
SELECT
    email,
    session_count,
    first_seen,
    last_seen,
    EXTRACT(EPOCH FROM (last_seen - first_seen)) / 3600 as total_hours,
    (user_metadata->>'total_requests')::integer as total_requests,
    preferences->>'theme' as theme_preference
FROM user_sessions
ORDER BY last_seen DESC;

-- =============================================================================
-- PERMISSIONS AND SECURITY
-- =============================================================================

-- Grant appropriate permissions (adjust based on your security requirements)
GRANT SELECT, INSERT, UPDATE ON user_sessions TO promptcraft_app;
GRANT SELECT, INSERT ON authentication_events TO promptcraft_app;
GRANT SELECT, INSERT, UPDATE ON service_tokens TO promptcraft_app;
GRANT SELECT ON valid_service_tokens TO promptcraft_app;
GRANT EXECUTE ON FUNCTION validate_service_token(VARCHAR, BOOLEAN) TO promptcraft_app;
GRANT EXECUTE ON FUNCTION upsert_user_session(VARCHAR, VARCHAR, JSONB, JSONB) TO promptcraft_app;
GRANT EXECUTE ON FUNCTION log_auth_event(VARCHAR, VARCHAR, INET, TEXT, VARCHAR, BOOLEAN, JSONB, JSONB) TO promptcraft_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO promptcraft_app;

-- Add final comment for migration tracking
COMMENT ON SCHEMA public IS 'PromptCraft Authentication Schema - Migration 001 - Complete AUTH-1 and AUTH-2 Setup';

-- =============================================================================
-- MIGRATION COMPLETION LOG
-- =============================================================================

DO $$
BEGIN
    RAISE NOTICE 'Migration 001_auth_schema.sql completed successfully';
    RAISE NOTICE 'AUTH-1 Components:';
    RAISE NOTICE '  - Created tables: user_sessions, authentication_events';
    RAISE NOTICE '  - Created functions: upsert_user_session, log_auth_event, update_user_session_last_seen';
    RAISE NOTICE '  - Created views: auth_analytics, user_session_summary';
    RAISE NOTICE 'AUTH-2 Components:';
    RAISE NOTICE '  - Created table: service_tokens';
    RAISE NOTICE '  - Created functions: validate_service_token, cleanup_expired_tokens, is_service_token_valid';
    RAISE NOTICE '  - Created view: valid_service_tokens';
    RAISE NOTICE '  - Created triggers for automatic timestamp updates';
    RAISE NOTICE 'Created comprehensive indexes for optimal query performance';
    RAISE NOTICE 'Both AUTH-1 and AUTH-2 systems are ready for production use';
END $$;
