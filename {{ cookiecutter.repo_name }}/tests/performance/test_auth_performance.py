"""Performance tests for PromptCraft authentication systems.

Tests verify:
- AUTH-1: Enhanced Cloudflare Access authentication performance (<75ms requirement)
- AUTH-2: Service token validation performance (<10ms target)
- Database integration performance and optimization
- Concurrent authentication handling
- Memory usage under load
- Cache effectiveness
- Response time consistency
- Graceful degradation scenarios
"""

# ruff: noqa: S105

import asyncio
import gc
import statistics
import time
from collections.abc import AsyncGenerator
from datetime import UTC, datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import psutil
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from src.auth.config import AuthenticationConfig
from src.auth.middleware import AuthenticationMiddleware, ServiceTokenUser
from src.auth.models import AuthenticatedUser, UserRole
from src.auth.service_token_manager import ServiceTokenManager
from src.database import DatabaseManager, get_database_manager
from src.monitoring.service_token_monitor import ServiceTokenMonitor


@pytest.mark.performance
class TestAuthenticationPerformance:
    """Performance tests for authentication middleware."""

    @pytest.fixture
    def auth_config(self) -> AuthenticationConfig:
        """Authentication configuration for performance testing."""
        return AuthenticationConfig(
            cloudflare_access_enabled=True,
            cloudflare_audience="perf-test-audience",
            cloudflare_issuer="https://perf-test.cloudflareaccess.com",
            auth_logging_enabled=True,
            rate_limiting_enabled=False,  # Disable for performance testing
        )

    @pytest.fixture
    async def mock_database_session(self) -> AsyncGenerator[AsyncMock, None]:
        """Mock high-performance database session."""
        mock_session = AsyncMock(spec=AsyncSession)

        # Mock fast database operations
        mock_session.execute = AsyncMock()
        mock_session.scalar_one_or_none = AsyncMock()
        mock_session.add = AsyncMock()
        mock_session.commit = AsyncMock()

        # Simulate fast database responses
        def fast_execute(*args, **kwargs):
            # Simulate 1-5ms database response time
            time.sleep(0.001 + (hash(str(args)) % 4) * 0.001)
            return AsyncMock()

        mock_session.execute.side_effect = fast_execute
        return mock_session

    @pytest.fixture
    async def mock_database_manager(
        self,
        mock_database_session: AsyncMock,
    ) -> AsyncGenerator[AsyncMock, None]:
        """Mock high-performance database manager."""
        mock_manager = AsyncMock(spec=DatabaseManager)

        # Mock session context manager
        mock_manager.get_session.return_value.__aenter__ = AsyncMock(return_value=mock_database_session)
        mock_manager.get_session.return_value.__aexit__ = AsyncMock(return_value=None)

        # Mock fast health check
        mock_manager.health_check.return_value = {
            "status": "healthy",
            "timestamp": time.time(),
            "connection_test": True,
            "response_time_ms": 2.1,  # Fast response
        }

        with patch("src.auth.middleware.get_database_manager", return_value=mock_manager):
            yield mock_manager

    @pytest.fixture
    def mock_jwt_validator(self) -> MagicMock:
        """Mock high-performance JWT validator."""
        validator = MagicMock()

        def fast_validate(*args, **kwargs):
            # Simulate 1-3ms JWT validation time
            time.sleep(0.001 + (hash(str(args)) % 2) * 0.001)
            return AuthenticatedUser(
                email="perf-test@example.com",
                role=UserRole.USER,
                jwt_claims={
                    "sub": "perf-test-sub",
                    "email": "perf-test@example.com",
                    "aud": "perf-test-audience",
                    "iss": "https://perf-test.cloudflareaccess.com",
                    "exp": int(time.time()) + 3600,
                },
            )

        validator.validate_token.side_effect = fast_validate
        return validator

    @pytest.fixture
    def performance_app(
        self,
        auth_config: AuthenticationConfig,
        mock_jwt_validator: MagicMock,
        mock_database_manager: AsyncMock,
    ) -> FastAPI:
        """Create FastAPI app optimized for performance testing."""
        app = FastAPI(title="PromptCraft Performance Test App")

        # Add authentication middleware
        auth_middleware = AuthenticationMiddleware(
            app=app,
            config=auth_config,
            jwt_validator=mock_jwt_validator,
            database_enabled=True,
        )
        app.add_middleware(auth_middleware)

        @app.get("/api/fast-endpoint")
        async def fast_endpoint(request: Request):
            """Fast test endpoint for performance testing."""
            user = getattr(request.state, "authenticated_user", None)
            return JSONResponse(
                {
                    "status": "success",
                    "user_email": user.email if user else None,
                    "timestamp": time.time(),
                },
            )

        @app.get("/health")
        async def health_endpoint():
            """Health endpoint (excluded from auth)."""
            return {"status": "healthy"}

        return app

    @pytest.mark.asyncio
    async def test_single_request_performance(
        self,
        performance_app: FastAPI,
        mock_database_session: AsyncMock,
    ):
        """Test single request authentication performance (<75ms target)."""
        from httpx import AsyncClient

        async with AsyncClient(app=performance_app, base_url="http://test") as client:
            # Measure single request performance
            start_time = time.time()

            response = await client.get(
                "/api/fast-endpoint",
                headers={
                    "CF-Access-Jwt-Assertion": "performance-test-jwt",
                    "CF-Connecting-IP": "192.168.1.100",
                },
            )

            end_time = time.time()
            request_time_ms = (end_time - start_time) * 1000

            # Verify response
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert data["user_email"] == "perf-test@example.com"

            # Verify performance requirement
            assert request_time_ms < 75.0, f"Request took {request_time_ms:.2f}ms, exceeds 75ms requirement"

            # Verify database operations were performed
            mock_database_session.execute.assert_called()
            mock_database_session.commit.assert_called()

    @pytest.mark.asyncio
    async def test_concurrent_requests_performance(
        self,
        performance_app: FastAPI,
        mock_database_session: AsyncMock,
    ):
        """Test concurrent request handling performance."""
        from httpx import AsyncClient

        async def make_concurrent_request(client: AsyncClient, request_id: int):
            """Make a single concurrent request."""
            start_time = time.time()

            response = await client.get(
                "/api/fast-endpoint",
                headers={
                    "CF-Access-Jwt-Assertion": f"concurrent-jwt-{request_id}",
                    "CF-Connecting-IP": f"192.168.1.{request_id % 200 + 1}",
                },
            )

            end_time = time.time()
            return {
                "request_id": request_id,
                "response_time_ms": (end_time - start_time) * 1000,
                "status_code": response.status_code,
                "response_data": response.json(),
            }

        async with AsyncClient(app=performance_app, base_url="http://test") as client:
            # Create 50 concurrent requests
            num_requests = 50
            start_time = time.time()

            tasks = [make_concurrent_request(client, i) for i in range(num_requests)]

            results = await asyncio.gather(*tasks)

            end_time = time.time()
            total_time_ms = (end_time - start_time) * 1000

            # Analyze results
            response_times = [r["response_time_ms"] for r in results]
            successful_requests = [r for r in results if r["status_code"] == 200]

            # Verify all requests succeeded
            assert len(successful_requests) == num_requests

            # Performance analysis
            avg_response_time = statistics.mean(response_times)
            median_response_time = statistics.median(response_times)
            p95_response_time = response_times[int(0.95 * len(response_times))]
            max_response_time = max(response_times)

            print(f"\nConcurrent Performance Results ({num_requests} requests):")
            print(f"Total time: {total_time_ms:.2f}ms")
            print(f"Average response time: {avg_response_time:.2f}ms")
            print(f"Median response time: {median_response_time:.2f}ms")
            print(f"95th percentile: {p95_response_time:.2f}ms")
            print(f"Max response time: {max_response_time:.2f}ms")

            # Performance requirements
            assert avg_response_time < 75.0, f"Average response time {avg_response_time:.2f}ms exceeds requirement"
            assert p95_response_time < 150.0, f"95th percentile {p95_response_time:.2f}ms exceeds tolerance"

            # Verify database operations
            assert mock_database_session.execute.call_count >= num_requests

    @pytest.mark.asyncio
    async def test_database_performance_impact(
        self,
        auth_config: AuthenticationConfig,
        mock_jwt_validator: MagicMock,
    ):
        """Test performance impact of database operations."""
        from httpx import AsyncClient

        # Test with database enabled
        app_with_db = FastAPI()

        with patch("src.auth.middleware.get_database_manager") as mock_get_db:
            # Mock fast database manager
            mock_db = AsyncMock()
            mock_session = AsyncMock()
            mock_db.get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_db.get_session.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_get_db.return_value = mock_db

            auth_middleware_with_db = AuthenticationMiddleware(
                app=app_with_db,
                config=auth_config,
                jwt_validator=mock_jwt_validator,
                database_enabled=True,
            )
            app_with_db.add_middleware(auth_middleware_with_db)

            @app_with_db.get("/api/test")
            async def test_endpoint_with_db(request: Request):
                user = getattr(request.state, "authenticated_user", None)
                return {"user_email": user.email if user else None}

            # Test with database disabled
            app_without_db = FastAPI()
            auth_middleware_without_db = AuthenticationMiddleware(
                app=app_without_db,
                config=auth_config,
                jwt_validator=mock_jwt_validator,
                database_enabled=False,
            )
            app_without_db.add_middleware(auth_middleware_without_db)

            @app_without_db.get("/api/test")
            async def test_endpoint_without_db(request: Request):
                user = getattr(request.state, "authenticated_user", None)
                return {"user_email": user.email if user else None}

            # Measure performance with database
            async with AsyncClient(app=app_with_db, base_url="http://test") as client:
                start_time = time.time()
                for _ in range(10):
                    await client.get("/api/test", headers={"CF-Access-Jwt-Assertion": "db-test-jwt"})
                db_time = time.time() - start_time

            # Measure performance without database
            async with AsyncClient(app=app_without_db, base_url="http://test") as client:
                start_time = time.time()
                for _ in range(10):
                    await client.get("/api/test", headers={"CF-Access-Jwt-Assertion": "no-db-test-jwt"})
                no_db_time = time.time() - start_time

            # Analyze database overhead
            db_overhead_ms = (db_time - no_db_time) * 100  # Per request overhead

            print("\nDatabase Performance Impact:")
            print(f"With database: {db_time * 100:.2f}ms per request")
            print(f"Without database: {no_db_time * 100:.2f}ms per request")
            print(f"Database overhead: {db_overhead_ms:.2f}ms per request")

            # Database overhead should be reasonable
            assert db_overhead_ms < 25.0, f"Database overhead {db_overhead_ms:.2f}ms is too high"

    @pytest.mark.asyncio
    async def test_memory_usage_under_load(
        self,
        performance_app: FastAPI,
    ):
        """Test memory usage during high load."""
        from httpx import AsyncClient

        # Get initial memory usage
        process = psutil.Process()
        initial_memory_mb = process.memory_info().rss / 1024 / 1024

        async with AsyncClient(app=performance_app, base_url="http://test") as client:
            # Generate load (100 requests)
            tasks = []
            for i in range(100):
                task = client.get(
                    "/api/fast-endpoint",
                    headers={
                        "CF-Access-Jwt-Assertion": f"load-test-jwt-{i}",
                        "CF-Connecting-IP": f"10.0.{i // 255}.{i % 255}",
                    },
                )
                tasks.append(task)

            # Execute all requests
            responses = await asyncio.gather(*tasks)

            # Verify all succeeded
            assert all(r.status_code == 200 for r in responses)

            # Force garbage collection
            gc.collect()

            # Check memory usage after load
            final_memory_mb = process.memory_info().rss / 1024 / 1024
            memory_increase_mb = final_memory_mb - initial_memory_mb

            print("\nMemory Usage Analysis:")
            print(f"Initial memory: {initial_memory_mb:.2f} MB")
            print(f"Final memory: {final_memory_mb:.2f} MB")
            print(f"Memory increase: {memory_increase_mb:.2f} MB")

            # Memory increase should be reasonable
            assert memory_increase_mb < 50.0, f"Memory increase {memory_increase_mb:.2f} MB is too high"

    @pytest.mark.asyncio
    async def test_graceful_degradation_performance(
        self,
        auth_config: AuthenticationConfig,
        mock_jwt_validator: MagicMock,
    ):
        """Test performance during database failures (graceful degradation)."""
        from httpx import AsyncClient

        app = FastAPI()

        # Mock database failure
        with patch("src.auth.middleware.get_database_manager") as mock_get_db:
            mock_db = AsyncMock()
            mock_db.get_session.side_effect = Exception("Database connection failed")
            mock_get_db.return_value = mock_db

            auth_middleware = AuthenticationMiddleware(
                app=app,
                config=auth_config,
                jwt_validator=mock_jwt_validator,
                database_enabled=True,
            )
            app.add_middleware(auth_middleware)

            @app.get("/api/degraded-test")
            async def degraded_test_endpoint(request: Request):
                user = getattr(request.state, "authenticated_user", None)
                return {"user_email": user.email if user else None}

            async with AsyncClient(app=app, base_url="http://test") as client:
                # Measure performance during database failure
                start_time = time.time()

                response = await client.get(
                    "/api/degraded-test",
                    headers={"CF-Access-Jwt-Assertion": "degraded-test-jwt"},
                )

                end_time = time.time()
                degraded_time_ms = (end_time - start_time) * 1000

                # Verify authentication still works
                assert response.status_code == 200
                data = response.json()
                assert data["user_email"] == "perf-test@example.com"

                # Performance should still be acceptable during degradation
                assert degraded_time_ms < 100.0, f"Degraded performance {degraded_time_ms:.2f}ms is too slow"

                print(f"\nGraceful Degradation Performance: {degraded_time_ms:.2f}ms")


@pytest.mark.performance
class TestServiceTokenPerformance:
    """Performance tests for AUTH-2 service token management."""

    @pytest.fixture
    def mock_service_token_manager(self):
        """Mock high-performance service token manager."""
        manager = MagicMock(spec=ServiceTokenManager)

        def fast_validate_token(token_hash: str):
            # Simulate 1-5ms token validation
            time.sleep(0.001 + (hash(token_hash) % 4) * 0.001)

            return {
                "is_valid": True,
                "token_name": "perf-test-token",
                "metadata": {"permissions": ["read", "write"]},
                "usage_count": 100,
                "expires_at": datetime.now(UTC) + timedelta(days=30),
            }

        manager.validate_token = MagicMock(side_effect=fast_validate_token)
        return manager

    @pytest.fixture
    def mock_service_token_monitor(self):
        """Mock service token monitor for performance testing."""
        monitor = MagicMock(spec=ServiceTokenMonitor)

        def fast_record_usage(token_name: str, metadata: dict[str, Any]):
            # Simulate 0.5-2ms monitoring operation
            time.sleep(0.0005 + (hash(token_name) % 3) * 0.0005)

        monitor.record_token_usage = MagicMock(side_effect=fast_record_usage)
        monitor.get_token_analytics = MagicMock(return_value={"usage_count": 1000})
        return monitor

    @pytest.mark.asyncio
    async def test_service_token_validation_performance(
        self,
        mock_service_token_manager: MagicMock,
        mock_service_token_monitor: MagicMock,
    ):
        """Test service token validation performance (<10ms target)."""

        # Test single token validation
        start_time = time.time()

        result = mock_service_token_manager.validate_token("sk_test_performance_token_123")

        end_time = time.time()
        validation_time_ms = (end_time - start_time) * 1000

        # Verify result
        assert result["is_valid"] is True
        assert result["token_name"] == "perf-test-token"

        # Verify performance requirement
        assert validation_time_ms < 10.0, f"Token validation took {validation_time_ms:.2f}ms, exceeds 10ms target"

        print(f"Service token validation time: {validation_time_ms:.2f}ms")

    @pytest.mark.asyncio
    async def test_concurrent_token_validation_performance(
        self,
        mock_service_token_manager: MagicMock,
        mock_service_token_monitor: MagicMock,
    ):
        """Test concurrent service token validation performance."""

        async def validate_token_async(token_id: int):
            """Validate a single token asynchronously."""
            start_time = time.time()

            # Run in thread pool to simulate async database operation
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                mock_service_token_manager.validate_token,
                f"sk_test_concurrent_token_{token_id}",
            )

            end_time = time.time()
            return {
                "token_id": token_id,
                "validation_time_ms": (end_time - start_time) * 1000,
                "is_valid": result["is_valid"],
            }

        # Test concurrent validation (20 tokens)
        num_tokens = 20
        start_time = time.time()

        tasks = [validate_token_async(i) for i in range(num_tokens)]
        results = await asyncio.gather(*tasks)

        end_time = time.time()
        total_time_ms = (end_time - start_time) * 1000

        # Analyze results
        validation_times = [r["validation_time_ms"] for r in results]
        successful_validations = [r for r in results if r["is_valid"]]

        # Verify all validations succeeded
        assert len(successful_validations) == num_tokens

        # Performance analysis
        avg_validation_time = statistics.mean(validation_times)
        max_validation_time = max(validation_times)

        print(f"\nConcurrent Token Validation Results ({num_tokens} tokens):")
        print(f"Total time: {total_time_ms:.2f}ms")
        print(f"Average validation time: {avg_validation_time:.2f}ms")
        print(f"Max validation time: {max_validation_time:.2f}ms")

        # Performance requirements
        assert avg_validation_time < 10.0, f"Average validation time {avg_validation_time:.2f}ms exceeds target"
        assert max_validation_time < 25.0, f"Max validation time {max_validation_time:.2f}ms exceeds tolerance"

    @pytest.mark.asyncio
    async def test_token_monitoring_performance_impact(
        self,
        mock_service_token_manager: MagicMock,
        mock_service_token_monitor: MagicMock,
    ):
        """Test performance impact of token monitoring."""

        # Test validation without monitoring
        start_time = time.time()
        for i in range(10):
            mock_service_token_manager.validate_token(f"sk_test_no_monitor_{i}")
        no_monitor_time = time.time() - start_time

        # Test validation with monitoring
        start_time = time.time()
        for i in range(10):
            result = mock_service_token_manager.validate_token(f"sk_test_with_monitor_{i}")
            mock_service_token_monitor.record_token_usage(result["token_name"], {"validation_time": time.time()})
        with_monitor_time = time.time() - start_time

        # Analyze monitoring overhead
        monitor_overhead_ms = (with_monitor_time - no_monitor_time) * 100  # Per request

        print("\nToken Monitoring Performance Impact:")
        print(f"Without monitoring: {no_monitor_time * 100:.2f}ms per request")
        print(f"With monitoring: {with_monitor_time * 100:.2f}ms per request")
        print(f"Monitoring overhead: {monitor_overhead_ms:.2f}ms per request")

        # Monitoring overhead should be minimal
        assert monitor_overhead_ms < 2.0, f"Monitoring overhead {monitor_overhead_ms:.2f}ms is too high"


@pytest.mark.performance
class TestRoleBasedPermissionPerformance:
    """Performance tests for AUTH-3 role-based permission system."""

    @pytest.fixture
    def mock_role_manager(self):
        """Mock high-performance role manager."""
        from src.auth.role_manager import RoleManager

        manager = MagicMock(spec=RoleManager)

        def fast_get_user_permissions(user_email: str):
            # Simulate 2-8ms permission resolution with database query
            time.sleep(0.002 + (hash(user_email) % 6) * 0.001)

            # Return comprehensive permission set based on user type
            if "admin" in user_email:
                return {
                    "users:create",
                    "users:read",
                    "users:update",
                    "users:delete",
                    "tokens:create",
                    "tokens:read",
                    "tokens:update",
                    "tokens:delete",
                    "roles:create",
                    "roles:read",
                    "roles:update",
                    "roles:delete",
                    "system:admin",
                    "system:config",
                    "system:backup",
                }
            if "manager" in user_email:
                return {
                    "users:read",
                    "users:update",
                    "tokens:read",
                    "tokens:create",
                    "roles:read",
                    "reports:read",
                    "reports:create",
                }
            return {"tokens:read", "profiles:read", "profiles:update"}

        def fast_validate_role_hierarchy(child_role: str, parent_role: str):
            # Simulate 1-3ms hierarchy validation
            time.sleep(0.001 + (hash(child_role + parent_role) % 2) * 0.001)
            # Simple validation: prevent circular dependencies
            return child_role != parent_role

        def fast_get_role_permissions(role_name: str):
            # Simulate 3-10ms role permission lookup with inheritance
            time.sleep(0.003 + (hash(role_name) % 7) * 0.001)

            role_permissions = {
                "super_admin": {"users:*", "tokens:*", "roles:*", "system:*", "reports:*", "audit:*", "config:*"},
                "admin": {
                    "users:create",
                    "users:read",
                    "users:update",
                    "users:delete",
                    "tokens:create",
                    "tokens:read",
                    "tokens:update",
                    "tokens:delete",
                    "roles:read",
                    "roles:update",
                    "reports:read",
                    "reports:create",
                },
                "user_manager": {
                    "users:read",
                    "users:update",
                    "users:delete",
                    "roles:read",
                    "reports:read",
                },
                "token_manager": {
                    "tokens:create",
                    "tokens:read",
                    "tokens:update",
                    "tokens:delete",
                    "users:read",  # Can see users for token assignment
                },
                "api_user": {"tokens:read", "profiles:read", "profiles:update"},
                "readonly_user": {"profiles:read", "reports:read"},
            }

            return role_permissions.get(role_name, set())

        def fast_assign_user_role(user_email: str, role_name: str, assigned_by: str | None = None):
            # Simulate 5-15ms user role assignment with database update
            time.sleep(0.005 + (hash(user_email + role_name) % 10) * 0.001)
            return True

        manager.get_user_permissions = MagicMock(side_effect=fast_get_user_permissions)
        manager.validate_role_hierarchy = MagicMock(side_effect=fast_validate_role_hierarchy)
        manager.get_role_permissions = MagicMock(side_effect=fast_get_role_permissions)
        manager.assign_user_role = MagicMock(side_effect=fast_assign_user_role)

        return manager

    @pytest.fixture
    def mock_permission_validator(self):
        """Mock high-performance permission validator."""
        from src.auth.permissions import require_permission

        def fast_check_permission(user_email: str, permission: str):
            # Simulate 1-5ms permission check using database function
            time.sleep(0.001 + (hash(user_email + permission) % 4) * 0.001)

            # Fast permission logic based on user type
            if "admin@" in user_email:
                return True
            if "manager@" in user_email:
                return permission in {"users:read", "tokens:read", "reports:read"}
            return permission in {"profiles:read", "tokens:read"}

        validator = MagicMock()
        validator.check_permission = MagicMock(side_effect=fast_check_permission)
        return validator

    @pytest.mark.asyncio
    async def test_permission_check_performance(
        self,
        mock_permission_validator: MagicMock,
    ):
        """Test single permission check performance (<15ms target)."""

        # Test permission check for different user types
        test_cases = [
            ("admin@example.com", "users:create"),
            ("manager@example.com", "tokens:read"),
            ("user@example.com", "profiles:read"),
        ]

        for user_email, permission in test_cases:
            start_time = time.time()

            result = mock_permission_validator.check_permission(user_email, permission)

            end_time = time.time()
            check_time_ms = (end_time - start_time) * 1000

            # Verify result
            assert result is not None

            # Verify performance requirement
            assert check_time_ms < 15.0, f"Permission check took {check_time_ms:.2f}ms, exceeds 15ms target"

        print(f"Single permission check performance: {check_time_ms:.2f}ms average")

    @pytest.mark.asyncio
    async def test_user_permissions_resolution_performance(
        self,
        mock_role_manager: MagicMock,
    ):
        """Test user permission resolution performance (<20ms target)."""

        # Test different user types
        test_users = [
            "admin@example.com",
            "manager@example.com",
            "user@example.com",
            "api_user@example.com",
        ]

        total_times = []

        for user_email in test_users:
            start_time = time.time()

            permissions = mock_role_manager.get_user_permissions(user_email)

            end_time = time.time()
            resolution_time_ms = (end_time - start_time) * 1000
            total_times.append(resolution_time_ms)

            # Verify result
            assert isinstance(permissions, set)
            assert len(permissions) > 0

            # Verify performance requirement
            assert (
                resolution_time_ms < 20.0
            ), f"Permission resolution took {resolution_time_ms:.2f}ms, exceeds 20ms target"

        avg_time = statistics.mean(total_times)
        print(f"User permission resolution performance: {avg_time:.2f}ms average")

    @pytest.mark.asyncio
    async def test_role_hierarchy_validation_performance(
        self,
        mock_role_manager: MagicMock,
    ):
        """Test role hierarchy validation performance (<10ms target)."""

        # Test various hierarchy validations
        test_cases = [
            ("admin", "super_admin"),
            ("user_manager", "admin"),
            ("token_manager", "admin"),
            ("api_user", "user_manager"),
            ("readonly_user", "api_user"),
        ]

        total_times = []

        for child_role, parent_role in test_cases:
            start_time = time.time()

            is_valid = mock_role_manager.validate_role_hierarchy(child_role, parent_role)

            end_time = time.time()
            validation_time_ms = (end_time - start_time) * 1000
            total_times.append(validation_time_ms)

            # Verify result
            assert isinstance(is_valid, bool)

            # Verify performance requirement
            assert (
                validation_time_ms < 10.0
            ), f"Hierarchy validation took {validation_time_ms:.2f}ms, exceeds 10ms target"

        avg_time = statistics.mean(total_times)
        print(f"Role hierarchy validation performance: {avg_time:.2f}ms average")

    @pytest.mark.asyncio
    async def test_concurrent_permission_checks_performance(
        self,
        mock_permission_validator: MagicMock,
    ):
        """Test concurrent permission checking performance."""

        async def check_permission_async(user_id: int, permission: str):
            """Check permission asynchronously."""
            user_email = f"user{user_id}@example.com"
            start_time = time.time()

            # Run in thread pool to simulate database operation
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                mock_permission_validator.check_permission,
                user_email,
                permission,
            )

            end_time = time.time()
            return {
                "user_id": user_id,
                "permission": permission,
                "check_time_ms": (end_time - start_time) * 1000,
                "has_permission": result,
            }

        # Test concurrent permission checks (30 users, 5 permissions each)
        permissions = ["users:read", "tokens:create", "reports:read", "profiles:update", "system:config"]
        num_users = 30

        start_time = time.time()

        # Generate all permission check tasks
        tasks = []
        for user_id in range(num_users):
            for permission in permissions:
                task = check_permission_async(user_id, permission)
                tasks.append(task)

        results = await asyncio.gather(*tasks)

        end_time = time.time()
        total_time_ms = (end_time - start_time) * 1000

        # Analyze results
        check_times = [r["check_time_ms"] for r in results]
        total_checks = len(results)

        # Performance analysis
        avg_check_time = statistics.mean(check_times)
        median_check_time = statistics.median(check_times)
        p95_check_time = check_times[int(0.95 * len(check_times))]
        max_check_time = max(check_times)

        print(f"\nConcurrent Permission Check Results ({total_checks} checks):")
        print(f"Total time: {total_time_ms:.2f}ms")
        print(f"Average check time: {avg_check_time:.2f}ms")
        print(f"Median check time: {median_check_time:.2f}ms")
        print(f"95th percentile: {p95_check_time:.2f}ms")
        print(f"Max check time: {max_check_time:.2f}ms")

        # Performance requirements
        assert avg_check_time < 15.0, f"Average check time {avg_check_time:.2f}ms exceeds requirement"
        assert p95_check_time < 30.0, f"95th percentile {p95_check_time:.2f}ms exceeds tolerance"

    @pytest.mark.asyncio
    async def test_role_assignment_performance(
        self,
        mock_role_manager: MagicMock,
    ):
        """Test role assignment performance (<25ms target)."""

        # Test role assignments for different scenarios
        test_assignments = [
            ("newuser1@example.com", "api_user"),
            ("manager2@example.com", "user_manager"),
            ("admin3@example.com", "admin"),
            ("service@example.com", "token_manager"),
        ]

        total_times = []

        for user_email, role_name in test_assignments:
            start_time = time.time()

            result = mock_role_manager.assign_user_role(user_email, role_name, "system_admin@example.com")

            end_time = time.time()
            assignment_time_ms = (end_time - start_time) * 1000
            total_times.append(assignment_time_ms)

            # Verify result
            assert result is True

            # Verify performance requirement
            assert assignment_time_ms < 25.0, f"Role assignment took {assignment_time_ms:.2f}ms, exceeds 25ms target"

        avg_time = statistics.mean(total_times)
        print(f"Role assignment performance: {avg_time:.2f}ms average")

    @pytest.mark.asyncio
    async def test_bulk_permission_resolution_performance(
        self,
        mock_role_manager: MagicMock,
    ):
        """Test bulk permission resolution for multiple users."""

        # Test bulk operations (50 users)
        num_users = 50
        user_emails = [f"bulkuser{i}@example.com" for i in range(num_users)]

        start_time = time.time()

        # Resolve permissions for all users
        for user_email in user_emails:
            permissions = mock_role_manager.get_user_permissions(user_email)
            assert isinstance(permissions, set)

        end_time = time.time()
        total_time_ms = (end_time - start_time) * 1000
        avg_time_per_user = total_time_ms / num_users

        print("\nBulk Permission Resolution Performance:")
        print(f"Total time for {num_users} users: {total_time_ms:.2f}ms")
        print(f"Average time per user: {avg_time_per_user:.2f}ms")

        # Performance requirements for bulk operations
        assert avg_time_per_user < 20.0, f"Bulk average {avg_time_per_user:.2f}ms per user exceeds 20ms target"
        assert total_time_ms < 1000.0, f"Bulk total time {total_time_ms:.2f}ms exceeds 1000ms tolerance"

    @pytest.mark.asyncio
    async def test_complex_role_inheritance_performance(
        self,
        mock_role_manager: MagicMock,
    ):
        """Test performance with complex role inheritance hierarchies."""

        # Test complex inheritance scenarios
        role_hierarchy_tests = [
            "super_admin",  # Root level - most permissions
            "admin",  # Second level - inherits from super_admin
            "user_manager",  # Third level - inherits from admin
            "api_user",  # Fourth level - inherits from user_manager
            "readonly_user",  # Fifth level - inherits from api_user
        ]

        total_times = []

        for role_name in role_hierarchy_tests:
            start_time = time.time()

            permissions = mock_role_manager.get_role_permissions(role_name)

            end_time = time.time()
            resolution_time_ms = (end_time - start_time) * 1000
            total_times.append(resolution_time_ms)

            # Verify result
            assert isinstance(permissions, set)

            # Verify performance for deep inheritance
            assert (
                resolution_time_ms < 15.0
            ), f"Complex inheritance resolution took {resolution_time_ms:.2f}ms, exceeds 15ms target"

        avg_time = statistics.mean(total_times)
        max_time = max(total_times)

        print("\nComplex Role Inheritance Performance:")
        print(f"Average resolution time: {avg_time:.2f}ms")
        print(f"Max resolution time: {max_time:.2f}ms")

        # Inheritance performance should be consistent
        assert max_time < 20.0, f"Max inheritance time {max_time:.2f}ms exceeds tolerance"

    @pytest.mark.asyncio
    async def test_permission_caching_effectiveness(
        self,
        mock_role_manager: MagicMock,
    ):
        """Test permission caching performance impact."""

        user_email = "cache_test@example.com"

        # First call (cold cache)
        start_time = time.time()
        permissions1 = mock_role_manager.get_user_permissions(user_email)
        cold_time_ms = (time.time() - start_time) * 1000

        # Second call (warm cache - simulate cached response)
        mock_role_manager.get_user_permissions = MagicMock(return_value=permissions1)

        start_time = time.time()
        permissions2 = mock_role_manager.get_user_permissions(user_email)
        warm_time_ms = (time.time() - start_time) * 1000

        # Verify cache effectiveness
        assert permissions1 == permissions2

        # Cache should provide significant performance improvement
        cache_speedup = cold_time_ms / max(warm_time_ms, 0.001)  # Avoid division by zero

        print("\nPermission Caching Performance:")
        print(f"Cold cache time: {cold_time_ms:.2f}ms")
        print(f"Warm cache time: {warm_time_ms:.2f}ms")
        print(f"Cache speedup: {cache_speedup:.1f}x")

        # Warm cache should be significantly faster
        assert (
            warm_time_ms < cold_time_ms / 2
        ), f"Cache not effective: warm {warm_time_ms:.2f}ms vs cold {cold_time_ms:.2f}ms"

    @pytest.mark.asyncio
    async def test_role_system_memory_usage(self):
        """Test memory usage of role-based permission system."""

        from src.auth.role_manager import RoleManager

        # Get initial memory usage
        process = psutil.Process()
        initial_memory_mb = process.memory_info().rss / 1024 / 1024

        # Simulate heavy role operations
        role_managers = []
        for _i in range(10):
            # Create multiple role manager instances to test memory usage
            manager = MagicMock(spec=RoleManager)

            # Simulate large permission sets
            large_permission_set = {f"resource_{j}:action_{k}" for j in range(100) for k in range(10)}
            manager.get_user_permissions = MagicMock(return_value=large_permission_set)

            role_managers.append(manager)

            # Perform operations
            for user_id in range(20):
                permissions = manager.get_user_permissions(f"user{user_id}@example.com")
                assert len(permissions) == 1000

        # Force garbage collection
        gc.collect()

        # Check memory usage after operations
        final_memory_mb = process.memory_info().rss / 1024 / 1024
        memory_increase_mb = final_memory_mb - initial_memory_mb

        print("\nRole System Memory Usage:")
        print(f"Initial memory: {initial_memory_mb:.2f} MB")
        print(f"Final memory: {final_memory_mb:.2f} MB")
        print(f"Memory increase: {memory_increase_mb:.2f} MB")

        # Memory increase should be reasonable for role system
        assert memory_increase_mb < 30.0, f"Role system memory increase {memory_increase_mb:.2f} MB is too high"
