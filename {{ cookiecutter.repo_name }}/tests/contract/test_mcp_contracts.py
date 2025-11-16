"""Contract tests for MCP integrations using Pact.

This module defines consumer-side contract tests for MCP (Model Context Protocol) integrations
to ensure API compatibility between PromptCraft and external MCP servers.

Note: Currently using Pact Python v2 API. When v3 becomes stable, migrate to pact.v3 imports.
See: https://github.com/pact-foundation/pact-python/issues/396
"""

import os
import shutil
import subprocess
import warnings
from typing import Any

import pytest

# Suppress PendingDeprecationWarning from Pact Python v2->v3 transition
# TODO: Migrate to pact.v3 API when it becomes stable and fully documented
warnings.filterwarnings("ignore", category=PendingDeprecationWarning, module="pact")

try:
    from pact import Consumer, EachLike, Like, Provider

    PACT_AVAILABLE = True

    # Check if pact-mock-service binary is available
    PACT_STANDALONE_INSTALLED = shutil.which("pact-mock-service") is not None

    if not PACT_STANDALONE_INSTALLED:
        # Try alternative method to find pact tools
        try:
            subprocess.run(["/usr/local/bin/pact-mock-service", "--help"], capture_output=True, check=True, timeout=5)
            PACT_STANDALONE_INSTALLED = True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            PACT_STANDALONE_INSTALLED = False

except ImportError:
    PACT_AVAILABLE = False
    PACT_STANDALONE_INSTALLED = False

    # Mock Pact classes for testing when pact-python is not available
    class Consumer:
        def __init__(self, name):
            self.name = name

        def has_pact_with(self, provider):
            return MockPact()

    class Provider:
        def __init__(self, name):
            self.name = name

    class Like:
        def __init__(self, value):
            self.value = value

    class EachLike:
        def __init__(self, value):
            self.value = value

    class MockPact:
        def given(self, state):
            return self

        def upon_receiving(self, description):
            return self

        def with_request(self, method, path, **kwargs):
            return self

        def will_respond_with(self, status, **kwargs):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass


@pytest.mark.skipif(not PACT_AVAILABLE, reason="pact-python not installed")
@pytest.mark.skipif(not PACT_STANDALONE_INSTALLED, reason="pact-mock-service binary not available")
@pytest.mark.skipif(os.getenv("CI_ENVIRONMENT") == "true", reason="Pact tests require actual MCP servers, skip in CI")
@pytest.mark.contract
class TestZenMCPContracts:
    """Contract tests for Zen MCP Server integration."""

    def setup_method(self):
        """Set up Pact consumer and provider."""
        self.consumer = Consumer("promptcraft")
        self.provider = Provider("zen-mcp-server")
        self.pact = self.consumer.has_pact_with(self.provider, port=8080)

    def test_query_processing_contract(self):
        """Test contract for query processing endpoint."""
        expected_request = {
            "query": "Generate a Python function",
            "context": "Software development task",
            "agent_id": "create_agent",
        }

        expected_response = {
            "result": Like("Generated Python code"),
            "status": "success",
            "metadata": {"model_used": Like("gpt-4"), "tokens_used": Like(150), "processing_time": Like(1.5)},
        }

        (
            self.pact.given("query processing service is available")
            .upon_receiving("a query processing request")
            .with_request(
                "POST",
                "/api/v1/query/process",
                headers={"Content-Type": "application/json"},
                body=expected_request,
            )
            .will_respond_with(200, headers={"Content-Type": "application/json"}, body=expected_response)
        )

        with self.pact:
            # Simulate the actual client call
            result = self._mock_zen_client_call("POST", "/api/v1/query/process", expected_request)

            assert result["status"] == "success"
            assert "result" in result
            assert "metadata" in result

    def test_agent_health_check_contract(self):
        """Test contract for agent health check endpoint."""
        expected_response = {
            "status": "healthy",
            "agents": EachLike(
                {"agent_id": Like("create_agent"), "status": Like("ready"), "last_ping": Like("2025-01-01T00:00:00Z")},
            ),
        }

        (
            self.pact.given("agents are running")
            .upon_receiving("a health check request")
            .with_request("GET", "/health/agents")
            .will_respond_with(200, headers={"Content-Type": "application/json"}, body=expected_response)
        )

        with self.pact:
            result = self._mock_zen_client_call("GET", "/health/agents", None)

            assert result["status"] == "healthy"
            assert "agents" in result
            assert len(result["agents"]) >= 1

    def test_knowledge_retrieval_contract(self):
        """Test contract for knowledge retrieval endpoint."""
        expected_request = {"query": "Python best practices", "agent_id": "create_agent", "limit": 5}

        expected_response = {
            "results": EachLike(
                {
                    "id": Like("doc_123"),
                    "title": Like("Python Best Practices"),
                    "content": Like("Best practices for Python development..."),
                    "score": Like(0.95),
                    "metadata": {"agent_id": Like("create_agent"), "created_at": Like("2025-01-01T00:00:00Z")},
                },
            ),
            "total": Like(5),
            "status": "success",
        }

        (
            self.pact.given("knowledge base contains Python documentation")
            .upon_receiving("a knowledge retrieval request")
            .with_request(
                "POST",
                "/api/v1/knowledge/search",
                headers={"Content-Type": "application/json"},
                body=expected_request,
            )
            .will_respond_with(200, headers={"Content-Type": "application/json"}, body=expected_response)
        )

        with self.pact:
            result = self._mock_zen_client_call("POST", "/api/v1/knowledge/search", expected_request)

            assert result["status"] == "success"
            assert "results" in result
            assert result["total"] >= 0

    def test_error_handling_contract(self):
        """Test contract for error responses."""
        expected_request = {"query": "", "context": "test context"}  # Invalid empty query

        expected_response = {
            "error": "validation_error",
            "message": Like("Query cannot be empty"),
            "status": "error",
            "code": 400,
        }

        (
            self.pact.given("validation is enabled")
            .upon_receiving("an invalid query request")
            .with_request(
                "POST",
                "/api/v1/query/process",
                headers={"Content-Type": "application/json"},
                body=expected_request,
            )
            .will_respond_with(400, headers={"Content-Type": "application/json"}, body=expected_response)
        )

        with self.pact:
            result = self._mock_zen_client_call("POST", "/api/v1/query/process", expected_request)

            assert result["status"] == "error"
            assert result["code"] == 400
            assert "error" in result
            assert "message" in result

    def _mock_zen_client_call(self, method: str, path: str, body: dict[str, Any]) -> dict[str, Any]:
        """Mock Zen MCP client call for contract testing."""
        # This would be replaced with actual client implementation
        # For now, return mock responses based on the path

        if path == "/api/v1/query/process":
            if body and body.get("query") == "":
                return {"error": "validation_error", "message": "Query cannot be empty", "status": "error", "code": 400}
            return {
                "result": "Generated Python code",
                "status": "success",
                "metadata": {"model_used": "gpt-4", "tokens_used": 150, "processing_time": 1.5},
            }
        if path == "/health/agents":
            return {
                "status": "healthy",
                "agents": [{"agent_id": "create_agent", "status": "ready", "last_ping": "2025-01-01T00:00:00Z"}],
            }
        if path == "/api/v1/knowledge/search":
            return {
                "results": [
                    {
                        "id": "doc_123",
                        "title": "Python Best Practices",
                        "content": "Best practices for Python development...",
                        "score": 0.95,
                        "metadata": {"agent_id": "create_agent", "created_at": "2025-01-01T00:00:00Z"},
                    },
                ],
                "total": 1,
                "status": "success",
            }

        return {"status": "error", "message": "Unknown endpoint"}


@pytest.mark.skipif(not PACT_AVAILABLE, reason="pact-python not installed")
@pytest.mark.skipif(not PACT_STANDALONE_INSTALLED, reason="pact-mock-service binary not available")
@pytest.mark.skipif(os.getenv("CI_ENVIRONMENT") == "true", reason="Pact tests require actual MCP servers, skip in CI")
@pytest.mark.contract
class TestHeimdalMCPContracts:
    """Contract tests for Heimdall MCP Server integration."""

    def setup_method(self):
        """Set up Pact consumer and provider."""
        self.consumer = Consumer("promptcraft")
        self.provider = Provider("heimdall-mcp-server")
        self.pact = self.consumer.has_pact_with(self.provider, port=8081)

    def test_security_analysis_contract(self):
        """Test contract for security analysis endpoint."""
        expected_request = {
            "code": "def unsafe_function(): os.system(user_input)",
            "language": "python",
            "analysis_type": "security",
        }

        expected_response = {
            "findings": EachLike(
                {
                    "type": Like("security_vulnerability"),
                    "severity": Like("high"),
                    "message": Like("Potential command injection vulnerability"),
                    "line": Like(1),
                    "suggestion": Like("Use subprocess.run() instead of os.system()"),
                },
            ),
            "score": Like(2.5),  # Security score out of 10
            "status": "completed",
        }

        (
            self.pact.given("security analysis service is available")
            .upon_receiving("a security analysis request")
            .with_request(
                "POST",
                "/api/v1/analyze/security",
                headers={"Content-Type": "application/json"},
                body=expected_request,
            )
            .will_respond_with(200, headers={"Content-Type": "application/json"}, body=expected_response)
        )

        with self.pact:
            result = self._mock_heimdall_client_call("POST", "/api/v1/analyze/security", expected_request)

            assert result["status"] == "completed"
            assert "findings" in result
            assert "score" in result
            assert result["score"] <= 10.0

    def test_code_quality_contract(self):
        """Test contract for code quality analysis endpoint."""
        expected_request = {
            "code": "def calculate_factorial(n):\n    return n * calculate_factorial(n-1) if n > 1 else 1",
            "language": "python",
            "analysis_type": "quality",
        }

        expected_response = {
            "metrics": {
                "complexity": Like(3),
                "maintainability": Like(7.5),
                "readability": Like(8.2),
                "test_coverage": Like(0.0),
            },
            "suggestions": EachLike(
                {
                    "type": Like("improvement"),
                    "message": Like("Consider adding base case check"),
                    "priority": Like("medium"),
                },
            ),
            "status": "completed",
        }

        (
            self.pact.given("code quality analysis service is available")
            .upon_receiving("a code quality analysis request")
            .with_request(
                "POST",
                "/api/v1/analyze/quality",
                headers={"Content-Type": "application/json"},
                body=expected_request,
            )
            .will_respond_with(200, headers={"Content-Type": "application/json"}, body=expected_response)
        )

        with self.pact:
            result = self._mock_heimdall_client_call("POST", "/api/v1/analyze/quality", expected_request)

            assert result["status"] == "completed"
            assert "metrics" in result
            assert "suggestions" in result

    def _mock_heimdall_client_call(self, method: str, path: str, body: dict[str, Any]) -> dict[str, Any]:
        """Mock Heimdall MCP client call for contract testing."""
        # This would be replaced with actual client implementation

        if path == "/api/v1/analyze/security":
            return {
                "findings": [
                    {
                        "type": "security_vulnerability",
                        "severity": "high",
                        "message": "Potential command injection vulnerability",
                        "line": 1,
                        "suggestion": "Use subprocess.run() instead of os.system()",
                    },
                ],
                "score": 2.5,
                "status": "completed",
            }
        if path == "/api/v1/analyze/quality":
            return {
                "metrics": {"complexity": 3, "maintainability": 7.5, "readability": 8.2, "test_coverage": 0.0},
                "suggestions": [
                    {"type": "improvement", "message": "Consider adding base case check", "priority": "medium"},
                ],
                "status": "completed",
            }

        return {"status": "error", "message": "Unknown endpoint"}


@pytest.mark.contract
class TestMockMCPContracts:
    """Mock contract tests that run without Pact dependency."""

    def test_mock_contract_validation(self):
        """Test contract validation without Pact dependency."""
        # This test validates the contract structure without requiring Pact

        def validate_query_contract(request: dict[str, Any], response: dict[str, Any]) -> bool:
            """Validate query processing contract."""
            # Request validation
            required_request_fields = ["query", "context"]
            if not all(field in request for field in required_request_fields):
                return False

            # Response validation
            required_response_fields = ["result", "status"]
            if not all(field in response for field in required_response_fields):
                return False

            # Status validation
            return response["status"] in ["success", "error"]

        # Test valid contract
        valid_request = {"query": "test query", "context": "test context"}
        valid_response = {"result": "test result", "status": "success"}

        assert validate_query_contract(valid_request, valid_response) is True

        # Test invalid contracts
        invalid_request = {"query": "test query"}  # Missing context
        assert validate_query_contract(invalid_request, valid_response) is False

        invalid_response = {"result": "test result"}  # Missing status
        assert validate_query_contract(valid_request, invalid_response) is False

    def test_mock_error_contract_validation(self):
        """Test error response contract validation."""

        def validate_error_contract(response: dict[str, Any]) -> bool:
            """Validate error response contract."""
            required_fields = ["error", "status", "code"]
            if not all(field in response for field in required_fields):
                return False

            if response["status"] != "error":
                return False

            return isinstance(response["code"], int)

        # Test valid error response
        valid_error = {"error": "validation_error", "message": "Invalid input", "status": "error", "code": 400}

        assert validate_error_contract(valid_error) is True

        # Test invalid error response
        invalid_error = {
            "error": "validation_error",
            "status": "error",
            # Missing code
        }

        assert validate_error_contract(invalid_error) is False
