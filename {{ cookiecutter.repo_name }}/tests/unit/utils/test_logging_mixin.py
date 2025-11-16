"""Comprehensive tests for src/utils/logging_mixin.py module."""

import logging
from unittest.mock import ANY, Mock, patch

import pytest

from src.utils.logging_mixin import (
    LoggerMixin,
    StructuredLoggerMixin,
    get_component_logger,
)


class TestLoggerMixin:
    """Test LoggerMixin class functionality."""

    def test_basic_initialization_default_logger_name(self):
        """Test LoggerMixin initialization with default logger name."""

        class TestComponent(LoggerMixin):
            pass

        component = TestComponent()

        # Should create logger with class-based name
        expected_name = f"promptcraft.{TestComponent.__module__}.TestComponent"
        assert component.logger.name == expected_name
        assert isinstance(component.logger, logging.Logger)

    def test_initialization_custom_logger_name(self):
        """Test LoggerMixin initialization with custom logger name."""

        class TestComponent(LoggerMixin):
            pass

        component = TestComponent(logger_name="custom_component")

        assert component.logger.name == "promptcraft.custom_component"

    def test_initialization_custom_logger_name_with_prefix(self):
        """Test LoggerMixin initialization with custom logger name that already has prefix."""

        class TestComponent(LoggerMixin):
            pass

        component = TestComponent(logger_name="promptcraft.already_prefixed")

        assert component.logger.name == "promptcraft.already_prefixed"

    def test_initialization_custom_log_level(self):
        """Test LoggerMixin initialization with custom log level."""

        class TestComponent(LoggerMixin):
            pass

        component = TestComponent(log_level=logging.DEBUG)

        assert component.logger.level == logging.DEBUG

    def test_initialization_default_log_level(self):
        """Test LoggerMixin sets INFO level when logger has no level set."""

        class TestComponent(LoggerMixin):
            pass

        with patch("logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_logger.level = 0  # No level set
            mock_get_logger.return_value = mock_logger

            TestComponent()

            mock_logger.setLevel.assert_called_with(logging.INFO)

    def test_initialization_preserves_existing_log_level(self):
        """Test LoggerMixin preserves existing log level when one is set."""

        class TestComponent(LoggerMixin):
            pass

        with patch("logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_logger.level = logging.WARNING  # Existing level
            mock_get_logger.return_value = mock_logger

            TestComponent()

            # Should not call setLevel since level is already set
            assert not any(call[0][0] == logging.INFO for call in mock_logger.setLevel.call_args_list)

    def test_initialization_with_parent_class_args(self):
        """Test LoggerMixin initialization passes arguments to parent classes."""

        class BaseClass:
            def __init__(self, base_arg, base_kwarg=None, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.base_arg = base_arg
                self.base_kwarg = base_kwarg

        class TestComponent(LoggerMixin, BaseClass):
            def __init__(self, base_arg, base_kwarg=None, logger_name=None, **kwargs):
                # Pass logger_name to LoggerMixin but not to BaseClass
                super().__init__(logger_name=logger_name, base_arg=base_arg, base_kwarg=base_kwarg, **kwargs)

        component = TestComponent("test_arg", base_kwarg="test_kwarg", logger_name="test")

        assert component.base_arg == "test_arg"
        assert component.base_kwarg == "test_kwarg"
        assert component.logger.name == "promptcraft.test"

    def test_log_method_entry_with_debug_enabled(self):
        """Test log_method_entry logs when DEBUG level is enabled."""

        class TestComponent(LoggerMixin):
            pass

        component = TestComponent()
        component.logger = Mock()
        component.logger.isEnabledFor.return_value = True

        component.log_method_entry("test_method", "arg1", "arg2", kwarg1="value1", kwarg2="value2")

        component.logger.isEnabledFor.assert_called_with(logging.DEBUG)
        component.logger.debug.assert_called_once_with(
            "Entering %s with args=%s, kwargs=%s",
            "test_method",
            ("arg1", "arg2"),
            {"kwarg1": "value1", "kwarg2": "value2"},
        )

    def test_log_method_entry_filters_private_kwargs(self):
        """Test log_method_entry filters out private kwargs starting with underscore."""

        class TestComponent(LoggerMixin):
            pass

        component = TestComponent()
        component.logger = Mock()
        component.logger.isEnabledFor.return_value = True

        component.log_method_entry("test_method", public_arg="public", _private_arg="private")

        # Should only include public kwargs
        component.logger.debug.assert_called_once_with(
            "Entering %s with args=%s, kwargs=%s",
            "test_method",
            (),
            {"public_arg": "public"},
        )

    def test_log_method_entry_with_debug_disabled(self):
        """Test log_method_entry does not log when DEBUG level is disabled."""

        class TestComponent(LoggerMixin):
            pass

        component = TestComponent()
        component.logger = Mock()
        component.logger.isEnabledFor.return_value = False

        component.log_method_entry("test_method", "arg1")

        component.logger.isEnabledFor.assert_called_with(logging.DEBUG)
        component.logger.debug.assert_not_called()

    def test_log_method_exit_with_result(self):
        """Test log_method_exit logs with result when DEBUG is enabled."""

        class TestComponent(LoggerMixin):
            pass

        component = TestComponent()
        component.logger = Mock()
        component.logger.isEnabledFor.return_value = True

        component.log_method_exit("test_method", result="test_result")

        component.logger.debug.assert_called_once_with("Exiting %s with result=%s", "test_method", "test_result")

    def test_log_method_exit_without_result(self):
        """Test log_method_exit logs without result when result is None."""

        class TestComponent(LoggerMixin):
            pass

        component = TestComponent()
        component.logger = Mock()
        component.logger.isEnabledFor.return_value = True

        component.log_method_exit("test_method")

        component.logger.debug.assert_called_once_with("Exiting %s", "test_method")

    def test_log_method_exit_with_debug_disabled(self):
        """Test log_method_exit does not log when DEBUG is disabled."""

        class TestComponent(LoggerMixin):
            pass

        component = TestComponent()
        component.logger = Mock()
        component.logger.isEnabledFor.return_value = False

        component.log_method_exit("test_method", result="test_result")

        component.logger.debug.assert_not_called()

    def test_log_error_with_context_minimal(self):
        """Test log_error_with_context with minimal parameters."""

        class TestComponent(LoggerMixin):
            pass

        component = TestComponent()
        component.logger = Mock()

        error = ValueError("Test error")
        component.log_error_with_context(error)

        component.logger.error.assert_called_once_with("Error%s: %s - Context: %s", "", error, {}, exc_info=True)

    def test_log_error_with_context_full(self):
        """Test log_error_with_context with all parameters."""

        class TestComponent(LoggerMixin):
            pass

        component = TestComponent()
        component.logger = Mock()

        error = ValueError("Test error")
        context = {"key": "value", "request_id": "123"}
        component.log_error_with_context(error, context=context, method_name="test_method")

        component.logger.error.assert_called_once_with(
            "Error%s: %s - Context: %s",
            " in test_method",
            error,
            context,
            exc_info=True,
        )

    def test_log_performance_metric_minimal(self):
        """Test log_performance_metric with minimal parameters."""

        class TestComponent(LoggerMixin):
            pass

        component = TestComponent()
        component.logger = Mock()

        component.log_performance_metric("response_time", 150.5)

        component.logger.info.assert_called_once_with(
            "PERF_METRIC: %s=%.2f%s - Context: %s",
            "response_time",
            150.5,
            "ms",
            {},
        )

    def test_log_performance_metric_full(self):
        """Test log_performance_metric with all parameters."""

        class TestComponent(LoggerMixin):
            pass

        component = TestComponent()
        component.logger = Mock()

        context = {"endpoint": "/api/test", "method": "GET"}
        component.log_performance_metric("response_time", 150.5, unit="seconds", context=context)

        component.logger.info.assert_called_once_with(
            "PERF_METRIC: %s=%.2f%s - Context: %s",
            "response_time",
            150.5,
            "seconds",
            context,
        )

    def test_log_state_change_minimal(self):
        """Test log_state_change with minimal parameters."""

        class TestComponent(LoggerMixin):
            pass

        component = TestComponent()
        component.logger = Mock()

        component.log_state_change("idle", "processing")

        component.logger.info.assert_called_once_with(
            "STATE_CHANGE: %s -> %s%s - Context: %s",
            "idle",
            "processing",
            "",
            {},
        )

    def test_log_state_change_full(self):
        """Test log_state_change with all parameters."""

        class TestComponent(LoggerMixin):
            pass

        component = TestComponent()
        component.logger = Mock()

        context = {"user_id": "123", "session_id": "abc"}
        component.log_state_change("idle", "processing", reason="user_request", context=context)

        component.logger.info.assert_called_once_with(
            "STATE_CHANGE: %s -> %s%s - Context: %s",
            "idle",
            "processing",
            " (reason: user_request)",
            context,
        )

    def test_log_business_event_minimal(self):
        """Test log_business_event with minimal parameters."""

        class TestComponent(LoggerMixin):
            pass

        component = TestComponent()
        component.logger = Mock()

        component.log_business_event("user_login")

        component.logger.log.assert_called_once_with(logging.INFO, "BUSINESS_EVENT: %s - Data: %s", "user_login", {})

    def test_log_business_event_full(self):
        """Test log_business_event with all parameters."""

        class TestComponent(LoggerMixin):
            pass

        component = TestComponent()
        component.logger = Mock()

        event_data = {"user_id": "123", "timestamp": "2023-01-01"}
        component.log_business_event("user_login", event_data=event_data, level=logging.WARNING)

        component.logger.log.assert_called_once_with(
            logging.WARNING,
            "BUSINESS_EVENT: %s - Data: %s",
            "user_login",
            event_data,
        )


class TestStructuredLoggerMixin:
    """Test StructuredLoggerMixin class functionality."""

    def test_initialization_default_values(self):
        """Test StructuredLoggerMixin initialization with default values."""

        class TestComponent(StructuredLoggerMixin):
            pass

        component = TestComponent()

        assert component.component_id == "TestComponent_instance"
        assert component.correlation_id is None

    def test_initialization_custom_values(self):
        """Test StructuredLoggerMixin initialization with custom values."""

        class TestComponent(StructuredLoggerMixin):
            pass

        component = TestComponent(component_id="custom_id", correlation_id="corr_123")

        assert component.component_id == "custom_id"
        assert component.correlation_id == "corr_123"

    def test_get_structured_context_minimal(self):
        """Test _get_structured_context with minimal setup."""

        class TestComponent(StructuredLoggerMixin):
            pass

        component = TestComponent(component_id="test_id")
        context = component._get_structured_context()

        expected_context = {"component_id": "test_id", "component_class": "TestComponent"}
        assert context == expected_context

    def test_get_structured_context_with_correlation_id(self):
        """Test _get_structured_context includes correlation ID when set."""

        class TestComponent(StructuredLoggerMixin):
            pass

        component = TestComponent(component_id="test_id", correlation_id="corr_123")
        context = component._get_structured_context()

        expected_context = {"component_id": "test_id", "component_class": "TestComponent", "correlation_id": "corr_123"}
        assert context == expected_context

    def test_get_structured_context_with_additional_context(self):
        """Test _get_structured_context merges additional context."""

        class TestComponent(StructuredLoggerMixin):
            pass

        component = TestComponent(component_id="test_id")
        additional = {"request_id": "req_456", "user_id": "user_789"}
        context = component._get_structured_context(additional)

        expected_context = {
            "component_id": "test_id",
            "component_class": "TestComponent",
            "request_id": "req_456",
            "user_id": "user_789",
        }
        assert context == expected_context

    def test_log_structured_minimal(self):
        """Test log_structured with minimal parameters."""

        class TestComponent(StructuredLoggerMixin):
            pass

        component = TestComponent(component_id="test_id")
        component.logger = Mock()

        component.log_structured(logging.INFO, "Test message")

        expected_context = {"component_id": "test_id", "component_class": "TestComponent"}
        component.logger.log.assert_called_once_with(logging.INFO, "%s - Context: %s", "Test message", expected_context)

    def test_log_structured_with_event_type(self):
        """Test log_structured includes event type when provided."""

        class TestComponent(StructuredLoggerMixin):
            pass

        component = TestComponent(component_id="test_id")
        component.logger = Mock()

        component.log_structured(logging.INFO, "Test message", event_type="api_call")

        expected_context = {"component_id": "test_id", "component_class": "TestComponent", "event_type": "api_call"}
        component.logger.log.assert_called_once_with(logging.INFO, "%s - Context: %s", "Test message", expected_context)

    def test_log_structured_with_kwargs(self):
        """Test log_structured includes kwargs in context."""

        class TestComponent(StructuredLoggerMixin):
            pass

        component = TestComponent(component_id="test_id")
        component.logger = Mock()

        component.log_structured(
            logging.INFO,
            "Test message",
            event_type="api_call",
            request_id="req_123",
            user_id="user_456",
        )

        expected_context = {
            "component_id": "test_id",
            "component_class": "TestComponent",
            "event_type": "api_call",
            "request_id": "req_123",
            "user_id": "user_456",
        }
        component.logger.log.assert_called_once_with(logging.INFO, "%s - Context: %s", "Test message", expected_context)

    def test_log_api_call_minimal(self):
        """Test log_api_call with minimal parameters."""

        class TestComponent(StructuredLoggerMixin):
            pass

        component = TestComponent(component_id="test_id")
        component.logger = Mock()

        component.log_api_call("/api/test")

        expected_context = {
            "component_id": "test_id",
            "component_class": "TestComponent",
            "event_type": "api_call",
            "endpoint": "/api/test",
            "method": "GET",
        }
        component.logger.log.assert_called_once_with(
            logging.INFO,
            "%s - Context: %s",
            "API call: GET /api/test",
            expected_context,
        )

    def test_log_api_call_full(self):
        """Test log_api_call with all parameters."""

        class TestComponent(StructuredLoggerMixin):
            pass

        component = TestComponent(component_id="test_id")
        component.logger = Mock()

        component.log_api_call("/api/test", method="POST", status_code=201, duration_ms=150.5, request_id="req_123")

        expected_context = {
            "component_id": "test_id",
            "component_class": "TestComponent",
            "event_type": "api_call",
            "endpoint": "/api/test",
            "method": "POST",
            "status_code": 201,
            "duration_ms": 150.5,
            "request_id": "req_123",
        }
        component.logger.log.assert_called_once_with(
            logging.INFO,
            "%s - Context: %s",
            "API call: POST /api/test",
            expected_context,
        )

    def test_log_api_call_partial_parameters(self):
        """Test log_api_call with some optional parameters."""

        class TestComponent(StructuredLoggerMixin):
            pass

        component = TestComponent(component_id="test_id")
        component.logger = Mock()

        component.log_api_call("/api/test", method="PUT", status_code=200)

        expected_context = {
            "component_id": "test_id",
            "component_class": "TestComponent",
            "event_type": "api_call",
            "endpoint": "/api/test",
            "method": "PUT",
            "status_code": 200,
        }
        component.logger.log.assert_called_once_with(
            logging.INFO,
            "%s - Context: %s",
            "API call: PUT /api/test",
            expected_context,
        )

    def test_inheritance_from_logger_mixin(self):
        """Test that StructuredLoggerMixin inherits from LoggerMixin."""

        class TestComponent(StructuredLoggerMixin):
            pass

        component = TestComponent()

        # Should have all LoggerMixin methods
        assert hasattr(component, "logger")
        assert hasattr(component, "log_method_entry")
        assert hasattr(component, "log_method_exit")
        assert hasattr(component, "log_error_with_context")

        # Should also have StructuredLoggerMixin methods
        assert hasattr(component, "log_structured")
        assert hasattr(component, "log_api_call")


class TestGetComponentLogger:
    """Test get_component_logger function."""

    @patch("src.utils.logging_mixin.logging.getLogger")
    def test_get_component_logger_default_level(self, mock_get_logger):
        """Test get_component_logger with default log level."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        result = get_component_logger("test_component")

        mock_get_logger.assert_called_once_with("promptcraft.test_component")
        mock_logger.setLevel.assert_called_once_with(logging.INFO)
        assert result == mock_logger

    @patch("src.utils.logging_mixin.logging.getLogger")
    def test_get_component_logger_custom_level(self, mock_get_logger):
        """Test get_component_logger with custom log level."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        result = get_component_logger("test_component", level=logging.DEBUG)

        mock_get_logger.assert_called_once_with("promptcraft.test_component")
        mock_logger.setLevel.assert_called_once_with(logging.DEBUG)
        assert result == mock_logger

    @patch("src.utils.logging_mixin.logging.getLogger")
    def test_get_component_logger_name_formatting(self, mock_get_logger):
        """Test get_component_logger formats name correctly."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        get_component_logger("my.component.name")

        mock_get_logger.assert_called_once_with("promptcraft.my.component.name")


class TestEdgeCasesAndComplexScenarios:
    """Test edge cases and complex scenarios."""

    def test_mixin_with_multiple_inheritance(self):
        """Test LoggerMixin works correctly with multiple inheritance."""

        class BaseClass:
            def __init__(self, base_value, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.base_value = base_value

        class OtherMixin:
            def __init__(self, other_value, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.other_value = other_value

        class ComplexComponent(LoggerMixin, OtherMixin, BaseClass):
            def __init__(self, base_value, other_value=None, logger_name=None, **kwargs):
                # Pass logger_name to LoggerMixin but not to other classes
                super().__init__(logger_name=logger_name, base_value=base_value, other_value=other_value, **kwargs)

        component = ComplexComponent("base", other_value="other", logger_name="complex")

        assert component.base_value == "base"
        assert component.other_value == "other"
        assert component.logger.name == "promptcraft.complex"

    def test_log_method_entry_with_complex_args(self):
        """Test log_method_entry handles complex argument types."""

        class TestComponent(LoggerMixin):
            pass

        component = TestComponent()
        component.logger = Mock()
        component.logger.isEnabledFor.return_value = True

        # Test with various argument types
        complex_args = [None, 123, {"key": "value"}, ["list", "items"]]
        complex_kwargs = {
            "none_value": None,
            "dict_value": {"nested": "dict"},
            "list_value": [1, 2, 3],
            "_private": "should be filtered",
        }

        component.log_method_entry("complex_method", *complex_args, **complex_kwargs)

        # Should handle complex types and filter private kwargs
        component.logger.debug.assert_called_once()
        call_args = component.logger.debug.call_args[0]
        assert call_args[1] == "complex_method"
        assert call_args[2] == tuple(complex_args)  # Args are converted to tuple
        assert "_private" not in call_args[3]
        assert "dict_value" in call_args[3]

    def test_structured_logger_context_merging(self):
        """Test that structured logger properly merges contexts."""

        class TestComponent(StructuredLoggerMixin):
            pass

        component = TestComponent(component_id="test_id", correlation_id="corr_123")

        # Test context merging with overlapping keys
        additional_context = {"component_id": "override_id", "new_key": "new_value"}  # Should override

        context = component._get_structured_context(additional_context)

        # additional_context should override base context
        assert context["component_id"] == "override_id"
        assert context["correlation_id"] == "corr_123"
        assert context["new_key"] == "new_value"

    def test_logger_name_generation_edge_cases(self):
        """Test logger name generation handles edge cases."""

        class ComponentWithLongName(LoggerMixin):
            pass

        # Test with very long module name
        with patch.object(ComponentWithLongName, "__module__", "very.long.module.path.that.is.quite.deep"):
            component = ComponentWithLongName()
            expected_name = "promptcraft.very.long.module.path.that.is.quite.deep.ComponentWithLongName"
            assert component.logger.name == expected_name

    def test_log_performance_metric_with_zero_and_negative_values(self):
        """Test log_performance_metric handles zero and negative values."""

        class TestComponent(LoggerMixin):
            pass

        component = TestComponent()
        component.logger = Mock()

        # Test zero value
        component.log_performance_metric("zero_metric", 0.0)
        component.logger.info.assert_called_with("PERF_METRIC: %s=%.2f%s - Context: %s", "zero_metric", 0.0, "ms", {})

        # Test negative value
        component.log_performance_metric("negative_metric", -15.5)
        assert any(call[0][2] == -15.5 for call in component.logger.info.call_args_list)

    def test_log_state_change_with_none_values(self):
        """Test log_state_change handles None values properly."""

        class TestComponent(LoggerMixin):
            pass

        component = TestComponent()
        component.logger = Mock()

        # Test with None states
        component.log_state_change(None, "active", reason=None)

        component.logger.info.assert_called_once_with("STATE_CHANGE: %s -> %s%s - Context: %s", None, "active", "", {})

    def test_empty_and_none_context_handling(self):
        """Test that empty and None contexts are handled properly."""

        class TestComponent(LoggerMixin):
            pass

        component = TestComponent()
        component.logger = Mock()

        # Test with None context
        component.log_error_with_context(ValueError("test"), context=None)

        component.logger.error.assert_called_with(
            "Error%s: %s - Context: %s",
            "",
            ANY,
            {},
            exc_info=True,  # The ValueError instance
        )

        # Test with empty context
        component.log_performance_metric("test_metric", 100.0, context={})
        assert any(call[0][4] == {} for call in component.logger.info.call_args_list)


@pytest.mark.parametrize(
    ("logger_name", "expected_name"),
    [
        ("simple", "promptcraft.simple"),
        ("component.name", "promptcraft.component.name"),
        ("promptcraft.already_prefixed", "promptcraft.already_prefixed"),
        ("promptcraft.nested.component", "promptcraft.nested.component"),
        (None, None),  # Will be handled by class-based naming
    ],
    ids=["simple", "dotted", "already_prefixed", "nested_prefixed", "none"],
)
def test_logger_name_formatting_parametrized(logger_name, expected_name):
    """Parametrized test for logger name formatting."""

    class TestComponent(LoggerMixin):
        pass

    if logger_name is not None:
        component = TestComponent(logger_name=logger_name)
        assert component.logger.name == expected_name
    else:
        component = TestComponent()
        # Should use class-based naming
        assert component.logger.name.startswith("promptcraft.")
        assert "TestComponent" in component.logger.name


@pytest.mark.parametrize(
    "log_level",
    [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL],
    ids=["debug", "info", "warning", "error", "critical"],
)
def test_custom_log_levels_parametrized(log_level):
    """Parametrized test for custom log levels."""

    class TestComponent(LoggerMixin):
        pass

    component = TestComponent(log_level=log_level)
    assert component.logger.level == log_level


class TestRealWorldUsageScenarios:
    """Test realistic usage scenarios."""

    def test_api_service_logging_pattern(self):
        """Test realistic API service logging pattern."""

        class APIService(StructuredLoggerMixin):
            def __init__(self, service_name, *args, **kwargs):
                super().__init__(
                    *args,
                    component_id=service_name,
                    correlation_id=None,
                    logger_name=f"api.{service_name}",
                    **kwargs,
                )

        service = APIService("user_service")
        service.logger = Mock()

        # Simulate API call logging
        service.log_api_call("/api/users/123", method="GET", status_code=200, duration_ms=45.2, user_id="user_123")

        # Verify structured logging
        service.logger.log.assert_called_once()
        call_args = service.logger.log.call_args[0]
        assert call_args[0] == logging.INFO
        assert call_args[1] == "%s - Context: %s"
        assert call_args[2] == "API call: GET /api/users/123"

        context = call_args[3]
        assert context["component_id"] == "user_service"
        assert context["endpoint"] == "/api/users/123"
        assert context["method"] == "GET"
        assert context["status_code"] == 200
        assert context["duration_ms"] == 45.2
        assert context["user_id"] == "user_123"

    def test_state_machine_logging_pattern(self):
        """Test realistic state machine logging pattern."""

        class StateMachine(LoggerMixin):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, logger_name="state_machine", **kwargs)
                self.state = "idle"

            def transition_to(self, new_state, reason=None):
                old_state = self.state
                self.state = new_state
                self.log_state_change(old_state, new_state, reason=reason)

        machine = StateMachine()
        machine.logger = Mock()

        # Simulate state transitions
        machine.transition_to("processing", reason="user_request")
        machine.transition_to("completed", reason="task_finished")

        # Verify state change logging
        assert machine.logger.info.call_count == 2

        # Check first transition
        first_call = machine.logger.info.call_args_list[0][0]
        # The format string is "STATE_CHANGE: %s -> %s%s - Context: %s"
        # first_call[0] is the format string, first_call[1] is "idle", first_call[2] is "processing", first_call[3] is " (reason: user_request)"
        assert first_call[0] == "STATE_CHANGE: %s -> %s%s - Context: %s"
        assert first_call[1] == "idle"
        assert first_call[2] == "processing"
        assert "(reason: user_request)" in first_call[3]

    def test_error_handling_with_context(self):
        """Test realistic error handling with context."""

        class DataProcessor(LoggerMixin):
            def process_data(self, data_id, user_id):
                try:
                    # Simulate processing
                    self.log_method_entry("process_data", data_id, user_id=user_id)

                    # Simulate error
                    raise ValueError("Invalid data format")

                except Exception as e:
                    self.log_error_with_context(
                        e,
                        context={"data_id": data_id, "user_id": user_id},
                        method_name="process_data",
                    )
                    raise

        processor = DataProcessor()
        processor.logger = Mock()
        processor.logger.isEnabledFor.return_value = True

        # Simulate error scenario
        with pytest.raises(ValueError, match="Invalid data format"):
            processor.process_data("data_123", "user_456")

        # Verify method entry logging
        processor.logger.debug.assert_called_once()

        # Verify error logging
        processor.logger.error.assert_called_once()
        error_call = processor.logger.error.call_args[0]
        assert "in process_data" in error_call[1]
        assert error_call[3] == {"data_id": "data_123", "user_id": "user_456"}
