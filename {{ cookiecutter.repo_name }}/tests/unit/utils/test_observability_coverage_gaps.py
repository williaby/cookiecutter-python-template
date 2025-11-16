"""Comprehensive tests for observability.py coverage gaps - targeting 0% coverage functions."""

import threading
from unittest.mock import Mock, patch

import pytest

import src.utils.observability as obs_module
from src.utils.observability import (
    AgentMetrics,
    NoOpSpan,
    OpenTelemetryInstrumentor,
    StructuredLogger,
)


class TestStructuredLoggerCoverageGaps:
    """Test StructuredLogger functions with 0% coverage."""

    def test_structured_logger_get_context_missing_coverage(self):
        """Test _get_context method covering missing branches."""
        logger = StructuredLogger("test_logger", correlation_id="test-id")

        # Mock the opentelemetry components to test trace context

        with (
            patch.object(obs_module, "OPENTELEMETRY_AVAILABLE", True),
            patch.object(obs_module, "trace", Mock(), create=True) as mock_trace,
        ):

            # Mock span with recording capability
            mock_span = Mock()
            mock_span.is_recording.return_value = True
            mock_span_context = Mock()
            mock_span_context.trace_id = 123456789
            mock_span_context.span_id = 987654321
            mock_span.get_span_context.return_value = mock_span_context
            mock_trace.get_current_span.return_value = mock_span

            context = logger._get_context()

            # Verify trace information is included
            assert "trace_id" in context
            assert "span_id" in context
            assert context["correlation_id"] == "test-id"

    def test_structured_logger_warning_method(self):
        """Test warning method with 0% coverage."""
        logger = StructuredLogger("test_logger")

        with patch.object(logger.logger, "warning") as mock_warning:
            logger.warning("Test warning message", extra_field="test_value")

            # Verify warning was called with structured data
            mock_warning.assert_called_once()
            call_args = mock_warning.call_args
            assert call_args[0][0] == "Test warning message"
            assert "structured_data" in call_args[1]["extra"]
            assert call_args[1]["extra"]["structured_data"]["extra_field"] == "test_value"

    def test_structured_logger_without_correlation_id(self):
        """Test _get_context without correlation_id."""
        logger = StructuredLogger("test_logger")  # No correlation_id

        context = logger._get_context()

        # Verify no correlation_id in context
        assert "correlation_id" not in context
        assert "timestamp" in context
        assert "logger" in context
        assert "thread_id" in context


class TestOpenTelemetryInstrumentorCoverageGaps:
    """Test OpenTelemetryInstrumentor functions with 0% coverage."""

    def test_setup_tracing_method(self):
        """Test _setup_tracing method with 0% coverage."""
        # Mock the entire observability module's OpenTelemetry imports
        mock_resource = Mock()
        mock_tracer_provider = Mock()
        mock_trace = Mock()
        mock_jaeger = Mock()
        mock_processor = Mock()
        mock_logging_inst = Mock()

        # Import observability module and add mocked attributes

        with (
            patch.object(obs_module, "OPENTELEMETRY_AVAILABLE", True),
            patch.object(obs_module, "Resource", mock_resource, create=True),
            patch.object(obs_module, "TracerProvider", mock_tracer_provider, create=True),
            patch.object(obs_module, "trace", mock_trace, create=True),
            patch.object(obs_module, "JaegerExporter", mock_jaeger, create=True),
            patch.object(obs_module, "BatchSpanProcessor", mock_processor, create=True),
            patch.object(obs_module, "LoggingInstrumentor", mock_logging_inst, create=True),
        ):

            # Setup mocks
            mock_resource.create.return_value = Mock()
            mock_tracer_provider_instance = Mock()
            mock_tracer_provider.return_value = mock_tracer_provider_instance
            mock_trace.get_tracer.return_value = Mock()
            mock_logging_inst.return_value.instrument = Mock()

            OpenTelemetryInstrumentor("test-service")

            # Verify setup was called
            mock_resource.create.assert_called_once()
            mock_tracer_provider.assert_called_once()
            mock_trace.set_tracer_provider.assert_called_once()
            mock_trace.get_tracer.assert_called_once()

    def test_setup_tracing_jaeger_exception(self):
        """Test _setup_tracing method when Jaeger export fails."""
        # Mock the observability module's OpenTelemetry imports
        mock_resource = Mock()
        mock_tracer_provider = Mock()
        mock_trace = Mock()
        mock_jaeger = Mock()
        mock_logging_inst = Mock()

        # Import observability module and add mocked attributes

        with (
            patch.object(obs_module, "OPENTELEMETRY_AVAILABLE", True),
            patch.object(obs_module, "Resource", mock_resource, create=True),
            patch.object(obs_module, "TracerProvider", mock_tracer_provider, create=True),
            patch.object(obs_module, "trace", mock_trace, create=True),
            patch.object(obs_module, "JaegerExporter", mock_jaeger, create=True),
            patch.object(obs_module, "LoggingInstrumentor", mock_logging_inst, create=True),
            patch("logging.getLogger") as mock_get_logger,
        ):

            # Setup mocks
            mock_resource.create.return_value = Mock()
            mock_tracer_provider_instance = Mock()
            mock_tracer_provider.return_value = mock_tracer_provider_instance
            mock_trace.get_tracer.return_value = Mock()
            mock_logging_inst.return_value.instrument = Mock()

            # Make Jaeger exporter raise exception
            mock_jaeger.side_effect = Exception("Jaeger not available")
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            OpenTelemetryInstrumentor("test-service")

            # Verify exception was logged
            mock_get_logger.assert_called()
            mock_logger.debug.assert_called()

    def test_start_span_method(self):
        """Test start_span method with 0% coverage."""
        # Mock all the OpenTelemetry imports that are used in the setup_tracing method

        mock_resource = Mock()
        mock_tracer_provider = Mock()
        mock_trace = Mock()

        with (
            patch.object(obs_module, "OPENTELEMETRY_AVAILABLE", True),
            patch.object(obs_module, "Resource", mock_resource, create=True),
            patch.object(obs_module, "TracerProvider", mock_tracer_provider, create=True),
            patch.object(obs_module, "trace", mock_trace, create=True),
            patch.object(obs_module, "LoggingInstrumentor", Mock(), create=True),
        ):

            # Setup the resource creation mock
            mock_resource.create.return_value = Mock()
            mock_tracer_provider_instance = Mock()
            mock_tracer_provider.return_value = mock_tracer_provider_instance

            # Create instrumentor
            instrumentor = OpenTelemetryInstrumentor("test-service")
            instrumentor.tracer = Mock()

            # Mock span
            mock_span = Mock()
            instrumentor.tracer.start_span.return_value = mock_span

            # Test with attributes
            attributes = {"key1": "value1", "key2": "value2"}
            result = instrumentor.start_span("test-span", attributes)

            # Verify span creation and attributes
            instrumentor.tracer.start_span.assert_called_once_with("test-span")
            mock_span.set_attribute.assert_any_call("key1", "value1")
            mock_span.set_attribute.assert_any_call("key2", "value2")
            assert result == mock_span

    def test_start_span_not_initialized(self):
        """Test start_span when not initialized."""
        instrumentor = OpenTelemetryInstrumentor("test-service")
        instrumentor.initialized = False

        result = instrumentor.start_span("test-span")

        # Should return NoOpSpan
        assert isinstance(result, NoOpSpan)

    def test_trace_operation_method(self):
        """Test trace_operation method with 0% coverage."""
        # Mock all the OpenTelemetry imports

        mock_resource = Mock()
        mock_tracer_provider = Mock()
        mock_trace = Mock()

        with (
            patch.object(obs_module, "OPENTELEMETRY_AVAILABLE", True),
            patch.object(obs_module, "Resource", mock_resource, create=True),
            patch.object(obs_module, "TracerProvider", mock_tracer_provider, create=True),
            patch.object(obs_module, "trace", mock_trace, create=True),
            patch.object(obs_module, "LoggingInstrumentor", Mock(), create=True),
        ):

            # Setup the resource creation mock
            mock_resource.create.return_value = Mock()
            mock_tracer_provider_instance = Mock()
            mock_tracer_provider.return_value = mock_tracer_provider_instance

            # Create instrumentor
            instrumentor = OpenTelemetryInstrumentor("test-service")
            instrumentor.tracer = Mock()

            # Mock context manager
            mock_span = Mock()
            mock_span.__enter__ = Mock(return_value=mock_span)
            mock_span.__exit__ = Mock(return_value=None)
            instrumentor.tracer.start_as_current_span.return_value = mock_span

            # Test successful operation
            with instrumentor.trace_operation("test-operation", attr1="value1"):
                pass

            # Verify span was created with attributes
            instrumentor.tracer.start_as_current_span.assert_called_once_with("test-operation")
            mock_span.set_attribute.assert_called_with("attr1", "value1")

    def test_trace_operation_with_exception(self):
        """Test trace_operation method when operation raises exception."""
        # Mock all the OpenTelemetry imports

        mock_resource = Mock()
        mock_tracer_provider = Mock()
        mock_trace = Mock()
        mock_status = Mock()
        mock_status_code = Mock()

        with (
            patch.object(obs_module, "OPENTELEMETRY_AVAILABLE", True),
            patch.object(obs_module, "Resource", mock_resource, create=True),
            patch.object(obs_module, "TracerProvider", mock_tracer_provider, create=True),
            patch.object(obs_module, "trace", mock_trace, create=True),
            patch.object(obs_module, "LoggingInstrumentor", Mock(), create=True),
            patch.object(obs_module, "Status", mock_status, create=True),
            patch.object(obs_module, "StatusCode", mock_status_code, create=True),
        ):

            # Setup the resource creation mock
            mock_resource.create.return_value = Mock()
            mock_tracer_provider_instance = Mock()
            mock_tracer_provider.return_value = mock_tracer_provider_instance

            # Create instrumentor
            instrumentor = OpenTelemetryInstrumentor("test-service")
            instrumentor.tracer = Mock()

            # Mock context manager and status
            mock_span = Mock()
            mock_span.__enter__ = Mock(return_value=mock_span)
            mock_span.__exit__ = Mock(return_value=None)
            instrumentor.tracer.start_as_current_span.return_value = mock_span
            mock_status_code.ERROR = "ERROR"

            # Test operation that raises exception
            test_exception = ValueError("Test error")
            with pytest.raises(ValueError, match="Test error"), instrumentor.trace_operation("test-operation"):
                raise test_exception

            # Verify error handling
            mock_span.set_status.assert_called_once()
            mock_span.record_exception.assert_called_once_with(test_exception)

    def test_trace_operation_not_initialized(self):
        """Test trace_operation when not initialized."""
        instrumentor = OpenTelemetryInstrumentor("test-service")
        instrumentor.initialized = False

        # Should not fail when not initialized
        with instrumentor.trace_operation("test-operation"):
            pass


class TestNoOpSpanCoverageGaps:
    """Test NoOpSpan methods with 0% coverage."""

    def test_noop_span_enter_method(self):
        """Test NoOpSpan __enter__ method."""
        span = NoOpSpan()
        result = span.__enter__()
        assert result is span

    def test_noop_span_exit_method(self):
        """Test NoOpSpan __exit__ method."""
        span = NoOpSpan()
        result = span.__exit__(None, None, None)
        assert result is None

    def test_noop_span_set_attribute_method(self):
        """Test NoOpSpan set_attribute method."""
        span = NoOpSpan()
        # Should not raise any exception
        span.set_attribute("key", "value")

    def test_noop_span_set_status_method(self):
        """Test NoOpSpan set_status method."""
        span = NoOpSpan()
        # Should not raise any exception
        span.set_status("some_status")

    def test_noop_span_record_exception_method(self):
        """Test NoOpSpan record_exception method."""
        span = NoOpSpan()
        # Should not raise any exception
        span.record_exception(Exception("test"))


class TestAgentMetricsCoverageGaps:
    """Test AgentMetrics methods with 0% coverage."""

    def test_agent_metrics_get_metrics_method(self):
        """Test get_metrics method with 0% coverage."""
        metrics = AgentMetrics()

        # Add some duration data
        metrics.record_duration("agent_execution_duration_seconds", 1.5)
        metrics.record_duration("agent_execution_duration_seconds", 2.0)
        metrics.record_duration("agent_execution_duration_seconds", 0.5)

        result = metrics.get_metrics()

        # Verify statistics are calculated
        assert "agent_execution_duration_stats" in result
        stats = result["agent_execution_duration_stats"]
        assert stats["count"] == 3
        assert stats["avg"] == (1.5 + 2.0 + 0.5) / 3
        assert stats["min"] == 0.5
        assert stats["max"] == 2.0
        assert "p95" in stats
        assert "p99" in stats

    def test_agent_metrics_percentile_method(self):
        """Test _percentile method with 0% coverage."""
        metrics = AgentMetrics()

        # Test with values
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        p95 = metrics._percentile(values, 95)
        assert p95 == 5.0  # 95th percentile of small dataset

        p50 = metrics._percentile(values, 50)
        assert p50 == 3.0  # 50th percentile (median)

        # Test with empty list
        empty_p95 = metrics._percentile([], 95)
        assert empty_p95 == 0.0

    def test_agent_metrics_get_metrics_empty_durations(self):
        """Test get_metrics with empty duration list."""
        metrics = AgentMetrics()

        result = metrics.get_metrics()

        # Should not have duration stats for empty list
        assert "agent_execution_duration_stats" not in result

    def test_agent_metrics_threading_safety(self):
        """Test AgentMetrics thread safety."""
        metrics = AgentMetrics()
        results = []

        def worker():
            for i in range(100):
                metrics.increment_counter("agent_executions_total")
                metrics.record_duration("agent_execution_duration_seconds", 0.1 * i)
            results.append(metrics.get_metrics())

        # Start multiple threads
        threads = []
        for _ in range(5):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # Verify final metrics
        final_metrics = metrics.get_metrics()
        assert final_metrics["agent_executions_total"] == 500  # 5 threads * 100 increments
        assert len(final_metrics["agent_execution_duration_seconds"]) == 500


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling scenarios."""

    def test_structured_logger_no_opentelemetry(self):
        """Test StructuredLogger when OpenTelemetry is not available."""
        with patch("src.utils.observability.OPENTELEMETRY_AVAILABLE", False):
            logger = StructuredLogger("test_logger", correlation_id="test-id")
            context = logger._get_context()

            # Should not include trace information
            assert "trace_id" not in context
            assert "span_id" not in context
            assert context["correlation_id"] == "test-id"

    def test_structured_logger_non_recording_span(self):
        """Test StructuredLogger with non-recording span."""

        with (
            patch.object(obs_module, "OPENTELEMETRY_AVAILABLE", True),
            patch.object(obs_module, "trace", Mock(), create=True) as mock_trace,
        ):

            # Mock span that is not recording
            mock_span = Mock()
            mock_span.is_recording.return_value = False
            mock_trace.get_current_span.return_value = mock_span

            logger = StructuredLogger("test_logger")
            context = logger._get_context()

            # Should not include trace information for non-recording span
            assert "trace_id" not in context
            assert "span_id" not in context

    def test_opentelemetry_instrumentor_no_opentelemetry(self):
        """Test OpenTelemetryInstrumentor when OpenTelemetry is not available."""
        with patch("src.utils.observability.OPENTELEMETRY_AVAILABLE", False):
            instrumentor = OpenTelemetryInstrumentor("test-service")

            # Should not be initialized
            assert instrumentor.tracer is None
            assert not instrumentor.initialized

    def test_agent_metrics_unknown_metric(self):
        """Test AgentMetrics with unknown metric names."""
        metrics = AgentMetrics()

        # Should not raise error for unknown metrics
        metrics.increment_counter("unknown_metric")
        metrics.record_duration("unknown_duration", 1.5)

        # Unknown metrics should not appear in results
        result = metrics.get_metrics()
        assert "unknown_metric" not in result
        assert "unknown_duration" not in result

    def test_agent_metrics_non_list_duration_metric(self):
        """Test AgentMetrics with corrupted duration metric."""
        metrics = AgentMetrics()

        # Corrupt the metric (make it not a list)
        metrics.metrics["agent_execution_duration_seconds"] = "not_a_list"

        # Should not fail
        metrics.record_duration("agent_execution_duration_seconds", 1.5)
        result = metrics.get_metrics()

        # Should handle gracefully
        assert "agent_execution_duration_stats" not in result

    def test_agent_metrics_memory_management(self):
        """Test AgentMetrics memory management for large datasets."""
        metrics = AgentMetrics()

        # Add more than the max measurements (1000)
        for i in range(1500):
            metrics.record_duration("agent_execution_duration_seconds", float(i))

        result = metrics.get_metrics()
        durations = result["agent_execution_duration_seconds"]

        # Should keep only last 1000 measurements
        assert len(durations) == 1000
        assert durations[0] == 500.0  # First of the last 1000
        assert durations[-1] == 1499.0  # Last measurement


@pytest.mark.parametrize("correlation_id", [None, "test-correlation-123"])
def test_structured_logger_correlation_scenarios(correlation_id):
    """Test StructuredLogger with different correlation ID scenarios."""
    logger = StructuredLogger("test_logger", correlation_id=correlation_id)

    context = logger._get_context()

    if correlation_id:
        assert context["correlation_id"] == correlation_id
    else:
        assert "correlation_id" not in context


@pytest.mark.parametrize("service_name", ["test-service", "promptcraft-agents", "custom-service"])
def test_opentelemetry_instrumentor_service_names(service_name):
    """Test OpenTelemetryInstrumentor with different service names."""
    with patch("src.utils.observability.OPENTELEMETRY_AVAILABLE", False):
        instrumentor = OpenTelemetryInstrumentor(service_name)
        assert instrumentor.service_name == service_name


@pytest.mark.parametrize(("percentile", "expected_index"), [(50, 2), (95, 4), (99, 4)])
def test_agent_metrics_percentile_calculations(percentile, expected_index):
    """Test AgentMetrics percentile calculations with different percentiles."""
    metrics = AgentMetrics()
    values = [1.0, 2.0, 3.0, 4.0, 5.0]

    result = metrics._percentile(values, percentile)
    expected = values[expected_index]

    assert result == expected
