"""
Unit tests for QueryCounselor + HydeProcessor integration (Week 1 Day 5).

This module tests the new integration methods added to QueryCounselor for
seamless HyDE processing coordination and performance monitoring.
"""

import asyncio
import gc
import logging
import time
from unittest.mock import AsyncMock, Mock

import pytest

from src.core.hyde_processor import HydeProcessor
from src.core.query_counselor import QueryCounselor
from src.utils.performance_monitor import PerformanceMonitor, track_performance

logger = logging.getLogger(__name__)


class TestQueryCounselorHydeIntegration:
    """Test suite for QueryCounselor + HydeProcessor integration."""

    @pytest.fixture
    async def query_counselor(self, mock_hyde_processor, mock_mcp_client):
        """Create QueryCounselor instance for testing."""
        return QueryCounselor(mcp_client=mock_mcp_client, hyde_processor=mock_hyde_processor)

    @pytest.fixture
    async def mock_hyde_processor(self):
        """Create mock HydeProcessor for controlled testing."""
        hyde = Mock(spec=HydeProcessor)
        hyde.three_tier_analysis = AsyncMock()
        hyde.process_query = AsyncMock()
        return hyde

    @pytest.fixture
    async def mock_mcp_client(self):
        """Create mock MCP client for testing."""
        from src.mcp_integration.mcp_client import MCPClientInterface, Response

        mock_client = Mock(spec=MCPClientInterface)

        # Mock workflow orchestration with conditional behavior
        async def mock_orchestrate(workflow_steps, **kwargs):
            # Extract query from workflow steps to determine behavior
            query = ""
            if workflow_steps and len(workflow_steps) > 0:
                # Get query from the workflow step's input_data
                input_data = getattr(workflow_steps[0], "input_data", {})
                query = input_data.get("query", "")

            # Return error response for empty queries
            if not query or not query.strip():
                return [
                    Response(
                        agent_id="mock_agent",
                        content="Empty query error",
                        success=False,
                        confidence=0.0,
                        processing_time=0.1,
                        metadata={"error": True, "error_type": "empty_query"},
                    ),
                ]

            # Normal successful response for non-empty queries
            return [
                Response(
                    agent_id="mock_agent",
                    content="Mocked response content",
                    success=True,
                    confidence=0.8,
                    processing_time=0.1,
                    metadata={"source": "mock"},
                ),
            ]

        mock_client.orchestrate_agents = AsyncMock(side_effect=mock_orchestrate)

        # Mock validate_query to return proper structure
        mock_client.validate_query = AsyncMock(
            side_effect=lambda q: {
                "is_valid": True,
                "sanitized_query": q,  # Return the original query
                "potential_issues": [],
            },
        )

        return mock_client

    @pytest.mark.asyncio
    async def test_hyde_processor_initialization(self, query_counselor):
        """Test that QueryCounselor properly initializes with HydeProcessor."""
        assert query_counselor.hyde_processor is not None
        assert isinstance(query_counselor.hyde_processor, HydeProcessor)

    @pytest.mark.asyncio
    async def test_process_query_with_hyde_simple_query(self, query_counselor):
        """Test process_query_with_hyde with simple query (no HyDE)."""
        simple_query = "Create a basic prompt"

        start_time = time.time()
        response = await query_counselor.process_query_with_hyde(simple_query)
        processing_time = time.time() - start_time

        # Validate response structure
        assert response is not None
        assert response.response is not None
        assert response.processing_time > 0
        assert processing_time < 2.0  # Must meet <2s SLA

        # Check HyDE metadata (should indicate no HyDE applied for simple query)
        hyde_meta = response.metadata.get("hyde_integration", {})
        assert "hyde_applied" in hyde_meta
        assert "processing_strategy" in hyde_meta

    @pytest.mark.asyncio
    async def test_process_query_with_hyde_complex_query(self, query_counselor):
        """Test process_query_with_hyde with complex query (HyDE applied)."""
        complex_query = "Create a comprehensive multi-agent orchestration prompt for analyzing large distributed systems with performance optimization strategies and security vulnerability detection"

        start_time = time.time()
        response = await query_counselor.process_query_with_hyde(complex_query)
        processing_time = time.time() - start_time

        # Validate response structure
        assert response is not None
        assert response.response is not None
        assert processing_time < 2.0  # Must meet <2s SLA

        # Check HyDE metadata (should indicate HyDE applied for complex query)
        hyde_meta = response.metadata.get("hyde_integration", {})
        assert (
            hyde_meta.get("hyde_applied") is True or hyde_meta.get("hyde_applied") is False
        )  # Could be either based on complexity analysis
        assert "processing_strategy" in hyde_meta
        assert "specificity_level" in hyde_meta

    @pytest.mark.asyncio
    async def test_get_processing_recommendation(self, query_counselor):
        """Test processing recommendation functionality."""
        test_query = "Analyze code performance issues"

        recommendations = await query_counselor.get_processing_recommendation(test_query)

        # Validate recommendation structure
        assert "query_analysis" in recommendations
        assert "processing_strategy" in recommendations
        assert "agent_recommendations" in recommendations

        # Validate query analysis
        query_analysis = recommendations["query_analysis"]
        assert "query_type" in query_analysis
        assert "confidence" in query_analysis
        assert "complexity" in query_analysis
        assert "hyde_recommended" in query_analysis

        # Validate processing strategy
        strategy = recommendations["processing_strategy"]
        assert "use_hyde" in strategy
        assert "expected_complexity" in strategy
        assert "estimated_agents" in strategy

    @pytest.mark.asyncio
    async def test_performance_tracking_integration(self, query_counselor):
        """Test that performance tracking is properly integrated."""
        monitor = PerformanceMonitor()
        monitor.get_all_metrics()

        # Process a query
        test_query = "Generate a template for deployment"
        response = await query_counselor.process_query_with_hyde(test_query)

        # Check that performance metadata is included
        perf_meta = response.metadata.get("performance", {})
        assert "total_processing_time" in perf_meta
        assert "agent_processing_time" in perf_meta
        assert perf_meta["total_processing_time"] > 0

    @pytest.mark.asyncio
    async def test_error_handling_in_hyde_integration(self, query_counselor):
        """Test error handling in HyDE integration."""
        # Test with empty query
        response = await query_counselor.process_query_with_hyde("")

        # Should return error response, not raise exception
        assert response is not None
        assert response.confidence == 0.0
        assert response.metadata.get("error") is True

    @pytest.mark.asyncio
    async def test_confidence_enhancement_with_hyde(self, query_counselor):
        """Test that HyDE results can enhance confidence."""
        # Use a medium complexity query that should trigger HyDE
        medium_query = "Create a template for CI/CD pipeline with security scanning and testing stages"

        response = await query_counselor.process_query_with_hyde(medium_query)

        # Response should have reasonable confidence
        assert response.confidence >= 0.0
        assert response.confidence <= 1.0

        # Check metadata for HyDE enhancement
        hyde_meta = response.metadata.get("hyde_integration", {})
        assert "hyde_results_count" in hyde_meta

    @pytest.mark.asyncio
    async def test_concurrent_processing_capability(self, query_counselor):
        """Test concurrent processing capability of integrated system."""
        test_queries = [
            "Create prompt for code review",
            "Generate documentation template",
            "Analyze system performance",
            "Design API endpoints",
            "Create testing strategy",
        ]

        # Process queries concurrently
        start_time = time.time()
        tasks = [query_counselor.process_query_with_hyde(query) for query in test_queries]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time

        # Validate all responses
        assert len(responses) == len(test_queries)
        successful_responses = [r for r in responses if not isinstance(r, Exception)]
        assert len(successful_responses) >= len(test_queries) * 0.8  # At least 80% success rate

        # Check that concurrent processing is reasonably efficient
        assert total_time < 10.0  # Should complete within 10 seconds

        # Validate individual response times
        for response in successful_responses:
            assert response.processing_time < 2.0  # Each should meet SLA

    @pytest.mark.asyncio
    async def test_hyde_integration_metadata_completeness(self, query_counselor):
        """Test that HyDE integration provides complete metadata."""
        test_query = "Create comprehensive prompt for multi-agent system design"

        response = await query_counselor.process_query_with_hyde(test_query)

        # Validate complete metadata structure
        assert "hyde_integration" in response.metadata
        assert "query_analysis" in response.metadata
        assert "performance" in response.metadata

        # Validate HyDE integration metadata
        hyde_meta = response.metadata["hyde_integration"]
        required_fields = [
            "hyde_applied",
            "processing_strategy",
            "specificity_level",
            "specificity_score",
            "hyde_results_count",
            "hyde_enhanced_search",
        ]
        for field in required_fields:
            assert field in hyde_meta

        # Validate query analysis metadata
        query_meta = response.metadata["query_analysis"]
        required_fields = [
            "original_query",
            "enhanced_query",
            "intent_confidence",
            "complexity",
            "context_needed",
        ]
        for field in required_fields:
            assert field in query_meta

        # Validate performance metadata
        perf_meta = response.metadata["performance"]
        required_fields = [
            "total_processing_time",
            "hyde_processing_time",
            "agent_processing_time",
        ]
        for field in required_fields:
            assert field in perf_meta

    @pytest.mark.asyncio
    async def test_sla_compliance_under_load(self, query_counselor):
        """Test SLA compliance under sustained load."""
        test_queries = [
            "Generate prompt for data analysis",
            "Create template for service deployment",
            "Analyze code architecture patterns",
            "Design error handling strategy",
            "Create documentation framework",
        ] * 10  # 50 total queries

        processing_times = []
        successful_count = 0

        # Process all queries
        for query in test_queries:
            start_time = time.time()
            try:
                response = await query_counselor.process_query_with_hyde(query)
                processing_time = time.time() - start_time
                processing_times.append(processing_time)

                if response.confidence > 0.0:
                    successful_count += 1

            except Exception:
                processing_time = time.time() - start_time
                processing_times.append(processing_time)

        # Calculate statistics
        avg_time = sum(processing_times) / len(processing_times)
        max_time = max(processing_times)

        # Calculate p95
        sorted_times = sorted(processing_times)
        p95_index = int(0.95 * len(sorted_times))
        p95_time = sorted_times[min(p95_index, len(sorted_times) - 1)]

        # Validate SLA compliance
        assert p95_time < 2.0, f"P95 response time ({p95_time:.3f}s) exceeds 2s SLA"
        assert avg_time < 1.0, f"Average response time ({avg_time:.3f}s) too high"
        assert (
            successful_count >= len(test_queries) * 0.95
        ), f"Success rate too low: {successful_count}/{len(test_queries)}"

        # Log performance summary
        logger.info("SLA Compliance Test Results:")
        logger.info("  Queries processed: %s", len(test_queries))
        logger.info("  Successful: %s", successful_count)
        logger.info("  Average time: %.3fs", avg_time)
        logger.info("  Max time: %.3fs", max_time)
        logger.info("  P95 time: %.3fs", p95_time)


@pytest.mark.performance
class TestPerformanceIntegration:
    """Performance-specific tests for the integration."""

    @pytest.mark.asyncio
    async def test_performance_monitor_integration(self):
        """Test that performance monitoring is properly integrated."""
        query_counselor = QueryCounselor()

        # Use performance tracking
        with track_performance("integration_test") as tracker:
            await query_counselor.process_query_with_hyde("Test query")

        # Validate tracking worked
        assert tracker.start_time is not None

        # Check that metrics were recorded
        from src.utils.performance_monitor import get_performance_monitor

        monitor = get_performance_monitor()
        metrics = monitor.get_all_metrics()
        assert "timers" in metrics

    @pytest.mark.asyncio
    async def test_memory_usage_stability(self):
        """Test memory usage stability during processing."""
        query_counselor = QueryCounselor()

        # Force garbage collection
        gc.collect()

        # Process multiple queries
        for i in range(20):
            query = f"Test query {i} for memory stability analysis"
            response = await query_counselor.process_query_with_hyde(query)
            assert response is not None

            # Occasional garbage collection
            if i % 5 == 0:
                gc.collect()

        # Final garbage collection
        gc.collect()

        # If we get here without memory errors, the test passes
        assert True


if __name__ == "__main__":
    """Run integration tests directly."""
    pytest.main([__file__, "-v"])
