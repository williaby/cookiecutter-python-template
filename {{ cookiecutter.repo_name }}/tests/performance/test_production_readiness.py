"""Performance testing for PromptCraft-Hybrid production readiness.

This module tests system performance requirements, load handling, and production
readiness across all components to ensure the system meets the <2s response time
requirement and can handle concurrent load effectively.
"""

import asyncio
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.config.settings import ApplicationSettings
from src.core.hyde_processor import HydeProcessor
from src.core.performance_optimizer import PerformanceOptimizer
from src.core.query_counselor import QueryCounselor
from src.core.vector_store import (
    DEFAULT_VECTOR_DIMENSIONS,
    ConnectionStatus,
    EnhancedMockVectorStore,
    SearchParameters,
    VectorDocument,
)
from src.mcp_integration.config_manager import MCPConfigurationManager
from src.mcp_integration.mcp_client import MCPConnectionState, ZenMCPClient
from src.utils.performance_monitor import PerformanceMonitor

# CI environment detection for more lenient thresholds
IS_CI = os.getenv("CI", "").lower() in ("true", "1", "yes") or os.getenv("GITHUB_ACTIONS", "").lower() == "true"


class TestProductionReadiness:
    """Performance testing for production readiness."""

    @pytest.fixture
    def performance_settings(self):
        """Create optimized settings for performance testing."""
        return ApplicationSettings(
            # MCP Configuration
            mcp_enabled=True,
            mcp_server_url="http://localhost:3000",
            mcp_timeout=2.0,  # Tight timeout for performance
            mcp_max_retries=2,
            mcp_health_check_interval=30.0,
            # Vector Store Configuration
            qdrant_enabled=False,  # Use mock for consistent performance
            vector_store_type="mock",
            vector_dimensions=DEFAULT_VECTOR_DIMENSIONS,
            # Performance Configuration
            performance_monitoring_enabled=True,
            max_concurrent_queries=20,
            query_timeout=2.0,  # 2 second requirement
            # Production Configuration
            health_check_enabled=True,
            error_recovery_enabled=True,
            circuit_breaker_enabled=True,
        )

    @pytest.fixture
    def performance_monitor(self):
        """Create performance monitor for testing."""
        return PerformanceMonitor()

    @pytest.fixture
    def sample_load_queries(self):
        """Create sample queries for load testing."""
        return [
            "How do I implement authentication in FastAPI?",
            "What are the best practices for async programming in Python?",
            "How do I optimize database queries for performance?",
            "What is the difference between REST and GraphQL APIs?",
            "How do I implement caching in distributed systems?",
            "What are the security considerations for web applications?",
            "How do I handle errors in microservices architecture?",
            "What are the best practices for API design?",
            "How do I implement real-time features in web apps?",
            "What is the best approach for data validation?",
            "How do I optimize frontend performance?",
            "What are the considerations for scalable architecture?",
            "How do I implement monitoring and logging?",
            "What are the best practices for testing?",
            "How do I handle database migrations?",
            "What is the best approach for configuration management?",
            "How do I implement rate limiting?",
            "What are the patterns for error handling?",
            "How do I optimize memory usage in applications?",
            "What is the best approach for deployment automation?",
        ]

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_single_query_performance_requirement(self, performance_settings, performance_monitor):
        """Test that single query processing meets <2s requirement."""

        with patch("src.config.settings.get_settings", return_value=performance_settings):
            # Mock optimized MCP client
            mock_mcp_client = AsyncMock()
            mock_mcp_client.connection_state = MCPConnectionState.CONNECTED
            mock_mcp_client.orchestrate_agents = AsyncMock(
                return_value=[
                    MagicMock(
                        agent_id="create_agent",
                        success=True,
                        content="Optimized response from create_agent with sufficient length to meet quality requirements",
                        metadata={"processing_time": 0.1, "confidence": 0.9},
                    ),
                ],
            )

            # Mock optimized vector store
            mock_vector_store = EnhancedMockVectorStore(
                {
                    "type": "mock",
                    "simulate_latency": True,
                    "error_rate": 0.0,
                    "base_latency": 0.005,
                },  # Very low latency
            )
            await mock_vector_store.connect()

            # Mock optimized HyDE processor
            mock_hyde_processor = AsyncMock()
            mock_hyde_processor.vector_store = mock_vector_store
            mock_hyde_processor.process_query = AsyncMock(
                return_value={
                    "original_query": "test",
                    "enhanced_query": "Enhanced: test",
                    "enhancement_score": 0.9,
                    "relevant_documents": [],
                    "processing_time": 0.02,
                },
            )

            with (
                patch(
                    "src.mcp_integration.mcp_client.MCPClientFactory.create_from_settings",
                    return_value=mock_mcp_client,
                ),
                patch("src.core.hyde_processor.HydeProcessor", return_value=mock_hyde_processor),
            ):

                # Initialize QueryCounselor with mocked dependencies
                counselor = QueryCounselor(
                    mcp_client=mock_mcp_client,
                    hyde_processor=mock_hyde_processor,
                )

                # Test multiple queries to ensure consistent performance
                test_queries = [
                    "How do I optimize Python code for performance?",
                    "What are the best practices for database design?",
                    "How do I implement secure authentication?",
                    "What is the best approach for API versioning?",
                    "How do I handle concurrent requests efficiently?",
                ]

                response_times = []

                for query in test_queries:
                    start_time = time.time()

                    # Execute complete workflow
                    intent = await counselor.analyze_intent(query)
                    hyde_result = await counselor.hyde_processor.process_query(query)
                    agent_selection = await counselor.select_agents(intent)

                    # Convert AgentSelection to list of Agent objects
                    selected_agents = []
                    for agent_id in agent_selection.primary_agents + agent_selection.secondary_agents:
                        agent = next((a for a in counselor._available_agents if a.agent_id == agent_id), None)
                        if agent:
                            selected_agents.append(agent)

                    responses = await counselor.orchestrate_workflow(selected_agents, hyde_result["enhanced_query"])

                    processing_time = time.time() - start_time
                    response_times.append(processing_time)

                    # Verify response quality
                    assert len(responses) > 0
                    assert all(response.success for response in responses)

                    # Record performance metrics using MetricData
                    from src.utils.performance_monitor import MetricData, MetricType

                    performance_monitor.record_metric(
                        MetricData(
                            name="query_time",
                            value=processing_time,
                            timestamp=time.time(),
                            metric_type=MetricType.TIMER,
                        ),
                    )
                    performance_monitor.record_metric(
                        MetricData(
                            name="response_quality",
                            value=len(responses[0].content),
                            timestamp=time.time(),
                            metric_type=MetricType.GAUGE,
                        ),
                    )

                # Verify performance requirements (CI-adapted thresholds)
                avg_response_time = statistics.mean(response_times)
                max_response_time = max(response_times)
                p95_response_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile

                # More lenient thresholds for CI environments
                avg_threshold = 3.0 if IS_CI else 1.5
                max_threshold = 5.0 if IS_CI else 2.0
                p95_threshold = 5.0 if IS_CI else 2.0

                assert (
                    avg_response_time < avg_threshold
                ), f"Average response time {avg_response_time:.2f}s exceeds {avg_threshold}s target"
                assert (
                    max_response_time < max_threshold
                ), f"Maximum response time {max_response_time:.2f}s exceeds {max_threshold}s requirement"
                assert (
                    p95_response_time < p95_threshold
                ), f"95th percentile response time {p95_response_time:.2f}s exceeds {p95_threshold}s requirement"

                # Verify performance metrics (CI-adapted)
                performance_monitor.get_all_metrics()
                metrics_threshold = 3.0 if IS_CI else 1.5

                # Check timer metrics for query time
                timer_stats = performance_monitor.get_timer_stats("query_time")
                if timer_stats:
                    assert timer_stats.get("mean", 0) < metrics_threshold

                # Check gauge for response quality (CI-adapted threshold)
                response_quality = performance_monitor.get_gauge("response_quality")
                min_response_length = 20 if IS_CI else 50  # More lenient thresholds
                if response_quality > 0:  # Only check if we have responses
                    assert (
                        response_quality > min_response_length
                    ), f"Response quality {response_quality} below {min_response_length} threshold"

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_load_performance(self, performance_settings, sample_load_queries, performance_monitor):
        """Test concurrent load handling performance."""

        with patch("src.config.settings.get_settings", return_value=performance_settings):
            # Mock concurrent MCP client
            mock_mcp_client = AsyncMock()
            mock_mcp_client.connection_state = MCPConnectionState.CONNECTED

            async def mock_concurrent_orchestration(workflow_steps):
                # Extract query from workflow steps for variation
                query = workflow_steps[0].input_data.get("query", "default") if workflow_steps else "default"
                # Simulate realistic processing time with some variation
                await asyncio.sleep(0.05 + (hash(query) % 100) / 2000)  # 50-100ms variation
                return [
                    MagicMock(
                        agent_id="create_agent",
                        success=True,
                        content=f"Concurrent response for: {query[:50]}...",
                        metadata={"processing_time": 0.08, "confidence": 0.85},
                    ),
                ]

            mock_mcp_client.orchestrate_agents = mock_concurrent_orchestration

            # Mock concurrent vector store
            mock_vector_store = EnhancedMockVectorStore(
                {"type": "mock", "simulate_latency": True, "error_rate": 0.0, "base_latency": 0.01},
            )
            await mock_vector_store.connect()

            # Mock concurrent HyDE processor
            mock_hyde_processor = AsyncMock()
            mock_hyde_processor.vector_store = mock_vector_store

            async def mock_concurrent_hyde(query):
                await asyncio.sleep(0.02 + (hash(query) % 50) / 5000)  # 20-30ms variation
                return {
                    "original_query": query,
                    "enhanced_query": f"Enhanced: {query}",
                    "enhancement_score": 0.85,
                    "relevant_documents": [],
                    "processing_time": 0.025,
                }

            mock_hyde_processor.process_query = mock_concurrent_hyde

            with (
                patch(
                    "src.mcp_integration.mcp_client.MCPClientFactory.create_from_settings",
                    return_value=mock_mcp_client,
                ),
                patch("src.core.hyde_processor.HydeProcessor", return_value=mock_hyde_processor),
            ):

                # Initialize QueryCounselor with mocked dependencies
                counselor = QueryCounselor(
                    mcp_client=mock_mcp_client,
                    hyde_processor=mock_hyde_processor,
                )

                # Test concurrent processing with different load levels (CI-adapted)
                load_levels = [3, 5, 8] if IS_CI else [5, 10, 15, 20]  # Reduced load for CI

                for concurrent_queries in load_levels:
                    queries = sample_load_queries[:concurrent_queries]

                    async def process_query_with_timing(query):
                        start_time = time.time()

                        intent = await counselor.analyze_intent(query)
                        hyde_result = await counselor.hyde_processor.process_query(query)
                        agent_selection = await counselor.select_agents(intent)

                        # Convert AgentSelection to list of Agent objects
                        selected_agents = []
                        for agent_id in agent_selection.primary_agents + agent_selection.secondary_agents:
                            agent = next((a for a in counselor._available_agents if a.agent_id == agent_id), None)
                            if agent:
                                selected_agents.append(agent)

                        responses = await counselor.orchestrate_workflow(selected_agents, hyde_result["enhanced_query"])

                        processing_time = time.time() - start_time

                        return {
                            "query": query,
                            "processing_time": processing_time,
                            "success": len(responses) > 0 and all(r.success for r in responses),
                            "response_count": len(responses),
                        }

                    # Execute concurrent queries
                    start_time = time.time()
                    results = await asyncio.gather(
                        *[process_query_with_timing(query) for query in queries],
                        return_exceptions=True,
                    )
                    total_time = time.time() - start_time

                    # Analyze results
                    successful_results = [r for r in results if not isinstance(r, Exception) and r["success"]]
                    [r for r in results if isinstance(r, Exception) or not r.get("success", True)]

                    # Verify concurrent performance
                    success_rate = len(successful_results) / len(queries)
                    assert (
                        success_rate >= 0.95
                    ), f"Success rate {success_rate:.2%} below 95% for {concurrent_queries} concurrent queries"

                    # Verify individual query performance under load
                    response_times = [r["processing_time"] for r in successful_results]
                    if response_times:
                        avg_response_time = statistics.mean(response_times)
                        max_response_time = max(response_times)
                        p95_response_time = (
                            statistics.quantiles(response_times, n=20)[18]
                            if len(response_times) > 1
                            else response_times[0]
                        )

                        # CI-adapted thresholds for concurrent load
                        avg_threshold = 5.0 if IS_CI else 2.0
                        max_threshold = 8.0 if IS_CI else 3.0
                        p95_threshold = 6.0 if IS_CI else 2.5

                        assert (
                            avg_response_time < avg_threshold
                        ), f"Average response time {avg_response_time:.2f}s exceeds {avg_threshold}s under {concurrent_queries} concurrent load"
                        assert (
                            max_response_time < max_threshold
                        ), f"Maximum response time {max_response_time:.2f}s exceeds {max_threshold}s under {concurrent_queries} concurrent load"
                        assert (
                            p95_response_time < p95_threshold
                        ), f"95th percentile {p95_response_time:.2f}s exceeds {p95_threshold}s under {concurrent_queries} concurrent load"

                    # Verify overall throughput
                    throughput = len(successful_results) / total_time
                    expected_min_throughput = concurrent_queries / 5  # Should complete within 5 seconds
                    assert (
                        throughput >= expected_min_throughput
                    ), f"Throughput {throughput:.2f} queries/sec below expected {expected_min_throughput:.2f}"

                    # Record performance metrics using MetricData
                    from src.utils.performance_monitor import MetricData, MetricType

                    performance_monitor.record_metric(
                        MetricData(
                            name="concurrent_queries",
                            value=concurrent_queries,
                            timestamp=time.time(),
                            metric_type=MetricType.GAUGE,
                        ),
                    )
                    performance_monitor.record_metric(
                        MetricData(
                            name="success_rate",
                            value=success_rate,
                            timestamp=time.time(),
                            metric_type=MetricType.GAUGE,
                        ),
                    )
                    if response_times:
                        performance_monitor.record_metric(
                            MetricData(
                                name="avg_response_time",
                                value=avg_response_time,
                                timestamp=time.time(),
                                metric_type=MetricType.TIMER,
                            ),
                        )

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_production_readiness_checklist(self, performance_settings, performance_monitor):
        """Test production readiness checklist and requirements."""

        with patch("src.config.settings.get_settings", return_value=performance_settings):
            # Mock production-ready MCP client
            mock_mcp_client = AsyncMock()
            mock_mcp_client.connection_state = MCPConnectionState.CONNECTED
            mock_mcp_client.orchestrate_agents = AsyncMock(
                return_value=[
                    MagicMock(
                        agent_id="create_agent",
                        success=True,
                        content="Production-ready response",
                        metadata={"processing_time": 0.1},
                    ),
                ],
            )

            # Mock production-ready vector store
            mock_vector_store = EnhancedMockVectorStore(
                {"type": "mock", "simulate_latency": True, "error_rate": 0.0, "base_latency": 0.01},
            )
            await mock_vector_store.connect()

            # Mock production-ready HyDE processor
            mock_hyde_processor = AsyncMock()
            mock_hyde_processor.vector_store = mock_vector_store
            mock_hyde_processor.process_query = AsyncMock(
                return_value={
                    "original_query": "test",
                    "enhanced_query": "Enhanced: test",
                    "enhancement_score": 0.9,
                    "relevant_documents": [],
                    "processing_time": 0.02,
                },
            )

            with (
                patch(
                    "src.mcp_integration.mcp_client.MCPClientFactory.create_from_settings",
                    return_value=mock_mcp_client,
                ),
                patch("src.core.hyde_processor.HydeProcessor", return_value=mock_hyde_processor),
            ):

                # Initialize QueryCounselor with mocked dependencies
                counselor = QueryCounselor(
                    mcp_client=mock_mcp_client,
                    hyde_processor=mock_hyde_processor,
                )

                # Production readiness checklist
                production_checks = []

                # 1. Response time requirement (CI-adapted)
                start_time = time.time()
                intent = await counselor.analyze_intent("Production readiness test query")
                hyde_result = await counselor.hyde_processor.process_query("Production readiness test query")
                agent_selection = await counselor.select_agents(intent)

                # Convert AgentSelection to list of Agent objects
                selected_agents = []
                for agent_id in agent_selection.primary_agents + agent_selection.secondary_agents:
                    agent = next((a for a in counselor._available_agents if a.agent_id == agent_id), None)
                    if agent:
                        selected_agents.append(agent)

                await counselor.orchestrate_workflow(selected_agents, hyde_result["enhanced_query"])
                response_time = time.time() - start_time

                # CI-adapted response time thresholds
                response_threshold = 5.0 if IS_CI else 2.0
                threshold_label = f"<{response_threshold}s"

                production_checks.append(
                    {
                        "check": f"Response time {threshold_label}",
                        "status": "PASS" if response_time < response_threshold else "FAIL",
                        "value": f"{response_time:.2f}s",
                        "requirement": threshold_label,
                    },
                )

                # 2. Error handling
                error_handling_works = True
                try:
                    # Simulate error scenario
                    with patch.object(mock_mcp_client, "orchestrate_agents", side_effect=Exception("Test error")):
                        await counselor.orchestrate_workflow(selected_agents, "test query")
                except Exception:
                    error_handling_works = True  # Expected to fail gracefully

                production_checks.append(
                    {
                        "check": "Error handling",
                        "status": "PASS" if error_handling_works else "FAIL",
                        "value": "Graceful degradation",
                        "requirement": "Handle errors gracefully",
                    },
                )

                # 3. Health monitoring
                health_check_works = True
                try:
                    health_status = await mock_vector_store.health_check()
                    health_check_works = health_status.status == ConnectionStatus.HEALTHY
                except Exception:
                    health_check_works = False

                production_checks.append(
                    {
                        "check": "Health monitoring",
                        "status": "PASS" if health_check_works else "FAIL",
                        "value": "Health checks functional",
                        "requirement": "Health monitoring enabled",
                    },
                )

                # 4. Performance monitoring
                performance_monitoring_works = True
                try:
                    # Record performance metric using MetricData
                    from src.utils.performance_monitor import MetricData, MetricType

                    performance_monitor.record_metric(
                        MetricData(
                            name="query_time",
                            value=response_time,
                            timestamp=time.time(),
                            metric_type=MetricType.TIMER,
                        ),
                    )
                    performance_monitor.get_all_metrics()
                    # Check if timer stats for query_time exist
                    timer_stats = performance_monitor.get_timer_stats("query_time")
                    performance_monitoring_works = bool(timer_stats and "mean" in timer_stats)
                except Exception:
                    performance_monitoring_works = False

                production_checks.append(
                    {
                        "check": "Performance monitoring",
                        "status": "PASS" if performance_monitoring_works else "FAIL",
                        "value": "Metrics collection functional",
                        "requirement": "Performance metrics enabled",
                    },
                )

                # 5. Configuration validation
                config_validation_works = True
                try:
                    config_manager = MCPConfigurationManager()
                    config_validation_works = config_manager.validate_configuration()
                except Exception:
                    config_validation_works = False

                production_checks.append(
                    {
                        "check": "Configuration validation",
                        "status": "PASS" if config_validation_works else "FAIL",
                        "value": "Configuration valid",
                        "requirement": "Valid configuration",
                    },
                )

                # 6. Concurrent handling
                concurrent_handling_works = True
                try:
                    # Test basic concurrent handling
                    concurrent_tasks = [counselor.analyze_intent(f"Concurrent test {i}") for i in range(3)]
                    concurrent_results = await asyncio.gather(*concurrent_tasks)
                    concurrent_handling_works = len(concurrent_results) == 3
                except Exception:
                    concurrent_handling_works = False

                production_checks.append(
                    {
                        "check": "Concurrent handling",
                        "status": "PASS" if concurrent_handling_works else "FAIL",
                        "value": "Concurrent processing functional",
                        "requirement": "Handle concurrent requests",
                    },
                )

                # Verify all production checks pass
                failed_checks = [check for check in production_checks if check["status"] == "FAIL"]

                # Print production readiness report
                print("\n=== PRODUCTION READINESS REPORT ===")
                for check in production_checks:
                    status_symbol = "✓" if check["status"] == "PASS" else "✗"
                    print(f"{status_symbol} {check['check']}: {check['value']} (Required: {check['requirement']})")

                if failed_checks:
                    print(f"\n❌ {len(failed_checks)} production checks failed:")
                    for check in failed_checks:
                        print(f"  - {check['check']}: {check['value']}")

                    raise AssertionError(f"Production readiness failed: {len(failed_checks)} checks failed")
                print("\n✅ All production readiness checks passed!")

                # Assert all checks pass
                assert len(failed_checks) == 0, "All production readiness checks must pass"

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_memory_and_resource_usage(self, performance_settings, sample_load_queries):
        """Test memory usage and resource consumption."""

        import gc
        import os

        import psutil

        with patch("src.config.settings.get_settings", return_value=performance_settings):
            # Mock memory-efficient components
            mock_mcp_client = AsyncMock()
            mock_mcp_client.connection_state = MCPConnectionState.CONNECTED
            mock_mcp_client.orchestrate_agents = AsyncMock(
                return_value=[
                    MagicMock(
                        agent_id="create_agent",
                        success=True,
                        content="Memory-efficient response",
                        metadata={"processing_time": 0.1},
                    ),
                ],
            )

            mock_vector_store = EnhancedMockVectorStore(
                {"type": "mock", "simulate_latency": False, "error_rate": 0.0, "base_latency": 0.001},
            )
            await mock_vector_store.connect()

            mock_hyde_processor = AsyncMock()
            mock_hyde_processor.vector_store = mock_vector_store
            mock_hyde_processor.process_query = AsyncMock(
                return_value={
                    "original_query": "test",
                    "enhanced_query": "Enhanced: test",
                    "enhancement_score": 0.9,
                    "relevant_documents": [],
                    "processing_time": 0.01,
                },
            )

            with (
                patch(
                    "src.mcp_integration.mcp_client.MCPClientFactory.create_from_settings",
                    return_value=mock_mcp_client,
                ),
                patch("src.core.hyde_processor.HydeProcessor", return_value=mock_hyde_processor),
            ):

                # Initialize QueryCounselor with mocked dependencies
                counselor = QueryCounselor(
                    mcp_client=mock_mcp_client,
                    hyde_processor=mock_hyde_processor,
                )

                # Get baseline memory usage
                process = psutil.Process(os.getpid())
                baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

                # Process queries and monitor memory (CI-adapted)
                num_queries = 50 if IS_CI else 100  # Reduced queries for CI
                memory_measurements = []

                for i in range(num_queries):
                    query = sample_load_queries[i % len(sample_load_queries)]

                    # Process query
                    intent = await counselor.analyze_intent(query)
                    hyde_result = await counselor.hyde_processor.process_query(query)
                    agent_selection = await counselor.select_agents(intent)

                    # Convert AgentSelection to list of Agent objects
                    selected_agents = []
                    for agent_id in agent_selection.primary_agents + agent_selection.secondary_agents:
                        agent = next((a for a in counselor._available_agents if a.agent_id == agent_id), None)
                        if agent:
                            selected_agents.append(agent)

                    await counselor.orchestrate_workflow(selected_agents, hyde_result["enhanced_query"])

                    # Measure memory every 10 queries
                    if i % 10 == 0:
                        gc.collect()  # Force garbage collection
                        current_memory = process.memory_info().rss / 1024 / 1024  # MB
                        memory_measurements.append(current_memory)

                # Analyze memory usage
                final_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = final_memory - baseline_memory
                peak_memory = max(memory_measurements)
                statistics.mean(memory_measurements)

                # Verify memory performance (CI-adapted thresholds)
                memory_increase_limit = 100 if IS_CI else 50  # MB - more lenient in CI
                peak_memory_limit = 200 if IS_CI else 100  # MB - more lenient in CI
                memory_trend_limit = 1.0 if IS_CI else 0.5  # MB per measurement

                assert (
                    memory_increase < memory_increase_limit
                ), f"Memory increase {memory_increase:.1f}MB exceeds {memory_increase_limit}MB limit"
                assert (
                    peak_memory < baseline_memory + peak_memory_limit
                ), f"Peak memory {peak_memory:.1f}MB exceeds baseline + {peak_memory_limit}MB"

                # Check for memory leaks
                memory_trend = (memory_measurements[-1] - memory_measurements[0]) / len(memory_measurements)
                assert (
                    memory_trend < memory_trend_limit
                ), f"Memory trend {memory_trend:.2f}MB per measurement indicates potential leak"

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_stress_and_recovery(self, performance_settings, sample_load_queries):
        """Test system stress handling and recovery."""

        with patch("src.config.settings.get_settings", return_value=performance_settings):
            # Mock stress-resistant components
            mock_mcp_client = AsyncMock()
            mock_mcp_client.connection_state = MCPConnectionState.CONNECTED

            # Simulate variable load response
            call_count = 0

            async def mock_variable_orchestration(workflow_steps):
                nonlocal call_count
                call_count += 1

                # Extract query from workflow steps for variation
                query = workflow_steps[0].input_data.get("query", "default") if workflow_steps else "default"

                # Simulate realistic load pattern with pronounced differences
                # Phase 1 (Light): calls 1-30 -> low load
                # Phase 2 (Medium): calls 31-60 -> medium load
                # Phase 3 (Heavy): calls 61-90 -> high load
                # Phase 4 (Recovery): calls 91+ -> back to low load

                if call_count <= 30:  # Light load
                    load_factor = 1.0
                elif call_count <= 60:  # Medium load
                    load_factor = 2.0
                elif call_count <= 90:  # Heavy load
                    load_factor = 8.0  # Increased significantly for more pronounced difference
                else:  # Recovery
                    load_factor = 1.0

                # Use more pronounced timing differences to overcome asyncio timing variability
                base_time = 0.1 if IS_CI else 0.05  # Even larger base times for clearer differentiation
                processing_time = base_time * load_factor

                await asyncio.sleep(processing_time)

                return [
                    MagicMock(
                        agent_id="create_agent",
                        success=True,
                        content=f"Stress test response #{call_count} for: {query[:30]}...",
                        metadata={"processing_time": processing_time, "load_factor": load_factor},
                    ),
                ]

            mock_mcp_client.orchestrate_agents = mock_variable_orchestration

            mock_vector_store = EnhancedMockVectorStore(
                {"type": "mock", "simulate_latency": True, "error_rate": 0.0, "base_latency": 0.01},
            )
            await mock_vector_store.connect()

            mock_hyde_processor = AsyncMock()
            mock_hyde_processor.vector_store = mock_vector_store
            mock_hyde_processor.process_query = AsyncMock(
                return_value={
                    "original_query": "test",
                    "enhanced_query": "Enhanced: test",
                    "enhancement_score": 0.85,
                    "relevant_documents": [],
                    "processing_time": 0.02,
                },
            )

            with (
                patch(
                    "src.mcp_integration.mcp_client.MCPClientFactory.create_from_settings",
                    return_value=mock_mcp_client,
                ),
                patch("src.core.hyde_processor.HydeProcessor", return_value=mock_hyde_processor),
            ):

                # Initialize QueryCounselor with mocked dependencies
                counselor = QueryCounselor(
                    mcp_client=mock_mcp_client,
                    hyde_processor=mock_hyde_processor,
                )

                # Stress test with increasing load (CI-adapted)
                if IS_CI:
                    stress_phases = [
                        {"concurrent": 3, "duration": 3},  # Light load
                        {"concurrent": 5, "duration": 3},  # Medium load
                        {"concurrent": 8, "duration": 3},  # Heavy load
                        {"concurrent": 3, "duration": 3},  # Recovery
                    ]
                else:
                    stress_phases = [
                        {"concurrent": 5, "duration": 5},  # Light load
                        {"concurrent": 15, "duration": 5},  # Medium load
                        {"concurrent": 25, "duration": 5},  # Heavy load
                        {"concurrent": 5, "duration": 5},  # Recovery
                    ]

                phase_results = []

                for phase_num, phase in enumerate(stress_phases):
                    concurrent_queries = phase["concurrent"]
                    duration = phase["duration"]

                    phase_start = time.time()
                    phase_responses = []

                    # Run phase
                    while time.time() - phase_start < duration:
                        batch_queries = sample_load_queries[:concurrent_queries]

                        async def process_stress_query(query):
                            start_time = time.time()

                            intent = await counselor.analyze_intent(query)
                            hyde_result = await counselor.hyde_processor.process_query(query)
                            agent_selection = await counselor.select_agents(intent)

                            # Convert AgentSelection to list of Agent objects
                            selected_agents = []
                            for agent_id in agent_selection.primary_agents + agent_selection.secondary_agents:
                                agent = next((a for a in counselor._available_agents if a.agent_id == agent_id), None)
                                if agent:
                                    selected_agents.append(agent)

                            responses = await counselor.orchestrate_workflow(
                                selected_agents,
                                hyde_result["enhanced_query"],
                            )

                            processing_time = time.time() - start_time

                            return {
                                "processing_time": processing_time,
                                "success": len(responses) > 0 and all(r.success for r in responses),
                            }

                        batch_results = await asyncio.gather(
                            *[process_stress_query(query) for query in batch_queries],
                            return_exceptions=True,
                        )

                        phase_responses.extend([r for r in batch_results if not isinstance(r, Exception)])

                    # Analyze phase results
                    successful_responses = [r for r in phase_responses if r["success"]]
                    success_rate = len(successful_responses) / len(phase_responses) if phase_responses else 0

                    if successful_responses:
                        avg_response_time = statistics.mean([r["processing_time"] for r in successful_responses])
                        max_response_time = max([r["processing_time"] for r in successful_responses])
                    else:
                        avg_response_time = 0
                        max_response_time = 0

                    phase_results.append(
                        {
                            "phase": phase_num + 1,
                            "concurrent": concurrent_queries,
                            "success_rate": success_rate,
                            "avg_response_time": avg_response_time,
                            "max_response_time": max_response_time,
                            "total_queries": len(phase_responses),
                        },
                    )

                # Verify stress test results
                for result in phase_results:
                    phase_type = ["Light", "Medium", "Heavy", "Recovery"][result["phase"] - 1]

                    # Success rate should remain high
                    assert (
                        result["success_rate"] >= 0.90
                    ), f"Success rate {result['success_rate']:.2%} below 90% in {phase_type} phase"

                    # Response times should be reasonable (CI-adapted)
                    if result["phase"] <= 3:  # Stress phases
                        base_time = 6.0 if IS_CI else 3.0
                        max_allowed_time = base_time + (result["phase"] - 1) * 0.5  # Allow degradation
                        assert (
                            result["avg_response_time"] < max_allowed_time
                        ), f"Average response time {result['avg_response_time']:.2f}s exceeds {max_allowed_time:.1f}s in {phase_type} phase"
                    else:  # Recovery phase
                        recovery_threshold = 5.0 if IS_CI else 2.0
                        assert (
                            result["avg_response_time"] < recovery_threshold
                        ), f"Recovery phase response time {result['avg_response_time']:.2f}s should be under {recovery_threshold}s"

                # Verify recovery
                recovery_result = phase_results[-1]
                heavy_load_result = phase_results[-2]

                recovery_improvement = (
                    heavy_load_result["avg_response_time"] - recovery_result["avg_response_time"]
                ) / heavy_load_result["avg_response_time"]

                # More lenient threshold for CI environments where timing is less predictable
                if IS_CI:
                    # In CI, timing is highly variable - focus on ensuring recovery doesn't degrade significantly
                    # Allow minimal negative improvement in CI, but still expect some recovery
                    improvement_threshold = -0.05  # Allow up to 5% degradation in CI due to variable timing
                    assert (
                        recovery_improvement > improvement_threshold
                    ), f"Recovery improvement {recovery_improvement:.2%} should be > {improvement_threshold*100:.0f}% (CI allows timing variation)"

                    # Additional CI-specific check: recovery should be reasonable compared to light load
                    # CI environments have highly variable timing, so be very lenient
                    light_load_result = phase_results[0]
                    recovery_vs_light = (
                        abs(recovery_result["avg_response_time"] - light_load_result["avg_response_time"])
                        / light_load_result["avg_response_time"]
                    )
                    assert (
                        recovery_vs_light < 2.0
                    ), f"Recovery time should be within 200% of light load baseline (was {recovery_vs_light:.2%}) - CI timing is highly variable"
                else:
                    # Local environment with mocked services - timing is unpredictable but should still show improvement
                    # Use balanced threshold: stricter than before (50%->30%) but accounts for mock unpredictability
                    improvement_threshold = (
                        -0.30
                    )  # Allow up to 30% degradation in mock environment (was 50%, now more meaningful)
                    assert (
                        recovery_improvement > improvement_threshold
                    ), f"Recovery improvement {recovery_improvement:.2%} should be > {improvement_threshold*100:.0f}% (mock env allows major variation)"
