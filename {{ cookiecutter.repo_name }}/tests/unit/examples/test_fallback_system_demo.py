"""
Comprehensive test suite for fallback_system_demo.py

Tests all functionality including:
- DemoTaskDetectionSystem behavior
- Normal operation demonstrations
- Failure scenario handling
- Circuit breaker protection
- Emergency mode activation
- Integration modes
- Health monitoring
- Performance impact measurement
"""

import asyncio
import contextlib
import logging
import time
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from typing import Any

import pytest

from examples.fallback_system_demo import (
    DemoTaskDetectionSystem,
    demonstrate_circuit_breaker,
    demonstrate_emergency_mode,
    demonstrate_failure_scenarios,
    demonstrate_health_monitoring,
    demonstrate_integration_modes,
    demonstrate_normal_operation,
    demonstrate_performance_impact,
    main,
)
from src.core.fallback_integration import IntegrationMode
from src.core.task_detection import DetectionResult
from src.core.task_detection_config import DetectionMode


class TestDemoTaskDetectionSystem:
    """Test the DemoTaskDetectionSystem class."""

    def test_initialization(self):
        """Test proper initialization of DemoTaskDetectionSystem."""
        system = DemoTaskDetectionSystem()
        
        assert system.failure_mode is None
        assert system.call_count == 0
        assert system.response_delay == 0.05

    def test_set_failure_mode(self):
        """Test setting different failure modes."""
        system = DemoTaskDetectionSystem()
        
        # Test setting failure modes
        system.set_failure_mode("timeout")
        assert system.failure_mode == "timeout"
        
        system.set_failure_mode("network")
        assert system.failure_mode == "network"
        
        system.set_failure_mode(None)
        assert system.failure_mode is None

    def test_set_response_delay(self):
        """Test setting response delay."""
        system = DemoTaskDetectionSystem()
        
        system.set_response_delay(0.1)
        assert system.response_delay == 0.1
        
        system.set_response_delay(0.001)
        assert system.response_delay == 0.001

    @pytest.mark.asyncio
    async def test_detect_categories_normal_operation(self):
        """Test normal detection without failures."""
        system = DemoTaskDetectionSystem()
        
        # Test git query
        result = await system.detect_categories("git commit changes")
        assert isinstance(result, DetectionResult)
        assert result.categories["git"] is True
        assert result.categories["core"] is True
        assert result.confidence_scores["git"] == 0.9
        assert system.call_count == 1

        # Test test query
        result = await system.detect_categories("run test suite")
        assert result.categories["test"] is True
        assert result.categories["quality"] is True
        assert result.confidence_scores["test"] == 0.8
        assert system.call_count == 2

        # Test debug query
        result = await system.detect_categories("debug this error")
        assert result.categories["debug"] is True
        assert result.categories["analysis"] is True
        assert result.confidence_scores["debug"] == 0.85
        assert system.call_count == 3

        # Test ambiguous query
        result = await system.detect_categories("help me with something")
        assert result.categories["analysis"] is True
        assert result.categories["core"] is False
        assert result.confidence_scores["analysis"] == 0.4
        assert system.call_count == 4

    @pytest.mark.asyncio
    async def test_detect_categories_timeout_failure(self):
        """Test timeout failure mode."""
        system = DemoTaskDetectionSystem()
        system.set_failure_mode("timeout")
        
        # Should take a very long time (10 seconds)
        start_time = time.time()
        
        # Use asyncio.wait_for to prevent test hanging
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                system.detect_categories("test query"), 
                timeout=0.1
            )

    @pytest.mark.asyncio
    async def test_detect_categories_network_failure(self):
        """Test network failure mode."""
        system = DemoTaskDetectionSystem()
        system.set_failure_mode("network")
        
        with pytest.raises(ConnectionError, match="Simulated network failure"):
            await system.detect_categories("test query")

    @pytest.mark.asyncio
    async def test_detect_categories_memory_failure(self):
        """Test memory failure mode."""
        system = DemoTaskDetectionSystem()
        system.set_failure_mode("memory")
        
        with pytest.raises(MemoryError, match="Simulated memory pressure"):
            await system.detect_categories("test query")

    @pytest.mark.asyncio
    async def test_detect_categories_generic_failure(self):
        """Test generic failure mode."""
        system = DemoTaskDetectionSystem()
        system.set_failure_mode("generic")
        
        with pytest.raises(Exception, match="Simulated generic error"):
            await system.detect_categories("test query")

    @pytest.mark.asyncio
    async def test_response_delay_timing(self):
        """Test that response delay is respected."""
        system = DemoTaskDetectionSystem()
        system.set_response_delay(0.1)  # 100ms
        
        start_time = time.time()
        await system.detect_categories("test query")
        elapsed_time = time.time() - start_time
        
        # Should take at least 100ms
        assert elapsed_time >= 0.1
        assert elapsed_time < 0.2  # But not too much longer

    @pytest.mark.asyncio
    async def test_call_count_tracking(self):
        """Test that call count is properly tracked."""
        system = DemoTaskDetectionSystem()
        
        assert system.call_count == 0
        
        await system.detect_categories("query 1")
        assert system.call_count == 1
        
        await system.detect_categories("query 2")
        assert system.call_count == 2
        
        # Even failed calls should increment counter
        system.set_failure_mode("network")
        with pytest.raises(ConnectionError):
            await system.detect_categories("query 3")
        assert system.call_count == 3


class TestDemonstrationFunctions:
    """Test the demonstration functions."""

    @pytest.mark.asyncio
    @patch("examples.fallback_system_demo.create_conservative_fallback_chain")
    async def test_demonstrate_normal_operation(self, mock_create_chain):
        """Test normal operation demonstration."""
        # Mock the fallback chain
        mock_chain = Mock()
        mock_chain.get_function_categories = AsyncMock(
            return_value=({"git": True, "core": True}, {"strategy": "optimized"})
        )
        mock_create_chain.return_value = mock_chain

        # Should run without exceptions
        await demonstrate_normal_operation()
        
        # Verify chain was created and called
        assert mock_create_chain.called
        assert mock_chain.get_function_categories.call_count == 5  # 5 test queries

    @pytest.mark.asyncio
    @patch("examples.fallback_system_demo.create_conservative_fallback_chain")
    async def test_demonstrate_failure_scenarios(self, mock_create_chain):
        """Test failure scenarios demonstration."""
        mock_chain = Mock()
        mock_chain.get_function_categories = AsyncMock(
            return_value=({"debug": True}, {"strategy": "fallback_level_1"})
        )
        mock_create_chain.return_value = mock_chain

        # Should run without exceptions
        await demonstrate_failure_scenarios()
        
        # Verify chain was created and called for each failure scenario
        assert mock_create_chain.called
        assert mock_chain.get_function_categories.call_count == 4  # 4 failure modes

    @pytest.mark.asyncio
    @patch("examples.fallback_system_demo.create_conservative_fallback_chain")
    async def test_demonstrate_circuit_breaker(self, mock_create_chain):
        """Test circuit breaker demonstration."""
        mock_chain = Mock()
        mock_chain.get_function_categories = AsyncMock(
            return_value=({"core": True}, {"strategy": "emergency"})
        )
        mock_chain.reset_circuit_breaker = Mock()
        mock_create_chain.return_value = mock_chain

        # Should run without exceptions
        await demonstrate_circuit_breaker()
        
        # Verify chain was created and circuit breaker was reset
        assert mock_create_chain.called
        assert mock_chain.reset_circuit_breaker.called
        # Should have at least 6 calls (5 failures + 1 after reset)
        assert mock_chain.get_function_categories.call_count >= 6

    @pytest.mark.asyncio
    @patch("examples.fallback_system_demo.create_conservative_fallback_chain")
    async def test_demonstrate_emergency_mode(self, mock_create_chain):
        """Test emergency mode demonstration."""
        mock_chain = Mock()
        mock_chain.get_function_categories = AsyncMock(
            return_value=({"core": True}, {"strategy": "emergency"})
        )
        mock_chain.exit_emergency_mode = Mock()
        mock_create_chain.return_value = mock_chain

        # Should run without exceptions
        await demonstrate_emergency_mode()
        
        # Verify emergency mode was set and exited
        assert mock_create_chain.called
        assert mock_chain.emergency_mode is True
        assert hasattr(mock_chain, 'emergency_mode_start')
        assert mock_chain.exit_emergency_mode.called

    @pytest.mark.asyncio
    @patch("examples.fallback_system_demo.create_enhanced_task_detection")
    async def test_demonstrate_integration_modes(self, mock_create_enhanced):
        """Test integration modes demonstration."""
        mock_enhanced_system = Mock()
        mock_enhanced_system.detect_categories = AsyncMock(
            return_value=DetectionResult(
                categories={"git": True},
                confidence_scores={"git": 0.9},
                detection_time_ms=50.0,
                signals_used={},
                fallback_applied=None,
            )
        )
        mock_create_enhanced.return_value = mock_enhanced_system

        # Should run without exceptions
        await demonstrate_integration_modes()
        
        # Verify enhanced system was created for each integration mode
        assert mock_create_enhanced.call_count == 3  # 3 integration modes
        assert mock_enhanced_system.detect_categories.call_count == 3

        # Verify all integration modes were tested
        calls = mock_create_enhanced.call_args_list
        modes_tested = [call[1]['integration_mode'] for call in calls]
        assert IntegrationMode.MONITORING in modes_tested
        assert IntegrationMode.SHADOW in modes_tested
        assert IntegrationMode.ACTIVE in modes_tested

    @pytest.mark.asyncio
    @patch("examples.fallback_system_demo.create_conservative_fallback_chain")
    async def test_demonstrate_health_monitoring(self, mock_create_chain):
        """Test health monitoring demonstration."""
        mock_chain = Mock()
        mock_chain.get_function_categories = AsyncMock(
            return_value=({"core": True}, {"strategy": "optimized"})
        )
        mock_chain.get_health_status = Mock(return_value={
            "performance": {"avg_response_time": 0.05, "success_rate": 0.95},
            "recovery": {"failures_handled": 3, "recovery_time": 0.1},
            "learning": {"insights": ["Insight 1", "Insight 2", "Insight 3"]}
        })
        mock_create_chain.return_value = mock_chain

        # Should run without exceptions
        await demonstrate_health_monitoring()
        
        # Verify health monitoring was called
        assert mock_create_chain.called
        assert mock_chain.get_health_status.called
        # Should have 10 calls (one for each iteration)
        assert mock_chain.get_function_categories.call_count == 10

    @pytest.mark.asyncio
    @patch("examples.fallback_system_demo.create_conservative_fallback_chain")
    async def test_demonstrate_performance_impact(self, mock_create_chain):
        """Test performance impact demonstration."""
        mock_chain = Mock()
        mock_chain.get_function_categories = AsyncMock(
            return_value=({"core": True}, {"strategy": "optimized"})
        )
        mock_create_chain.return_value = mock_chain

        # Should run without exceptions
        await demonstrate_performance_impact()
        
        # Verify performance tests were run
        assert mock_create_chain.called
        # Should have 110 calls (100 with chain + 10 under failure)
        assert mock_chain.get_function_categories.call_count == 110

    @pytest.mark.asyncio
    @patch("examples.fallback_system_demo.demonstrate_normal_operation")
    @patch("examples.fallback_system_demo.demonstrate_failure_scenarios")
    @patch("examples.fallback_system_demo.demonstrate_circuit_breaker")
    @patch("examples.fallback_system_demo.demonstrate_emergency_mode")
    @patch("examples.fallback_system_demo.demonstrate_integration_modes")
    @patch("examples.fallback_system_demo.demonstrate_health_monitoring")
    @patch("examples.fallback_system_demo.demonstrate_performance_impact")
    async def test_main_function(
        self,
        mock_performance,
        mock_health,
        mock_integration,
        mock_emergency,
        mock_circuit,
        mock_failure,
        mock_normal,
    ):
        """Test the main demonstration function."""
        # Mock all demonstration functions
        mock_normal.return_value = None
        mock_failure.return_value = None
        mock_circuit.return_value = None
        mock_emergency.return_value = None
        mock_integration.return_value = None
        mock_health.return_value = None
        mock_performance.return_value = None

        # Should run without exceptions
        await main()
        
        # Verify all demonstrations were called
        assert mock_normal.called
        assert mock_failure.called
        assert mock_circuit.called
        assert mock_emergency.called
        assert mock_integration.called
        assert mock_health.called
        assert mock_performance.called

    @pytest.mark.asyncio
    @patch("examples.fallback_system_demo.demonstrate_normal_operation")
    async def test_main_function_with_exception(self, mock_normal):
        """Test main function handles exceptions gracefully."""
        # Make one demonstration raise an exception
        mock_normal.side_effect = Exception("Test exception")

        # Should not raise exception (it's caught and printed)
        await main()
        
        # Exception should have been caught
        assert mock_normal.called


class TestIntegrationScenarios:
    """Test integration scenarios with real components."""

    @pytest.mark.asyncio
    async def test_demo_system_integration_with_real_detection_result(self):
        """Test demo system produces valid DetectionResult objects."""
        system = DemoTaskDetectionSystem()
        
        result = await system.detect_categories("git commit changes")
        
        # Validate DetectionResult structure
        assert isinstance(result, DetectionResult)
        assert isinstance(result.categories, dict)
        assert isinstance(result.confidence_scores, dict)
        assert isinstance(result.detection_time_ms, float)
        assert isinstance(result.signals_used, dict)
        assert result.fallback_applied is None

        # Validate content makes sense
        assert len(result.categories) > 0
        assert len(result.confidence_scores) > 0
        assert result.detection_time_ms > 0
        assert all(isinstance(v, bool) for v in result.categories.values())
        assert all(isinstance(v, (int, float)) for v in result.confidence_scores.values())

    @pytest.mark.asyncio
    async def test_system_behavior_consistency(self):
        """Test that system behavior is consistent across calls."""
        system = DemoTaskDetectionSystem()
        
        # Same query should produce same results
        result1 = await system.detect_categories("git commit changes")
        result2 = await system.detect_categories("git commit changes")
        
        assert result1.categories == result2.categories
        assert result1.confidence_scores == result2.confidence_scores
        # Detection time may vary slightly, but should be similar
        assert abs(result1.detection_time_ms - result2.detection_time_ms) < 10

    @pytest.mark.asyncio
    async def test_failure_mode_isolation(self):
        """Test that failure modes don't affect each other."""
        system = DemoTaskDetectionSystem()
        
        # Test normal operation
        result = await system.detect_categories("test query")
        assert isinstance(result, DetectionResult)
        
        # Set failure mode and verify it fails
        system.set_failure_mode("network")
        with pytest.raises(ConnectionError):
            await system.detect_categories("test query")
        
        # Reset and verify normal operation returns
        system.set_failure_mode(None)
        result = await system.detect_categories("test query")
        assert isinstance(result, DetectionResult)

    @pytest.mark.asyncio
    async def test_concurrent_detection_calls(self):
        """Test system handles concurrent calls correctly."""
        system = DemoTaskDetectionSystem()
        
        # Make multiple concurrent calls
        tasks = [
            system.detect_categories("git commit"),
            system.detect_categories("run tests"),
            system.detect_categories("debug error"),
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All should succeed and be different
        assert len(results) == 3
        assert all(isinstance(r, DetectionResult) for r in results)
        assert results[0].categories != results[1].categories  # Different queries -> different results
        assert system.call_count == 3


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_empty_query_handling(self):
        """Test handling of empty queries."""
        system = DemoTaskDetectionSystem()
        
        result = await system.detect_categories("")
        assert isinstance(result, DetectionResult)
        # Should fall back to default (ambiguous) result
        assert result.categories["analysis"] is True
        assert result.confidence_scores["analysis"] == 0.4

    @pytest.mark.asyncio
    async def test_none_query_handling(self):
        """Test handling of None query."""
        system = DemoTaskDetectionSystem()
        
        # None query should raise AttributeError (expected behavior)
        with pytest.raises(AttributeError, match="'NoneType' object has no attribute 'lower'"):
            await system.detect_categories(None)

    @pytest.mark.asyncio
    async def test_unicode_query_handling(self):
        """Test handling of unicode queries."""
        system = DemoTaskDetectionSystem()
        
        result = await system.detect_categories("git commit 变更")
        assert isinstance(result, DetectionResult)
        # Should still detect git keywords
        assert result.categories["git"] is True

    @pytest.mark.asyncio
    async def test_very_long_query(self):
        """Test handling of very long queries."""
        system = DemoTaskDetectionSystem()
        
        long_query = "git " + "x" * 10000  # Very long query with git keyword
        result = await system.detect_categories(long_query)
        assert isinstance(result, DetectionResult)
        # Should still detect git keywords
        assert result.categories["git"] is True

    @pytest.mark.asyncio
    async def test_case_insensitive_detection(self):
        """Test that detection is case insensitive."""
        system = DemoTaskDetectionSystem()
        
        queries = ["git commit", "GIT COMMIT", "Git Commit", "gIt CoMmIt"]
        results = []
        
        for query in queries:
            result = await system.detect_categories(query)
            results.append(result)
        
        # All should detect git
        for result in results:
            assert result.categories["git"] is True
            assert result.confidence_scores["git"] == 0.9

    def test_response_delay_bounds(self):
        """Test response delay validation."""
        system = DemoTaskDetectionSystem()
        
        # Test various delay values
        system.set_response_delay(0.0)
        assert system.response_delay == 0.0
        
        system.set_response_delay(0.001)
        assert system.response_delay == 0.001
        
        system.set_response_delay(10.0)
        assert system.response_delay == 10.0
        
        # Negative delays should be allowed (implementation choice)
        system.set_response_delay(-1.0)
        assert system.response_delay == -1.0


class TestSecurityAndSafety:
    """Test security and safety aspects."""

    @pytest.mark.asyncio
    async def test_failure_mode_validation(self):
        """Test that only expected failure modes are handled."""
        system = DemoTaskDetectionSystem()
        
        # Valid failure modes should work
        valid_modes = ["timeout", "network", "memory", "generic", None]
        
        for mode in valid_modes:
            system.set_failure_mode(mode)
            assert system.failure_mode == mode
        
        # Invalid failure modes should not cause issues
        system.set_failure_mode("invalid_mode")
        assert system.failure_mode == "invalid_mode"
        
        # Should still work normally (no failure triggered)
        result = await system.detect_categories("test query")
        assert isinstance(result, DetectionResult)

    @pytest.mark.asyncio
    async def test_context_parameter_safety(self):
        """Test that context parameter is handled safely."""
        system = DemoTaskDetectionSystem()
        
        # Test with None context
        result = await system.detect_categories("test query", None)
        assert isinstance(result, DetectionResult)
        
        # Test with empty context
        result = await system.detect_categories("test query", {})
        assert isinstance(result, DetectionResult)
        
        # Test with complex context
        complex_context = {
            "user_id": "test_user",
            "session_id": 12345,
            "nested": {"data": ["a", "b", "c"]},
            "unicode": "测试数据",
        }
        result = await system.detect_categories("test query", complex_context)
        assert isinstance(result, DetectionResult)

    @pytest.mark.asyncio
    async def test_exception_safety(self):
        """Test that exceptions don't leave system in bad state."""
        system = DemoTaskDetectionSystem()
        
        # Set failure mode and trigger exception
        system.set_failure_mode("network")
        initial_call_count = system.call_count
        
        with pytest.raises(ConnectionError):
            await system.detect_categories("test query")
        
        # Call count should still be incremented
        assert system.call_count == initial_call_count + 1
        
        # Reset and verify system still works
        system.set_failure_mode(None)
        result = await system.detect_categories("test query")
        assert isinstance(result, DetectionResult)
        assert system.call_count == initial_call_count + 2

    @pytest.mark.asyncio 
    async def test_timeout_behavior_safety(self):
        """Test that timeout behavior doesn't cause resource leaks."""
        system = DemoTaskDetectionSystem()
        system.set_failure_mode("timeout")
        
        # Create multiple timeout tasks
        tasks = []
        for i in range(3):
            task = asyncio.create_task(
                asyncio.wait_for(
                    system.detect_categories(f"query {i}"),
                    timeout=0.01
                )
            )
            tasks.append(task)
        
        # All should timeout
        results = await asyncio.gather(*tasks, return_exceptions=True)
        assert all(isinstance(r, asyncio.TimeoutError) for r in results)
        
        # Reset and verify system still works
        system.set_failure_mode(None)
        result = await system.detect_categories("test query")
        assert isinstance(result, DetectionResult)


class TestPerformanceCharacteristics:
    """Test performance characteristics of the demo system."""

    @pytest.mark.asyncio
    async def test_response_time_accuracy(self):
        """Test that response delay is accurate within reasonable bounds."""
        system = DemoTaskDetectionSystem()
        delays_to_test = [0.01, 0.05, 0.1]
        
        for delay in delays_to_test:
            system.set_response_delay(delay)
            
            start_time = time.time()
            await system.detect_categories("test query")
            actual_time = time.time() - start_time
            
            # Should be within 50% of target (accounting for system overhead)
            assert actual_time >= delay * 0.8
            assert actual_time <= delay * 1.5

    @pytest.mark.asyncio
    async def test_concurrent_performance(self):
        """Test performance under concurrent load."""
        system = DemoTaskDetectionSystem()
        system.set_response_delay(0.01)  # 10ms
        
        # Run 10 concurrent calls
        start_time = time.time()
        tasks = [
            system.detect_categories(f"query {i}")
            for i in range(10)
        ]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # All should succeed
        assert len(results) == 10
        assert all(isinstance(r, DetectionResult) for r in results)
        
        # Should take about 10ms (concurrent), not 100ms (sequential)
        assert total_time < 0.1  # Less than 100ms for 10 concurrent 10ms calls
        assert system.call_count == 10

    @pytest.mark.asyncio
    async def test_memory_usage_stability(self):
        """Test that repeated calls don't accumulate memory."""
        system = DemoTaskDetectionSystem()
        system.set_response_delay(0.001)  # Very fast
        
        # Run many calls
        for i in range(1000):
            result = await system.detect_categories(f"query {i}")
            assert isinstance(result, DetectionResult)
        
        assert system.call_count == 1000
        
        # Verify system is still responsive
        result = await system.detect_categories("final test")
        assert isinstance(result, DetectionResult)
        assert system.call_count == 1001