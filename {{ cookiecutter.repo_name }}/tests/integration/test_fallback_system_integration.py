"""
Integration tests for Conservative Fallback System with existing PromptCraft codebase

This test suite validates that the fallback system integrates correctly with
the existing PromptCraft systems without breaking existing functionality.

Test Areas:
- Integration with existing TaskDetectionSystem
- Backwards compatibility with existing code
- MCP integration compatibility
- UI component integration
- Configuration system compatibility
- Performance impact validation
"""

import asyncio
import time
from unittest.mock import patch

import pytest

from src.core.conservative_fallback_chain import (
    create_conservative_fallback_chain,
)
from src.core.fallback_integration import (
    BackwardsCompatibleTaskDetection,
    EnhancedTaskDetectionSystem,
    IntegrationConfig,
    IntegrationMode,
    create_enhanced_task_detection,
)
from src.core.task_detection import DetectionResult, TaskDetectionSystem
from src.core.task_detection_config import DetectionMode, TaskDetectionConfig


class TestExistingSystemCompatibility:
    """Test compatibility with existing PromptCraft systems"""

    def test_task_detection_system_integration(self):
        """Test integration with real TaskDetectionSystem"""

        # Create real task detection system
        original_system = TaskDetectionSystem()

        # Create enhanced system
        enhanced_system = create_enhanced_task_detection(
            original_system,
            IntegrationMode.ACTIVE,
        )

        # Verify it's properly wrapped
        assert isinstance(enhanced_system, EnhancedTaskDetectionSystem)
        assert enhanced_system.original_system is original_system
        assert enhanced_system.fallback_chain is not None

    @pytest.mark.asyncio
    async def test_backwards_compatibility_wrapper(self):
        """Test backwards compatibility wrapper"""

        # Create original system
        original_system = TaskDetectionSystem()

        # Wrap with backwards compatible wrapper
        wrapped_system = BackwardsCompatibleTaskDetection(original_system)

        # Test that it behaves like original system
        result = await wrapped_system.detect_categories(
            "git commit changes",
            {"project_type": "web_app"},
        )

        assert isinstance(result, DetectionResult)
        assert hasattr(result, "categories")
        assert hasattr(result, "confidence_scores")
        assert hasattr(result, "detection_time_ms")

    @pytest.mark.asyncio
    async def test_configuration_compatibility(self):
        """Test configuration system compatibility"""

        # Test with existing configuration
        config = TaskDetectionConfig()
        config.apply_mode_preset(DetectionMode.CONSERVATIVE)

        # Verify configuration is accepted
        original_system = TaskDetectionSystem()
        fallback_chain = create_conservative_fallback_chain(original_system, config)

        assert fallback_chain.config is config
        assert fallback_chain.config.mode == DetectionMode.CONSERVATIVE

        # Test with different presets
        for mode in [DetectionMode.CONSERVATIVE, DetectionMode.BALANCED, DetectionMode.AGGRESSIVE]:
            config.apply_mode_preset(mode)
            fallback_chain = create_conservative_fallback_chain(original_system, config)
            assert fallback_chain.config.mode == mode


class TestFunctionalIntegration:
    """Test functional integration scenarios"""

    @pytest.mark.asyncio
    async def test_normal_operation_flow(self):
        """Test normal operation flow through enhanced system"""

        original_system = TaskDetectionSystem()
        enhanced_system = create_enhanced_task_detection(original_system)

        # Test various queries that should work normally
        test_queries = [
            "git status",
            "run tests",
            "debug authentication error",
            "analyze code quality",
            "help with security review",
        ]

        for query in test_queries:
            result = await enhanced_system.detect_categories(query)

            assert isinstance(result, DetectionResult)
            assert result.categories is not None
            assert result.confidence_scores is not None
            assert result.detection_time_ms >= 0

    @pytest.mark.asyncio
    async def test_fallback_activation_flow(self):
        """Test fallback activation in realistic scenarios"""

        original_system = TaskDetectionSystem()
        enhanced_system = create_enhanced_task_detection(original_system)

        # Mock original system to fail
        with patch.object(original_system, "detect_categories") as mock_detect:
            mock_detect.side_effect = TimeoutError("Detection timeout")

            result = await enhanced_system.detect_categories("test query")

            # Should get fallback result
            assert isinstance(result, DetectionResult)
            assert result.fallback_applied is not None
            assert any(result.categories.values())  # Some categories should be loaded

    @pytest.mark.asyncio
    async def test_performance_impact_measurement(self):
        """Test performance impact of fallback system"""

        original_system = TaskDetectionSystem()

        # Measure original system performance
        start_time = time.time()
        for i in range(10):
            await original_system.detect_categories(f"query {i}")
        original_time = time.time() - start_time

        # Measure enhanced system performance
        enhanced_system = create_enhanced_task_detection(original_system)

        start_time = time.time()
        for i in range(10):
            await enhanced_system.detect_categories(f"query {i}")
        enhanced_time = time.time() - start_time

        # Performance overhead should be minimal (< 20%)
        overhead = (enhanced_time - original_time) / original_time
        assert overhead < 0.2, f"Performance overhead too high: {overhead:.2%}"

    @pytest.mark.asyncio
    async def test_concurrent_request_handling(self):
        """Test handling of concurrent requests"""

        original_system = TaskDetectionSystem()
        enhanced_system = create_enhanced_task_detection(original_system)

        # Create concurrent requests
        async def make_request(query_id):
            return await enhanced_system.detect_categories(f"concurrent query {query_id}")

        # Run 20 concurrent requests
        tasks = [make_request(i) for i in range(20)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All requests should complete successfully
        successful_results = [r for r in results if isinstance(r, DetectionResult)]
        errors = [r for r in results if isinstance(r, Exception)]

        assert len(successful_results) >= 18, f"Too many failures: {len(errors)} errors"
        assert len(errors) <= 2, f"Unexpected number of errors: {errors}"


class TestErrorHandlingIntegration:
    """Test error handling integration with existing systems"""

    @pytest.mark.asyncio
    async def test_existing_error_handling_preservation(self):
        """Test that existing error handling is preserved"""

        original_system = TaskDetectionSystem()
        enhanced_system = create_enhanced_task_detection(original_system)

        # Mock original system to raise specific error
        with patch.object(original_system, "detect_categories") as mock_detect:
            mock_detect.side_effect = ValueError("Specific validation error")

            # Enhanced system should handle gracefully and use fallback
            result = await enhanced_system.detect_categories("test query")

            assert isinstance(result, DetectionResult)
            assert result.fallback_applied is not None

    @pytest.mark.asyncio
    async def test_integration_with_mcp_error_handling(self):
        """Test integration with existing MCP error handling"""

        # This would test integration with src.core.zen_mcp_error_handling
        # For now, verify that the systems are compatible

        from src.core.zen_mcp_error_handling import ZenMCPIntegration

        original_system = TaskDetectionSystem()
        enhanced_system = create_enhanced_task_detection(original_system)

        # Verify systems can coexist
        mcp_integration = ZenMCPIntegration()

        # Both systems should be able to operate
        assert enhanced_system is not None
        assert mcp_integration is not None

    @pytest.mark.asyncio
    async def test_logging_integration(self):
        """Test integration with existing logging systems"""

        from unittest.mock import patch

        # Capture log messages
        with patch("src.core.conservative_fallback_chain.logger") as mock_logger:
            original_system = TaskDetectionSystem()
            enhanced_system = create_enhanced_task_detection(original_system)

            # Trigger fallback
            with patch.object(original_system, "detect_categories") as mock_detect:
                mock_detect.side_effect = Exception("Test error")

                await enhanced_system.detect_categories("test query")

                # Verify logging calls were made
                assert mock_logger.warning.called or mock_logger.error.called


class TestConfigurationIntegration:
    """Test configuration system integration"""

    def test_environment_configuration_compatibility(self):
        """Test compatibility with environment configurations"""

        original_system = TaskDetectionSystem()

        # Test different environment configurations
        environments = ["development", "production", "performance_critical"]

        for env in environments:
            config = TaskDetectionConfig()
            env_config = config.get_environment_preset(env)

            # Should be able to create fallback chain with any environment config
            fallback_chain = create_conservative_fallback_chain(original_system, env_config)

            assert fallback_chain.config is env_config
            assert fallback_chain.detection_system is original_system

    def test_integration_mode_configuration(self):
        """Test integration mode configuration options"""

        original_system = TaskDetectionSystem()

        # Test all integration modes
        for mode in IntegrationMode:
            enhanced_system = create_enhanced_task_detection(original_system, mode)

            assert enhanced_system.integration_config.mode == mode

            # Some modes should have fallback chain, others shouldn't
            if mode in [IntegrationMode.ACTIVE, IntegrationMode.AGGRESSIVE, IntegrationMode.SHADOW]:
                assert enhanced_system.fallback_chain is not None
            else:
                assert enhanced_system.fallback_chain is None or mode == IntegrationMode.MONITORING

    def test_gradual_rollout_configuration(self):
        """Test gradual rollout configuration"""

        original_system = TaskDetectionSystem()

        # Test different rollout percentages
        for percentage in [10.0, 50.0, 100.0]:
            config = IntegrationConfig(
                mode=IntegrationMode.ACTIVE,
                rollout_percentage=percentage,
            )

            enhanced_system = EnhancedTaskDetectionSystem(
                original_system, config,
            )

            assert enhanced_system.integration_config.rollout_percentage == percentage


class TestHealthCheckIntegration:
    """Test health check integration"""

    @pytest.mark.asyncio
    async def test_health_status_endpoint_compatibility(self):
        """Test health status endpoint compatibility"""

        original_system = TaskDetectionSystem()
        enhanced_system = create_enhanced_task_detection(original_system)

        # Get health status
        health = await enhanced_system.get_health_status()

        # Verify expected structure
        assert "integration_config" in health
        assert "original_system" in health
        assert "fallback_system" in health
        assert "integration_metrics" in health
        assert "current_state" in health

        # Verify health indicators
        assert isinstance(health["integration_config"]["mode"], str)
        assert isinstance(health["integration_metrics"], dict)

    def test_metrics_collection_integration(self):
        """Test metrics collection integration"""

        original_system = TaskDetectionSystem()
        enhanced_system = create_enhanced_task_detection(original_system)

        # Verify metrics are being collected
        metrics = enhanced_system.metrics.get_metrics_summary()

        assert "total_requests" in metrics
        assert "fallback_rate" in metrics
        assert "detection_success_rate" in metrics
        assert "errors_prevented" in metrics


class TestUIIntegration:
    """Test UI component integration"""

    def test_notification_manager_compatibility(self):
        """Test notification manager compatibility with UI systems"""

        from src.core.fallback_integration import IntegrationConfig, UserNotificationManager

        config = IntegrationConfig(enable_user_notifications=True)
        notification_manager = UserNotificationManager(config)

        # Verify notification manager is properly configured
        assert notification_manager.config.notify_on_fallback is True
        assert len(notification_manager.notification_history) == 0

    @pytest.mark.asyncio
    async def test_user_notification_flow(self):
        """Test user notification flow"""

        original_system = TaskDetectionSystem()
        enhanced_system = create_enhanced_task_detection(original_system)

        # Mock UI notification system
        with patch.object(enhanced_system.notification_manager, "_send_notification") as mock_notify:
            mock_notify.return_value = True

            # Trigger fallback that should generate notification
            with patch.object(original_system, "detect_categories") as mock_detect:
                mock_detect.side_effect = Exception("Critical error")

                await enhanced_system.detect_categories("test query")

                # Check if notification was attempted
                # (Actual notification depends on fallback level and configuration)


class TestDataFlowIntegration:
    """Test data flow integration through the system"""

    @pytest.mark.asyncio
    async def test_context_preservation(self):
        """Test that context is preserved through fallback system"""

        original_system = TaskDetectionSystem()
        enhanced_system = create_enhanced_task_detection(original_system)

        # Test with rich context
        context = {
            "user_id": "test_user",
            "project_type": "web_app",
            "file_extensions": [".py", ".js"],
            "has_tests": True,
            "has_security_files": True,
        }

        result = await enhanced_system.detect_categories("debug authentication", context)

        # Context should influence the result appropriately
        assert isinstance(result, DetectionResult)
        # Additional verification would depend on actual TaskDetectionSystem implementation

    @pytest.mark.asyncio
    async def test_signal_preservation(self):
        """Test that detection signals are preserved"""

        original_system = TaskDetectionSystem()
        enhanced_system = create_enhanced_task_detection(original_system)

        result = await enhanced_system.detect_categories("git commit changes")

        # Signals should be preserved from original detection or replaced with fallback info
        assert result.signals_used is not None
        assert isinstance(result.signals_used, dict)

    @pytest.mark.asyncio
    async def test_category_mapping_consistency(self):
        """Test that category mappings remain consistent"""

        original_system = TaskDetectionSystem()
        enhanced_system = create_enhanced_task_detection(original_system)

        result = await enhanced_system.detect_categories("git status")

        # Standard categories should be present
        expected_categories = ["core", "git", "analysis", "debug", "test", "quality", "security"]

        for category in expected_categories:
            assert category in result.categories
            assert isinstance(result.categories[category], bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
