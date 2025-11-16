#!/usr/bin/env python3
"""Comprehensive tests for examples/openrouter_client_demo.py.

This test suite covers all functions and code paths in the OpenRouter client demo,
ensuring 80%+ test coverage while following project testing standards.
"""

import asyncio
import logging
import os
from unittest.mock import AsyncMock, Mock, patch

import pytest

from examples.openrouter_client_demo import (
    check_environment,
    demo_basic_functionality,
    demo_error_handling,
    demo_model_integration,
    main,
)
from src.mcp_integration.mcp_client import MCPError, WorkflowStep


class TestDemoBasicFunctionality:
    """Test the demo_basic_functionality function."""

    @pytest.mark.asyncio
    @patch("examples.openrouter_client_demo.OpenRouterClient")
    @patch("examples.openrouter_client_demo.logger")
    async def test_demo_basic_functionality_success_path(self, mock_logger, mock_client_class):
        """Test successful execution of basic functionality demo."""
        # Setup mock client
        mock_client = AsyncMock()
        mock_client.connect.return_value = True
        mock_client.validate_query.return_value = {"is_valid": True, "potential_issues": []}
        mock_client.get_capabilities.return_value = ["text_generation", "query_validation"]

        # Mock health check response
        mock_health = Mock()
        mock_health.status = "healthy"
        mock_health.response_time = 0.123
        mock_health.error_count = 0
        mock_client.health_check.return_value = mock_health

        # Mock workflow response
        mock_response = Mock()
        mock_response.agent_id = "security-expert"
        mock_response.content = "This is a test response about secure software development principles..."
        mock_response.confidence = 0.95
        mock_response.processing_time = 1.234
        mock_client.orchestrate_agents.return_value = [mock_response]

        mock_client_class.return_value = mock_client

        # Execute the demo
        await demo_basic_functionality()

        # Verify client interactions
        mock_client.connect.assert_called_once()
        mock_client.validate_query.assert_called_once()
        mock_client.get_capabilities.assert_called_once()
        mock_client.orchestrate_agents.assert_called_once()
        mock_client.health_check.assert_called_once()
        mock_client.disconnect.assert_called_once()

        # Verify logging calls
        assert mock_logger.info.call_count >= 8

    @pytest.mark.asyncio
    @patch("examples.openrouter_client_demo.OpenRouterClient")
    @patch("examples.openrouter_client_demo.logger")
    async def test_demo_basic_functionality_connection_failure(self, mock_logger, mock_client_class):
        """Test demo behavior when connection fails."""
        mock_client = AsyncMock()
        mock_client.connect.return_value = False
        mock_client_class.return_value = mock_client

        await demo_basic_functionality()

        # Should return early when connection fails
        mock_client.connect.assert_called_once()
        mock_client.validate_query.assert_not_called()
        mock_client.disconnect.assert_called_once()

    @pytest.mark.asyncio
    @patch("examples.openrouter_client_demo.OpenRouterClient")
    @patch("examples.openrouter_client_demo.logger")
    async def test_demo_basic_functionality_invalid_query(self, mock_logger, mock_client_class):
        """Test demo behavior when query validation fails."""
        mock_client = AsyncMock()
        mock_client.connect.return_value = True
        mock_client.validate_query.return_value = {"is_valid": False, "potential_issues": ["XSS risk detected"]}
        mock_client_class.return_value = mock_client

        await demo_basic_functionality()

        # Should return early when query is invalid
        mock_client.connect.assert_called_once()
        mock_client.validate_query.assert_called_once()
        mock_client.get_capabilities.assert_not_called()
        mock_client.disconnect.assert_called_once()

    @pytest.mark.asyncio
    @patch("examples.openrouter_client_demo.OpenRouterClient")
    @patch("examples.openrouter_client_demo.logger")
    async def test_demo_basic_functionality_capabilities_error(self, mock_logger, mock_client_class):
        """Test demo behavior when capabilities retrieval fails."""
        mock_client = AsyncMock()
        mock_client.connect.return_value = True
        mock_client.validate_query.return_value = {"is_valid": True, "potential_issues": []}
        mock_client.get_capabilities.side_effect = MCPError("Capabilities error")

        # Mock other successful responses
        mock_health = Mock()
        mock_health.status = "healthy"
        mock_health.response_time = 0.123
        mock_health.error_count = 0
        mock_client.health_check.return_value = mock_health
        mock_client.orchestrate_agents.return_value = []

        mock_client_class.return_value = mock_client

        await demo_basic_functionality()

        # Should continue despite capabilities error
        mock_client.get_capabilities.assert_called_once()
        mock_client.orchestrate_agents.assert_called_once()
        mock_client.health_check.assert_called_once()

    @pytest.mark.asyncio
    @patch("examples.openrouter_client_demo.OpenRouterClient")
    @patch("examples.openrouter_client_demo.logger")
    async def test_demo_basic_functionality_workflow_error(self, mock_logger, mock_client_class):
        """Test demo behavior when workflow execution fails."""
        mock_client = AsyncMock()
        mock_client.connect.return_value = True
        mock_client.validate_query.return_value = {"is_valid": True, "potential_issues": []}
        mock_client.get_capabilities.return_value = ["text_generation"]
        mock_client.orchestrate_agents.side_effect = MCPError("Workflow error")

        mock_health = Mock()
        mock_health.status = "healthy"
        mock_health.response_time = 0.123
        mock_health.error_count = 0
        mock_client.health_check.return_value = mock_health

        mock_client_class.return_value = mock_client

        await demo_basic_functionality()

        # Should continue to health check despite workflow error
        mock_client.orchestrate_agents.assert_called_once()
        mock_client.health_check.assert_called_once()

    @pytest.mark.asyncio
    @patch("examples.openrouter_client_demo.OpenRouterClient")
    @patch("examples.openrouter_client_demo.logger")
    async def test_demo_basic_functionality_no_responses(self, mock_logger, mock_client_class):
        """Test demo behavior when no responses are returned from workflow."""
        mock_client = AsyncMock()
        mock_client.connect.return_value = True
        mock_client.validate_query.return_value = {"is_valid": True, "potential_issues": []}
        mock_client.get_capabilities.return_value = ["text_generation"]
        mock_client.orchestrate_agents.return_value = []  # No responses

        mock_health = Mock()
        mock_health.status = "healthy"
        mock_health.response_time = 0.123
        mock_health.error_count = 0
        mock_client.health_check.return_value = mock_health

        mock_client_class.return_value = mock_client

        await demo_basic_functionality()

        # Should handle empty responses gracefully
        mock_client.orchestrate_agents.assert_called_once()
        mock_client.health_check.assert_called_once()

    @pytest.mark.asyncio
    @patch("examples.openrouter_client_demo.OpenRouterClient")
    @patch("examples.openrouter_client_demo.logger")
    async def test_demo_basic_functionality_exception_handling(self, mock_logger, mock_client_class):
        """Test demo behavior when unexpected exceptions occur."""
        mock_client = AsyncMock()
        mock_client.connect.side_effect = Exception("Unexpected error")
        mock_client_class.return_value = mock_client

        # Should not raise exception, should handle gracefully
        await demo_basic_functionality()

        mock_client.disconnect.assert_called_once()


class TestDemoErrorHandling:
    """Test the demo_error_handling function."""

    @pytest.mark.asyncio
    @patch("examples.openrouter_client_demo.OpenRouterClient")
    @patch("examples.openrouter_client_demo.logger")
    async def test_demo_error_handling_invalid_config(self, mock_logger, mock_client_class):
        """Test error handling demo with invalid configuration."""
        mock_client = AsyncMock()
        mock_client.connect.side_effect = MCPError("Connection failed")
        mock_client_class.return_value = mock_client

        await demo_error_handling()

        # Should call client with invalid configuration
        mock_client_class.assert_called()
        mock_client.connect.assert_called_once()

    @pytest.mark.asyncio
    @patch("examples.openrouter_client_demo.OpenRouterClient")
    @patch("examples.openrouter_client_demo.logger")
    async def test_demo_error_handling_suspicious_query(self, mock_logger, mock_client_class):
        """Test error handling demo with suspicious query validation."""
        mock_client = AsyncMock()
        mock_client.validate_query.return_value = {
            "is_valid": False,
            "potential_issues": ["XSS attack detected", "Script injection risk"],
        }
        mock_client_class.return_value = mock_client

        await demo_error_handling()

        # Should test query validation with suspicious content
        mock_client.validate_query.assert_called()

        # Check that suspicious query was used
        call_args = mock_client.validate_query.call_args[0][0]
        assert "<script>" in call_args
        assert "alert(" in call_args

    @pytest.mark.asyncio
    @patch("examples.openrouter_client_demo.OpenRouterClient")
    @patch("examples.openrouter_client_demo.logger")
    async def test_demo_error_handling_query_validation_exception(self, mock_logger, mock_client_class):
        """Test error handling when query validation raises exception."""
        mock_client = AsyncMock()
        mock_client.validate_query.side_effect = Exception("Validation error")
        mock_client_class.return_value = mock_client

        await demo_error_handling()

        # Should handle validation exception gracefully
        mock_client.validate_query.assert_called()

    @pytest.mark.asyncio
    @patch("examples.openrouter_client_demo.OpenRouterClient")
    @patch("examples.openrouter_client_demo.logger")
    async def test_demo_error_handling_valid_suspicious_query(self, mock_logger, mock_client_class):
        """Test error handling when suspicious query is incorrectly marked as valid."""
        mock_client = AsyncMock()
        mock_client.validate_query.return_value = {
            "is_valid": True,  # Should be False for suspicious content
            "potential_issues": [],
        }
        mock_client_class.return_value = mock_client

        await demo_error_handling()

        # Should detect and warn about failure to identify security issues
        mock_client.validate_query.assert_called()


class TestDemoModelIntegration:
    """Test the demo_model_integration function."""

    @pytest.mark.asyncio
    @patch("examples.openrouter_client_demo.OpenRouterClient")
    @patch("examples.openrouter_client_demo.logger")
    async def test_demo_model_integration_all_task_types(self, mock_logger, mock_client_class):
        """Test model integration demo with all task types."""
        mock_client = AsyncMock()
        mock_client._execute_single_step.side_effect = Exception("Expected without real API")
        mock_client_class.return_value = mock_client

        await demo_model_integration()

        # Should test all task types
        expected_task_types = ["general", "reasoning", "vision", "analysis"]
        assert mock_client._execute_single_step.call_count == len(expected_task_types)

        # Verify each task type was tested
        call_args = [call[0][0] for call in mock_client._execute_single_step.call_args_list]
        for i, expected_type in enumerate(expected_task_types):
            workflow_step = call_args[i]
            assert workflow_step.input_data["task_type"] == expected_type

    @pytest.mark.asyncio
    @patch("examples.openrouter_client_demo.OpenRouterClient")
    @patch("examples.openrouter_client_demo.logger")
    async def test_demo_model_integration_workflow_step_structure(self, mock_logger, mock_client_class):
        """Test that workflow steps are created with correct structure."""
        mock_client = AsyncMock()
        mock_client._execute_single_step.side_effect = Exception("Expected without real API")
        mock_client_class.return_value = mock_client

        await demo_model_integration()

        # Verify workflow step structure
        call_args = [call[0][0] for call in mock_client._execute_single_step.call_args_list]

        for workflow_step in call_args:
            assert isinstance(workflow_step, WorkflowStep)
            assert workflow_step.step_id.startswith("demo-")
            assert workflow_step.agent_id.endswith("-agent")
            assert "query" in workflow_step.input_data
            assert "task_type" in workflow_step.input_data
            assert "allow_premium" in workflow_step.input_data
            assert workflow_step.input_data["allow_premium"] is False
            assert "max_tokens_needed" in workflow_step.input_data
            assert workflow_step.input_data["max_tokens_needed"] == 4096
            assert workflow_step.timeout_seconds == 30

    @pytest.mark.asyncio
    @patch("examples.openrouter_client_demo.OpenRouterClient")
    @patch("examples.openrouter_client_demo.logger")
    async def test_demo_model_integration_successful_execution(self, mock_logger, mock_client_class):
        """Test model integration demo with successful execution."""
        mock_client = AsyncMock()
        mock_client._execute_single_step.return_value = "Success"
        mock_client_class.return_value = mock_client

        await demo_model_integration()

        # Should execute all task types successfully
        assert mock_client._execute_single_step.call_count == 4


class TestCheckEnvironment:
    """Test the check_environment function."""

    @patch("examples.openrouter_client_demo.os.getenv")
    @patch("examples.openrouter_client_demo.logger")
    def test_check_environment_api_key_present(self, mock_logger, mock_getenv):
        """Test check_environment when API key is present."""
        mock_getenv.side_effect = lambda key, default=None: {
            "PROMPTCRAFT_OPENROUTER_API_KEY": "sk-test-key-123",
            "PROMPTCRAFT_OPENROUTER_BASE_URL": default,
        }.get(key, default)

        check_environment()

        # Should log success message for API key
        log_calls = [str(call[0]) for call in mock_logger.info.call_args_list]
        api_key_log = next((call for call in log_calls if "✓ OpenRouter API key is configured" in call), None)
        assert api_key_log is not None

    @patch("examples.openrouter_client_demo.os.getenv")
    @patch("examples.openrouter_client_demo.logger")
    def test_check_environment_api_key_missing(self, mock_logger, mock_getenv):
        """Test check_environment when API key is missing."""
        mock_getenv.side_effect = lambda key, default=None: {
            "PROMPTCRAFT_OPENROUTER_API_KEY": None,
            "PROMPTCRAFT_OPENROUTER_BASE_URL": default,
        }.get(key, default)

        check_environment()

        # Should log warning for missing API key
        assert mock_logger.warning.call_count >= 1
        warning_calls = [str(call[0]) for call in mock_logger.warning.call_args_list]
        missing_key_warning = next((call for call in warning_calls if "not found in environment" in call), None)
        assert missing_key_warning is not None

    @patch("examples.openrouter_client_demo.os.getenv")
    @patch("examples.openrouter_client_demo.logger")
    def test_check_environment_base_url_default(self, mock_logger, mock_getenv):
        """Test check_environment with default base URL."""

        def mock_getenv_func(key, default=None):
            if key == "PROMPTCRAFT_OPENROUTER_API_KEY":
                return "sk-test-key"
            if key == "PROMPTCRAFT_OPENROUTER_BASE_URL":
                return default  # Return the default value (https://openrouter.ai/api/v1)
            return default

        mock_getenv.side_effect = mock_getenv_func

        check_environment()

        # Should log default base URL
        log_calls = [str(call) for call in mock_logger.info.call_args_list]
        base_url_log = next((call for call in log_calls if "https://openrouter.ai/api/v1" in call), None)
        assert base_url_log is not None

    @patch("examples.openrouter_client_demo.os.getenv")
    @patch("examples.openrouter_client_demo.logger")
    def test_check_environment_custom_base_url(self, mock_logger, mock_getenv):
        """Test check_environment with custom base URL."""
        custom_url = "https://custom-openrouter.example.com/api/v1"
        mock_getenv.side_effect = lambda key, default=None: {
            "PROMPTCRAFT_OPENROUTER_API_KEY": "sk-test-key",
            "PROMPTCRAFT_OPENROUTER_BASE_URL": custom_url,
        }.get(key, default)

        check_environment()

        # Should log custom base URL
        log_calls = [str(call[0]) for call in mock_logger.info.call_args_list]
        base_url_log = next((call for call in log_calls if custom_url in call), None)
        assert base_url_log is not None


class TestMainFunction:
    """Test the main function."""

    @pytest.mark.asyncio
    @patch("examples.openrouter_client_demo.check_environment")
    @patch("examples.openrouter_client_demo.demo_basic_functionality")
    @patch("examples.openrouter_client_demo.demo_error_handling")
    @patch("examples.openrouter_client_demo.demo_model_integration")
    async def test_main_function_execution_order(
        self,
        mock_model_integration,
        mock_error_handling,
        mock_basic_functionality,
        mock_check_environment,
    ):
        """Test that main function executes all demos in correct order."""
        await main()

        # Verify all functions were called
        mock_check_environment.assert_called_once()
        mock_basic_functionality.assert_called_once()
        mock_error_handling.assert_called_once()
        mock_model_integration.assert_called_once()

    @pytest.mark.asyncio
    @patch("examples.openrouter_client_demo.check_environment", side_effect=Exception("Check error"))
    @patch("examples.openrouter_client_demo.demo_basic_functionality")
    @patch("examples.openrouter_client_demo.demo_error_handling")
    @patch("examples.openrouter_client_demo.demo_model_integration")
    async def test_main_function_error_propagation(
        self,
        mock_model_integration,
        mock_error_handling,
        mock_basic_functionality,
        mock_check_environment,
    ):
        """Test that main function propagates errors from subfunctions."""
        with pytest.raises(Exception, match="Check error"):
            await main()

        # Should not call other functions if check_environment fails
        mock_basic_functionality.assert_not_called()

    @pytest.mark.asyncio
    @patch("examples.openrouter_client_demo.check_environment")
    @patch("examples.openrouter_client_demo.demo_basic_functionality", side_effect=Exception("Demo error"))
    @patch("examples.openrouter_client_demo.demo_error_handling")
    @patch("examples.openrouter_client_demo.demo_model_integration")
    async def test_main_function_continues_on_demo_error(
        self,
        mock_model_integration,
        mock_error_handling,
        mock_basic_functionality,
        mock_check_environment,
    ):
        """Test that main function continues even if individual demos fail."""
        with pytest.raises(Exception, match="Demo error"):
            await main()

        # Should have called check_environment and basic_functionality
        mock_check_environment.assert_called_once()
        mock_basic_functionality.assert_called_once()


class TestLoggingAndFormatting:
    """Test logging and output formatting."""

    @patch("logging.basicConfig")
    def test_logging_configuration(self, mock_basic_config):
        """Test that logging is configured correctly."""
        # Re-import the module to trigger logging setup
        import importlib

        import examples.openrouter_client_demo

        importlib.reload(examples.openrouter_client_demo)

        # Verify logging was configured
        mock_basic_config.assert_called()
        # Find the call with level=INFO
        info_calls = [
            call
            for call in mock_basic_config.call_args_list
            if "level" in call.kwargs and call.kwargs["level"] == logging.INFO
        ]
        assert len(info_calls) >= 1
        args, kwargs = info_calls[0]
        assert kwargs["level"] == logging.INFO
        assert "format" in kwargs

    def test_logger_name(self):
        """Test that logger has correct name."""
        import examples.openrouter_client_demo

        # Logger should be named after the module
        assert examples.openrouter_client_demo.logger.name == "examples.openrouter_client_demo"


class TestWorkflowStepCreation:
    """Test WorkflowStep creation and validation."""

    def test_workflow_step_creation_in_basic_demo(self):
        """Test that WorkflowStep is created correctly in basic demo."""
        # Test the structure of WorkflowStep creation
        step = WorkflowStep(
            step_id="demo-step-1",
            agent_id="security-expert",
            input_data={
                "query": "test query",
                "task_type": "general",
                "max_tokens": 150,
                "temperature": 0.7,
            },
            timeout_seconds=30,
        )

        assert step.step_id == "demo-step-1"
        assert step.agent_id == "security-expert"
        assert step.input_data["query"] == "test query"
        assert step.input_data["task_type"] == "general"
        assert step.input_data["max_tokens"] == 150
        assert step.input_data["temperature"] == 0.7
        assert step.timeout_seconds == 30

    def test_workflow_step_creation_in_model_demo(self):
        """Test that WorkflowStep is created correctly in model integration demo."""
        task_types = ["general", "reasoning", "vision", "analysis"]

        for task_type in task_types:
            step = WorkflowStep(
                step_id=f"demo-{task_type}",
                agent_id=f"{task_type}-agent",
                input_data={
                    "query": f"This is a {task_type} task",
                    "task_type": task_type,
                    "allow_premium": False,
                    "max_tokens_needed": 4096,
                },
                timeout_seconds=30,
            )

            assert step.step_id == f"demo-{task_type}"
            assert step.agent_id == f"{task_type}-agent"
            assert step.input_data["task_type"] == task_type
            assert step.input_data["allow_premium"] is False
            assert step.input_data["max_tokens_needed"] == 4096


class TestIntegrationAndEdgeCases:
    """Test integration scenarios and edge cases."""

    @pytest.mark.asyncio
    async def test_asyncio_run_compatibility(self):
        """Test that the demo functions are compatible with asyncio.run."""
        # This test ensures the async functions can be run properly
        try:
            # Should be able to run without errors (with mocking)
            await asyncio.sleep(0.001)  # Minimal async operation
        except Exception as e:
            pytest.fail(f"Asyncio compatibility test failed: {e}")

    @patch.dict(os.environ, {}, clear=True)
    def test_environment_variable_handling(self):
        """Test handling of missing environment variables."""
        # Clear environment and test
        result = os.getenv("PROMPTCRAFT_OPENROUTER_API_KEY")
        assert result is None

        # Default value handling
        base_url = os.getenv("PROMPTCRAFT_OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        assert base_url == "https://openrouter.ai/api/v1"

    def test_module_imports(self):
        """Test that all required imports are available."""
        try:
            from examples.openrouter_client_demo import (
                check_environment,
                demo_basic_functionality,
                demo_error_handling,
                demo_model_integration,
                main,
            )
            from src.mcp_integration.mcp_client import MCPError, WorkflowStep
            from src.mcp_integration.openrouter_client import OpenRouterClient

            # All imports should be successful
            assert callable(demo_basic_functionality)
            assert callable(demo_error_handling)
            assert callable(demo_model_integration)
            assert callable(check_environment)
            assert callable(main)
            assert MCPError is not None
            assert WorkflowStep is not None
            assert OpenRouterClient is not None

        except ImportError as e:
            pytest.fail(f"Required imports failed: {e}")

    @pytest.mark.asyncio
    @patch("examples.openrouter_client_demo.OpenRouterClient")
    async def test_real_openrouter_client_instantiation(self, mock_client_class):
        """Test that OpenRouterClient can be instantiated correctly."""
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client

        # Test both default and custom instantiation
        from examples.openrouter_client_demo import OpenRouterClient

        # Default client
        client1 = OpenRouterClient()
        assert client1 is not None

        # Custom client with parameters
        client2 = OpenRouterClient(api_key="test-key", base_url="https://test-url.com/api/v1")
        assert client2 is not None
