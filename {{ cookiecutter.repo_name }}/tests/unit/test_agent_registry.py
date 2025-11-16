"""
Unit tests for AgentRegistry system.

This module tests the AgentRegistry class and its functionality including
registration, discovery, capability matching, and error handling.
"""

from unittest.mock import patch

import pytest

from src.agents.base_agent import BaseAgent
from src.agents.exceptions import AgentRegistrationError
from src.agents.models import AgentOutput


class TestAgentRegistry:
    """Test suite for AgentRegistry class."""

    def test_agent_registry_initialization(self, fresh_agent_registry):
        """Test AgentRegistry initialization."""
        registry = fresh_agent_registry

        assert len(registry) == 0
        assert registry.list_agents() == []
        assert registry.list_agent_classes() == {}
        assert registry.get_registry_status()["total_agents"] == 0

    def test_agent_registry_register_decorator(self, fresh_agent_registry, mock_agent_class):
        """Test agent registration via decorator."""
        registry = fresh_agent_registry

        # Register agent class
        decorated_class = registry.register("test_agent")(mock_agent_class)

        assert decorated_class == mock_agent_class
        assert "test_agent" in registry
        assert len(registry) == 1
        assert registry.list_agents() == ["test_agent"]

    def test_agent_registry_register_duplicate_agent_id(self, fresh_agent_registry, mock_agent_class):
        """Test registration with duplicate agent ID."""
        registry = fresh_agent_registry

        # Register first agent
        registry.register("duplicate_agent")(mock_agent_class)

        # Try to register another agent with same ID
        with pytest.raises(AgentRegistrationError) as excinfo:
            registry.register("duplicate_agent")(mock_agent_class)

        assert "Agent ID 'duplicate_agent' already exists" in str(excinfo.value)
        assert excinfo.value.error_code == "DUPLICATE_AGENT_ID"

    @pytest.mark.parametrize(
        ("agent_id", "expected_error"),
        [
            ("", "Agent ID must be a non-empty string"),
            (None, "Agent ID must be a non-empty string"),
            ("test-agent", "Agent ID 'test-agent' must contain only alphanumeric characters and underscores"),
            ("test agent", "Agent ID 'test agent' must contain only alphanumeric characters and underscores"),
            ("test@agent", "Agent ID 'test@agent' must contain only alphanumeric characters and underscores"),
        ],
        ids=["empty", "none", "dash", "space", "special-char"],
    )
    def test_agent_registry_invalid_agent_id(self, fresh_agent_registry, mock_agent_class, agent_id, expected_error):
        """Test registration with invalid agent ID."""
        registry = fresh_agent_registry

        with pytest.raises(AgentRegistrationError) as excinfo:
            registry.register(agent_id)(mock_agent_class)

        assert expected_error in str(excinfo.value)
        assert excinfo.value.error_code == "INVALID_AGENT_ID"

    def test_agent_registry_invalid_agent_class(self, fresh_agent_registry):
        """Test registration with invalid agent class."""
        registry = fresh_agent_registry

        class InvalidAgent:
            pass  # Does not inherit from BaseAgent

        with pytest.raises(AgentRegistrationError) as excinfo:
            registry.register("invalid_agent")(InvalidAgent)

        assert "Agent class 'InvalidAgent' must inherit from BaseAgent" in str(excinfo.value)
        assert excinfo.value.error_code == "INVALID_AGENT_CLASS"

    def test_agent_registry_missing_execute_method(self, fresh_agent_registry):
        """Test registration with agent class missing execute method."""
        registry = fresh_agent_registry

        class IncompleteAgent(BaseAgent):
            pass  # Missing execute method

        with pytest.raises(AgentRegistrationError) as excinfo:
            registry.register("incomplete_agent")(IncompleteAgent)

        assert "Agent class 'IncompleteAgent' missing required method 'execute'" in str(excinfo.value)
        assert excinfo.value.error_code == "MISSING_REQUIRED_METHOD"

    def test_agent_registry_non_async_execute_method(self, fresh_agent_registry):
        """Test registration with non-async execute method."""
        registry = fresh_agent_registry

        class SyncAgent(BaseAgent):
            def execute(self, agent_input):  # Not async
                return AgentOutput(content="sync response", confidence=0.9, processing_time=0.1, agent_id=self.agent_id)

        with pytest.raises(AgentRegistrationError) as excinfo:
            registry.register("sync_agent")(SyncAgent)

        assert "Agent class 'SyncAgent' execute method must be async" in str(excinfo.value)
        assert excinfo.value.error_code == "INVALID_METHOD_SIGNATURE"

    def test_agent_registry_get_agent_class(self, fresh_agent_registry, mock_agent_class):
        """Test getting agent class by ID."""
        registry = fresh_agent_registry

        # Register agent
        registry.register("test_agent")(mock_agent_class)

        # Get agent class
        retrieved_class = registry.get_agent_class("test_agent")
        assert retrieved_class == mock_agent_class

    def test_agent_registry_get_agent_class_not_found(self, fresh_agent_registry):
        """Test getting non-existent agent class."""
        registry = fresh_agent_registry

        with pytest.raises(AgentRegistrationError) as excinfo:
            registry.get_agent_class("non_existent")

        assert "Agent 'non_existent' not found in registry" in str(excinfo.value)
        assert excinfo.value.error_code == "AGENT_NOT_FOUND"

    def test_agent_registry_get_agent_instance(self, fresh_agent_registry, mock_agent_class):
        """Test getting agent instance."""
        registry = fresh_agent_registry

        # Register agent
        registry.register("test_agent")(mock_agent_class)

        # Get agent instance
        config = {"agent_id": "test_agent"}
        agent = registry.get_agent("test_agent", config)

        assert isinstance(agent, mock_agent_class)
        assert agent.agent_id == "test_agent"

    def test_agent_registry_get_agent_with_auto_agent_id(self, fresh_agent_registry, mock_agent_class):
        """Test getting agent instance with auto-injected agent_id."""
        registry = fresh_agent_registry

        # Register agent
        registry.register("test_agent")(mock_agent_class)

        # Get agent instance without agent_id in config
        config = {"other_param": "value"}
        agent = registry.get_agent("test_agent", config)

        assert isinstance(agent, mock_agent_class)
        assert agent.agent_id == "test_agent"

    def test_agent_registry_get_agent_instantiation_error(self, fresh_agent_registry):
        """Test agent instantiation error."""
        registry = fresh_agent_registry

        class ErrorAgent(BaseAgent):
            def __init__(self, config):
                raise ValueError("Instantiation error")

            async def execute(self, agent_input):
                return AgentOutput(content="response", confidence=0.9, processing_time=0.1, agent_id=self.agent_id)

        # Register agent
        registry.register("error_agent")(ErrorAgent)

        with pytest.raises(AgentRegistrationError) as excinfo:
            registry.get_agent("error_agent", {"agent_id": "error_agent"})

        assert "Failed to instantiate agent 'error_agent'" in str(excinfo.value)
        assert excinfo.value.error_code == "INSTANTIATION_FAILED"

    def test_agent_registry_get_cached_agent(self, fresh_agent_registry, mock_agent_class):
        """Test cached agent retrieval."""
        registry = fresh_agent_registry

        # Register agent
        registry.register("test_agent")(mock_agent_class)

        # Get agent instance (should be cached)
        config = {"agent_id": "test_agent", "param": "value"}
        agent1 = registry.get_cached_agent("test_agent", config)
        agent2 = registry.get_cached_agent("test_agent", config)

        # Should return same instance
        assert agent1 is agent2

    def test_agent_registry_get_cached_agent_different_config(self, fresh_agent_registry, mock_agent_class):
        """Test cached agent with different config."""
        registry = fresh_agent_registry

        # Register agent
        registry.register("test_agent")(mock_agent_class)

        # Get agent instances with different configs
        config1 = {"agent_id": "test_agent", "param": "value1"}
        config2 = {"agent_id": "test_agent", "param": "value2"}

        agent1 = registry.get_cached_agent("test_agent", config1)
        agent2 = registry.get_cached_agent("test_agent", config2)

        # Should return different instances
        assert agent1 is not agent2

    def test_agent_registry_capabilities_storage(self, fresh_agent_registry):
        """Test agent capabilities storage."""
        registry = fresh_agent_registry

        class CapableAgent(BaseAgent):
            def __init__(self, config):
                super().__init__(config)

            async def execute(self, agent_input):
                return AgentOutput(content="response", confidence=0.9, processing_time=0.1, agent_id=self.agent_id)

            def get_capabilities(self):
                return {"input_types": ["text", "code"], "output_types": ["analysis"], "custom_capability": True}

        # Register agent
        registry.register("capable_agent")(CapableAgent)

        # Get agent instance (should store capabilities)
        config = {"agent_id": "capable_agent"}
        _agent = registry.get_agent("capable_agent", config)

        # Verify capabilities are stored
        assert "capable_agent" in registry._capabilities
        capabilities = registry._capabilities["capable_agent"]
        assert capabilities["input_types"] == ["text", "code"]
        assert capabilities["custom_capability"] is True

    def test_agent_registry_find_agents_by_capability(self, fresh_agent_registry):
        """Test finding agents by capability."""
        registry = fresh_agent_registry

        class Agent1(BaseAgent):
            def __init__(self, config):
                super().__init__(config)

            async def execute(self, agent_input):
                return AgentOutput(content="response", confidence=0.9, processing_time=0.1, agent_id=self.agent_id)

            def get_capabilities(self):
                return {"feature": "analysis", "type": "security"}

        class Agent2(BaseAgent):
            def __init__(self, config):
                super().__init__(config)

            async def execute(self, agent_input):
                return AgentOutput(content="response", confidence=0.9, processing_time=0.1, agent_id=self.agent_id)

            def get_capabilities(self):
                return {"feature": "generation", "type": "security"}

        # Register agents
        registry.register("agent1")(Agent1)
        registry.register("agent2")(Agent2)

        # Create instances to populate capabilities
        registry.get_agent("agent1", {"agent_id": "agent1"})
        registry.get_agent("agent2", {"agent_id": "agent2"})

        # Find agents by capability
        analysis_agents = registry.find_agents_by_capability("feature", "analysis")
        security_agents = registry.find_agents_by_capability("type", "security")
        non_existent = registry.find_agents_by_capability("non_existent")

        assert analysis_agents == ["agent1"]
        assert set(security_agents) == {"agent1", "agent2"}
        assert non_existent == []

    def test_agent_registry_find_agents_by_type(self, fresh_agent_registry):
        """Test finding agents by input/output type."""
        registry = fresh_agent_registry

        class TextAgent(BaseAgent):
            def __init__(self, config):
                super().__init__(config)

            async def execute(self, agent_input):
                return AgentOutput(content="response", confidence=0.9, processing_time=0.1, agent_id=self.agent_id)

            def get_capabilities(self):
                return {"input_types": ["text"], "output_types": ["text", "analysis"]}

        class CodeAgent(BaseAgent):
            def __init__(self, config):
                super().__init__(config)

            async def execute(self, agent_input):
                return AgentOutput(content="response", confidence=0.9, processing_time=0.1, agent_id=self.agent_id)

            def get_capabilities(self):
                return {"input_types": ["code", "text"], "output_types": ["analysis"]}

        # Register agents
        registry.register("text_agent")(TextAgent)
        registry.register("code_agent")(CodeAgent)

        # Create instances to populate capabilities
        registry.get_agent("text_agent", {"agent_id": "text_agent"})
        registry.get_agent("code_agent", {"agent_id": "code_agent"})

        # Find agents by type
        text_input_agents = registry.find_agents_by_type("text")
        code_input_agents = registry.find_agents_by_type("code")
        analysis_output_agents = registry.find_agents_by_type("text", "analysis")

        assert set(text_input_agents) == {"text_agent", "code_agent"}
        assert code_input_agents == ["code_agent"]
        assert set(analysis_output_agents) == {"text_agent", "code_agent"}

    def test_agent_registry_get_agent_info(self, fresh_agent_registry, mock_agent_class):
        """Test getting agent information."""
        registry = fresh_agent_registry

        # Register agent
        registry.register("test_agent")(mock_agent_class)

        # Create instance to populate info
        config = {"agent_id": "test_agent", "param": "value"}
        registry.get_cached_agent("test_agent", config)

        # Get agent info
        info = registry.get_agent_info("test_agent")

        assert info["agent_id"] == "test_agent"
        assert info["agent_class"] == mock_agent_class.__name__
        assert info["module"] == mock_agent_class.__module__
        assert info["is_cached"] is True
        assert info["config"] == config

    def test_agent_registry_get_agent_info_not_found(self, fresh_agent_registry):
        """Test getting info for non-existent agent."""
        registry = fresh_agent_registry

        with pytest.raises(AgentRegistrationError) as excinfo:
            registry.get_agent_info("non_existent")

        assert "Agent 'non_existent' not found in registry" in str(excinfo.value)
        assert excinfo.value.error_code == "AGENT_NOT_FOUND"

    def test_agent_registry_unregister(self, fresh_agent_registry, mock_agent_class):
        """Test agent unregistration."""
        registry = fresh_agent_registry

        # Register agent
        registry.register("test_agent")(mock_agent_class)

        # Create cached instance
        config = {"agent_id": "test_agent"}
        registry.get_cached_agent("test_agent", config)

        # Verify agent is registered
        assert "test_agent" in registry
        assert len(registry) == 1

        # Unregister agent
        registry.unregister("test_agent")

        # Verify agent is removed
        assert "test_agent" not in registry
        assert len(registry) == 0
        assert registry.list_agents() == []

    def test_agent_registry_unregister_not_found(self, fresh_agent_registry):
        """Test unregistering non-existent agent."""
        registry = fresh_agent_registry

        with pytest.raises(AgentRegistrationError) as excinfo:
            registry.unregister("non_existent")

        assert "Agent 'non_existent' not found in registry" in str(excinfo.value)
        assert excinfo.value.error_code == "AGENT_NOT_FOUND"

    def test_agent_registry_clear(self, fresh_agent_registry, mock_agent_class):
        """Test clearing all registrations."""
        registry = fresh_agent_registry

        # Register multiple agents
        registry.register("agent1")(mock_agent_class)
        registry.register("agent2")(mock_agent_class)

        # Create cached instances
        registry.get_cached_agent("agent1", {"agent_id": "agent1"})
        registry.get_cached_agent("agent2", {"agent_id": "agent2"})

        # Verify agents are registered
        assert len(registry) == 2

        # Clear all registrations
        registry.clear()

        # Verify registry is empty
        assert len(registry) == 0
        assert registry.list_agents() == []
        assert registry.get_registry_status()["total_agents"] == 0

    def test_agent_registry_status(self, fresh_agent_registry, mock_agent_class):
        """Test registry status reporting."""
        registry = fresh_agent_registry

        # Register agents
        registry.register("agent1")(mock_agent_class)
        registry.register("agent2")(mock_agent_class)

        # Create one cached instance
        registry.get_cached_agent("agent1", {"agent_id": "agent1"})

        # Check status
        status = registry.get_registry_status()

        assert status["total_agents"] == 2
        assert status["cached_instances"] == 1
        assert status["agents_with_capabilities"] == 1  # agent1 has capabilities
        assert set(status["registered_agents"]) == {"agent1", "agent2"}

    def test_agent_registry_container_methods(self, fresh_agent_registry, mock_agent_class):
        """Test container methods (__contains__, __len__, __iter__)."""
        registry = fresh_agent_registry

        # Register agents
        registry.register("agent1")(mock_agent_class)
        registry.register("agent2")(mock_agent_class)

        # Test __contains__
        assert "agent1" in registry
        assert "agent2" in registry
        assert "non_existent" not in registry

        # Test __len__
        assert len(registry) == 2

        # Test __iter__
        agent_ids = list(registry)
        assert set(agent_ids) == {"agent1", "agent2"}

    def test_agent_registry_logging(self, fresh_agent_registry, mock_agent_class):
        """Test registry logging behavior."""
        registry = fresh_agent_registry

        with patch.object(registry.logger, "info") as mock_info:
            # Register agent
            registry.register("test_agent")(mock_agent_class)

            # Verify logging
            mock_info.assert_called_once()
            args, kwargs = mock_info.call_args
            assert "Registered agent with class" in args[0]
            assert kwargs["extra"]["agent_id"] == "test_agent"

    def test_global_agent_registry_singleton(self):
        """Test that global agent_registry is a singleton."""
        # pylint: disable=import-outside-toplevel
        from src.agents.registry import agent_registry as registry1
        from src.agents.registry import agent_registry as registry2

        # Should be the same instance
        assert registry1 is registry2

    @pytest.mark.integration
    def test_agent_registry_full_lifecycle(self, fresh_agent_registry):
        """Test complete agent lifecycle through registry."""
        registry = fresh_agent_registry

        # Define test agent
        class LifecycleAgent(BaseAgent):
            def __init__(self, config):
                super().__init__(config)
                self.processed_count = 0

            async def execute(self, agent_input):
                self.processed_count += 1
                return AgentOutput(
                    content=f"Processed {self.processed_count} times",
                    confidence=0.9,
                    processing_time=0.1,
                    agent_id=self.agent_id,
                )

            def get_capabilities(self):
                return {"input_types": ["text"], "output_types": ["text"], "stateful": True}

        # Register agent
        registry.register("lifecycle_agent")(LifecycleAgent)

        # Get agent instance
        config = {"agent_id": "lifecycle_agent"}
        agent = registry.get_cached_agent("lifecycle_agent", config)

        # Verify agent is working
        assert isinstance(agent, LifecycleAgent)
        assert agent.processed_count == 0

        # Test capabilities
        capabilities = registry.find_agents_by_capability("stateful", True)
        assert "lifecycle_agent" in capabilities

        # Test agent info
        info = registry.get_agent_info("lifecycle_agent")
        assert info["agent_id"] == "lifecycle_agent"
        assert info["is_cached"] is True

        # Test registry status
        status = registry.get_registry_status()
        assert status["total_agents"] == 1
        assert status["cached_instances"] == 1

        # Cleanup
        registry.unregister("lifecycle_agent")
        assert len(registry) == 0

    @pytest.mark.security
    def test_agent_registry_security_validation(self, fresh_agent_registry, security_test_inputs):
        """Test registry security with malicious inputs."""
        registry = fresh_agent_registry

        # Define a simple agent class
        class SecurityTestAgent(BaseAgent):
            async def execute(self, agent_input):
                return AgentOutput(
                    content="secure response",
                    confidence=0.9,
                    processing_time=0.1,
                    agent_id=self.agent_id,
                )

        # Test malicious agent IDs (should fail validation)
        for malicious_input in security_test_inputs:
            try:
                registry.register(malicious_input)(SecurityTestAgent)
                # If registration succeeds, clean up
                registry.unregister(malicious_input)
            except AgentRegistrationError:
                # Expected for most malicious inputs
                pass

        # Test malicious config values (should be handled safely)
        registry.register("security_agent")(SecurityTestAgent)

        for malicious_input in security_test_inputs:
            try:
                config = {"agent_id": "security_agent", "malicious_param": malicious_input}
                agent = registry.get_agent("security_agent", config)
                assert agent is not None
            except Exception:  # nosec B110 - Expected for security testing  # noqa: S112
                # Some malicious inputs might cause instantiation errors
                # This is expected behavior for security validation
                continue
