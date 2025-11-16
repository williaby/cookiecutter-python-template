"""
Unit tests for registry collision scenarios.

This module tests collision scenarios in agent registration, including
duplicate agent IDs, namespace conflicts, capability conflicts, and
other edge cases that could occur in the registry system.
"""

import threading
import time
from typing import Any

import pytest

from src.agents.base_agent import BaseAgent
from src.agents.exceptions import AgentRegistrationError
from src.agents.models import AgentInput, AgentOutput
from src.agents.registry import AgentRegistry, agent_registry


class TestRegistryCollisionScenarios:
    """Test various collision scenarios in agent registry."""

    @pytest.fixture
    def registry(self):
        """Fresh registry for testing."""
        registry = AgentRegistry()
        yield registry
        registry.clear()

    @pytest.fixture
    def sample_agent_classes(self):
        """Multiple agent classes for collision testing."""

        class FirstAgent(BaseAgent):
            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                return AgentOutput(
                    content="First agent response",
                    confidence=0.9,
                    processing_time=0.1,
                    agent_id=self.agent_id,
                )

            def get_capabilities(self) -> dict[str, Any]:
                return {"input_types": ["text"], "output_types": ["text"], "specialization": "first_specialty"}

        class SecondAgent(BaseAgent):
            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                return AgentOutput(
                    content="Second agent response",
                    confidence=0.8,
                    processing_time=0.2,
                    agent_id=self.agent_id,
                )

            def get_capabilities(self) -> dict[str, Any]:
                return {"input_types": ["text"], "output_types": ["text"], "specialization": "second_specialty"}

        class ConflictingAgent(BaseAgent):
            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                return AgentOutput(
                    content="Conflicting agent response",
                    confidence=0.7,
                    processing_time=0.15,
                    agent_id=self.agent_id,
                )

            def get_capabilities(self) -> dict[str, Any]:
                return {
                    "input_types": ["text"],
                    "output_types": ["text"],
                    "specialization": "first_specialty",  # Same as FirstAgent
                }

        return {"FirstAgent": FirstAgent, "SecondAgent": SecondAgent, "ConflictingAgent": ConflictingAgent}

    @pytest.mark.unit
    def test_duplicate_agent_id_collision(self, registry, sample_agent_classes):
        """Test collision when registering agents with duplicate IDs."""
        # Register first agent
        registry.register("duplicate_id")(sample_agent_classes["FirstAgent"])

        # Verify first registration succeeded
        assert "duplicate_id" in registry
        assert len(registry) == 1

        # Attempt to register second agent with same ID
        with pytest.raises(AgentRegistrationError) as excinfo:
            registry.register("duplicate_id")(sample_agent_classes["SecondAgent"])

        assert excinfo.value.error_code == "DUPLICATE_AGENT_ID"
        assert "duplicate_id" in excinfo.value.message
        assert excinfo.value.agent_id == "duplicate_id"

        # Verify registry state unchanged
        assert len(registry) == 1
        retrieved_class = registry.get_agent_class("duplicate_id")
        assert retrieved_class == sample_agent_classes["FirstAgent"]

    @pytest.mark.unit
    def test_case_sensitive_agent_id_collision(self, registry, sample_agent_classes):
        """Test that agent IDs are case sensitive."""
        # Register agents with different cases
        registry.register("test_agent")(sample_agent_classes["FirstAgent"])
        registry.register("TEST_AGENT")(sample_agent_classes["SecondAgent"])
        registry.register("Test_Agent")(sample_agent_classes["ConflictingAgent"])

        # All should be registered successfully
        assert len(registry) == 3
        assert "test_agent" in registry
        assert "TEST_AGENT" in registry
        assert "Test_Agent" in registry

        # Each should retrieve the correct class
        assert registry.get_agent_class("test_agent") == sample_agent_classes["FirstAgent"]
        assert registry.get_agent_class("TEST_AGENT") == sample_agent_classes["SecondAgent"]
        assert registry.get_agent_class("Test_Agent") == sample_agent_classes["ConflictingAgent"]

    @pytest.mark.unit
    def test_similar_agent_id_collision_prevention(self, registry, sample_agent_classes):
        """Test registration of agents with similar but different IDs."""
        similar_ids = [
            "text_processor",
            "text_processor_v2",
            "text_processor_advanced",
            "text_processor_1",
            "textprocessor",
            "text_processor_",
        ]

        # Register all similar IDs
        for i, agent_id in enumerate(similar_ids):
            agent_class = sample_agent_classes["FirstAgent"] if i % 2 == 0 else sample_agent_classes["SecondAgent"]
            registry.register(agent_id)(agent_class)

        # All should be registered successfully
        assert len(registry) == len(similar_ids)

        # Each should be retrievable
        for agent_id in similar_ids:
            assert agent_id in registry
            assert registry.get_agent_class(agent_id) is not None

    @pytest.mark.unit
    def test_capability_collision_handling(self, registry, sample_agent_classes):
        """Test handling of agents with conflicting capabilities."""
        # Register agents with conflicting capabilities
        registry.register("first_agent")(sample_agent_classes["FirstAgent"])
        registry.register("conflicting_agent")(sample_agent_classes["ConflictingAgent"])

        # Both should be registered successfully
        assert len(registry) == 2

        # Create instances to populate capabilities
        registry.get_agent("first_agent", {"agent_id": "first_agent"})
        registry.get_agent("conflicting_agent", {"agent_id": "conflicting_agent"})

        # Both should be found when searching by specialization
        specialty_agents = registry.find_agents_by_capability("specialization", "first_specialty")
        assert len(specialty_agents) == 2
        assert set(specialty_agents) == {"first_agent", "conflicting_agent"}

    @pytest.mark.unit
    def test_namespace_collision_simulation(self, registry, sample_agent_classes):
        """Test simulation of namespace collisions."""
        # Simulate namespace-like agent IDs
        namespace_ids = ["org_team_agent", "org_team_agent_v2", "org_other_agent", "different_org_agent"]

        # Register all namespace-like IDs
        for agent_id in namespace_ids:
            registry.register(agent_id)(sample_agent_classes["FirstAgent"])

        # Test prefix-based retrieval simulation
        org_team_agents = [aid for aid in registry.list_agents() if aid.startswith("org_team")]
        assert len(org_team_agents) == 2
        assert "org_team_agent" in org_team_agents
        assert "org_team_agent_v2" in org_team_agents

        # Test organization-based retrieval simulation
        org_agents = [aid for aid in registry.list_agents() if aid.startswith("org_")]
        assert len(org_agents) == 3

    @pytest.mark.unit
    def test_concurrent_registration_collision(self, registry, sample_agent_classes):
        """Test collision detection during concurrent registration attempts."""

        registration_results = []
        registration_errors = []

        def register_agent(agent_id, agent_class):
            try:
                # Add small delay to increase collision probability
                time.sleep(0.01)
                registry.register(agent_id)(agent_class)
                registration_results.append(agent_id)
            except AgentRegistrationError as e:
                registration_errors.append((agent_id, e))

        # Create multiple threads trying to register the same agent ID
        threads = []
        for _i in range(5):
            thread = threading.Thread(
                target=register_agent,
                args=("concurrent_agent", sample_agent_classes["FirstAgent"]),
            )
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Only one registration should succeed
        assert len(registration_results) == 1
        assert len(registration_errors) == 4

        # All errors should be duplicate ID errors
        for agent_id, error in registration_errors:
            assert error.error_code == "DUPLICATE_AGENT_ID"
            assert agent_id == "concurrent_agent"

    @pytest.mark.unit
    def test_registration_rollback_on_collision(self, registry, sample_agent_classes):
        """Test that registration is properly rolled back on collision."""
        # Register first agent
        registry.register("rollback_test")(sample_agent_classes["FirstAgent"])

        # Store initial state
        initial_count = len(registry)
        initial_agents = registry.list_agents()

        # Attempt to register duplicate
        with pytest.raises(AgentRegistrationError):
            registry.register("rollback_test")(sample_agent_classes["SecondAgent"])

        # Verify state is unchanged
        assert len(registry) == initial_count
        assert registry.list_agents() == initial_agents

        # Verify original agent is still accessible
        original_class = registry.get_agent_class("rollback_test")
        assert original_class == sample_agent_classes["FirstAgent"]

    @pytest.mark.unit
    def test_global_registry_collision_isolation(self, sample_agent_classes):
        """Test collision isolation between different registry instances."""
        # Use global registry

        # Store original state
        _original_agents = list(agent_registry.list_agents())

        try:
            # Create local registry
            local_registry = AgentRegistry()

            # Register same agent ID in both registries
            agent_registry.register("collision_test")(sample_agent_classes["FirstAgent"])
            local_registry.register("collision_test")(sample_agent_classes["SecondAgent"])

            # Both registrations should succeed
            assert "collision_test" in agent_registry
            assert "collision_test" in local_registry

            # Each should have different implementations
            global_class = agent_registry.get_agent_class("collision_test")
            local_class = local_registry.get_agent_class("collision_test")

            assert global_class == sample_agent_classes["FirstAgent"]
            assert local_class == sample_agent_classes["SecondAgent"]

            # Clear local registry
            local_registry.clear()

        finally:
            # Cleanup global registry
            if "collision_test" in agent_registry:
                agent_registry.unregister("collision_test")

    @pytest.mark.unit
    def test_agent_id_format_collision_prevention(self, registry, sample_agent_classes):
        """Test prevention of agent ID format collisions."""
        # Test various invalid formats that might cause collisions
        invalid_ids = [
            "agent-with-dashes",
            "agent.with.dots",
            "agent with spaces",
            "agent/with/slashes",
            "agent\\with\\backslashes",
            "agent:with:colons",
        ]

        for invalid_id in invalid_ids:
            with pytest.raises(AgentRegistrationError) as excinfo:
                registry.register(invalid_id)(sample_agent_classes["FirstAgent"])

            # Should fail during agent registration with invalid ID
            assert excinfo.value.error_code == "INVALID_AGENT_ID"

    @pytest.mark.unit
    def test_capability_conflict_resolution(self, registry, sample_agent_classes):
        """Test resolution of capability conflicts."""
        # Register multiple agents with overlapping capabilities
        registry.register("text_agent_1")(sample_agent_classes["FirstAgent"])
        registry.register("text_agent_2")(sample_agent_classes["SecondAgent"])
        registry.register("text_agent_3")(sample_agent_classes["ConflictingAgent"])

        # Create instances to populate capabilities
        for agent_id in ["text_agent_1", "text_agent_2", "text_agent_3"]:
            registry.get_agent(agent_id, {"agent_id": agent_id})

        # Test capability discovery with conflicts
        text_agents = registry.find_agents_by_type("text")
        assert len(text_agents) == 3

        # Test specific capability filtering
        first_specialty_agents = registry.find_agents_by_capability("specialization", "first_specialty")
        second_specialty_agents = registry.find_agents_by_capability("specialization", "second_specialty")

        # Should handle capability conflicts correctly
        assert len(first_specialty_agents) == 2  # FirstAgent and ConflictingAgent
        assert len(second_specialty_agents) == 1  # SecondAgent only

        assert "text_agent_1" in first_specialty_agents
        assert "text_agent_3" in first_specialty_agents
        assert "text_agent_2" in second_specialty_agents

    @pytest.mark.unit
    def test_registry_state_consistency_after_collisions(self, registry, sample_agent_classes):
        """Test registry state consistency after handling collisions."""
        # Register several agents
        successful_registrations = []

        for i in range(5):
            agent_id = f"consistent_agent_{i}"
            registry.register(agent_id)(sample_agent_classes["FirstAgent"])
            successful_registrations.append(agent_id)

        # Attempt some collisions
        for i in range(3):
            with pytest.raises(AgentRegistrationError):
                registry.register(f"consistent_agent_{i}")(sample_agent_classes["SecondAgent"])

        # Verify state consistency
        assert len(registry) == 5
        assert set(registry.list_agents()) == set(successful_registrations)

        # Test registry operations still work correctly
        status = registry.get_registry_status()
        assert status["total_agents"] == 5
        assert status["cached_instances"] == 0  # No instances created yet

        # Create instances and verify they work
        for agent_id in successful_registrations:
            agent = registry.get_cached_agent(agent_id, {"agent_id": agent_id})
            assert agent.agent_id == agent_id

        # Verify cached instances updated
        status_after = registry.get_registry_status()
        assert status_after["cached_instances"] == 5

    @pytest.mark.unit
    def test_decorator_collision_behavior(self, registry, sample_agent_classes):
        """Test decorator behavior during collisions."""

        # Register first agent using decorator
        @registry.register("decorator_test")
        class DecoratorAgent(BaseAgent):
            async def execute(self, agent_input: AgentInput) -> AgentOutput:
                return AgentOutput(
                    content="Decorator agent",
                    confidence=1.0,
                    processing_time=0.1,
                    agent_id=self.agent_id,
                )

        # Verify decorator returns the class
        assert DecoratorAgent is not None
        assert "decorator_test" in registry

        # Attempt to register another agent with same ID
        with pytest.raises(AgentRegistrationError):

            @registry.register("decorator_test")
            class AnotherDecoratorAgent(BaseAgent):
                async def execute(self, agent_input: AgentInput) -> AgentOutput:
                    return AgentOutput(
                        content="Another decorator agent",
                        confidence=1.0,
                        processing_time=0.1,
                        agent_id=self.agent_id,
                    )

        # Original agent should still be registered
        assert registry.get_agent_class("decorator_test") == DecoratorAgent

    @pytest.mark.unit
    def test_mass_registration_collision_handling(self, registry, sample_agent_classes):
        """Test collision handling during mass registration."""
        # Register many agents with some collisions
        successful_count = 0
        collision_count = 0

        for i in range(100):
            agent_id = f"mass_agent_{i % 50}"  # Will create collisions after 50

            try:
                registry.register(agent_id)(sample_agent_classes["FirstAgent"])
                successful_count += 1
            except AgentRegistrationError:
                collision_count += 1

        # Should have exactly 50 successful registrations
        assert successful_count == 50
        assert collision_count == 50
        assert len(registry) == 50

        # Verify all registered agents are accessible
        for i in range(50):
            agent_id = f"mass_agent_{i}"
            assert agent_id in registry
            agent_class = registry.get_agent_class(agent_id)
            assert agent_class == sample_agent_classes["FirstAgent"]

    @pytest.mark.unit
    def test_unregister_after_collision(self, registry, sample_agent_classes):
        """Test unregistration after collision attempts."""
        # Register original agent
        registry.register("unregister_test")(sample_agent_classes["FirstAgent"])

        # Attempt collision
        with pytest.raises(AgentRegistrationError):
            registry.register("unregister_test")(sample_agent_classes["SecondAgent"])

        # Unregister original
        registry.unregister("unregister_test")
        assert "unregister_test" not in registry

        # Now should be able to register new agent with same ID
        registry.register("unregister_test")(sample_agent_classes["SecondAgent"])
        assert "unregister_test" in registry

        # Verify new agent is registered
        new_class = registry.get_agent_class("unregister_test")
        assert new_class == sample_agent_classes["SecondAgent"]
