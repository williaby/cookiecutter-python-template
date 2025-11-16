"""
Unit tests for CreateAgent class.

This module provides comprehensive test coverage for the CreateAgent class,
testing C.R.E.A.T.E. framework prompt generation, agent identification,
and configuration management according to development.md standards.
"""

import pytest

from src.agents.create_agent import CreateAgent


@pytest.mark.unit
class TestCreateAgent:
    """Test cases for CreateAgent class."""

    def test_init_default_values(self):
        """Test CreateAgent initialization with proper naming conventions."""
        agent = CreateAgent()

        assert agent.agent_id == "create_agent"
        assert agent.knowledge_base_path == "/knowledge/create_agent/"
        assert agent.qdrant_collection == "create_agent"

    def test_agent_id_follows_naming_convention(self):
        """Test that agent_id follows snake_case convention per development.md."""
        agent = CreateAgent()

        # Should be snake_case format
        assert agent.agent_id == "create_agent"
        assert "_" in agent.agent_id
        assert agent.agent_id.islower()
        assert not agent.agent_id.startswith("_")
        assert not agent.agent_id.endswith("_")

    def test_knowledge_base_path_follows_convention(self):
        """Test knowledge base path follows development.md 3.9 convention."""
        agent = CreateAgent()

        expected_path = f"/knowledge/{agent.agent_id}/"
        assert agent.knowledge_base_path == expected_path
        assert agent.knowledge_base_path.startswith("/knowledge/")
        assert agent.knowledge_base_path.endswith("/")

    def test_qdrant_collection_matches_agent_id(self):
        """Test Qdrant collection name matches agent_id per development.md 3.1."""
        agent = CreateAgent()

        assert agent.qdrant_collection == agent.agent_id
        assert agent.qdrant_collection == "create_agent"

    def test_get_agent_id(self):
        """Test get_agent_id method returns correct identifier."""
        agent = CreateAgent()

        result = agent.get_agent_id()
        assert result == "create_agent"
        assert isinstance(result, str)

    def test_get_agent_id_immutable(self):
        """Test that get_agent_id returns consistent value."""
        agent = CreateAgent()

        # Multiple calls should return the same value
        result1 = agent.get_agent_id()
        result2 = agent.get_agent_id()
        assert result1 == result2
        assert result1 is agent.agent_id  # Should return same object reference

    def test_get_knowledge_path(self):
        """Test get_knowledge_path method returns correct path."""
        agent = CreateAgent()

        result = agent.get_knowledge_path()
        assert result == "/knowledge/create_agent/"
        assert isinstance(result, str)

    def test_get_knowledge_path_immutable(self):
        """Test that get_knowledge_path returns consistent value."""
        agent = CreateAgent()

        # Multiple calls should return the same value
        result1 = agent.get_knowledge_path()
        result2 = agent.get_knowledge_path()
        assert result1 == result2
        assert result1 is agent.knowledge_base_path  # Should return same object reference

    def test_get_qdrant_collection(self):
        """Test get_qdrant_collection method returns correct collection name."""
        agent = CreateAgent()

        result = agent.get_qdrant_collection()
        assert result == "create_agent"
        assert isinstance(result, str)

    def test_get_qdrant_collection_immutable(self):
        """Test that get_qdrant_collection returns consistent value."""
        agent = CreateAgent()

        # Multiple calls should return the same value
        result1 = agent.get_qdrant_collection()
        result2 = agent.get_qdrant_collection()
        assert result1 == result2
        assert result1 is agent.qdrant_collection  # Should return same object reference

    def test_generate_prompt_basic_context(self):
        """Test generate_prompt with basic context."""
        agent = CreateAgent()
        context = {"query": "Generate a Python function", "domain": "programming"}

        result = agent.generate_prompt(context)

        assert isinstance(result, str)
        assert "# Generated C.R.E.A.T.E. Prompt" in result
        assert "Agent: create_agent" in result
        assert str(context) in result
        assert "Preferences:" not in result  # No preferences provided

    def test_generate_prompt_with_preferences(self):
        """Test generate_prompt with context and preferences."""
        agent = CreateAgent()
        context = {"query": "Write documentation", "domain": "technical"}
        preferences = {"tone": "formal", "format": "markdown"}

        result = agent.generate_prompt(context, preferences)

        assert isinstance(result, str)
        assert "# Generated C.R.E.A.T.E. Prompt" in result
        assert "Agent: create_agent" in result
        assert str(context) in result
        assert f"Preferences: {preferences}" in result

    def test_generate_prompt_empty_context(self):
        """Test generate_prompt with empty context."""
        agent = CreateAgent()
        context = {}

        result = agent.generate_prompt(context)

        assert isinstance(result, str)
        assert "# Generated C.R.E.A.T.E. Prompt" in result
        assert "Agent: create_agent" in result
        assert "Context: {}" in result

    def test_generate_prompt_none_preferences(self):
        """Test generate_prompt with None preferences (default)."""
        agent = CreateAgent()
        context = {"task": "summarize document"}

        result = agent.generate_prompt(context, None)

        assert isinstance(result, str)
        assert "Preferences:" not in result
        assert str(context) in result

    def test_generate_prompt_empty_preferences(self):
        """Test generate_prompt with empty preferences dict."""
        agent = CreateAgent()
        context = {"task": "analyze code"}
        preferences = {}

        result = agent.generate_prompt(context, preferences)

        assert isinstance(result, str)
        assert "Preferences:" not in result  # Empty dict is falsy, no preferences section
        assert str(context) in result

    def test_generate_prompt_complex_context(self):
        """Test generate_prompt with complex nested context."""
        agent = CreateAgent()
        context = {
            "query": "Create API documentation",
            "domain": "software development",
            "requirements": {
                "format": "OpenAPI 3.0",
                "authentication": "JWT",
                "endpoints": ["users", "products", "orders"],
            },
            "constraints": ["RESTful", "JSON responses", "rate limiting"],
        }
        preferences = {"style": "comprehensive", "examples": True, "validation": "strict"}

        result = agent.generate_prompt(context, preferences)

        assert isinstance(result, str)
        assert "# Generated C.R.E.A.T.E. Prompt" in result
        assert "Agent: create_agent" in result
        assert "Create API documentation" in result
        assert "OpenAPI 3.0" in result
        assert "comprehensive" in result

    def test_generate_prompt_special_characters(self):
        """Test generate_prompt handles special characters properly."""
        agent = CreateAgent()
        context = {"query": "Process text with symbols: !@#$%^&*()", "special_chars": "Unicode: αβγ δεζ ηθι"}
        preferences = {"encoding": "UTF-8", "escape": "none"}

        result = agent.generate_prompt(context, preferences)

        assert isinstance(result, str)
        assert "!@#$%^&*()" in result
        assert "αβγ δεζ ηθι" in result
        assert "UTF-8" in result

    def test_generate_prompt_large_context(self):
        """Test generate_prompt with large context data."""
        agent = CreateAgent()

        # Create large context with repeated data
        large_data = "x" * 1000
        context = {
            "query": "Process large dataset",
            "data": large_data,
            "metadata": {"size": len(large_data), "type": "text"},
        }
        preferences = {"compression": "enabled", "streaming": True}

        result = agent.generate_prompt(context, preferences)

        assert isinstance(result, str)
        assert large_data in result
        assert "Process large dataset" in result
        assert "compression" in result
        assert len(result) > 1000  # Should contain all the data

    def test_generate_prompt_numeric_values(self):
        """Test generate_prompt with numeric context values."""
        agent = CreateAgent()
        context = {"query": "Calculate statistics", "values": [1, 2, 3, 4, 5], "threshold": 3.14159, "count": 42}
        preferences = {"precision": 2, "format": "scientific"}

        result = agent.generate_prompt(context, preferences)

        assert isinstance(result, str)
        assert "Calculate statistics" in result
        assert "3.14159" in result
        assert "42" in result
        assert "precision" in result

    def test_generate_prompt_boolean_values(self):
        """Test generate_prompt with boolean context values."""
        agent = CreateAgent()
        context = {"query": "Configure settings", "debug_mode": True, "production": False, "auto_save": True}
        preferences = {"strict_mode": False, "validation": True}

        result = agent.generate_prompt(context, preferences)

        assert isinstance(result, str)
        assert "Configure settings" in result
        assert "True" in result
        assert "False" in result

    def test_generate_prompt_maintains_structure(self):
        """Test that generate_prompt maintains expected structure."""
        agent = CreateAgent()
        context = {"task": "test structure"}
        preferences = {"format": "structured"}

        result = agent.generate_prompt(context, preferences)
        lines = result.split("\n")

        # Should have expected structure
        assert lines[0] == "# Generated C.R.E.A.T.E. Prompt"
        assert lines[1] == ""  # Empty line
        assert lines[2].startswith("Agent: ")
        assert lines[3].startswith("Context: ")
        assert lines[4].startswith("Preferences: ")

    def test_multiple_agents_independent(self):
        """Test that multiple CreateAgent instances are independent."""
        agent1 = CreateAgent()
        agent2 = CreateAgent()

        # Should have same values but be different objects
        assert agent1.agent_id == agent2.agent_id
        assert agent1.knowledge_base_path == agent2.knowledge_base_path
        assert agent1.qdrant_collection == agent2.qdrant_collection

        # But should be independent objects
        assert agent1 is not agent2
        assert id(agent1) != id(agent2)

    def test_method_complexity_time_efficiency(self):
        """Test that methods have expected O(1) time complexity."""
        agent = CreateAgent()

        # These should be O(1) operations
        import time

        # Test multiple calls to ensure consistent performance
        start_time = time.perf_counter()
        for _ in range(1000):
            _ = agent.get_agent_id()
            _ = agent.get_knowledge_path()
            _ = agent.get_qdrant_collection()
        end_time = time.perf_counter()

        # Should complete quickly (less than 0.01 seconds for 1000 calls)
        duration = end_time - start_time
        assert duration < 0.01, f"Methods took too long: {duration}s"

    def test_string_representations_valid(self):
        """Test that all string representations are valid and non-empty."""
        agent = CreateAgent()

        # All string methods should return non-empty strings
        assert len(agent.get_agent_id()) > 0
        assert len(agent.get_knowledge_path()) > 0
        assert len(agent.get_qdrant_collection()) > 0

        # Should contain expected characters
        assert all(c.isalnum() or c in "_" for c in agent.get_agent_id())
        assert "/" in agent.get_knowledge_path()

    def test_agent_interface_compatibility(self):
        """Test compatibility with expected agent interface."""
        agent = CreateAgent()

        # Should have all required methods for agent interface
        assert hasattr(agent, "get_agent_id")
        assert hasattr(agent, "get_knowledge_path")
        assert hasattr(agent, "get_qdrant_collection")
        assert hasattr(agent, "generate_prompt")

        # Methods should be callable
        assert callable(agent.get_agent_id)
        assert callable(agent.get_knowledge_path)
        assert callable(agent.get_qdrant_collection)
        assert callable(agent.generate_prompt)

    def test_documentation_completeness(self):
        """Test that the class has proper documentation."""
        agent = CreateAgent()

        # Class should have docstring
        assert CreateAgent.__doc__ is not None
        assert len(CreateAgent.__doc__) > 0

        # Methods should have docstrings
        assert agent.get_agent_id.__doc__ is not None
        assert agent.get_knowledge_path.__doc__ is not None
        assert agent.get_qdrant_collection.__doc__ is not None
        assert agent.generate_prompt.__doc__ is not None
