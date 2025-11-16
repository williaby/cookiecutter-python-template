"""Comprehensive tests for CreateAgent to achieve 100% coverage.

This test module covers all functionality of the CreateAgent class including:
- Initialization and naming conventions
- Agent identification methods
- Prompt generation functionality
- Edge cases and error conditions
"""

from src.agents.create_agent import CreateAgent


class TestCreateAgent:
    """Test suite for CreateAgent class."""

    def test_create_agent_initialization(self):
        """Test CreateAgent initialization with proper naming conventions."""
        agent = CreateAgent()

        # Verify proper initialization according to development.md naming conventions
        assert agent.agent_id == "create_agent"
        assert agent.knowledge_base_path == "/knowledge/create_agent/"
        assert agent.qdrant_collection == "create_agent"

    def test_get_agent_id(self):
        """Test get_agent_id method returns correct snake_case identifier."""
        agent = CreateAgent()

        # Test agent ID retrieval
        agent_id = agent.get_agent_id()
        assert agent_id == "create_agent"
        assert isinstance(agent_id, str)

    def test_get_knowledge_path(self):
        """Test get_knowledge_path method returns correct path format."""
        agent = CreateAgent()

        # Test knowledge path retrieval
        knowledge_path = agent.get_knowledge_path()
        assert knowledge_path == "/knowledge/create_agent/"
        assert isinstance(knowledge_path, str)
        assert knowledge_path.startswith("/knowledge/")
        assert knowledge_path.endswith("/")

    def test_get_qdrant_collection(self):
        """Test get_qdrant_collection method returns correct collection name."""
        agent = CreateAgent()

        # Test Qdrant collection name retrieval
        collection_name = agent.get_qdrant_collection()
        assert collection_name == "create_agent"
        assert isinstance(collection_name, str)

    def test_generate_prompt_basic_context(self):
        """Test generate_prompt with basic context only."""
        agent = CreateAgent()

        # Test basic prompt generation
        context = {"query": "Test query", "domain": "testing"}
        result = agent.generate_prompt(context)

        # Verify prompt structure
        assert isinstance(result, str)
        assert "# Generated C.R.E.A.T.E. Prompt" in result
        assert "Agent: create_agent" in result
        assert "Context: {'query': 'Test query', 'domain': 'testing'}" in result
        assert "Preferences:" not in result  # No preferences provided

    def test_generate_prompt_with_preferences(self):
        """Test generate_prompt with both context and preferences."""
        agent = CreateAgent()

        # Test prompt generation with preferences
        context = {"task": "Generate code", "language": "Python"}
        preferences = {"style": "detailed", "format": "markdown"}
        result = agent.generate_prompt(context, preferences)

        # Verify prompt structure includes preferences
        assert isinstance(result, str)
        assert "# Generated C.R.E.A.T.E. Prompt" in result
        assert "Agent: create_agent" in result
        assert "Context: {'task': 'Generate code', 'language': 'Python'}" in result
        assert "Preferences: {'style': 'detailed', 'format': 'markdown'}" in result

    def test_generate_prompt_empty_context(self):
        """Test generate_prompt with empty context."""
        agent = CreateAgent()

        # Test with empty context
        context = {}
        result = agent.generate_prompt(context)

        # Verify basic structure is maintained
        assert isinstance(result, str)
        assert "# Generated C.R.E.A.T.E. Prompt" in result
        assert "Agent: create_agent" in result
        assert "Context: {}" in result

    def test_generate_prompt_none_preferences(self):
        """Test generate_prompt with None preferences explicitly."""
        agent = CreateAgent()

        # Test with None preferences (explicit)
        context = {"test": "data"}
        result = agent.generate_prompt(context, None)

        # Verify no preferences section is added
        assert isinstance(result, str)
        assert "# Generated C.R.E.A.T.E. Prompt" in result
        assert "Preferences:" not in result

    def test_generate_prompt_empty_preferences(self):
        """Test generate_prompt with empty preferences dict."""
        agent = CreateAgent()

        # Test with empty preferences dict
        context = {"test": "data"}
        preferences = {}
        result = agent.generate_prompt(context, preferences)

        # Verify preferences section is NOT included for empty dict (falsy value)
        assert isinstance(result, str)
        assert "# Generated C.R.E.A.T.E. Prompt" in result
        assert "Preferences:" not in result  # Empty dict is falsy, so no preferences section

    def test_generate_prompt_complex_data(self):
        """Test generate_prompt with complex nested data structures."""
        agent = CreateAgent()

        # Test with complex nested data
        context = {
            "user_query": "Create a REST API",
            "requirements": {
                "framework": "FastAPI",
                "database": "PostgreSQL",
                "features": ["authentication", "rate limiting"],
            },
            "constraints": ["secure", "scalable"],
        }
        preferences = {
            "output_format": "step-by-step",
            "code_style": {"indentation": 4, "naming": "snake_case"},
            "documentation": True,
        }
        result = agent.generate_prompt(context, preferences)

        # Verify complex data is properly included
        assert isinstance(result, str)
        assert "# Generated C.R.E.A.T.E. Prompt" in result
        assert "Agent: create_agent" in result
        assert "REST API" in result
        assert "FastAPI" in result
        assert "step-by-step" in result

    def test_agent_id_consistency(self):
        """Test that agent_id is consistent across all methods."""
        agent = CreateAgent()

        # Verify consistency across all identification methods
        agent_id = agent.get_agent_id()
        knowledge_path = agent.get_knowledge_path()
        collection_name = agent.get_qdrant_collection()

        # All should be based on the same agent_id
        assert agent_id == "create_agent"
        assert f"/knowledge/{agent_id}/" == knowledge_path
        assert agent_id == collection_name

    def test_multiple_instances_independence(self):
        """Test that multiple CreateAgent instances are independent."""
        agent1 = CreateAgent()
        agent2 = CreateAgent()

        # Verify instances are independent but identical
        assert agent1 is not agent2
        assert agent1.agent_id == agent2.agent_id
        assert agent1.get_agent_id() == agent2.get_agent_id()

        # Test that they generate identical output for same input
        context = {"test": "same input"}
        result1 = agent1.generate_prompt(context)
        result2 = agent2.generate_prompt(context)
        assert result1 == result2

    def test_string_methods_return_types(self):
        """Test that all string-returning methods return proper string types."""
        agent = CreateAgent()

        # Verify all methods return strings
        assert isinstance(agent.get_agent_id(), str)
        assert isinstance(agent.get_knowledge_path(), str)
        assert isinstance(agent.get_qdrant_collection(), str)
        assert isinstance(agent.generate_prompt({}), str)

    def test_generate_prompt_special_characters(self):
        """Test generate_prompt handles special characters properly."""
        agent = CreateAgent()

        # Test with special characters and unicode
        context = {
            "special_chars": "!@#$%^&*()",
            "unicode": "café, naïve, résumé",
            "quotes": "He said \"Hello\" and 'Goodbye'",
            "newlines": "Line 1\nLine 2\tTabbed",
        }
        preferences = {"format": 'JSON with "quotes"', "symbols": "Use → arrows and ★ stars"}
        result = agent.generate_prompt(context, preferences)

        # Verify special characters are preserved
        assert isinstance(result, str)
        assert "!@#$%^&*()" in result
        assert "café" in result
        assert "→ arrows" in result
        assert "★ stars" in result
