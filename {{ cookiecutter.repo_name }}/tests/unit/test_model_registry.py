"""Unit tests for ModelRegistry and ModelCapabilities.

Tests cover model configuration loading, capability validation, fallback chain processing,
and integration with environment variables. Ensures 80% coverage requirement.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml

from src.mcp_integration.model_registry import (
    ModelCapabilities,
    ModelRegistry,
    ModelRegistryConfig,
    get_model_registry,
    reload_model_registry,
)


class TestModelCapabilities:
    """Test ModelCapabilities dataclass validation and properties."""

    def test_valid_model_capabilities(self):
        """Test creating valid ModelCapabilities instance."""
        capabilities = ModelCapabilities(
            model_id="test/model:free",
            display_name="Test Model",
            provider="test",
            category="free_general",
            context_window=4096,
            max_tokens_per_request=1024,
            rate_limit_requests_per_minute=30,
        )

        assert capabilities.model_id == "test/model:free"
        assert capabilities.display_name == "Test Model"
        assert capabilities.provider == "test"
        assert capabilities.category == "free_general"
        assert capabilities.context_window == 4096
        assert capabilities.max_tokens_per_request == 1024
        assert capabilities.rate_limit_requests_per_minute == 30
        assert capabilities.timeout_seconds == 30.0  # Default value
        assert capabilities.supports_streaming is True  # Default value
        assert capabilities.enabled is True  # Default value

    def test_model_capabilities_validation_errors(self):
        """Test ModelCapabilities validation with invalid values."""
        # Test negative rate limit
        with pytest.raises(ValueError, match="Rate limit must be positive"):
            ModelCapabilities(
                model_id="test/model",
                display_name="Test",
                provider="test",
                category="free_general",
                context_window=4096,
                max_tokens_per_request=1024,
                rate_limit_requests_per_minute=-1,
            )

        # Test negative context window
        with pytest.raises(ValueError, match="Context window must be positive"):
            ModelCapabilities(
                model_id="test/model",
                display_name="Test",
                provider="test",
                category="free_general",
                context_window=-1,
                max_tokens_per_request=1024,
                rate_limit_requests_per_minute=30,
            )

        # Test negative timeout
        with pytest.raises(ValueError, match="Timeout must be positive"):
            ModelCapabilities(
                model_id="test/model",
                display_name="Test",
                provider="test",
                category="free_general",
                context_window=4096,
                max_tokens_per_request=1024,
                rate_limit_requests_per_minute=30,
                timeout_seconds=-5.0,
            )

    def test_is_free_property(self):
        """Test is_free property logic."""
        # Test free model by name
        free_model = ModelCapabilities(
            model_id="test/model:free",
            display_name="Test",
            provider="test",
            category="free_general",
            context_window=4096,
            max_tokens_per_request=1024,
            rate_limit_requests_per_minute=30,
        )
        assert free_model.is_free is True

        # Test free model by no cost
        free_model_no_cost = ModelCapabilities(
            model_id="test/model",
            display_name="Test",
            provider="test",
            category="free_general",
            context_window=4096,
            max_tokens_per_request=1024,
            rate_limit_requests_per_minute=30,
            cost_per_input_token=None,
            cost_per_output_token=None,
        )
        assert free_model_no_cost.is_free is True

        # Test paid model
        paid_model = ModelCapabilities(
            model_id="test/model:paid",
            display_name="Test",
            provider="test",
            category="premium_general",
            context_window=4096,
            max_tokens_per_request=1024,
            rate_limit_requests_per_minute=30,
            cost_per_input_token=0.001,
            cost_per_output_token=0.002,
        )
        assert paid_model.is_free is False

    def test_estimated_cost_property(self):
        """Test estimated_cost_per_1k_tokens property."""
        # Test free model
        free_model = ModelCapabilities(
            model_id="test/model:free",
            display_name="Test",
            provider="test",
            category="free_general",
            context_window=4096,
            max_tokens_per_request=1024,
            rate_limit_requests_per_minute=30,
        )
        assert free_model.estimated_cost_per_1k_tokens is None

        # Test paid model
        paid_model = ModelCapabilities(
            model_id="test/model:paid",
            display_name="Test",
            provider="test",
            category="premium_general",
            context_window=4096,
            max_tokens_per_request=1024,
            rate_limit_requests_per_minute=30,
            cost_per_input_token=0.001,
            cost_per_output_token=0.002,
        )
        expected_cost = (0.001 + 0.002) * 1000
        assert paid_model.estimated_cost_per_1k_tokens == expected_cost

    def test_model_capabilities_with_warnings(self, caplog):
        """Test ModelCapabilities with warning for free model with cost."""
        ModelCapabilities(
            model_id="test/model:free",
            display_name="Test",
            provider="test",
            category="free_general",
            context_window=4096,
            max_tokens_per_request=1024,
            rate_limit_requests_per_minute=30,
            cost_per_input_token=0.001,  # This should trigger warning
        )
        assert "Free model test/model:free has cost specified" in caplog.text


class TestModelRegistryConfig:
    """Test ModelRegistryConfig validation."""

    def test_valid_config(self):
        """Test valid configuration."""
        config = ModelRegistryConfig(
            default_category="free_general",
            enable_fallback=True,
            max_fallback_depth=3,
        )
        assert config.default_category == "free_general"
        assert config.enable_fallback is True
        assert config.max_fallback_depth == 3

    def test_fallback_depth_validation(self):
        """Test fallback depth validation."""
        # Test negative depth
        with pytest.raises(ValueError, match="Fallback depth must be between 0 and 10"):
            ModelRegistryConfig(max_fallback_depth=-1)

        # Test too large depth
        with pytest.raises(ValueError, match="Fallback depth must be between 0 and 10"):
            ModelRegistryConfig(max_fallback_depth=15)

    def test_unknown_category_warning(self, caplog):
        """Test warning for unknown category."""
        ModelRegistryConfig(default_category="unknown_category")
        assert "Unknown category 'unknown_category'" in caplog.text


class TestModelRegistry:
    """Test ModelRegistry functionality."""

    @pytest.fixture
    def temp_config_file(self):
        """Create temporary YAML config file for testing."""
        config_data = {
            "models": [
                {
                    "model_id": "test/model1:free",
                    "display_name": "Test Model 1",
                    "provider": "test",
                    "category": "free_general",
                    "context_window": 4096,
                    "max_tokens_per_request": 1024,
                    "rate_limit_requests_per_minute": 30,
                    "supports_reasoning": True,
                },
                {
                    "model_id": "test/model2:free",
                    "display_name": "Test Model 2",
                    "provider": "test",
                    "category": "free_reasoning",
                    "context_window": 8192,
                    "max_tokens_per_request": 2048,
                    "rate_limit_requests_per_minute": 20,
                    "supports_vision": True,
                },
            ],
            "fallback_chains": {
                "free_general": ["test/model1:free", "test/model2:free"],
                "free_reasoning": ["test/model2:free", "test/model1:free"],
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        yield temp_path
        Path(temp_path).unlink()

    @patch("src.mcp_integration.model_registry.get_settings")
    def test_registry_initialization_with_config_file(self, mock_get_settings, temp_config_file):
        """Test ModelRegistry initialization with valid config file."""
        mock_get_settings.return_value = Mock()

        registry = ModelRegistry(config_path=temp_config_file)

        assert len(registry._models) == 2
        assert "test/model1:free" in registry._models
        assert "test/model2:free" in registry._models
        assert registry._fallback_chains["free_general"] == ["test/model1:free", "test/model2:free"]

    @patch("src.mcp_integration.model_registry.get_settings")
    def test_registry_initialization_with_missing_config(self, mock_get_settings):
        """Test ModelRegistry initialization when config file is missing."""
        mock_get_settings.return_value = Mock()

        # Use non-existent path
        registry = ModelRegistry(config_path="/nonexistent/path.yaml")

        # Should fall back to default models
        assert len(registry._models) > 0
        assert "deepseek/deepseek-chat-v3-0324:free" in registry._models

    @patch("src.mcp_integration.model_registry.get_settings")
    def test_registry_initialization_with_invalid_config(self, mock_get_settings):
        """Test ModelRegistry initialization with invalid config file."""
        mock_get_settings.return_value = Mock()

        # Create invalid YAML file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_path = f.name

        try:
            registry = ModelRegistry(config_path=temp_path)
            # Should fall back to default models
            assert len(registry._models) > 0
        finally:
            Path(temp_path).unlink()

    @patch("src.mcp_integration.model_registry.get_settings")
    def test_get_model_capabilities(self, mock_get_settings, temp_config_file):
        """Test getting model capabilities."""
        mock_get_settings.return_value = Mock()
        registry = ModelRegistry(config_path=temp_config_file)

        # Test existing model
        capabilities = registry.get_model_capabilities("test/model1:free")
        assert capabilities is not None
        assert capabilities.model_id == "test/model1:free"
        assert capabilities.supports_reasoning is True

        # Test non-existent model
        capabilities = registry.get_model_capabilities("nonexistent/model")
        assert capabilities is None

    @patch("src.mcp_integration.model_registry.get_settings")
    def test_list_models(self, mock_get_settings, temp_config_file):
        """Test listing models with filters."""
        mock_get_settings.return_value = Mock()
        registry = ModelRegistry(config_path=temp_config_file)

        # Test listing all models
        all_models = registry.list_models()
        assert len(all_models) == 2

        # Test filtering by category
        general_models = registry.list_models(category="free_general")
        assert len(general_models) == 1
        assert general_models[0].model_id == "test/model1:free"

        # Test filtering by provider
        test_models = registry.list_models(provider="test")
        assert len(test_models) == 2

    @patch("src.mcp_integration.model_registry.get_settings")
    def test_get_fallback_chain(self, mock_get_settings, temp_config_file):
        """Test getting fallback chains."""
        mock_get_settings.return_value = Mock()
        registry = ModelRegistry(config_path=temp_config_file)

        # Test existing chain
        chain = registry.get_fallback_chain("free_general")
        assert chain == ["test/model1:free", "test/model2:free"]

        # Test non-existent chain
        empty_chain = registry.get_fallback_chain("nonexistent_category")
        assert empty_chain == []

    @patch("src.mcp_integration.model_registry.get_settings")
    def test_fallback_chain_depth_limiting(self, mock_get_settings):
        """Test fallback chain depth limiting."""
        mock_get_settings.return_value = Mock()

        # Mock config with max depth of 2
        with patch.object(ModelRegistry, "_load_config") as mock_load_config:
            mock_config = ModelRegistryConfig(max_fallback_depth=2)
            mock_load_config.return_value = mock_config

            registry = ModelRegistry()
            registry._fallback_chains = {"test_category": ["model1", "model2", "model3", "model4"]}
            registry._models = {
                "model1": Mock(enabled=True),
                "model2": Mock(enabled=True),
                "model3": Mock(enabled=True),
                "model4": Mock(enabled=True),
            }

            chain = registry.get_fallback_chain("test_category")
            assert len(chain) == 2  # Limited by max_fallback_depth

    @patch("src.mcp_integration.model_registry.get_settings")
    def test_convert_model_name(self, mock_get_settings, temp_config_file):
        """Test model name conversion with aliases."""
        mock_get_settings.return_value = Mock()
        registry = ModelRegistry(config_path=temp_config_file)

        # Test direct model ID
        result = registry.convert_model_name("test/model1:free")
        assert result == "test/model1:free"

        # Test alias (should fallback to first available model since deepseek isn't in test config)
        result = registry.convert_model_name("deepseek")
        assert result == "test/model1:free"  # First in fallback chain

        # Test unknown model (should fallback)
        result = registry.convert_model_name("unknown/model")
        assert result == "test/model1:free"  # First in fallback chain

    @patch("src.mcp_integration.model_registry.get_settings")
    def test_select_best_model(self, mock_get_settings, temp_config_file):
        """Test smart model selection."""
        mock_get_settings.return_value = Mock()
        registry = ModelRegistry(config_path=temp_config_file)

        # Test reasoning task - should pick model with supports_reasoning=True
        model = registry.select_best_model("reasoning", allow_premium=False)
        # In our temp config, model1 has supports_reasoning=True, model2 has supports_vision=True
        # The reasoning category should find the first model that supports reasoning
        # Looking at our config: model1 has supports_reasoning=True
        assert model == "test/model1:free"  # Model1 has supports_reasoning in config

        # Test vision task - should pick model with supports_vision=True
        model = registry.select_best_model("vision", allow_premium=False)
        assert model == "test/model2:free"  # Has supports_vision in config

        # Test general task - should pick from free_general chain
        model = registry.select_best_model("general", allow_premium=False)
        assert model == "test/model1:free"  # First in free_general chain

    @patch("src.mcp_integration.model_registry.get_settings")
    def test_get_rate_limit(self, mock_get_settings, temp_config_file):
        """Test getting rate limits."""
        mock_get_settings.return_value = Mock()
        registry = ModelRegistry(config_path=temp_config_file)

        # Test existing model
        rate_limit = registry.get_rate_limit("test/model1:free")
        assert rate_limit == 30

        # Test non-existent model (should return default)
        rate_limit = registry.get_rate_limit("nonexistent/model")
        assert rate_limit == 20  # Default value

    @patch("src.mcp_integration.model_registry.get_settings")
    def test_is_model_available(self, mock_get_settings, temp_config_file):
        """Test model availability checking."""
        mock_get_settings.return_value = Mock()
        registry = ModelRegistry(config_path=temp_config_file)

        # Test available model
        assert registry.is_model_available("test/model1:free") is True

        # Test non-existent model
        assert registry.is_model_available("nonexistent/model") is False

    @patch("src.mcp_integration.model_registry.get_settings")
    def test_get_model_cost(self, mock_get_settings):
        """Test model cost calculation."""
        mock_get_settings.return_value = Mock()
        registry = ModelRegistry()

        # Add a paid model for testing
        paid_model = ModelCapabilities(
            model_id="test/paid:model",
            display_name="Test Paid",
            provider="test",
            category="premium_general",
            context_window=4096,
            max_tokens_per_request=1024,
            rate_limit_requests_per_minute=100,
            cost_per_input_token=0.001,
            cost_per_output_token=0.002,
        )
        registry._models["test/paid:model"] = paid_model

        # Test paid model cost calculation
        cost = registry.get_model_cost("test/paid:model", 1000, 500)
        expected_cost = (1000 * 0.001) + (500 * 0.002)
        assert cost == expected_cost

        # Test free model (should return None)
        cost = registry.get_model_cost("deepseek/deepseek-chat-v3-0324:free", 1000, 500)
        assert cost is None

    @patch("src.mcp_integration.model_registry.get_settings")
    def test_reload_config(self, mock_get_settings, temp_config_file):
        """Test configuration reloading."""
        mock_get_settings.return_value = Mock()
        registry = ModelRegistry(config_path=temp_config_file)

        original_count = len(registry._models)

        # Reload should work without errors
        registry.reload_config()

        # Should have same number of models after reload
        assert len(registry._models) == original_count

    @patch("src.mcp_integration.model_registry.get_settings")
    def test_environment_variable_integration(self, mock_get_settings):
        """Test environment variable integration."""
        mock_settings = Mock()
        mock_get_settings.return_value = mock_settings

        with patch.dict(
            os.environ,
            {
                "PROMPTCRAFT_DEFAULT_MODEL_CATEGORY": "free_reasoning",
                "PROMPTCRAFT_ENABLE_MODEL_FALLBACK": "false",
                "PROMPTCRAFT_MAX_FALLBACK_DEPTH": "5",
            },
        ):
            registry = ModelRegistry()

            assert registry._config.default_category == "free_reasoning"
            assert registry._config.enable_fallback is False
            assert registry._config.max_fallback_depth == 5

    @patch("src.mcp_integration.model_registry.get_settings")
    def test_disabled_model_filtering(self, mock_get_settings):
        """Test that disabled models are filtered from results."""
        mock_get_settings.return_value = Mock()
        registry = ModelRegistry()

        # Add a disabled model
        disabled_model = ModelCapabilities(
            model_id="test/disabled:model",
            display_name="Disabled Model",
            provider="test",
            category="free_general",
            context_window=4096,
            max_tokens_per_request=1024,
            rate_limit_requests_per_minute=30,
            enabled=False,  # Disabled
        )
        registry._models["test/disabled:model"] = disabled_model

        # Should not appear in list_models
        models = registry.list_models()
        model_ids = [m.model_id for m in models]
        assert "test/disabled:model" not in model_ids

        # Should not be available
        assert registry.is_model_available("test/disabled:model") is False

    @patch("src.mcp_integration.model_registry.get_settings")
    def test_select_best_model_with_context_requirement(self, mock_get_settings, temp_config_file):
        """Test model selection with context window requirements."""
        mock_get_settings.return_value = Mock()
        registry = ModelRegistry(config_path=temp_config_file)

        # Test with context requirement that model2 meets but model1 doesn't
        model = registry.select_best_model(
            "general",
            allow_premium=False,
            max_tokens_needed=6000,  # model1 has 4096, model2 has 8192
        )
        assert model == "test/model2:free"


class TestGlobalRegistry:
    """Test global registry functions."""

    def test_get_model_registry_singleton(self):
        """Test global registry singleton behavior."""
        # Clear any existing global registry
        import src.mcp_integration.model_registry as registry_module

        registry_module._global_registry = None

        # First call should create instance
        registry1 = get_model_registry()
        assert registry1 is not None

        # Second call should return same instance
        registry2 = get_model_registry()
        assert registry1 is registry2

    def test_reload_model_registry(self):
        """Test global registry reload."""
        # Clear any existing global registry
        import src.mcp_integration.model_registry as registry_module

        registry_module._global_registry = None

        # Should work even if no registry exists yet
        reload_model_registry()

        # Should create registry
        registry = get_model_registry()
        assert registry is not None

        # Reload should work on existing registry
        reload_model_registry()


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @patch("src.mcp_integration.model_registry.get_settings")
    def test_empty_yaml_config(self, mock_get_settings):
        """Test handling of empty YAML config."""
        mock_get_settings.return_value = Mock()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({}, f)  # Empty config
            temp_path = f.name

        try:
            registry = ModelRegistry(config_path=temp_path)
            # Should fall back to default models
            assert len(registry._models) > 0
        finally:
            Path(temp_path).unlink()

    @patch("src.mcp_integration.model_registry.get_settings")
    def test_malformed_model_in_config(self, mock_get_settings):
        """Test handling of malformed model configuration."""
        mock_get_settings.return_value = Mock()

        config_data = {
            "models": [
                {
                    "model_id": "valid/model:free",
                    "display_name": "Valid Model",
                    "provider": "test",
                    "category": "free_general",
                    "context_window": 4096,
                    "max_tokens_per_request": 1024,
                    "rate_limit_requests_per_minute": 30,
                },
                {
                    "model_id": "invalid/model",
                    # Missing required fields
                },
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            registry = ModelRegistry(config_path=temp_path)
            # Should load valid model and skip invalid one
            assert "valid/model:free" in registry._models
            assert "invalid/model" not in registry._models
        finally:
            Path(temp_path).unlink()

    @patch("src.mcp_integration.model_registry.get_settings")
    def test_select_best_model_no_candidates(self, mock_get_settings):
        """Test model selection when no candidates meet criteria."""
        mock_get_settings.return_value = Mock()
        registry = ModelRegistry()

        # Clear all models to test fallback
        registry._models.clear()

        # Should return ultimate fallback
        model = registry.select_best_model("any_task")
        assert model == "deepseek/deepseek-chat-v3-0324:free"
