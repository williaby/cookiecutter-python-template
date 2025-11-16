"""Model Registry and Configuration System for AI Tool Routing Implementation.

This module provides centralized model configuration management, extending the patterns
from .claude/commands/shared/model_utils.sh for Python integration. It supports
OpenRouter API integration with hybrid routing, rate limiting, and fallback chains.

Architecture:
    - ModelCapabilities: Dataclass defining model characteristics and constraints
    - ModelRegistry: Central registry for model configuration and selection
    - YAML configuration loading with validation
    - Environment variable integration following .env.local patterns

Key Features:
    - Rate limit enforcement per OpenRouter documentation
    - Context window and cost tracking for optimal model selection
    - Fallback chain logic for resilient model routing
    - Environment-based configuration with gradual rollout support
    - Compatibility with existing model_utils.sh patterns

Supported Models (Phase 1):
    - deepseek/deepseek-chat-v3-0324:free (DeepSeek V3)
    - google/gemini-2.0-flash-exp:free (Gemini 2.0 Flash)
    - qwen/qwen3-32b:free (Qwen 32B)

Called by:
    - AI Tool Routing: For model selection and capability checking
    - MCP Integration: For model-specific configurations
    - Hybrid routing logic: For fallback chain management

Dependencies:
    - pydantic: For data validation and settings management
    - PyYAML: For configuration file loading
    - src.config.settings: For environment variable integration

Environment Variables:
    - PROMPTCRAFT_OPENROUTER_API_KEY: OpenRouter API authentication
    - PROMPTCRAFT_MODEL_REGISTRY_CONFIG: Custom registry configuration path
    - PROMPTCRAFT_ENABLE_MODEL_FALLBACK: Enable/disable fallback chains
    - PROMPTCRAFT_DEFAULT_MODEL_CATEGORY: Default model category for selection

Example Usage:
    ```python
    registry = ModelRegistry()

    # Get model capabilities
    capabilities = registry.get_model_capabilities("deepseek/deepseek-chat-v3-0324:free")
    print(f"Rate limit: {capabilities.rate_limit_requests_per_minute}")

    # Get fallback chain for free models
    chain = registry.get_fallback_chain("free_general")

    # Select best available model
    selected = registry.select_best_model("reasoning", allow_premium=False)
    ```

Complexity: O(1) for model lookups, O(n) for fallback chain processing where n is chain length
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

from src.config.settings import get_settings

logger = logging.getLogger(__name__)

# Model registry constants
MAX_FALLBACK_DEPTH = 10  # Maximum depth for fallback chains to prevent infinite loops


@dataclass
class ModelCapabilities:
    """Model capability specifications for AI tool routing.

    Defines comprehensive model characteristics including performance constraints,
    cost metrics, and technical specifications. Based on OpenRouter API documentation
    and extended for hybrid routing needs.

    Attributes:
        model_id: Unique identifier for the model (e.g., "deepseek/deepseek-chat-v3-0324:free")
        display_name: Human-readable model name for UI display
        provider: Model provider (e.g., "deepseek", "google", "qwen")
        category: Model category for fallback chains ("free_general", "premium_reasoning", etc.)
        context_window: Maximum context length in tokens
        max_tokens_per_request: Maximum output tokens per request
        rate_limit_requests_per_minute: Requests per minute rate limit
        rate_limit_tokens_per_minute: Tokens per minute rate limit (if applicable)
        cost_per_input_token: Cost per input token in USD (None for free models)
        cost_per_output_token: Cost per output token in USD (None for free models)
        timeout_seconds: Request timeout in seconds
        supports_streaming: Whether model supports streaming responses
        supports_function_calling: Whether model supports function/tool calling
        supports_vision: Whether model supports image inputs
        supports_reasoning: Whether model has enhanced reasoning capabilities
        available_regions: List of geographic regions where model is available
        fallback_models: Ordered list of fallback model IDs
        enabled: Whether model is currently enabled for use
        metadata: Additional provider-specific metadata
    """

    model_id: str
    display_name: str
    provider: str
    category: str
    context_window: int
    max_tokens_per_request: int
    rate_limit_requests_per_minute: int
    rate_limit_tokens_per_minute: int | None = None
    cost_per_input_token: float | None = None
    cost_per_output_token: float | None = None
    timeout_seconds: float = 30.0
    supports_streaming: bool = True
    supports_function_calling: bool = False
    supports_vision: bool = False
    supports_reasoning: bool = False
    available_regions: list[str] = field(default_factory=lambda: ["global"])
    fallback_models: list[str] = field(default_factory=list)
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate model capabilities after initialization."""
        if self.rate_limit_requests_per_minute <= 0:
            raise ValueError(f"Rate limit must be positive: {self.rate_limit_requests_per_minute}")

        if self.context_window <= 0:
            raise ValueError(f"Context window must be positive: {self.context_window}")

        if self.timeout_seconds <= 0:
            raise ValueError(f"Timeout must be positive: {self.timeout_seconds}")

        # Validate cost consistency for free models
        if "free" in self.model_id.lower():
            if self.cost_per_input_token is not None or self.cost_per_output_token is not None:
                logger.warning(f"Free model {self.model_id} has cost specified")

    @property
    def is_free(self) -> bool:
        """Check if model is free to use."""
        return "free" in self.model_id.lower() or (
            self.cost_per_input_token is None and self.cost_per_output_token is None
        )

    @property
    def estimated_cost_per_1k_tokens(self) -> float | None:
        """Estimate cost per 1000 tokens assuming 1:1 input/output ratio."""
        if self.cost_per_input_token is None or self.cost_per_output_token is None:
            return None
        return (self.cost_per_input_token + self.cost_per_output_token) * 1000


class ModelRegistryConfig(BaseModel):
    """Configuration for ModelRegistry with validation.

    Supports loading from YAML files and environment variables with
    comprehensive validation and sensible defaults.
    """

    default_category: str = Field(default="free_general", description="Default model category")
    enable_fallback: bool = Field(default=True, description="Enable fallback chain processing")
    max_fallback_depth: int = Field(default=3, description="Maximum fallback chain depth")
    openrouter_api_key: str | None = Field(default=None, description="OpenRouter API key")
    custom_config_path: str | None = Field(default=None, description="Custom model config file path")

    @field_validator("max_fallback_depth")
    @classmethod
    def validate_fallback_depth(cls, v: int) -> int:
        if v < 0 or v > MAX_FALLBACK_DEPTH:
            raise ValueError(f"Fallback depth must be between 0 and {MAX_FALLBACK_DEPTH}")
        return v

    @field_validator("default_category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        valid_categories = {"free_general", "free_reasoning", "premium_reasoning", "premium_analysis", "large_context"}
        if v not in valid_categories:
            logger.warning(f"Unknown category '{v}', valid options: {valid_categories}")
        return v


class ModelRegistry:
    """Central registry for AI model configuration and selection.

    Provides unified access to model capabilities, fallback chains, and routing logic.
    Extends the patterns from .claude/commands/shared/model_utils.sh for Python integration
    with enhanced configuration management and validation.

    Features:
        - Model capability lookup and validation
        - Fallback chain processing with depth limits
        - Environment-based configuration loading
        - Rate limit and cost tracking
        - Smart model selection based on task requirements
        - Integration with existing model_utils.sh patterns

    Architecture:
        The registry loads model configurations from YAML files and provides
        both direct lookup and smart selection capabilities. It maintains
        compatibility with the bash model utilities while adding Python-specific
        enhancements.
    """

    def __init__(self, config_path: str | Path | None = None) -> None:
        """Initialize ModelRegistry with configuration.

        Args:
            config_path: Optional path to model configuration YAML file.
                        Defaults to config/openrouter_models.yaml
        """
        self.logger = logging.getLogger(__name__ + ".ModelRegistry")
        self._models: dict[str, ModelCapabilities] = {}
        self._fallback_chains: dict[str, list[str]] = {}
        self._config: ModelRegistryConfig = self._load_config()

        # Determine configuration file path
        if config_path:
            self._config_path = Path(config_path)
        elif self._config.custom_config_path:
            self._config_path = Path(self._config.custom_config_path)
        else:
            # Default to project config directory
            project_root = Path(__file__).parent.parent.parent
            self._config_path = project_root / "config" / "openrouter_models.yaml"

        # Load model configurations
        self._load_models()
        self.logger.info(f"ModelRegistry initialized with {len(self._models)} models")

    def _load_config(self) -> ModelRegistryConfig:
        """Load registry configuration from environment variables."""
        get_settings()

        return ModelRegistryConfig(
            default_category=os.getenv("PROMPTCRAFT_DEFAULT_MODEL_CATEGORY", "free_general"),
            enable_fallback=os.getenv("PROMPTCRAFT_ENABLE_MODEL_FALLBACK", "true").lower() == "true",
            max_fallback_depth=int(os.getenv("PROMPTCRAFT_MAX_FALLBACK_DEPTH", "3")),
            openrouter_api_key=os.getenv("PROMPTCRAFT_OPENROUTER_API_KEY"),
            custom_config_path=os.getenv("PROMPTCRAFT_MODEL_REGISTRY_CONFIG"),
        )

    def _load_models(self) -> None:
        """Load model configurations from YAML file."""
        try:
            if not self._config_path.exists():
                self.logger.warning(f"Model config file not found: {self._config_path}")
                self._load_default_models()
                return

            with self._config_path.open(encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

            if not config_data or "models" not in config_data:
                self.logger.error("Invalid model configuration format")
                self._load_default_models()
                return

            # Load models from configuration
            for model_data in config_data["models"]:
                try:
                    capabilities = ModelCapabilities(**model_data)
                    self._models[capabilities.model_id] = capabilities
                except Exception as e:
                    self.logger.error(f"Failed to load model {model_data.get('model_id', 'unknown')}: {e}")

            # Load fallback chains
            if "fallback_chains" in config_data:
                self._fallback_chains = config_data["fallback_chains"]
            else:
                self._load_default_fallback_chains()

            self.logger.info(f"Loaded {len(self._models)} models from {self._config_path}")

        except Exception as e:
            self.logger.error(f"Failed to load model configuration: {e}")
            self._load_default_models()

    def _load_default_models(self) -> None:
        """Load default model configurations when YAML file is unavailable."""
        self.logger.info("Loading default model configurations")

        # Default models based on model_utils.sh patterns
        default_models = [
            ModelCapabilities(
                model_id="deepseek/deepseek-chat-v3-0324:free",
                display_name="DeepSeek V3 Chat (Free)",
                provider="deepseek",
                category="free_general",
                context_window=163840,  # 163K context
                max_tokens_per_request=8192,
                rate_limit_requests_per_minute=20,  # Conservative default for free tier
                timeout_seconds=30.0,
                supports_streaming=True,
                supports_function_calling=False,
                supports_reasoning=True,
                fallback_models=["google/gemini-2.0-flash-exp:free", "qwen/qwen3-32b:free"],
            ),
            ModelCapabilities(
                model_id="google/gemini-2.0-flash-exp:free",
                display_name="Gemini 2.0 Flash Experimental (Free)",
                provider="google",
                category="free_general",
                context_window=1000000,  # 1M context
                max_tokens_per_request=8192,
                rate_limit_requests_per_minute=15,  # Conservative for experimental model
                timeout_seconds=30.0,
                supports_streaming=True,
                supports_function_calling=True,
                supports_vision=True,
                fallback_models=["deepseek/deepseek-chat-v3-0324:free", "qwen/qwen3-32b:free"],
            ),
            ModelCapabilities(
                model_id="qwen/qwen3-32b:free",
                display_name="Qwen 3 32B (Free)",
                provider="qwen",
                category="free_general",
                context_window=32768,  # 32K context
                max_tokens_per_request=4096,
                rate_limit_requests_per_minute=25,  # Slightly higher for established model
                timeout_seconds=25.0,
                supports_streaming=True,
                supports_function_calling=False,
                fallback_models=["deepseek/deepseek-chat-v3-0324:free"],
            ),
        ]

        for model in default_models:
            self._models[model.model_id] = model

        self._load_default_fallback_chains()

    def _load_default_fallback_chains(self) -> None:
        """Load default fallback chains matching model_utils.sh patterns."""
        self._fallback_chains = {
            "free_general": [
                "deepseek/deepseek-chat-v3-0324:free",
                "google/gemini-2.0-flash-exp:free",
                "qwen/qwen3-32b:free",
            ],
            "free_reasoning": [
                "deepseek/deepseek-r1-0528:free",
                "deepseek/deepseek-chat-v3-0324:free",
                "microsoft/mai-ds-r1:free",
            ],
            "large_context": [
                "google/gemini-2.5-pro",
                "google/gemini-2.5-flash",
                "google/gemini-2.0-flash-exp:free",
                "deepseek/deepseek-chat-v3-0324:free",
            ],
        }

    def get_model_capabilities(self, model_id: str) -> ModelCapabilities | None:
        """Get capabilities for a specific model.

        Args:
            model_id: Model identifier (e.g., "deepseek/deepseek-chat-v3-0324:free")

        Returns:
            ModelCapabilities if found, None otherwise
        """
        return self._models.get(model_id)

    def list_models(self, category: str | None = None, provider: str | None = None) -> list[ModelCapabilities]:
        """List available models with optional filtering.

        Args:
            category: Filter by model category (e.g., "free_general")
            provider: Filter by provider (e.g., "deepseek")

        Returns:
            List of matching ModelCapabilities
        """
        models = list(self._models.values())

        if category:
            models = [m for m in models if m.category == category]

        if provider:
            models = [m for m in models if m.provider == provider]

        return [m for m in models if m.enabled]

    def get_fallback_chain(self, category: str) -> list[str]:
        """Get fallback chain for a model category.

        Implements the same logic as get_fallback_chain() from model_utils.sh
        with additional validation and depth limiting.

        Args:
            category: Model category for fallback chain

        Returns:
            Ordered list of model IDs for fallback
        """
        chain = self._fallback_chains.get(category, [])

        # Apply depth limiting
        max_depth = self._config.max_fallback_depth
        if len(chain) > max_depth:
            chain = chain[:max_depth]
            self.logger.debug(f"Truncated fallback chain for '{category}' to {max_depth} models")

        # Filter to only enabled models
        enabled_chain = []
        for model_id in chain:
            if model_id in self._models and self._models[model_id].enabled:
                enabled_chain.append(model_id)

        return enabled_chain

    def convert_model_name(self, user_input: str) -> str:
        """Convert user-friendly model names to proper OpenRouter format.

        Extends the convert_model_name() function from model_utils.sh with
        additional validation and registry integration.

        Args:
            user_input: User-provided model name or alias

        Returns:
            Canonical model ID or fallback model
        """
        # Direct lookup first
        if user_input in self._models:
            return user_input

        # Common aliases based on model_utils.sh
        aliases = {
            # DeepSeek aliases
            "deepseek": "deepseek/deepseek-chat-v3-0324:free",
            "deepseek-v3": "deepseek/deepseek-chat-v3-0324:free",
            "deepseek-r1": "deepseek/deepseek-r1-0528:free",
            # Gemini aliases
            "gemini-free": "google/gemini-2.0-flash-exp:free",
            "gemini-2.0-flash": "google/gemini-2.0-flash-exp:free",
            "gemini-flash": "google/gemini-2.5-flash",
            "gemini-pro": "google/gemini-2.5-pro",
            # Qwen aliases
            "qwen-32b": "qwen/qwen3-32b:free",
            "qwen-14b": "qwen/qwen3-14b:free",
            "qwq": "qwen/qwq-32b:free",
        }

        canonical_id = aliases.get(user_input.lower(), user_input)

        # Validate the canonical ID exists
        if canonical_id in self._models:
            return canonical_id

        # Fallback to default category if not found
        fallback_chain = self.get_fallback_chain(self._config.default_category)
        if fallback_chain:
            self.logger.warning(f"Model '{user_input}' not found, using fallback: {fallback_chain[0]}")
            return fallback_chain[0]

        # Last resort - return the first available model
        available_models = [m.model_id for m in self._models.values() if m.enabled]
        if available_models:
            self.logger.error(f"No fallback found for '{user_input}', using: {available_models[0]}")
            return available_models[0]

        # Ultimate fallback
        return "deepseek/deepseek-chat-v3-0324:free"

    def select_best_model(
        self,
        task_type: str,
        allow_premium: bool = False,
        max_tokens_needed: int | None = None,
    ) -> str:
        """Select the best available model for a task type.

        Implements smart model selection similar to smart_model_select()
        from model_utils.sh with enhanced capability matching.

        Args:
            task_type: Type of task ("reasoning", "general", "vision", etc.)
            allow_premium: Whether to consider paid models
            max_tokens_needed: Minimum context window required

        Returns:
            Best available model ID
        """
        # Map task types to categories
        task_to_category = {
            "reasoning": "free_reasoning" if not allow_premium else "premium_reasoning",
            "analysis": "free_general" if not allow_premium else "premium_analysis",
            "planning": "large_context",
            "general": "free_general",
            "vision": "free_general",  # Will filter by supports_vision
        }

        category = task_to_category.get(task_type, self._config.default_category)
        candidates = self.get_fallback_chain(category)

        # Apply additional filters
        for model_id in candidates:
            capabilities = self._models.get(model_id)
            if not capabilities or not capabilities.enabled:
                continue

            # Check premium constraint
            if not allow_premium and not capabilities.is_free:
                continue

            # Check context window requirement
            if max_tokens_needed and capabilities.context_window < max_tokens_needed:
                continue

            # Check task-specific capabilities
            if task_type == "vision" and not capabilities.supports_vision:
                continue

            if task_type == "reasoning" and not capabilities.supports_reasoning:
                continue

            # Found suitable model
            self.logger.debug(f"Selected {model_id} for task '{task_type}'")
            return model_id

        # Fallback to any available model
        for model_id in candidates:
            capabilities = self._models.get(model_id)
            if capabilities and capabilities.enabled:
                self.logger.warning(f"Using fallback model {model_id} for task '{task_type}'")
                return model_id

        # Ultimate fallback
        return self.convert_model_name("deepseek")

    def get_rate_limit(self, model_id: str) -> int:
        """Get rate limit for a specific model.

        Args:
            model_id: Model identifier

        Returns:
            Requests per minute rate limit, or default if unknown
        """
        capabilities = self.get_model_capabilities(model_id)
        return capabilities.rate_limit_requests_per_minute if capabilities else 20

    def is_model_available(self, model_id: str) -> bool:
        """Check if a model is available and enabled.

        Args:
            model_id: Model identifier

        Returns:
            True if model is available and enabled
        """
        capabilities = self.get_model_capabilities(model_id)
        return capabilities is not None and capabilities.enabled

    def get_model_cost(self, model_id: str, input_tokens: int, output_tokens: int) -> float | None:
        """Calculate estimated cost for model usage.

        Args:
            model_id: Model identifier
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Estimated cost in USD, or None for free models
        """
        capabilities = self.get_model_capabilities(model_id)
        if not capabilities or capabilities.is_free:
            return None

        cost = 0.0
        if capabilities.cost_per_input_token:
            cost += input_tokens * capabilities.cost_per_input_token
        if capabilities.cost_per_output_token:
            cost += output_tokens * capabilities.cost_per_output_token

        return cost

    def reload_config(self) -> None:
        """Reload model configuration from file.

        Useful for dynamic configuration updates without restart.
        """
        self.logger.info("Reloading model registry configuration")
        self._models.clear()
        self._fallback_chains.clear()
        self._config = self._load_config()
        self._load_models()
        self.logger.info(f"Reloaded {len(self._models)} models")


# Global registry instance for convenient access
_global_registry: ModelRegistry | None = None


def get_model_registry() -> ModelRegistry:
    """Get global ModelRegistry instance.

    Returns:
        Singleton ModelRegistry instance
    """
    global _global_registry  # noqa: PLW0603
    if _global_registry is None:
        _global_registry = ModelRegistry()
    return _global_registry


def reload_model_registry() -> None:
    """Reload global ModelRegistry configuration."""
    global _global_registry  # noqa: PLW0603
    if _global_registry is not None:
        _global_registry.reload_config()
    else:
        _global_registry = ModelRegistry()


def clear_model_registry() -> None:
    """Clear global ModelRegistry (for testing)."""
    global _global_registry  # noqa: PLW0603
    _global_registry = None
