"""
Agent Data Models for PromptCraft-Hybrid.

This module defines the core data models used throughout the agent system,
providing standardized input/output contracts with comprehensive validation.

The models implement:
- AgentInput: Standardized input data model for agent execution
- AgentOutput: Standardized output data model for agent responses
- Configuration models for dependency injection
- Validation rules for all agent operations

Architecture:
    All models use Pydantic for data validation, serialization, and type safety.
    The models support runtime configuration overrides and provide comprehensive
    error handling for malformed data.

Example:
    ```python
    from src.agents.models import AgentInput, AgentOutput

    # Create input for agent execution
    agent_input = AgentInput(
        content="Analyze this code",
        context={"language": "python", "framework": "fastapi"},
        config_overrides={"max_tokens": 1000}
    )

    # Create output from agent execution
    agent_output = AgentOutput(
        content="Code analysis complete",
        metadata={"analysis_type": "security", "issues_found": 0},
        confidence=0.95,
        processing_time=1.234
    )
    ```

Dependencies:
    - pydantic: For data validation and serialization
    - typing: For type annotations
    - datetime: For timestamp handling
    - uuid: For unique identifier generation

Called by:
    - src/agents/base_agent.py: BaseAgent interface contracts
    - src/agents/registry.py: Agent registration and execution
    - All agent implementations throughout the system

Complexity: O(1) - Data models with validation have constant time complexity
"""

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class AgentInput(BaseModel):
    """
    Standardized input data model for agent execution.

    This model defines the contract for all data passed to agents, including
    content, context, and configuration overrides. It provides comprehensive
    validation and supports runtime configuration modifications.

    Attributes:
        content (str): The primary content/query for the agent to process
        context (Optional[Dict[str, Any]]): Additional context information
        config_overrides (Optional[Dict[str, Any]]): Runtime configuration overrides
        request_id (str): Unique identifier for this request
        timestamp (datetime): When the request was created

    Example:
        ```python
        agent_input = AgentInput(
            content="Generate documentation for this API",
            context={
                "api_type": "fastapi",
                "version": "0.1.0",
                "endpoints": ["/health", "/docs"]
            },
            config_overrides={
                "max_tokens": 2000,
                "temperature": 0.7
            }
        )
        ```
    """

    content: str = Field(..., description="Primary content/query for the agent to process", max_length=100000)

    context: dict[str, Any] | None = Field(default=None, description="Additional context information for the agent")

    config_overrides: dict[str, Any] | None = Field(
        default=None,
        description="Runtime configuration overrides for the agent",
    )

    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for this request")

    timestamp: datetime = Field(default_factory=datetime.now, description="When the request was created")

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        """Validate content is not empty or whitespace-only."""
        if not v or not v.strip():
            raise ValueError("Content cannot be empty or whitespace-only")
        return v.strip()

    @field_validator("context", mode="before")
    @classmethod
    def validate_context(cls, v: dict[str, Any] | None) -> dict[str, Any] | None:
        """Validate context dictionary if provided."""
        if v is not None:
            if not isinstance(v, dict):
                raise ValueError("Context must be a dictionary")
            # Ensure all keys are strings
            for key in v:
                if not isinstance(key, str):
                    raise ValueError("All context keys must be strings")
        return v

    @field_validator("config_overrides", mode="before")
    @classmethod
    def validate_config_overrides(cls, v: dict[str, Any] | None) -> dict[str, Any] | None:
        """Validate configuration overrides if provided."""
        if v is not None:
            if not isinstance(v, dict):
                raise ValueError("Config overrides must be a dictionary")
            # Ensure all keys are strings
            for key in v:
                if not isinstance(key, str):
                    raise ValueError("All config override keys must be strings")
        return v

    @model_validator(mode="before")
    @classmethod
    def validate_model(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Perform cross-field validation."""
        content = values.get("content", "")
        context = values.get("context", {})

        # Ensure content and context are consistent
        if context and "content_type" in context:
            content_type = context["content_type"]
            max_code_length = 50000
            if content_type == "code" and len(content) > max_code_length:
                raise ValueError("Code content cannot exceed 50,000 characters")

        return values

    model_config = {
        "validate_assignment": True,
        "arbitrary_types_allowed": True,
        "json_schema_extra": {
            "example": {
                "content": "Analyze this Python function for security vulnerabilities",
                "context": {"language": "python", "framework": "fastapi", "security_focus": "injection_attacks"},
                "config_overrides": {"max_tokens": 1500, "temperature": 0.3},
            },
        },
    }


class AgentOutput(BaseModel):
    """
    Standardized output data model for agent responses.

    This model defines the contract for all data returned by agents, including
    content, metadata, confidence scores, and performance metrics. It provides
    comprehensive validation and consistent structure across all agent types.

    Attributes:
        content (str): The primary response content from the agent
        metadata (Dict[str, Any]): Additional metadata about the response
        confidence (float): Confidence score for the response (0.0 to 1.0)
        processing_time (float): Time taken to process the request in seconds
        request_id (str): Unique identifier linking to the original request
        agent_id (str): Identifier of the agent that produced this output
        timestamp (datetime): When the response was generated

    Example:
        ```python
        agent_output = AgentOutput(
            content="Security analysis complete. No vulnerabilities found.",
            metadata={
                "analysis_type": "security",
                "rules_checked": 15,
                "issues_found": 0
            },
            confidence=0.92,
            processing_time=2.15,
            agent_id="security_agent"
        )
        ```
    """

    content: str = Field(..., description="Primary response content from the agent", max_length=500000)

    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata about the response")

    confidence: float = Field(..., description="Confidence score for the response (0.0 to 1.0)", ge=0.0, le=1.0)

    processing_time: float = Field(..., description="Time taken to process the request in seconds", ge=0.0)

    request_id: str | None = Field(default=None, description="Unique identifier linking to the original request")

    agent_id: str = Field(..., description="Identifier of the agent that produced this output")

    timestamp: datetime = Field(default_factory=datetime.now, description="When the response was generated")

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        """Validate content is not empty or whitespace-only."""
        if not v or not v.strip():
            raise ValueError("Content cannot be empty or whitespace-only")
        return v.strip()

    @field_validator("metadata", mode="before")
    @classmethod
    def validate_metadata(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Validate metadata dictionary."""
        if not isinstance(v, dict):
            raise ValueError("Metadata must be a dictionary")

        # Ensure all keys are strings
        for key in v:
            if not isinstance(key, str):
                raise ValueError("All metadata keys must be strings")

        return v

    @field_validator("agent_id")
    @classmethod
    def validate_agent_id(cls, v: str) -> str:
        """Validate agent_id format."""
        if not v or not v.strip():
            raise ValueError("Agent ID cannot be empty or whitespace-only")

        # Trim whitespace first
        trimmed = v.strip()

        # Agent IDs should follow snake_case convention
        if not trimmed.replace("_", "").isalnum():
            raise ValueError("Agent ID must contain only alphanumeric characters and underscores")

        return trimmed

    @model_validator(mode="before")
    @classmethod
    def validate_model(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Perform cross-field validation."""
        confidence = values.get("confidence", 0.0)
        processing_time = values.get("processing_time", 0.0)

        # Validate confidence makes sense with processing time (only if both are numbers)
        max_processing_time = 30.0
        high_confidence_threshold = 0.95
        if (
            isinstance(processing_time, int | float)
            and isinstance(confidence, int | float)
            and processing_time > max_processing_time
            and confidence > high_confidence_threshold
        ):
            # Warning: High processing time with very high confidence might indicate issues
            pass

        return values

    model_config = {
        "validate_assignment": True,
        "arbitrary_types_allowed": True,
        "json_schema_extra": {
            "example": {
                "content": "Analysis complete. The code follows security best practices.",
                "metadata": {
                    "analysis_type": "security",
                    "rules_applied": ["injection", "auth", "validation"],
                    "issues_found": 0,
                    "suggestions": 2,
                },
                "confidence": 0.89,
                "processing_time": 1.45,
                "agent_id": "security_agent",
            },
        },
    }


class AgentConfig(BaseModel):
    """
    Configuration model for agent dependency injection.

    This model defines the configuration structure for agents, supporting
    both static configuration and runtime overrides. It provides validation
    for configuration parameters and supports nested configuration structures.

    Attributes:
        agent_id (str): Unique identifier for the agent
        name (str): Human-readable name for the agent
        description (str): Description of the agent's capabilities
        config (Dict[str, Any]): Agent-specific configuration parameters
        enabled (bool): Whether the agent is enabled for execution

    Example:
        ```python
        config = AgentConfig(
            agent_id="security_agent",
            name="Security Analysis Agent",
            description="Analyzes code for security vulnerabilities",
            config={
                "max_tokens": 2000,
                "temperature": 0.3,
                "rules": ["injection", "auth", "validation"]
            }
        )
        ```
    """

    agent_id: str = Field(..., description="Unique identifier for the agent")

    name: str = Field(..., description="Human-readable name for the agent")

    description: str = Field(..., description="Description of the agent's capabilities")

    config: dict[str, Any] = Field(default_factory=dict, description="Agent-specific configuration parameters")

    enabled: bool = Field(default=True, description="Whether the agent is enabled for execution")

    @field_validator("agent_id")
    @classmethod
    def validate_agent_id(cls, v: str) -> str:
        """Validate agent_id format."""
        if not v or not v.strip():
            raise ValueError("Agent ID cannot be empty")

        # Trim whitespace first
        trimmed = v.strip()

        # Agent IDs should follow snake_case convention
        if not trimmed.replace("_", "").isalnum():
            raise ValueError("Agent ID must contain only alphanumeric characters and underscores")

        return trimmed

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name is not empty."""
        if not v or not v.strip():
            raise ValueError("Name cannot be empty")
        return v.strip()

    @field_validator("description")
    @classmethod
    def validate_description(cls, v: str) -> str:
        """Validate description is not empty."""
        if not v or not v.strip():
            raise ValueError("Description cannot be empty")
        return v.strip()

    model_config = {
        "validate_assignment": True,
        "arbitrary_types_allowed": True,
        "json_schema_extra": {
            "example": {
                "agent_id": "create_agent",
                "name": "C.R.E.A.T.E. Framework Agent",
                "description": "Enhances prompts using the C.R.E.A.T.E. framework",
                "config": {"max_tokens": 4000, "temperature": 0.7, "frameworks": ["create", "analysis", "enhancement"]},
                "enabled": True,
            },
        },
    }


# Type aliases for better code readability
AgentInputType = AgentInput
AgentOutputType = AgentOutput
AgentConfigType = AgentConfig

# Export all models
__all__ = ["AgentConfig", "AgentConfigType", "AgentInput", "AgentInputType", "AgentOutput", "AgentOutputType"]
