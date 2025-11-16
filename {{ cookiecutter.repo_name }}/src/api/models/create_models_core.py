"""Core Pydantic models for C.R.E.A.T.E. framework API endpoints.

This module defines the essential data models for Phase 1 Issue 4,
providing request/response validation for the C.R.E.A.T.E. framework API.
"""

from typing import Any

from pydantic import BaseModel, Field, field_validator


class CreateRequestModel(BaseModel):
    """Request model for C.R.E.A.T.E. framework processing."""

    input_prompt: str = Field(
        ...,
        description="The input prompt to enhance using C.R.E.A.T.E. framework",
        min_length=1,
        max_length=50000,
    )
    domain: str | None = Field(
        None,
        description="Optional domain for specialized processing",
        pattern=r"^(general|technical|legal|business|academic)$",
    )
    context: dict[str, Any] | None = Field(
        None,
        description="Optional additional context information",
    )
    settings: dict[str, Any] | None = Field(
        None,
        description="Optional processing settings",
    )

    @field_validator("input_prompt")
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        """Validate the input prompt."""
        if not v or not v.strip():
            raise ValueError("Input prompt cannot be empty or whitespace only")
        return v.strip()

    @field_validator("domain")
    @classmethod
    def validate_domain(cls, v: str | None) -> str | None:
        """Validate the domain if provided."""
        if v is not None:
            valid_domains = {"general", "technical", "legal", "business", "academic"}
            if v not in valid_domains:
                raise ValueError(f"Domain must be one of: {', '.join(valid_domains)}")
        return v


class CreateResponseModel(BaseModel):
    """Response model for C.R.E.A.T.E. framework processing."""

    enhanced_prompt: str = Field(
        ...,
        description="The enhanced prompt using C.R.E.A.T.E. framework",
    )
    framework_components: dict[str, Any] = Field(
        ...,
        description="The extracted C.R.E.A.T.E. framework components",
    )
    metadata: dict[str, Any] = Field(
        ...,
        description="Processing metadata including timing and statistics",
    )
    processing_time: float = Field(
        ...,
        description="Time taken to process the prompt in seconds",
        ge=0,
    )
    success: bool = Field(
        ...,
        description="Whether processing was successful",
    )
    errors: list[str] = Field(
        default_factory=list,
        description="List of any errors encountered during processing",
    )


class ErrorResponseModel(BaseModel):
    """Error response model for API endpoints."""

    error: str = Field(
        ...,
        description="Error type or category",
    )
    detail: str = Field(
        ...,
        description="Detailed error message",
    )
    timestamp: float = Field(
        ...,
        description="Timestamp when error occurred",
    )
    request_id: str | None = Field(
        None,
        description="Optional request identifier for debugging",
    )


class HealthResponseModel(BaseModel):
    """Health check response model."""

    status: str = Field(
        ...,
        description="Health status",
        pattern=r"^(healthy|unhealthy)$",
    )
    service: str = Field(
        ...,
        description="Service name",
    )
    version: str = Field(
        ...,
        description="Service version",
    )
    environment: str = Field(
        ...,
        description="Environment name",
    )
    debug: bool = Field(
        ...,
        description="Debug mode status",
    )
    timestamp: float = Field(
        ...,
        description="Response timestamp",
    )


class DomainResponseModel(BaseModel):
    """Available domains response model."""

    domains: list[str] = Field(
        ...,
        description="List of available processing domains",
    )
    default_domain: str = Field(
        ...,
        description="Default domain used when none specified",
    )


class FrameworkInfoResponseModel(BaseModel):
    """C.R.E.A.T.E. framework information response model."""

    framework: str = Field(
        ...,
        description="Framework name",
    )
    components: list[str] = Field(
        ...,
        description="List of framework components",
    )
    description: str = Field(
        ...,
        description="Framework description",
    )
    version: str = Field(
        ...,
        description="Framework version",
    )
