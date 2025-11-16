"""Unit tests for create_models_core API models."""

import pytest
from pydantic import ValidationError

from src.api.models.create_models_core import (
    CreateRequestModel,
    CreateResponseModel,
    DomainResponseModel,
    ErrorResponseModel,
    FrameworkInfoResponseModel,
    HealthResponseModel,
)


class TestCreateModels:
    """Test cases for create API models."""

    def test_create_request_valid(self):
        """Test CreateRequestModel with valid data."""
        request = CreateRequestModel(
            input_prompt="Help me write a professional email",
            domain="business",
            context={"urgency": "high"},
            settings={"format": "formal"},
        )

        assert request.input_prompt == "Help me write a professional email"
        assert request.domain == "business"
        assert request.context == {"urgency": "high"}
        assert request.settings == {"format": "formal"}

    def test_create_request_minimal(self):
        """Test CreateRequestModel with minimal required data."""
        request = CreateRequestModel(input_prompt="Simple prompt")

        assert request.input_prompt == "Simple prompt"
        assert request.domain is None
        assert request.context is None
        assert request.settings is None

    def test_create_request_empty_prompt_fails(self):
        """Test CreateRequestModel fails with empty prompt."""
        with pytest.raises(ValidationError):
            CreateRequestModel(input_prompt="")

    def test_create_request_invalid_domain(self):
        """Test CreateRequestModel with invalid domain."""
        with pytest.raises(ValidationError):
            CreateRequestModel(input_prompt="Test prompt", domain="invalid_domain")

    def test_create_request_valid_domains(self):
        """Test CreateRequestModel with valid domains."""
        valid_domains = ["general", "technical", "legal", "business", "academic"]

        for domain in valid_domains:
            request = CreateRequestModel(input_prompt="Test prompt", domain=domain)
            assert request.domain == domain

    def test_create_response_valid(self):
        """Test CreateResponseModel with valid data."""
        response = CreateResponseModel(
            enhanced_prompt="Enhanced prompt text",
            framework_components={"context": {"role": "writer"}, "request": {"task": "write email"}},
            metadata={"processing_time": 1.5},
            processing_time=1.5,
            success=True,
            errors=[],
        )

        assert response.enhanced_prompt == "Enhanced prompt text"
        assert response.success is True
        assert response.errors == []
        assert "context" in response.framework_components

    def test_create_response_with_errors(self):
        """Test CreateResponseModel with error state."""
        response = CreateResponseModel(
            enhanced_prompt="",
            framework_components={},
            metadata={"error": True},
            processing_time=0.1,
            success=False,
            errors=["Validation failed", "Processing error"],
        )

        assert response.enhanced_prompt == ""
        assert response.success is False
        assert len(response.errors) == 2
        assert "Validation failed" in response.errors

    def test_error_response_model(self):
        """Test ErrorResponseModel."""
        error = ErrorResponseModel(
            error="ValidationError",
            detail="Input prompt is required",
            timestamp=1234567890.0,
            request_id="req-123",
        )

        assert error.error == "ValidationError"
        assert error.detail == "Input prompt is required"
        assert error.timestamp == 1234567890.0
        assert error.request_id == "req-123"

    def test_health_response_model(self):
        """Test HealthResponseModel."""
        health = HealthResponseModel(
            status="healthy",
            service="create-processor",
            version="1.0.0",
            environment="test",
            debug=True,
            timestamp=1234567890.0,
        )

        assert health.status == "healthy"
        assert health.service == "create-processor"
        assert health.version == "1.0.0"

    def test_health_response_invalid_status(self):
        """Test HealthResponseModel with invalid status."""
        with pytest.raises(ValidationError):
            HealthResponseModel(
                status="unknown",  # Invalid status
                service="test",
                version="1.0.0",
                environment="test",
                debug=True,
                timestamp=1234567890.0,
            )

    def test_domain_response_model(self):
        """Test DomainResponseModel."""
        domains = DomainResponseModel(domains=["general", "technical", "business"], default_domain="general")

        assert len(domains.domains) == 3
        assert "technical" in domains.domains
        assert domains.default_domain == "general"

    def test_framework_info_response_model(self):
        """Test FrameworkInfoResponseModel."""
        framework = FrameworkInfoResponseModel(
            framework="C.R.E.A.T.E.",
            components=["Context", "Request", "Examples"],
            description="Framework for prompt enhancement",
            version="1.0.0",
        )

        assert framework.framework == "C.R.E.A.T.E."
        assert len(framework.components) == 3
        assert "Context" in framework.components

    def test_negative_processing_time_fails(self):
        """Test CreateResponseModel fails with negative processing time."""
        with pytest.raises(ValidationError):
            CreateResponseModel(
                enhanced_prompt="Test",
                framework_components={},
                metadata={},
                processing_time=-1.0,  # Negative time
                success=True,
                errors=[],
            )
