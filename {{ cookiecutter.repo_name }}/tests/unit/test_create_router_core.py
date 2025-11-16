"""Unit tests for create_core API router."""

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from src.core.create_processor_core import ValidationError
from src.main_core import app


class TestCreateRouter:
    """Test cases for create_core API router."""

    def setup_method(self):
        """Setup test fixtures."""
        self.client = TestClient(app)

    def test_get_available_domains(self):
        """Test GET /api/v1/create/domains endpoint."""
        response = self.client.get("/api/v1/create/domains")
        assert response.status_code == 200

        data = response.json()
        assert "domains" in data
        assert "default_domain" in data
        assert isinstance(data["domains"], list)

        # Check that expected domains are present
        domains = data["domains"]
        expected_domains = ["general", "technical", "legal", "business", "academic"]
        for domain in expected_domains:
            assert domain in domains
        assert data["default_domain"] == "general"

    def test_get_framework_info(self):
        """Test GET /api/v1/create/framework endpoint."""
        response = self.client.get("/api/v1/create/framework")
        assert response.status_code == 200

        data = response.json()
        assert data["framework"] == "C.R.E.A.T.E."
        assert "components" in data
        assert "version" in data
        assert "description" in data

        # Check that all 6 C.R.E.A.T.E. components are present
        components = data["components"]
        expected_components = ["Context", "Request", "Examples", "Augmentations", "Tone & Format", "Evaluation"]
        assert len(components) == 6
        for component in expected_components:
            assert component in components

    def test_health_check(self):
        """Test GET /api/v1/create/health endpoint."""
        response = self.client.get("/api/v1/create/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "promptcraft-hybrid-create-core"
        assert "timestamp" in data
        assert "version" in data

    @patch("src.api.routers.create_core.processor.process_prompt")
    def test_process_prompt_success(self, mock_process_prompt):
        """Test POST /api/v1/create/ with successful processing."""
        # Setup mock
        mock_response = AsyncMock()
        mock_response.success = True
        mock_response.enhanced_prompt = "Enhanced prompt text"
        mock_response.framework_components = {"context": {"role": "writer"}}
        mock_response.metadata = {"processing_time": 1.5}
        mock_response.processing_time = 1.5
        mock_response.errors = []

        mock_process_prompt.return_value = mock_response

        # Make request
        request_data = {"input_prompt": "Help me write a professional email", "domain": "business"}

        response = self.client.post("/api/v1/create/", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["enhanced_prompt"] == "Enhanced prompt text"
        assert data["success"] is True
        assert data["errors"] == []

    @patch("src.api.routers.create_core.processor.process_prompt")
    def test_process_prompt_validation_error(self, mock_process_prompt):
        """Test POST /api/v1/create/ with validation error."""
        # Setup mock to raise ValidationError
        mock_process_prompt.side_effect = ValidationError("Input prompt must be a non-empty string")

        # Make request with valid JSON (error comes from processor)
        request_data = {"input_prompt": "Test prompt", "domain": "general"}

        response = self.client.post("/api/v1/create/", json=request_data)
        assert response.status_code == 400

        data = response.json()
        assert "detail" in data
        assert "Validation error" in data["detail"]

    def test_process_prompt_missing_prompt(self):
        """Test POST /api/v1/create/ with missing input_prompt."""
        request_data = {
            "domain": "general",
            # Missing input_prompt
        }

        response = self.client.post("/api/v1/create/", json=request_data)
        assert response.status_code == 422  # Validation error

    def test_process_prompt_invalid_json(self):
        """Test POST /api/v1/create/ with invalid JSON."""
        response = self.client.post(
            "/api/v1/create/",
            data="invalid json",
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 422

    def test_process_prompt_empty_prompt_validation(self):
        """Test POST /api/v1/create/ with empty prompt (Pydantic validation)."""
        request_data = {"input_prompt": "", "domain": "general"}  # Empty prompt should fail Pydantic validation

        response = self.client.post("/api/v1/create/", json=request_data)
        assert response.status_code == 422  # Pydantic validation error

    def test_process_prompt_invalid_domain_validation(self):
        """Test POST /api/v1/create/ with invalid domain (Pydantic validation)."""
        request_data = {"input_prompt": "Valid prompt", "domain": "invalid_domain"}  # Should fail Pydantic validation

        response = self.client.post("/api/v1/create/", json=request_data)
        assert response.status_code == 422  # Pydantic validation error

    @patch("src.api.routers.create_core.processor.process_prompt")
    def test_process_prompt_internal_error(self, mock_process_prompt):
        """Test POST /api/v1/create/ with internal server error."""
        # Setup mock to raise generic exception
        mock_process_prompt.side_effect = Exception("Internal error")

        request_data = {"input_prompt": "Test prompt", "domain": "general"}

        response = self.client.post("/api/v1/create/", json=request_data)
        assert response.status_code == 500

        data = response.json()
        assert "detail" in data
        assert "Processing error" in data["detail"]

    @patch("src.api.routers.create_core.processor.process_prompt")
    def test_process_prompt_optional_fields(self, mock_process_prompt):
        """Test POST /api/v1/create/ with only required fields."""
        # Setup mock
        mock_response = AsyncMock()
        mock_response.success = True
        mock_response.enhanced_prompt = "Enhanced text"
        mock_response.framework_components = {}
        mock_response.metadata = {}
        mock_response.processing_time = 1.0
        mock_response.errors = []

        mock_process_prompt.return_value = mock_response

        # Make request with minimal data
        request_data = {
            "input_prompt": "Simple prompt",
            # domain, context, settings are optional
        }

        response = self.client.post("/api/v1/create/", json=request_data)
        assert response.status_code == 200

        # Verify processor was called with None for optional domain
        mock_process_prompt.assert_called_once_with(input_prompt="Simple prompt", domain=None)
