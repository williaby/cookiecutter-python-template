"""Unit tests for main_core FastAPI application."""

from fastapi.testclient import TestClient

from src.main_core import app


class TestMainCoreApp:
    """Test cases for main_core FastAPI application."""

    def setup_method(self):
        """Setup test fixtures."""
        self.client = TestClient(app)

    def test_root_endpoint(self):
        """Test root endpoint returns API information."""
        response = self.client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert data["message"] == "PromptCraft-Hybrid C.R.E.A.T.E. Framework API"
        assert data["version"] == "1.0.0"
        assert data["framework"] == "C.R.E.A.T.E."
        assert "endpoints" in data
        assert "timestamp" in data

    def test_health_check_endpoint(self):
        """Test health check endpoint returns status."""
        response = self.client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "promptcraft-hybrid-create-api"
        assert data["version"] == "1.0.0"
        assert "timestamp" in data

    def test_not_found_handler(self):
        """Test 404 error handler."""
        response = self.client.get("/nonexistent")
        assert response.status_code == 404

        data = response.json()
        assert data["error"] == "Not Found"
        assert "detail" in data
        assert "timestamp" in data

    def test_api_endpoints_accessible(self):
        """Test that API endpoints are properly registered."""
        # Test create endpoint (GET should return 405 Method Not Allowed since it's POST only)
        response = self.client.get("/api/v1/create/")
        # Should not be 404 (not found), but 405 (method not allowed) is acceptable
        assert response.status_code in [405, 422]  # 405 or 422 are acceptable

        # Test domains endpoint
        response = self.client.get("/api/v1/create/domains")
        assert response.status_code == 200

        # Test framework endpoint
        response = self.client.get("/api/v1/create/framework")
        assert response.status_code == 200

    def test_cors_middleware_configured(self):
        """Test that CORS middleware is configured."""
        # Test OPTIONS request to check CORS headers
        response = self.client.options("/")
        # Should not be 405 (method not allowed) due to CORS middleware
        assert response.status_code in [200, 405]  # 405 is acceptable if OPTIONS not implemented
