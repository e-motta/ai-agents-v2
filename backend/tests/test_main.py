"""
Unit tests for Main application module.

These tests verify the FastAPI application setup,
middleware configuration, and endpoint functionality.
"""

from unittest.mock import Mock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.main import app, global_exception_handler, health_check, lifespan


class TestLifespan:
    """Test the lifespan context manager."""

    @pytest.mark.asyncio
    @patch("app.main.get_math_llm")
    @patch("app.main.get_router_llm")
    @patch("app.main.get_knowledge_engine")
    async def test_lifespan_success(
        self, mock_get_knowledge_engine, mock_get_router_llm, mock_get_math_llm
    ):
        """Test successful lifespan execution."""
        # Mock dependencies
        mock_math_llm = Mock()
        mock_router_llm = Mock()
        mock_knowledge_engine = Mock()

        mock_get_math_llm.return_value = mock_math_llm
        mock_get_router_llm.return_value = mock_router_llm
        mock_get_knowledge_engine.return_value = mock_knowledge_engine

        # Mock app
        mock_app = Mock(spec=FastAPI)

        # Test the lifespan
        async with lifespan(mock_app) as result:
            # Verify dependencies were called
            mock_get_math_llm.assert_called_once()
            mock_get_router_llm.assert_called_once()
            mock_get_knowledge_engine.assert_called_once()

            # Verify result is None (yield)
            assert result is None


class TestGlobalExceptionHandler:
    """Test the global exception handler."""

    @pytest.mark.asyncio
    @patch("app.main.logger")
    async def test_global_exception_handler_success(self, mock_logger):
        """Test successful global exception handling."""
        # Mock request
        mock_request = Mock()
        mock_request.url = "http://test.com/api/test"
        mock_request.method = "GET"

        # Mock exception
        mock_exception = Exception("Test error")

        # Call the handler
        response = await global_exception_handler(mock_request, mock_exception)

        # Verify response
        assert response.status_code == 500
        assert response.body == b'{"detail":"An internal error occurred."}'

        # Verify logging
        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args
        assert "Unhandled exception caught by global handler" in call_args[0][0]
        assert call_args[1]["path"] == "http://test.com/api/test"
        assert call_args[1]["method"] == "GET"
        assert call_args[1]["exc_info"] is True

    @pytest.mark.asyncio
    @patch("app.main.logger")
    async def test_global_exception_handler_with_different_exception(self, mock_logger):
        """Test global exception handling with different exception types."""
        # Mock request
        mock_request = Mock()
        mock_request.url = "http://test.com/api/error"
        mock_request.method = "POST"

        # Mock different exception
        mock_exception = ValueError("Validation error")

        # Call the handler
        response = await global_exception_handler(mock_request, mock_exception)

        # Verify response
        assert response.status_code == 500
        assert response.body == b'{"detail":"An internal error occurred."}'

        # Verify logging
        mock_logger.error.assert_called_once()


class TestHealthCheck:
    """Test the health check endpoint."""

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Test successful health check."""
        # Call the function
        result = await health_check()

        # Verify result
        assert result == {"status": "healthy"}


class TestCorsConfiguration:
    """Test CORS configuration."""

    def test_cors_allows_development_origins(self):
        """Test that CORS allows development origins."""
        client = TestClient(app)

        # Test with development origin
        response = client.get("/health", headers={"Origin": "http://localhost:3000"})
        assert response.status_code == 200

        # Check CORS headers
        assert "access-control-allow-origin" in response.headers

    def test_cors_allows_frontend_container(self):
        """Test that CORS allows frontend container origins."""
        client = TestClient(app)

        # Test with container origin
        response = client.get("/health", headers={"Origin": "http://frontend:80"})
        assert response.status_code == 200

    def test_cors_allows_agents_frontend(self):
        """Test that CORS allows agents frontend origin."""
        client = TestClient(app)

        # Test with agents frontend origin
        response = client.get("/health", headers={"Origin": "http://agents-frontend"})
        assert response.status_code == 200
