"""
Unit tests for Dependencies module.

These tests verify the dependency injection functions
and their caching behavior.
"""

from unittest.mock import Mock, patch

from app.dependencies import (
    get_knowledge_engine,
    get_math_llm,
    get_redis_service,
    get_router_llm,
    get_sanitized_message_from_request,
)
from app.models import ChatRequest


class TestGetSanitizedMessageFromRequest:
    """Test the get_sanitized_message_from_request function."""

    @patch("app.dependencies.sanitize_user_input")
    def test_get_sanitized_message_from_request_success(self, mock_sanitize):
        """Test successful message sanitization."""
        # Mock sanitization
        mock_sanitize.return_value = "sanitized message"

        # Create mock request
        mock_request = Mock(spec=ChatRequest)
        mock_request.message = "  test message  "

        # Call the function
        result = get_sanitized_message_from_request(mock_request)

        # Verify result
        assert result == "sanitized message"
        mock_sanitize.assert_called_once_with("  test message  ")

    @patch("app.dependencies.sanitize_user_input")
    def test_get_sanitized_message_from_request_with_whitespace(self, mock_sanitize):
        """Test message sanitization with whitespace handling."""
        # Mock sanitization
        mock_sanitize.return_value = "clean message"

        # Create mock request with whitespace
        mock_request = Mock(spec=ChatRequest)
        mock_request.message = "\n\t  test message  \n\t"

        # Call the function
        result = get_sanitized_message_from_request(mock_request)

        # Verify result
        assert result == "clean message"
        mock_sanitize.assert_called_once_with("\n\t  test message  \n\t")

    @patch("app.dependencies.sanitize_user_input")
    def test_get_sanitized_message_from_request_empty_message(self, mock_sanitize):
        """Test message sanitization with empty message."""
        # Mock sanitization
        mock_sanitize.return_value = ""

        # Create mock request with empty message
        mock_request = Mock(spec=ChatRequest)
        mock_request.message = ""

        # Call the function
        result = get_sanitized_message_from_request(mock_request)

        # Verify result
        assert result == ""
        mock_sanitize.assert_called_once_with("")


class TestGetMathLlm:
    """Test the get_math_llm function."""

    @patch("app.dependencies.get_math_agent_llm")
    def test_get_math_llm_calls_agent_function(self, mock_get_math_agent_llm):
        """Test that get_math_llm calls the agent function."""
        # Mock the agent function
        mock_llm = Mock()
        mock_get_math_agent_llm.return_value = mock_llm

        # Call the function
        result = get_math_llm()

        # Verify result
        assert result == mock_llm
        mock_get_math_agent_llm.assert_called_once()


class TestGetRouterLlm:
    """Test the get_router_llm function."""

    @patch("app.dependencies.get_router_agent_llm")
    def test_get_router_llm_calls_agent_function(self, mock_get_router_agent_llm):
        """Test that get_router_llm calls the agent function."""
        # Mock the agent function
        mock_llm = Mock()
        mock_get_router_agent_llm.return_value = mock_llm

        # Call the function
        result = get_router_llm()

        # Verify result
        assert result == mock_llm
        mock_get_router_agent_llm.assert_called_once()


class TestGetKnowledgeEngine:
    """Test the get_knowledge_engine function."""

    @patch("app.dependencies.get_query_engine")
    def test_get_knowledge_engine_calls_query_function(self, mock_get_query_engine):
        """Test that get_knowledge_engine calls the query function."""
        # Mock the query function
        mock_engine = Mock()
        mock_get_query_engine.return_value = mock_engine

        # Call the function
        result = get_knowledge_engine()

        # Verify result
        assert result == mock_engine
        mock_get_query_engine.assert_called_once()


class TestGetRedisService:
    """Test the get_redis_service function."""

    @patch("app.dependencies.RedisService")
    def test_get_redis_service_creates_service(self, mock_redis_service_class):
        """Test that get_redis_service creates a Redis service."""
        # Mock the Redis service class
        mock_service = Mock()
        mock_redis_service_class.return_value = mock_service

        # Call the function
        result = get_redis_service()

        # Verify result
        assert result == mock_service
        mock_redis_service_class.assert_called_once()
