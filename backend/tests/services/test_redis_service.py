"""
Unit tests for Redis Service.

These tests verify the Redis service functionality for managing
conversation history and user conversations.
"""

import json
from unittest.mock import Mock, patch

import pytest
from redis.exceptions import RedisError

from app.services.redis_service import RedisService


class TestRedisServiceInit:
    """Test RedisService initialization."""

    @patch("app.services.redis_service.get_settings")
    @patch("app.services.redis_service.redis.Redis")
    def test_redis_service_init_success(self, mock_redis_class, mock_get_settings):
        """Test successful Redis service initialization."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.REDIS_HOST = "localhost"
        mock_settings.REDIS_PORT = 6379
        mock_settings.REDIS_DB = 0
        mock_settings.REDIS_SOCKET_CONNECT_TIMEOUT = 5
        mock_settings.REDIS_SOCKET_TIMEOUT = 5
        mock_settings.get_redis_password.return_value = "test_password"
        mock_get_settings.return_value = mock_settings

        # Mock Redis client
        mock_redis_client = Mock()
        mock_redis_client.ping.return_value = True
        mock_redis_class.return_value = mock_redis_client

        # Create service
        service = RedisService()

        # Verify Redis client was created with correct parameters
        mock_redis_class.assert_called_once_with(
            host="localhost",
            port=6379,
            db=0,
            password="test_password",
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_error=[TimeoutError, ConnectionError],
        )

        # Verify ping was called
        mock_redis_client.ping.assert_called_once()

        assert service.redis_client == mock_redis_client
        assert service.settings == mock_settings

    @patch("app.services.redis_service.get_settings")
    @patch("app.services.redis_service.redis.Redis")
    def test_redis_service_init_connection_error(
        self, mock_redis_class, mock_get_settings
    ):
        """Test Redis service initialization with connection error."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.REDIS_HOST = "localhost"
        mock_settings.REDIS_PORT = 6379
        mock_settings.REDIS_DB = 0
        mock_settings.REDIS_SOCKET_CONNECT_TIMEOUT = 5
        mock_settings.REDIS_SOCKET_TIMEOUT = 5
        mock_settings.get_redis_password.return_value = "test_password"
        mock_get_settings.return_value = mock_settings

        # Mock Redis client to raise error
        mock_redis_client = Mock()
        mock_redis_client.ping.side_effect = RedisError("Connection failed")
        mock_redis_class.return_value = mock_redis_client

        # Create service and expect exception
        with pytest.raises(RedisError, match="Connection failed"):
            RedisService()


class TestAddMessageToHistory:
    """Test the add_message_to_history method."""

    def setup_method(self):
        """Set up test fixtures."""
        with patch("app.services.redis_service.get_settings") as mock_get_settings:
            mock_settings = Mock()
            mock_settings.REDIS_CONVERSATION_TTL = 3600
            mock_get_settings.return_value = mock_settings

            with patch("app.services.redis_service.redis.Redis") as mock_redis_class:
                self.mock_redis_client = Mock()
                self.mock_redis_client.ping.return_value = True
                mock_redis_class.return_value = self.mock_redis_client

                self.service = RedisService()

    def test_add_message_to_history_success(self):
        """Test successful message addition to history."""
        # Call the method
        result = self.service.add_message_to_history(
            conversation_id="conv_123",
            user_message="Hello",
            agent_response="Hi there!",
            user_id="user_456",
            agent="MathAgent",
        )

        # Verify result
        assert result is True

        # Verify Redis operations
        self.mock_redis_client.rpush.assert_called_once()
        self.mock_redis_client.expire.assert_called()
        self.mock_redis_client.sadd.assert_called_once()

        # Verify the message was added to conversation
        call_args = self.mock_redis_client.rpush.call_args
        assert call_args[0][0] == "conversation:conv_123"

        # Verify the message content
        message_data = json.loads(call_args[0][1])
        assert message_data["user_id"] == "user_456"
        assert message_data["user_message"] == "Hello"
        assert message_data["agent_response"] == "Hi there!"
        assert message_data["agent"] == "MathAgent"
        assert "timestamp" in message_data

        # Verify user conversation mapping
        sadd_call_args = self.mock_redis_client.sadd.call_args
        assert sadd_call_args[0][0] == "user_conversations:user_456"
        assert sadd_call_args[0][1] == "conv_123"

    def test_add_message_to_history_redis_error(self):
        """Test handling of Redis errors in add_message_to_history."""
        # Mock Redis error
        self.mock_redis_client.rpush.side_effect = RedisError("Redis error")

        # Call the method
        result = self.service.add_message_to_history(
            conversation_id="conv_123",
            user_message="Hello",
            agent_response="Hi there!",
            user_id="user_456",
            agent="MathAgent",
        )

        # Verify result
        assert result is False

    def test_add_message_to_history_unexpected_error(self):
        """Test handling of unexpected errors in add_message_to_history."""
        # Mock unexpected error
        self.mock_redis_client.rpush.side_effect = Exception("Unexpected error")

        # Call the method
        result = self.service.add_message_to_history(
            conversation_id="conv_123",
            user_message="Hello",
            agent_response="Hi there!",
            user_id="user_456",
            agent="MathAgent",
        )

        # Verify result
        assert result is False


class TestGetHistory:
    """Test the get_history method."""

    def setup_method(self):
        """Set up test fixtures."""
        with patch("app.services.redis_service.get_settings") as mock_get_settings:
            mock_settings = Mock()
            mock_get_settings.return_value = mock_settings

            with patch("app.services.redis_service.redis.Redis") as mock_redis_class:
                self.mock_redis_client = Mock()
                self.mock_redis_client.ping.return_value = True
                mock_redis_class.return_value = self.mock_redis_client

                self.service = RedisService()

    def test_get_history_success(self):
        """Test successful history retrieval."""
        # Mock Redis response
        mock_messages = [
            json.dumps(
                {
                    "timestamp": "2023-01-01T00:00:00Z",
                    "user_id": "user_456",
                    "user_message": "Hello",
                    "agent_response": "Hi there!",
                    "agent": "MathAgent",
                }
            ),
            json.dumps(
                {
                    "timestamp": "2023-01-01T00:01:00Z",
                    "user_id": "user_456",
                    "user_message": "How are you?",
                    "agent_response": "I'm doing well!",
                    "agent": "MathAgent",
                }
            ),
        ]
        self.mock_redis_client.lrange.return_value = mock_messages

        # Call the method
        result = self.service.get_history("conv_123")

        # Verify result
        assert len(result) == 2
        assert result[0]["user_message"] == "Hello"
        assert result[0]["agent_response"] == "Hi there!"
        assert result[1]["user_message"] == "How are you?"
        assert result[1]["agent_response"] == "I'm doing well!"

        # Verify Redis call
        self.mock_redis_client.lrange.assert_called_once_with(
            "conversation:conv_123", 0, -1
        )

    def test_get_history_empty_conversation(self):
        """Test history retrieval for empty conversation."""
        # Mock empty Redis response
        self.mock_redis_client.lrange.return_value = []

        # Call the method
        result = self.service.get_history("conv_123")

        # Verify result
        assert result == []

    def test_get_history_invalid_json(self):
        """Test handling of invalid JSON in history."""
        # Mock Redis response with invalid JSON
        mock_messages = [
            json.dumps(
                {
                    "timestamp": "2023-01-01T00:00:00Z",
                    "user_id": "user_456",
                    "user_message": "Hello",
                    "agent_response": "Hi there!",
                    "agent": "MathAgent",
                }
            ),
            "invalid json",
            json.dumps(
                {
                    "timestamp": "2023-01-01T00:01:00Z",
                    "user_id": "user_456",
                    "user_message": "How are you?",
                    "agent_response": "I'm doing well!",
                    "agent": "MathAgent",
                }
            ),
        ]
        self.mock_redis_client.lrange.return_value = mock_messages

        # Call the method
        result = self.service.get_history("conv_123")

        # Verify result - should skip invalid JSON
        assert len(result) == 2
        assert result[0]["user_message"] == "Hello"
        assert result[1]["user_message"] == "How are you?"

    def test_get_history_redis_error(self):
        """Test handling of Redis errors in get_history."""
        # Mock Redis error
        self.mock_redis_client.lrange.side_effect = RedisError("Redis error")

        # Call the method
        result = self.service.get_history("conv_123")

        # Verify result
        assert result == []

    def test_get_history_unexpected_error(self):
        """Test handling of unexpected errors in get_history."""
        # Mock unexpected error
        self.mock_redis_client.lrange.side_effect = Exception("Unexpected error")

        # Call the method
        result = self.service.get_history("conv_123")

        # Verify result
        assert result == []


class TestGetUserConversations:
    """Test the get_user_conversations method."""

    def setup_method(self):
        """Set up test fixtures."""
        with patch("app.services.redis_service.get_settings") as mock_get_settings:
            mock_settings = Mock()
            mock_get_settings.return_value = mock_settings

            with patch("app.services.redis_service.redis.Redis") as mock_redis_class:
                self.mock_redis_client = Mock()
                self.mock_redis_client.ping.return_value = True
                mock_redis_class.return_value = self.mock_redis_client

                self.service = RedisService()

    def test_get_user_conversations_success(self):
        """Test successful user conversations retrieval."""
        # Mock Redis response
        mock_conversations = {"conv_123", "conv_456", "conv_789"}
        self.mock_redis_client.smembers.return_value = mock_conversations

        # Call the method
        result = self.service.get_user_conversations("user_456")

        # Verify result
        assert len(result) == 3
        assert "conv_123" in result
        assert "conv_456" in result
        assert "conv_789" in result
        assert result == sorted(result)  # Should be sorted

        # Verify Redis call
        self.mock_redis_client.smembers.assert_called_once_with(
            "user_conversations:user_456"
        )

    def test_get_user_conversations_empty(self):
        """Test user conversations retrieval for user with no conversations."""
        # Mock empty Redis response
        self.mock_redis_client.smembers.return_value = set()

        # Call the method
        result = self.service.get_user_conversations("user_456")

        # Verify result
        assert result == []

    def test_get_user_conversations_redis_error(self):
        """Test handling of Redis errors in get_user_conversations."""
        # Mock Redis error
        self.mock_redis_client.smembers.side_effect = RedisError("Redis error")

        # Call the method
        result = self.service.get_user_conversations("user_456")

        # Verify result
        assert result == []

    def test_get_user_conversations_unexpected_error(self):
        """Test handling of unexpected errors in get_user_conversations."""
        # Mock unexpected error
        self.mock_redis_client.smembers.side_effect = Exception("Unexpected error")

        # Call the method
        result = self.service.get_user_conversations("user_456")

        # Verify result
        assert result == []

    def test_get_user_conversations_sorts_result(self):
        """Test that user conversations are sorted."""
        # Mock Redis response with unsorted conversations
        mock_conversations = {"conv_789", "conv_123", "conv_456"}
        self.mock_redis_client.smembers.return_value = mock_conversations

        # Call the method
        result = self.service.get_user_conversations("user_456")

        # Verify result is sorted
        assert result == ["conv_123", "conv_456", "conv_789"]


class TestRedisServiceIntegration:
    """Integration tests for Redis service."""

    def setup_method(self):
        """Set up test fixtures."""
        with patch("app.services.redis_service.get_settings") as mock_get_settings:
            mock_settings = Mock()
            mock_settings.REDIS_CONVERSATION_TTL = 3600
            mock_get_settings.return_value = mock_settings

            with patch("app.services.redis_service.redis.Redis") as mock_redis_class:
                self.mock_redis_client = Mock()
                self.mock_redis_client.ping.return_value = True
                mock_redis_class.return_value = self.mock_redis_client

                self.service = RedisService()

    def test_full_conversation_flow(self):
        """Test the complete conversation flow."""
        # Add messages to history
        self.service.add_message_to_history(
            conversation_id="conv_123",
            user_message="Hello",
            agent_response="Hi there!",
            user_id="user_456",
            agent="MathAgent",
        )

        self.service.add_message_to_history(
            conversation_id="conv_123",
            user_message="How are you?",
            agent_response="I'm doing well!",
            user_id="user_456",
            agent="MathAgent",
        )

        # Mock history retrieval
        mock_messages = [
            json.dumps(
                {
                    "timestamp": "2023-01-01T00:00:00Z",
                    "user_id": "user_456",
                    "user_message": "Hello",
                    "agent_response": "Hi there!",
                    "agent": "MathAgent",
                }
            ),
            json.dumps(
                {
                    "timestamp": "2023-01-01T00:01:00Z",
                    "user_id": "user_456",
                    "user_message": "How are you?",
                    "agent_response": "I'm doing well!",
                    "agent": "MathAgent",
                }
            ),
        ]
        self.mock_redis_client.lrange.return_value = mock_messages

        # Get conversation history
        history = self.service.get_history("conv_123")

        # Verify history
        assert len(history) == 2
        assert history[0]["user_message"] == "Hello"
        assert history[1]["user_message"] == "How are you?"

        # Mock user conversations
        self.mock_redis_client.smembers.return_value = {"conv_123", "conv_456"}

        # Get user conversations
        conversations = self.service.get_user_conversations("user_456")

        # Verify conversations
        assert len(conversations) == 2
        assert "conv_123" in conversations
        assert "conv_456" in conversations

        # Verify all Redis operations were called
        assert self.mock_redis_client.rpush.call_count == 2
        assert (
            self.mock_redis_client.expire.call_count == 4
        )  # 2 for conversation, 2 for user mapping
        assert self.mock_redis_client.sadd.call_count == 2
        assert self.mock_redis_client.lrange.call_count == 1
        assert self.mock_redis_client.smembers.call_count == 1
