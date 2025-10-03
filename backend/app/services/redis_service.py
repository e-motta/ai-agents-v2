"""
Redis service for managing conversation history.

This module provides functions to store and retrieve conversation history
using Redis as the storage backend.
"""

import json
from datetime import UTC, datetime
from typing import Any

import redis
from redis.exceptions import RedisError

from app.core.logging import get_logger
from app.core.settings import get_settings

logger = get_logger(__name__)


class RedisService:
    """Redis service for conversation history management."""

    def __init__(self, settings=None):
        """
        Initialize Redis connection using application settings.

        Args:
            settings: Optional settings instance. If None, will use get_settings()
        """
        if settings is None:
            settings = get_settings()

        self.settings = settings

        try:
            self.redis_client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                password=settings.get_redis_password(),
                decode_responses=True,
                socket_connect_timeout=settings.REDIS_SOCKET_CONNECT_TIMEOUT,
                socket_timeout=settings.REDIS_SOCKET_TIMEOUT,
                retry_on_error=[TimeoutError, ConnectionError],
            )
            # Test connection
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {settings.REDIS_HOST}:{settings.REDIS_PORT}")
        except RedisError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def add_message_to_history(
        self,
        conversation_id: str,
        user_message: str,
        agent_response: str,
        user_id: str,
        agent: str,
    ) -> bool:
        """
        Add a message exchange to conversation history.

        Args:
            conversation_id: Unique identifier for the conversation
            user_message: The user's message
            agent_response: The agent's response
            user_id: The user's identifier
            agent: The agent responsible for the response

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create message entry
            message_entry = {
                "timestamp": datetime.now(UTC).isoformat(),
                "user_id": user_id,
                "user_message": user_message,
                "agent_response": agent_response,
                "agent": agent,
            }

            # Use Redis list to store conversation history
            # Key format: "conversation:{conversation_id}"
            key = f"conversation:{conversation_id}"

            # Add message to the end of the list
            self.redis_client.rpush(key, json.dumps(message_entry))

            # Set expiration for the conversation using settings TTL
            self.redis_client.expire(key, self.settings.REDIS_CONVERSATION_TTL)

            # Maintain user-conversation mapping
            # Key format: "user_conversations:{user_id}"
            user_key = f"user_conversations:{user_id}"

            # Add conversation_id to user's conversation set if not already present
            self.redis_client.sadd(user_key, conversation_id)

            # Set expiration for the user conversations mapping using settings TTL
            self.redis_client.expire(user_key, self.settings.REDIS_CONVERSATION_TTL)

            logger.info(f"Added message to conversation {conversation_id}")
            return True

        except RedisError as e:
            logger.exception(f"Failed to add message to history: {e}")
            return False
        except Exception as e:
            logger.exception(f"Unexpected error adding message to history: {e}")
            return False

    def get_history(self, conversation_id: str) -> list[dict[str, Any]]:
        """
        Retrieve conversation history.

        Args:
            conversation_id: Unique identifier for the conversation

        Returns:
            list[dict[str, Any]]: List of message exchanges in chronological order
        """
        try:
            key = f"conversation:{conversation_id}"

            # Get all messages from the list
            messages = self.redis_client.lrange(key, 0, -1)

            # Parse JSON messages
            history = []
            for message_json in messages:
                try:
                    message_data = json.loads(message_json)
                    history.append(message_data)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse message JSON: {e}")
                    continue

            logger.info(f"Retrieved {len(history)} messages for conversation {conversation_id}")
            return history

        except RedisError as e:
            logger.exception(f"Failed to get conversation history: {e}")
            return []
        except Exception as e:
            logger.exception(f"Unexpected error getting conversation history: {e}")
            return []

    def clear_history(self, conversation_id: str) -> bool:
        """
        Clear conversation history.

        Args:
            conversation_id: Unique identifier for the conversation

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            key = f"conversation:{conversation_id}"
            result = self.redis_client.delete(key)
            logger.info(f"Cleared history for conversation {conversation_id}")
            return result > 0
        except RedisError as e:
            logger.exception(f"Failed to clear conversation history: {e}")
            return False

    def get_conversation_count(self, conversation_id: str) -> int:
        """
        Get the number of messages in a conversation.

        Args:
            conversation_id: Unique identifier for the conversation

        Returns:
            int: Number of messages in the conversation
        """
        try:
            key = f"conversation:{conversation_id}"
            return self.redis_client.llen(key)
        except RedisError as e:
            logger.exception(f"Failed to get conversation count: {e}")
            return 0

    def get_user_conversations(self, user_id: str) -> list[str]:
        """
        Get all conversation IDs for a specific user.

        Args:
            user_id: Unique identifier for the user

        Returns:
            list[str]: List of conversation IDs belonging to the user
        """
        try:
            user_key = f"user_conversations:{user_id}"

            # Get all conversation IDs from the set
            conversation_ids = self.redis_client.smembers(user_key)

            # Convert to list and sort for consistent ordering
            result = sorted(list(conversation_ids))

            logger.info(f"Retrieved {len(result)} conversations for user {user_id}")
            return result

        except RedisError as e:
            logger.exception(f"Failed to get user conversations: {e}")
            return []
        except Exception as e:
            logger.exception(f"Unexpected error getting user conversations: {e}")
            return []
