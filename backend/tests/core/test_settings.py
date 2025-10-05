"""
Unit tests for Settings validation and model configuration.

These tests verify that the settings correctly validate model names and
support configuration via environment variables.
"""

import os
from unittest.mock import patch

import pytest

from app.core.settings import Settings, get_settings


class TestSettingsValidation:
    """Test the Settings class validation."""

    def test_default_settings(self):
        """Test that default settings are valid."""
        settings = Settings()
        assert settings.LLM_MODEL == "gpt-3.5-turbo"
        assert settings.EMBEDDING_MODEL == "text-embedding-3-small"

    def test_model_whitespace_handling(self):
        """Test that whitespace in model names is handled correctly."""
        settings = Settings(LLM_MODEL="  gpt-4  ")
        assert settings.LLM_MODEL == "gpt-4"

        settings = Settings(EMBEDDING_MODEL="  text-embedding-3-small  ")
        assert settings.EMBEDDING_MODEL == "text-embedding-3-small"


class TestSettingsEnvironmentVariables:
    """Test that settings can be configured via environment variables."""

    def test_llm_model_from_env(self):
        """Test that LLM_MODEL can be set via environment variable."""
        with patch.dict(os.environ, {"LLM_MODEL": "gpt-4"}):
            # Clear the settings cache to pick up new environment variables
            get_settings.cache_clear()
            settings = Settings()
            assert settings.LLM_MODEL == "gpt-4"

    def test_embedding_model_from_env(self):
        """Test that EMBEDDING_MODEL can be set via environment variable."""
        with patch.dict(os.environ, {"EMBEDDING_MODEL": "text-embedding-3-large"}):
            # Clear the settings cache to pick up new environment variables
            get_settings.cache_clear()
            settings = Settings()
            assert settings.EMBEDDING_MODEL == "text-embedding-3-large"

    def test_multiple_models_from_env(self):
        """Test that multiple models can be set via environment variables."""
        with patch.dict(
            os.environ,
            {"LLM_MODEL": "gpt-4o", "EMBEDDING_MODEL": "text-embedding-3-large"},
        ):
            # Clear the settings cache to pick up new environment variables
            get_settings.cache_clear()
            settings = Settings()
            assert settings.LLM_MODEL == "gpt-4o"
            assert settings.EMBEDDING_MODEL == "text-embedding-3-large"


class TestSettingsHelpers:
    """Test the helper methods in Settings."""

    def test_ensure_openai_api_key_with_key(self):
        """Test ensure_openai_api_key with valid key."""
        settings = Settings(OPENAI_API_KEY="test-key")  # pyright: ignore[reportArgumentType]
        result = settings.ensure_openai_api_key()
        assert result == "test-key"

    def test_ensure_openai_api_key_without_key(self):
        """Test ensure_openai_api_key without key raises error."""
        settings = Settings(OPENAI_API_KEY=None)
        with pytest.raises(
            ValueError, match="OPENAI_API_KEY environment variable is required"
        ):
            settings.ensure_openai_api_key()

    def test_ensure_openai_api_key_empty_key(self):
        """Test ensure_openai_api_key with empty key raises error."""
        settings = Settings(OPENAI_API_KEY="")  # pyright: ignore[reportArgumentType]
        with pytest.raises(
            ValueError, match="OPENAI_API_KEY environment variable is required"
        ):
            settings.ensure_openai_api_key()

    def test_get_redis_password_with_password(self):
        """Test get_redis_password with password."""
        settings = Settings(REDIS_PASSWORD="test-password")  # pyright: ignore[reportArgumentType]
        result = settings.get_redis_password()
        assert result == "test-password"

    def test_get_redis_password_without_password(self):
        """Test get_redis_password without password returns None."""
        settings = Settings(REDIS_PASSWORD=None)
        result = settings.get_redis_password()
        assert result is None
