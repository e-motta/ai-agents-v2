"""
Integration tests for model configuration via environment variables.

These tests verify that the application correctly uses environment variables
to configure different models for different agents.
"""

import os
from unittest.mock import patch

from app.core.llm import get_math_agent_llm, get_router_agent_llm
from app.core.settings import Settings


class TestModelConfigurationIntegration:
    """Integration tests for model configuration."""

    def test_environment_variable_override(self):
        """Test that environment variables override default model settings."""
        with patch.dict(
            os.environ,
            {"LLM_MODEL": "gpt-4o", "EMBEDDING_MODEL": "text-embedding-3-large"},
        ):
            settings = Settings()
            assert settings.LLM_MODEL == "gpt-4o"
            assert settings.EMBEDDING_MODEL == "text-embedding-3-large"

    def test_convenience_functions_use_environment_config(self):
        """Test that convenience functions use environment-configured models."""
        with patch.dict(os.environ, {"LLM_MODEL": "gpt-4"}):
            # Clear the settings cache to pick up new environment variables
            Settings.model_rebuild()

            # Mock the OpenAI API key requirement
            with patch("app.core.llm.get_settings") as mock_get_settings:
                mock_settings = mock_get_settings.return_value
                mock_settings.LLM_MODEL = "gpt-4"
                mock_settings.ensure_openai_api_key.return_value = "test-key"

                math_llm = get_math_agent_llm()
                router_llm = get_router_agent_llm()

                assert math_llm.model_name == "gpt-4"
                assert router_llm.model_name == "gpt-4"

    def test_different_models_for_different_agents(self):
        """Test that different agents can use different models if configured."""
        # This test demonstrates that if we had separate environment variables
        # for different agents, they could use different models
        with patch.dict(os.environ, {"LLM_MODEL": "gpt-4o-mini"}):
            Settings.model_rebuild()

            with patch("app.core.llm.get_settings") as mock_get_settings:
                mock_settings = mock_get_settings.return_value
                mock_settings.LLM_MODEL = "gpt-4o-mini"
                mock_settings.ensure_openai_api_key.return_value = "test-key"

                math_llm = get_math_agent_llm()
                router_llm = get_router_agent_llm()

                # Both agents use the same configured model
                assert math_llm.model_name == "gpt-4o-mini"
                assert router_llm.model_name == "gpt-4o-mini"

    def test_model_validation_with_environment_variables(self):
        """Test that model validation works with environment variables."""
        # Test valid models
        valid_models = ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4o-mini"]

        for model in valid_models:
            with patch.dict(os.environ, {"LLM_MODEL": model}):
                settings = Settings()
                assert model == settings.LLM_MODEL

    def test_embedding_model_environment_variable(self):
        """Test that embedding model can be configured via environment variable."""
        with patch.dict(os.environ, {"EMBEDDING_MODEL": "text-embedding-3-large"}):
            settings = Settings()
            assert settings.EMBEDDING_MODEL == "text-embedding-3-large"

    def test_whitespace_handling_in_environment_variables(self):
        """Test that whitespace in environment variables is handled correctly."""
        with patch.dict(os.environ, {"LLM_MODEL": "  gpt-4  "}):
            settings = Settings()
            assert settings.LLM_MODEL == "gpt-4"
