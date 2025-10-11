"""
Integration tests for model configuration via environment variables.

These tests verify that the application correctly uses environment variables
to configure different models for different agents.
"""

import os
from unittest.mock import Mock, patch

from app.core.llm import get_math_agent_llm_client, get_router_agent_llm_client
from app.core.settings import Settings
from app.services.llm_client import LLMClient


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

    @patch("app.core.llm.ChatOpenAI")
    def test_different_models_for_different_agents(self, mock_chat_openai):
        """Test that different agents can use different models if configured."""
        with patch.dict(os.environ, {"LLM_MODEL": "gpt-4o-mini"}):
            Settings.model_rebuild()

            with patch("app.core.llm.get_settings") as mock_get_settings:
                mock_settings = mock_get_settings.return_value
                mock_settings.MATH_LLM_MODEL = "gpt-4o-mini"
                mock_settings.ROUTER_LLM_MODEL = "gpt-4o"
                mock_settings.ensure_openai_api_key.return_value = "test-key"

                # Mock the ChatOpenAI instances
                mock_math_llm = Mock()
                mock_math_llm.model_name = "gpt-4o-mini"
                mock_math_llm.temperature = 0

                mock_router_llm = Mock()
                mock_router_llm.model_name = "gpt-4o"
                mock_router_llm.temperature = 0

                mock_chat_openai.side_effect = [mock_math_llm, mock_router_llm]

                math_llm_client = get_math_agent_llm_client()
                router_llm_client = get_router_agent_llm_client()

                # Both agents use the same configured model
                assert isinstance(math_llm_client, LLMClient)
                assert isinstance(router_llm_client, LLMClient)
                assert math_llm_client.llm.model_name == "gpt-4o-mini"
                assert router_llm_client.llm.model_name == "gpt-4o"

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
