"""
Unit tests for LLM factory functions and convenience functions.

These tests verify that the LLM functions correctly use configured models
instead of hardcoded values.
"""

import os
from unittest.mock import Mock, patch

from langchain_openai import ChatOpenAI

from app.core.llm import (
    get_chat_openai_llm,
    get_math_agent_llm,
    get_router_agent_llm,
    setup_knowledge_agent_settings,
    setup_llamaindex_settings,
)


class TestChatOpenAILLM:
    """Test the get_chat_openai_llm function."""

    @patch("app.core.llm.get_settings")
    def test_get_chat_openai_llm_with_default_model(self, mock_get_settings):
        """Test that get_chat_openai_llm uses default model from settings."""
        mock_settings = Mock()
        mock_settings.LLM_MODEL = "gpt-4"
        mock_settings.ensure_openai_api_key.return_value = "test-key"
        mock_get_settings.return_value = mock_settings

        llm = get_chat_openai_llm()

        assert isinstance(llm, ChatOpenAI)
        assert llm.model_name == "gpt-4"
        assert llm.temperature == 0

    @patch("app.core.llm.get_settings")
    def test_get_chat_openai_llm_with_custom_model(self, mock_get_settings):
        """Test that get_chat_openai_llm uses custom model when provided."""
        mock_settings = Mock()
        mock_settings.LLM_MODEL = "gpt-3.5-turbo"
        mock_settings.ensure_openai_api_key.return_value = "test-key"
        mock_get_settings.return_value = mock_settings

        llm = get_chat_openai_llm(model="gpt-4o", temperature=0.5)

        assert isinstance(llm, ChatOpenAI)
        assert llm.model_name == "gpt-4o"
        assert llm.temperature == 0.5

    @patch("app.core.llm.get_settings")
    def test_get_chat_openai_llm_with_custom_temperature(self, mock_get_settings):
        """Test that get_chat_openai_llm uses custom temperature."""
        mock_settings = Mock()
        mock_settings.LLM_MODEL = "gpt-3.5-turbo"
        mock_settings.ensure_openai_api_key.return_value = "test-key"
        mock_get_settings.return_value = mock_settings

        llm = get_chat_openai_llm(model="gpt-3.5-turbo", temperature=0.7)

        assert isinstance(llm, ChatOpenAI)
        assert llm.temperature == 0.7


class TestConvenienceFunctions:
    """Test the convenience functions for agent LLMs."""

    @patch("app.core.llm.get_settings")
    def test_get_math_agent_llm_uses_default_model(self, mock_get_settings):
        """Test that get_math_agent_llm uses default model from settings."""
        mock_settings = Mock()
        mock_settings.MATH_LLM_MODEL = None
        mock_settings.LLM_MODEL = "gpt-3.5"
        mock_settings.ensure_openai_api_key.return_value = "test-key"
        mock_get_settings.return_value = mock_settings

        llm = get_math_agent_llm()

        assert isinstance(llm, ChatOpenAI)
        assert llm.model_name == "gpt-3.5"
        assert llm.temperature == 0

    @patch("app.core.llm.get_settings")
    def test_get_math_agent_llm_uses_configured_model(self, mock_get_settings):
        """Test that get_math_agent_llm uses configured model from settings."""
        mock_settings = Mock()
        mock_settings.MATH_LLM_MODEL = "gpt-4"
        mock_settings.LLM_MODEL = "gpt-3.5"
        mock_settings.ensure_openai_api_key.return_value = "test-key"
        mock_get_settings.return_value = mock_settings

        llm = get_math_agent_llm()

        assert isinstance(llm, ChatOpenAI)
        assert llm.model_name == "gpt-4"
        assert llm.temperature == 0

    @patch("app.core.llm.get_settings")
    def test_get_router_agent_llm_uses_default_model(self, mock_get_settings):
        """Test that get_router_agent_llm uses default model from settings."""
        mock_settings = Mock()
        mock_settings.ROUTER_LLM_MODEL = None
        mock_settings.LLM_MODEL = "gpt-3.5"
        mock_settings.ensure_openai_api_key.return_value = "test-key"
        mock_get_settings.return_value = mock_settings

        llm = get_router_agent_llm()

        assert isinstance(llm, ChatOpenAI)
        assert llm.model_name == "gpt-3.5"
        assert llm.temperature == 0

    @patch("app.core.llm.get_settings")
    def test_get_router_agent_llm_uses_configured_model(self, mock_get_settings):
        """Test that get_router_agent_llm uses configured model from settings."""
        mock_settings = Mock()
        mock_settings.LLM_MODEL = "gpt-4"
        mock_settings.ROUTER_LLM_MODEL = "gpt-3.5"
        mock_settings.ensure_openai_api_key.return_value = "test-key"
        mock_get_settings.return_value = mock_settings

        llm = get_router_agent_llm()

        assert isinstance(llm, ChatOpenAI)
        assert llm.model_name == "gpt-3.5"
        assert llm.temperature == 0

    @patch("app.core.llm.get_settings")
    def test_different_agents_can_use_different_models(self, mock_get_settings):
        """Test that different agents can use different models if configured."""
        # First call with gpt-4
        mock_settings = Mock()
        mock_settings.MATH_LLM_MODEL = "gpt-4"
        mock_settings.ensure_openai_api_key.return_value = "test-key"
        mock_get_settings.return_value = mock_settings

        math_llm = get_math_agent_llm()
        assert math_llm.model_name == "gpt-4"

        # Second call with gpt-4o (simulating different configuration)
        mock_settings.ROUTER_LLM_MODEL = "gpt-4o"
        router_llm = get_router_agent_llm()
        assert router_llm.model_name == "gpt-4o"


class TestLlamaIndexSettings:
    """Test the LlamaIndex settings functions."""

    @patch("app.core.llm.get_settings")
    @patch("app.core.llm.Settings")
    def test_setup_llamaindex_settings_with_defaults(
        self, mock_settings_class, mock_get_settings
    ):
        """Test that setup_llamaindex_settings uses default models from settings."""
        mock_settings = Mock()
        mock_settings.LLM_MODEL = "gpt-4"
        mock_settings.EMBEDDING_MODEL = "text-embedding-3-large"
        mock_settings.CHUNK_SIZE = 512
        mock_settings.CHUNK_OVERLAP = 10
        mock_settings.ensure_openai_api_key.return_value = "test-key"
        mock_get_settings.return_value = mock_settings

        # Mock the LlamaIndex Settings
        mock_llm = Mock()
        mock_embed_model = Mock()
        mock_node_parser = Mock()

        mock_settings_class.llm = mock_llm
        mock_settings_class.embed_model = mock_embed_model
        mock_settings_class.node_parser = mock_node_parser

        setup_llamaindex_settings()

        # Verify that the settings were called with default values
        mock_get_settings.assert_called_once()

    @patch("app.core.llm.get_settings")
    @patch("app.core.llm.Settings")
    def test_setup_llamaindex_settings_with_custom_values(
        self, mock_settings_class, mock_get_settings
    ):
        """Test that setup_llamaindex_settings uses custom values when provided."""
        mock_settings = Mock()
        mock_settings.LLM_MODEL = "gpt-3.5-turbo"
        mock_settings.EMBEDDING_MODEL = "text-embedding-3-small"
        mock_settings.CHUNK_SIZE = 1024
        mock_settings.CHUNK_OVERLAP = 20
        mock_settings.ensure_openai_api_key.return_value = "test-key"
        mock_get_settings.return_value = mock_settings

        # Mock the LlamaIndex Settings
        mock_llm = Mock()
        mock_embed_model = Mock()
        mock_node_parser = Mock()

        mock_settings_class.llm = mock_llm
        mock_settings_class.embed_model = mock_embed_model
        mock_settings_class.node_parser = mock_node_parser

        setup_llamaindex_settings(
            llm_model="gpt-4o",
            embedding_model="text-embedding-3-large",
            chunk_size=512,
            chunk_overlap=10,
        )

        # Verify that the settings were called
        mock_get_settings.assert_called_once()

    @patch("app.core.llm.setup_llamaindex_settings")
    def test_setup_knowledge_agent_settings_calls_setup(self, mock_setup):
        """Test that setup_knowledge_agent_settings calls setup_llamaindex_settings."""
        setup_knowledge_agent_settings()
        mock_setup.assert_called_once()


class TestEnvironmentVariableConfiguration:
    """Test that LLM functions work with environment variable configuration."""

    @patch.dict(
        os.environ,
        {"MATH_LLM_MODEL": "gpt-4o", "EMBEDDING_MODEL": "text-embedding-3-large"},
    )
    @patch("app.core.llm.get_settings")
    def test_math_agent_with_env_config(self, mock_get_settings):
        """Test that math agent uses environment-configured model."""
        mock_settings = Mock()
        mock_settings.MATH_LLM_MODEL = "gpt-4o"  # From env var
        mock_settings.ensure_openai_api_key.return_value = "test-key"
        mock_get_settings.return_value = mock_settings

        llm = get_math_agent_llm()
        assert llm.model_name == "gpt-4o"

    @patch.dict(
        os.environ,
        {"ROUTER_LLM_MODEL": "gpt-4", "EMBEDDING_MODEL": "text-embedding-3-small"},
    )
    @patch("app.core.llm.get_settings")
    def test_router_agent_with_env_config(self, mock_get_settings):
        """Test that router agent uses environment-configured model."""
        mock_settings = Mock()
        mock_settings.ROUTER_LLM_MODEL = "gpt-4"  # From env var
        mock_settings.ensure_openai_api_key.return_value = "test-key"
        mock_get_settings.return_value = mock_settings

        llm = get_router_agent_llm()
        assert llm.model_name == "gpt-4"
