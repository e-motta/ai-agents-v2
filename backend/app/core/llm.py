"""
Centralized LLM factory functions for all agents.

This module provides factory functions to create LLM instances consistently
across all agents, ensuring uniform configuration and easy maintenance.
"""

from langchain_openai import ChatOpenAI
from llama_index.core import Settings
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI

from app.core.settings import get_settings


def get_chat_openai_llm(model: str | None = None, temperature: float = 0) -> ChatOpenAI:
    """
    Create a ChatOpenAI instance for LangChain agents.

    Args:
        model: The OpenAI model to use (defaults from settings)
        temperature: Temperature for response generation

    Returns:
        ChatOpenAI instance configured for the agent
    """
    settings = get_settings()
    # Validate presence of API key but do not pass it explicitly
    settings.ensure_openai_api_key()

    model_name = model or settings.LLM_MODEL
    return ChatOpenAI(model=model_name, temperature=temperature)


def setup_llamaindex_settings(
    llm_model: str | None = None,
    embedding_model: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> None:
    """
    Setup LlamaIndex global settings for LLM and embeddings.

    Args:
        llm_model: The OpenAI model to use for LLM (defaults from settings)
        embedding_model: The OpenAI model to use for embeddings (defaults from settings)
        chunk_size: Size of text chunks for processing (defaults from settings)
        chunk_overlap: Overlap between chunks (defaults from settings)
    """
    settings = get_settings()
    # Validate presence of API key but do not pass it explicitly
    settings.ensure_openai_api_key()

    # Configure LlamaIndex settings
    Settings.llm = LlamaIndexOpenAI(
        model=llm_model or settings.LLM_MODEL, temperature=0
    )
    Settings.embed_model = OpenAIEmbedding(
        model=embedding_model or settings.EMBEDDING_MODEL
    )
    Settings.node_parser = SimpleNodeParser.from_defaults(
        chunk_size=(chunk_size if chunk_size is not None else settings.CHUNK_SIZE),
        chunk_overlap=(
            chunk_overlap if chunk_overlap is not None else settings.CHUNK_OVERLAP
        ),
    )


# Convenience functions for common configurations
def get_math_agent_llm() -> ChatOpenAI:
    """Get ChatOpenAI LLM configured for math agent."""
    return get_chat_openai_llm(model="gpt-3.5-turbo", temperature=0)


def get_router_agent_llm() -> ChatOpenAI:
    """Get ChatOpenAI LLM configured for router agent."""
    return get_chat_openai_llm(model="gpt-3.5-turbo", temperature=0)


def setup_knowledge_agent_settings() -> None:
    """Setup LlamaIndex settings for knowledge agent."""
    setup_llamaindex_settings(
        llm_model="gpt-3.5-turbo",
        embedding_model="text-embedding-3-small",
        chunk_size=1024,
        chunk_overlap=20,
    )
