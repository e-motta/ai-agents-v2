"""
Unit tests for Knowledge Agent main functions.

These tests verify the knowledge agent's core functionality including
index building, query engine initialization, and knowledge querying.
"""

import errno
from unittest.mock import AsyncMock, Mock, patch

import pytest
from llama_index.core import Document
from llama_index.core.base.base_query_engine import BaseQueryEngine

from app.agents.knowledge_agent.main import (
    build_index_from_scratch,
    get_query_engine,
    query_knowledge,
)
from app.enums import KnowledgeAgentMessages


class TestBuildIndexFromScratch:
    """Test the build_index_from_scratch function."""

    @patch("app.agents.knowledge_agent.main.get_settings")
    @patch("app.agents.knowledge_agent.main.setup_knowledge_agent_settings")
    @patch("app.agents.knowledge_agent.main.crawl_help_center")
    @patch("app.agents.knowledge_agent.main.chromadb")
    @patch("app.agents.knowledge_agent.main.VectorStoreIndex")
    @patch("app.agents.knowledge_agent.main.StorageContext")
    @patch("app.agents.knowledge_agent.main.ChromaVectorStore")
    def test_build_index_from_scratch_success(
        self,
        mock_chroma_vector_store,
        mock_storage_context,
        mock_vector_store_index,
        mock_chromadb,
        mock_crawl_help_center,
        mock_setup_knowledge_agent_settings,
        mock_get_settings,
    ):
        """Test successful index building from scratch."""
        # Mock settings
        mock_settings = Mock()
        mock_vector_store_path = Mock()
        mock_vector_store_path.exists.return_value = False
        mock_vector_store_path.__truediv__ = Mock(
            return_value="/test/vector_store/chroma_db"
        )
        mock_settings.VECTOR_STORE_PATH = mock_vector_store_path
        mock_settings.COLLECTION_NAME = "test_collection"
        mock_get_settings.return_value = mock_settings

        # Mock crawled documents
        mock_documents = [
            Document(text="Test content 1", metadata={"url": "http://test1.com"}),
            Document(text="Test content 2", metadata={"url": "http://test2.com"}),
        ]
        mock_crawl_help_center.return_value = mock_documents

        # Mock ChromaDB
        mock_chroma_client = Mock()
        mock_chroma_collection = Mock()
        mock_chroma_client.create_collection.return_value = mock_chroma_collection
        mock_chromadb.PersistentClient.return_value = mock_chroma_client

        # Mock vector store and index
        mock_vector_store = Mock()
        mock_chroma_vector_store.return_value = mock_vector_store
        mock_storage_context.from_defaults.return_value = Mock()
        mock_index = Mock()
        mock_vector_store_index.from_documents.return_value = mock_index

        # Call the function
        build_index_from_scratch()

        # Verify setup was called
        mock_setup_knowledge_agent_settings.assert_called_once()
        mock_crawl_help_center.assert_called_once()

        # Verify ChromaDB client was created
        mock_chromadb.PersistentClient.assert_called_once()
        mock_chroma_client.create_collection.assert_called_once_with("test_collection")

        # Verify index was created and persisted
        mock_vector_store_index.from_documents.assert_called_once()
        mock_index.storage_context.persist.assert_called_once()

    @patch("app.agents.knowledge_agent.main.get_settings")
    @patch("app.agents.knowledge_agent.main.setup_knowledge_agent_settings")
    @patch("app.agents.knowledge_agent.main.crawl_help_center")
    @patch("app.agents.knowledge_agent.main.shutil")
    def test_build_index_from_scratch_existing_directory_cleanup(
        self,
        mock_shutil,
        mock_crawl_help_center,
        mock_setup_knowledge_agent_settings,
        mock_get_settings,
    ):
        """Test index building when vector store directory already exists."""
        # Mock settings
        mock_settings = Mock()
        mock_vector_store_path = Mock()
        mock_vector_store_path.exists.return_value = True
        mock_vector_store_path.__truediv__ = Mock(
            return_value="/test/vector_store/chroma_db"
        )
        mock_settings.VECTOR_STORE_PATH = mock_vector_store_path
        mock_settings.COLLECTION_NAME = "test_collection"
        mock_get_settings.return_value = mock_settings

        # Mock directory contents
        mock_item1 = Mock()
        mock_item1.is_dir.return_value = True
        mock_item2 = Mock()
        mock_item2.is_dir.return_value = False
        mock_vector_store_path.iterdir.return_value = [mock_item1, mock_item2]

        # Mock crawled documents
        mock_documents = [
            Document(text="Test content", metadata={"url": "http://test.com"})
        ]
        mock_crawl_help_center.return_value = mock_documents

        # Mock ChromaDB and other dependencies
        with (
            patch("app.agents.knowledge_agent.main.chromadb") as mock_chromadb,
            patch(
                "app.agents.knowledge_agent.main.VectorStoreIndex"
            ) as mock_vector_store_index,
        ):
            mock_chroma_client = Mock()
            mock_chroma_collection = Mock()
            mock_chroma_client.create_collection.return_value = mock_chroma_collection
            mock_chromadb.PersistentClient.return_value = mock_chroma_client

            mock_index = Mock()
            mock_vector_store_index.from_documents.return_value = mock_index

            # Call the function
            build_index_from_scratch()

        # Verify cleanup was attempted
        mock_shutil.rmtree.assert_called_once_with(mock_item1)
        mock_item2.unlink.assert_called_once()

    @patch("app.agents.knowledge_agent.main.get_settings")
    @patch("app.agents.knowledge_agent.main.setup_knowledge_agent_settings")
    @patch("app.agents.knowledge_agent.main.crawl_help_center")
    @patch("app.agents.knowledge_agent.main.shutil")
    def test_build_index_from_scratch_ebusy_error_handling(
        self,
        mock_shutil,
        mock_crawl_help_center,
        mock_setup_knowledge_agent_settings,
        mock_get_settings,
    ):
        """Test handling of EBUSY error during cleanup."""
        # Mock settings
        mock_settings = Mock()
        mock_vector_store_path = Mock()
        mock_vector_store_path.exists.return_value = True
        mock_vector_store_path.__truediv__ = Mock(
            return_value="/test/vector_store/chroma_db"
        )
        mock_settings.VECTOR_STORE_PATH = mock_vector_store_path
        mock_settings.COLLECTION_NAME = "test_collection"
        mock_get_settings.return_value = mock_settings

        # Mock directory contents
        mock_item = Mock()
        mock_item.is_dir.return_value = True
        mock_vector_store_path.iterdir.return_value = [mock_item]

        # Mock EBUSY error
        ebusy_error = OSError("Device or resource busy")
        ebusy_error.errno = errno.EBUSY
        mock_shutil.rmtree.side_effect = ebusy_error

        # Mock crawled documents
        mock_documents = [
            Document(text="Test content", metadata={"url": "http://test.com"})
        ]
        mock_crawl_help_center.return_value = mock_documents

        # Mock ChromaDB and other dependencies
        with (
            patch("app.agents.knowledge_agent.main.chromadb") as mock_chromadb,
            patch(
                "app.agents.knowledge_agent.main.VectorStoreIndex"
            ) as mock_vector_store_index,
        ):
            mock_chroma_client = Mock()
            mock_chroma_collection = Mock()
            mock_chroma_client.create_collection.return_value = mock_chroma_collection
            mock_chromadb.PersistentClient.return_value = mock_chroma_client

            mock_index = Mock()
            mock_vector_store_index.from_documents.return_value = mock_index

            # Call the function - should not raise exception
            build_index_from_scratch()

        # Verify cleanup was attempted
        mock_shutil.rmtree.assert_called_once_with(mock_item)

    @patch("app.agents.knowledge_agent.main.get_settings")
    @patch("app.agents.knowledge_agent.main.setup_knowledge_agent_settings")
    @patch("app.agents.knowledge_agent.main.crawl_help_center")
    def test_build_index_from_scratch_no_documents_error(
        self,
        mock_crawl_help_center,
        mock_setup_knowledge_agent_settings,
        mock_get_settings,
    ):
        """Test error handling when no documents are created."""
        # Mock settings
        mock_settings = Mock()
        mock_vector_store_path = Mock()
        mock_vector_store_path.exists.return_value = False
        mock_settings.VECTOR_STORE_PATH = mock_vector_store_path
        mock_settings.COLLECTION_NAME = "test_collection"
        mock_get_settings.return_value = mock_settings

        # Mock no documents returned
        mock_crawl_help_center.return_value = []

        # Call the function and expect ValueError
        with pytest.raises(
            ValueError, match=r"No documents were created during crawling."
        ):
            build_index_from_scratch()


class TestGetQueryEngine:
    """Test the get_query_engine function."""

    @patch("app.agents.knowledge_agent.main.get_settings")
    @patch("app.agents.knowledge_agent.main.setup_knowledge_agent_settings")
    @patch("app.agents.knowledge_agent.main.chromadb")
    @patch("app.agents.knowledge_agent.main.VectorStoreIndex")
    @patch("app.agents.knowledge_agent.main.ChromaVectorStore")
    def test_get_query_engine_success(
        self,
        mock_chroma_vector_store,
        mock_vector_store_index,
        mock_chromadb,
        mock_setup_knowledge_agent_settings,
        mock_get_settings,
    ):
        """Test successful query engine initialization."""
        # Mock settings
        mock_settings = Mock()
        mock_vector_store_path = Mock()
        mock_vector_store_path.exists.return_value = True
        mock_vector_store_path.__truediv__ = Mock(
            return_value="/test/vector_store/chroma_db"
        )
        mock_settings.VECTOR_STORE_PATH = mock_vector_store_path
        mock_settings.COLLECTION_NAME = "test_collection"
        mock_get_settings.return_value = mock_settings

        # Mock ChromaDB
        mock_chroma_client = Mock()
        mock_chroma_collection = Mock()
        mock_chroma_client.get_collection.return_value = mock_chroma_collection
        mock_chromadb.PersistentClient.return_value = mock_chroma_client

        # Mock vector store and index
        mock_vector_store = Mock()
        mock_chroma_vector_store.return_value = mock_vector_store
        mock_index = Mock()
        mock_query_engine = Mock()
        mock_index.as_query_engine.return_value = mock_query_engine
        mock_vector_store_index.from_vector_store.return_value = mock_index

        # Call the function
        result = get_query_engine()

        # Verify setup was called
        mock_setup_knowledge_agent_settings.assert_called_once()

        # Verify ChromaDB client was created
        mock_chromadb.PersistentClient.assert_called_once()
        mock_chroma_client.get_collection.assert_called_once_with("test_collection")

        # Verify index was loaded and query engine created
        mock_vector_store_index.from_vector_store.assert_called_once_with(
            vector_store=mock_vector_store
        )
        mock_index.as_query_engine.assert_called_once()

        assert result == mock_query_engine

    @patch("app.agents.knowledge_agent.main.get_settings")
    def test_get_query_engine_vector_store_not_found(self, mock_get_settings):
        """Test query engine initialization when vector store doesn't exist."""
        # Mock settings
        mock_settings = Mock()
        mock_vector_store_path = Mock()
        mock_vector_store_path.exists.return_value = False
        mock_settings.VECTOR_STORE_PATH = mock_vector_store_path
        mock_settings.COLLECTION_NAME = "test_collection"
        mock_get_settings.return_value = mock_settings

        # Call the function
        result = get_query_engine()

        # Should return None
        assert result is None

    @patch("app.agents.knowledge_agent.main.get_settings")
    @patch("app.agents.knowledge_agent.main.setup_knowledge_agent_settings")
    @patch("app.agents.knowledge_agent.main.chromadb")
    def test_get_query_engine_chromadb_error(
        self,
        mock_chromadb,
        mock_setup_knowledge_agent_settings,
        mock_get_settings,
    ):
        """Test query engine initialization when ChromaDB fails."""
        # Mock settings
        mock_settings = Mock()
        mock_vector_store_path = Mock()
        mock_vector_store_path.exists.return_value = True
        mock_vector_store_path.__truediv__ = Mock(
            return_value="/test/vector_store/chroma_db"
        )
        mock_settings.VECTOR_STORE_PATH = mock_vector_store_path
        mock_settings.COLLECTION_NAME = "test_collection"
        mock_get_settings.return_value = mock_settings

        # Mock ChromaDB to raise exception
        mock_chromadb.PersistentClient.side_effect = Exception("ChromaDB error")

        # Call the function
        result = get_query_engine()

        # Should return None due to error
        assert result is None


class TestQueryKnowledge:
    """Test the query_knowledge function."""

    @pytest.mark.asyncio
    async def test_query_knowledge_success(self):
        """Test successful knowledge query."""
        # Mock query engine
        mock_query_engine = AsyncMock(spec=BaseQueryEngine)
        mock_response = Mock()
        mock_response.__str__ = Mock(return_value="Test answer")
        mock_response.source_nodes = []
        mock_query_engine.aquery.return_value = mock_response

        # Call the function
        result = await query_knowledge("Test query", mock_query_engine)

        # Verify result
        assert result == "Test answer"
        mock_query_engine.aquery.assert_called_once_with("Test query")

    @pytest.mark.asyncio
    async def test_query_knowledge_empty_query(self):
        """Test query knowledge with empty query."""
        mock_query_engine = AsyncMock(spec=BaseQueryEngine)

        # Call the function with empty query
        with pytest.raises(ValueError, match=r"Query cannot be empty."):
            await query_knowledge("", mock_query_engine)

    @pytest.mark.asyncio
    async def test_query_knowledge_no_information(self):
        """Test query knowledge when no information is found."""
        # Mock query engine
        mock_query_engine = AsyncMock(spec=BaseQueryEngine)
        mock_response = Mock()
        mock_response.__str__ = Mock(return_value="")
        mock_response.source_nodes = []
        mock_query_engine.aquery.return_value = mock_response

        # Call the function
        result = await query_knowledge("Test query", mock_query_engine)

        # Should return no information message
        assert result == KnowledgeAgentMessages.KNOWLEDGE_NO_INFORMATION

    @pytest.mark.asyncio
    async def test_query_knowledge_with_sources(self):
        """Test query knowledge with source information."""
        # Mock query engine
        mock_query_engine = AsyncMock(spec=BaseQueryEngine)
        mock_response = Mock()
        mock_response.__str__ = Mock(return_value="Test answer with sources")

        # Mock source nodes
        mock_source_node = Mock()
        mock_source_node.score = 0.95
        mock_source_node.node = Mock()
        mock_source_node.node.metadata = {
            "url": "http://test.com",
            "source": "test_source",
        }
        mock_response.source_nodes = [mock_source_node]

        mock_query_engine.aquery.return_value = mock_response

        # Call the function
        result = await query_knowledge("Test query", mock_query_engine)

        # Verify result
        assert result == "Test answer with sources"
        mock_query_engine.aquery.assert_called_once_with("Test query")

    @pytest.mark.asyncio
    async def test_query_knowledge_exception_handling(self):
        """Test query knowledge exception handling."""
        # Mock query engine
        mock_query_engine = AsyncMock(spec=BaseQueryEngine)
        mock_query_engine.aquery.side_effect = Exception("Query failed")

        # Call the function
        with pytest.raises(
            ValueError, match=KnowledgeAgentMessages.KNOWLEDGE_QUERY_FAILED
        ):
            await query_knowledge("Test query", mock_query_engine)

    @pytest.mark.asyncio
    async def test_query_knowledge_none_response(self):
        """Test query knowledge with None response."""
        # Mock query engine
        mock_query_engine = AsyncMock(spec=BaseQueryEngine)
        mock_response = Mock()
        mock_response.__str__ = Mock(return_value="None")
        mock_response.source_nodes = []
        mock_query_engine.aquery.return_value = mock_response

        # Call the function
        result = await query_knowledge("Test query", mock_query_engine)

        # Should return no information message
        assert result == KnowledgeAgentMessages.KNOWLEDGE_NO_INFORMATION

    @pytest.mark.asyncio
    async def test_query_knowledge_null_response(self):
        """Test query knowledge with null response."""
        # Mock query engine
        mock_query_engine = AsyncMock(spec=BaseQueryEngine)
        mock_response = Mock()
        mock_response.__str__ = Mock(return_value="null")
        mock_response.source_nodes = []
        mock_query_engine.aquery.return_value = mock_response

        # Call the function
        result = await query_knowledge("Test query", mock_query_engine)

        # Should return no information message
        assert result == KnowledgeAgentMessages.KNOWLEDGE_NO_INFORMATION
