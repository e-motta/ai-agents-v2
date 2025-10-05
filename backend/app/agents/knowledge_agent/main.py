import errno
import shutil
import time

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.vector_stores.chroma import ChromaVectorStore

from app.agents.knowledge_agent.scraping import crawl_help_center
from app.core.llm import setup_knowledge_agent_settings
from app.core.logging import get_logger
from app.core.settings import get_settings
from app.enums import ErrorMessage
from app.security.prompts import KNOWLEDGE_AGENT_SYSTEM_PROMPT

logger = get_logger(__name__)


def build_index_from_scratch() -> None:
    """
    Crawls, scrapes, and builds the vector store from scratch.
    """
    settings = get_settings()
    vector_store_path = settings.VECTOR_STORE_PATH
    collection_name = settings.COLLECTION_NAME

    start_time = time.time()

    if vector_store_path.exists():
        logger.warning(
            "Vector store already exists. Deleting contents to rebuild.",
            vector_store_path=str(vector_store_path),
        )
        try:
            # Only delete contents, not the directory itself (since it's mounted)
            for item in vector_store_path.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
        except OSError as e:
            if e.errno == errno.EBUSY:  # Device or resource busy
                logger.warning(
                    "Cannot delete vector store contents (resource busy). "
                    "This may be due to multiple pods accessing the same PVC. "
                    "Attempting to build index in existing directory.",
                    vector_store_path=str(vector_store_path),
                    error=str(e),
                )
                # Don't exit, continue with building in the existing directory
            else:
                raise

    logger.info(
        "Creating new vector store",
        vector_store_path=str(vector_store_path),
        collection_name=collection_name,
    )
    setup_knowledge_agent_settings()

    documents = crawl_help_center()
    if not documents:
        error_msg = "No documents were created during crawling."
        logger.error(error_msg)
        raise ValueError(error_msg)

    chroma_client = chromadb.PersistentClient(path=str(vector_store_path / "chroma_db"))
    chroma_collection = chroma_client.create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, show_progress=True
    )
    index.storage_context.persist(persist_dir=str(vector_store_path))

    execution_time = time.time() - start_time
    logger.info(
        "Vector store built and persisted successfully",
        documents_count=len(documents),
        execution_time=execution_time,
        vector_store_path=str(vector_store_path),
    )


def get_query_engine() -> BaseQueryEngine | None:
    """
    FastAPI Dependency: Loads the pre-built index from disk and returns a
    configured query engine. Returns None if the vector store is not found.
    """
    settings = get_settings()
    vector_store_path = settings.VECTOR_STORE_PATH
    collection_name = settings.COLLECTION_NAME

    start_time = time.time()

    logger.info(
        "Initializing query engine from persisted store",
        vector_store_path=str(vector_store_path),
        collection_name=collection_name,
    )

    if not vector_store_path.exists():
        logger.warning(
            "Vector store not found. "
            "Knowledge agent is disabled until the index is built.",
            vector_store_path=str(vector_store_path),
        )
        return None

    # Setup LLM settings for LlamaIndex (this is safe to call multiple times)
    setup_knowledge_agent_settings()

    # Load the persisted ChromaDB store
    try:
        chroma_client = chromadb.PersistentClient(
            path=str(vector_store_path / "chroma_db")
        )
        chroma_collection = chroma_client.get_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    except Exception as e:
        logger.warning(
            "Failed to load vector store collection. "
            "Knowledge agent will be disabled until the index is built.",
            collection_name=collection_name,
            error=str(e),
            vector_store_path=str(vector_store_path),
        )
        return None

    # Load the index from the vector store
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    execution_time = time.time() - start_time
    logger.info(
        "Query engine initialized successfully",
        execution_time=execution_time,
        vector_store_path=str(vector_store_path),
    )

    # Return the configured query engine
    return index.as_query_engine(
        system_prompt=KNOWLEDGE_AGENT_SYSTEM_PROMPT,
        similarity_top_k=5,
        response_mode="compact",
    )


async def query_knowledge(query: str, query_engine: BaseQueryEngine) -> str:
    """
    Asynchronously queries the knowledge base.

    Args:
        query: The question to ask.
        query_engine: The query engine instance provided by the dependency.

    Returns:
        The answer from the knowledge base.
    """
    start_time = time.time()

    if not query:
        error_msg = "Query cannot be empty."
        raise ValueError(error_msg)

    logger.info("Starting knowledge base query", query=query, query_preview=query[:100])

    try:
        # Use the native async method for non-blocking I/O
        response = await query_engine.aquery(query)
        answer = str(response).strip()
        execution_time = time.time() - start_time

        # Extract source information from the response
        sources = []
        if hasattr(response, "source_nodes") and response.source_nodes:
            for node in response.source_nodes:
                if hasattr(node, "node") and hasattr(node.node, "metadata"):
                    source_info = {
                        "url": node.node.metadata.get("url", "Unknown"),
                        "source": node.node.metadata.get("source", "Unknown"),
                        "score": getattr(node, "score", None),
                    }
                    sources.append(source_info)

        if not answer or answer.lower() in ["", "none", "null"]:
            logger.info(
                "No information found in knowledge base",
                query=query,
                execution_time=execution_time,
                sources=sources,
            )
            return ErrorMessage.KNOWLEDGE_NO_INFORMATION

        logger.info(
            "Knowledge base query completed",
            query=query,
            answer_preview=answer[:100],
            execution_time=execution_time,
            sources=sources,
        )
        return answer
    except Exception as e:
        execution_time = time.time() - start_time
        logger.exception(
            "Error querying knowledge base",
            query=query,
            error=str(e),
            execution_time=execution_time,
        )
        error_msg = f"{ErrorMessage.KNOWLEDGE_QUERY_FAILED}: {e!s}"
        raise ValueError(error_msg) from e
