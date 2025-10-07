import errno
import shutil

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.schema import NodeWithScore
from llama_index.vector_stores.chroma import ChromaVectorStore

from app.agents.knowledge_agent.scraping import crawl_help_center
from app.core.llm import setup_knowledge_agent_settings
from app.core.logging import get_logger
from app.core.settings import get_settings
from app.enums import KnowledgeAgentMessages
from app.security.prompts import KNOWLEDGE_AGENT_SYSTEM_PROMPT

logger = get_logger(__name__)


def build_index_from_scratch() -> None:
    """
    Crawls, scrapes, and builds the vector store from scratch.
    """
    settings = get_settings()
    vector_store_path = settings.VECTOR_STORE_PATH
    collection_name = settings.COLLECTION_NAME

    if vector_store_path.exists():
        logger.warning(
            KnowledgeAgentMessages.VECTOR_STORE_EXISTS,
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
                    KnowledgeAgentMessages.VECTOR_STORE_CANNOT_DELETE,
                    vector_store_path=str(vector_store_path),
                    error=str(e),
                )
                # Don't exit, continue with building in the existing directory
            else:
                raise

    logger.info(
        KnowledgeAgentMessages.VECTOR_STORE_CREATING,
        vector_store_path=str(vector_store_path),
        collection_name=collection_name,
    )
    setup_knowledge_agent_settings()

    documents = crawl_help_center()
    if not documents:
        error_msg = KnowledgeAgentMessages.DOCUMENTS_CREATING_ERROR
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

    logger.info(
        KnowledgeAgentMessages.VECTOR_STORE_CREATED,
        documents_count=len(documents),
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

    logger.info(
        KnowledgeAgentMessages.QUERY_ENGINE_INITIALIZING,
        vector_store_path=str(vector_store_path),
        collection_name=collection_name,
    )

    if not vector_store_path.exists():
        logger.warning(
            KnowledgeAgentMessages.QUERY_ENGINE_NOT_FOUND,
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
            KnowledgeAgentMessages.QUERY_ENGINE_NOT_FOUND,
            collection_name=collection_name,
            error=str(e),
            vector_store_path=str(vector_store_path),
        )
        return None

    # Load the index from the vector store
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    logger.info(
        KnowledgeAgentMessages.QUERY_ENGINE_INITIALIZED,
        vector_store_path=str(vector_store_path),
    )

    # Return the configured query engine
    return index.as_query_engine(
        system_prompt=KNOWLEDGE_AGENT_SYSTEM_PROMPT,
        similarity_top_k=5,
        response_mode="compact",
    )


def _validate_query(query: str) -> None:
    """Validates that the query string is not empty."""
    if not query:
        raise ValueError(KnowledgeAgentMessages.QUERY_CANNOT_BE_EMPTY)


def _extract_source_from_node(
    node: NodeWithScore,
) -> dict[str, str | float | None] | None:
    """Extracts metadata from a single source node if available."""
    if not (hasattr(node, "node") and hasattr(node.node, "metadata")):
        return None

    metadata = node.node.metadata
    return {
        "url": metadata.get("url", "Unknown"),
        "source": metadata.get("source", "Unknown"),
        "score": getattr(node, "score", None),
    }


def _process_engine_response(
    response: RESPONSE_TYPE,
) -> tuple[str, list[dict[str, str | float | None]]]:
    """Processes the raw query engine response to extract a clean answer and sources."""
    answer = str(response).strip()
    sources = []

    if hasattr(response, "source_nodes") and response.source_nodes:
        sources = [
            source_info
            for node in response.source_nodes
            if (source_info := _extract_source_from_node(node)) is not None
        ]

    return answer, sources


async def query_knowledge(query: str, query_engine: BaseQueryEngine) -> str:
    """
    Queries the knowledge base, processes the response, and handles errors.
    """
    _validate_query(query)

    logger.info(
        KnowledgeAgentMessages.QUERY_INITIALIZING,
        query=query,
        query_preview=query[:100],
    )

    try:
        raw_response: RESPONSE_TYPE = await query_engine.aquery(query)
        answer, sources = _process_engine_response(raw_response)

        if not answer or answer.lower() in {"", "none", "null"}:
            logger.info(
                KnowledgeAgentMessages.KNOWLEDGE_NO_INFORMATION,
                query=query,
                sources=sources,
            )
            return KnowledgeAgentMessages.KNOWLEDGE_NO_INFORMATION

        logger.info(
            KnowledgeAgentMessages.QUERY_COMPLETED,
            query=query,
            answer_preview=answer[:100],
            sources=sources,
        )
        return answer

    except Exception as e:
        logger.exception(
            KnowledgeAgentMessages.KNOWLEDGE_QUERY_FAILED, query=query, error=str(e)
        )
        raise ValueError(KnowledgeAgentMessages.KNOWLEDGE_QUERY_FAILED) from e
