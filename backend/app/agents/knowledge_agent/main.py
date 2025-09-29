import shutil
import time

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

from app.security.prompts import KNOWLEDGE_AGENT_SYSTEM_PROMPT
from app.core.settings import get_settings
from app.core.llm import setup_knowledge_agent_settings
from app.agents.knowledge_agent.scraping import crawl_help_center
from app.core.logging import get_logger
from app.enums import ErrorMessage

logger = get_logger(__name__)
_settings = get_settings()
VECTOR_STORE_PATH = _settings.VECTOR_STORE_PATH
COLLECTION_NAME = _settings.COLLECTION_NAME


def build_index_from_scratch():
    """
    Crawls, scrapes, and builds the vector store from scratch.
    """
    start_time = time.time()

    if VECTOR_STORE_PATH.exists():
        logger.warning(
            "Vector store already exists. Deleting contents to rebuild.",
            vector_store_path=str(VECTOR_STORE_PATH),
        )
        try:
            # Only delete contents, not the directory itself (since it's mounted)
            for item in VECTOR_STORE_PATH.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
        except OSError as e:
            if e.errno == 16:  # Device or resource busy
                logger.warning(
                    "Cannot delete vector store contents (resource busy). "
                    "This may be due to multiple pods accessing the same PVC. "
                    "Attempting to build index in existing directory.",
                    vector_store_path=str(VECTOR_STORE_PATH),
                    error=str(e),
                )
                # Don't exit, continue with building in the existing directory
            else:
                raise

    logger.info(
        "Creating new vector store",
        vector_store_path=str(VECTOR_STORE_PATH),
        collection_name=COLLECTION_NAME,
    )
    setup_knowledge_agent_settings()

    documents = crawl_help_center()
    if not documents:
        logger.error("No documents were created during crawling")
        raise ValueError("No documents were created during crawling.")

    chroma_client = chromadb.PersistentClient(path=str(VECTOR_STORE_PATH / "chroma_db"))
    chroma_collection = chroma_client.create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, show_progress=True
    )
    index.storage_context.persist(persist_dir=str(VECTOR_STORE_PATH))

    execution_time = time.time() - start_time
    logger.info(
        "Vector store built and persisted successfully",
        documents_count=len(documents),
        execution_time=execution_time,
        vector_store_path=str(VECTOR_STORE_PATH),
    )


async def build_index_background():
    """Build the vector index in the background if it doesn't exist."""
    try:
        # Check if the collection actually exists, not just the directory
        if _settings.VECTOR_STORE_PATH.exists():
            try:
                chroma_client = chromadb.PersistentClient(
                    path=str(_settings.VECTOR_STORE_PATH / "chroma_db")
                )
                chroma_client.get_collection(_settings.COLLECTION_NAME)
                logger.info(
                    "Vector store and collection already exist, skipping background build"
                )
                return
            except Exception:
                # Collection doesn't exist, proceed with building
                pass

        logger.info(
            "Vector store or collection not found, starting background index build..."
        )
        from app.agents.knowledge_agent.main import build_index_from_scratch

        build_index_from_scratch()
        logger.info("Background index build completed successfully")
    except Exception as e:
        logger.warning(f"Background index build failed: {e}")
        logger.info("Index will be built when first knowledge query is made")


def get_query_engine() -> BaseQueryEngine | None:
    """
    FastAPI Dependency: Loads the pre-built index from disk and returns a
    configured query engine. Returns None if the vector store is not found.
    """
    start_time = time.time()

    logger.info(
        "Initializing query engine from persisted store",
        vector_store_path=str(VECTOR_STORE_PATH),
        collection_name=COLLECTION_NAME,
    )

    if not VECTOR_STORE_PATH.exists():
        logger.warning(
            "Vector store not found. Knowledge agent is disabled until the index is built.",
            vector_store_path=str(VECTOR_STORE_PATH),
        )
        return None

    # Setup LLM settings for LlamaIndex (this is safe to call multiple times)
    setup_knowledge_agent_settings()

    # Load the persisted ChromaDB store
    try:
        chroma_client = chromadb.PersistentClient(
            path=str(VECTOR_STORE_PATH / "chroma_db")
        )
        chroma_collection = chroma_client.get_collection(COLLECTION_NAME)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    except Exception as e:
        logger.warning(
            "Failed to load vector store collection. Knowledge agent will be disabled until the index is built.",
            collection_name=COLLECTION_NAME,
            error=str(e),
            vector_store_path=str(VECTOR_STORE_PATH),
        )
        return None

    # Load the index from the vector store
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    execution_time = time.time() - start_time
    logger.info(
        "Query engine initialized successfully",
        execution_time=execution_time,
        vector_store_path=str(VECTOR_STORE_PATH),
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
        raise ValueError("Query cannot be empty.")

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
        logger.error(
            "Error querying knowledge base",
            query=query,
            error=str(e),
            execution_time=execution_time,
        )
        raise ValueError(f"{ErrorMessage.KNOWLEDGE_QUERY_FAILED}: {str(e)}")
