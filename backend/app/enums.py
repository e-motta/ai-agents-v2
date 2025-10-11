from enum import StrEnum


class Agents(StrEnum):
    MathAgent = "MathAgent"
    KnowledgeAgent = "KnowledgeAgent"


class WorkflowSignals(StrEnum):
    UnsupportedLanguage = "UnsupportedLanguage"
    Error = "Error"


class SystemMessages(StrEnum):
    """Standardized error messages for the application."""

    # General errors
    GENERIC_ERROR = (
        "Sorry, I could not process your request. "
        "/ Desculpe, não consegui processar a sua pergunta."
    )
    UNSUPPORTED_LANGUAGE = (
        "Unsupported language. Please ask in English or Portuguese. "
        "/ Por favor, pergunte em inglês ou português."
    )

    # API errors
    API_VALIDATION_ERROR = "Request validation failed."
    API_INTERNAL_ERROR = "An internal error occurred."
    API_SERVICE_UNAVAILABLE = "Service temporarily unavailable."

    # Redis errors
    REDIS_OPERATION_FAILED = "Redis operation failed."


class KnowledgeAgentMessages(StrEnum):
    VECTOR_STORE_EXISTS = "Vector store already exists. Deleting contents to rebuild."
    VECTOR_STORE_CANNOT_DELETE = (
        "Cannot delete vector store contents (resource busy). "
        "Attempting to build index in existing directory."
    )
    VECTOR_STORE_CREATING = "Creating new vector store"
    VECTOR_STORE_CREATED = "Vector store built and persisted successfully"
    DOCUMENTS_CREATING_ERROR = "No documents were created during crawling."

    QUERY_ENGINE_INITIALIZING = "Initializing query engine from persisted store"
    QUERY_ENGINE_NOT_FOUND = (
        "Vector store not found. Knowledge agent is disabled until the index is built."
    )
    QUERY_ENGINE_INITIALIZED = "Query engine initialized successfully"

    QUERY_CANNOT_BE_EMPTY = "Query cannot be empty."
    QUERY_INITIALIZING = "Starting knowledge base query"
    QUERY_COMPLETED = "Knowledge base query completed"

    KNOWLEDGE_BASE_UNAVAILABLE = (
        "The knowledge base is not available at the moment. It may be initializing."
    )
    KNOWLEDGE_NO_INFORMATION = (
        "I don't have information about that in the available documentation."
    )
    KNOWLEDGE_QUERY_FAILED = "Error querying the knowledge base."

    # Scraping messages
    SCRAPING_STARTING = "Starting comprehensive crawl of InfinitePay help center"
    SCRAPING_CONTENT_FROM_URL = "Scraping content from URL"
    SCRAPING_SUCCESS = "Successfully scraped content"
    SCRAPING_ERROR = "Error scraping URL"
    SCRAPING_FINDING_COLLECTIONS = "Finding collection links"
    SCRAPING_FOUND_COLLECTION = "Found collection link"
    SCRAPING_COLLECTIONS_COMPLETED = "Collection links search completed"
    SCRAPING_FINDING_ARTICLES = "Finding article links"
    SCRAPING_FOUND_ARTICLE = "Found article link"
    SCRAPING_ARTICLES_COMPLETED = "Article links search completed"
    SCRAPING_PROCESSING_ARTICLE = "Processing article"
    SCRAPING_CREATED_DOCUMENT = "Created document"
    SCRAPING_NO_CONTENT = "No content found"
    SCRAPING_ERROR_PROCESSING = "Error processing article"
    SCRAPING_COMPLETED = "Crawling completed"
    SCRAPING_ERROR_DURING = "Error during crawling"
    SCRAPING_COLLECTION_ERROR = "Error finding collection links"
    SCRAPING_ARTICLE_ERROR = "Error finding article links"

    # Index and storage messages
    INDEX_ERROR_CREATING = "Error creating or persisting vector index"
    INDEX_ERROR_LOADING = "Failed to load vector index"
    INDEX_ERROR_QUERY_ENGINE = "Failed to create query engine"
    STORAGE_ERROR_CREATING = "Failed to create ChromaDB client or collection"
    STORAGE_ERROR_LOADING = "Failed to load ChromaDB client or collection"


class RouterAgentMessages(StrEnum):
    # Validation messages
    QUERY_CANNOT_BE_EMPTY = "Query cannot be empty"

    # Routing messages
    ROUTING_QUERY = "Routing query"
    ROUTING_ERROR = "Error routing query"
    ROUTING_INVALID_RESPONSE = "Invalid response from router"

    # Security messages
    SECURITY_SUSPICIOUS_CONTENT = "Suspicious content detected in query"
    SECURITY_SUSPICIOUS_RETURN_KNOWLEDGE = (
        "Suspicious content detected, returning KnowledgeAgent for safety"
    )

    # Conversion messages
    CONVERSION_STARTING = "Starting response conversion"
    CONVERSION_COMPLETED = "Response conversion completed"
    CONVERSION_FAILED_NO_RESULT = "Response conversion failed - no result"
    CONVERSION_ERROR = "Response conversion error"
    CONVERSION_FALLBACK = "Falling back to original response due to conversion failure"


class MathAgentMessages(StrEnum):
    MATH_EVALUATION_STARTING = "Starting math evaluation"
    MATH_EVALUATION_COMPLETED = "Math evaluation completed"

    MATH_NON_NUMERICAL_RESULT = "The result is not a valid number."
    MATH_VALIDATION_FAILED = "Math validation failed."
    MATH_EVALUATION_FAILED = "An unexpected error occurred during math evaluation"

    MATH_VALIDATION_EXCEEDS_LIMIT = (
        "Result magnitude |{value}| exceeds the limit of {max_result_value}."
    )
    MATH_VALIDATION_NAN = "Result is Not a Number (NaN)."
    MATH_VALIDATION_NO_NUMERIC_DATA = (
        "Result '{result_text}' contains no valid numeric data."
    )
    MATH_VALIDATION_ERROR = "LLM returned an empty or explicit error message."
    MATH_CONVERSION_FAILED = "Failed to convert '{cleaned_text}' to float: {error}"
    MATH_LLM_QUERY = "Evaluate this mathematical expression: {query}"
