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
