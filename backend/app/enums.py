from enum import StrEnum


class Agent(StrEnum):
    MathAgent = "MathAgent"
    KnowledgeAgent = "KnowledgeAgent"


class WorkflowSignal(StrEnum):
    UnsupportedLanguage = "UnsupportedLanguage"
    Error = "Error"


class ErrorMessage(StrEnum):
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

    # Math Agent errors
    MATH_EVALUATION_FAILED = "I couldn't solve that mathematical expression."
    MATH_NON_NUMERICAL_RESULT = "The result is not a valid number."

    # Knowledge Agent errors
    KNOWLEDGE_BASE_UNAVAILABLE = (
        "The knowledge base is not available at the moment. It may be initializing."
    )
    KNOWLEDGE_NO_INFORMATION = (
        "I don't have information about that in the available documentation."
    )
    KNOWLEDGE_QUERY_FAILED = "Error querying the knowledge base."

    # API errors
    API_VALIDATION_ERROR = "Request validation failed."
    API_INTERNAL_ERROR = "An internal error occurred."
    API_SERVICE_UNAVAILABLE = "Service temporarily unavailable."

    # Redis errors
    REDIS_OPERATION_FAILED = "Redis operation failed."
