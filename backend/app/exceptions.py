"""
Custom exceptions for the AI Agents application.

This module defines custom exception classes that provide more specific
error handling and better debugging capabilities compared to generic exceptions.
"""

from typing import Any


class MathAgentError(Exception):
    """Base exception class for all math agent related errors."""

    def __init__(self, message: str, details: Any | None = None) -> None:
        """
        Initialize the exception with a message and optional details.

        Args:
            message: The error message describing what went wrong.
            details: Optional additional details about the error context.
        """
        super().__init__(message)
        self.message = message
        self.details = details


class MathValidationError(MathAgentError):
    """Raised when mathematical validation fails."""

    def __init__(
        self, message: str, result_text: str | None = None, details: Any | None = None
    ) -> None:
        """
        Initialize the validation error.

        Args:
            message: The validation error message.
            result_text: The text that failed validation.
            details: Optional additional details about the validation failure.
        """
        super().__init__(message, details)
        self.result_text = result_text


class MathResultError(MathAgentError):
    """Raised when the mathematical result is invalid or out of bounds."""

    def __init__(
        self,
        message: str,
        value: float | None = None,
        max_value: float | None = None,
        details: Any | None = None,
    ) -> None:
        """
        Initialize the result error.

        Args:
            message: The result error message.
            value: The invalid value that caused the error.
            max_value: The maximum allowed value (if applicable).
            details: Optional additional details about the result error.
        """
        super().__init__(message, details)
        self.value = value
        self.max_value = max_value


class MathEvaluationError(MathAgentError):
    """Raised when the mathematical evaluation process fails."""

    def __init__(
        self, message: str, query: str | None = None, details: Any | None = None
    ) -> None:
        """
        Initialize the evaluation error.

        Args:
            message: The evaluation error message.
            query: The mathematical query that failed to evaluate.
            details: Optional additional details about the evaluation failure.
        """
        super().__init__(message, details)
        self.query = query


class MathConversionError(MathAgentError):
    """Raised when converting text to numeric values fails."""

    def __init__(
        self, message: str, input_text: str | None = None, details: Any | None = None
    ) -> None:
        """
        Initialize the conversion error.

        Args:
            message: The conversion error message.
            input_text: The text that failed to convert.
            details: Optional additional details about the conversion failure.
        """
        super().__init__(message, details)
        self.input_text = input_text


class KnowledgeAgentError(Exception):
    """Base exception class for all knowledge agent related errors."""

    def __init__(self, message: str, details: Any | None = None) -> None:
        """
        Initialize the exception with a message and optional details.

        Args:
            message: The error message describing what went wrong.
            details: Optional additional details about the error context.
        """
        super().__init__(message)
        self.message = message
        self.details = details


class KnowledgeValidationError(KnowledgeAgentError):
    """Raised when knowledge agent validation fails."""

    def __init__(
        self, message: str, query: str | None = None, details: Any | None = None
    ) -> None:
        """
        Initialize the validation error.

        Args:
            message: The validation error message.
            query: The query that failed validation.
            details: Optional additional details about the validation failure.
        """
        super().__init__(message, details)
        self.query = query


class KnowledgeQueryError(KnowledgeAgentError):
    """Raised when knowledge base querying fails."""

    def __init__(
        self, message: str, query: str | None = None, details: Any | None = None
    ) -> None:
        """
        Initialize the query error.

        Args:
            message: The query error message.
            query: The query that failed.
            details: Optional additional details about the query failure.
        """
        super().__init__(message, details)
        self.query = query


class KnowledgeIndexError(KnowledgeAgentError):
    """Raised when knowledge index operations fail."""

    def __init__(
        self, message: str, operation: str | None = None, details: Any | None = None
    ) -> None:
        """
        Initialize the index error.

        Args:
            message: The index error message.
            operation: The operation that failed (e.g., 'build', 'load', 'persist').
            details: Optional additional details about the index failure.
        """
        super().__init__(message, details)
        self.operation = operation


class KnowledgeScrapingError(KnowledgeAgentError):
    """Raised when web scraping operations fail."""

    def __init__(
        self, message: str, url: str | None = None, details: Any | None = None
    ) -> None:
        """
        Initialize the scraping error.

        Args:
            message: The scraping error message.
            url: The URL that failed to scrape.
            details: Optional additional details about the scraping failure.
        """
        super().__init__(message, details)
        self.url = url


class KnowledgeStorageError(KnowledgeAgentError):
    """Raised when vector store operations fail."""

    def __init__(
        self,
        message: str,
        operation: str | None = None,
        path: str | None = None,
        details: Any | None = None,
    ) -> None:
        """
        Initialize the storage error.

        Args:
            message: The storage error message.
            operation: The storage operation that failed.
            path: The storage path where the error occurred.
            details: Optional additional details about the storage failure.
        """
        super().__init__(message, details)
        self.operation = operation
        self.path = path


class RouterAgentError(Exception):
    """Base exception class for all router agent related errors."""

    def __init__(self, message: str, details: Any | None = None) -> None:
        """
        Initialize the exception with a message and optional details.

        Args:
            message: The error message describing what went wrong.
            details: Optional additional details about the error context.
        """
        super().__init__(message)
        self.message = message
        self.details = details


class RouterValidationError(RouterAgentError):
    """Raised when router agent validation fails."""

    def __init__(
        self, message: str, query: str | None = None, details: Any | None = None
    ) -> None:
        """
        Initialize the validation error.

        Args:
            message: The validation error message.
            query: The query that failed validation.
            details: Optional additional details about the validation failure.
        """
        super().__init__(message, details)
        self.query = query


class RouterRoutingError(RouterAgentError):
    """Raised when query routing fails."""

    def __init__(
        self, message: str, query: str | None = None, details: Any | None = None
    ) -> None:
        """
        Initialize the routing error.

        Args:
            message: The routing error message.
            query: The query that failed to route.
            details: Optional additional details about the routing failure.
        """
        super().__init__(message, details)
        self.query = query


class RouterConversionError(RouterAgentError):
    """Raised when response conversion fails."""

    def __init__(
        self,
        message: str,
        agent_type: str | None = None,
        original_response: str | None = None,
        details: Any | None = None,
    ) -> None:
        """
        Initialize the conversion error.

        Args:
            message: The conversion error message.
            agent_type: The type of agent that generated the response.
            original_response: The original response that failed to convert.
            details: Optional additional details about the conversion failure.
        """
        super().__init__(message, details)
        self.agent_type = agent_type
        self.original_response = original_response


class RouterSecurityError(RouterAgentError):
    """Raised when security-related issues are detected."""

    def __init__(
        self,
        message: str,
        query: str | None = None,
        pattern: str | None = None,
        details: Any | None = None,
    ) -> None:
        """
        Initialize the security error.

        Args:
            message: The security error message.
            query: The query that triggered the security issue.
            pattern: The suspicious pattern that was detected.
            details: Optional additional details about the security issue.
        """
        super().__init__(message, details)
        self.query = query
        self.pattern = pattern
