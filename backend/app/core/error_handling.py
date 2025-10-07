"""
Standardized error handling utilities for the FastAPI application.

This module provides utilities for creating consistent error responses
across the application using the ErrorResponse model and ErrorMessage enum.
"""

from enum import StrEnum

from fastapi import HTTPException, status

from app.enums import KnowledgeAgentMessages, SystemMessages
from app.models import ErrorResponse


def create_error_response(
    error_message: StrEnum,
    code: str,
    details: str | None = None,
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
) -> HTTPException:
    """
    Create a standardized HTTPException with ErrorResponse format.

    Args:
        error_message: The standardized error message from ErrorMessage enum
        code: Error code identifier
        details: Optional additional details about the error
        status_code: HTTP status code

    Returns:
        HTTPException with standardized error response
    """
    error_response = ErrorResponse(error=error_message, code=code, details=details)

    return HTTPException(status_code=status_code, detail=error_response.model_dump())


def create_validation_error(details: str | None = None) -> HTTPException:
    """Create a validation error response."""
    return create_error_response(
        error_message=SystemMessages.API_VALIDATION_ERROR,
        code="VALIDATION_ERROR",
        details=details,
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
    )


def create_math_error(details: str | None = None) -> HTTPException:
    """Create a math evaluation error response."""
    return create_error_response(
        error_message=SystemMessages.MATH_EVALUATION_FAILED,
        code="MATH_ERROR",
        details=details,
        status_code=status.HTTP_400_BAD_REQUEST,
    )


def create_knowledge_error(details: str | None = None) -> HTTPException:
    """Create a knowledge base error response."""
    return create_error_response(
        error_message=KnowledgeAgentMessages.KNOWLEDGE_QUERY_FAILED,
        code="KNOWLEDGE_ERROR",
        details=details,
        status_code=status.HTTP_400_BAD_REQUEST,
    )


def create_service_unavailable_error(
    service_name: str, details: str | None = None
) -> HTTPException:
    """Create a service unavailable error response."""
    return create_error_response(
        error_message=SystemMessages.API_SERVICE_UNAVAILABLE,
        code="SERVICE_UNAVAILABLE",
        details=details or f"{service_name} is currently unavailable",
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
    )


def create_redis_error(details: str | None = None) -> HTTPException:
    """Create a Redis error response."""
    return create_error_response(
        error_message=SystemMessages.REDIS_OPERATION_FAILED,
        code="REDIS_ERROR",
        details=details,
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
    )
