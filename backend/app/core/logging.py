"""
Structured logging configuration using structlog.

This module configures Python's structlog to output structured JSON logs
with all required fields for the agent system.
"""

import logging
import sys
from collections.abc import MutableMapping
from datetime import UTC, datetime
from typing import Any

import structlog
from structlog.stdlib import LoggerFactory


def add_timestamp(
    logger: Any,  # noqa: ARG001
    method_name: str,  # noqa: ARG001
    event_dict: MutableMapping[str, Any],
) -> MutableMapping[str, Any]:
    """Add ISO 8601 timestamp to log events."""
    event_dict["timestamp"] = datetime.now(UTC).isoformat()
    return event_dict


def add_agent_context(
    logger: Any,
    method_name: str,  # noqa: ARG001
    event_dict: MutableMapping[str, Any],
) -> MutableMapping[str, Any]:
    """Add agent context to log events."""
    # Extract agent name from logger name if available
    logger_name = logger.name if hasattr(logger, "name") else ""
    if "router_agent" in logger_name:
        event_dict["agent"] = "RouterAgent"
    elif "math_agent" in logger_name:
        event_dict["agent"] = "MathAgent"
    elif "knowledge_agent" in logger_name:
        event_dict["agent"] = "KnowledgeAgent"
    else:
        event_dict["agent"] = "System"

    return event_dict


def configure_logging() -> None:
    """
    Configure structlog for structured JSON logging.

    This function sets up:
    - JSON output format
    - Required fields: timestamp, level, agent, conversation_id, user_id
    - Agent-specific fields: decision (RouterAgent) or processed_content (other agents)
    - Execution time tracking
    """

    # Configure structlog processors
    structlog.configure(
        processors=[
            # Add timestamp
            add_timestamp,
            # Add agent context
            add_agent_context,
            # Add log level
            structlog.stdlib.add_log_level,
            # Add logger name
            structlog.stdlib.add_logger_name,
            # Format as JSON
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.INFO,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)  # type: ignore[no-any-return]


def log_agent_decision(
    logger: structlog.stdlib.BoundLogger,
    conversation_id: str,
    user_id: str,
    decision: str,
    execution_time: float | None = None,
    **kwargs: Any,
) -> None:
    """
    Log agent decision with structured fields.

    Args:
        logger: Structured logger instance
        conversation_id: Unique conversation identifier
        user_id: User identifier
        decision: Agent decision (for RouterAgent)
        execution_time: Time taken for execution in seconds
        **kwargs: Additional context fields
    """
    log_data = {
        "conversation_id": conversation_id,
        "user_id": user_id,
        "decision": decision,
    }

    if execution_time is not None:
        log_data["execution_time"] = f"{execution_time:.3f}s"

    log_data.update(kwargs)
    logger.info("Agent decision made", **log_data)


def log_agent_processing(
    logger: structlog.stdlib.BoundLogger,
    conversation_id: str,
    user_id: str,
    processed_content: str,
    execution_time: float | None = None,
    **kwargs: Any,
) -> None:
    """
    Log agent processing with structured fields.

    Args:
        logger: Structured logger instance
        conversation_id: Unique conversation identifier
        user_id: User identifier
        processed_content: Content processed by agent (for MathAgent/KnowledgeAgent)
        execution_time: Time taken for execution in seconds
        **kwargs: Additional context fields
    """
    log_data = {
        "conversation_id": conversation_id,
        "user_id": user_id,
        "processed_content": processed_content,
    }

    if execution_time is not None:
        log_data["execution_time"] = f"{execution_time:.3f}s"

    log_data.update(kwargs)
    logger.info("Agent processing completed", **log_data)


def log_system_event(
    logger: structlog.stdlib.BoundLogger,
    event: str,
    conversation_id: str | None = None,
    user_id: str | None = None,
    execution_time: float | None = None,
    **kwargs: Any,
) -> None:
    """
    Log system events with structured fields.

    Args:
        logger: Structured logger instance
        event: Event description
        conversation_id: Unique conversation identifier (optional)
        user_id: User identifier (optional)
        execution_time: Time taken for execution in seconds
        **kwargs: Additional context fields
    """
    log_data = {}

    if conversation_id:
        log_data["conversation_id"] = conversation_id
    if user_id:
        log_data["user_id"] = user_id
    if execution_time is not None:
        log_data["execution_time"] = f"{execution_time:.3f}s"

    log_data.update(kwargs)
    logger.info(event, **log_data)
