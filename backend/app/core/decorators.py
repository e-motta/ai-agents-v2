import time
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any

from fastapi import HTTPException
from structlog.stdlib import BoundLogger

from app.core.error_handling import create_knowledge_error, create_math_error
from app.core.logging import log_agent_processing
from app.models import ChatRequest, WorkflowStep


def log_and_handle_agent_errors(
    logger: BoundLogger, agent_name: str, error_status_code: int = 500
) -> Callable[
    [Callable[..., Awaitable[str]]],
    Callable[[dict[str, Any]], Awaitable[tuple[str, WorkflowStep]]],
]:
    """Decorator to handle timing, logging, and exceptions for agent processing."""

    def decorator(
        func: Callable[..., Awaitable[str]],
    ) -> Callable[[dict[str, Any]], Awaitable[tuple[str, WorkflowStep]]]:
        @wraps(func)
        async def wrapper(context: dict[str, Any]) -> tuple[str, WorkflowStep]:
            payload: ChatRequest = context["payload"]
            start_time = time.time()
            query_preview = payload.message[:100]
            try:
                final_response = await func(context)
                execution_time = time.time() - start_time
                log_agent_processing(
                    logger=logger,
                    conversation_id=payload.conversation_id,
                    user_id=payload.user_id,
                    processed_content=final_response,
                    execution_time=execution_time,
                    query_preview=query_preview,
                )
                return final_response, WorkflowStep(
                    agent=agent_name, action=func.__name__, result=final_response
                )
            except Exception as e:
                execution_time = time.time() - start_time
                logger.exception(
                    "%s processing failed",
                    agent_name,
                    conversation_id=payload.conversation_id,
                    user_id=payload.user_id,
                    error=str(e),
                    execution_time=execution_time,
                    query_preview=query_preview,
                )
                # Create appropriate error based on agent type
                if agent_name == "MathAgent":
                    raise create_math_error(details=str(e)) from e
                if agent_name == "KnowledgeAgent":
                    raise create_knowledge_error(details=str(e)) from e
                raise HTTPException(status_code=error_status_code, detail=str(e)) from e

        return wrapper

    return decorator
