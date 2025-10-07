# mypy: disable-error-code=no-untyped-def

import inspect
import time
from functools import wraps
from typing import Any

from fastapi import HTTPException
from structlog.stdlib import BoundLogger

from app.core.error_handling import create_knowledge_error, create_math_error
from app.core.logging import log_agent_processing
from app.models import ChatRequest  # noqa: TC001


def log_agent(logger: BoundLogger, agent_name: str):
    """Decorator to handle timing, logging, and exceptions for agent processing.

    Works for both sync and async functions.
    """

    def decorator(func):
        is_async = inspect.iscoroutinefunction(func)

        @wraps(func)
        async def async_wrapper(context: dict[str, Any]):
            payload: ChatRequest = context["payload"]
            start_time = time.time()
            query_preview = payload.message[:100]
            final_response, workflow_step = None, None

            try:
                result = await func(context) if is_async else func(context)
                final_response, workflow_step = result
                execution_time = time.time() - start_time

                log_agent_processing(
                    agent_name=agent_name,
                    logger=logger,
                    conversation_id=payload.conversation_id,
                    user_id=payload.user_id,
                    processed_content=final_response,
                    execution_time=execution_time,
                    query_preview=query_preview,
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

            return final_response, workflow_step

        # always return an async function to support use in async workflows
        return async_wrapper

    return decorator


def raise_agent_exception(agent_name: str, error_status_code: int = 500):
    """Decorator to map exceptions to agent-specific errors."""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if agent_name == "MathAgent":
                    raise create_math_error(details=str(e)) from e
                if agent_name == "KnowledgeAgent":
                    raise create_knowledge_error(details=str(e)) from e
                raise HTTPException(status_code=error_status_code, detail=str(e)) from e

        return wrapper

    return decorator
