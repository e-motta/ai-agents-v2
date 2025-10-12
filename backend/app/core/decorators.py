# mypy: disable-error-code=no-untyped-def

import inspect
import time
from functools import wraps

from structlog.stdlib import BoundLogger

from app.core.logging import log_agent_processing
from app.enums import SystemMessages
from app.schemas import GenericContext, WorkflowStep
from app.security.constants import GRACEFUL_AGENT_EXCEPTIONS


def log_process(logger: BoundLogger, agent_name: str):
    """Decorator to handle logging for agent processing.

    Works for both sync and async functions.
    """

    def decorator(func):
        is_async = inspect.iscoroutinefunction(func)

        @wraps(func)
        async def async_wrapper(context: GenericContext):
            start_time = time.time()
            execution_time: float | None = None

            payload = context.payload
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
                    "%s Processing failed",
                    agent_name,
                    conversation_id=payload.conversation_id,
                    user_id=payload.user_id,
                    error=str(e),
                    execution_time=execution_time,
                    query_preview=query_preview,
                )
                raise

            return final_response, workflow_step

        # always return an async function to support use in async workflows
        return async_wrapper

    return decorator


def handle_agent_errors(
    agent_name: str,
    catch: tuple[type[Exception], ...] = GRACEFUL_AGENT_EXCEPTIONS,
):
    """Decorator to catch a specific tuple of exceptions and map them
    to a standardized error response."""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except catch:
                final_response = SystemMessages.GENERIC_ERROR
                workflow_step = WorkflowStep(
                    agent=agent_name,
                    action=func.__name__,
                    result=final_response,
                )
                return final_response, workflow_step

        return wrapper

    return decorator
