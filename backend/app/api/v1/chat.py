import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import APIRouter, Depends
from langchain_openai.chat_models.base import ChatOpenAI
from llama_index.core.base.base_query_engine import BaseQueryEngine

from app.agents.knowledge_agent import query_knowledge
from app.agents.math_agent import solve_math
from app.agents.router_agent import convert_response, route_query
from app.core.decorators import log_and_handle_agent_errors
from app.core.error_handling import (
    create_redis_error,
    create_service_unavailable_error,
    create_validation_error,
)
from app.core.logging import get_logger, log_system_event
from app.dependencies import (
    RedisServiceDep,
    SanitizedMessage,
    get_knowledge_engine,
    get_math_llm,
    get_router_llm,
)
from app.enums import ErrorMessage, ResponseEnum
from app.models import ChatRequest, ChatResponse, WorkflowStep

router = APIRouter()
logger = get_logger(__name__)


@log_and_handle_agent_errors(logger, agent_name="MathAgent", error_status_code=400)
async def _process_math(context: dict[str, Any]) -> str:
    """Handle MathAgent flow."""
    return await solve_math(context["sanitized_message"], context["math_llm"])


@log_and_handle_agent_errors(logger, agent_name="KnowledgeAgent", error_status_code=400)
async def _process_knowledge(context: dict[str, Any]) -> str:
    """Handle KnowledgeAgent flow."""
    knowledge_engine = context["knowledge_engine"]
    if knowledge_engine is None:
        raise create_service_unavailable_error(
            service_name="Knowledge Base",
            details=ErrorMessage.KNOWLEDGE_BASE_UNAVAILABLE,
        )
    return await query_knowledge(context["sanitized_message"], knowledge_engine)


def _process_unsupported_language(context: dict[str, Any]) -> tuple[str, WorkflowStep]:
    """Handle unsupported language decision."""
    payload = context["payload"]
    final_response = ErrorMessage.UNSUPPORTED_LANGUAGE
    log_system_event(
        logger=logger,
        event="unsupported_language_rejected",
        conversation_id=payload.conversation_id,
        user_id=payload.user_id,
    )
    return final_response, WorkflowStep(
        agent="System", action="reject", result=str(ResponseEnum.UnsupportedLanguage)
    )


def _process_error(context: dict[str, Any]) -> tuple[str, WorkflowStep]:
    """Handle generic error decision."""
    payload = context["payload"]
    final_response = ErrorMessage.GENERIC_ERROR
    log_system_event(
        logger=logger,
        event="error_processing_request",
        conversation_id=payload.conversation_id,
        user_id=payload.user_id,
    )
    return final_response, WorkflowStep(
        agent="System", action="error", result=str(ResponseEnum.Error)
    )


HANDLER_BY_DECISION: dict[
    ResponseEnum,
    Callable[
        [dict[str, Any]], Awaitable[tuple[str, WorkflowStep]] | tuple[str, WorkflowStep]
    ],
] = {
    ResponseEnum.MathAgent: _process_math,
    ResponseEnum.KnowledgeAgent: _process_knowledge,
    ResponseEnum.UnsupportedLanguage: _process_unsupported_language,
    ResponseEnum.Error: _process_error,
}


def _save_conversation_to_redis(
    redis_service: RedisServiceDep,
    conversation_id: str,
    user_id: str,
    message: str,
    agent_response: str,
    agent: str,
):
    if redis_service is None:
        logger.warning(
            "Redis service unavailable, conversation not saved",
            conversation_id=conversation_id,
            user_id=user_id,
        )
        return

    try:
        redis_success = redis_service.add_message_to_history(
            conversation_id=conversation_id,
            user_message=message,
            agent_response=agent_response,
            user_id=user_id,
            agent=agent,
        )
        if redis_success:
            logger.info(
                "Conversation saved to Redis",
                conversation_id=conversation_id,
                user_id=user_id,
            )
        else:
            logger.warning(
                "Failed to save conversation to Redis",
                conversation_id=conversation_id,
                user_id=user_id,
            )
    except Exception as e:
        logger.exception(
            "Error saving conversation to Redis",
            conversation_id=conversation_id,
            user_id=user_id,
            error=str(e),
        )


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    sanitized_message: SanitizedMessage,
    redis_service: RedisServiceDep,
    router_llm: ChatOpenAI = Depends(get_router_llm),
    math_llm: ChatOpenAI = Depends(get_math_llm),
    knowledge_engine: BaseQueryEngine | None = Depends(get_knowledge_engine),
) -> ChatResponse:
    start_time = time.time()

    if not sanitized_message or not sanitized_message.strip():
        raise create_validation_error(details="'message' cannot be empty")

    logger.info(
        "Chat request received",
        conversation_id=payload.conversation_id,
        user_id=payload.user_id,
        message_preview=sanitized_message[:100],
    )

    try:
        decision = await route_query(
            sanitized_message,
            llm=router_llm,
            conversation_id=payload.conversation_id,
            user_id=payload.user_id,
        )
    except Exception as e:
        logger.exception(
            "Router agent failed",
            conversation_id=payload.conversation_id,
            user_id=payload.user_id,
            error=str(e),
        )
        decision = ResponseEnum.Error

    agent_workflow: list[WorkflowStep] = [
        WorkflowStep(agent="RouterAgent", action="route_query", result=str(decision))
    ]

    context = {
        "payload": payload,
        "sanitized_message": sanitized_message,
        "math_llm": math_llm,
        "knowledge_engine": knowledge_engine,
    }
    handler = HANDLER_BY_DECISION.get(decision, _process_error)  # type: ignore[assignment]

    if asyncio.iscoroutinefunction(handler):
        source_agent_response, step = await handler(context)
    else:
        source_agent_response, step = handler(context)

    agent_workflow.append(step)

    try:
        if decision in [ResponseEnum.MathAgent]:
            final_response = await convert_response(
                original_query=sanitized_message,
                agent_response=source_agent_response,
                agent_type=str(decision),
                llm=router_llm,
            )
        else:
            final_response = source_agent_response
    except Exception as e:
        logger.exception(
            "Response conversion failed, using original response",
            conversation_id=payload.conversation_id,
            user_id=payload.user_id,
            error=str(e),
        )
        final_response = source_agent_response

    total_execution_time = time.time() - start_time
    logger.info(
        "Chat request completed",
        conversation_id=payload.conversation_id,
        user_id=payload.user_id,
        router_decision=str(decision),
        execution_time=total_execution_time,
        response_preview=final_response[:100],
    )

    _save_conversation_to_redis(
        redis_service,
        payload.conversation_id,
        payload.user_id,
        sanitized_message,
        source_agent_response,
        str(decision),
    )

    return ChatResponse(
        user_id=payload.user_id,
        conversation_id=payload.conversation_id,
        router_decision=str(decision),
        response=final_response,
        source_agent_response=source_agent_response,
        agent_workflow=agent_workflow,
    )


@router.get("/chat/history/{conversation_id}")
async def get_conversation_history(
    conversation_id: str,
    redis_service: RedisServiceDep,
) -> dict:
    """
    Retrieve conversation history for a given conversation ID.

    Args:
        conversation_id: The unique identifier for the conversation

    Returns:
        dict: Contains the conversation history or error information
    """
    logger.info(
        "Conversation history requested",
        conversation_id=conversation_id,
    )

    if redis_service is None:
        logger.warning(
            "Redis service unavailable, returning empty history",
            conversation_id=conversation_id,
        )
        return {
            "conversation_id": conversation_id,
            "message_count": 0,
            "history": [],
        }

    try:
        history = redis_service.get_history(conversation_id)

        logger.info(
            "Conversation history retrieved",
            conversation_id=conversation_id,
            message_count=len(history),
        )

        return {
            "conversation_id": conversation_id,
            "message_count": len(history),
            "history": history,
        }

    except Exception as e:
        logger.exception(
            "Error retrieving conversation history",
            conversation_id=conversation_id,
            error=str(e),
        )
        raise create_redis_error(
            details=f"Failed to retrieve conversation history: {e!s}"
        ) from e


@router.get("/chat/user/{user_id}/conversations")
async def get_user_conversations_endpoint(
    user_id: str,
    redis_service: RedisServiceDep,
) -> dict:
    """
    Retrieve all conversation IDs for a specific user.

    Args:
        user_id: The unique identifier for the user

    Returns:
        dict: Contains the list of conversation IDs or error information
    """
    logger.info(
        "User conversations requested",
        user_id=user_id,
    )

    if redis_service is None:
        logger.warning(
            "Redis service unavailable, returning empty conversation list",
            user_id=user_id,
        )
        return {
            "user_id": user_id,
            "conversation_count": 0,
            "conversation_ids": [],
        }

    try:
        conversation_ids = redis_service.get_user_conversations(user_id)

        logger.info(
            "User conversations retrieved",
            user_id=user_id,
            conversation_count=len(conversation_ids),
        )

        return {
            "user_id": user_id,
            "conversation_count": len(conversation_ids),
            "conversation_ids": conversation_ids,
        }

    except Exception as e:
        logger.exception(
            "Error retrieving user conversations",
            user_id=user_id,
            error=str(e),
        )
        raise create_redis_error(
            details=f"Failed to retrieve user conversations: {e!s}"
        ) from e
