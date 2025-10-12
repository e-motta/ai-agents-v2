import time

from fastapi import APIRouter, Depends
from llama_index.core.base.base_query_engine import BaseQueryEngine

from app.core.error_handling import create_redis_error, create_validation_error
from app.core.logging import get_logger
from app.dependencies import (
    RedisServiceDep,
    SanitizedMessage,
    get_knowledge_engine,
    get_math_llm,
    get_router_llm,
)
from app.enums import Agents, WorkflowSignals
from app.schemas import ChatRequest, ChatResponse, ProcessingContext, RoutingContext
from app.services.chat_dispatcher import dispatch_chat_workflow
from app.services.llm_client import LLMClient

router = APIRouter()
logger = get_logger(__name__)


def _save_conversation_to_redis(
    redis_service: RedisServiceDep,
    conversation_id: str,
    user_id: str,
    message: str,
    agent_response: str,
    agent: str,
) -> None:
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
    router_llm: LLMClient = Depends(get_router_llm),
    math_llm: LLMClient = Depends(get_math_llm),
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

    routing_context = RoutingContext(
        payload=payload,
        sanitized_message=sanitized_message,
        llm_client=router_llm,
    )
    decision, step = await dispatch_chat_workflow(Agents.RouterAgent, routing_context)
    workflow_history = [step]

    processing_context = ProcessingContext(
        payload=payload,
        sanitized_message=sanitized_message,
        llm_client=math_llm,
        knowledge_engine=knowledge_engine,
    )
    agent_response, step = await dispatch_chat_workflow(decision, processing_context)
    workflow_history.append(step)

    conversion_context = routing_context.model_copy(
        update={"agent_response": agent_response, "agent_type": str(decision)}
    )
    final_response, step = await dispatch_chat_workflow(
        WorkflowSignals.ResponseConversion, conversion_context
    )
    workflow_history.append(step)

    total_execution_time = time.time() - start_time
    logger.info(
        "Chat request completed",
        conversation_id=payload.conversation_id,
        user_id=payload.user_id,
        router_decision=str(decision),
        execution_time=total_execution_time,
        response_preview=final_response[:100],
        workflow_history=[s.model_dump() for s in workflow_history],
    )

    _save_conversation_to_redis(
        redis_service,
        payload.conversation_id,
        payload.user_id,
        sanitized_message,
        agent_response,
        str(decision),
    )

    return ChatResponse(
        user_id=payload.user_id,
        conversation_id=payload.conversation_id,
        router_decision=str(decision),
        response=final_response,
        source_agent_response=agent_response,
        workflow_history=workflow_history,
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
