import asyncio
from collections.abc import Awaitable, Callable
from typing import Literal, cast, overload

from app.agents.knowledge_agent import query_knowledge
from app.agents.math_agent import solve_math
from app.agents.router_agent import convert_response, route_query
from app.core.decorators import handle_agent_errors, log_process
from app.core.error_handling import create_service_unavailable_error
from app.core.logging import get_logger
from app.enums import Agents, KnowledgeAgentMessages, SystemMessages, WorkflowSignals
from app.schemas import ProcessingContext, RoutingContext, WorkflowStep
from app.security.constants import CONVERT_RESPONSE_AGENTS

logger = get_logger(__name__)


@handle_agent_errors(Agents.RouterAgent, "route_query")
@log_process(logger, Agents.RouterAgent)
async def _route_query(context: RoutingContext) -> tuple[str, WorkflowStep]:
    """Handle RouterAgent flow."""
    response = await route_query(
        context.sanitized_message,
        llm_client=context.llm_client,
        conversation_id=context.payload.conversation_id,
        user_id=context.payload.user_id,
    )
    return response, WorkflowStep(
        agent="RouterAgent", action="route_query", result=str(response)
    )


@handle_agent_errors(Agents.RouterAgent, "convert_response")
@log_process(logger, Agents.RouterAgent)
async def _convert_response(context: RoutingContext) -> tuple[str, WorkflowStep]:
    if context.agent_type in CONVERT_RESPONSE_AGENTS:
        response = await convert_response(
            original_query=context.sanitized_message,
            agent_response=context.agent_response or SystemMessages.GENERIC_ERROR,
            agent_type=context.agent_type or WorkflowSignals.Error,
            llm_client=context.llm_client,
        )
    else:
        response = context.agent_response or SystemMessages.GENERIC_ERROR
    return response, WorkflowStep(
        agent="RouterAgent", action="convert_response", result=str(response)
    )


@handle_agent_errors(Agents.MathAgent, "process_math")
@log_process(logger, Agents.MathAgent)
async def _process_math(context: ProcessingContext) -> tuple[str, WorkflowStep]:
    """Handle MathAgent flow."""
    final_response = await solve_math(context.sanitized_message, context.llm_client)
    return final_response, WorkflowStep(
        agent=Agents.MathAgent, action="process_math", result=final_response
    )


@handle_agent_errors(Agents.KnowledgeAgent, "process_knowledge")
@log_process(logger, Agents.KnowledgeAgent)
async def _process_knowledge(context: ProcessingContext) -> tuple[str, WorkflowStep]:
    """Handle KnowledgeAgent flow."""
    knowledge_engine = context.knowledge_engine
    if knowledge_engine is None:
        raise create_service_unavailable_error(
            service_name="Knowledge Base",
            details=KnowledgeAgentMessages.KNOWLEDGE_BASE_UNAVAILABLE,
        )
    final_response = await query_knowledge(context.sanitized_message, knowledge_engine)
    return final_response, WorkflowStep(
        agent=Agents.KnowledgeAgent,
        action="process_knowledge",
        result=final_response,
    )


@log_process(logger, "UnsupportedLanguage")
def _process_unsupported_language(_: ProcessingContext) -> tuple[str, WorkflowStep]:
    """Handle unsupported language decision."""
    return SystemMessages.UNSUPPORTED_LANGUAGE, WorkflowStep(
        agent="System", action="reject", result=str(WorkflowSignals.UnsupportedLanguage)
    )


@log_process(logger, "Error")
def _process_error(_: ProcessingContext) -> tuple[str, WorkflowStep]:
    """Handle generic error decision."""
    return SystemMessages.GENERIC_ERROR, WorkflowStep(
        agent="System", action="error", result=str(WorkflowSignals.Error)
    )


Context = ProcessingContext | RoutingContext

SyncChatHandler = Callable[
    [Context], tuple[str | Agents | WorkflowSignals, WorkflowStep]
]
AsyncChatHandler = Callable[
    [Context], Awaitable[tuple[str | Agents | WorkflowSignals, WorkflowStep]]
]

HANDLER_MAP: dict[Agents | WorkflowSignals, SyncChatHandler | AsyncChatHandler] = {
    Agents.RouterAgent: _route_query,
    Agents.MathAgent: _process_math,
    Agents.KnowledgeAgent: _process_knowledge,
    WorkflowSignals.UnsupportedLanguage: _process_unsupported_language,
    WorkflowSignals.Error: _process_error,
    WorkflowSignals.ResponseConversion: _convert_response,
}


@overload
async def dispatch_chat_workflow(
    signal: Literal[Agents.RouterAgent], context: RoutingContext
) -> tuple[Agents | WorkflowSignals, WorkflowStep]: ...


@overload
async def dispatch_chat_workflow(
    signal: Literal[WorkflowSignals.ResponseConversion], context: RoutingContext
) -> tuple[str | WorkflowSignals, WorkflowStep]: ...


@overload
async def dispatch_chat_workflow(
    signal: Agents | WorkflowSignals, context: ProcessingContext
) -> tuple[str, WorkflowStep]: ...


async def dispatch_chat_workflow(
    signal: Agents | WorkflowSignals, context: Context
) -> tuple[str | Agents | WorkflowSignals, WorkflowStep]:
    """Selects and executes the appropriate handler based on the message."""
    handler = HANDLER_MAP.get(signal, _process_error)

    if asyncio.iscoroutinefunction(handler):
        handler = cast("AsyncChatHandler", handler)
        response, step = await handler(context)
    else:
        handler = cast("SyncChatHandler", handler)
        response, step = handler(context)

    return response, step
