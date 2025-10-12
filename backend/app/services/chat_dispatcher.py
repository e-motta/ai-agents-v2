import asyncio
from collections.abc import Awaitable, Callable
from typing import cast

from app.agents.knowledge_agent import query_knowledge
from app.agents.math_agent import solve_math
from app.core.decorators import handle_agent_errors, log_process
from app.core.error_handling import create_service_unavailable_error
from app.core.logging import get_logger
from app.enums import Agents, KnowledgeAgentMessages, SystemMessages, WorkflowSignals
from app.models import ChatContext, WorkflowStep

logger = get_logger(__name__)


@handle_agent_errors(Agents.MathAgent, "process_math")
@log_process(logger, Agents.MathAgent)
async def _process_math(context: ChatContext) -> tuple[str, WorkflowStep]:
    """Handle MathAgent flow."""
    final_response = await solve_math(context.sanitized_message, context.llm_client)
    return final_response, WorkflowStep(
        agent=Agents.MathAgent, action="process_math", result=final_response
    )


@handle_agent_errors(Agents.KnowledgeAgent, "process_knowledge")
@log_process(logger, Agents.KnowledgeAgent)
async def _process_knowledge(context: ChatContext) -> tuple[str, WorkflowStep]:
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
def _process_unsupported_language(_: ChatContext) -> tuple[str, WorkflowStep]:
    """Handle unsupported language decision."""
    return SystemMessages.UNSUPPORTED_LANGUAGE, WorkflowStep(
        agent="System", action="reject", result=str(WorkflowSignals.UnsupportedLanguage)
    )


@log_process(logger, "Error")
def _process_error(_: ChatContext) -> tuple[str, WorkflowStep]:
    """Handle generic error decision."""
    return SystemMessages.GENERIC_ERROR, WorkflowStep(
        agent="System", action="error", result=str(WorkflowSignals.Error)
    )


SyncChatHandler = Callable[[ChatContext], tuple[str, WorkflowStep]]
AsyncChatHandler = Callable[[ChatContext], Awaitable[tuple[str, WorkflowStep]]]

HANDLER_MAP: dict[Agents | WorkflowSignals, SyncChatHandler | AsyncChatHandler] = {
    Agents.MathAgent: _process_math,
    Agents.KnowledgeAgent: _process_knowledge,
    WorkflowSignals.UnsupportedLanguage: _process_unsupported_language,
    WorkflowSignals.Error: _process_error,
}


async def dispatch_agent_request(
    decision: Agents | WorkflowSignals, context: ChatContext
) -> tuple[str, WorkflowStep]:
    """
    Selects and executes the appropriate agent handler based on the routing decision.
    """
    handler = HANDLER_MAP.get(decision, _process_error)

    if asyncio.iscoroutinefunction(handler):
        handler = cast("AsyncChatHandler", handler)
        source_agent_response, step = await handler(context)
    else:
        handler = cast("SyncChatHandler", handler)
        source_agent_response, step = handler(context)

    return source_agent_response, step
