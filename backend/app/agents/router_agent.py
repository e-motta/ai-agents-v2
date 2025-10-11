"""
Router Agent module for classifying user queries between MathAgent and KnowledgeAgent.

This module provides a function to route user queries to the appropriate agent
based on the query content using an LLM classifier.
"""

from app.core.logging import get_logger, log_agent_decision
from app.enums import Agents, RouterAgentMessages, WorkflowSignals
from app.exceptions import RouterValidationError
from app.security.constants import SUSPICIOUS_PATTERNS
from app.security.prompts import ROUTER_CONVERSION_PROMPT, ROUTER_SYSTEM_PROMPT
from app.services.llm_client import LLMClient

logger = get_logger(__name__)


def _validate_response(response: str) -> Agents | WorkflowSignals:
    """
    Validate and clean the LLM response.

    Args:
        response: Raw response from the LLM

    Returns:
        Cleaned response
        ("MathAgent", "KnowledgeAgent", "UnsupportedLanguage", or "Error")
    """
    cleaned_response = response.strip().lower()

    canonical_responses: list[Agents | WorkflowSignals] = [
        Agents.MathAgent,
        Agents.KnowledgeAgent,
        WorkflowSignals.UnsupportedLanguage,
        WorkflowSignals.Error,
    ]

    for r in canonical_responses:
        if r.lower() in cleaned_response.lower():
            return r

    # Default to Error for safety
    logger.warning(
        RouterAgentMessages.ROUTING_INVALID_RESPONSE,
        response=response,
        default_action=WorkflowSignals.Error,
    )
    return WorkflowSignals.Error


def _detect_suspicious_content(query: str) -> bool:
    """
    Detect potentially suspicious or malicious content in the query.

    Args:
        query: User query to analyze

    Returns:
        True if suspicious content is detected, False otherwise
    """
    query_lower = query.lower()

    for pattern in SUSPICIOUS_PATTERNS:
        if pattern in query_lower:
            logger.warning(
                RouterAgentMessages.SECURITY_SUSPICIOUS_CONTENT,
                pattern=pattern,
                query_preview=query[:50],
            )
            return True

    return False


async def route_query(
    query: str,
    llm_client: LLMClient,
    conversation_id: str | None = None,
    user_id: str | None = None,
) -> Agents | WorkflowSignals:
    """
    Route a user query to the appropriate agent or return error status.

    Args:
        query: The user's query string
        llm_client: LLMClient instance to use for routing

    Returns:
        str: Either "MathAgent", "KnowledgeAgent", "UnsupportedLanguage", or "Error"

    Raises:
        RouterValidationError: If the query is empty
    """
    if not query or not query.strip():
        raise RouterValidationError(
            message=RouterAgentMessages.QUERY_CANNOT_BE_EMPTY, query=query
        )

    # Clean the query
    cleaned_query = query.strip()

    # Check for suspicious content
    if _detect_suspicious_content(cleaned_query):
        logger.warning(
            RouterAgentMessages.SECURITY_SUSPICIOUS_RETURN_KNOWLEDGE,
            conversation_id=conversation_id,
            user_id=user_id,
            query_preview=cleaned_query[:100],
        )
        return Agents.KnowledgeAgent

    try:
        logger.info(
            RouterAgentMessages.ROUTING_QUERY,
            conversation_id=conversation_id,
            user_id=user_id,
            query_preview=cleaned_query[:100],
        )

        content = await llm_client.ask(
            message=cleaned_query,
            system_prompt=ROUTER_SYSTEM_PROMPT,
        )

        log_agent_decision(
            logger=logger,
            conversation_id=conversation_id or "unknown",
            user_id=user_id or "unknown",
            decision=content,
            query_preview=cleaned_query[:100],
        )

        return _validate_response(content)

    except Exception as e:
        logger.exception(
            RouterAgentMessages.ROUTING_ERROR,
            conversation_id=conversation_id,
            user_id=user_id,
            error=str(e),
            query_preview=cleaned_query[:100],
        )
        # Default to Error for safety
        return WorkflowSignals.Error


async def convert_response(
    original_query: str,
    agent_response: str,
    agent_type: str,
    llm_client: LLMClient,
) -> str:
    """
    Convert raw agent response into conversational format
    using the Router Agent's conversion system.

    Args:
        original_query: The user's original query
        agent_response: The raw response from the specialized agent
        agent_type: The type of agent that generated
            the response (MathAgent or KnowledgeAgent)
        llm_client: LLMClient instance to use for conversion

    Returns:
        str: The converted conversational response

    Raises:
        RouterConversionError: If the conversion fails
        (though this is handled gracefully)
    """
    logger.info(
        RouterAgentMessages.CONVERSION_STARTING,
        agent_type=agent_type,
        response_preview=agent_response[:100],
        query_preview=original_query[:100],
    )

    try:
        message = f"""Original Query: "{original_query}"
Agent Type: {agent_type}
Agent Response: "{agent_response}"

Please convert this agent response into a conversational format
 while preserving all factual accuracy."""

        content = await llm_client.ask(
            message=message,
            system_prompt=ROUTER_CONVERSION_PROMPT,
        )

        if not content:
            logger.warning(
                RouterAgentMessages.CONVERSION_FAILED_NO_RESULT,
                agent_type=agent_type,
            )
            return agent_response

        logger.info(
            RouterAgentMessages.CONVERSION_COMPLETED,
            agent_type=agent_type,
            original_response_preview=agent_response[:100],
            converted_response_preview=content[:100],
        )

        return content

    except Exception as e:
        logger.exception(
            RouterAgentMessages.CONVERSION_ERROR,
            agent_type=agent_type,
            error=str(e),
        )
        # Fallback to original response if conversion fails
        return agent_response
