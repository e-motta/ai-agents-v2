"""
Router Agent module for classifying user queries between MathAgent and KnowledgeAgent.

This module provides a function to route user queries to the appropriate agent
based on the query content using an LLM classifier.
"""

from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.core.logging import get_logger, log_agent_decision
from app.enums import ResponseEnum
from app.security.prompts import ROUTER_CONVERSION_PROMPT, ROUTER_SYSTEM_PROMPT

logger = get_logger(__name__)


def _validate_response(response: str) -> str:
    """
    Validate and clean the LLM response.

    Args:
        response: Raw response from the LLM

    Returns:
        Cleaned response
        ("MathAgent", "KnowledgeAgent", "UnsupportedLanguage", or "Error")
    """
    # Clean the response
    cleaned_response = response.strip().lower()

    # Mapping of keywords to canonical agent names
    response_map = {
        "mathagent": ResponseEnum.MathAgent,
        "knowledgeagent": ResponseEnum.KnowledgeAgent,
        "unsupportedlanguage": ResponseEnum.UnsupportedLanguage,
        "error": ResponseEnum.Error,
    }

    for key, value in response_map.items():
        if key in cleaned_response:
            return value

    # Default to Error for safety
    logger.warning(
        "Invalid response from router", response=response, default_action="Error"
    )
    return ResponseEnum.Error


def _detect_suspicious_content(query: str) -> bool:
    """
    Detect potentially suspicious or malicious content in the query.

    Args:
        query: User query to analyze

    Returns:
        True if suspicious content is detected, False otherwise
    """
    suspicious_patterns = [
        "ignore previous instructions",
        "forget everything",
        "system prompt",
        "you are now",
        "act as",
        "pretend to be",
        "roleplay",
        "jailbreak",
        "developer mode",
        "admin mode",
        "override",
        "bypass",
        "exploit",
        "hack",
        "inject",
        "execute",
        "run command",
        "system call",
        "file://",
        "http://",
        "https://",
        "<script>",
        "javascript:",
        "data:",
        "eval(",
        "exec(",
        "import os",
        "subprocess",
        "shell",
        "terminal",
        "command line",
        "prompt injection",
        "llm injection",
        # Portuguese patterns
        "ignore as instruções anteriores",
        "esqueça tudo",
        "prompt do sistema",
        "você agora é",
        "aja como",
        "finja ser",
        "interprete o papel de",
    ]

    query_lower = query.lower()

    for pattern in suspicious_patterns:
        if pattern in query_lower:
            logger.warning(
                "Suspicious content detected in query",
                pattern=pattern,
                query_preview=query[:50],
            )
            return True

    return False


async def route_query(
    query: str,
    llm: ChatOpenAI,
    conversation_id: str | None = None,
    user_id: str | None = None,
) -> str:
    """
    Route a user query to the appropriate agent or return error status.

    Args:
        query: The user's query string
        llm: ChatOpenAI LLM instance to use for routing

    Returns:
        str: Either "MathAgent", "KnowledgeAgent", "UnsupportedLanguage", or "Error"

    Raises:
        ValueError: If the query is empty or if there's an error processing the query
    """
    if not query or not query.strip():
        error_msg = "Query cannot be empty"
        raise ValueError(error_msg)

    # Clean the query
    cleaned_query = query.strip()

    # Check for suspicious content
    if _detect_suspicious_content(cleaned_query):
        logger.warning(
            "Suspicious content detected, returning KnowledgeAgent for safety",
            conversation_id=conversation_id,
            user_id=user_id,
            query_preview=cleaned_query[:100],
        )
        return ResponseEnum.KnowledgeAgent

    try:
        logger.info(
            "Routing query",
            conversation_id=conversation_id,
            user_id=user_id,
            query_preview=cleaned_query[:100],
        )

        # Create messages
        messages = [
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=f'Query: "{cleaned_query}"'),
        ]

        # Get response from LLM asynchronously
        response = await llm.ainvoke(messages)

        # Handle different response formats
        if isinstance(response.content, list):
            response_text = " ".join(str(item) for item in response.content).strip()
        else:
            response_text = response.content.strip()

        # Log the decision with structured fields
        log_agent_decision(
            logger=logger,
            conversation_id=conversation_id or "unknown",
            user_id=user_id or "unknown",
            decision=response_text,
            query_preview=cleaned_query[:100],
        )

        # Validate and return the cleaned response
        return _validate_response(response_text)

    except Exception as e:
        logger.exception(
            "Error routing query",
            conversation_id=conversation_id,
            user_id=user_id,
            error=str(e),
            query_preview=cleaned_query[:100],
        )
        # Default to Error for safety
        return ResponseEnum.Error


async def convert_response(
    original_query: str,
    agent_response: str,
    agent_type: str,
    llm: ChatOpenAI,
) -> str:
    """
    Convert raw agent response into conversational format
    using the Router Agent's conversion system.

    Args:
        original_query: The user's original query
        agent_response: The raw response from the specialized agent
        agent_type: The type of agent that generated
            the response (MathAgent or KnowledgeAgent)
        llm: ChatOpenAI LLM instance to use for conversion

    Returns:
        str: The converted conversational response

    Raises:
        ValueError: If the conversion fails
    """
    logger.info(
        "Starting response conversion",
        agent_type=agent_type,
        response_preview=agent_response[:100],
        query_preview=original_query[:100],
    )

    try:
        # Create messages for conversion
        messages = [
            SystemMessage(content=ROUTER_CONVERSION_PROMPT),
            HumanMessage(
                content=f"""Original Query: "{original_query}"
Agent Type: {agent_type}
Agent Response: "{agent_response}"

Please convert this agent response into a conversational format
 while preserving all factual accuracy."""
            ),
        ]

        # Get response from LLM asynchronously
        response = await llm.ainvoke(messages)

        # Handle different response formats
        if isinstance(response.content, list):
            converted_response = " ".join(
                str(item) for item in response.content
            ).strip()
        else:
            converted_response = response.content.strip()

        # Validate that we got a reasonable response
        if not converted_response:
            logger.error(
                "Response conversion failed - no result",
                agent_type=agent_type,
            )
            # Fallback to original response if conversion fails
            return agent_response

        logger.info(
            "Response conversion completed",
            agent_type=agent_type,
            original_response_preview=agent_response[:100],
            converted_response_preview=converted_response[:100],
        )

        return converted_response

    except Exception as e:
        logger.exception(
            "Response conversion error",
            agent_type=agent_type,
            error=str(e),
        )
        # Fallback to original response if conversion fails
        return agent_response
