"""
Math Agent module for solving mathematical expressions using LangChain.
"""

import math
import time

from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.core.logging import get_logger
from app.enums import ErrorMessage
from app.security.prompts import MATH_AGENT_SYSTEM_PROMPT

logger = get_logger(__name__)


def _validate_math_response(result: str, query: str, execution_time: float):
    if not result or result.lower() == "error":
        logger.error(
            "Math evaluation failed - no result",
            query=query,
            execution_time=execution_time,
        )
        error_msg = f"{ErrorMessage.MATH_EVALUATION_FAILED}: {query}"
        raise ValueError(error_msg)

    try:
        result_float = float(result)
        MAX_RESULT_VALUE = 1e10  # Prevent overflow
        if abs(result_float) > MAX_RESULT_VALUE:
            error_msg = f"Result too large: {result}"
            raise ValueError(error_msg)  # noqa: TRY301
        if math.isnan(result_float):
            error_msg = f"Invalid result: {result}"
            raise ValueError(error_msg)  # noqa: TRY301
    except ValueError:
        logger.exception(
            "Math evaluation failed - non-numerical or invalid result",
            query=query,
            result=result,
            execution_time=execution_time,
        )
        error_msg = f"{ErrorMessage.MATH_NON_NUMERICAL_RESULT}: '{result}'"
        raise ValueError(error_msg) from None


async def solve_math(query: str, llm: ChatOpenAI) -> str:
    """
    Solve a mathematical expression using an LLM calculator.

    Args:
        query (str): The mathematical expression to evaluate
        llm (ChatOpenAI): The LLM instance to use for calculations

    Returns:
        str: The numerical result as a string

    Raises:
        ValueError: If the query cannot be evaluated
    """
    start_time = time.time()

    logger.info("Starting math evaluation", query=query, query_preview=query[:50])

    # Create messages
    messages = [
        SystemMessage(content=MATH_AGENT_SYSTEM_PROMPT),
        HumanMessage(content=f"Evaluate this mathematical expression: {query}"),
    ]

    try:
        # Get response from LLM asynchronously
        response = await llm.ainvoke(messages)
        # Handle different response formats
        if isinstance(response.content, list):
            result = " ".join(str(item) for item in response.content).strip()
        else:
            result = response.content.strip()

        execution_time = time.time() - start_time

        _validate_math_response(result, query, execution_time)

        logger.info(
            "Math evaluation completed",
            query=query,
            result=result,
            execution_time=execution_time,
        )

        return result
    except Exception as e:
        execution_time = time.time() - start_time
        logger.exception(
            "Math evaluation error",
            query=query,
            error=str(e),
            execution_time=execution_time,
        )
        error_msg = f"{ErrorMessage.MATH_EVALUATION_FAILED} '{query}': {e!s}"
        raise ValueError(error_msg) from e
