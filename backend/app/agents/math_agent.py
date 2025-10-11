"""
Math Agent module for solving mathematical expressions using LangChain.
"""

import math
import re

from app.core.logging import get_logger
from app.enums import MathAgentMessages
from app.security.prompts import MATH_AGENT_SYSTEM_PROMPT
from app.services.llm_client import LLMClient

logger = get_logger(__name__)

MAX_RESULT_VALUE = 1e10


def _clean_and_convert_to_float(result_text: str) -> float:
    """
    Clean a string to keep only numeric characters and convert it to a float.

    Raises:
        ValueError: If the cleaned string cannot be converted to a valid float.
    """
    if not result_text or result_text.lower() == "error":
        raise ValueError(MathAgentMessages.MATH_VALIDATION_ERROR)

    cleaned_text = re.sub(r"[^0-9.\-]", "", result_text)
    if cleaned_text in {"", "-", "."}:
        raise ValueError(
            MathAgentMessages.MATH_VALIDATION_NO_NUMERIC_DATA.format(
                result_text=result_text
            )
        )

    return float(cleaned_text)


def _validate_numeric_result(value: float) -> None:
    """
    Validate that a numeric result is within acceptable boundaries
    (e.g., not NaN or too large).

    Raises:
        ValueError: If the number is outside the defined limits.
    """
    if math.isnan(value):
        raise ValueError(MathAgentMessages.MATH_VALIDATION_NAN)

    if abs(value) > MAX_RESULT_VALUE:
        raise ValueError(
            MathAgentMessages.MATH_VALIDATION_EXCEEDS_LIMIT.format(
                value=value, max_result_value=MAX_RESULT_VALUE
            )
        )


async def solve_math(query: str, llm_client: LLMClient) -> str:
    """
    Solve a mathematical expression using an LLM-based calculator.

    Args:
        query: The mathematical expression to evaluate.
        llm_client: LLMClient instance to use for calculations.

    Returns:
        The numerical result as a string, as returned by the LLM.

    Raises:
        ValueError: If the query fails to evaluate or the result is invalid.
    """
    logger.info(MathAgentMessages.MATH_EVALUATION_STARTING, query=query)

    try:
        raw_result = await llm_client.ask(
            message=f"Evaluate this mathematical expression: {query}",
            system_prompt=MATH_AGENT_SYSTEM_PROMPT,
        )

        numeric_value = _clean_and_convert_to_float(raw_result)
        _validate_numeric_result(numeric_value)

        logger.info(
            MathAgentMessages.MATH_EVALUATION_COMPLETED, query=query, result=raw_result
        )
        return raw_result

    except ValueError as e:
        logger.exception(
            MathAgentMessages.MATH_VALIDATION_FAILED,
            query=query,
            error=str(e),
        )
        raise ValueError(MathAgentMessages.MATH_VALIDATION_FAILED) from e

    except Exception as e:
        logger.exception(MathAgentMessages.MATH_EVALUATION_FAILED, query=query)
        raise ValueError(MathAgentMessages.MATH_EVALUATION_FAILED) from e
