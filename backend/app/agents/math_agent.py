"""
Math Agent module for solving mathematical expressions using LangChain.
"""

import math
import re

from app.core.logging import get_logger
from app.enums import MathAgentMessages
from app.exceptions import (
    MathConversionError,
    MathEvaluationError,
    MathResultError,
    MathValidationError,
)
from app.security.prompts import MATH_AGENT_SYSTEM_PROMPT
from app.services.llm_client import LLMClient

logger = get_logger(__name__)

MAX_RESULT_VALUE = 1e10


def _clean_and_convert_to_float(result_text: str) -> float:
    """
    Clean a string to keep only numeric characters and convert it to a float.

    Raises:
        MathValidationError: If the result text is empty or explicitly an error.
        MathConversionError: If the cleaned string cannot be converted to a valid float.
    """
    if not result_text or result_text.lower() == "error":
        raise MathValidationError(
            message=MathAgentMessages.MATH_VALIDATION_ERROR, result_text=result_text
        )

    # Handle special cases like NaN, inf, -inf before cleaning
    result_lower = result_text.lower().strip()
    if result_lower in {"nan", "inf", "-inf", "infinity", "-infinity"}:
        try:
            return float(result_lower)
        except ValueError:
            pass

    cleaned_text = re.sub(r"[^0-9.\-eE]", "", result_text)
    if cleaned_text in {"", "-", "."}:
        raise MathConversionError(
            message=MathAgentMessages.MATH_VALIDATION_NO_NUMERIC_DATA.format(
                result_text=result_text
            ),
            input_text=result_text,
        )

    try:
        return float(cleaned_text)
    except ValueError as e:
        raise MathConversionError(
            message=MathAgentMessages.MATH_CONVERSION_FAILED.format(
                cleaned_text=cleaned_text, error=str(e)
            ),
            input_text=result_text,
            details={"cleaned_text": cleaned_text, "original_error": str(e)},
        ) from e


def _validate_numeric_result(value: float) -> None:
    """
    Validate that a numeric result is within acceptable boundaries
    (e.g., not NaN or too large).

    Raises:
        MathResultError: If the number is outside the defined limits.
    """
    if math.isnan(value):
        raise MathResultError(
            message=MathAgentMessages.MATH_VALIDATION_NAN, value=value
        )

    if abs(value) > MAX_RESULT_VALUE:
        raise MathResultError(
            message=MathAgentMessages.MATH_VALIDATION_EXCEEDS_LIMIT.format(
                value=value, max_result_value=MAX_RESULT_VALUE
            ),
            value=value,
            max_value=MAX_RESULT_VALUE,
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
        MathValidationError: If the result text validation fails.
        MathConversionError: If the result cannot be converted to a number.
        MathResultError: If the numeric result is invalid or out of bounds.
        MathEvaluationError: If the evaluation process fails unexpectedly.
    """
    logger.info(MathAgentMessages.MATH_EVALUATION_STARTING, query=query)

    try:
        raw_result = await llm_client.ask(
            message=MathAgentMessages.MATH_LLM_QUERY.format(query=query),
            system_prompt=MATH_AGENT_SYSTEM_PROMPT,
        )

        numeric_value = _clean_and_convert_to_float(raw_result)
        _validate_numeric_result(numeric_value)

        logger.info(
            MathAgentMessages.MATH_EVALUATION_COMPLETED, query=query, result=raw_result
        )
        return raw_result

    except (MathValidationError, MathConversionError, MathResultError) as e:
        logger.exception(
            MathAgentMessages.MATH_VALIDATION_FAILED, query=query, error=str(e)
        )
        raise

    except Exception as e:
        logger.exception(
            MathAgentMessages.MATH_EVALUATION_FAILED, query=query, error=str(e)
        )
        raise MathEvaluationError(
            message=MathAgentMessages.MATH_EVALUATION_FAILED,
            query=query,
            details={"original_error": str(e), "error_type": type(e).__name__},
        ) from e
