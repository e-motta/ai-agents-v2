"""
Unit tests for Math Agent simple expressions.

These tests verify that the math agent correctly evaluates mathematical
expressions without making external LLM calls.
"""

from unittest.mock import AsyncMock

import pytest

from app.agents.math_agent import solve_math
from app.enums import MathAgentMessages


class TestSolveMath:
    """Test the solve_math function."""

    @pytest.mark.asyncio
    async def test_solve_simple_addition(self, mock_llm):
        """Test solving simple addition."""
        # Mock LLM response
        mock_response = AsyncMock()
        mock_response.content = "4"
        mock_llm.ainvoke.return_value = mock_response

        result = await solve_math("2 + 2", mock_llm)
        assert result == "4"
        mock_llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_solve_simple_subtraction(self, mock_llm):
        """Test solving simple subtraction."""
        # Mock LLM response
        mock_response = AsyncMock()
        mock_response.content = "3"
        mock_llm.ainvoke.return_value = mock_response

        result = await solve_math("5 - 2", mock_llm)
        assert result == "3"
        mock_llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_solve_simple_multiplication(self, mock_llm):
        """Test solving simple multiplication."""
        # Mock LLM response
        mock_response = AsyncMock()
        mock_response.content = "10"
        mock_llm.ainvoke.return_value = mock_response

        result = await solve_math("2 * 5", mock_llm)
        assert result == "10"
        mock_llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_solve_simple_division(self, mock_llm):
        """Test solving simple division."""
        # Mock LLM response
        mock_response = AsyncMock()
        mock_response.content = "3"
        mock_llm.ainvoke.return_value = mock_response

        result = await solve_math("6 / 2", mock_llm)
        assert result == "3"
        mock_llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_solve_complex_expression(self, mock_llm):
        """Test solving complex mathematical expressions."""
        # Mock LLM response
        mock_response = AsyncMock()
        mock_response.content = "14"
        mock_llm.ainvoke.return_value = mock_response

        result = await solve_math("(2 + 3) * 4 - 6", mock_llm)
        assert result == "14"
        mock_llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_solve_decimal_expression(self, mock_llm):
        """Test solving expressions with decimals."""
        # Mock LLM response
        mock_response = AsyncMock()
        mock_response.content = "2.5"
        mock_llm.ainvoke.return_value = mock_response

        result = await solve_math("1.5 + 1.0", mock_llm)
        assert result == "2.5"
        mock_llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_solve_power_expression(self, mock_llm):
        """Test solving power expressions."""
        # Mock LLM response
        mock_response = AsyncMock()
        mock_response.content = "8"
        mock_llm.ainvoke.return_value = mock_response

        result = await solve_math("2^3", mock_llm)
        assert result == "8"
        mock_llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_solve_square_root(self, mock_llm):
        """Test solving square root expressions."""
        # Mock LLM response
        mock_response = AsyncMock()
        mock_response.content = "4"
        mock_llm.ainvoke.return_value = mock_response

        result = await solve_math("sqrt(16)", mock_llm)
        assert result == "4"
        mock_llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_solve_trigonometric_function(self, mock_llm):
        """Test solving trigonometric functions."""
        # Mock LLM response
        mock_response = AsyncMock()
        mock_response.content = "1"
        mock_llm.ainvoke.return_value = mock_response

        result = await solve_math("sin(pi/2)", mock_llm)
        assert result == "1"
        mock_llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_solve_negative_result(self, mock_llm):
        """Test solving expressions that result in negative numbers."""
        # Mock LLM response
        mock_response = AsyncMock()
        mock_response.content = "-3"
        mock_llm.ainvoke.return_value = mock_response

        result = await solve_math("2 - 5", mock_llm)
        assert result == "-3"
        mock_llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_solve_zero_result(self, mock_llm):
        """Test solving expressions that result in zero."""
        # Mock LLM response
        mock_response = AsyncMock()
        mock_response.content = "0"
        mock_llm.ainvoke.return_value = mock_response

        result = await solve_math("5 - 5", mock_llm)
        assert result == "0"
        mock_llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_solve_large_number(self, mock_llm):
        """Test solving expressions with large numbers."""
        # Mock LLM response
        mock_response = AsyncMock()
        mock_response.content = "1000000"
        mock_llm.ainvoke.return_value = mock_response

        result = await solve_math("1000 * 1000", mock_llm)
        assert result == "1000000"
        mock_llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_solve_list_content_response(self, mock_llm):
        """Test handling of list content in LLM response."""
        # Mock LLM response with list content
        mock_response = AsyncMock()
        mock_response.content = ["4"]
        mock_llm.ainvoke.return_value = mock_response

        result = await solve_math("2 + 2", mock_llm)
        assert result == "4"
        mock_llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_solve_empty_response_raises_error(self, mock_llm):
        """Test that empty LLM response raises ValueError."""
        # Mock LLM response with empty content
        mock_response = AsyncMock()
        mock_response.content = ""
        mock_llm.ainvoke.return_value = mock_response

        with pytest.raises(ValueError, match=MathAgentMessages.MATH_VALIDATION_FAILED):
            await solve_math("2 + 2", mock_llm)

    @pytest.mark.asyncio
    async def test_solve_error_response_raises_error(self, mock_llm):
        """Test that 'Error' response raises ValueError."""
        # Mock LLM response with error
        mock_response = AsyncMock()
        mock_response.content = "Error"
        mock_llm.ainvoke.return_value = mock_response

        with pytest.raises(ValueError, match=MathAgentMessages.MATH_VALIDATION_FAILED):
            await solve_math("invalid expression", mock_llm)

    @pytest.mark.asyncio
    async def test_solve_non_numerical_response_raises_error(self, mock_llm):
        """Test that non-numerical response raises ValueError."""
        # Mock LLM response with non-numerical content
        mock_response = AsyncMock()
        mock_response.content = "This is not a number"
        mock_llm.ainvoke.return_value = mock_response

        with pytest.raises(ValueError, match="Math validation failed"):
            await solve_math("2 + 2", mock_llm)

    @pytest.mark.asyncio
    async def test_solve_llm_exception_raises_error(self, mock_llm):
        """Test that LLM exceptions raise ValueError."""
        # Mock LLM to raise an exception
        mock_llm.ainvoke.side_effect = Exception("LLM Error")

        with pytest.raises(ValueError, match=MathAgentMessages.MATH_EVALUATION_FAILED):
            await solve_math("2 + 2", mock_llm)

    @pytest.mark.asyncio
    async def test_solve_whitespace_in_response(self, mock_llm):
        """Test handling of whitespace in LLM response."""
        # Mock LLM response with whitespace
        mock_response = AsyncMock()
        mock_response.content = "  4  "
        mock_llm.ainvoke.return_value = mock_response

        result = await solve_math("2 + 2", mock_llm)
        assert result == "4"
        mock_llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_solve_float_result(self, mock_llm):
        """Test solving expressions that result in float values."""
        # Mock LLM response
        mock_response = AsyncMock()
        mock_response.content = "2.5"
        mock_llm.ainvoke.return_value = mock_response

        result = await solve_math("5 / 2", mock_llm)
        assert result == "2.5"
        mock_llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_solve_very_small_decimal(self, mock_llm):
        """Test solving expressions with very small decimal results."""
        # Mock LLM response
        mock_response = AsyncMock()
        mock_response.content = "0.001"
        mock_llm.ainvoke.return_value = mock_response

        result = await solve_math("1 / 1000", mock_llm)
        assert result == "0.001"
        mock_llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_solve_very_large_decimal(self, mock_llm):
        """Test solving expressions with very large decimal results."""
        # Mock LLM response
        mock_response = AsyncMock()
        mock_response.content = "1000000.5"
        mock_llm.ainvoke.return_value = mock_response

        result = await solve_math("1000000 + 0.5", mock_llm)
        assert result == "1000000.5"
        mock_llm.ainvoke.assert_called_once()
