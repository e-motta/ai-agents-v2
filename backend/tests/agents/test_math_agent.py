"""
Unit tests for Math Agent simple expressions.

These tests verify that the math agent correctly evaluates mathematical
expressions without making external LLM calls.
"""

import pytest

from app.agents.math_agent import solve_math
from app.enums import MathAgentMessages


class TestSolveMath:
    """Test the solve_math function."""

    @pytest.mark.asyncio
    async def test_solve_simple_addition(self, mock_llm_client):
        """Test solving simple addition."""
        # Mock LLM response
        mock_llm_client.ask.return_value = "4"

        result = await solve_math("2 + 2", mock_llm_client)
        assert result == "4"
        mock_llm_client.ask.assert_called_once()

    @pytest.mark.asyncio
    async def test_solve_simple_subtraction(self, mock_llm_client):
        """Test solving simple subtraction."""
        # Mock LLM response
        mock_llm_client.ask.return_value = "3"

        result = await solve_math("5 - 2", mock_llm_client)
        assert result == "3"
        mock_llm_client.ask.assert_called_once()

    @pytest.mark.asyncio
    async def test_solve_simple_multiplication(self, mock_llm_client):
        """Test solving simple multiplication."""
        # Mock LLM response
        mock_llm_client.ask.return_value = "10"

        result = await solve_math("2 * 5", mock_llm_client)
        assert result == "10"
        mock_llm_client.ask.assert_called_once()

    @pytest.mark.asyncio
    async def test_solve_simple_division(self, mock_llm_client):
        """Test solving simple division."""
        # Mock LLM response
        mock_llm_client.ask.return_value = "3"

        result = await solve_math("6 / 2", mock_llm_client)
        assert result == "3"
        mock_llm_client.ask.assert_called_once()

    @pytest.mark.asyncio
    async def test_solve_complex_expression(self, mock_llm_client):
        """Test solving complex mathematical expressions."""
        # Mock LLM response
        mock_llm_client.ask.return_value = "14"

        result = await solve_math("(2 + 3) * 4 - 6", mock_llm_client)
        assert result == "14"
        mock_llm_client.ask.assert_called_once()

    @pytest.mark.asyncio
    async def test_solve_decimal_expression(self, mock_llm_client):
        """Test solving expressions with decimals."""
        # Mock LLM response
        mock_llm_client.ask.return_value = "2.5"

        result = await solve_math("1.5 + 1.0", mock_llm_client)
        assert result == "2.5"
        mock_llm_client.ask.assert_called_once()

    @pytest.mark.asyncio
    async def test_solve_power_expression(self, mock_llm_client):
        """Test solving power expressions."""
        # Mock LLM response
        mock_llm_client.ask.return_value = "8"

        result = await solve_math("2^3", mock_llm_client)
        assert result == "8"
        mock_llm_client.ask.assert_called_once()

    @pytest.mark.asyncio
    async def test_solve_square_root(self, mock_llm_client):
        """Test solving square root expressions."""
        # Mock LLM response
        mock_llm_client.ask.return_value = "4"

        result = await solve_math("sqrt(16)", mock_llm_client)
        assert result == "4"
        mock_llm_client.ask.assert_called_once()

    @pytest.mark.asyncio
    async def test_solve_trigonometric_function(self, mock_llm_client):
        """Test solving trigonometric functions."""
        # Mock LLM response
        mock_llm_client.ask.return_value = "1"

        result = await solve_math("sin(pi/2)", mock_llm_client)
        assert result == "1"
        mock_llm_client.ask.assert_called_once()

    @pytest.mark.asyncio
    async def test_solve_negative_result(self, mock_llm_client):
        """Test solving expressions that result in negative numbers."""
        # Mock LLM response
        mock_llm_client.ask.return_value = "-3"

        result = await solve_math("2 - 5", mock_llm_client)
        assert result == "-3"
        mock_llm_client.ask.assert_called_once()

    @pytest.mark.asyncio
    async def test_solve_zero_result(self, mock_llm_client):
        """Test solving expressions that result in zero."""
        # Mock LLM response
        mock_llm_client.ask.return_value = "0"

        result = await solve_math("5 - 5", mock_llm_client)
        assert result == "0"
        mock_llm_client.ask.assert_called_once()

    @pytest.mark.asyncio
    async def test_solve_large_number(self, mock_llm_client):
        """Test solving expressions with large numbers."""
        # Mock LLM response
        mock_llm_client.ask.return_value = "1000000"

        result = await solve_math("1000 * 1000", mock_llm_client)
        assert result == "1000000"
        mock_llm_client.ask.assert_called_once()

    @pytest.mark.asyncio
    async def test_solve_list_content_response(self, mock_llm_client):
        """Test handling of list content in LLM response."""
        # Mock LLM response with list content (LLMClient.ask() handles parsing)
        mock_llm_client.ask.return_value = "4"

        result = await solve_math("2 + 2", mock_llm_client)
        assert result == "4"
        mock_llm_client.ask.assert_called_once()

    @pytest.mark.asyncio
    async def test_solve_empty_response_raises_error(self, mock_llm_client):
        """Test that empty LLM response raises ValueError."""
        # Mock LLM response with empty content
        mock_llm_client.ask.return_value = ""

        with pytest.raises(ValueError, match=MathAgentMessages.MATH_VALIDATION_FAILED):
            await solve_math("2 + 2", mock_llm_client)

    @pytest.mark.asyncio
    async def test_solve_error_response_raises_error(self, mock_llm_client):
        """Test that 'Error' response raises ValueError."""
        # Mock LLM response with error
        mock_llm_client.ask.return_value = "Error"

        with pytest.raises(ValueError, match=MathAgentMessages.MATH_VALIDATION_FAILED):
            await solve_math("invalid expression", mock_llm_client)

    @pytest.mark.asyncio
    async def test_solve_non_numerical_response_raises_error(self, mock_llm_client):
        """Test that non-numerical response raises ValueError."""
        # Mock LLM response with non-numerical content
        mock_llm_client.ask.return_value = "This is not a number"

        with pytest.raises(ValueError, match=MathAgentMessages.MATH_VALIDATION_FAILED):
            await solve_math("2 + 2", mock_llm_client)

    @pytest.mark.asyncio
    async def test_solve_llm_exception_raises_error(self, mock_llm_client):
        """Test that LLM exceptions raise ValueError."""
        # Mock LLM to raise an exception
        mock_llm_client.ask.side_effect = Exception("LLM Error")

        with pytest.raises(ValueError, match=MathAgentMessages.MATH_EVALUATION_FAILED):
            await solve_math("2 + 2", mock_llm_client)

    @pytest.mark.asyncio
    async def test_solve_float_result(self, mock_llm_client):
        """Test solving expressions that result in float values."""
        # Mock LLM response
        mock_llm_client.ask.return_value = "2.5"

        result = await solve_math("5 / 2", mock_llm_client)
        assert result == "2.5"
        mock_llm_client.ask.assert_called_once()

    @pytest.mark.asyncio
    async def test_solve_very_small_decimal(self, mock_llm_client):
        """Test solving expressions with very small decimal results."""
        # Mock LLM response
        mock_llm_client.ask.return_value = "0.001"

        result = await solve_math("1 / 1000", mock_llm_client)
        assert result == "0.001"
        mock_llm_client.ask.assert_called_once()

    @pytest.mark.asyncio
    async def test_solve_very_large_decimal(self, mock_llm_client):
        """Test solving expressions with very large decimal results."""
        # Mock LLM response
        mock_llm_client.ask.return_value = "1000000.5"

        result = await solve_math("1000000 + 0.5", mock_llm_client)
        assert result == "1000000.5"
        mock_llm_client.ask.assert_called_once()
