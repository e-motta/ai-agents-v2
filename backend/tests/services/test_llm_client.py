from unittest.mock import AsyncMock, Mock

import pytest
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.services.llm_client import LLMClient


class TestLLMClientInit:
    """Test LLMClient initialization."""

    def test_llm_client_init_success(self):
        """Test successful LLMClient initialization."""
        # Create mock ChatOpenAI instance
        mock_llm = Mock(spec=ChatOpenAI)

        # Create LLMClient
        client = LLMClient(mock_llm)

        # Verify initialization
        assert client.llm == mock_llm


class TestParseLLMContent:
    """Test the _parse_llm_content static method."""

    def test_parse_string_content_with_whitespace(self):
        """Test parsing string content with leading/trailing whitespace."""
        content = "  Hello, world!  "
        result = LLMClient._parse_llm_content(content)

        assert result == "Hello, world!"

    def test_parse_list_content(self):
        """Test parsing list content."""
        content = ["Hello", "world", "!"]
        result = LLMClient._parse_llm_content(content)

        assert result == "Hello world !"

    def test_parse_list_content_with_mixed_types(self):
        """Test parsing list content with mixed types."""
        content = ["Hello", 123, "world", True]
        result = LLMClient._parse_llm_content(content)

        assert result == "Hello 123 world True"

    def test_parse_list_content_with_whitespace(self):
        """Test parsing list content with whitespace."""
        content = ["  Hello  ", "  world  ", "  !  "]
        result = LLMClient._parse_llm_content(content)

        assert result == "Hello world !"

    def test_parse_empty_list(self):
        """Test parsing empty list."""
        content = []
        result = LLMClient._parse_llm_content(content)

        assert result == ""

    def test_parse_none_values_in_list(self):
        """Test parsing list with None values."""
        content = ["Hello", None, "world", None]
        result = LLMClient._parse_llm_content(content)

        assert result == "Hello world"


class TestAskMethod:
    """Test the ask method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_llm = AsyncMock(spec=ChatOpenAI)
        self.client = LLMClient(self.mock_llm)

    @pytest.mark.asyncio
    async def test_ask_success(self):
        """Test successful ask method call."""
        # Mock LLM response
        mock_response = AIMessage(content="Hello! How can I help you?")
        self.mock_llm.ainvoke.return_value = mock_response

        # Call the method
        result = await self.client.ask(
            message="Hello", system_prompt="You are a helpful assistant."
        )

        # Verify result
        assert result == "Hello! How can I help you?"

        # Verify LLM was called with correct messages
        self.mock_llm.ainvoke.assert_called_once()
        call_args = self.mock_llm.ainvoke.call_args[0][0]
        assert len(call_args) == 2
        assert isinstance(call_args[0], SystemMessage)
        assert call_args[0].content == "You are a helpful assistant."
        assert isinstance(call_args[1], HumanMessage)
        assert call_args[1].content == "Hello"

    @pytest.mark.asyncio
    async def test_ask_with_string_response(self):
        """Test ask method with string response."""
        # Mock LLM response with string content
        mock_response = AIMessage(content="Simple response")
        self.mock_llm.ainvoke.return_value = mock_response

        # Call the method
        result = await self.client.ask(
            message="Test message", system_prompt="Test system prompt"
        )

        # Verify result
        assert result == "Simple response"

    @pytest.mark.asyncio
    async def test_ask_with_list_response(self):
        """Test ask method with list response."""
        # Mock LLM response with list content
        mock_response = AIMessage(content=["This", "is", "a", "list", "response"])
        self.mock_llm.ainvoke.return_value = mock_response

        # Call the method
        result = await self.client.ask(
            message="Test message", system_prompt="Test system prompt"
        )

        # Verify result
        assert result == "This is a list response"

    @pytest.mark.asyncio
    async def test_ask_with_whitespace_response(self):
        """Test ask method with response containing whitespace."""
        # Mock LLM response with whitespace
        mock_response = AIMessage(content="  Response with whitespace  ")
        self.mock_llm.ainvoke.return_value = mock_response

        # Call the method
        result = await self.client.ask(
            message="Test message", system_prompt="Test system prompt"
        )

        # Verify result is trimmed
        assert result == "Response with whitespace"

    @pytest.mark.asyncio
    async def test_ask_with_empty_response(self):
        """Test ask method with empty response."""
        # Mock LLM response with empty content
        mock_response = AIMessage(content="")
        self.mock_llm.ainvoke.return_value = mock_response

        # Call the method
        result = await self.client.ask(
            message="Test message", system_prompt="Test system prompt"
        )

        # Verify result
        assert result == ""

    @pytest.mark.asyncio
    async def test_ask_with_empty_list_response(self):
        """Test ask method with empty list response."""
        # Mock LLM response with empty list
        mock_response = AIMessage(content=[])
        self.mock_llm.ainvoke.return_value = mock_response

        # Call the method
        result = await self.client.ask(
            message="Test message", system_prompt="Test system prompt"
        )

        # Verify result
        assert result == ""

    @pytest.mark.asyncio
    async def test_ask_llm_exception(self):
        """Test ask method when LLM raises an exception."""
        # Mock LLM to raise exception
        self.mock_llm.ainvoke.side_effect = Exception("LLM error")

        # Call the method and expect exception
        with pytest.raises(Exception, match="LLM error"):
            await self.client.ask(
                message="Test message", system_prompt="Test system prompt"
            )

    @pytest.mark.asyncio
    async def test_ask_with_special_characters(self):
        """Test ask method with special characters in message and prompt."""
        # Mock LLM response
        mock_response = AIMessage(content="Response with special chars: @#$%")
        self.mock_llm.ainvoke.return_value = mock_response

        # Call the method with special characters
        result = await self.client.ask(
            message="Message with @#$% special chars",
            system_prompt="System prompt with @#$% special chars",
        )

        # Verify result
        assert result == "Response with special chars: @#$%"

        # Verify messages were passed correctly
        call_args = self.mock_llm.ainvoke.call_args[0][0]
        assert call_args[0].content == "System prompt with @#$% special chars"
        assert call_args[1].content == "Message with @#$% special chars"

    @pytest.mark.asyncio
    async def test_ask_with_multiline_content(self):
        """Test ask method with multiline content."""
        # Mock LLM response with multiline content
        mock_response = AIMessage(content="Line 1\nLine 2\nLine 3")
        self.mock_llm.ainvoke.return_value = mock_response

        # Call the method
        result = await self.client.ask(
            message="Multiline\nmessage", system_prompt="Multiline\nsystem\nprompt"
        )

        # Verify result
        assert result == "Line 1\nLine 2\nLine 3"

    @pytest.mark.asyncio
    async def test_ask_with_very_long_content(self):
        """Test ask method with very long content."""
        # Create long content
        long_message = "A" * 1000
        long_system_prompt = "B" * 1000
        long_response = "C" * 1000

        # Mock LLM response
        mock_response = AIMessage(content=long_response)
        self.mock_llm.ainvoke.return_value = mock_response

        # Call the method
        result = await self.client.ask(
            message=long_message, system_prompt=long_system_prompt
        )

        # Verify result
        assert result == long_response


class TestLLMClientIntegration:
    """Integration tests for LLMClient."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_llm = AsyncMock(spec=ChatOpenAI)
        self.client = LLMClient(self.mock_llm)

    @pytest.mark.asyncio
    async def test_multiple_ask_calls(self):
        """Test multiple ask calls with the same client."""
        # Mock different responses for different calls
        responses = [
            AIMessage(content="First response"),
            AIMessage(content="Second response"),
            AIMessage(content="Third response"),
        ]
        self.mock_llm.ainvoke.side_effect = responses

        # Make multiple calls
        result1 = await self.client.ask("First message", "System prompt")
        result2 = await self.client.ask("Second message", "System prompt")
        result3 = await self.client.ask("Third message", "System prompt")

        # Verify results
        assert result1 == "First response"
        assert result2 == "Second response"
        assert result3 == "Third response"

        # Verify LLM was called 3 times
        assert self.mock_llm.ainvoke.call_count == 3

    @pytest.mark.asyncio
    async def test_ask_with_different_system_prompts(self):
        """Test ask method with different system prompts."""
        # Mock LLM response
        mock_response = AIMessage(content="Response")
        self.mock_llm.ainvoke.return_value = mock_response

        # Call with different system prompts
        await self.client.ask("Message", "Math assistant")
        await self.client.ask("Message", "Code assistant")
        await self.client.ask("Message", "General assistant")

        # Verify LLM was called 3 times with different system messages
        assert self.mock_llm.ainvoke.call_count == 3

        # Check the system messages
        calls = self.mock_llm.ainvoke.call_args_list
        assert calls[0][0][0][0].content == "Math assistant"
        assert calls[1][0][0][0].content == "Code assistant"
        assert calls[2][0][0][0].content == "General assistant"
