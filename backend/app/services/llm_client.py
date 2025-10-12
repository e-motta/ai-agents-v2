from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


class LLMClient:
    """Wrapper around ChatOpenAI for clean, testable use in FastAPI."""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    @staticmethod
    def _parse_llm_content(content: str | list) -> str:
        """
        Parses the content of a LangChain AIMessage into a single, clean string.
        """
        if isinstance(content, list):
            return " ".join(str(item).strip() for item in content if item)
        return content.strip()

    async def ask(self, message: str, system_prompt: str) -> str:
        # todo: add optional message history
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=message),
        ]

        response = await self.llm.ainvoke(messages)

        return self._parse_llm_content(response.content)
