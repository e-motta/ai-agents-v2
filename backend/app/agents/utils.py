def parse_llm_content(content: str | list) -> str:
    """
    Parses the content of a LangChain AIMessage into a single, clean string.
    """
    if isinstance(content, list):
        return " ".join(str(item) for item in content).strip()
    return content.strip()
