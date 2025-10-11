from llama_index.core.base.base_query_engine import BaseQueryEngine
from pydantic import BaseModel

from app.services.llm_client import LLMClient


class ChatRequest(BaseModel):
    message: str
    user_id: str
    conversation_id: str


class ChatContext(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    payload: ChatRequest
    sanitized_message: str
    llm_client: LLMClient
    knowledge_engine: BaseQueryEngine | None


class WorkflowStep(BaseModel):
    agent: str
    action: str
    result: str


class ChatResponse(BaseModel):
    user_id: str
    conversation_id: str
    router_decision: str
    response: str
    source_agent_response: str
    agent_workflow: list[WorkflowStep]


class ErrorResponse(BaseModel):
    error: str
    code: str
    details: str | None = None
