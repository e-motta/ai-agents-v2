from llama_index.core.base.base_query_engine import BaseQueryEngine
from pydantic import BaseModel

from app.services.llm_client import LLMClient


class ChatRequest(BaseModel):
    message: str
    user_id: str
    conversation_id: str


class GenericContext(BaseModel):
    payload: ChatRequest


class ProcessingContext(GenericContext):
    sanitized_message: str
    llm_client: LLMClient
    knowledge_engine: BaseQueryEngine | None

    model_config = {"arbitrary_types_allowed": True}


class RoutingContext(GenericContext):
    sanitized_message: str
    llm_client: LLMClient
    agent_response: str | None = None
    agent_type: str | None = None

    model_config = {"arbitrary_types_allowed": True}


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
