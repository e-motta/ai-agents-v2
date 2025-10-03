from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    user_id: str
    conversation_id: str


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
