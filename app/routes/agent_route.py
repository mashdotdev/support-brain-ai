from app.schema.agent_schema import AgentMessage
from app.ai_agent.triage import agent
from app.core.dependency import get_current_user
from app.models.user import User

from fastapi import APIRouter, Depends, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(get_remote_address)
ai_agent_router = APIRouter(prefix="/agent", tags=["ai-agent"])


@ai_agent_router.post(path="/chat", description="Chat with AI Agent")
@limiter.limit("5/minute")
async def chat(
    request: Request,
    agent_req: AgentMessage,
    current_user: User = Depends(get_current_user),
):
    return await agent(message=agent_req.message)
