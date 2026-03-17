from typing import cast

from app.routes.user import user_router
from app.routes.agent_route import ai_agent_router
from app.routes.rag_route import rag_router
from app.database.db import lifespan

from fastapi import FastAPI, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from starlette.types import ExceptionHandler


limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title="Support Brain",
    description="Chat with AI in plain english about any documentation",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)
app.add_exception_handler(
    RateLimitExceeded, cast(ExceptionHandler, _rate_limit_exceeded_handler)
)

app.include_router(router=user_router)
app.include_router(router=ai_agent_router)
app.include_router(router=rag_router)


@app.get(path="/health", description="Check API health and running status")
@limiter.limit("5/minute")
def check_health(request: Request) -> dict:
    return {"status": "running", "api_name": "support_brain_ai"}
