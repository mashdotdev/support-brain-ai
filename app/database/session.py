from fastapi import Request


async def get_session(request: Request):
    async with request.app.state.async_session() as session:
        yield session
