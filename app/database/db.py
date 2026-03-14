from app.core.config import get_settings, Settings


from fastapi import FastAPI
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from contextlib import asynccontextmanager
from colorama import Fore

settings: Settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"{Fore.GREEN}Creating database connection")
    engine = create_async_engine(
        url=settings.database_url, pool_pre_ping=True, pool_size=5
    )
    app.state.async_session = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

    async with engine.begin() as connection:
        await connection.run_sync(SQLModel.metadata.create_all)

    print(f"{Fore.GREEN}Database connection created")

    yield

    await engine.dispose()
