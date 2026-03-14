from sqlmodel import select

from app.models.user import User
from app.schema.user_schema import UserCreate, UserResponse
from app.database.session import get_session
from app.database.db import AsyncSession
from app.core.security import encrypt_password

from fastapi import APIRouter, Depends, HTTPException, Request, status
from slowapi import Limiter
from slowapi.util import get_remote_address


user_router = APIRouter(prefix="/user", tags=["user"])

limiter = Limiter(key_func=get_remote_address)


@user_router.post(
    path="/create",
    description="Create a new user",
    status_code=status.HTTP_201_CREATED,
    response_model=UserResponse,
)
@limiter.limit("5/minute")
async def create_user(
    request: Request,
    user_data: UserCreate,
    session: AsyncSession = Depends(get_session),
) -> UserResponse:
    user = (
        await session.exec(select(User).where(User.email == user_data.email))
    ).first()

    if user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this email already exists",
        )

    new_user = User(
        **user_data.model_dump(),
        hashed_password=encrypt_password(plain_password=user_data.password),
    )

    session.add(instance=new_user)
    await session.commit()
    await session.refresh(instance=new_user)

    return UserResponse(id=str(new_user.id), email=new_user.email)


@user_router.get(path="/me")
@limiter.limit("5/minute")
async def get_user_details(request: Request) -> dict:
    return {"name": "Mashhood", "age": 20}
