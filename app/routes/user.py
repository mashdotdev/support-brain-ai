from datetime import timedelta

from app.models.user import User
from app.schema.user_schema import UserCreate, UserResponse, CurrentUserResponse
from app.schema.sign_in import SignInResponse
from app.database.session import get_session
from app.database.db import AsyncSession
from app.core.security import encrypt_password, verify_password, generate_access_token
from app.core.config import Settings, get_settings
from app.core.dependency import get_current_user

from sqlmodel import select
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security.oauth2 import OAuth2PasswordRequestForm
from slowapi import Limiter
from slowapi.util import get_remote_address

settings: Settings = get_settings()

user_router = APIRouter(prefix="/user", tags=["user"])

limiter = Limiter(key_func=get_remote_address)


@user_router.post(
    path="/sign-up",
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


@user_router.post(
    path="/sign-in",
    description="Sign in user",
    response_model=SignInResponse,
    status_code=status.HTTP_201_CREATED,
)
@limiter.limit("5/minute")
async def sign_in(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    session: AsyncSession = Depends(get_session),
) -> SignInResponse:
    user = (
        await session.exec(select(User).where(User.email == form_data.username))
    ).first()

    if not user or not verify_password(
        plain_password=form_data.password, hashed_password=user.hashed_password
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid Credentials"
        )

    access_token = generate_access_token(
        data={"sub": user.email},
        expires=timedelta(minutes=settings.access_token_expire_time),
    )

    return SignInResponse(access_token=access_token, type="Bearer")


@user_router.get(
    path="/me",
    description="Get current signed in user info",
    response_model=CurrentUserResponse,
    status_code=status.HTTP_200_OK,
)
@limiter.limit("5/minute")
async def get_user_details(
    request: Request, current_user: User = Depends(get_current_user)
) -> CurrentUserResponse:
    return CurrentUserResponse(
        id=str(current_user.id),
        name=current_user.name,
        email=current_user.email,
        created_at=str(current_user.created_at),
    )
