from app.models.user import User
from app.database.db import AsyncSession
from app.database.session import get_session
from app.core.security import decode_access_token

from sqlmodel import select
from fastapi import Depends, HTTPException, status
from fastapi.security.oauth2 import OAuth2PasswordBearer


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/user/sign-in")


async def get_current_user(
    token: str = Depends(oauth2_scheme), session: AsyncSession = Depends(get_session)
) -> User:
    credential_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not Authenticated",
        headers={"WWW-AUTHENTICATION": "Bearer"},
    )

    payload = decode_access_token(token=token)
    if not payload:
        raise credential_exception

    email: str | None = payload.get("sub")
    if not email:
        raise credential_exception

    user = (await session.exec(select(User).where(User.email == email))).first()
    if not user:
        raise credential_exception

    return user
