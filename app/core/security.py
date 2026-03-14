from typing import Optional
from datetime import timedelta, datetime

from app.core.config import Settings, get_settings

from pwdlib import PasswordHash
from pwdlib.hashers.argon2 import Argon2Hasher
from jose import jwt, JWTError

settings: Settings = get_settings()
hash_password = PasswordHash((Argon2Hasher(),))


def encrypt_password(plain_password: str) -> str:
    """Hash plain password"""
    return hash_password.hash(password=plain_password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return hash_password.verify(password=plain_password, hash=hashed_password)


def generate_access_token(data: dict, expires: Optional[timedelta]) -> str:
    to_encode = data.copy()
    expire = datetime.now() + (expires or timedelta(minutes=15))
    to_encode.update({"exp": expire})

    return jwt.encode(
        claims=to_encode, key=settings.secret_key, algorithm=settings.jwt_algorithm
    )


def decode_access_token(token: str):
    try:
        return jwt.decode(
            token=token, key=settings.secret_key, algorithms=[settings.jwt_algorithm]
        )
    except JWTError:
        return None
