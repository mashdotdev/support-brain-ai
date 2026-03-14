from pwdlib import PasswordHash
from pwdlib.hashers.argon2 import Argon2Hasher

hash_password = PasswordHash((Argon2Hasher(),))


def encrypt_password(plain_password: str) -> str:
    """Hash plain password"""
    return hash_password.hash(password=plain_password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return hash_password.verify(password=plain_password, hash=hashed_password)
