from pydantic import BaseModel
from typing import Optional


class UserCreate(BaseModel):
    name: Optional[str] = None
    email: str
    password: str


class UserResponse(BaseModel):
    id: str
    email: str
