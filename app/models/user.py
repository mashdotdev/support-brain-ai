import uuid
from typing import Optional
from sqlmodel import SQLModel, Field
from datetime import datetime


class User(SQLModel, table=True):
    """User table in the Neon database"""

    id: Optional[uuid.UUID] = Field(default_factory=uuid.uuid4, primary_key=True)
    name: Optional[str] = Field(
        default=None, description="User name", min_length=4, max_length=18
    )
    email: str = Field(index=True)
    hashed_password: str
    created_at: Optional[datetime] = Field(
        default_factory=datetime.now, description="Account creation date"
    )
