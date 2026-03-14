from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    """Environmental and configuration"""

    # Gemini configurations
    gemini_api_key: str | None = Field(
        default=None, description="Gemini api key for the AI modal"
    )
    embedding_model: str | None = Field(
        default=None, description="Embedding model for the vectors"
    )
    embedding_dimension: int = 768

    # Qdrant Configuration
    qdrant_url: str | None = Field(
        default=None, description="Qdrant database url (docker)"
    )

    # Authentication Configurations
    secret_key: str | None = Field(
        default=None,
        description="Secret key for the authentication for JWT authentication",
    )

    # Database Configuration
    database_url: str = Field(
        default="", description="PostgreSQL database url Neon or Supabase"
    )

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()
