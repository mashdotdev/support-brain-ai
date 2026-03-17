from pydantic_settings import BaseSettings
from pydantic import Field, SecretStr
from functools import lru_cache


class Settings(BaseSettings):
    """Environmental and configuration"""

    # Gemini configurations
    gemini_api_key: SecretStr = Field(
        default=SecretStr(""), description="Gemini api key for the AI modal"
    )
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    embedding_model: str = Field(
        default="gemini-embedding-001", description="Embedding model for the vectors"
    )
    embedding_dimension: int = 1024

    # Qdrant Configuration
    qdrant_url: str = Field(
        default="http://localhost:6333", description="Qdrant database url (docker)"
    )
    qdrant_collection_name: str = "docs_collection"

    # Authentication Configurations
    secret_key: str = Field(
        default="",
        description="Secret key for the authentication for JWT authentication",
    )
    jwt_algorithm: str = "HS256"
    access_token_expire_time: int = 30

    # Database Configuration
    database_url: str = Field(
        default="", description="PostgreSQL database url Neon or Supabase"
    )

    # jina config
    jina_api_key: SecretStr = Field(default=SecretStr(""))

    # open router
    open_router_api_key: str = ""

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()
