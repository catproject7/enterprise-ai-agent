"""Typed application configuration loaded from environment variables."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Application settings with environment-based overrides."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="ENTERPRISE_AI_AGENT_",
        extra="ignore",
    )

    environment: Literal["local", "test", "staging", "production"] = "local"
    log_level: Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"] = "INFO"

    llm_model: str = "gpt-4o-mini"
    llm_api_key: SecretStr | None = None
    llm_base_url: str | None = None

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimension: int = Field(default=384, gt=0)
    embedding_batch_size: int = Field(default=32, gt=0)
    embedding_cache_dir: str | None = None

    qdrant_url: str = ":memory:"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection_name: str = "document_chunks"

    @field_validator("llm_model", "embedding_model", "qdrant_url", "qdrant_collection_name")
    @classmethod
    def validate_non_empty_text(cls, value: str) -> str:
        """Reject empty runtime configuration values."""

        if not value.strip():
            raise ValueError("runtime configuration values must not be empty")
        return value

    @field_validator("llm_api_key", "qdrant_api_key", mode="before")
    @classmethod
    def normalize_optional_secret(cls, value: object) -> object:
        """Treat blank environment values as missing secrets."""

        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("llm_base_url", "embedding_cache_dir", mode="before")
    @classmethod
    def normalize_optional_text(cls, value: object) -> object:
        """Treat blank optional text values as missing."""

        if isinstance(value, str) and not value.strip():
            return None
        return value


@lru_cache
def get_settings() -> AppSettings:
    """Return the process-wide application settings instance."""

    return AppSettings()
