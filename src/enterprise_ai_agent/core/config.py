"""Typed application configuration loaded from environment variables."""

from functools import lru_cache
from typing import Literal

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


@lru_cache
def get_settings() -> AppSettings:
    """Return the process-wide application settings instance."""

    return AppSettings()
