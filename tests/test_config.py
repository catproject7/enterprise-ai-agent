"""Tests for environment-based application configuration."""

import pytest
from pydantic import ValidationError

from enterprise_ai_agent.core.config import AppSettings


def test_settings_defaults() -> None:
    settings = AppSettings(_env_file=None)

    assert settings.environment == "local"
    assert settings.log_level == "INFO"


def test_settings_environment_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENTERPRISE_AI_AGENT_ENVIRONMENT", "test")
    monkeypatch.setenv("ENTERPRISE_AI_AGENT_LOG_LEVEL", "DEBUG")

    settings = AppSettings(_env_file=None)

    assert settings.environment == "test"
    assert settings.log_level == "DEBUG"


def test_settings_reject_invalid_environment() -> None:
    with pytest.raises(ValidationError):
        AppSettings(environment="invalid", _env_file=None)
