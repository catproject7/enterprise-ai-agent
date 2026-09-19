"""Tests for the package's public import surface."""

from enterprise_ai_agent.core.config import AppSettings
from enterprise_ai_agent.core.logging import configure_logging


def test_package_modules_are_importable() -> None:
    assert AppSettings is not None
    assert callable(configure_logging)
