"""Tests for logging initialization."""

import logging
from collections.abc import Iterator

import pytest

from enterprise_ai_agent.core.logging import configure_logging


def _application_handlers() -> list[logging.Handler]:
    return [
        handler
        for handler in logging.getLogger().handlers
        if handler.get_name() == "enterprise_ai_agent"
    ]


@pytest.fixture
def isolated_root_logger() -> Iterator[None]:
    root_logger = logging.getLogger()
    original_handlers = root_logger.handlers[:]
    original_level = root_logger.level
    root_logger.handlers.clear()

    try:
        yield
    finally:
        root_logger.handlers.clear()
        root_logger.handlers.extend(original_handlers)
        root_logger.setLevel(original_level)


def test_configure_logging_initializes_root_logger(
    isolated_root_logger: None,
) -> None:
    configure_logging("DEBUG")

    root_logger = logging.getLogger()

    assert root_logger.level == logging.DEBUG
    assert len(_application_handlers()) == 1


def test_configure_logging_does_not_add_duplicate_handlers(
    isolated_root_logger: None,
) -> None:
    configure_logging("DEBUG")
    configure_logging("INFO")

    root_logger = logging.getLogger()

    assert root_logger.level == logging.INFO
    assert len(_application_handlers()) == 1
